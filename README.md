# FLUX VAE + DiT 潜空间扩散模型

## 一、项目概述

本项目使用 **FLUX.1-dev 预训练 VAE** 作为图像编解码器，在其潜在空间上训练 **DiT (Diffusion Transformer)** 扩散模型，实现无条件图像生成。

**核心思路：** 利用强大的预训练 VAE 跳过自训练编码器步骤，直接在高质量潜在空间中学习数据分布。

**流程简图：**

```
输入图像 x ∈ [B, 3, 256, 256]
    │
    ▼ FLUX VAE Encoder (冻结)
    │
    z_raw = Encoder(x).sample()              # [B, 16, 32, 32]
    z_scaled = (z_raw - shift) × scale       # FLUX 内置 scale/shift
    z_norm = (z_scaled - μ_ch) / σ_ch        # 逐通道归一化 → ~N(0,1)
    │
    ▼ DiT 扩散训练 (在 z_norm 空间)
    │
    采样: z_t → DiT → z_0_norm (DDIM, 50步)
    │
    ▼ 反归一化 + FLUX VAE Decoder
    │
    z_0_scaled = z_0_norm × σ_ch + μ_ch      # 反归一化
    z_0_raw = z_0_scaled / scale + shift      # 反 scale/shift
    x_gen = Decoder(z_0_raw)                  # [B, 3, 256, 256]
```

**支持数据集：**
- `cifar10` / `cifar100` — 32×32 小图，自动上采样到 `image_size` 编码
- `imagenet` — ImageNet-1k 256×256 (via HuggingFace `evanarlian/imagenet_1k_resized_256`)
- `imagefolder` — 本地 ImageFolder 格式 (需含 `train/` 和 `val/` 子目录)

---

## 二、环境要求

### 硬件
- GPU: 建议 ≥ 16GB 显存 (RTX 4090 / A100 / B300 等)
- 磁盘: CIFAR ~2GB, ImageNet ~30GB (HuggingFace 缓存)

### 软件依赖

```bash
pip install torch torchvision numpy diffusers transformers datasets huggingface_hub
```

核心依赖：
| 包 | 最低版本 | 用途 |
|---|---------|------|
| `torch` | ≥ 2.0 | 训练框架 |
| `torchvision` | ≥ 0.15 | 数据加载、Inception-V3 |
| `diffusers` | ≥ 0.25 | FLUX.1-dev VAE (`AutoencoderKL`) |
| `datasets` | ≥ 2.0 | HuggingFace ImageNet 加载 |
| `numpy` | ≥ 1.20 | FID 计算 |

---

## 三、架构设计

### 3.1 FLUX.1-dev VAE

使用 `black-forest-labs/FLUX.1-dev` 的预训练 `AutoencoderKL`：

| 参数 | 值 | 说明 |
|------|---|------|
| `latent_channels` | 16 | 潜在通道数 (比 SD 的 4 通道更丰富) |
| 空间下采样 | 8× | 256×256 → 32×32 |
| `scaling_factor` | 0.3611 | 内置 latent 缩放因子 |
| `shift_factor` | 0.1159 | 内置 latent 偏移因子 |
| 参数量 | ~168M | 完全冻结，不参与训练 |

**编码公式：**
$$z_{\text{scaled}} = (z_{\text{raw}} - \text{shift}) \times \text{scale}$$

**解码公式：**
$$z_{\text{raw}} = z_{\text{scaled}} / \text{scale} + \text{shift}$$

### 3.2 逐通道归一化

FLUX VAE 输出的 latent 各通道统计量差异较大 (均值范围 -1.08 ~ +1.70，标准差 ~1.37)。扩散模型假设数据近似标准正态分布，所以需要逐通道归一化：

$$z_{\text{norm}}^{(c)} = \frac{z_{\text{scaled}}^{(c)} - \mu_c}{\sigma_c}$$

其中 $\mu_c, \sigma_c$ 在整个训练集上统计。归一化后每通道 ~ $\mathcal{N}(0, 1)$。

生成时需要反归一化：$z_{\text{scaled}}^{(c)} = z_{\text{norm}}^{(c)} \times \sigma_c + \mu_c$

> 通道统计量 ($\mu_c, \sigma_c$) 保存在每个 checkpoint 中，确保推理复现。

### 3.3 DiT (Diffusion Transformer)

DiT 在归一化后的潜在空间中运行：

| 配置 | 层数 | 头数 | 隐藏维度 | 参数量 | 适用场景 |
|------|------|------|---------|--------|---------|
| DiT-T (Tiny) | 6 | 3 | 192 | ~5M | 快速实验 |
| DiT-S (Small) | 12 | 6 | 384 | ~33M | CIFAR-10 |
| DiT-B (Base) | 12 | 12 | 768 | ~130M | ImageNet |

**关键设计：**
- **Patch Embedding**: 将 32×32×16 latent 分割为 16×16 = 256 个 token (patch_size=2)
- **adaLN-Zero**: 时间步嵌入通过 Adaptive LayerNorm 调制每层参数
- **Zero-Init 输出层**: 初始化时每个 DiT 块 ≈ 恒等映射
- **QK-Norm**: 防止注意力权重发散
- **2D 正弦位置编码**: patch 级绝对位置信息

> 本项目不使用 cross-attention (FLUX VAE 无多尺度量化输出)。

### 3.4 扩散过程

| 组件 | 选择 | 说明 |
|------|------|------|
| 噪声调度 | Cosine | $\bar{\alpha}(t) = \cos^2(\frac{\pi}{2} \cdot \frac{t/T + s}{1+s})$, $s=0.008$ |
| 预测目标 | v-prediction | $v = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1-\bar{\alpha}_t} \cdot x_0$, 高 SNR 端更稳定 |
| 损失加权 | min-SNR-$\gamma$ | $w(t) = \min(\text{SNR}(t), \gamma) / \text{SNR}(t)$, $\gamma=5.0$ |
| 采样方法 | DDIM | 确定性采样，默认 50 步 |
| 时间步数 | 1000 | 训练时均匀采样 |

**训练损失：**
$$\mathcal{L} = w_{\text{SNR}}(t) \cdot \|v_\theta(z_t, t) - v_{\text{target}}\|^2$$

### 3.5 训练优化

| 优化器 | 设置 |
|--------|------|
| AdamW | lr=2e-4, weight_decay=0.01, betas=(0.9, 0.99) |
| LR Schedule | Cosine annealing + 20-epoch linear warmup |
| EMA | decay=0.9999, 评估时使用 EMA 权重 |
| AMP | bf16 混合精度 (默认开启) |
| Grad Clip | max_norm=1.0 |

---

## 四、训练指南

### 4.1 CIFAR-10

```bash
cd diffusion_torch/diffusion_torch

python3 train_flux_latent_diffusion.py \
    --dataset cifar10 \
    --data_dir ./data \
    --image_size 256 \
    --dit_size B \
    --epochs 500 \
    --batch_size 256 \
    --lr 2e-4 \
    --warmup_epochs 20 \
    --fid_interval 25 \
    --vis_interval 5 \
    --output_dir ./checkpoints/flux_latent_diffusion
```

> CIFAR-10 的 32×32 图像会被上采样到 256×256 送入 FLUX VAE。FID 在原始 32×32 分辨率下计算。

### 4.2 ImageNet 256×256

```bash
cd diffusion_torch/diffusion_torch

python3 train_flux_latent_diffusion.py \
    --dataset imagenet \
    --data_dir ./data \
    --image_size 256 \
    --dit_size B \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --warmup_epochs 10 \
    --fid_interval 10 \
    --fid_n_samples 2048 \
    --vis_interval 5 \
    --save_interval 10 \
    --output_dir ./checkpoints/flux_imagenet
```

> 首次运行时会自动从 HuggingFace 下载 `evanarlian/imagenet_1k_resized_256` (~25GB)，缓存到 `--data_dir/imagenet_hf_cache/`。
> 1.28M 训练图像的 latent 预计算约需 30-60 分钟，自动缓存到 `--latent_cache_dir`。

### 4.3 ImageNet 快速实验

使用 `--max_train_samples` 限制训练集大小，快速验证流程：

```bash
python3 train_flux_latent_diffusion.py \
    --dataset imagenet \
    --data_dir ./data \
    --image_size 256 \
    --dit_size B \
    --max_train_samples 50000 \
    --epochs 50 \
    --batch_size 128 \
    --lr 2e-4 \
    --fid_interval 10 \
    --output_dir ./checkpoints/flux_imagenet_50k
```

### 4.4 本地 ImageFolder

```bash
python3 train_flux_latent_diffusion.py \
    --dataset imagefolder \
    --data_dir /path/to/your/dataset \
    --image_size 256 \
    --dit_size B \
    --epochs 200 \
    --batch_size 64 \
    --output_dir ./checkpoints/flux_custom
```

目录结构需为：
```
/path/to/your/dataset/
  train/
    class_a/
      img001.jpg
      ...
    class_b/
      ...
  val/
    class_a/
      ...
```

### 4.5 恢复训练

```bash
python3 train_flux_latent_diffusion.py \
    --dataset imagenet \
    --data_dir ./data \
    --resume ./checkpoints/flux_imagenet/epoch0050.pt \
    --epochs 100 \
    --output_dir ./checkpoints/flux_imagenet
```

---

## 五、训练流程详解

完整训练流程如下：

### Step 1: 加载 FLUX VAE

```python
FluxVAEWrapper(device, dtype=torch.float32)
# 从 HuggingFace 加载 black-forest-labs/FLUX.1-dev 的 vae 子模块
# 完全冻结, requires_grad=False
```

### Step 2: 预计算 Latent

```python
precompute_latents(flux_vae, train_set, batch_size=64, cache_path=...)
# 遍历整个训练集, 通过 FLUX VAE encoder 得到 latent
# 缓存到磁盘 (.pt 文件), 后续 epoch 直接加载
```

输出形状: `(N, 16, 32, 32)` float32

### Step 3: 逐通道归一化

```python
normalize_latents(train_latents)
# 计算每通道 μ_c, σ_c
# 归一化: z_norm = (z - μ_c) / σ_c
# 返回 (归一化 latent, channel_mean, channel_std)
```

### Step 4: 训练 DiT

每个 epoch：
1. 从归一化 latent cache 采样 mini-batch $z_0$
2. 均匀采样 $t \sim U\{1, T\}$
3. 加噪: $z_t = \sqrt{\bar\alpha_t} \cdot z_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon$
4. DiT 预测: $\hat{v} = \text{DiT}(z_t, t)$
5. 计算 min-SNR-$\gamma$ 加权 MSE 损失
6. 梯度更新 + EMA 更新

### Step 5: 评估与采样

- **可视化**: 每 `vis_interval` epoch 用 EMA 权重 DDIM 采样 64 张图
- **FID**: 每 `fid_interval` epoch 生成 `fid_n_samples` 张图与验证集比较
  - CIFAR: 生成 256×256 → 下采样 32×32 → Inception 特征
  - ImageNet: 生成 256×256 → 直接送 Inception 特征

---

## 六、完整参数参考

### train_flux_latent_diffusion.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | cifar10 | 数据集: cifar10, cifar100, imagenet, imagefolder |
| `--data_dir` | str | ./data | 数据目录 |
| `--image_size` | int | 256 | 输入图像尺寸 (FLUX VAE 最佳 256) |
| `--dit_size` | str | B | DiT 规模: T, S, B |
| `--patch_size` | int | 2 | DiT patch 大小 |
| `--num_timesteps` | int | 1000 | 扩散时间步数 |
| `--beta_schedule` | str | cosine | 噪声调度: linear, cosine |
| `--pred_type` | str | v | 预测目标: eps, v |
| `--min_snr_gamma` | float | 5.0 | min-SNR-γ 截断值 |
| `--svd_aux_weight` | float | 0.0 | SVD 辅助损失权重 (默认关闭) |
| `--epochs` | int | 500 | 训练轮数 |
| `--batch_size` | int | 256 | 批大小 |
| `--lr` | float | 2e-4 | 初始学习率 |
| `--min_lr` | float | 1e-6 | 最小学习率 (cosine annealing 终点) |
| `--warmup_epochs` | int | 20 | linear warmup 轮数 |
| `--ema_decay` | float | 0.9999 | EMA 衰减率 |
| `--grad_accum` | int | 1 | 梯度累积步数 |
| `--max_train_samples` | int | 0 | 限制训练样本数 (0=全部) |
| `--fid_interval` | int | 25 | FID 评估间隔 (epochs) |
| `--fid_n_samples` | int | 2048 | FID 评估生成样本数 |
| `--ddim_steps` | int | 50 | DDIM 采样步数 |
| `--output_dir` | str | ./checkpoints/flux_latent_diffusion | 输出目录 |
| `--vis_interval` | int | 5 | 可视化间隔 (epochs) |
| `--save_interval` | int | 25 | checkpoint 保存间隔 |
| `--resume` | str | None | 恢复训练的 checkpoint 路径 |
| `--latent_cache_dir` | str | ./latent_cache | latent 缓存目录 |
| `--use_amp` | flag | True | 使用 bf16 混合精度 |
| `--compile` | flag | False | 使用 torch.compile 加速 |

---

## 七、输出文件说明

```
checkpoints/flux_latent_diffusion/
  best.pt              # 最佳 FID 的 checkpoint
  epoch0025.pt         # 定期保存
  epoch0050.pt
  ...
  final.pt             # 最终 epoch
  vis/
    samples_epoch0005.png   # DDIM 采样可视化 (8×8 网格)
    samples_epoch0010.png
    ...

latent_cache/
  cifar10_train_256.pt       # CIFAR-10 训练集 latent 缓存
  imagenet_train_256.pt      # ImageNet 训练集 latent 缓存
  imagenet_train_256_50000.pt  # max_train_samples=50000 时的缓存
```

**Checkpoint 内容：**
```python
{
    'epoch': int,
    'model_state_dict': dict,     # DiT 模型权重
    'optimizer_state_dict': dict, # AdamW 优化器状态
    'ema_state_dict': dict,       # EMA 权重 (评估用)
    'ema_decay': float,
    'fid': float,                 # 当前最佳 FID
    'channel_mean': Tensor,       # (16,) 逐通道均值
    'channel_std': Tensor,        # (16,) 逐通道标准差
}
```

---

## 八、项目文件结构

```
diffusion_torch/diffusion_torch/
  train_flux_latent_diffusion.py   # 主训练脚本 (FLUX VAE + DiT)
  models/
    dit.py                         # DiT 架构 (Transformer + adaLN-Zero)
    vqvae.py                       # 多尺度 VQVAE (旧, 本流程不使用)
  latent_diffusion.py              # 扩散过程 (噪声调度/损失/DDIM采样)
  ema.py                           # EMA + checkpoint 工具
  fid_utils.py                     # Inception-V3 特征提取
  classifier_metrics_numpy.py      # FID 数值计算 (NumPy)
  data/                            # 数据目录
    cifar-10-batches-py/           # CIFAR-10 数据
    imagenet_hf_cache/             # ImageNet HuggingFace 缓存
```

---

## 九、CIFAR-10 实验结果

### 训练配置
- DiT-B (130M params), FLUX VAE (冻结)
- 256×256 编码 → 32×32×16 latent → 逐通道归一化
- v-prediction, cosine schedule, min-SNR-γ=5.0
- AdamW (lr=2e-4), bf16 AMP, batch_size=256

### FID 曲线 (2048 样本, 32×32)

| Epoch | FID | 说明 |
|-------|-----|------|
| 25 | 428 | 初期噪声 |
| 50 | 336 | 开始学到结构 |
| 100 | 182 | 明显改善 |
| 150 | 87 | 快速收敛阶段 |
| 200 | 57 | |
| 250 | 45 | |
| 300 | 43 | 开始收敛 |
| **350** | **42.64** | **最佳 FID** |
| 375 | 43.13 | 略有上升 |

> FID 在 ~350 epoch 后收敛到 ~42-43，可能因 CIFAR 的 32×32 分辨率上采样到 256×256 存在信息损失。ImageNet 原生 256×256 应有更好的效果。

---

## 十、理论背景

### 10.1 为什么用预训练 VAE？

传统两阶段方法需要自训练 VQVAE → 再训练扩散模型。使用 FLUX.1-dev 预训练 VAE 的优势：

1. **跳过 Stage 1**: 无需训练 VQVAE，直接获得高质量连续潜在空间
2. **更好的重建质量**: FLUX VAE 在 256×256 上 PSNR > 47dB
3. **16 通道 latent**: 比 Stable Diffusion 的 4 通道保留更多信息
4. **连续 latent**: 不需要向量量化，避免 codebook 坍缩等问题

### 10.2 v-prediction vs ε-prediction

v-prediction 定义: $v = \sqrt{\bar\alpha_t} \cdot \epsilon - \sqrt{1-\bar\alpha_t} \cdot x_0$

优势：
- 在高 SNR 端 (小 t) 不退化，训练信号更稳定
- 与 min-SNR-γ 结合效果好
- Progressive Distillation 友好

### 10.3 min-SNR-γ 加权

标准 MSE 损失在不同噪声水平下梯度量级差异大。min-SNR-γ 通过截断：

$$w(t) = \frac{\min(\text{SNR}(t), \gamma)}{\text{SNR}(t)}$$

平衡高低噪声水平的训练信号，$\gamma=5.0$ 是推荐默认值。

### 10.4 为什么需要逐通道归一化？

扩散模型的前向过程假设 $z_0 \sim \mathcal{N}(0, I)$，噪声 $\epsilon \sim \mathcal{N}(0, I)$。
如果 $z_0$ 的各通道分布差异大 (如 FLUX VAE latent)，加噪/去噪过程中
不同通道的信噪比不一致，影响训练效率和样本质量。

逐通道归一化确保所有通道在同一尺度上，扩散过程假设成立。
