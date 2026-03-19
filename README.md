# SpectralDiff V2 — 多尺度 VQVAE + 潜空间频谱扩散模型

## 一、项目概述

**SpectralDiff V2** 是一个基于 **VAR (Visual Autoregressive)** 思想的两阶段图像生成模型：

- **Stage 1**: 训练多尺度 VQVAE，将图像压缩为离散潜码
- **Stage 2**: 在 VQVAE 潜空间中训练 DiT 扩散模型，用 SVD 频域辅助损失引导

:::::
> **粗尺度** (1×1, 2×2) 自然捕获 **低频信息** (轮廓、外观) = SVD 的角色
> **细尺度** (4×4, 8×8) 捕获 **高频信息** (纹理、光照) = Diffusion 的角色

reset: **CIFAR-10**, **CIFAR-100**, **ImageNet**

---

## 二、环境要求

```
Python >= 3.8
PyTorch >= 2.0
torchvision
numpy
```

#reset 
:
```bash
pip install torch torchvision numpy
```

---

## 三、架构设计

### 3.1 总体流程

```
 x ∈ [B, 3, H, H]
    │
    ▼

  Multi-Scale VQVAE           │
  Encoder → 下采样            │
  → MultiScaleQuantizer       │
    (scales=[1,2,4,8,...])    │
  → 潜码 z ∈ [B, D, H', H'] │

    │           │
    │ z0        │ 多尺度上下文 ctx
    ▼           ▼

  Latent DiT                  │
  + SVD 频域辅助损失           │
  + 多尺度交叉注意力 (VAR)    │
  v-prediction / min-SNR-γ   │

    │
    ▼

  VQVAE Decoder               │
  潜码 → 重建图像              │

```

### 3.2 多尺度 VQVAE (参考 VAR)

 z 按不同尺度做自适应池化，每个尺度独立向量量化（残差式），量化结果上采样累加:

```
z ∈ [B, D, H', W']
  ├─ AdaptiveAvgPool → 1×1 → VQ → 上采样 → z_hat_1   (最粗: 全局结构)
  ├─ AdaptiveAvgPool → 2×2 → VQ(残差) → 上采样 → z_hat_2
  ├─ AdaptiveAvgPool → 4×4 → VQ(残差) → 上采样 → z_hat_4
  └─ 无池化 → 8×8 → VQ(残差) → z_hat_8              (最细: 纹理细节)
  → z_hat = z_hat_1 + z_hat_2 + z_hat_4 + z_hat_8
```

:
- EMA 编码本更新 (不走梯度, Laplace 平滑防止坍缩)
- 直通估计器 (Straight-Through Estimator) 传递梯度
- 每个尺度独立 codebook

### 3.3 Latent DiT + 交叉注意力

DiT 在 VQVAE 潜空间中运行，通过 **多尺度交叉注意力** 接收 VQVAE 粗尺度条件:

- Q ← DiT 隐状态 (潜码 tokens)
- K, V ← 多尺度上下文 (VQVAE 各尺度量化结果展平拼接)
- 每隔 2 个 DiTBlock 加一层交叉注意力
- 零初始化门控 (cross_gate) 保证初始不破坏训练
- QK-Norm 防止注意力发散

### 3.4 损失函数

```
L_total = w_snr(t) · L_main + λ_svd · L_svd

--------:
  L_main = MSE(v_pred, v_target)       (v-prediction)
  w_snr(t) = min(SNR(t), γ) / SNR(t)  (min-SNR-γ 加权)
  L_svd = ||P_L(ẑ₀ - z₀)||² + ||(I-P_L)(ẑ₀ - z₀)||²  (SVD 投影误差)
  P_L = U_k · U_k^T                    (前 k 个奇异向量投影)
```

### 3.5 采样方法

| 方法 | 步数 | 特点 |
|------|------|------|
| DDIM | 50-250 | 快速, 确定性 |
| DDPM | 1000 | 完整马尔可夫链, 最佳质量 |
| 多尺度级联 (Cascade) | 50-250 | 先去噪70%取低频 → SVD截断 → 重加噪继续精修 |

---

## 四、模型规模

### VQVAE

| 配置 | 推荐场景 | hidden_dim | latent_dim | ch_mult | 编码本 | 多尺度 |
|------|---------|-----------|-----------|---------|--------|--------|
| `small` | CIFAR (32×32) | 128 | 32 | [1,2,4] | 512 | [1,2,4,8] |
| `base` | CIFAR/ImageNet-64 | 256 | 32 | [1,2,4] | 1024 | [1,2,4,8] |
| `large` | ImageNet (64-256) | 256 | 64 | [1,2,4,8] | 2048 | [1,2,4,8,16] |

### DiT

| 配置 | 参数量 | 层数 | 头数 | 隐藏维度 | 交叉注意力层 |
|------|--------|------|------|---------|------------|
| `T` (Tiny) | ~4.7M | 6 | 6 | 192 | 3 |
| `S` (Small) | ~16M | 12 | 6 | 384 | 6 |
| `B` (Base) | ~59M | 12 | 12 | 768 | 6 |

---

## 五、数据集准备

### CIFAR-10 / CIFAR-100

pip install h5py :
```bash
# 自动下载到 ./data 目录
python3 train_vqvae.py --dataset cifar10 --data_dir ./data
```

### ImageNet

:
```
/path/to/imagenet/
 train/
   ├── n01440764/
   │   ├── n01440764_10026.JPEG
   │   └── ...
   ├── n01443537/
   └── ... (共 1000 个类别)
 val/
    ├── n01440764/
    └── ... (共 1000 个类别)
```

---

## 六、训练指南

### Stage 1: 训练 VQVAE

#### CIFAR-10 (32×32)

```bash
cd diffusion_torch/diffusion_torch

python3 train_vqvae.py \
    --dataset cifar10 \
    --data_dir ./data \
    --image_size 32 \
    --model_size small \
    --codebook_size 512 \
    --latent_dim 32 \
    --commitment_weight 0.25 \
    --epochs 200 \
    --batch_size 128 \
    --lr 1e-3 \
    --ema_decay 0.999 \
    --vis_interval 10 \
    --save_interval 20 \
    --output_dir ./checkpoints/vqvae_cifar10
```

#### ImageNet 64×64

```bash
python3 train_vqvae.py \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --image_size 64 \
    --model_size large \
    --codebook_size 2048 \
    --latent_dim 64 \
    --commitment_weight 0.25 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --ema_decay 0.999 \
    --vis_interval 5 \
    --save_interval 10 \
    --output_dir ./checkpoints/vqvae_imagenet64
```

#### ImageNet 256×256

```bash
python3 train_vqvae.py \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --image_size 256 \
    --model_size large \
    --codebook_size 2048 \
    --latent_dim 64 \
    --commitment_weight 0.25 \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --ema_decay 0.999 \
    --vis_interval 5 \
    --save_interval 10 \
    --output_dir ./checkpoints/vqvae_imagenet256
```

**关键监控指标:**
- `PSNR`: ≥ 25dB 表示重建质量可接受 (CIFAR), ≥ 22dB (ImageNet)
- `Codebook使用率`: ≥ 80% (低于此值说明 codebook 坍缩)
- `loss_recon`: 应持续下降
- `loss_commit`: 应保持稳定在 0.1-1.0 范围

**输出文件:**
```
checkpoints/vqvae_cifar10/
 best.pt          # 最佳 PSNR 的 checkpoint (含 EMA 权重)
 final.pt         # 最终 epoch 的 checkpoint
 epoch0020.pt     # 定期保存的 checkpoint
 vis/
    ├── recon_epoch0010.png  # 重建可视化
    └── ...
```

---

### Stage 2: 训练潜空间扩散

#### CIFAR-10

```bash
python3 train_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_cifar10/best.pt \
    --vqvae_size small \
    --dataset cifar10 \
    --data_dir ./data \
    --image_size 32 \
    --dit_size S \
    --pred_type v \
    --beta_schedule cosine \
    --num_timesteps 1000 \
    --min_snr_gamma 5.0 \
    --svd_aux_weight 0.1 \
    --use_svd_aux \
    --use_multiscale_cond \
    --cond_scales 1 2 \
    --epochs 500 \
    --batch_size 128 \
    --lr 1e-4 \
    --ema_decay 0.9999 \
    --ddim_steps 50 \
    --fid_interval 50 \
    --fid_n_samples 5000 \
    --vis_interval 10 \
    --save_interval 50 \
    --output_dir ./checkpoints/latent_diff_cifar10
```

#### ImageNet 64×64

```bash
python3 train_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_imagenet64/best.pt \
    --vqvae_size large \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --image_size 64 \
    --dit_size B \
    --pred_type v \
    --beta_schedule cosine \
    --num_timesteps 1000 \
    --min_snr_gamma 5.0 \
    --svd_aux_weight 0.1 \
    --use_svd_aux \
    --use_multiscale_cond \
    --cond_scales 1 2 \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-4 \
    --ema_decay 0.9999 \
    --ddim_steps 50 \
    --fid_interval 50 \
    --fid_n_samples 5000 \
    --vis_interval 10 \
    --save_interval 50 \
    --output_dir ./checkpoints/latent_diff_imagenet64
```

#### ImageNet 256×256

```bash
python3 train_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_imagenet256/best.pt \
    --vqvae_size large \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --image_size 256 \
    --dit_size B \
    --pred_type v \
    --beta_schedule cosine \
    --num_timesteps 1000 \
    --min_snr_gamma 5.0 \
    --svd_aux_weight 0.1 \
    --use_svd_aux \
    --use_multiscale_cond \
    --cond_scales 1 2 \
    --epochs 300 \
    --batch_size 8 \
    --lr 5e-5 \
    --ema_decay 0.9999 \
    --ddim_steps 50 \
    --fid_interval 100 \
    --fid_n_samples 5000 \
    --vis_interval 10 \
    --save_interval 50 \
    --output_dir ./checkpoints/latent_diff_imagenet256
```

**关键参数说明:**
| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `--pred_type v` | v-prediction, 高 SNR 端更稳定 | `v` (推荐) 或 `eps` |
| `--beta_schedule cosine` | 更均匀的信噪比分布 | `cosine` |
| `--min_snr_gamma 5.0` | 平衡不同噪声水平的训练信号 | 5.0 |
| `--svd_aux_weight 0.1` | SVD 辅助损失权重 | 0.05-0.2 |
| `--cond_scales 1 2` | 用粗尺度 (1×1, 2×2) 做条件 | [1, 2] |
| `--use_multiscale_cond` | 开启 VAR 风格多尺度条件 | 默认开启 |
| `--no_svd_aux` | 关闭 SVD 辅助损失 | 调试时可用 |
| `--no_multiscale_cond` | 关闭多尺度条件 (无条件生成) | 消融实验 |

**输出文件:**
```
checkpoints/latent_diff_cifar10/
 best.pt          # 最佳 FID 的 checkpoint
 final.pt         # 最终 checkpoint
 epoch0050.pt     # 定期保存
 vis/
    ├── samples_epoch0010.png  # DDIM 采样可视化
    └── ...
```

**注意:** Stage 2 的 `--image_size` 和 `--vqvae_size` 必须与 Stage 1 训练时一致。

---

## 七、推理 (采样) 指南

### 7.1 DDIM 采样 (快速, 推荐)

```bash
# CIFAR-10
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_cifar10/best.pt \
    --vqvae_size small \
    --dit_ckpt ./checkpoints/latent_diff_cifar10/best.pt \
    --dit_size S \
    --image_size 32 \
    --mode ddim \
    --ddim_steps 50 \
    --n_samples 64 \
    --batch_size 64 \
    --output_dir ./samples/cifar10_ddim

# ImageNet 64×64
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_imagenet64/best.pt \
    --vqvae_size large \
    --dit_ckpt ./checkpoints/latent_diff_imagenet64/best.pt \
    --dit_size B \
    --image_size 64 \
    --mode ddim \
    --ddim_steps 100 \
    --n_samples 64 \
    --output_dir ./samples/imagenet64_ddim
```

### 7.2 多尺度级联采样

```bash
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_cifar10/best.pt \
    --vqvae_size small \
    --dit_ckpt ./checkpoints/latent_diff_cifar10/best.pt \
    --dit_size S \
    --image_size 32 \
    --mode cascade \
    --ddim_steps 50 \
    --n_samples 64 \
    --output_dir ./samples/cifar10_cascade
```

:
```
Phase A: 噪声 → DDIM 去噪 70% → 得到粗糙 z₀
         ↓
    SVD 截断: z₀ → 保留前 k 奇异值 → z_low (低频结构)
         ↓
Phase B: z_low → 重新加噪到 t_mid → DDIM 继续去噪到 t=0 → z_final
         ↓
    VQVAE Decode → 输出图像
```

### 7.3 DDPM 完整采样

```bash
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_cifar10/best.pt \
    --vqvae_size small \
    --dit_ckpt ./checkpoints/latent_diff_cifar10/best.pt \
    --dit_size S \
    --image_size 32 \
    --mode ddpm \
    --n_samples 16 \
    --output_dir ./samples/cifar10_ddpm
```

**采样输出:**
```
samples/cifar10_ddim/
 ddim_samples.png       # 网格可视化 (最多 64 张)
 ddim_images/
    ├── 00000.png          # 单张图像
    ├── 00001.png
    └── ...
```

---

## 八、测评指南

### 8.1 FID + IS 评估

```bash
# CIFAR-10 FID-10k + IS
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_cifar10/best.pt \
    --vqvae_size small \
    --dit_ckpt ./checkpoints/latent_diff_cifar10/best.pt \
    --dit_size S \
    --image_size 32 \
    --dataset cifar10 \
    --data_dir ./data \
    --mode eval \
    --n_gen 10000 \
    --ddim_steps 250 \
    --batch_size 64 \
    --output_dir ./eval/cifar10

# ImageNet 64×64 FID-50k
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_imagenet64/best.pt \
    --vqvae_size large \
    --dit_ckpt ./checkpoints/latent_diff_imagenet64/best.pt \
    --dit_size B \
    --image_size 64 \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --mode eval \
    --n_gen 50000 \
    --ddim_steps 250 \
    --batch_size 64 \
    --output_dir ./eval/imagenet64
```

**评估输出:**
```
eval/cifar10/
 eval_results.txt   # FID, IS 数值
 eval_samples.png   # 样本可视化
```

`eval_results.txt` 内容:
```
FID: 12.3456
IS: 8.5432 ± 0.2345
n_gen: 10000
ddim_steps: 250
```

### 8.2 可视化 (频率分解 + 采样对比)

```bash
# CIFAR-10
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_cifar10/best.pt \
    --vqvae_size small \
    --dit_ckpt ./checkpoints/latent_diff_cifar10/best.pt \
    --dit_size S \
    --image_size 32 \
    --dataset cifar10 \
    --data_dir ./data \
    --mode visualize \
    --ddim_steps 50 \
    --output_dir ./vis/cifar10

# ImageNet 64×64
python3 sample_latent_diffusion.py \
    --vqvae_ckpt ./checkpoints/vqvae_imagenet64/best.pt \
    --vqvae_size large \
    --dit_ckpt ./checkpoints/latent_diff_imagenet64/best.pt \
    --dit_size B \
    --image_size 64 \
    --dataset imagenet \
    --data_dir /path/to/imagenet \
    --mode visualize \
    --ddim_steps 50 \
    --output_dir ./vis/imagenet64
```

**可视化输出:**
```
vis/cifar10/
 freq_decomposition.png  # 多尺度频率分解
   行1: 原图
   行2: 完整重建
   行3: 低频 (scale 1+2)
   行4: 中频 (scale 4)
   行5: 高频 (scale 8)
   行6+: 各尺度独立重建
 ddim_vs_cascade.png     # DDIM vs 级联采样对比
    行1: DDIM 样本
    行2: 级联采样样本
```

---

## 九、完整参数参考

### train_vqvae.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | cifar10 | 数据集: cifar10, cifar100, imagenet |
| `--data_dir` | str | ./data | 数据目录 (ImageNet 需含 train/ 和 val/) |
| `--image_size` | int | 32 | 图像尺寸 (CIFAR=32, ImageNet=64/128/256) |
| `--model_size` | str | small | VQVAE 规模: small, base, large |
| `--hidden_dim` | int | 随model_size | Encoder/Decoder 隐藏通道数 (覆盖工厂默认) |
| `--latent_dim` | int | 随model_size | 潜在维度 (覆盖工厂默认) |
| `--codebook_size` | int | 随model_size | 每个尺度的 codebook 大小 (覆盖工厂默认) |
| `--commitment_weight` | float | 0.25 | commitment loss 权重 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 128 | 批大小 |
| `--lr` | float | 1e-3 | 初始学习率 |
| `--min_lr` | float | 1e-5 | 最小学习率 |
| `--warmup_epochs` | int | 5 | warmup 轮数 |
| `--ema_decay` | float | 0.999 | EMA 衰减率 |
| `--output_dir` | str | ./checkpoints/vqvae | 输出目录 |
| `--log_interval` | int | 1 | 日志打印间隔 (epochs) |
| `--vis_interval` | int | 10 | 可视化+验证间隔 |
| `--save_interval` | int | 20 | checkpoint 保存间隔 |
| `--resume` | str | None | 恢复训练的 checkpoint 路径 |

### train_latent_diffusion.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--vqvae_ckpt` | str | **必填** | Stage 1 VQVAE checkpoint 路径 |
| `--vqvae_size` | str | small | VQVAE 规模: small, base, large |
| `--dataset` | str | cifar10 | 数据集: cifar10, cifar100, imagenet |
| `--data_dir` | str | ./data | 数据目录 |
| `--image_size` | int | 32 | 图像尺寸 (须与 Stage 1 一致) |
| `--dit_size` | str | S | DiT 规模: T, S, B |
| `--num_timesteps` | int | 1000 | 扩散步数 |
| `--beta_schedule` | str | cosine | 噪声调度: linear, cosine |
| `--pred_type` | str | v | 预测目标: eps, v |
| `--min_snr_gamma` | float | 5.0 | min-SNR-γ 加权 |
| `--svd_aux_weight` | float | 0.1 | SVD 辅助损失权重 |
| `--use_svd_aux` | flag | True | 开启 SVD 辅助损失 |
| `--no_svd_aux` | flag | - | 关闭 SVD 辅助损失 |
| `--use_multiscale_cond` | flag | True | 开启多尺度条件 |
| `--no_multiscale_cond` | flag | - | 关闭多尺度条件 |
| `--cond_scales` | int[] | [1, 2] | 条件化的粗尺度列表 |
| `--epochs` | int | 500 | 训练轮数 |
| `--batch_size` | int | 128 | 批大小 |
| `--lr` | float | 1e-4 | 初始学习率 |
| `--min_lr` | float | 1e-6 | 最小学习率 |
| `--warmup_epochs` | int | 10 | warmup 轮数 |
| `--ema_decay` | float | 0.9999 | EMA 衰减率 |
| `--fid_interval` | int | 50 | FID 评估间隔 |
| `--fid_n_samples` | int | 5000 | FID 评估样本数 |
| `--ddim_steps` | int | 50 | DDIM 采样步数 |
| `--output_dir` | str | ./checkpoints/latent_diffusion | 输出目录 |
| `--vis_interval` | int | 10 | 可视化间隔 |
| `--save_interval` | int | 50 | checkpoint 保存间隔 |
| `--resume` | str | None | 恢复训练路径 |

### sample_latent_diffusion.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--vqvae_ckpt` | str | **必填** | VQVAE checkpoint |
| `--vqvae_size` | str | small | VQVAE 规模: small, base, large |
| `--dit_ckpt` | str | **必填** | DiT checkpoint |
| `--dit_size` | str | S | DiT 规模: T, S, B |
| `--use_cross_attn` | flag | True | 使用交叉注意力 |
| `--no_cross_attn` | flag | - | 关闭交叉注意力 |
| `--mode` | str | ddim | 模式: ddim, cascade, ddpm, eval, visualize |
| `--ddim_steps` | int | 50 | DDIM 步数 |
| `--n_samples` | int | 64 | 生成样本数 (ddim/cascade/ddpm) |
| `--n_gen` | int | 10000 | eval 模式生成数量 |
| `--batch_size` | int | 64 | 采样批大小 |
| `--num_timesteps` | int | 1000 | 扩散步数 |
| `--beta_schedule` | str | cosine | 噪声调度 |
| `--pred_type` | str | v | 预测目标 |
| `--dataset` | str | cifar10 | 数据集 (eval/visualize 需要) |
| `--data_dir` | str | ./data | 数据目录 |
| `--image_size` | int | 32 | 图像尺寸 |
| `--output_dir` | str | ./samples | 输出目录 |

---

## 十、推荐配置

### CIFAR-10 (32×32, 50k 训练样本)

| 阶段 | model_size | dit_size | batch | lr | epochs | 预计显存 |
|------|-----------|---------|-------|-----|--------|---------|
| Stage 1 | small | - | 128 | 1e-3 | 200 | ~2GB |
| Stage 2 | - | S | 128 | 1e-4 | 500 | ~3GB |

### ImageNet 64×64 (1.28M 训练样本)

| 阶段 | model_size | dit_size | batch | lr | epochs | 预计显存 |
|------|-----------|---------|-------|-----|--------|---------|
| Stage 1 | large | - | 64 | 1e-3 | 100 | ~8GB |
| Stage 2 | - | B | 64 | 1e-4 | 300 | ~12GB |

### ImageNet 256×256

| 阶段 | model_size | dit_size | batch | lr | epochs | 预计显存 |
|------|-----------|---------|-------|-----|--------|---------|
| Stage 1 | large | - | 16 | 5e-4 | 100 | ~16GB |
| Stage 2 | - | B | 8 | 5e-5 | 300 | ~20GB |

---

## 十一、项目文件说明

```
diffusion_torch/diffusion_torch/
 models/
   ├── vqvae.py              (762行)  多尺度 VQVAE (VAR 风格)
   └── dit.py                (630行)  DiT + 交叉注意力
 latent_diffusion.py        (583行)  潜空间扩散过程
 ema.py                     (111行)  EMA 滑动平均 + Checkpoint
 train_vqvae.py             (465行)  Stage 1 训练
 train_latent_diffusion.py  (562行)  Stage 2 训练
 sample_latent_diffusion.py (508行)  采样与评估
 fid_utils.py                        FID 计算工具
```

---

## 十二、创新点总结

| # | 创新点 | 说明 |
|---|--------|------|
| 1 | **多尺度 VQVAE (VAR 启发)** | 残差式多分辨率量化, 自然形成频率分解 |
| 2 | **潜空间 SVD 辅助损失** | 在 VQVAE latent 空间做 SVD 投影, 引导低频/高频分别学习 |
| 3 | **多尺度交叉注意力** | DiT 通过粗尺度条件接受全局结构信息 |
| 4 | **多尺度级联采样** | 先去噪取低频 → SVD 截断 → 重加噪精修高频 |
| 5 | **v-prediction + min-SNR-γ** | 稳定训练, 平衡不同噪声水平的梯度信号 |
| 6 | **EMA 编码本** | 向量量化训练稳定, 无需大 codebook 也能避免坍缩 |

---

## 十三、自检清单

| 检查项 | 结果 |
|--------|------|
| VQVAE 32×32: shape (3,32,32) → z(32,8,8) → recon(3,32,32) | ✅ |
| VQVAE 64×64: shape (3,64,64) → z(64,8,8) → recon(3,64,64) | ✅ |
| VQVAE 128×128: shape (3,128,128) → z(32,32,32) → recon(3,128,128) | ✅ |
| VQVAE 256×256: shape (3,256,256) → z(64,32,32) → recon(3,256,256) | ✅ |
| DiT latent_size=8 前向传播 | ✅ |
| DiT latent_size=32 前向传播 (ImageNet) | ✅ |
| v-prediction 数学可逆性 (误差 < 5e-7) | ✅ |
| SVD 截断精确低秩 | ✅ |
| 训练损失梯度回传 (130/130 DiT 参数) | ✅ |
| DDIM 采样正确输出 | ✅ |
| 多尺度级联采样正确输出 | ✅ |
| DDPM 采样正确输出 | ✅ |
| GPU 完整集成测试 (VQVAE→EMA→DiT→采样) | ✅ |
| EMA apply_shadow/restore 正确性 | ✅ |
| train_vqvae.py --dataset imagenet argparse | ✅ |
| train_latent_diffusion.py --dataset imagenet argparse | ✅ |
| sample_latent_diffusion.py --dataset imagenet argparse | ✅ |
| 全部中文注释覆盖 | ✅ |
