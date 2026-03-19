# **SpectralDiff V2 — VAR 风格多尺度 VQVAE + 潜空间频谱扩散模型**

## 一、项目名称与核心思想

**SpectralDiff V2: Multi-Scale VQVAE + Latent Spectral Diffusion**

:  **参考 VAR (Visual Autoregressive) 模型的多尺度思想，将 SVD 频域分解扩散迁移到 VQVAE 潜空间中。**

### V1 → V2 演进

| | V1 (像素空间) | V2 (潜空间) |
|---|---|---|
| **骨干** | DiT 在像素空间 | DiT 在 VQVAE 潜空间 |
| **压缩** | 无 | Multi-Scale VQVAE (4x 下采样) |
| **条件机制** | 无 | 多尺度交叉注意力 (VAR 启发) |
| **编码本** | 无 | EMA 向量量化 + Laplace 平滑 |
| **SVD 辅助** | 像素空间 SVD | 潜空间 SVD |
| **采样** | DDIM / 级联 | DDIM / DDPM / 多尺度级联 |

## 二、架构总览

```
 x ∈ [B,3,32,32]
    │
    ▼

  Stage 1: Multi-Scale    │
  VQVAE (冻结后使用)       │
  Encoder → 4x 下采样     │
  → MultiScaleQuantizer   │
    (scales=[1,2,4,8])    │
  → 潜码 z ∈ [B,32,8,8]  │

    │         │
    │ z0      │ 多尺度上下文 ctx
    ▼         ▼

  Stage 2: Latent DiT     │
  + SVD 频域辅助损失       │
  v-prediction / min-SNR  │
  CrossAttention (VAR)    │
  → 预测潜码 ẑ0           │

    │
    ▼

  VQVAE Decoder           │
  潜码 → 重建图像          │
  ẑ0 → x̂ ∈ [B,3,32,32]  │

```

## 三、Multi-Scale VQVAE (参考 VAR)

### 3.1 多尺度量化器

VAR 模型的核心洞察：图像信息天然具有多尺度结构。我们的 `MultiScaleQuantizer` 实现：

1. 对潜码 z 按不同尺度做自适应池化（scales = [1, 2, 4, 8]）
2. 每个尺度独立做向量量化（残差式：每级量化上一级的残差）
3. 量化结果上采样回原始分辨率后累加

```
z ∈ [B,D,H,W]
  ├─ AdaptiveAvgPool → 1×1 → VQ → 上采样 → z_hat_1
  ├─ AdaptiveAvgPool → 2×2 → VQ(residual) → 上采样 → z_hat_2
  ├─ AdaptiveAvgPool → 4×4 → VQ(residual) → 上采样 → z_hat_4
  └─ AdaptiveAvgPool → 8×8 → VQ(residual) → 上采样 → z_hat_8
  → z_hat = z_hat_1 + z_hat_2 + z_hat_4 + z_hat_8
```

### 3.2 EMA 编码本更新

- 指数移动平均更新编码本向量（不走梯度）
- Laplace 平滑防止编码本坍缩
- 直通估计器 (Straight-Through Estimator) 传递梯度

### 3.3 模型规模

| 配置 | 参数量 | 隐藏维度 | 编码本大小 | 潜码维度 |
|------|--------|---------|-----------|---------|
| VQVAE_Small | ~47M | 128 | 256 | 32 |
| VQVAE_Base | ~98M | 192 | 512 | 64 |
| VQVAE_Large | ~188M | 256 | 1024 | 128 |

## 四、Latent DiT + 多尺度交叉注意力

DiT 在 VQVAE 潜空间中操作，核心改进：

### 4.1 CrossAttention (VAR 风格条件注入)

```
Q ← 潜码 tokens (DiT 隐状态)
K, V ← 多尺度上下文 (VQVAE 各尺度量化结果展平拼接)
```

- 每隔 `cross_attn_every_n` 个 DiTBlock 添加一层交叉注意力
- QK-Norm 防止注意力发散
- 零初始化门控 (cross_gate) 保证初始阶段不破坏预训练

### 4.2 模型规模

| 配置 | 参数量 | 层数 | 头数 | 隐藏维度 | 交叉注意力层 |
|------|--------|------|------|---------|------------|
| LatentDiT_T | ~4.7M | 6 | 6 | 192 | 3 |
| LatentDiT_S | ~16M | 12 | 6 | 384 | 6 |
| LatentDiT_B | ~59M | 12 | 12 | 768 | 6 |

## 五、频域辅助损失 (潜空间版)

--------执行 SVD 分解，引导扩散模型：

$$L_{total} = w_{snr}(t) \cdot L_{main} + \lambda_{svd} \cdot L_{svd}$$

--------：
- $L_{main}$: v-prediction MSE 损失 (或 eps-prediction)
- $w_{snr}(t) = \min(\text{SNR}(t), \gamma) / \text{SNR}(t)$: min-SNR-γ 加权
- $L_{svd} = \|P_L(\hat{z}_0 - z_0)\|^2 + \|(I-P_L)(\hat{z}_0 - z_0)\|^2$: SVD 频率投影误差
- $P_L = U_k U_k^T$: 潜空间 SVD 前 k 个奇异向量投影

## 六、采样方法

### 6.1 DDIM 采样
 DDIM，在潜空间去噪后用 VQVAE Decoder 解码。

### 6.2 多尺度级联采样 (Multiscale Cascade)
```
Phase A: noise → DDIM 去噪 70% → 得到粗糙 z0_pred
         ↓
    SVD 截断: z0_pred → 保留前 k 奇异值 → z_low (低频结构)
         ↓
Phase B: z_low → 重新加噪到 t_mid → DDIM 去噪到 t=0 → z_final
         ↓
    VQVAE Decode → 输出图像
```

### 6.3 DDPM 采样
.    T 步马尔可夫链采样。

## 七、项目文件说明

### V2 文件 (当前版本)

| 文件 | 行数 | 功能 |
|------|------|------|
| `models/vqvae.py` | 762 | VAR 风格 Multi-Scale VQVAE |
| `models/dit.py` | 630 | DiT + 多尺度交叉注意力 |
| `latent_diffusion.py` | 583 | 潜空间扩散: SVD 辅助损失 + 采样 |
| `train_vqvae.py` | 430 | Stage 1: VQVAE 训练脚本 |
| `train_latent_diffusion.py` | 524 | Stage 2: 潜空间扩散训练脚本 |
| `sample_latent_diffusion.py` | 477 | 推理与评估脚本 |
| `ema.py` | 111 | EMA 滑动平均 + Checkpoint 工具 |

### V1 文件 (保留)

| 文件 | 功能 |
|------|------|
| `spectral_diffusion.py` | V1 像素空间扩散 |
| `train_spectral_diffusion.py` | V1 训练脚本 |
| `sample_spectral_diffusion.py` | V1 采样脚本 |

## 八、使用命令

### Stage 1: 训练 VQVAE

```bash
cd diffusion_torch/diffusion_torch

python3 train_vqvae.py \
    --dataset cifar10 --epochs 200 --batch 128 --lr 1e-3 \
    --model_size small --image_size 32 \
    --codebook_size 256 --latent_dim 32 --commit_weight 0.25 \
    --save_every 20 --eval_every 10 \
    --out_dir ./outputs_vqvae
```

### Stage 2: 训练潜空间扩散

```bash
python3 train_latent_diffusion.py \
    --vqvae_ckpt outputs_vqvae/best_ckpt.pt \
    --dataset cifar10 --epochs 500 --batch 128 --lr 2e-4 \
    --model_size T --pred_type v --beta_schedule cosine \
    --timesteps 1000 --k_truncate 4 --lambda_svd 0.1 \
    --min_snr_gamma 5.0 --sample_steps 50 \
    --save_every 10 --eval_fid --eval_every 25 \
    --out_dir ./outputs_latent
```

### 推理 — DDIM 采样

```bash
python3 sample_latent_diffusion.py \
    --vqvae_ckpt outputs_vqvae/best_ckpt.pt \
    --dit_ckpt outputs_latent/best_ckpt.pt \
    --mode ddim --n 256 --sample_steps 250 \
    --out gen_ddim.png
```

### 推理 — 多尺度级联采样

```bash
python3 sample_latent_diffusion.py \
    --vqvae_ckpt outputs_vqvae/best_ckpt.pt \
    --dit_ckpt outputs_latent/best_ckpt.pt \
    --mode cascade --n 256 --sample_steps 250 \
    --k_truncate 4 --cascade_t_frac 0.3 \
    --out gen_cascade.png
```

### 完整评估 (FID-50k)

```bash
python3 sample_latent_diffusion.py \
    --vqvae_ckpt outputs_vqvae/best_ckpt.pt \
    --dit_ckpt outputs_latent/best_ckpt.pt \
    --mode eval --fid_num 50000 --sample_steps 250 \
    --eval_cascade --compute_is
```

### 可视化

```bash
python3 sample_latent_diffusion.py \
    --vqvae_ckpt outputs_vqvae/best_ckpt.pt \
    --dit_ckpt outputs_latent/best_ckpt.pt \
    --mode visualize --sample_steps 50
```

## 九、技术栈

| 技术 | 来源 | 作用 |
|------|------|------|
| VQVAE + 多尺度量化 | VAR (Tian'24) | 多分辨率潜码表示 |
| DiT (Diffusion Transformer) | Peebles'23 | 扩散骨干网络 |
| 多尺度交叉注意力 | VAR 条件注入思想 | 潜码多尺度条件 |
| v-prediction | Salimans'22 | 高 SNR 端更稳定 |
| min-SNR-γ 加权 | Hang'23 | 平衡不同噪声水平 |
| adaLN-Zero | Peebles'23 | 条件注入 + 初始恒等映射 |
| QK-Norm | Dehghani'23 | 防止 attention logit 发散 |
| Cosine schedule | Nichol'21 | 均匀信噪比分布 |
| EMA 编码本 | van den Oord'17 | 稳定向量量化训练 |
| SVD 频域辅助损失 | 本项目原创 | 潜空间频率先验 |

## 十、自检清单

| 检查项 | 结果 |
|--------|------|
| VQVAE 前向传播形状 (3,32,32) → z(32,8,8) → recon(3,32,32) | ✓ |
| VQVAE 梯度回传 (164/168 参数有梯度) | ✓ |
| 多尺度频率分解正确性 | ✓ |
| LatentDiT 交叉注意力集成 (3/6 层) | ✓ |
| v-prediction 数学可逆性 (误差 < 5e-7) | ✓ |
| SVD 截断精确低秩 | ✓ |
| 训练损失梯度回传 (130/130 DiT 参数有梯度) | ✓ |
| DDIM 采样正确输出 | ✓ |
| 多尺度级联采样正确输出 | ✓ |
| DDPM 采样正确输出 | ✓ |
| GPU 完整集成测试 (VQVAE→EMA→DiT→采样) | ✓ |
| EMA 滑动平均 apply/restore 正确性 | ✓ |
| GPU 峰值显存 ~1.6GB (32×32, batch=16) | ✓ |
| 全部中文注释覆盖 | ✓ |
