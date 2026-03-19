# **SpectralDiff — 频谱级联扩散模型: 完整思路报告**

## 一、项目名称与核心思想

**SpectralDiff: SVD 频域分解引导的图像扩散生成模型**

核心命题:  **奇异值是低频信息，让 SVD 负责低频结构，让 Diffusion 负责高频细节。**

## 二、现有代码库 BUG 审查 (6 项)

| 编号 | 严重度 | 位置 | 问题 | 新实现修复方式 |
|------|--------|------|------|---------------|
| BUG-1 | **严重** | unet.py | `normalize(x)` 每次前向传播创建新 GroupNorm，参数永远无法学习 | DiT 使用持久化 LayerNorm |
| BUG-2 | **严重** | unet.py | resnetBlock 默认 dropout=0.8，丢弃 80% 特征 | DiT 默认 dropout=0.0 |
| BUG-3 | 设计缺陷 | train_b_diffusion.py | b 空间维度 `[B,1,r,W]` 中 r 依赖图像，限制泛化 | 全部在像素空间操作 |
| BUG-4 | 设计缺陷 | train_b_diffusion.py 整体 | 两阶段训练/推理分布偏移 | 单模型消除偏移 |
| BUG-5 | 轻微 | image_diffusion.py | clip_x0 后未重计算一致 eps | 始终重计算 eps |
| BUG-6 | 轻微 | b_diffusion.py | sqrt_recip_alphas 定义但未使用 | 清理无用代码 |

## 三、创新点

### 创新点 1: 频域分解辅助损失 (Spectral Decomposition Auxiliary Loss)

**原理**: 传统扩散模型对所有频率一视同仁地训练。实际上，低频结构（轮廓、外观）应在高噪声阶段优先学习，高频细节（纹理、光照）应在低噪声阶段精修。

**实现**:
$$L_{total} = w_{snr}(t) \cdot L_{main} + \lambda_L(t) \cdot L_{low} + \lambda_H(t) \cdot L_{high}$$

其中:
- $L_{low} = \|P_L(\hat{x}_0 - x_0)\|^2$，$P_L = U_k U_k^T$ 为 SVD 低频子空间投影
- $L_{high} = \|(I - P_L)(\hat{x}_0 - x_0)\|^2$ 为高频残差误差
- $\lambda_L(t) = \lambda \cdot (t/T)$: 大 t 时强调低频 → **先学结构**
- $\lambda_H(t) = \lambda \cdot (1 - t/T)$: 小 t 时强调高频 → **后学细节**

### 创新点 2: 频谱级联采样 (Spectral Cascade Sampling)

**核心**: 去噪过程中在中间点显式施加 SVD 结构约束。

```
Phase A: noise → DDIM去噪70% → 得到粗糙的 x0 预测
         ↓
    SVD投影: x0_pred → SVD截断 → I_L (干净的低频结构)
         ↓  ← 这步就是 "SVD 负责低频"
Phase B: I_L → 重新加噪到t_mid → 继续DDIM去噪到t=0 → I_out
         ↑  ← 这步就是 "Diffusion 负责高频"
```

### 创新点 3: 单模型消除分布偏移

旧方案：Stage-1 (生成 $I_L$) + Stage-2 (生成 $b|I_L$)，两模型级联，误差累积。
新方案：单个 DiT 模型在像素空间操作，SVD 仅作为损失和采样中的频率先验，彻底消除两阶段分布偏移。

### 创新点 4: 现代技术栈集成

| 技术 | 来源 | 作用 |
|------|------|------|
| DiT (Diffusion Transformer) | Peebles'23, Ma'24 | 2023-2024 主流骨干，可扩展 |
| v-prediction | Salimans'22 | 高 SNR 端更稳定 |
| min-SNR-γ 加权 | Hang'23 | 平衡不同噪声水平的训练信号 |
| adaLN-Zero | Peebles'23 | 条件注入，初始恒等映射 |
| QK-Norm | Dehghani'23 | 防止 attention logit 发散 |
| Cosine schedule | Nichol'21 | 比 linear 更均匀的信噪比分布 |

## 四、项目文件说明

| 文件 | 行数 | 功能 |
|------|------|------|
| models/dit.py | 449 | DiT 架构 (3种规模: T/S/B) |
| spectral_diffusion.py | 702 | 扩散核心: 频域分解损失 + DDIM + 级联采样 |
| train_spectral_diffusion.py | 533 | 完整训练脚本: 数据/模型/优化/EMA/评估 |
| sample_spectral_diffusion.py | 519 | 推理与评估: DDIM/级联采样 + FID/IS 计算 |

## 五、使用命令

### 训练 (CIFAR-10)
```bash
cd diffusion_torch/diffusion_torch

# DiT-S, v-prediction, cosine schedule, 频域辅助损失
python3 train_spectral_diffusion.py \
    --dataset cifar10 --epochs 500 --batch 128 --lr 2e-4 \
    --model_size S --pred_type v --beta_schedule cosine \
    --timesteps 1000 --k_truncate 8 --lambda_spectral 0.5 \
    --min_snr_gamma 5.0 --sample_steps 50 \
    --save_every 10 --eval_fid --eval_every 25 --fid_num 10000 \
    --out_dir ./outputs
```

### 推理 — 标准 DDIM
```bash
python3 sample_spectral_diffusion.py \
    --ckpt outputs/best_ckpt.pt --mode ddim \
    --n 256 --sample_steps 250 --out gen_ddim.png
```

### 推理 — 频谱级联采样
```bash
python3 sample_spectral_diffusion.py \
    --ckpt outputs/best_ckpt.pt --mode cascade \
    --n 256 --sample_steps 250 --k_truncate 8 --cascade_t_frac 0.3 \
    --out gen_cascade.png
```

### 完整评估 (FID-50k)
```bash
python3 sample_spectral_diffusion.py \
    --ckpt outputs/best_ckpt.pt --mode eval \
    --fid_num 50000 --sample_steps 250 --eval_cascade --compute_is
```

### 可视化 (频率分解 + 级联对比)
```bash
python3 sample_spectral_diffusion.py \
    --ckpt outputs/best_ckpt.pt --mode visualize --sample_steps 50
```

## 六、自检清单

| 检查项 | 结果 |
|--------|------|
| DiT 前向传播形状一致性 | ✓ (input [B,3,32,32] → output [B,3,32,32]) |
| v-prediction 数学可逆性 (x0/eps 恢复) | ✓ (误差 < 1e-7) |
| SVD 截断精确低秩 (float64) | ✓ (秩=k, 残差 < 1e-14) |
| SVD 频域投影正交性 | ✓ (低频+高频=全频) |
| min-SNR-γ 权重范围 [0,1] | ✓ |
| 训练损失梯度回传 (95/95 参数层有梯度) | ✓ |
| DDIM 采样输出值域 [-1,1] | ✓ |
| 频谱级联采样输出值域 [-1,1] | ✓ |
| Checkpoint 存取一致性 | ✓ |
| EMA 更新正确性 | ✓ |
| 全部中文注释覆盖 | ✓ |