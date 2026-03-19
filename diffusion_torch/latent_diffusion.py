"""
Latent SpectralDiffusion — 在 VQVAE 潜在空间上的频率感知扩散模型
================================================================

参考论文:
  - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
  - Tian et al., "Visual Autoregressive Modeling: Scalable Image Generation
    via Next-Scale Prediction", NeurIPS 2024
  - Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
  - Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy", ICCV 2023

核心设计:
  1. 两阶段训练:
     - Stage 1: 训练 Multi-Scale VQVAE (图像 → 多尺度离散 latent)
     - Stage 2: 冻结 VQVAE, 在连续 latent 空间上训练 Diffusion (本模块)
  
  2. VAR 启发的频率分解:
     - VQVAE 的多尺度量化自然形成 coarse-to-fine 的频率层级
     - 粗尺度 (scale 1, 2): 全局语义、轮廓 = 低频
     - 细尺度 (scale 4, 8): 纹理、边缘 = 高频
     
  3. 多尺度条件化:
     - 粗尺度量化结果作为 context, 通过 cross-attention 条件化 DiT
     - DiT 主要负责学习高频残差 (高分辨率尺度的信息)
     - 这等价于用 VQVAE 粗尺度替代了原来的 SVD 低秩投影
  
  4. SVD 辅助频率分析 (保留):
     - 仍可在 latent 空间做 SVD 分析，监控频率分布
     - 但不再作为核心采样策略，而是辅助诊断工具

  5. 采样策略:
     - DDIM: 标准 latent DDIM 采样
     - 多尺度级联: 先生成粗尺度 latent, 再以粗尺度为条件生成完整 latent
     - DDPM: 标准 T-step 采样

关键改进 (相比 V1 pixel-space):
  - latent space 远比 pixel space 紧凑 (32×32×3 → 8×8×32), 计算效率提升
  - VQVAE 的离散化提供了天然的频率分解, 替代手工 SVD
  - 多尺度条件让 diffusion 专注于高频细节, 降低学习难度
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  配置
# ============================================================

@dataclass
class LatentDiffusionConfig:
    """Latent Diffusion 配置"""
    # 噪声调度
    num_timesteps: int = 1000
    beta_schedule: str = 'cosine'    # 'linear' 或 'cosine'
    beta_start: float = 1e-4         # linear schedule 参数
    beta_end: float = 0.02           # linear schedule 参数
    
    # 预测模式
    pred_type: str = 'v'             # 'eps' 或 'v' (v-prediction)
    
    # 损失权重
    min_snr_gamma: float = 5.0       # min-SNR-γ 截断值
    svd_aux_weight: float = 0.1      # SVD 辅助损失权重 (latent 空间)
    svd_rank: int = 4                # SVD 截断秩 (在 latent 空间)
    use_svd_aux: bool = True         # 是否使用 SVD 辅助损失
    
    # 多尺度条件
    use_multiscale_cond: bool = True  # 是否使用多尺度条件化
    cond_scales: List[int] = field(default_factory=lambda: [1, 2])  # 用于条件化的粗尺度

    # VQVAE latent 参数 (需要与 VQVAE 配置匹配)
    latent_dim: int = 32
    latent_size: int = 8


# ============================================================
#  噪声调度
# ============================================================

def make_beta_schedule(
    schedule: str, num_timesteps: int,
    beta_start: float = 1e-4, beta_end: float = 0.02,
) -> torch.Tensor:
    """
    生成 beta 调度序列
    
    cosine 调度 (Nichol & Dhariwal, 2021):
      alpha_bar(t) = cos²(π/2 · (t/T + s) / (1 + s))
      其中 s = 0.008 是偏移量, 避免 t=0 时 beta 过小
    """
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    elif schedule == 'cosine':
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(min=1e-8, max=0.999)
    else:
        raise ValueError(f"未知的 beta schedule: {schedule}")
    return betas


# ============================================================
#  核心: Latent SpectralDiffusion
# ============================================================

class LatentSpectralDiffusion(nn.Module):
    """
    在 VQVAE 潜在空间上的扩散过程
    
    管理:
    - 前向扩散: q(z_t | z_0) = N(z_t; α_t * z_0, σ_t² * I)
    - 训练损失: min-SNR-γ 加权 MSE + 多尺度频率辅助损失
    - 逆向采样: DDIM / DDPM / 多尺度级联
    
    z 为 VQVAE encoder 输出的连续 latent (非量化), shape (B, D, H_lat, W_lat)
    """

    def __init__(self, cfg: LatentDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        T = cfg.num_timesteps

        # ---- 噪声调度参数 (float64 精度计算, 存储为 float32) ----
        betas = make_beta_schedule(cfg.beta_schedule, T, cfg.beta_start, cfg.beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 前向扩散参数
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # SNR = α²/(1-α²), 用于 min-SNR-γ 加权
        snr = alphas_cumprod / (1.0 - alphas_cumprod)

        # DDPM 后验参数: q(z_{t-1} | z_t, z_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        # 注册为 buffer (不参与梯度, 跟随 .to(device))
        def _reg(name, tensor):
            self.register_buffer(name, tensor.float())

        _reg('betas', betas)
        _reg('alphas_cumprod', alphas_cumprod)
        _reg('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        _reg('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        _reg('snr', snr)
        _reg('posterior_variance', posterior_variance)
        _reg('posterior_log_variance', posterior_log_variance)
        _reg('posterior_mean_coef1', posterior_mean_coef1)
        _reg('posterior_mean_coef2', posterior_mean_coef2)

    # ----------------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------------

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """从参数序列 a 中按时间步 t 提取值, 并广播到 shape"""
        batch_size = t.shape[0]
        out = a.gather(0, t.long())
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(
        self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向扩散加噪: q(z_t | z_0) = α_t * z_0 + σ_t * ε
        
        Args:
            z_0: (B, D, H, W)  干净 latent
            t:   (B,)           时间步
            noise: (B, D, H, W) 噪声 (可选, 默认随机)
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        alpha_t = self._extract(self.sqrt_alphas_cumprod, t, z_0.shape)
        sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)
        return alpha_t * z_0 + sigma_t * noise

    def predict_z0(
        self, model_out: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """从模型输出恢复 z_0 预测"""
        alpha_t = self._extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)

        if self.cfg.pred_type == 'eps':
            return (z_t - sigma_t * model_out) / alpha_t.clamp(min=1e-8)
        elif self.cfg.pred_type == 'v':
            return alpha_t * z_t - sigma_t * model_out
        else:
            raise ValueError(f"未知 pred_type: {self.cfg.pred_type}")

    def predict_eps(
        self, model_out: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """从模型输出恢复噪声 ε 预测"""
        alpha_t = self._extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)

        if self.cfg.pred_type == 'eps':
            return model_out
        elif self.cfg.pred_type == 'v':
            return sigma_t * z_t + alpha_t * model_out
        else:
            raise ValueError(f"未知 pred_type: {self.cfg.pred_type}")

    def compute_v_target(
        self, z_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """计算 v-prediction 目标: v = α_t * ε - σ_t * z_0"""
        alpha_t = self._extract(self.sqrt_alphas_cumprod, t, z_0.shape)
        sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)
        return alpha_t * noise - sigma_t * z_0

    # ----------------------------------------------------------------
    #  SVD 频率分析 (latent 空间版, 辅助诊断)
    # ----------------------------------------------------------------

    @staticmethod
    def svd_truncate(z: torch.Tensor, rank: int) -> torch.Tensor:
        """
        对 latent map 的每个通道做 SVD 低秩截断
        
        z: (B, D, H, W) → 对每个 (B, d, H, W) 的 H×W 矩阵做 SVD
        返回: (B, D, H, W) 低秩近似
        
        在 latent 空间中, SVD 截断捕获 latent feature map 的空间低频模式
        """
        B, D, H, W = z.shape
        z_flat = z.reshape(B * D, H, W)
        U, S, Vt = torch.linalg.svd(z_flat, full_matrices=False)
        k = min(rank, min(H, W))
        U_k = U[:, :, :k]
        S_k = S[:, :k]
        Vt_k = Vt[:, :k, :]
        z_low = U_k @ torch.diag_embed(S_k) @ Vt_k
        return z_low.reshape(B, D, H, W)

    @staticmethod
    def svd_project_error(
        error: torch.Tensor, rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将预测误差分解为 SVD 低频和高频分量
        
        error: (B, D, H, W) — 预测误差 (z_0_pred - z_0_true)
        返回: (low_error, high_error), 分别在低秩/高秩子空间的误差
        """
        low = LatentSpectralDiffusion.svd_truncate(error, rank)
        high = error - low
        return low, high

    # ----------------------------------------------------------------
    #  训练损失
    # ----------------------------------------------------------------

    def training_loss(
        self,
        model: nn.Module,
        z_0: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 Latent Diffusion 训练损失
        
        Args:
            model:   DiT 网络 (接受 z_t, t, context)
            z_0:     (B, D, H, W) 干净 latent (VQVAE encoder 输出)
            context: (B, M, D_ctx) 多尺度条件 tokens (可选)
        
        Returns:
            loss:     scalar 总损失
            log_dict: dict 各项损失数值
        
        损失组成:
          L_total = L_main + λ_svd * (L_low + L_high)
          
          L_main:  min-SNR-γ 加权的逐样本 MSE
          L_low:   低频子空间误差 (SVD latent 分析)
          L_high:  高频子空间误差
        """
        B = z_0.shape[0]
        device = z_0.device

        # 1. 随机时间步
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=device)

        # 2. 加噪
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)

        # 3. 模型预测
        model_out = model(z_t, t, context=context)

        # 4. 计算目标
        if self.cfg.pred_type == 'v':
            target = self.compute_v_target(z_0, noise, t)
        else:
            target = noise

        # 5. 逐样本 MSE
        mse_per_sample = (model_out - target).pow(2).mean(dim=[1, 2, 3])  # (B,)

        # 6. min-SNR-γ 加权
        snr_t = self.snr.gather(0, t.long())  # (B,)
        weight = torch.clamp(snr_t, max=self.cfg.min_snr_gamma) / snr_t
        loss_main = (weight * mse_per_sample).mean()

        log_dict = {
            'loss_main': loss_main.item(),
            'mse_mean': mse_per_sample.mean().item(),
        }

        # 7. SVD 辅助频率损失 (可选, 在 latent 空间)
        loss_svd = torch.tensor(0.0, device=device)
        if self.cfg.use_svd_aux and self.cfg.svd_aux_weight > 0:
            z_0_pred = self.predict_z0(model_out, z_t, t)
            error = z_0_pred - z_0
            low_err, high_err = self.svd_project_error(error, self.cfg.svd_rank)
            loss_low = low_err.pow(2).mean()
            loss_high = high_err.pow(2).mean()
            loss_svd = self.cfg.svd_aux_weight * (loss_low + loss_high)
            log_dict['loss_svd_low'] = loss_low.item()
            log_dict['loss_svd_high'] = loss_high.item()

        loss_total = loss_main + loss_svd
        log_dict['loss_total'] = loss_total.item()

        return loss_total, log_dict

    # ----------------------------------------------------------------
    #  DDIM 采样
    # ----------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = False,
        clip_range: float = 5.0,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDIM 采样 (在 latent 空间)
        
        Args:
            model:   DiT 网络
            shape:   (B, D, H_lat, W_lat) 输出形状
            num_steps: DDIM 步数
            eta:     DDIM 随机性参数 (0 = 确定性, 1 ≈ DDPM)
            clip_denoised: 是否裁剪去噪结果
            clip_range: 裁剪范围 (latent 空间不用 [-1,1], 用更大范围)
            context: 多尺度条件
        Returns:
            z_0:     (B, D, H_lat, W_lat) 生成的 latent
        """
        device = self.betas.device
        B = shape[0]
        T = self.cfg.num_timesteps

        # DDIM 时间步子集 (均匀间隔)
        step_size = T // num_steps
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        # 从纯噪声开始
        z_t = torch.randn(shape, device=device)

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # 模型预测
            model_out = model(z_t, t_batch, context=context)

            # 恢复 z_0 预测
            z_0_pred = self.predict_z0(model_out, z_t, t_batch)
            if clip_denoised:
                z_0_pred = z_0_pred.clamp(-clip_range, clip_range)

            # 恢复 eps 预测 (从 clipped z_0 重算, 更准确)
            eps_pred = self.predict_eps(model_out, z_t, t_batch)
            if clip_denoised:
                alpha_t = self._extract(self.sqrt_alphas_cumprod, t_batch, z_t.shape)
                sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, z_t.shape)
                eps_pred = (z_t - alpha_t * z_0_pred) / sigma_t.clamp(min=1e-8)

            # t_next 的噪声参数
            alpha_next = self.sqrt_alphas_cumprod[t_next]
            sigma_next = self.sqrt_one_minus_alphas_cumprod[t_next]

            # DDIM 更新
            if eta > 0 and t_next > 0:
                alpha_cur = self.alphas_cumprod[t_cur]
                alpha_nxt = self.alphas_cumprod[t_next]
                sigma_ddim = eta * torch.sqrt(
                    (1 - alpha_nxt) / (1 - alpha_cur) * (1 - alpha_cur / alpha_nxt)
                )
                dir_z = torch.sqrt(1 - alpha_nxt - sigma_ddim ** 2) * eps_pred
                z_t = torch.sqrt(alpha_nxt) * z_0_pred + dir_z
                z_t = z_t + sigma_ddim * torch.randn_like(z_t)
            else:
                z_t = alpha_next * z_0_pred + sigma_next * eps_pred

        return z_t

    # ----------------------------------------------------------------
    #  多尺度级联采样 (VAR 启发)
    # ----------------------------------------------------------------

    @torch.no_grad()
    def multiscale_cascade_sample(
        self,
        model: nn.Module,
        vqvae: nn.Module,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        clip_denoised: bool = False,
        clip_range: float = 5.0,
    ) -> torch.Tensor:
        """
        多尺度级联采样 (VAR 启发)
        
        流程:
        1. Phase A: 无条件 DDIM 采样生成粗略 latent
        2. 通过 VQVAE 量化器提取粗尺度:
           - 将生成的 latent 输入 multi-scale quantizer
           - 取粗尺度 (scale 1, 2) 的量化结果
        3. Phase B: 以粗尺度为条件, 再次 DDIM 采样
           生成精细 latent (包含高频细节)
        
        这模拟了 VAR 的 "next-scale prediction":
        先生成低频 → 以低频为条件生成高频
        
        Args:
            model:  DiT 网络 (需支持 cross-attention)
            vqvae:  训练好的 VQVAE (用于提取粗尺度)
            shape:  (B, D, H_lat, W_lat) 目标形状
        Returns:
            z_final: 最终 latent
        """
        device = self.betas.device
        B = shape[0]

        # ---- Phase A: 无条件采样 → 粗略 latent ----
        z_coarse = self.ddim_sample(
            model, shape, num_steps=num_steps,
            clip_denoised=clip_denoised, clip_range=clip_range,
            context=None,  # 无条件
        )

        # ---- 提取粗尺度条件 ----
        # 通过 VQVAE quantizer 获取各尺度量化结果
        z_q, z_q_scales, indices_scales, _ = vqvae.quantizer.encode(z_coarse)
        
        # 取粗尺度 (前 N_cond 个尺度)
        cond_scales = self.cfg.cond_scales  # 如 [1, 2]
        all_scales = vqvae.cfg.multi_scales
        
        # 拼接粗尺度 tokens 作为 context
        context_tokens = []
        for i, scale in enumerate(all_scales):
            if scale in cond_scales:
                # z_q_scales[i]: (B, D, H_lat, W_lat) — 该尺度上采样后的特征
                # 将空间展平为 tokens
                z_s = z_q_scales[i]  # (B, D, H, W)
                tokens = z_s.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
                context_tokens.append(tokens)

        if context_tokens:
            context = torch.cat(context_tokens, dim=1)  # (B, M_total, D)
        else:
            context = None

        # ---- Phase B: 以粗尺度为条件, 精细采样 ----
        z_fine = self.ddim_sample(
            model, shape, num_steps=num_steps,
            clip_denoised=clip_denoised, clip_range=clip_range,
            context=context,
        )

        return z_fine

    # ----------------------------------------------------------------
    #  DDPM 采样 (全步)
    # ----------------------------------------------------------------

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDPM 标准 T-step 采样
        
        Args:
            model: DiT 网络
            shape: (B, D, H_lat, W_lat)
            context: 多尺度条件
        Returns:
            z_0: 生成的 latent
        """
        device = self.betas.device
        B = shape[0]
        T = self.cfg.num_timesteps

        z_t = torch.randn(shape, device=device)

        for t_val in reversed(range(T)):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            model_out = model(z_t, t_batch, context=context)
            z_0_pred = self.predict_z0(model_out, z_t, t_batch)

            # 后验均值
            mean = (
                self._extract(self.posterior_mean_coef1, t_batch, z_t.shape) * z_0_pred
                + self._extract(self.posterior_mean_coef2, t_batch, z_t.shape) * z_t
            )

            if t_val > 0:
                log_var = self._extract(self.posterior_log_variance, t_batch, z_t.shape)
                noise = torch.randn_like(z_t)
                z_t = mean + torch.exp(0.5 * log_var) * noise
            else:
                z_t = mean

        return z_t


# ============================================================
#  多尺度条件提取辅助函数
# ============================================================

def extract_multiscale_context(
    vqvae: nn.Module,
    z: torch.Tensor,
    cond_scales: List[int],
) -> torch.Tensor:
    """
    从 VQVAE 的多尺度量化中提取粗尺度 context tokens
    
    Args:
        vqvae:       MultiScaleVQVAE 模型
        z:           (B, D, H_lat, W_lat) 输入 latent
        cond_scales: 要提取的粗尺度列表, 如 [1, 2]
    
    Returns:
        context: (B, M_total, D) — 拼接的 context tokens
    """
    z_q, z_q_scales, indices_scales, _ = vqvae.quantizer.encode(z)
    all_scales = vqvae.cfg.multi_scales

    context_tokens = []
    for i, scale in enumerate(all_scales):
        if scale in cond_scales:
            z_s = z_q_scales[i]  # (B, D, H, W)
            tokens = z_s.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
            context_tokens.append(tokens)

    if context_tokens:
        return torch.cat(context_tokens, dim=1)
    return None
