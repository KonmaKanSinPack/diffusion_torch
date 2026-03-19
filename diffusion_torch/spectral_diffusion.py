"""
SpectralDiffusion — 带 SVD 频域分解的扩散过程核心模块
====================================================
核心创新:
  1. 频域分解损失: 将预测误差投影到 SVD 定义的低频/高频子空间,
     对不同时间步施加不同的频率权重, 实现"先结构后细节"的学习
  2. 频谱级联采样: 去噪中途对预测 x0 做 SVD 截断 (提取低频结构),
     再继续去噪补全高频细节, 实现 "SVD 负责低频, Diffusion 负责高频"
  3. 支持 ε-prediction 和 v-prediction 两种目标
  4. min-SNR-γ 损失加权: 平衡不同噪声水平的训练信号强度
  5. DDIM 采样: 支持确定性/随机采样, 可变步数

数学符号:
  x_0: 干净图像
  x_t: 第 t 步的带噪图像
  ε: 高斯噪声
  α_t = √(ᾱ_t), σ_t = √(1-ᾱ_t)
  v_t = α_t ε - σ_t x_0  (velocity)
  SNR(t) = ᾱ_t / (1 - ᾱ_t)
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 噪声调度: Beta Schedule
# ---------------------------------------------------------------------------

def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """生成扩散过程的 beta 序列

    参数:
      schedule: "linear" | "cosine"
      timesteps: 总扩散步数 T
      beta_start, beta_end: linear schedule 的起止值
      device, dtype: 张量设备与精度 (用 float64 保证数值稳定)

    返回:
      betas: [T] — 每步的噪声增量 β_t
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps 必须 > 0, 当前为 {timesteps}")

    schedule = schedule.lower()
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=dtype)
    elif schedule == "cosine":
        # Improved DDPM cosine schedule (Nichol & Dhariwal, 2021)
        # ᾱ_t = cos²((t/T + s) / (1+s) * π/2)
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device, dtype=dtype)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-8, 0.999)
    else:
        raise ValueError(f"未知的 beta schedule: {schedule}")

    return betas


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """从参数向量 a 中按时间步 t 取值, 并广播到 x 的形状"""
    if t.dtype != torch.long:
        t = t.long()
    out = a.gather(0, t)
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class SpectralDiffusionConfig:
    """扩散过程配置"""
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    # 预测目标: "eps" = 噪声预测, "v" = 速度预测
    pred_type: str = "v"
    # min-SNR-γ 加权参数 (0 表示不使用)
    min_snr_gamma: float = 5.0
    # SVD 频域分解参数
    k_truncate: int = 8          # 低频保留的奇异值个数
    lambda_spectral: float = 0.5 # 频域辅助损失权重
    # 频谱级联采样参数
    cascade_t_frac: float = 0.3  # 级联中点 (从末尾算起的比例, 0.3 = 去噪 70% 后做 SVD)


# ---------------------------------------------------------------------------
# SpectralDiffusion 主类
# ---------------------------------------------------------------------------

class SpectralDiffusion:
    """
    带 SVD 频域分解的扩散过程

    包含:
      - 前向加噪 q_sample
      - x0 预测 (从 ε/v 输出转换)
      - 频域分解损失
      - min-SNR-γ 加权
      - DDIM 采样
      - 频谱级联采样 (核心创新)
    """

    def __init__(
        self,
        config: SpectralDiffusionConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.T = config.timesteps

        # ---------- 噪声调度参数 (float64 计算, float32 存储) ----------
        betas = make_beta_schedule(
            schedule=config.beta_schedule,
            timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=device,
            dtype=torch.float64,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device, dtype=torch.float64), alphas_cumprod[:-1]]
        )

        # 存储为 float32 以匹配模型精度
        self.betas = betas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # α_t = √(ᾱ_t),  σ_t = √(1 - ᾱ_t)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()

        # 后验分布 q(x_{t-1} | x_t, x_0) 的参数
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).clamp(min=1e-20)
        self.posterior_variance = posterior_variance.float()

        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).float()

        # SNR(t) = ᾱ_t / (1 - ᾱ_t), 用于 min-SNR-γ 加权
        self.snr = (alphas_cumprod / (1.0 - alphas_cumprod)).float()

    # ------------------------------------------------------------------
    # 前向加噪
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向扩散: x_t = α_t x_0 + σ_t ε"""
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return alpha_t * x_start + sigma_t * noise

    # ------------------------------------------------------------------
    # 从模型输出恢复 x0 / ε
    # ------------------------------------------------------------------

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """从模型输出推断 x_0

        eps-prediction:  x_0 = (x_t - σ_t ε̂) / α_t
        v-prediction:    x_0 = α_t x_t - σ_t v̂
        """
        alpha_t = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if self.config.pred_type == "eps":
            return (x_t - sigma_t * model_output) / alpha_t.clamp(min=1e-8)
        elif self.config.pred_type == "v":
            return alpha_t * x_t - sigma_t * model_output
        else:
            raise ValueError(f"未知的 pred_type: {self.config.pred_type}")

    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """从模型输出推断噪声 ε"""
        alpha_t = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if self.config.pred_type == "eps":
            return model_output
        elif self.config.pred_type == "v":
            return sigma_t * x_t + alpha_t * model_output
        else:
            raise ValueError(f"未知的 pred_type: {self.config.pred_type}")

    def compute_v_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """计算 v-prediction 的训练目标: v = α_t ε - σ_t x_0"""
        alpha_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return alpha_t * noise - sigma_t * x_start

    # ------------------------------------------------------------------
    # SVD 频域分解
    # ------------------------------------------------------------------

    @staticmethod
    def svd_truncate(
        images: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """对图像进行 SVD 截断, 保留前 k 个奇异值 (低频分量)

        参数:
          images: [B, C, H, W]
          k: 保留的奇异值个数

        返回:
          images_low: [B, C, H, W] — 低秩近似 (低频分量)
        """
        B, C, H, W = images.shape
        flat = images.reshape(B, C * H, W)
        U, S, Vt = torch.linalg.svd(flat, full_matrices=False)
        # full_matrices=False: U ∈ [B, CH, r], S ∈ [B, r], Vt ∈ [B, r, W]
        # 其中 r = min(CH, W)
        r = S.shape[1]
        k_use = min(k, r)
        # 截断: 只保留前 k 个奇异值/向量 (精确低秩, 避免浮点噪声)
        U_k = U[:, :, :k_use]                              # [B, CH, k]
        S_k = torch.diag_embed(S[:, :k_use])               # [B, k, k]
        Vt_k = Vt[:, :k_use, :]                            # [B, k, W]
        # 重建低秩图像: I_L = U_k Σ_k V_k^T
        low_flat = U_k @ S_k @ Vt_k
        return low_flat.reshape(B, C, H, W)

    @staticmethod
    def svd_project_error(
        x0: torch.Tensor,
        x0_pred: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将预测误差投影到 SVD 低频/高频子空间

        基于 x0 的 SVD 分解定义频率子空间:
          - U_k: 前 k 个左奇异向量 → 低频子空间
          - U_rest: 其余左奇异向量 → 高频子空间

        参数:
          x0: [B, C, H, W] — 干净图像
          x0_pred: [B, C, H, W] — 模型预测的 x0
          k: 低频截止奇异值个数

        返回:
          loss_low:  标量 — 低频空间的 MSE
          loss_high: 标量 — 高频空间的 MSE
        """
        B, C, H, W = x0.shape
        flat_gt = x0.reshape(B, C * H, W)
        flat_pred = x0_pred.reshape(B, C * H, W)

        # 对干净图像做 SVD, 获取频率基
        U, S, Vt = torch.linalg.svd(flat_gt, full_matrices=False)
        r = S.shape[1]
        k_use = min(k, r)

        # 预测误差
        delta = flat_pred - flat_gt  # [B, CH, W]

        # 低频子空间投影: P_L(δ) = U_k U_k^T δ
        U_k = U[:, :, :k_use]                           # [B, CH, k]
        delta_low = U_k @ (U_k.transpose(1, 2) @ delta) # [B, CH, W]

        # 高频子空间: 互补投影
        delta_high = delta - delta_low                    # [B, CH, W]

        # 分频段 MSE
        loss_low = (delta_low ** 2).mean()
        loss_high = (delta_high ** 2).mean()

        return loss_low, loss_high

    # ------------------------------------------------------------------
    # 训练损失
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """计算带频域分解的训练损失

        L_total = w_snr(t) * L_main + λ_L(t) * L_low + λ_H(t) * L_high

        其中:
          - L_main: 标准预测损失 (eps 或 v 的 MSE)
          - L_low:  低频子空间预测误差 (x0 空间)
          - L_high: 高频子空间预测误差 (x0 空间)
          - w_snr(t): min-SNR-γ 加权
          - λ_L(t) = λ * (t/T): 大 t 时强调低频 → 先学结构
          - λ_H(t) = λ * (1 - t/T): 小 t 时强调高频 → 后学细节

        参数:
          model: 扩散网络 (eps/v 预测)
          x_start: [B, C, H, W] — 干净图像
          t: [B] — 随机采样的时间步
          noise: [B, C, H, W] — 高斯噪声 (可选, 默认随机生成)

        返回:
          loss: 标量损失
          log_dict: 用于日志记录的中间值
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 前向加噪
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 模型预测
        model_output = model(x_t, t)

        # ---- 主损失 ----
        if self.config.pred_type == "eps":
            target = noise
        elif self.config.pred_type == "v":
            target = self.compute_v_target(x_start, noise, t)
        else:
            raise ValueError(f"未知的 pred_type: {self.config.pred_type}")

        # 逐样本 MSE (不取均值, 便于加权)
        mse_per_sample = F.mse_loss(model_output, target, reduction="none").mean(dim=[1, 2, 3])

        # min-SNR-γ 加权
        if self.config.min_snr_gamma > 0:
            snr_t = self.snr.gather(0, t.long())  # [B] — 直接索引, 无需广播
            gamma = self.config.min_snr_gamma
            if self.config.pred_type == "eps":
                # w(t) = min(SNR(t), γ) / SNR(t)
                weight = torch.clamp(snr_t, max=gamma) / snr_t.clamp(min=1e-8)
            else:
                # v-prediction: w(t) = min(SNR(t), γ) / (SNR(t) + 1)
                weight = torch.clamp(snr_t, max=gamma) / (snr_t + 1.0)
            loss_main = (weight * mse_per_sample).mean()
        else:
            loss_main = mse_per_sample.mean()

        # ---- 频域辅助损失 ----
        loss_low = torch.tensor(0.0, device=self.device)
        loss_high = torch.tensor(0.0, device=self.device)
        if self.config.lambda_spectral > 0:
            with torch.no_grad():
                x0_pred = self.predict_x0(x_t, t, model_output).detach()

            # 注意: 对 model_output 的梯度仍然通过 loss_main 流通
            # 频域损失使用 detach 的预测, 仅作为辅助监督信号
            # 但为了让梯度回传, 重新计算 x0_pred (不 detach)
            x0_pred_grad = self.predict_x0(x_t, t, model_output)
            x0_pred_grad = x0_pred_grad.clamp(-1.0, 1.0)

            loss_low, loss_high = self.svd_project_error(
                x_start, x0_pred_grad, self.config.k_truncate
            )

            # 时间依赖权重: 大 t → 强调低频, 小 t → 强调高频
            t_frac = t.float() / max(self.T - 1, 1)  # [0, 1]
            lambda_L = self.config.lambda_spectral * t_frac.mean()
            lambda_H = self.config.lambda_spectral * (1.0 - t_frac.mean())

            loss_spectral = lambda_L * loss_low + lambda_H * loss_high
        else:
            loss_spectral = torch.tensor(0.0, device=self.device)

        loss_total = loss_main + loss_spectral

        log_dict = {
            "loss_total": loss_total.item(),
            "loss_main": loss_main.item(),
            "loss_low": loss_low.item(),
            "loss_high": loss_high.item(),
        }

        return loss_total, log_dict

    # ------------------------------------------------------------------
    # DDIM 采样
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        shape: torch.Size,
        steps: int = 50,
        eta: float = 0.0,
        clip_x0: bool = True,
        init_noise: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """DDIM 采样 — 从高斯噪声逐步去噪至干净图像

        参数:
          model: 扩散网络
          shape: 输出形状 [B, C, H, W]
          steps: 采样步数 (≤ T)
          eta: 0.0 = 确定性 DDIM, >0 = 随机 DDIM
          clip_x0: 是否将 x0 预测裁剪到 [-1, 1]
          init_noise: 可选固定初始噪声
          return_intermediates: 是否返回中间状态列表

        返回:
          x: [B, C, H, W] — 生成的图像
          (可选) intermediates: 中间状态列表
        """
        if steps <= 0:
            raise ValueError(f"steps 必须 > 0, 当前为 {steps}")
        T = self.T
        steps = min(int(steps), T)

        # 构建时间序列: 从 T-1 递减到 0
        t_seq = torch.linspace(0, T - 1, steps, device=self.device)
        t_seq = torch.round(t_seq).long().unique(sorted=True)
        t_seq = t_seq.flip(0)  # 降序

        x = init_noise if init_noise is not None else \
            torch.randn(shape, device=self.device)
        intermediates = [x] if return_intermediates else None

        for idx, t_val in enumerate(t_seq):
            t = torch.full((shape[0],), int(t_val.item()), device=self.device, dtype=torch.long)

            # 模型预测
            model_out = model(x, t)
            x0_pred = self.predict_x0(x, t, model_out)
            eps_pred = self.predict_eps(x, t, model_out)

            if clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                # 重新计算一致的 eps
                alpha_t = _extract(self.sqrt_alphas_cumprod, t, x.shape)
                sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                eps_pred = (x - alpha_t * x0_pred) / sigma_t.clamp(min=1e-8)

            # 最后一步直接返回 x0
            if idx == len(t_seq) - 1:
                x = x0_pred
                break

            # DDIM 更新
            t_prev_val = int(t_seq[idx + 1].item())
            t_prev = torch.full((shape[0],), t_prev_val, device=self.device, dtype=torch.long)

            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)

            # DDIM sigma
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8))
                * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8)).clamp(min=0.0))
            )
            pred_dir = torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)) * eps_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + pred_dir

            if eta > 0:
                x = x + sigma * torch.randn_like(x)

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    # ------------------------------------------------------------------
    # 频谱级联采样 (核心创新)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def spectral_cascade_sample(
        self,
        model: torch.nn.Module,
        shape: torch.Size,
        steps: int = 50,
        k_truncate: Optional[int] = None,
        t_mid_frac: Optional[float] = None,
        eta: float = 0.0,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        """频谱级联采样 — SVD 负责低频, Diffusion 负责高频

        算法流程:
          Phase A (去噪主体, 恢复低频结构):
            从纯噪声 x_T 开始, 使用 DDIM 去噪到中间时间步 t_mid
            → 此时模型已大致恢复了图像的低频结构

          SVD 投影 (提取/增强低频结构):
            对 Phase A 的 x0 预测做 SVD 截断, 得到 I_L (低秩近似)
            → SVD 截断去除了残余高频噪声, 保留干净的结构信息
            → 这就是 "SVD 负责低频" 的体现

          Phase B (继续去噪, 补全高频细节):
            将 I_L 重新加噪到 t_mid, 然后继续 DDIM 去噪到 t=0
            → 扩散模型在已有结构的基础上自然生成纹理/光照等高频细节
            → 这就是 "Diffusion 负责高频" 的体现

        参数:
          model: 扩散网络
          shape: [B, C, H, W]
          steps: 总采样步数 (会在两个 phase 之间分配)
          k_truncate: SVD 截断保留奇异值数 (默认用 config 中的值)
          t_mid_frac: 级联中点比例 (0~1, 从时间轴末端算起)
          eta: DDIM 随机性参数
          clip_x0: 是否裁剪 x0 预测

        返回:
          x: [B, C, H, W] — 生成的图像
        """
        k = k_truncate if k_truncate is not None else self.config.k_truncate
        t_frac = t_mid_frac if t_mid_frac is not None else self.config.cascade_t_frac
        T = self.T

        if steps <= 0:
            raise ValueError(f"steps 必须 > 0")
        steps = min(int(steps), T)

        # 确定中间时间步
        t_mid = max(1, int(T * t_frac))

        # 构建完整时间序列
        t_full = torch.linspace(0, T - 1, steps, device=self.device)
        t_full = torch.round(t_full).long().unique(sorted=True).flip(0)

        # 分割为 Phase A 和 Phase B
        phase_a_mask = t_full >= t_mid
        phase_b_mask = t_full < t_mid
        t_phase_a = t_full[phase_a_mask]
        t_phase_b = t_full[phase_b_mask]

        # === Phase A: 从噪声去噪到 t_mid ===
        x = torch.randn(shape, device=self.device)

        for idx, t_val in enumerate(t_phase_a):
            t = torch.full((shape[0],), int(t_val.item()), device=self.device, dtype=torch.long)
            model_out = model(x, t)
            x0_pred = self.predict_x0(x, t, model_out)
            eps_pred = self.predict_eps(x, t, model_out)

            if clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                alpha_t = _extract(self.sqrt_alphas_cumprod, t, x.shape)
                sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                eps_pred = (x - alpha_t * x0_pred) / sigma_t.clamp(min=1e-8)

            # 如果 Phase A 结束后还有 Phase B, 做 DDIM step 到 t_mid
            if idx == len(t_phase_a) - 1:
                if len(t_phase_b) > 0:
                    # 保存 Phase A 末尾的 x0 预测用于 SVD 投影
                    x0_phase_a = x0_pred
                    break
                else:
                    x = x0_pred
                    return x

            # 正常 DDIM step
            t_next_val = int(t_phase_a[idx + 1].item())
            t_next = torch.full((shape[0],), t_next_val, device=self.device, dtype=torch.long)

            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_next = _extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp(min=1e-8))
                * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_next.clamp(min=1e-8)).clamp(min=0.0))
            )
            pred_dir = torch.sqrt((1.0 - alpha_bar_next - sigma ** 2).clamp(min=0.0)) * eps_pred
            x = torch.sqrt(alpha_bar_next) * x0_pred + pred_dir
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        # === SVD 投影: 提取低频结构 ===
        x0_low = self.svd_truncate(x0_phase_a, k=k)

        # === Phase B: 从低频结构重新加噪到 t_mid, 然后继续去噪 ===
        # 将低频图像加噪到 t_mid 水平
        t_mid_tensor = torch.full((shape[0],), t_mid, device=self.device, dtype=torch.long)
        noise_fresh = torch.randn_like(x0_low)
        x = self.q_sample(x_start=x0_low, t=t_mid_tensor, noise=noise_fresh)

        # Phase B DDIM 去噪
        for idx, t_val in enumerate(t_phase_b):
            t = torch.full((shape[0],), int(t_val.item()), device=self.device, dtype=torch.long)
            model_out = model(x, t)
            x0_pred = self.predict_x0(x, t, model_out)
            eps_pred = self.predict_eps(x, t, model_out)

            if clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                alpha_t = _extract(self.sqrt_alphas_cumprod, t, x.shape)
                sigma_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                eps_pred = (x - alpha_t * x0_pred) / sigma_t.clamp(min=1e-8)

            if idx == len(t_phase_b) - 1:
                x = x0_pred
                break

            t_next_val = int(t_phase_b[idx + 1].item())
            t_next = torch.full((shape[0],), t_next_val, device=self.device, dtype=torch.long)

            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_next = _extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp(min=1e-8))
                * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_next.clamp(min=1e-8)).clamp(min=0.0))
            )
            pred_dir = torch.sqrt((1.0 - alpha_bar_next - sigma ** 2).clamp(min=0.0)) * eps_pred
            x = torch.sqrt(alpha_bar_next) * x0_pred + pred_dir
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        return x

    # ------------------------------------------------------------------
    # DDPM 采样 (完整步数, 参考用)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: torch.Size,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        """完整的 DDPM 采样 (T 步), 主要用于参考和验证"""
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            model_out = model(x, t)
            x0_pred = self.predict_x0(x, t, model_out)
            if clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            mean = (
                _extract(self.posterior_mean_coef1, t, x.shape) * x0_pred
                + _extract(self.posterior_mean_coef2, t, x.shape) * x
            )
            var = _extract(self.posterior_variance, t, x.shape)

            if step == 0:
                x = mean
            else:
                x = mean + torch.sqrt(var) * torch.randn_like(x)
        return x
