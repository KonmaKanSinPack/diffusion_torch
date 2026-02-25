import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def make_beta_schedule(
    *,
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError(f"timesteps must be > 0, got {timesteps}")

    schedule = schedule.lower()
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=dtype)
    elif schedule == "cosine":
        # Improved DDPM cosine schedule (Nichol & Dhariwal).
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device, dtype=dtype)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-8, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract a[t] and reshape to [B, 1, 1, 1]... to broadcast."""
    if t.dtype != torch.long:
        t = t.long()
    out = a.gather(0, t)
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


@dataclass
class DDPMParams:
    timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class ConditionalBDDPM:
    """DDPM on b-space with a condition image (e.g., I_L).

    Model signature expected: eps_pred = model(x_in, t)
      - x_in: concat([cond, b_t]) along channel dim
      - t: int64 tensor shape [B]

    b_t is a 1-channel image-like tensor.
    """

    def __init__(
        self,
        *,
        params: DDPMParams,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.params = params
        self.device = device
        self.dtype = dtype

        betas = make_beta_schedule(
            schedule=params.beta_schedule,
            timesteps=params.timesteps,
            beta_start=params.beta_start,
            beta_end=params.beta_end,
            device=device,
            dtype=dtype,
        )
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device, dtype=dtype), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # posterior variance q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

        # for posterior mean using eps-pred form
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
        ) / (1.0 - self.alphas_cumprod)

    def q_sample(self, *, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, *, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            x_t - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
        ) / _extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(
        self,
        *,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_x0: bool = False,
    ):
        x_in = torch.cat([cond, x_t], dim=1)
        eps_pred = model(x_in, t)

        x0_pred = self.predict_x0_from_eps(x_t=x_t, t=t, eps=eps_pred)
        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = _extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred, eps_pred

    @torch.no_grad()
    def p_sample(
        self,
        *,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_x0: bool = False,
    ) -> torch.Tensor:
        mean, var, _, _ = self.p_mean_variance(model=model, x_t=x_t, t=t, cond=cond, clip_x0=clip_x0)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_loop(
        self,
        *,
        model,
        shape: torch.Size,
        cond: torch.Tensor,
        clip_x0: bool = False,
        return_all: bool = False,
    ):
        x = torch.randn(shape, device=self.device, dtype=self.dtype)
        all_steps = [x] if return_all else None

        for step in reversed(range(self.params.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model=model, x_t=x, t=t, cond=cond, clip_x0=clip_x0)
            if return_all:
                all_steps.append(x)

        return (x, all_steps) if return_all else x

    def training_loss(
        self,
        *,
        model,
        x_start: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_in = torch.cat([cond, x_t], dim=1)
        eps_pred = model(x_in, t)
        return F.mse_loss(eps_pred, noise)
