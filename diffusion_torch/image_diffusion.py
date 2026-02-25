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


class ImageDDPM:
    """Unconditional DDPM for images, noise-prediction objective."""

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

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

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

    def p_mean_variance(self, *, model, x_t: torch.Tensor, t: torch.Tensor, clip_x0: bool = True):
        eps_pred = model(x_t, t)
        x0_pred = self.predict_x0_from_eps(x_t=x_t, t=t, eps=eps_pred)
        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = _extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred

    @torch.no_grad()
    def p_sample(self, *, model, x_t: torch.Tensor, t: torch.Tensor, clip_x0: bool = True) -> torch.Tensor:
        mean, var, _ = self.p_mean_variance(model=model, x_t=x_t, t=t, clip_x0=clip_x0)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_loop(self, *, model, shape: torch.Size, clip_x0: bool = True) -> torch.Tensor:
        x = torch.randn(shape, device=self.device, dtype=self.dtype)
        for step in reversed(range(self.params.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model=model, x_t=x, t=t, clip_x0=clip_x0)
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        *,
        model,
        shape: torch.Size,
        steps: int = 50,
        eta: float = 0.0,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling with a reduced number of steps.

        - steps: number of sampling steps (<= timesteps)
        - eta: 0.0 => deterministic DDIM; >0 adds noise
        """
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        T = self.params.timesteps
        steps = min(int(steps), int(T))

        # Pick a monotone decreasing set of timesteps.
        t_seq = torch.linspace(0, T - 1, steps, device=self.device)
        t_seq = torch.round(t_seq).long().unique(sorted=True)
        t_seq = t_seq.flip(0)  # descending

        x = torch.randn(shape, device=self.device, dtype=self.dtype)
        for idx, t_val in enumerate(t_seq):
            t = torch.full((shape[0],), int(t_val.item()), device=self.device, dtype=torch.long)

            eps = model(x, t)
            x0 = self.predict_x0_from_eps(x_t=x, t=t, eps=eps)
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)

            if idx == len(t_seq) - 1:
                x = x0
                break

            t_prev_val = int(t_seq[idx + 1].item())
            t_prev = torch.full((shape[0],), t_prev_val, device=self.device, dtype=torch.long)

            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)

            # DDIM sigma
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_prev).clamp(min=0.0))
            )
            noise = torch.randn_like(x)
            pred_dir = torch.sqrt((1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0)) * eps
            x = torch.sqrt(alpha_bar_prev) * x0 + pred_dir + sigma * noise

        return x

    @torch.no_grad()
    def ddim_denoise_from(
        self,
        *,
        model,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        steps: int = 25,
        eta: float = 0.0,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        """DDIM denoise starting from a provided x_t at timestep t_start.

        This is useful for "conditioning augmentation": take a real target x0,
        noise it to x_t, then denoise with a trained model to get a model-like sample.
        """
        if t_start.dtype != torch.long:
            t_start = t_start.long()
        if t_start.dim() != 1:
            raise ValueError(f"t_start must have shape [B], got {t_start.shape}")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")

        # We only support a single shared start timestep for simplicity.
        if not torch.all(t_start == t_start[0]):
            raise ValueError("ddim_denoise_from currently requires all t_start equal within batch")

        t0 = int(t_start[0].item())
        if t0 < 0 or t0 >= self.params.timesteps:
            raise ValueError(f"t_start out of range: {t0}")

        steps = min(int(steps), t0 + 1)
        t_seq = torch.linspace(0, t0, steps, device=self.device)
        t_seq = torch.round(t_seq).long().unique(sorted=True).flip(0)  # descending to 0

        x = x_t
        bsz = x.shape[0]
        for idx, t_val in enumerate(t_seq):
            t = torch.full((bsz,), int(t_val.item()), device=self.device, dtype=torch.long)
            eps = model(x, t)
            x0 = self.predict_x0_from_eps(x_t=x, t=t, eps=eps)
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)

            if idx == len(t_seq) - 1:
                x = x0
                break

            t_prev_val = int(t_seq[idx + 1].item())
            t_prev = torch.full((bsz,), t_prev_val, device=self.device, dtype=torch.long)

            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_prev).clamp(min=0.0))
            )
            noise = torch.randn_like(x)
            pred_dir = torch.sqrt((1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0)) * eps
            x = torch.sqrt(alpha_bar_prev) * x0 + pred_dir + sigma * noise

        return x

    def training_loss(self, *, model, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        eps_pred = model(x_t, t)
        return F.mse_loss(eps_pred, noise)
