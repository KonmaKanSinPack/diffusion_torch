import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.utils import save_image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.unet import UNet
from b_diffusion import ConditionalBDDPM, DDPMParams
import classifier_metrics_numpy


def grad_clip(params, mode: str = "norm", value: float = 1.0, **kwargs) -> None:
    assert mode in ["value", "norm"], "mode should be value or norm"
    if mode == "norm":
        nn.utils.clip_grad.clip_grad_norm_(parameters=params, max_norm=value, **kwargs)
    else:
        nn.utils.clip_grad.clip_grad_value_(parameters=params, clip_value=value)


def pad_to_square(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    if h == w:
        return x
    if h > w:
        diff = h - w
        return F.pad(x, (diff // 2, diff - diff // 2, 0, 0), "constant", 0)
    diff = w - h
    return F.pad(x, (0, 0, diff // 2, diff - diff // 2), "constant", 0)


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def psnr_from_batch(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-10) -> float:
    x_true_01 = denorm_to_unit(x_true)
    x_pred_01 = denorm_to_unit(x_pred)
    mse = torch.mean((x_true_01 - x_pred_01) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return float(psnr.mean().item())


def build_inception_feature_extractor(device: torch.device) -> nn.Module:
    try:
        from torchvision.models import Inception_V3_Weights

        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    except Exception:
        model = inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def inception_activations(x_01: torch.Tensor, inception: nn.Module) -> np.ndarray:
    x_299 = F.interpolate(x_01, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_299.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_299.device).view(1, 3, 1, 1)
    x_norm = (x_299 - mean) / std
    acts = inception(x_norm)
    if isinstance(acts, tuple):
        acts = acts[0]
    return acts.detach().cpu().numpy()


@torch.no_grad()
def make_low_rank_and_b(
    *,
    images: torch.Tensor,
    k_truncate: int,
    do_pad: bool,
):
    """Construct I_L by truncating singular values of I_H, then compute b using SVD(I_L).

    Shapes follow existing repo convention:
      images: [B, C, H, W]
      images_flat: [B, C*H, W]
      b_true: [B, r, W] (typically [B, W, W])
    """
    if do_pad:
        images = pad_to_square(images)

    b, c, h, w = images.shape
    images_flat = torch.reshape(images, [b, c * h, w])

    # Build I_L via truncation (low-rank approximation)
    U_h, S_h, Vt_h = torch.linalg.svd(images_flat)
    r = S_h.shape[1]
    k = min(k_truncate, r)
    S_truncated = torch.zeros_like(S_h)
    S_truncated[:, :k] = S_h[:, :k]
    S_L_diag = torch.diag_embed(S_truncated)
    I_L_flat = U_h[:, :, :r] @ S_L_diag @ Vt_h
    I_L = torch.reshape(I_L_flat, [b, c, h, w])

    # Compute b in the SVD basis of I_L (matches your "SVD(I_L)" definition)
    U_l, S_l, Vt_l = torch.linalg.svd(I_L_flat)
    r_l = S_l.shape[1]
    residual_flat = images_flat - I_L_flat
    b_true = torch.transpose(U_l, 1, 2)[:, :r_l, :] @ residual_flat  # [B, r_l, W]

    return I_L, I_L_flat, U_l, r_l, b_true


@torch.no_grad()
def evaluate_fid_psnr(
    *,
    model: nn.Module,
    ddpm: ConditionalBDDPM,
    testloader: DataLoader,
    device: torch.device,
    k_truncate: int,
    do_pad: bool,
    eval_batches: int,
    inception: nn.Module,
):
    model.eval()
    all_real_acts = []
    all_fake_acts = []
    psnr_values = []

    for batch_idx, (images, _) in enumerate(testloader):
        if batch_idx >= eval_batches:
            break

        images = images.to(device)
        I_L, I_L_flat, U_l, r_l, _ = make_low_rank_and_b(images=images, k_truncate=k_truncate, do_pad=do_pad)

        b_sample = ddpm.sample_loop(
            model=model,
            shape=torch.Size([images.shape[0], 1, r_l, images.shape[-1]]),
            cond=I_L,
            clip_x0=False,
        )

        b_sample_flat = b_sample.squeeze(1)
        recon_flat = I_L_flat + U_l[:, :, :r_l] @ b_sample_flat
        recon = torch.reshape(recon_flat, images.shape)

        psnr_values.append(psnr_from_batch(images, recon))

        real_acts = inception_activations(denorm_to_unit(images), inception)
        fake_acts = inception_activations(denorm_to_unit(recon), inception)
        all_real_acts.append(real_acts)
        all_fake_acts.append(fake_acts)

    real_acts_np = np.concatenate(all_real_acts, axis=0)
    fake_acts_np = np.concatenate(all_fake_acts, axis=0)
    fid = float(
        classifier_metrics_numpy.frechet_classifier_distance_from_activations(
            real_acts_np, fake_acts_np
        )
    )
    psnr = float(np.mean(psnr_values))
    return fid, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_truncate", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--out", type=str, default="b_diffusion_recon.jpg")
    parser.add_argument("--eval_metrics", action="store_true", help="Compute FID and PSNR during eval interval")
    parser.add_argument("--eval_batches", type=int, default=10, help="How many test batches to use for metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=0)

    # Model predicts eps in b-space. Input is [I_L (3ch), b_t (1ch)] => 4 channels.
    model = UNet(t_emb_dim=128, ch=64, out_ch=1, in_ch=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ddpm = ConditionalBDDPM(
        params=DDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )

    inception = build_inception_feature_extractor(device) if args.eval_metrics else None

    do_pad = True

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
        for images, _ in pbar:
            images = images.to(device)

            with torch.no_grad():
                I_L, _, _, r_l, b_true = make_low_rank_and_b(images=images, k_truncate=args.k_truncate, do_pad=do_pad)

            # b-space target as 1-channel image
            b0 = b_true.unsqueeze(1)
            t = torch.randint(0, ddpm.params.timesteps, (b0.shape[0],), device=device).long()

            loss = ddpm.training_loss(model=model, x_start=b0, cond=I_L, t=t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip(model.parameters(), mode="norm", value=1.0)
            opt.step()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), r=int(r_l))

        # quick sample visualization
        if (epoch + 1) % args.save_every == 0:
            model.eval()
            with torch.no_grad():
                images, _ = next(iter(testloader))
                images = images.to(device)

                I_L, I_L_flat, U_l, r_l, _ = make_low_rank_and_b(
                    images=images, k_truncate=args.k_truncate, do_pad=do_pad
                )

                b_sample = ddpm.sample_loop(
                    model=model,
                    shape=torch.Size([images.shape[0], 1, r_l, images.shape[-1]]),
                    cond=I_L,
                    clip_x0=False,
                )

                # Reconstruct: I_out = I_L + U @ b
                b_sample_flat = b_sample.squeeze(1)  # [B, r, W]
                recon_flat = I_L_flat + U_l[:, :, :r_l] @ b_sample_flat
                recon = torch.reshape(recon_flat, images.shape)

                pad_img = F.pad(images, (2, 2, 0, 0), "constant", 1)
                pad_low = F.pad(I_L, (2, 2, 0, 0), "constant", 1)
                pad_recon = F.pad(recon, (2, 2, 0, 0), "constant", 1)
                grid = torch.cat([pad_img, pad_low, pad_recon], dim=3)
                save_image(grid, args.out, normalize=True)
                print(f"Saved {args.out}")

            if args.eval_metrics:
                fid, psnr = evaluate_fid_psnr(
                    model=model,
                    ddpm=ddpm,
                    testloader=testloader,
                    device=device,
                    k_truncate=args.k_truncate,
                    do_pad=do_pad,
                    eval_batches=args.eval_batches,
                    inception=inception,
                )
                print(
                    f"[Eval @ epoch {epoch + 1}] FID: {fid:.4f}, PSNR: {psnr:.4f} dB "
                    f"(batches={args.eval_batches})"
                )


if __name__ == "__main__":
    main()
