import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.unet import UNet
from b_diffusion import ConditionalBDDPM, DDPMParams
from ema import EMA, save_checkpoint, load_checkpoint


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
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ckpt", type=str, default="b_ddpm_ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--out", type=str, default="b_diffusion_recon.jpg")
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
    ema = EMA(model, decay=args.ema_decay)

    ddpm = ConditionalBDDPM(
        params=DDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )

    do_pad = True

    global_step = 0
    if args.resume and os.path.exists(args.ckpt):
        global_step = load_checkpoint(args.ckpt, model=model, opt=opt, ema=ema, map_location=device)
        print(f"Resumed from {args.ckpt} at step {global_step}")

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
            ema.update(model)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), r=int(r_l))

        # quick sample visualization
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.ckpt, model=model, opt=opt, ema=ema, step=global_step)

            model_ema = ema.ema_model.to(device)
            model_ema.eval()
            with torch.no_grad():
                images, _ = next(iter(testloader))
                images = images.to(device)

                I_L, I_L_flat, U_l, r_l, _ = make_low_rank_and_b(
                    images=images, k_truncate=args.k_truncate, do_pad=do_pad
                )

                # Faster sampling for preview
                b_sample = ddpm.ddim_sample_loop(
                    model=model_ema,
                    shape=torch.Size([images.shape[0], 1, r_l, images.shape[-1]]),
                    cond=I_L,
                    steps=args.sample_steps,
                    eta=args.ddim_eta,
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
                print(f"Saved checkpoint {args.ckpt}")


if __name__ == "__main__":
    main()
