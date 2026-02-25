import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.unet import UNet
from image_diffusion import ImageDDPM, DDPMParams as ImgDDPMParams
from b_diffusion import ConditionalBDDPM, DDPMParams as BDDPMParams
from ema import EMA, load_checkpoint


def reconstruct_from_il_and_b(I_L: torch.Tensor, b_sample: torch.Tensor) -> torch.Tensor:
    # I_L: [B,3,H,W]
    b, c, h, w = I_L.shape
    I_L_flat = torch.reshape(I_L, [b, c * h, w])
    U_l, S_l, _ = torch.linalg.svd(I_L_flat)
    r_l = S_l.shape[1]

    b_sample_flat = b_sample.squeeze(1)  # [B, r, W]
    recon_flat = I_L_flat + U_l[:, :, :r_l] @ b_sample_flat
    recon = torch.reshape(recon_flat, [b, c, h, w])
    return recon


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    p.add_argument("--il_ckpt", type=str, required=True)
    p.add_argument("--b_ckpt", type=str, required=True)

    p.add_argument("--out", type=str, default="two_stage_samples.jpg")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: unconditional I_L generator
    il_model = UNet(t_emb_dim=128, ch=128, out_ch=3, in_ch=3).to(device)
    il_opt_dummy = torch.optim.AdamW(il_model.parameters(), lr=1e-4)
    il_ema = EMA(il_model)
    load_checkpoint(args.il_ckpt, model=il_model, opt=il_opt_dummy, ema=il_ema, map_location=device)
    il_model = il_ema.ema_model.to(device).eval()

    il_ddpm = ImageDDPM(
        params=ImgDDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )

    # Stage 2: conditional b generator
    b_model = UNet(t_emb_dim=128, ch=64, out_ch=1, in_ch=4).to(device)
    b_opt_dummy = torch.optim.AdamW(b_model.parameters(), lr=1e-4)
    b_ema = EMA(b_model)
    load_checkpoint(args.b_ckpt, model=b_model, opt=b_opt_dummy, ema=b_ema, map_location=device)
    b_model = b_ema.ema_model.to(device).eval()

    b_ddpm = ConditionalBDDPM(
        params=BDDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )

    all_out = []
    generated = 0
    for _ in range((args.n + args.batch - 1) // args.batch):
        cur = min(args.batch, args.n - generated)
        if cur <= 0:
            break
        I_L = il_ddpm.ddim_sample_loop(
            model=il_model,
            shape=torch.Size([cur, 3, 32, 32]),
            steps=args.sample_steps,
            eta=args.ddim_eta,
            clip_x0=True,
        )

        # b shape depends on r_l from SVD(I_L)
        I_L_flat = torch.reshape(I_L, [cur, 3 * 32, 32])
        _, S_l, _ = torch.linalg.svd(I_L_flat)
        r_l = S_l.shape[1]

        b_sample = b_ddpm.ddim_sample_loop(
            model=b_model,
            shape=torch.Size([cur, 1, r_l, 32]),
            cond=I_L,
            steps=args.sample_steps,
            eta=args.ddim_eta,
            clip_x0=False,
        )

        out = reconstruct_from_il_and_b(I_L, b_sample)
        all_out.append(out)
        generated += cur

    imgs = torch.cat(all_out, dim=0)[: args.n]
    save_image(imgs, args.out, nrow=8, normalize=True)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
