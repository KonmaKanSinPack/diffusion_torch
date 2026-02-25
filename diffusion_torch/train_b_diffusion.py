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
from torchvision.utils import save_image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.unet import UNet
from b_diffusion import ConditionalBDDPM, DDPMParams
from ema import EMA, save_checkpoint, load_checkpoint
from image_diffusion import ImageDDPM, DDPMParams as ImgDDPMParams
from fid_utils import InceptionFeatureExtractor, compute_fid, collect_n_images_from_loader, maybe_load_cached_acts, save_cached_acts


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
def make_low_rank(
    *,
    images: torch.Tensor,
    k_truncate: int,
    do_pad: bool,
) -> torch.Tensor:
    if do_pad:
        images = pad_to_square(images)

    b, c, h, w = images.shape
    images_flat = torch.reshape(images, [b, c * h, w])
    U_h, S_h, Vt_h = torch.linalg.svd(images_flat)
    r = S_h.shape[1]
    k = min(k_truncate, r)
    S_truncated = torch.zeros_like(S_h)
    S_truncated[:, :k] = S_h[:, :k]
    S_L_diag = torch.diag_embed(S_truncated)
    I_L_flat = U_h[:, :, :r] @ S_L_diag @ Vt_h
    I_L = torch.reshape(I_L_flat, [b, c, h, w])
    return I_L


@torch.no_grad()
def compute_b_from_cond(
    *,
    images: torch.Tensor,
    cond: torch.Tensor,
    do_pad: bool,
):
    """Given I_H=images and a condition image cond (same shape), compute SVD(cond) and b = U^T (I_H - cond)."""
    if do_pad:
        images = pad_to_square(images)
        cond = pad_to_square(cond)

    b, c, h, w = images.shape
    images_flat = torch.reshape(images, [b, c * h, w])
    cond_flat = torch.reshape(cond, [b, c * h, w])

    U_l, S_l, _ = torch.linalg.svd(cond_flat)
    r_l = S_l.shape[1]
    residual_flat = images_flat - cond_flat
    b_true = torch.transpose(U_l, 1, 2)[:, :r_l, :] @ residual_flat
    return cond_flat, U_l, r_l, b_true


@torch.no_grad()
def psnr_db(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """PSNR in dB, expects x,y in [-1,1]."""
    x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    y01 = ((y + 1.0) / 2.0).clamp(0.0, 1.0)
    mse = torch.mean((x01 - y01) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


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

    # Conditioning augmentation (reduce IL->b distribution shift)
    parser.add_argument("--il_ckpt", type=str, default=None, help="Stage-1 I_L DDPM checkpoint (for conditioning augmentation + FID).")
    parser.add_argument("--cond_aug_prob", type=float, default=0.25, help="Probability of replacing true I_L with IL-model denoised sample.")
    parser.add_argument("--cond_aug_t", type=int, default=50, help="Forward noising timestep for I_L conditioning augmentation.")
    parser.add_argument("--cond_aug_steps", type=int, default=25, help="DDIM steps to denoise from cond_aug_t back to 0.")
    parser.add_argument("--cond_aug_warmup_epochs", type=int, default=5, help="Start conditioning augmentation after this epoch.")

    # Per-epoch evaluation
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_psnr", action="store_true")
    parser.add_argument("--psnr_num", type=int, default=2048)
    parser.add_argument("--eval_fid", action="store_true")
    parser.add_argument("--fid_num", type=int, default=2048)
    parser.add_argument("--real_split", type=str, default="train", choices=["train", "test"], help="Real split used for FID reference.")
    parser.add_argument("--cache_real", type=str, default="cifar10_real_inception_acts.npy")

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

    # Optional stage-1 I_L model for conditioning augmentation and FID.
    il_model = None
    il_ddpm = None
    if args.il_ckpt is not None:
        il_net = UNet(t_emb_dim=128, ch=128, out_ch=3, in_ch=3).to(device)
        il_opt_dummy = torch.optim.AdamW(il_net.parameters(), lr=1e-4)
        il_ema = EMA(il_net)
        load_checkpoint(args.il_ckpt, model=il_net, opt=il_opt_dummy, ema=il_ema, map_location=device)
        il_model = il_ema.ema_model.to(device).eval()
        il_ddpm = ImageDDPM(
            params=ImgDDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
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
                I_L_true = make_low_rank(images=images, k_truncate=args.k_truncate, do_pad=do_pad)

                # Conditioning augmentation: turn true I_L into a model-like sample \hat I_L.
                use_aug = (
                    il_model is not None
                    and il_ddpm is not None
                    and epoch >= args.cond_aug_warmup_epochs
                    and args.cond_aug_prob > 0
                    and (torch.rand(()) < args.cond_aug_prob)
                )
                if use_aug:
                    t_aug = torch.full((I_L_true.shape[0],), int(args.cond_aug_t), device=device, dtype=torch.long)
                    noise = torch.randn_like(I_L_true)
                    I_L_noisy = il_ddpm.q_sample(x_start=I_L_true, t=t_aug, noise=noise)
                    I_L = il_ddpm.ddim_denoise_from(
                        model=il_model,
                        x_t=I_L_noisy,
                        t_start=t_aug,
                        steps=args.cond_aug_steps,
                        eta=0.0,
                        clip_x0=True,
                    )
                else:
                    I_L = I_L_true

                I_L_flat, U_l, r_l, b_true = compute_b_from_cond(images=images, cond=I_L, do_pad=do_pad)

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

        # quick sample visualization + checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.ckpt, model=model, opt=opt, ema=ema, step=global_step)

            model_ema = ema.ema_model.to(device)
            model_ema.eval()
            with torch.no_grad():
                images, _ = next(iter(testloader))
                images = images.to(device)

                I_L = make_low_rank(images=images, k_truncate=args.k_truncate, do_pad=do_pad)
                I_L_flat, U_l, r_l, _ = compute_b_from_cond(images=images, cond=I_L, do_pad=do_pad)

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

        # === Per-epoch evaluation ===
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0):
            model_ema = ema.ema_model.to(device).eval()

            if args.eval_psnr:
                psnr_vals = []
                seen = 0
                with torch.no_grad():
                    for images, _ in testloader:
                        images = images.to(device)
                        I_L = make_low_rank(images=images, k_truncate=args.k_truncate, do_pad=do_pad)
                        I_L_flat, U_l, r_l, _ = compute_b_from_cond(images=images, cond=I_L, do_pad=do_pad)

                        b_sample = ddpm.ddim_sample_loop(
                            model=model_ema,
                            shape=torch.Size([images.shape[0], 1, r_l, images.shape[-1]]),
                            cond=I_L,
                            steps=args.sample_steps,
                            eta=args.ddim_eta,
                            clip_x0=False,
                        )

                        recon_flat = I_L_flat + U_l[:, :, :r_l] @ b_sample.squeeze(1)
                        recon = torch.reshape(recon_flat, images.shape)
                        psnr_vals.append(psnr_db(recon, images))

                        seen += images.shape[0]
                        if seen >= args.psnr_num:
                            break
                psnr_mean = torch.cat(psnr_vals, dim=0)[: args.psnr_num].mean().item()
                print(f"[Eval] epoch={epoch} PSNR(dB)={psnr_mean:.3f} (n={min(seen, args.psnr_num)})")

            if args.eval_fid:
                if il_model is None or il_ddpm is None:
                    print("[Eval] FID skipped: --il_ckpt not provided")
                else:
                    # real activations cache
                    is_train = args.real_split == "train"
                    real_dset = torchvision.datasets.CIFAR10(root=args.data, train=is_train, download=True, transform=transform)
                    real_loader = DataLoader(real_dset, batch_size=args.batch, shuffle=False, num_workers=0)

                    feat = InceptionFeatureExtractor(device=device)
                    real_acts = maybe_load_cached_acts(args.cache_real)
                    if real_acts is None or real_acts.shape[0] < args.fid_num:
                        real_imgs = collect_n_images_from_loader(real_loader, args.fid_num, device=device)
                        real_acts = feat.activations(real_imgs, batch_size=args.batch)
                        save_cached_acts(args.cache_real, real_acts)
                    else:
                        real_acts = real_acts[: args.fid_num]

                    # generate unconditional two-stage samples
                    gen_acts_list = []
                    remaining = args.fid_num
                    with torch.no_grad():
                        while remaining > 0:
                            cur = min(args.batch, remaining)
                            I_L_s = il_ddpm.ddim_sample_loop(
                                model=il_model,
                                shape=torch.Size([cur, 3, 32, 32]),
                                steps=args.sample_steps,
                                eta=args.ddim_eta,
                                clip_x0=True,
                            )

                            I_L_flat = torch.reshape(I_L_s, [cur, 3 * 32, 32])
                            U_l, S_l, _ = torch.linalg.svd(I_L_flat)
                            r_l = S_l.shape[1]

                            b_s = ddpm.ddim_sample_loop(
                                model=model_ema,
                                shape=torch.Size([cur, 1, r_l, 32]),
                                cond=I_L_s,
                                steps=args.sample_steps,
                                eta=args.ddim_eta,
                                clip_x0=False,
                            )
                            out_flat = I_L_flat + U_l[:, :, :r_l] @ b_s.squeeze(1)
                            out = torch.reshape(out_flat, [cur, 3, 32, 32])

                            gen_acts_list.append(feat.activations(out, batch_size=args.batch))
                            remaining -= cur

                    gen_acts = np.concatenate(gen_acts_list, axis=0)[: args.fid_num]
                    fid_val = compute_fid(real_acts, gen_acts)
                    print(f"[Eval] epoch={epoch} FID={fid_val:.4f} (n={args.fid_num})")


if __name__ == "__main__":
    main()
