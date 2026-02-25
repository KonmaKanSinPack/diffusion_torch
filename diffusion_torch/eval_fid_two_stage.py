import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from fid_utils import InceptionFeatureExtractor, compute_fid, collect_n_images_from_loader, maybe_load_cached_acts, save_cached_acts
from two_stage_sample import reconstruct_from_il_and_b
from models.unet import UNet
from image_diffusion import ImageDDPM, DDPMParams as ImgDDPMParams
from b_diffusion import ConditionalBDDPM, DDPMParams as BDDPMParams
from ema import EMA, load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--real_split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--num", type=int, default=10000)
    p.add_argument("--gen_batch", type=int, default=64)
    p.add_argument("--inception_batch", type=int, default=64)

    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    p.add_argument("--il_ckpt", type=str, required=True)
    p.add_argument("--b_ckpt", type=str, required=True)

    p.add_argument("--cache_real", type=str, default="cifar10_real_inception_acts.npy")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # real data loader (normalized to [-1,1])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    is_train = args.real_split == "train"
    dset = torchvision.datasets.CIFAR10(root=args.data, train=is_train, download=True, transform=transform)
    loader = DataLoader(dset, batch_size=args.gen_batch, shuffle=False, num_workers=0)

    feat = InceptionFeatureExtractor(device=device)

    # real activations (cache)
    real_acts = maybe_load_cached_acts(args.cache_real)
    if real_acts is None or real_acts.shape[0] < args.num:
        print("Computing real activations...")
        real_imgs = collect_n_images_from_loader(loader, args.num, device=device)
        real_acts = feat.activations(real_imgs, batch_size=args.inception_batch)
        save_cached_acts(args.cache_real, real_acts)
        print(f"Cached real activations to {args.cache_real}")
    else:
        real_acts = real_acts[: args.num]
        print(f"Loaded cached real activations: {real_acts.shape}")

    # load models (EMA)
    il_model = UNet(t_emb_dim=128, ch=128, out_ch=3, in_ch=3).to(device)
    il_opt_dummy = torch.optim.AdamW(il_model.parameters(), lr=1e-4)
    il_ema = EMA(il_model)
    load_checkpoint(args.il_ckpt, model=il_model, opt=il_opt_dummy, ema=il_ema, map_location=device)
    il_model = il_ema.ema_model.to(device).eval()

    b_model = UNet(t_emb_dim=128, ch=64, out_ch=1, in_ch=4).to(device)
    b_opt_dummy = torch.optim.AdamW(b_model.parameters(), lr=1e-4)
    b_ema = EMA(b_model)
    load_checkpoint(args.b_ckpt, model=b_model, opt=b_opt_dummy, ema=b_ema, map_location=device)
    b_model = b_ema.ema_model.to(device).eval()

    il_ddpm = ImageDDPM(
        params=ImgDDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )
    b_ddpm = ConditionalBDDPM(
        params=BDDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )

    # generate activations
    print("Generating samples and computing activations...")
    gen_acts_list = []
    remaining = args.num
    pbar = tqdm(total=args.num)
    while remaining > 0:
        cur = min(args.gen_batch, remaining)
        I_L = il_ddpm.ddim_sample_loop(
            model=il_model,
            shape=torch.Size([cur, 3, 32, 32]),
            steps=args.sample_steps,
            eta=args.ddim_eta,
            clip_x0=True,
        )

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

        acts = feat.activations(out, batch_size=args.inception_batch)
        gen_acts_list.append(acts)

        remaining -= cur
        pbar.update(cur)
    pbar.close()

    gen_acts = np.concatenate(gen_acts_list, axis=0)[: args.num]

    fid = compute_fid(real_acts, gen_acts)
    print(f"FID({args.num}): {fid:.4f}")


if __name__ == "__main__":
    main()
