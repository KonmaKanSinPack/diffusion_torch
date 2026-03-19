"""
Latent Diffusion 采样与评估脚本
================================

支持多种采样模式:
  1. ddim       — DDIM 采样 (快速, 50步)
  2. cascade    — 多尺度级联采样 (VAR 启发, 先粗后细)
  3. ddpm       — DDPM 完整采样 (1000步, 最佳质量)
  4. eval       — 批量生成 + FID/IS 评估
  5. visualize  — 频率分解可视化 (展示 VQVAE 多尺度分解效果)

完整管线:
  噪声 z_T → [DiT 去噪] → 连续 latent z_0 → [VQVAE Decoder] → 图像 x

使用方法:
  # DDIM 采样
  python sample_latent_diffusion.py \\
      --vqvae_ckpt checkpoints/vqvae/best.pt \\
      --dit_ckpt checkpoints/latent_diffusion/best.pt \\
      --mode ddim --n_samples 64

  # 多尺度级联采样
  python sample_latent_diffusion.py \\
      --vqvae_ckpt checkpoints/vqvae/best.pt \\
      --dit_ckpt checkpoints/latent_diffusion/best.pt \\
      --mode cascade

  # FID/IS 评估
  python sample_latent_diffusion.py \\
      --vqvae_ckpt checkpoints/vqvae/best.pt \\
      --dit_ckpt checkpoints/latent_diffusion/best.pt \\
      --mode eval --n_gen 10000
"""

import argparse
import os
import sys
import time
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_torch.models.vqvae import (
    MultiScaleVQVAE, VQVAEConfig, FrequencyDecomposer,
    VQVAE_Small, VQVAE_Base,
)
from diffusion_torch.models.dit import (
    DiT, LatentDiT_T, LatentDiT_S, LatentDiT_B,
)
from diffusion_torch.latent_diffusion import (
    LatentSpectralDiffusion, LatentDiffusionConfig, extract_multiscale_context,
)
from diffusion_torch.ema import load_checkpoint
from diffusion_torch.fid_utils import compute_fid, InceptionFeatureExtractor


# ============================================================
#  模型加载
# ============================================================

def load_vqvae(
    ckpt_path: str, device: torch.device,
    model_size: str = 'small', image_size: int = 32,
) -> MultiScaleVQVAE:
    """加载 VQVAE 并冻结"""
    from diffusion_torch.models.vqvae import VQVAE_Large
    vqvae_factory = {'small': VQVAE_Small, 'base': VQVAE_Base, 'large': VQVAE_Large}
    if model_size not in vqvae_factory:
        raise ValueError(f"不支持: {model_size}")
    vqvae = vqvae_factory[model_size](image_size=image_size)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        vqvae.load_state_dict(ckpt['ema_state_dict'])
    elif 'model_state_dict' in ckpt:
        vqvae.load_state_dict(ckpt['model_state_dict'])
    else:
        vqvae.load_state_dict(ckpt)

    vqvae = vqvae.to(device).eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    return vqvae


def load_dit(
    ckpt_path: str, device: torch.device,
    dit_size: str, vqvae: MultiScaleVQVAE,
    use_cross_attn: bool = True,
) -> DiT:
    """加载 DiT (使用 EMA 权重)"""
    latent_size = vqvae.cfg.latent_size
    latent_dim = vqvae.cfg.latent_dim
    context_dim = latent_dim if use_cross_attn else 0

    factory = {'T': LatentDiT_T, 'S': LatentDiT_S, 'B': LatentDiT_B}
    dit = factory[dit_size](
        latent_size=latent_size, latent_dim=latent_dim,
        context_dim=context_dim, use_cross_attn=use_cross_attn,
    )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        dit.load_state_dict(ckpt['ema_state_dict'])
    elif 'model_state_dict' in ckpt:
        dit.load_state_dict(ckpt['model_state_dict'])
    else:
        dit.load_state_dict(ckpt)

    return dit.to(device).eval()


# ============================================================
#  采样
# ============================================================

@torch.no_grad()
def generate_samples(
    dit: nn.Module,
    diffusion: LatentSpectralDiffusion,
    vqvae: MultiScaleVQVAE,
    n_samples: int,
    device: torch.device,
    mode: str = 'ddim',
    ddim_steps: int = 50,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    批量生成图像
    
    Returns:
        images: (N, 3, 32, 32) 范围 [0, 1]
    """
    dit.eval()
    vqvae.eval()

    latent_shape_batch = (
        batch_size,
        vqvae.cfg.latent_dim,
        vqvae.cfg.latent_size,
        vqvae.cfg.latent_size,
    )

    all_images = []
    n_remaining = n_samples

    while n_remaining > 0:
        bs = min(batch_size, n_remaining)
        shape = (bs, *latent_shape_batch[1:])

        if mode == 'ddim':
            z = diffusion.ddim_sample(dit, shape, num_steps=ddim_steps)
        elif mode == 'cascade':
            z = diffusion.multiscale_cascade_sample(
                dit, vqvae, shape, num_steps=ddim_steps,
            )
        elif mode == 'ddpm':
            z = diffusion.p_sample_loop(dit, shape)
        else:
            raise ValueError(f"未知采样模式: {mode}")

        # 解码
        images = vqvae.decode_continuous(z)
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        all_images.append(images.cpu())
        n_remaining -= bs

    return torch.cat(all_images, dim=0)[:n_samples]


# ============================================================
#  Inception Score 计算
# ============================================================

@torch.no_grad()
def compute_is(
    images: torch.Tensor,
    device: torch.device,
    n_splits: int = 10,
) -> Tuple[float, float]:
    """
    计算 Inception Score (IS)
    
    IS = exp(E_x[KL(p(y|x) || p(y))])
    越高越好 (CIFAR-10 真实数据 ~11)
    """
    # 使用 Inception-V3 (需要 299×299 输入)
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True, transform_input=False).to(device).eval()

    N = images.shape[0]
    all_preds = []

    for i in range(0, N, 64):
        batch = images[i:i+64].to(device)
        # 上采样到 299×299
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        logits = model(batch)
        preds = F.softmax(logits, dim=1)
        all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()

    # 分 split 计算
    split_scores = []
    split_size = N // n_splits
    for k in range(n_splits):
        part = all_preds[k * split_size: (k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        split_scores.append(np.exp(kl))

    return float(np.mean(split_scores)), float(np.std(split_scores))


# ============================================================
#  频率分解可视化
# ============================================================

@torch.no_grad()
def visualize_frequency_decomposition(
    vqvae: MultiScaleVQVAE,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    n_samples: int = 8,
):
    """
    可视化 VQVAE 的多尺度频率分解
    
    展示: 原图 | 完整重建 | 低频 | 中频 | 高频 | 各尺度独立重建
    """
    vqvae.eval()
    images, _ = next(iter(val_loader))
    images = images[:n_samples].to(device)

    decomposer = FrequencyDecomposer()
    low, mid, high, scales = decomposer.decompose(vqvae, images)

    # 完整重建
    x_recon, _, _, _ = vqvae(images)

    rows = [images, x_recon, low, mid, high] + scales
    all_imgs = torch.cat(rows, dim=0)
    all_imgs = (all_imgs + 1) / 2
    all_imgs = all_imgs.clamp(0, 1)

    grid = torchvision.utils.make_grid(all_imgs, nrow=n_samples, padding=2)
    torchvision.utils.save_image(grid, save_path)
    print(f"频率分解可视化保存到 {save_path}")
    print(f"  行1: 原图, 行2: 完整重建, 行3: 低频, 行4: 中频, 行5: 高频")
    for i in range(len(scales)):
        print(f"  行{6+i}: scale {vqvae.cfg.multi_scales[i]}×{vqvae.cfg.multi_scales[i]}")


@torch.no_grad()
def visualize_cascade_comparison(
    dit: nn.Module,
    diffusion: LatentSpectralDiffusion,
    vqvae: MultiScaleVQVAE,
    device: torch.device,
    save_path: str,
    n_samples: int = 8,
    ddim_steps: int = 50,
):
    """
    对比 DDIM vs 级联采样
    
    展示: DDIM 样本 | 级联样本
    """
    dit.eval()
    vqvae.eval()

    shape = (n_samples, vqvae.cfg.latent_dim, vqvae.cfg.latent_size, vqvae.cfg.latent_size)

    # DDIM
    z_ddim = diffusion.ddim_sample(dit, shape, num_steps=ddim_steps)
    img_ddim = vqvae.decode_continuous(z_ddim)

    # 级联
    z_cascade = diffusion.multiscale_cascade_sample(
        dit, vqvae, shape, num_steps=ddim_steps,
    )
    img_cascade = vqvae.decode_continuous(z_cascade)

    rows = [img_ddim, img_cascade]
    all_imgs = torch.cat(rows, dim=0)
    all_imgs = (all_imgs + 1) / 2
    all_imgs = all_imgs.clamp(0, 1)

    grid = torchvision.utils.make_grid(all_imgs, nrow=n_samples, padding=2)
    torchvision.utils.save_image(grid, save_path)
    print(f"采样对比保存到 {save_path}")
    print(f"  行1: DDIM, 行2: 多尺度级联")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Latent Diffusion 采样与评估')

    # 模型
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--vqvae_size', type=str, default='small', choices=['small', 'base', 'large'])
    parser.add_argument('--dit_ckpt', type=str, required=True)
    parser.add_argument('--dit_size', type=str, default='S', choices=['T', 'S', 'B'])
    parser.add_argument('--use_cross_attn', action='store_true', default=True)
    parser.add_argument('--no_cross_attn', action='store_true')

    # 采样
    parser.add_argument('--mode', type=str, default='ddim',
                        choices=['ddim', 'cascade', 'ddpm', 'eval', 'visualize'],
                        help='采样/评估模式')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--n_samples', type=int, default=64,
                        help='生成样本数 (ddim/cascade/ddpm 模式)')
    parser.add_argument('--n_gen', type=int, default=10000,
                        help='eval 模式生成数量')
    parser.add_argument('--batch_size', type=int, default=64)

    # Diffusion 配置
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--pred_type', type=str, default='v')

    # 数据 (eval/visualize 模式需要)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=32,
                        help='图像尺寸, 必须与训练时一致')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./samples')

    args = parser.parse_args()

    if args.no_cross_attn:
        args.use_cross_attn = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 加载模型 ----
    print("加载 VQVAE...")
    vqvae = load_vqvae(args.vqvae_ckpt, device, args.vqvae_size, args.image_size)

    print(f"加载 DiT-{args.dit_size}...")
    dit = load_dit(args.dit_ckpt, device, args.dit_size, vqvae, args.use_cross_attn)

    # ---- Diffusion ----
    diff_cfg = LatentDiffusionConfig(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        pred_type=args.pred_type,
        latent_dim=vqvae.cfg.latent_dim,
        latent_size=vqvae.cfg.latent_size,
    )
    diffusion = LatentSpectralDiffusion(diff_cfg).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 模式分发 ----
    if args.mode in ('ddim', 'cascade', 'ddpm'):
        print(f"\n{args.mode.upper()} 采样 {args.n_samples} 张图像...")
        t0 = time.time()

        images = generate_samples(
            dit, diffusion, vqvae, args.n_samples, device,
            mode=args.mode, ddim_steps=args.ddim_steps,
            batch_size=args.batch_size,
        )

        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.1f}s ({elapsed/args.n_samples:.2f}s/张)")

        # 保存
        save_path = os.path.join(args.output_dir, f'{args.mode}_samples.png')
        grid = torchvision.utils.make_grid(images[:min(64, args.n_samples)], nrow=8, padding=2)
        torchvision.utils.save_image(grid, save_path)
        print(f"  保存到 {save_path}")

        # 保存单独图片
        img_dir = os.path.join(args.output_dir, f'{args.mode}_images')
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(images):
            torchvision.utils.save_image(img, os.path.join(img_dir, f'{i:05d}.png'))
        print(f"  单独图像保存到 {img_dir}/")

    elif args.mode == 'eval':
        print(f"\n评估模式: 生成 {args.n_gen} 张并计算 FID/IS...")

        # 加载验证集
        if args.dataset == 'imagenet':
            val_transform = T.Compose([
                T.Resize(int(args.image_size * 1.14)),
                T.CenterCrop(args.image_size),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3),
            ])
            val_set = torchvision.datasets.ImageFolder(
                os.path.join(args.data_dir, 'val'), transform=val_transform)
        else:
            resize_ops = [T.Resize((args.image_size, args.image_size))] \
                if args.image_size != 32 else []
            transform = T.Compose(resize_ops + [
                T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
            ])
            ds_cls = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' \
                else torchvision.datasets.CIFAR100
            val_set = ds_cls(args.data_dir, train=False, download=True, transform=transform)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)

        # 生成
        t0 = time.time()
        images = generate_samples(
            dit, diffusion, vqvae, args.n_gen, device,
            mode='ddim', ddim_steps=args.ddim_steps,
            batch_size=args.batch_size,
        )
        gen_time = time.time() - t0
        print(f"  生成耗时: {gen_time:.1f}s")

        # FID
        real_images = []
        for imgs, _ in val_loader:
            real_images.append((imgs + 1) / 2)
            if sum(r.shape[0] for r in real_images) >= args.n_gen:
                break
        real_images = torch.cat(real_images, dim=0)[:args.n_gen]

        fid = compute_fid(real_images, images, device=device)
        print(f"  FID: {fid:.2f}")

        # IS
        is_mean, is_std = compute_is(images, device)
        print(f"  IS: {is_mean:.2f} ± {is_std:.2f}")

        # 保存结果
        result_path = os.path.join(args.output_dir, 'eval_results.txt')
        with open(result_path, 'w') as f:
            f.write(f"FID: {fid:.4f}\n")
            f.write(f"IS: {is_mean:.4f} ± {is_std:.4f}\n")
            f.write(f"n_gen: {args.n_gen}\n")
            f.write(f"ddim_steps: {args.ddim_steps}\n")
        print(f"  结果保存到 {result_path}")

        # 样本可视化
        grid = torchvision.utils.make_grid(images[:64], nrow=8, padding=2)
        torchvision.utils.save_image(grid, os.path.join(args.output_dir, 'eval_samples.png'))

    elif args.mode == 'visualize':
        print("\n可视化模式...")

        # 加载验证集
        if args.dataset == 'imagenet':
            val_transform = T.Compose([
                T.Resize(int(args.image_size * 1.14)),
                T.CenterCrop(args.image_size),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3),
            ])
            val_set = torchvision.datasets.ImageFolder(
                os.path.join(args.data_dir, 'val'), transform=val_transform)
        else:
            resize_ops = [T.Resize((args.image_size, args.image_size))] \
                if args.image_size != 32 else []
            transform = T.Compose(resize_ops + [
                T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
            ])
            ds_cls = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' \
                else torchvision.datasets.CIFAR100
            val_set = ds_cls(args.data_dir, train=False, download=True, transform=transform)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)

        # 1. VQVAE 频率分解
        visualize_frequency_decomposition(
            vqvae, val_loader, device,
            os.path.join(args.output_dir, 'freq_decomposition.png'),
        )

        # 2. DDIM vs 级联采样对比
        visualize_cascade_comparison(
            dit, diffusion, vqvae, device,
            os.path.join(args.output_dir, 'ddim_vs_cascade.png'),
            ddim_steps=args.ddim_steps,
        )

        print("\n可视化完成!")

    print("\n完成!")


if __name__ == '__main__':
    main()
