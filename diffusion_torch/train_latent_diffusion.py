"""
Stage 2: Latent SpectralDiffusion 训练脚本
==========================================

前提: Stage 1 已训练好 Multi-Scale VQVAE

训练流程:
  1. 加载冻结的 VQVAE
  2. 构建 Latent DiT (在 VQVAE latent 空间操作)
  3. 构建 LatentSpectralDiffusion (管理扩散过程)
  4. 训练循环:
     a. 图像 → VQVAE encoder → 连续 latent z
     b. (可选) 提取粗尺度 context tokens
     c. 随机加噪 z → z_t
     d. DiT 预测去噪
     e. min-SNR-γ 加权损失 + SVD 辅助损失
  5. 定期生成样本 → VQVAE decoder → 像素图像 → FID 评估

使用方法:
  python train_latent_diffusion.py \\
      --vqvae_ckpt checkpoints/vqvae/best.pt \\
      --dataset cifar10 \\
      --dit_size S \\
      --epochs 500 \\
      --batch_size 128
"""

import argparse
import os
import sys
import time
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_torch.models.vqvae import (
    MultiScaleVQVAE, VQVAEConfig, VQVAE_Small, VQVAE_Base,
)
from diffusion_torch.models.dit import (
    DiT, LatentDiT_T, LatentDiT_S, LatentDiT_B,
)
from diffusion_torch.latent_diffusion import (
    LatentSpectralDiffusion, LatentDiffusionConfig, extract_multiscale_context,
)
from diffusion_torch.ema import EMA, save_checkpoint, load_checkpoint
from diffusion_torch.fid_utils import compute_fid


# ============================================================
#  数据集
# ============================================================

def get_dataset(
    name: str,
    data_dir: str = './data',
    batch_size: int = 128,
    image_size: int = 32,
):
    """加载数据集, 归一化到 [-1, 1]"""
    need_resize = (name in ('cifar10', 'cifar100') and image_size != 32) or \
                  name == 'imagenet'
    resize_ops = [T.Resize((image_size, image_size))] if need_resize else []

    transform_train = T.Compose(
        resize_ops + [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    transform_val = T.Compose(
        resize_ops + [
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    if name == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform_val)
    elif name == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform_val)
    elif name == 'imagenet':
        train_transform = T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        val_transform = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        train_set = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform=train_transform)
        val_set = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'val'), transform=val_transform)
    else:
        raise ValueError(f"不支持的数据集: {name}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=min(256, batch_size * 2), shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


# ============================================================
#  学习率调度
# ============================================================

def cosine_lr_schedule(
    optimizer, epoch, total_epochs, warmup_epochs=10, base_lr=1e-4, min_lr=1e-6,
) -> float:
    """余弦退火 + 线性 warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ============================================================
#  FID 评估
# ============================================================

@torch.no_grad()
def evaluate_fid(
    dit_model: nn.Module,
    diffusion: LatentSpectralDiffusion,
    vqvae: MultiScaleVQVAE,
    val_loader: DataLoader,
    device: torch.device,
    n_gen: int = 5000,
    ddim_steps: int = 50,
    use_cascade: bool = False,
) -> float:
    """
    生成样本并计算 FID
    
    流程:
    1. 采样 latent z (通过 DDIM 或级联采样)
    2. VQVAE decoder: z → 图像
    3. 计算 FID (与验证集比较)
    """
    dit_model.eval()
    vqvae.eval()

    latent_shape = (
        min(64, n_gen),  # 每批最多 64 个
        vqvae.cfg.latent_dim,
        vqvae.cfg.latent_size,
        vqvae.cfg.latent_size,
    )

    generated = []
    n_remaining = n_gen

    while n_remaining > 0:
        bs = min(latent_shape[0], n_remaining)
        shape = (bs, *latent_shape[1:])

        if use_cascade:
            z = diffusion.multiscale_cascade_sample(
                dit_model, vqvae, shape, num_steps=ddim_steps,
            )
        else:
            z = diffusion.ddim_sample(
                dit_model, shape, num_steps=ddim_steps,
            )

        # VQVAE 解码: latent → 图像
        images = vqvae.decode_continuous(z)  # (B, 3, 32, 32)
        images = (images + 1) / 2  # [-1,1] → [0,1]
        images = images.clamp(0, 1)
        generated.append(images.cpu())
        n_remaining -= bs

    generated = torch.cat(generated, dim=0)[:n_gen]

    # 收集真实图像
    real_images = []
    for imgs, _ in val_loader:
        real_images.append((imgs + 1) / 2)  # [-1,1] → [0,1]
        if sum(r.shape[0] for r in real_images) >= n_gen:
            break
    real_images = torch.cat(real_images, dim=0)[:n_gen]

    # 计算 FID
    fid = compute_fid(real_images, generated, device=device)
    return fid


# ============================================================
#  可视化
# ============================================================

@torch.no_grad()
def save_samples_vis(
    dit_model: nn.Module,
    diffusion: LatentSpectralDiffusion,
    vqvae: MultiScaleVQVAE,
    device: torch.device,
    save_path: str,
    n_samples: int = 64,
    ddim_steps: int = 50,
):
    """生成样本并保存可视化"""
    dit_model.eval()
    vqvae.eval()

    shape = (n_samples, vqvae.cfg.latent_dim, vqvae.cfg.latent_size, vqvae.cfg.latent_size)
    z = diffusion.ddim_sample(dit_model, shape, num_steps=ddim_steps)
    images = vqvae.decode_continuous(z)
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    torchvision.utils.save_image(grid, save_path)
    print(f"  [可视化] 样本保存到 {save_path}")


# ============================================================
#  加载冻结的 VQVAE
# ============================================================

def load_frozen_vqvae(
    ckpt_path: str, device: torch.device, model_size: str = 'small',
    image_size: int = 32,
) -> MultiScaleVQVAE:
    """
    加载训练好的 VQVAE 并冻结全部参数
    
    注意: 使用 EMA 权重 (如果有的话), 因为 EMA 重建质量通常更好
    """
    from diffusion_torch.models.vqvae import VQVAE_Large
    vqvae_factory = {'small': VQVAE_Small, 'base': VQVAE_Base, 'large': VQVAE_Large}
    if model_size not in vqvae_factory:
        raise ValueError(f"不支持的 VQVAE 规模: {model_size}")
    vqvae = vqvae_factory[model_size](image_size=image_size)

    # 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 优先使用 EMA 权重
    if 'ema_state_dict' in ckpt:
        vqvae.load_state_dict(ckpt['ema_state_dict'])
        print(f"  [VQVAE] 加载 EMA 权重 (from {ckpt_path})")
    elif 'model_state_dict' in ckpt:
        vqvae.load_state_dict(ckpt['model_state_dict'])
        print(f"  [VQVAE] 加载模型权重 (from {ckpt_path})")
    else:
        # 兼容直接保存的 state_dict
        vqvae.load_state_dict(ckpt)
        print(f"  [VQVAE] 加载 state_dict (from {ckpt_path})")

    vqvae = vqvae.to(device)

    # 冻结全部参数
    for param in vqvae.parameters():
        param.requires_grad = False
    vqvae.eval()

    print(f"  [VQVAE] 已冻结, latent_dim={vqvae.cfg.latent_dim}, "
          f"latent_size={vqvae.cfg.latent_size}, scales={vqvae.cfg.multi_scales}")

    return vqvae


# ============================================================
#  训练主循环
# ============================================================

def train_epoch(
    dit_model: nn.Module,
    diffusion: LatentSpectralDiffusion,
    vqvae: MultiScaleVQVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: EMA,
    use_multiscale_cond: bool = True,
    cond_scales: list = None,
) -> dict:
    """单个 epoch 训练"""
    dit_model.train()
    total_loss = 0.0
    total_main = 0.0
    total_svd = 0.0
    n_batches = 0

    for images, _ in train_loader:
        images = images.to(device)

        # 1. VQVAE 编码: 图像 → 连续 latent
        with torch.no_grad():
            z_0 = vqvae.get_latent(images)  # (B, D, H_lat, W_lat)

            # 2. (可选) 提取多尺度条件
            context = None
            if use_multiscale_cond and cond_scales:
                context = extract_multiscale_context(vqvae, z_0, cond_scales)

        # 3. Diffusion 训练损失
        loss, log_dict = diffusion.training_loss(dit_model, z_0, context=context)

        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update(dit_model)

        total_loss += log_dict['loss_total']
        total_main += log_dict['loss_main']
        total_svd += log_dict.get('loss_svd_low', 0.0) + log_dict.get('loss_svd_high', 0.0)
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'main': total_main / n_batches,
        'svd': total_svd / n_batches,
    }


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: 在 VQVAE latent 空间训练 Diffusion'
    )

    # VQVAE
    parser.add_argument('--vqvae_ckpt', type=str, required=True,
                        help='Stage 1 训练好的 VQVAE checkpoint 路径')
    parser.add_argument('--vqvae_size', type=str, default='small',
                        choices=['small', 'base', 'large'])

    # 数据
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录 (ImageNet 需含 train/ 和 val/ 子目录)')
    parser.add_argument('--image_size', type=int, default=32,
                        help='图像尺寸, 必须与 VQVAE 训练时一致')

    # DiT 模型
    parser.add_argument('--dit_size', type=str, default='S',
                        choices=['T', 'S', 'B'],
                        help='DiT 模型规模')

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'])
    parser.add_argument('--pred_type', type=str, default='v',
                        choices=['eps', 'v'])
    parser.add_argument('--min_snr_gamma', type=float, default=5.0)
    parser.add_argument('--svd_aux_weight', type=float, default=0.1)
    parser.add_argument('--use_svd_aux', action='store_true', default=True)
    parser.add_argument('--no_svd_aux', action='store_true')

    # 多尺度条件
    parser.add_argument('--use_multiscale_cond', action='store_true', default=True)
    parser.add_argument('--no_multiscale_cond', action='store_true')
    parser.add_argument('--cond_scales', type=int, nargs='+', default=[1, 2],
                        help='用于条件化的粗尺度')

    # 训练
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    # 评估
    parser.add_argument('--fid_interval', type=int, default=50,
                        help='每 N 个 epoch 计算 FID')
    parser.add_argument('--fid_n_samples', type=int, default=5000)
    parser.add_argument('--ddim_steps', type=int, default=50)

    # 输出
    parser.add_argument('--output_dir', type=str, default='./checkpoints/latent_diffusion')
    parser.add_argument('--vis_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)

    # 恢复
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # 处理互斥参数
    if args.no_svd_aux:
        args.use_svd_aux = False
    if args.no_multiscale_cond:
        args.use_multiscale_cond = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 加载冻结的 VQVAE ----
    print(f"\n加载 VQVAE ({args.vqvae_size})...")
    vqvae = load_frozen_vqvae(args.vqvae_ckpt, device, args.vqvae_size, args.image_size)

    # ---- 数据集 ----
    print(f"\n加载数据集: {args.dataset}")
    train_loader, val_loader = get_dataset(
        args.dataset, args.data_dir, args.batch_size, args.image_size,
    )
    print(f"  训练集: {len(train_loader.dataset)} 样本")

    # ---- Diffusion 配置 ----
    diff_cfg = LatentDiffusionConfig(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        pred_type=args.pred_type,
        min_snr_gamma=args.min_snr_gamma,
        svd_aux_weight=args.svd_aux_weight,
        use_svd_aux=args.use_svd_aux,
        use_multiscale_cond=args.use_multiscale_cond,
        cond_scales=args.cond_scales,
        latent_dim=vqvae.cfg.latent_dim,
        latent_size=vqvae.cfg.latent_size,
    )
    diffusion = LatentSpectralDiffusion(diff_cfg).to(device)

    # ---- DiT 模型 ----
    latent_size = vqvae.cfg.latent_size
    latent_dim = vqvae.cfg.latent_dim
    use_cross = args.use_multiscale_cond
    context_dim = latent_dim if use_cross else 0

    dit_factory = {'T': LatentDiT_T, 'S': LatentDiT_S, 'B': LatentDiT_B}
    dit_model = dit_factory[args.dit_size](
        latent_size=latent_size,
        latent_dim=latent_dim,
        context_dim=context_dim,
        use_cross_attn=use_cross,
    ).to(device)

    n_params = sum(p.numel() for p in dit_model.parameters())
    print(f"\nDiT-{args.dit_size} (Latent)")
    print(f"  参数量: {n_params:,}")
    print(f"  输入: ({latent_dim}, {latent_size}, {latent_size})")
    print(f"  CrossAttn: {use_cross}")
    if use_cross:
        print(f"  条件尺度: {args.cond_scales}")

    # ---- 优化器与 EMA ----
    optimizer = torch.optim.AdamW(
        dit_model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99),
    )
    ema = EMA(dit_model, decay=args.ema_decay)

    # ---- 恢复训练 ----
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, dit_model, ema, optimizer)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"从 epoch {start_epoch} 恢复训练")

    # ---- 输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- 训练循环 ----
    print(f"\n开始 Stage 2 训练 (共 {args.epochs} epochs)...")
    best_fid = float('inf')

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr, min_lr=args.min_lr,
        )

        t0 = time.time()
        log = train_epoch(
            dit_model, diffusion, vqvae, train_loader, optimizer, device, ema,
            use_multiscale_cond=args.use_multiscale_cond,
            cond_scales=args.cond_scales,
        )
        t_elapsed = time.time() - t0

        print(
            f"[Epoch {epoch+1:>4d}/{args.epochs}]  "
            f"loss={log['loss']:.4f}  main={log['main']:.4f}  "
            f"svd={log['svd']:.4f}  lr={lr:.2e}  time={t_elapsed:.1f}s"
        )

        # ---- 可视化 ----
        if (epoch + 1) % args.vis_interval == 0:
            ema.apply_shadow(dit_model)
            vis_path = os.path.join(vis_dir, f'samples_epoch{epoch+1:04d}.png')
            save_samples_vis(
                dit_model, diffusion, vqvae, device, vis_path,
                ddim_steps=args.ddim_steps,
            )
            ema.restore(dit_model)

        # ---- FID 评估 ----
        if (epoch + 1) % args.fid_interval == 0:
            ema.apply_shadow(dit_model)
            fid = evaluate_fid(
                dit_model, diffusion, vqvae, val_loader, device,
                n_gen=args.fid_n_samples, ddim_steps=args.ddim_steps,
            )
            print(f"  [FID] {fid:.2f}")
            ema.restore(dit_model)

            if fid < best_fid:
                best_fid = fid
                save_checkpoint(
                    os.path.join(args.output_dir, 'best.pt'),
                    dit_model, ema, optimizer,
                    epoch=epoch, extra={'fid': best_fid},
                )
                print(f"  [最佳模型] FID={best_fid:.2f} 保存")

        # ---- 定期保存 ----
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f'epoch{epoch+1:04d}.pt'),
                dit_model, ema, optimizer, epoch=epoch,
            )

    # ---- 最终保存 ----
    save_checkpoint(
        os.path.join(args.output_dir, 'final.pt'),
        dit_model, ema, optimizer, epoch=args.epochs - 1,
    )
    print(f"\n训练完成! 最佳 FID: {best_fid:.2f}")


if __name__ == '__main__':
    main()
