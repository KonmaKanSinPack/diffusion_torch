"""
Stage 1: Multi-Scale VQVAE 训练脚本
====================================

训练流程:
  1. 加载数据集 (CIFAR-10/100)
  2. 构建 Multi-Scale VQVAE
  3. 训练 VQVAE (重建损失 + commitment 损失)
  4. 定期保存 checkpoint, 可视化重建质量
  5. 训练完成后冻结 VQVAE, 供 Stage 2 (Latent Diffusion) 使用

训练目标:
  L = L_recon + β * L_commit
  其中:
    L_recon = ||x - x_recon||^2  (像素级重建)
    L_commit = ||z_e - sg(z_q)||^2  (encoder 输出接近 codebook)
    β = commitment_weight (默认 0.25)

关键监控指标:
  - 重建 MSE / PSNR: 衡量重建质量
  - Codebook 使用率: 监控 codebook collapse (理想 > 80%)
  - 各尺度独立重建: 验证频率分解效果

使用方法:
  python train_vqvae.py --dataset cifar10 --epochs 100 --batch_size 128
"""

import argparse
import os
import sys
import time
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_torch.models.vqvae import (
    MultiScaleVQVAE, VQVAEConfig, FrequencyDecomposer,
    VQVAE_Small, VQVAE_Base,
)
from diffusion_torch.ema import EMA, save_checkpoint, load_checkpoint


# ============================================================
#  数据集
# ============================================================

def get_dataset(
    name: str,
    data_dir: str = './data',
    image_size: int = 32,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """
    加载数据集, 图像归一化到 [-1, 1]
    
    Args:
        name: 'cifar10', 'cifar100', 'imagenet'
        data_dir: 数据存放路径
        image_size: 目标图像尺寸
        batch_size: 训练 batch size
    Returns:
        train_loader, val_loader
    """
    # CIFAR 系列原始 32×32, 若 image_size != 32 则 Resize
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
        # ImageNet: data_dir 应含 train/ 和 val/ 子目录
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
        raise ValueError(f"不支持的数据集: {name}, 可选: cifar10, cifar100, imagenet")

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
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 5,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
) -> float:
    """余弦退火学习率, 带线性 warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ============================================================
#  评估
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module, val_loader: DataLoader, device: torch.device,
    image_size: int = 32,
) -> dict:
    """
    在验证集上评估 VQVAE 重建质量
    
    Returns:
        dict: {
            'mse': 平均 MSE,
            'psnr': 平均 PSNR,
            'codebook_usage': List[float] 各尺度利用率,
        }
    """
    model.eval()
    total_mse = 0.0
    total_count = 0

    for images, _ in val_loader:
        images = images.to(device)
        x_recon, _, _, _ = model(images)
        mse = F.mse_loss(x_recon, images, reduction='sum')
        total_mse += mse.item()
        total_count += images.shape[0]

    n_pixels = total_count * 3 * image_size * image_size
    avg_mse = total_mse / n_pixels  # 逐像素平均
    # PSNR: 20 * log10(2 / sqrt(MSE)), 因为值域 [-1,1] 范围为 2
    psnr = 20 * math.log10(2.0 / math.sqrt(avg_mse + 1e-10))

    codebook_usage = model.quantizer.get_codebook_usage()

    return {
        'mse': avg_mse,
        'psnr': psnr,
        'codebook_usage': codebook_usage,
    }


# ============================================================
#  可视化
# ============================================================

@torch.no_grad()
def save_reconstruction_vis(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    n_samples: int = 8,
):
    """保存重建对比图: 原图 | 重建 | 各尺度独立重建"""
    model.eval()

    # 取一小批
    images, _ = next(iter(val_loader))
    images = images[:n_samples].to(device)

    # 重建
    x_recon, z, z_q_scales, _ = model(images)

    # 各尺度独立重建
    scale_recons = []
    for z_q_s in z_q_scales:
        x_s = model.decode(z_q_s)
        scale_recons.append(x_s)

    # 组装网格: 第1行原图, 第2行重建, 后面每行一个尺度
    rows = [images, x_recon] + scale_recons
    all_imgs = torch.cat(rows, dim=0)

    # 反归一化 [-1,1] → [0,1]
    all_imgs = (all_imgs + 1) / 2
    all_imgs = all_imgs.clamp(0, 1)

    nrow = n_samples
    grid = torchvision.utils.make_grid(all_imgs, nrow=nrow, padding=2)
    torchvision.utils.save_image(grid, save_path)
    print(f"  [可视化] 保存到 {save_path}")


# ============================================================
#  训练主循环
# ============================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: EMA,
) -> dict:
    """单个 epoch 训练"""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_commit = 0.0
    n_batches = 0

    for images, _ in train_loader:
        images = images.to(device)
        
        loss, log_dict = model.compute_loss(images)

        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪, 防止训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        ema.update(model)

        total_loss += log_dict['loss_total']
        total_recon += log_dict['loss_recon']
        total_commit += log_dict['loss_commit']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'commit': total_commit / n_batches,
    }


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 1: 训练 Multi-Scale VQVAE')
    
    # 数据
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录 (ImageNet 需含 train/ 和 val/ 子目录)')
    parser.add_argument('--image_size', type=int, default=32,
                        help='输入图像尺寸 (CIFAR=32, ImageNet 推荐 64/128/256)')
    
    # 模型
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='VQVAE 模型规模 (large 适合 ImageNet 64+)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Encoder/Decoder 隐藏通道数 (默认随 model_size)')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='潜在维度 (默认随 model_size)')
    parser.add_argument('--codebook_size', type=int, default=None,
                        help='每个尺度的 codebook 大小 (默认随 model_size)')
    parser.add_argument('--commitment_weight', type=float, default=0.25,
                        help='commitment loss 权重 β')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./checkpoints/vqvae')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='每 N 个 epoch 打印一次日志')
    parser.add_argument('--vis_interval', type=int, default=10,
                        help='每 N 个 epoch 保存可视化')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='每 N 个 epoch 保存 checkpoint')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    
    args = parser.parse_args()

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 数据集 ----
    print(f"加载数据集: {args.dataset} (image_size={args.image_size})")
    train_loader, val_loader = get_dataset(
        args.dataset, args.data_dir, args.image_size, args.batch_size,
    )
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")

    # ---- 模型 ----
    from diffusion_torch.models.vqvae import VQVAE_Large
    vqvae_factory = {'small': VQVAE_Small, 'base': VQVAE_Base, 'large': VQVAE_Large}

    # 收集非 None 的模型参数作为 kwargs 传入工厂函数 (覆盖工厂默认值)
    model_kwargs = {'commitment_weight': args.commitment_weight}
    if args.hidden_dim is not None:
        model_kwargs['hidden_dim'] = args.hidden_dim
    if args.latent_dim is not None:
        model_kwargs['latent_dim'] = args.latent_dim
    if args.codebook_size is not None:
        model_kwargs['codebook_size'] = args.codebook_size

    model = vqvae_factory[args.model_size](
        image_size=args.image_size,
        **model_kwargs,
    )

    model = model.to(device)

    # 参数统计
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: VQVAE-{args.model_size}")
    print(f"  总参数: {n_params:,}")
    print(f"  可训练: {n_trainable:,}")
    print(f"  Latent 空间: D={model.cfg.latent_dim}, 尺寸={model.cfg.latent_size}×{model.cfg.latent_size}")
    print(f"  多尺度: {model.cfg.multi_scales}")
    print(f"  Codebook: {model.cfg.codebook_size} codes/scale × {model.cfg.num_scales} scales")

    # ---- 优化器与 EMA ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99),
    )
    ema = EMA(model, decay=args.ema_decay)

    # ---- 恢复训练 ----
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, ema, optimizer)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"从 epoch {start_epoch} 恢复训练")

    # ---- 输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- 训练循环 ----
    print(f"\n开始训练 (共 {args.epochs} epochs)...")
    best_psnr = 0.0

    for epoch in range(start_epoch, args.epochs):
        # 学习率调度
        lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr, min_lr=args.min_lr,
        )

        # 训练
        t0 = time.time()
        train_log = train_epoch(model, train_loader, optimizer, device, ema)
        t_elapsed = time.time() - t0

        # 日志
        if (epoch + 1) % args.log_interval == 0:
            print(
                f"[Epoch {epoch+1:>4d}/{args.epochs}]  "
                f"loss={train_log['loss']:.4f}  "
                f"recon={train_log['recon']:.4f}  "
                f"commit={train_log['commit']:.4f}  "
                f"lr={lr:.2e}  "
                f"time={t_elapsed:.1f}s"
            )

        # 验证 + 可视化
        if (epoch + 1) % args.vis_interval == 0:
            # 使用 EMA 模型评估
            ema.apply_shadow(model)
            eval_result = evaluate(model, val_loader, device, args.image_size)
            print(
                f"  [验证] MSE={eval_result['mse']:.6f}  "
                f"PSNR={eval_result['psnr']:.2f}dB  "
                f"Codebook使用率={[f'{u:.1%}' for u in eval_result['codebook_usage']]}"
            )
            
            # 可视化
            vis_path = os.path.join(vis_dir, f'recon_epoch{epoch+1:04d}.png')
            save_reconstruction_vis(model, val_loader, device, vis_path)
            
            ema.restore(model)

            # 保存最佳模型
            if eval_result['psnr'] > best_psnr:
                best_psnr = eval_result['psnr']
                save_checkpoint(
                    os.path.join(args.output_dir, 'best.pt'),
                    model, ema, optimizer,
                    epoch=epoch, extra={'psnr': best_psnr},
                )
                print(f"  [最佳模型] PSNR={best_psnr:.2f}dB 保存")

        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f'epoch{epoch+1:04d}.pt'),
                model, ema, optimizer,
                epoch=epoch,
            )

    # ---- 最终保存 ----
    save_checkpoint(
        os.path.join(args.output_dir, 'final.pt'),
        model, ema, optimizer,
        epoch=args.epochs - 1,
    )
    print(f"\n训练完成! 最佳 PSNR: {best_psnr:.2f}dB")
    print(f"Checkpoint 保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
