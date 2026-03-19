"""
train_spectral_diffusion.py — 频谱级联扩散模型完整训练脚本
===========================================================
实现: SVD 负责低频信息 + Diffusion 负责高频信息 的无条件图像生成

核心特性:
  1. DiT (Diffusion Transformer) 骨干 — 2023-2024 主流架构
  2. v-prediction 训练目标 — 在高 SNR 端更稳定
  3. min-SNR-γ 损失加权 — 平衡不同噪声水平的训练信号
  4. SVD 频域分解辅助损失 — 引导模型按频率优先级学习
  5. DDIM + 频谱级联采样 — 推理时显式实现 "SVD→低频, Diffusion→高频"
  6. EMA 权重平滑 — 稳定生成质量
  7. 梯度裁剪 + 学习率 warmup + cosine 衰减
  8. 定期 FID 评估与可视化

支持数据集:
  - CIFAR-10 (32×32, 自动下载)
  - 可扩展至 CelebA-HQ, LSUN, ImageNet 等

使用示例:
  python3 train_spectral_diffusion.py \\
      --epochs 500 --batch 128 --lr 2e-4 \\
      --model_size S --pred_type v \\
      --beta_schedule cosine --timesteps 1000 \\
      --k_truncate 8 --lambda_spectral 0.5 \\
      --min_snr_gamma 5.0 \\
      --sample_steps 50 --save_every 10 \\
      --eval_fid --fid_num 10000 --eval_every 25
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# 确保本目录在 import 路径中
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.dit import DiT, DiT_T, DiT_S, DiT_B
from spectral_diffusion import SpectralDiffusion, SpectralDiffusionConfig
from ema import EMA, save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def grad_clip(params, max_norm: float = 1.0) -> float:
    """梯度范数裁剪, 返回裁剪前的梯度范数"""
    return nn.utils.clip_grad_norm_(params, max_norm=max_norm).item()


def cosine_lr_schedule(optimizer, step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6):
    """余弦退火学习率调度, 带线性 warmup"""
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 数据集准备
# ---------------------------------------------------------------------------

def get_dataset(name: str, data_dir: str, img_size: int = 32):
    """获取数据集和对应的 transform

    目前支持:
      - cifar10: 32×32, 自动下载
      - cifar100: 32×32, 自动下载
    可扩展: CelebA-HQ, LSUN, ImageNet
    """
    name = name.lower()

    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),                      # 数据增强: 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化到 [-1, 1]
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        return trainset, testset, 3, img_size

    elif name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        return trainset, testset, 3, img_size

    else:
        raise ValueError(f"不支持的数据集: {name}. 当前支持: cifar10, cifar100")


# ---------------------------------------------------------------------------
# FID 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fid(
    model: nn.Module,
    diffusion: SpectralDiffusion,
    num_samples: int,
    batch_size: int,
    img_channels: int,
    img_size: int,
    sample_steps: int,
    use_cascade: bool,
    device: torch.device,
    real_acts: np.ndarray,
    feat_extractor,
) -> float:
    """生成样本并计算 FID

    参数:
      model: 扩散网络 (EMA 版本)
      diffusion: SpectralDiffusion 实例
      num_samples: 生成样本数
      batch_size: 每批生成数
      img_channels, img_size: 图像规格
      sample_steps: DDIM 采样步数
      use_cascade: 是否使用频谱级联采样
      device: 计算设备
      real_acts: 预计算的真实图像 Inception 特征
      feat_extractor: InceptionFeatureExtractor 实例

    返回:
      fid: FID 值
    """
    model.eval()
    gen_acts_list = []
    remaining = num_samples

    while remaining > 0:
        cur_batch = min(batch_size, remaining)
        shape = torch.Size([cur_batch, img_channels, img_size, img_size])

        if use_cascade:
            samples = diffusion.spectral_cascade_sample(
                model=model, shape=shape, steps=sample_steps, eta=0.0, clip_x0=True,
            )
        else:
            samples = diffusion.ddim_sample(
                model=model, shape=shape, steps=sample_steps, eta=0.0, clip_x0=True,
            )

        # 计算 Inception 特征
        acts = feat_extractor.activations(samples, batch_size=cur_batch)
        gen_acts_list.append(acts)
        remaining -= cur_batch

    gen_acts = np.concatenate(gen_acts_list, axis=0)[:num_samples]

    from fid_utils import compute_fid
    fid_val = compute_fid(real_acts, gen_acts)
    return fid_val


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="频谱级联扩散模型训练")

    # ---- 数据 ----
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--img_size", type=int, default=32)

    # ---- 模型 ----
    parser.add_argument("--model_size", type=str, default="S", choices=["T", "S", "B"],
                        help="DiT 模型规模: T(iny)=5M, S(mall)=33M, B(ase)=130M 参数")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="DiT Transformer 块内的 dropout 率")

    # ---- 扩散过程 ----
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--pred_type", type=str, default="v", choices=["eps", "v"],
                        help="预测目标: eps=噪声, v=速度 (推荐)")
    parser.add_argument("--min_snr_gamma", type=float, default=5.0,
                        help="min-SNR-γ 损失加权参数, 0=不使用")

    # ---- SVD 频域分解 ----
    parser.add_argument("--k_truncate", type=int, default=8,
                        help="SVD 低频保留奇异值个数 (越小=结构越简洁)")
    parser.add_argument("--lambda_spectral", type=float, default=0.5,
                        help="频域辅助损失权重")
    parser.add_argument("--cascade_t_frac", type=float, default=0.3,
                        help="频谱级联采样的中间时间步比例 (从末端算)")

    # ---- 训练 ----
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=5000,
                        help="学习率线性 warmup 步数")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # ---- 采样与保存 ----
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="DDIM/级联采样步数")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每 N 个 epoch 保存一次 checkpoint")
    parser.add_argument("--ckpt", type=str, default="spectral_diff_ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./outputs")

    # ---- 评估 ----
    parser.add_argument("--eval_fid", action="store_true",
                        help="启用 FID 评估")
    parser.add_argument("--eval_every", type=int, default=25,
                        help="每 N 个 epoch 做一次 FID 评估")
    parser.add_argument("--fid_num", type=int, default=10000,
                        help="FID 评估使用的样本数")
    parser.add_argument("--fid_cache", type=str, default="real_inception_acts.npy",
                        help="真实图像 Inception 特征缓存文件")
    parser.add_argument("--use_cascade", action="store_true",
                        help="评估时使用频谱级联采样 (否则使用标准 DDIM)")

    # ---- 其他 ----
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- 随机种子 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练配置]")
    print(f"  设备: {device}")
    print(f"  数据集: {args.dataset}, 图像尺寸: {args.img_size}")
    print(f"  模型: DiT-{args.model_size}, 预测目标: {args.pred_type}")
    print(f"  SVD 截断 k={args.k_truncate}, 频域损失权重 λ={args.lambda_spectral}")
    print(f"  min-SNR-γ = {args.min_snr_gamma}")

    # ---- 输出目录 ----
    os.makedirs(args.out_dir, exist_ok=True)

    # =========================================================================
    # 1. 数据集
    # =========================================================================
    trainset, testset, img_channels, img_size = get_dataset(
        args.dataset, args.data_dir, args.img_size
    )
    trainloader = DataLoader(
        trainset, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    testloader = DataLoader(
        testset, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # =========================================================================
    # 2. 模型
    # =========================================================================
    # 根据尺寸选择 DiT 配置
    model_builders = {"T": DiT_T, "S": DiT_S, "B": DiT_B}
    model = model_builders[args.model_size](
        img_size=img_size,
        in_channels=img_channels,
        out_channels=img_channels,
        dropout=args.dropout,
    ).to(device)

    num_params = count_parameters(model)
    print(f"  模型参数量: {num_params / 1e6:.2f}M")

    # =========================================================================
    # 3. 优化器 + EMA
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    ema = EMA(model, decay=args.ema_decay)

    # =========================================================================
    # 4. 扩散过程
    # =========================================================================
    diff_config = SpectralDiffusionConfig(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        pred_type=args.pred_type,
        min_snr_gamma=args.min_snr_gamma,
        k_truncate=args.k_truncate,
        lambda_spectral=args.lambda_spectral,
        cascade_t_frac=args.cascade_t_frac,
    )
    diffusion = SpectralDiffusion(config=diff_config, device=device)

    # =========================================================================
    # 5. 断点续训
    # =========================================================================
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.ckpt):
        global_step = load_checkpoint(args.ckpt, model=model, opt=optimizer, ema=ema, map_location=device)
        # 从 global_step 推算 epoch
        steps_per_epoch = len(trainloader)
        start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        print(f"  从 {args.ckpt} 恢复训练, step={global_step}, epoch≈{start_epoch}")

    # =========================================================================
    # 6. FID 评估准备
    # =========================================================================
    feat_extractor = None
    real_acts = None
    if args.eval_fid:
        from fid_utils import (
            InceptionFeatureExtractor,
            collect_n_images_from_loader,
            maybe_load_cached_acts,
            save_cached_acts,
        )
        feat_extractor = InceptionFeatureExtractor(device=device)
        real_acts = maybe_load_cached_acts(args.fid_cache)
        if real_acts is None or real_acts.shape[0] < args.fid_num:
            print(f"  计算真实图像 Inception 特征 (n={args.fid_num}) ...")
            real_imgs = collect_n_images_from_loader(trainloader, args.fid_num, device=device)
            real_acts = feat_extractor.activations(real_imgs, batch_size=args.batch)
            save_cached_acts(args.fid_cache, real_acts)
            print(f"  已缓存到 {args.fid_cache}")
        else:
            real_acts = real_acts[:args.fid_num]
            print(f"  已加载缓存的真实图像特征 ({real_acts.shape[0]} 个)")

    # =========================================================================
    # 7. 训练总步数 (用于 LR schedule)
    # =========================================================================
    total_steps = args.epochs * len(trainloader)
    print(f"  总训练步数: {total_steps}")
    print(f"  学习率: {args.lr}, warmup: {args.warmup_steps} 步, cosine 衰减")
    print("=" * 60)

    # =========================================================================
    # 8. 训练循环
    # =========================================================================
    best_fid = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_main = 0.0
        epoch_loss_low = 0.0
        epoch_loss_high = 0.0
        num_batches = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            # 学习率调度
            lr = cosine_lr_schedule(
                optimizer, global_step, total_steps,
                args.warmup_steps, args.lr
            )

            # 随机采样时间步
            t = torch.randint(0, args.timesteps, (batch_size,), device=device, dtype=torch.long)

            # 计算损失 (含频域分解)
            loss, log = diffusion.training_loss(model=model, x_start=images, t=t)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = grad_clip(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # EMA 更新
            ema.update(model)

            # 记录
            global_step += 1
            epoch_loss += log["loss_total"]
            epoch_loss_main += log["loss_main"]
            epoch_loss_low += log["loss_low"]
            epoch_loss_high += log["loss_high"]
            num_batches += 1

            pbar.set_postfix(
                loss=f'{log["loss_total"]:.4f}',
                main=f'{log["loss_main"]:.4f}',
                low=f'{log["loss_low"]:.4f}',
                high=f'{log["loss_high"]:.4f}',
                lr=f'{lr:.2e}',
                gnorm=f'{gnorm:.2f}',
            )

        # ---- Epoch 统计 ----
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_main = epoch_loss_main / max(num_batches, 1)
        avg_low = epoch_loss_low / max(num_batches, 1)
        avg_high = epoch_loss_high / max(num_batches, 1)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}  main={avg_main:.4f}  "
              f"low={avg_low:.4f}  high={avg_high:.4f}  step={global_step}")

        # ---- 保存 checkpoint + 可视化 ----
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, args.ckpt)
            save_checkpoint(ckpt_path, model=model, opt=optimizer, ema=ema, step=global_step)

            # 用 EMA 模型生成可视化样本
            ema_model = ema.ema_model.to(device).eval()
            sample_shape = torch.Size([64, img_channels, img_size, img_size])

            with torch.no_grad():
                # 标准 DDIM 采样
                samples_ddim = diffusion.ddim_sample(
                    model=ema_model, shape=sample_shape,
                    steps=args.sample_steps, eta=0.0, clip_x0=True,
                )
                # 频谱级联采样
                samples_cascade = diffusion.spectral_cascade_sample(
                    model=ema_model, shape=sample_shape,
                    steps=args.sample_steps, eta=0.0, clip_x0=True,
                )
                # 低频分量可视化 (SVD 截断)
                samples_low = SpectralDiffusion.svd_truncate(
                    samples_cascade, k=args.k_truncate
                )

            # 保存:  上排=DDIM, 中排=级联, 下排=级联的低频分量
            ddim_grid = (samples_ddim[:16] + 1.0) / 2.0
            cascade_grid = (samples_cascade[:16] + 1.0) / 2.0
            low_grid = (samples_low[:16] + 1.0) / 2.0

            vis = torch.cat([ddim_grid, cascade_grid, low_grid], dim=0).clamp(0, 1)
            save_image(vis, os.path.join(args.out_dir, f"samples_epoch{epoch}.png"), nrow=16)
            print(f"  → 已保存 checkpoint 和可视化到 {args.out_dir}/")

        # ---- FID 评估 ----
        if (
            args.eval_fid
            and feat_extractor is not None
            and real_acts is not None
            and (epoch + 1) % args.eval_every == 0
        ):
            ema_model = ema.ema_model.to(device).eval()

            # 标准 DDIM FID
            fid_ddim = evaluate_fid(
                model=ema_model, diffusion=diffusion,
                num_samples=args.fid_num, batch_size=args.batch,
                img_channels=img_channels, img_size=img_size,
                sample_steps=args.sample_steps, use_cascade=False,
                device=device, real_acts=real_acts,
                feat_extractor=feat_extractor,
            )
            print(f"  [FID-DDIM]    epoch={epoch}  FID={fid_ddim:.2f}  (n={args.fid_num})")

            # 频谱级联 FID
            fid_cascade = evaluate_fid(
                model=ema_model, diffusion=diffusion,
                num_samples=args.fid_num, batch_size=args.batch,
                img_channels=img_channels, img_size=img_size,
                sample_steps=args.sample_steps, use_cascade=True,
                device=device, real_acts=real_acts,
                feat_extractor=feat_extractor,
            )
            print(f"  [FID-Cascade] epoch={epoch}  FID={fid_cascade:.2f}  (n={args.fid_num})")

            # 保存最佳模型
            min_fid = min(fid_ddim, fid_cascade)
            if min_fid < best_fid:
                best_fid = min_fid
                best_path = os.path.join(args.out_dir, "best_ckpt.pt")
                save_checkpoint(best_path, model=model, opt=optimizer, ema=ema, step=global_step)
                print(f"  → 新的最佳 FID={best_fid:.2f}, 已保存到 {best_path}")

    # ---- 训练结束 ----
    print("=" * 60)
    print(f"训练完成! 最佳 FID = {best_fid:.2f}")
    final_path = os.path.join(args.out_dir, "final_ckpt.pt")
    save_checkpoint(final_path, model=model, opt=optimizer, ema=ema, step=global_step)
    print(f"最终 checkpoint 已保存到 {final_path}")


if __name__ == "__main__":
    main()
