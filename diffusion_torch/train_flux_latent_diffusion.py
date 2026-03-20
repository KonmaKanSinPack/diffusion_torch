"""
FLUX.1-dev VAE + DiT Latent Diffusion 训练脚本
================================================

使用 black-forest-labs/FLUX.1-dev 的预训练 VAE 替代自训练 VQVAE,
直接在 FLUX VAE 的潜在空间上训练扩散模型。

FLUX VAE 参数:
  - latent_channels: 16
  - 下采样倍率: 8x (256x256 → 32x32)
  - scaling_factor: 0.3611
  - shift_factor: 0.1159

训练流程:
  1. 加载冻结的 FLUX VAE
  2. 预计算所有训练/验证集的 latent (避免重复编码)
  3. 构建 DiT 模型 (在 FLUX latent 空间操作)
  4. 训练循环:
     a. 从缓存加载 latent z_0
     b. 随机加噪 z_0 → z_t
     c. DiT 预测去噪
     d. min-SNR-γ 加权损失
  5. 定期生成样本 → FLUX VAE decoder → 图像 → FID 评估

使用方法:
  python train_flux_latent_diffusion.py \\
      --dataset cifar10 \\
      --image_size 256 \\
      --dit_size B \\
      --epochs 500 \\
      --batch_size 128 \\
      --lr 1e-4
"""

import argparse
import os
import sys
import time
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_torch.models.dit import DiT, LatentDiT_T, LatentDiT_S, LatentDiT_B
from diffusion_torch.latent_diffusion import LatentSpectralDiffusion, LatentDiffusionConfig
from diffusion_torch.ema import EMA, save_checkpoint, load_checkpoint


# ============================================================
#  FLUX VAE 封装
# ============================================================

class FluxVAEWrapper:
    """
    FLUX.1-dev VAE 封装类

    处理 FLUX VAE 的 scaling/shift 以及 encode/decode 接口。
    训练时 latent 空间:  z = (raw_latent - shift_factor) * scaling_factor
    解码时需要 unscale: raw_latent = z / scaling_factor + shift_factor
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            subfolder='vae',
            torch_dtype=dtype,
        ).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.scaling_factor = self.vae.config.scaling_factor  # 0.3611
        self.shift_factor = self.vae.config.shift_factor      # 0.1159
        self.latent_channels = self.vae.config.latent_channels  # 16
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 scaled latent 空间。

        Args:
            images: (B, 3, H, W) 范围 [-1, 1]
        Returns:
            z: (B, 16, H/8, W/8) scaled latent
        """
        raw_latent = self.vae.encode(images.to(self.dtype)).latent_dist.sample()
        z = (raw_latent - self.shift_factor) * self.scaling_factor
        return z.float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从 scaled latent 解码到图像。

        Args:
            z: (B, 16, H_lat, W_lat) scaled latent
        Returns:
            images: (B, 3, H, W) 范围约 [-1, 1]
        """
        raw_latent = z.to(self.dtype) / self.scaling_factor + self.shift_factor
        images = self.vae.decode(raw_latent).sample
        return images.float()


# ============================================================
#  数据集 & 预计算 Latent
# ============================================================

class HFImageDataset(torch.utils.data.Dataset):
    """将 HuggingFace Dataset 包装为 PyTorch Dataset"""

    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataset(
    name: str,
    data_dir: str = './data',
    image_size: int = 256,
):
    """加载数据集, 返回 [-1, 1] 范围的图像"""
    transform_train = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    transform_val = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

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
        from datasets import load_dataset
        print("  从 HuggingFace 加载 ImageNet-1k 256×256...")
        hf_train = load_dataset(
            'evanarlian/imagenet_1k_resized_256',
            split='train',
            cache_dir=os.path.join(data_dir, 'imagenet_hf_cache'),
        )
        hf_val = load_dataset(
            'evanarlian/imagenet_1k_resized_256',
            split='val',
            cache_dir=os.path.join(data_dir, 'imagenet_hf_cache'),
        )
        train_set = HFImageDataset(hf_train, transform_train)
        val_set = HFImageDataset(hf_val, transform_val)
    elif name == 'imagefolder':
        train_set = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform=transform_train)
        val_set = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'val'), transform=transform_val)
    else:
        raise ValueError(f"不支持的数据集: {name}，支持: cifar10, cifar100, imagenet, imagefolder")

    return train_set, val_set


@torch.no_grad()
def precompute_latents(
    flux_vae: FluxVAEWrapper,
    dataset,
    batch_size: int = 64,
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """
    预计算整个数据集的 FLUX VAE latent，缓存到磁盘。

    Returns:
        all_latents: (N, 16, H_lat, W_lat) float32 tensor
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  从缓存加载 latent: {cache_path}")
        return torch.load(cache_path, map_location='cpu', weights_only=True)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    all_latents = []
    print(f"  预计算 latent ({len(dataset)} 样本)...")
    t0 = time.time()

    for i, (images, _) in enumerate(loader):
        images = images.to(flux_vae.device)
        z = flux_vae.encode(images)
        all_latents.append(z.cpu())
        if (i + 1) % 50 == 0:
            print(f"    {(i+1)*batch_size}/{len(dataset)}")

    all_latents = torch.cat(all_latents, dim=0)
    elapsed = time.time() - t0
    print(f"  预计算完成: {all_latents.shape}, 耗时 {elapsed:.1f}s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        torch.save(all_latents, cache_path)
        print(f"  缓存已保存: {cache_path}")

    return all_latents


def normalize_latents(latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对预计算的 latent 做逐通道归一化，使每个通道 ~ N(0,1)。

    这对扩散模型至关重要：扩散过程假设数据分布接近标准正态分布。

    Returns:
        normalized: (N, C, H, W) 归一化 latent
        channel_mean: (C,) 每通道均值
        channel_std: (C,) 每通道标准差
    """
    # 计算每通道统计量 (在 N, H, W 维度上)
    channel_mean = latents.mean(dim=(0, 2, 3))   # (C,)
    channel_std = latents.std(dim=(0, 2, 3))      # (C,)
    channel_std = channel_std.clamp(min=1e-6)

    # 归一化
    normalized = (latents - channel_mean[None, :, None, None]) / channel_std[None, :, None, None]
    print(f"  [归一化] 归一化前: mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"  [归一化] 归一化后: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    return normalized, channel_mean, channel_std


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
    flux_vae: FluxVAEWrapper,
    val_images: torch.Tensor,
    device: torch.device,
    latent_size: int = 32,
    latent_dim: int = 16,
    n_gen: int = 2048,
    ddim_steps: int = 50,
    channel_mean: Optional[torch.Tensor] = None,
    channel_std: Optional[torch.Tensor] = None,
    fid_resolution: int = 256,
) -> float:
    """
    生成样本并计算 FID

    流程:
    1. DDIM 采样 → 生成归一化 latent
    2. 反归一化 → FLUX VAE decode → 图像
    3. 统一到 fid_resolution, 送入 Inception-V3 提取特征
    4. 比较生成 vs 真实的 FID

    val_images 可以是任意分辨率, 会被 InceptionFeatureExtractor 内部处理
    """
    from diffusion_torch.fid_utils import InceptionFeatureExtractor
    import classifier_metrics_numpy

    dit_model.eval()
    batch_gen = min(32, n_gen)

    generated = []
    n_remaining = n_gen

    while n_remaining > 0:
        bs = min(batch_gen, n_remaining)
        shape = (bs, latent_dim, latent_size, latent_size)

        z_norm = diffusion.ddim_sample(dit_model, shape, num_steps=ddim_steps)

        # 反归一化
        if channel_mean is not None and channel_std is not None:
            cm = channel_mean.to(device)
            cs = channel_std.to(device)
            z = z_norm * cs[None, :, None, None] + cm[None, :, None, None]
        else:
            z = z_norm

        images = flux_vae.decode(z)
        # 统一分辨率用于 FID
        if images.shape[-1] != fid_resolution:
            images = F.interpolate(images, size=fid_resolution, mode='bilinear', align_corners=False)
        images = images.clamp(-1, 1)
        generated.append(images.cpu())
        n_remaining -= bs

    generated = torch.cat(generated, dim=0)[:n_gen]

    extractor = InceptionFeatureExtractor(device)
    real_for_fid = val_images[:n_gen]
    # 统一真实图像分辨率
    if real_for_fid.shape[-1] != fid_resolution:
        real_for_fid = F.interpolate(real_for_fid, size=fid_resolution, mode='bilinear', align_corners=False)
    gen_acts = extractor.activations(generated, batch_size=64)
    real_acts = extractor.activations(real_for_fid, batch_size=64)
    fid = float(classifier_metrics_numpy.frechet_classifier_distance_from_activations(
        real_acts, gen_acts))

    return fid


# ============================================================
#  可视化
# ============================================================

@torch.no_grad()
def save_samples_vis(
    dit_model: nn.Module,
    diffusion: LatentSpectralDiffusion,
    flux_vae: FluxVAEWrapper,
    device: torch.device,
    save_path: str,
    latent_size: int = 32,
    latent_dim: int = 16,
    n_samples: int = 64,
    ddim_steps: int = 50,
    channel_mean: Optional[torch.Tensor] = None,
    channel_std: Optional[torch.Tensor] = None,
):
    """生成样本并保存可视化"""
    dit_model.eval()
    shape = (n_samples, latent_dim, latent_size, latent_size)
    z_norm = diffusion.ddim_sample(dit_model, shape, num_steps=ddim_steps)

    # 反归一化
    if channel_mean is not None and channel_std is not None:
        cm = channel_mean.to(device)
        cs = channel_std.to(device)
        z = z_norm * cs[None, :, None, None] + cm[None, :, None, None]
    else:
        z = z_norm

    images = flux_vae.decode(z)
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    torchvision.utils.save_image(grid, save_path)
    print(f"  [可视化] 样本保存到 {save_path}")


# ============================================================
#  训练
# ============================================================

def train_epoch(
    dit_model: nn.Module,
    diffusion: LatentSpectralDiffusion,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: EMA,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> dict:
    """单个 epoch 训练 (支持 AMP 混合精度)"""
    dit_model.train()
    total_loss = 0.0
    total_main = 0.0
    total_svd = 0.0
    n_batches = 0

    optimizer.zero_grad()
    for step, (z_0,) in enumerate(train_loader):
        z_0 = z_0.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            loss, log_dict = diffusion.training_loss(dit_model, z_0, context=None)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            ema.update(dit_model)

        total_loss += log_dict['loss_total']
        total_main += log_dict['loss_main']
        total_svd += log_dict.get('loss_svd_low', 0.0) + log_dict.get('loss_svd_high', 0.0)
        n_batches += 1

    if n_batches % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
        ema.update(dit_model)

    return {
        'loss': total_loss / max(n_batches, 1),
        'main': total_main / max(n_batches, 1),
        'svd': total_svd / max(n_batches, 1),
    }


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLUX VAE + DiT Latent Diffusion 训练'
    )

    # 数据
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'imagefolder'],
                        help='数据集: cifar10/cifar100/imagenet/imagefolder')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸 (FLUX VAE 最佳 256)')

    # DiT 模型
    parser.add_argument('--dit_size', type=str, default='B',
                        choices=['T', 'S', 'B'],
                        help='DiT 模型规模 (推荐 B)')
    parser.add_argument('--patch_size', type=int, default=2)

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'])
    parser.add_argument('--pred_type', type=str, default='v',
                        choices=['eps', 'v'])
    parser.add_argument('--min_snr_gamma', type=float, default=5.0)
    parser.add_argument('--svd_aux_weight', type=float, default=0.0,
                        help='SVD 辅助损失权重 (默认关闭)')

    # 训练
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='限制训练样本数 (0=全部, 用于快速实验)')

    # 评估
    parser.add_argument('--fid_interval', type=int, default=25,
                        help='每 N 个 epoch 计算 FID')
    parser.add_argument('--fid_n_samples', type=int, default=2048)
    parser.add_argument('--ddim_steps', type=int, default=50)

    # 输出
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints/flux_latent_diffusion')
    parser.add_argument('--vis_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=25)

    # 恢复
    parser.add_argument('--resume', type=str, default=None)

    # 缓存
    parser.add_argument('--latent_cache_dir', type=str, default='./latent_cache')

    # 性能
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='使用 bf16 混合精度加速')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='使用 torch.compile 加速')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- FLUX VAE ----
    print("\n加载 FLUX.1-dev VAE...")
    flux_vae = FluxVAEWrapper(device, dtype=torch.float32)
    latent_dim = flux_vae.latent_channels  # 16
    latent_size = args.image_size // 8      # 256/8 = 32
    print(f"  latent_dim={latent_dim}, latent_size={latent_size}")
    print(f"  scaling_factor={flux_vae.scaling_factor}, shift_factor={flux_vae.shift_factor}")

    # ---- 数据集 ----
    print(f"\n加载数据集: {args.dataset}")
    train_set, val_set = get_dataset(args.dataset, args.data_dir, args.image_size)
    print(f"  训练集: {len(train_set)} 样本, 图像尺寸: {args.image_size}x{args.image_size}")

    # 限制训练样本数 (用于快速实验)
    if args.max_train_samples > 0 and args.max_train_samples < len(train_set):
        train_set = torch.utils.data.Subset(train_set, range(args.max_train_samples))
        print(f"  [限制] 使用前 {args.max_train_samples} 个训练样本")

    # ---- 预计算 latent ----
    os.makedirs(args.latent_cache_dir, exist_ok=True)
    n_samples = len(train_set)
    suffix = f"_{n_samples}" if args.max_train_samples > 0 else ""
    train_cache = os.path.join(
        args.latent_cache_dir, f'{args.dataset}_train_{args.image_size}{suffix}.pt')

    print("\n预计算训练集 latent...")
    train_latents = precompute_latents(
        flux_vae, train_set, batch_size=64, cache_path=train_cache)

    # ---- 逐通道归一化 latent ----
    print("\n归一化 latent...")
    train_latents, channel_mean, channel_std = normalize_latents(train_latents)

    # 收集验证集图像 (for FID)
    # FID 分辨率: ImageNet 用 256, CIFAR 用 32 (原始分辨率)
    if args.dataset in ('imagenet', 'imagefolder'):
        fid_resolution = 256
    else:
        fid_resolution = 32

    print(f"\n加载验证集图像 (for FID, resolution={fid_resolution})...")
    fid_transform = T.Compose([
        T.Resize((fid_resolution, fid_resolution), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    if args.dataset == 'cifar10':
        fid_val_set = torchvision.datasets.CIFAR10(
            args.data_dir, train=False, download=True, transform=fid_transform)
    elif args.dataset == 'cifar100':
        fid_val_set = torchvision.datasets.CIFAR100(
            args.data_dir, train=False, download=True, transform=fid_transform)
    else:
        # imagenet / imagefolder: 复用 val_set 但用 fid_transform
        _, fid_val_set = get_dataset(args.dataset, args.data_dir, fid_resolution)

    # 取前 n_fid 张 (避免加载全部 50k)
    n_fid = min(args.fid_n_samples * 2, len(fid_val_set))
    val_images_fid = torch.stack([fid_val_set[i][0] for i in range(n_fid)])
    print(f"  验证集: {val_images_fid.shape} (resolution={fid_resolution})")

    # 创建 DataLoader (从缓存的 latent)
    train_ds = TensorDataset(train_latents)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # ---- Diffusion ----
    diff_cfg = LatentDiffusionConfig(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        pred_type=args.pred_type,
        min_snr_gamma=args.min_snr_gamma,
        svd_aux_weight=args.svd_aux_weight,
        use_svd_aux=(args.svd_aux_weight > 0),
        use_multiscale_cond=False,
        cond_scales=[],
        latent_dim=latent_dim,
        latent_size=latent_size,
    )
    diffusion = LatentSpectralDiffusion(diff_cfg).to(device)

    # ---- DiT 模型 ----
    # 不使用 cross-attention (FLUX VAE 无多尺度量化)
    dit_factory = {'T': LatentDiT_T, 'S': LatentDiT_S, 'B': LatentDiT_B}
    dit_model = dit_factory[args.dit_size](
        latent_size=latent_size,
        latent_dim=latent_dim,
        context_dim=0,
        use_cross_attn=False,
    ).to(device)

    n_params = sum(p.numel() for p in dit_model.parameters())
    print(f"\nDiT-{args.dit_size} (FLUX Latent)")
    print(f"  参数量: {n_params:,}")
    print(f"  输入: ({latent_dim}, {latent_size}, {latent_size})")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Tokens: {(latent_size // args.patch_size) ** 2}")

    # ---- 优化器与 EMA ----
    optimizer = torch.optim.AdamW(
        dit_model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99),
    )
    ema = EMA(dit_model, decay=args.ema_decay)

    # ---- AMP & Compile ----
    scaler = None
    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print(f"  [AMP] bf16 混合精度已启用")
    if args.compile:
        dit_model = torch.compile(dit_model)
        print(f"  [Compile] torch.compile 已启用")

    # ---- 恢复训练 ----
    start_epoch = 0
    best_fid = float('inf')
    if args.resume:
        ckpt = load_checkpoint(args.resume, dit_model, ema, optimizer)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_fid = ckpt.get('fid', float('inf'))
        print(f"从 epoch {start_epoch} 恢复训练, 当前最佳 FID: {best_fid:.2f}")

    # ---- 输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- 释放 VAE 显存 (训练时不需要 encoder) ----
    # 保留 VAE 用于 decode (生成可视化)
    # flux_vae 已在 GPU, 作为冻结模型占用固定显存

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"开始训练: FLUX VAE + DiT-{args.dit_size} Latent Diffusion")
    print(f"  数据集: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} × {args.grad_accum} grad_accum")
    print(f"  LR: {args.lr} → {args.min_lr}")
    print(f"  Latent: {latent_dim}×{latent_size}×{latent_size}")
    print(f"  Pred: {args.pred_type}, Schedule: {args.beta_schedule}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr, min_lr=args.min_lr,
        )

        t0 = time.time()
        log = train_epoch(
            dit_model, diffusion, train_loader, optimizer,
            device, ema, grad_accum_steps=args.grad_accum,
            scaler=scaler, use_amp=args.use_amp,
        )
        t_elapsed = time.time() - t0

        print(
            f"[Epoch {epoch+1:>4d}/{args.epochs}]  "
            f"loss={log['loss']:.4f}  main={log['main']:.4f}  "
            f"lr={lr:.2e}  time={t_elapsed:.1f}s"
        )

        # ---- 可视化 ----
        if (epoch + 1) % args.vis_interval == 0:
            ema.apply_shadow(dit_model)
            vis_path = os.path.join(vis_dir, f'samples_epoch{epoch+1:04d}.png')
            save_samples_vis(
                dit_model, diffusion, flux_vae, device, vis_path,
                latent_size=latent_size, latent_dim=latent_dim,
                n_samples=64, ddim_steps=args.ddim_steps,
                channel_mean=channel_mean, channel_std=channel_std,
            )
            ema.restore(dit_model)

        # ---- FID 评估 ----
        if (epoch + 1) % args.fid_interval == 0:
            ema.apply_shadow(dit_model)
            fid = evaluate_fid(
                dit_model, diffusion, flux_vae, val_images_fid, device,
                latent_size=latent_size, latent_dim=latent_dim,
                n_gen=args.fid_n_samples, ddim_steps=args.ddim_steps,
                channel_mean=channel_mean, channel_std=channel_std,
                fid_resolution=fid_resolution,
            )
            print(f"  [FID] {fid:.2f}")
            ema.restore(dit_model)

            if fid < best_fid:
                best_fid = fid
                save_checkpoint(
                    os.path.join(args.output_dir, 'best.pt'),
                    dit_model, ema, optimizer,
                    epoch=epoch, extra={
                        'fid': best_fid,
                        'channel_mean': channel_mean,
                        'channel_std': channel_std,
                    },
                )
                print(f"  [最佳模型] FID={best_fid:.2f} 已保存")

        # ---- 定期保存 ----
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f'epoch{epoch+1:04d}.pt'),
                dit_model, ema, optimizer,
                epoch=epoch, extra={
                    'fid': best_fid,
                    'channel_mean': channel_mean,
                    'channel_std': channel_std,
                },
            )

    # ---- 最终保存 ----
    save_checkpoint(
        os.path.join(args.output_dir, 'final.pt'),
        dit_model, ema, optimizer,
        epoch=args.epochs - 1, extra={
            'fid': best_fid,
            'channel_mean': channel_mean,
            'channel_std': channel_std,
        },
    )
    print(f"\n训练完成! 最佳 FID: {best_fid:.2f}")


if __name__ == '__main__':
    main()
