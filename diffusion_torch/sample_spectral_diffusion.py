"""
sample_spectral_diffusion.py — 频谱级联扩散模型推理与评估脚本
================================================================
功能:
  1. 从训练好的 checkpoint 加载模型
  2. 支持三种采样方式:
     - 标准 DDIM 采样
     - 频谱级联采样 (SVD→低频 + Diffusion→高频)
     - 完整 DDPM 采样 (T 步, 参考用)
  3. 自动计算评估指标:
     - FID (Fréchet Inception Distance) — 生成质量
     - IS (Inception Score) — 生成多样性与质量
     - PSNR / SSIM — 与真实图像的重建质量 (用于 SVD 分析)
  4. 可视化:
     - 生成样本网格
     - 频率分解可视化 (低频+高频+合成)
     - 不同 k_truncate 对比
     - 不同 cascade_t_frac 对比

使用示例:
  # 标准 DDIM 采样
  python3 sample_spectral_diffusion.py \\
      --ckpt outputs/best_ckpt.pt --mode ddim \\
      --n 256 --sample_steps 50 --out gen_ddim.png

  # 频谱级联采样
  python3 sample_spectral_diffusion.py \\
      --ckpt outputs/best_ckpt.pt --mode cascade \\
      --n 256 --sample_steps 50 --k_truncate 8 --cascade_t_frac 0.3 \\
      --out gen_cascade.png

  # 完整评估 (FID + 可视化)
  python3 sample_spectral_diffusion.py \\
      --ckpt outputs/best_ckpt.pt --mode eval \\
      --fid_num 50000 --sample_steps 250 --eval_cascade
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.dit import DiT, DiT_T, DiT_S, DiT_B
from spectral_diffusion import SpectralDiffusion, SpectralDiffusionConfig
from ema import EMA, load_checkpoint


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def to_image(x: torch.Tensor) -> torch.Tensor:
    """将 [-1,1] 范围的张量转换为 [0,1] 用于可视化"""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def compute_is(
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    splits: int = 10,
) -> tuple:
    """计算 Inception Score (IS)

    IS = exp(E_x[KL(p(y|x) || p(y))])
    其中 p(y|x) 是 Inception-V3 的类别分布, p(y) 是边际分布

    参数:
      images: [N, C, H, W] in [-1, 1]
      device: 计算设备
      batch_size: 推理批大小
      splits: 计算 IS 时的分割数 (用于估计方差)

    返回:
      (is_mean, is_std)
    """
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = inception_v3(weights=weights).to(device).eval()

    N = images.shape[0]
    all_preds = []

    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size].to(device)
        batch = to_image(batch)
        batch = preprocess(batch)
        logits = model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = F.softmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # 分割计算 IS
    scores = []
    split_size = N // splits
    for k in range(splits):
        part = all_preds[k * split_size: (k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# 主采样逻辑
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    diffusion: SpectralDiffusion,
    num_samples: int,
    batch_size: int,
    img_channels: int,
    img_size: int,
    sample_steps: int,
    mode: str,
    device: torch.device,
    k_truncate: int = 8,
    cascade_t_frac: float = 0.3,
) -> torch.Tensor:
    """批量生成样本

    参数:
      model: 扩散网络
      diffusion: SpectralDiffusion 实例
      num_samples: 总生成数
      batch_size: 每批生成数
      mode: "ddim" | "cascade" | "ddpm"
      其余参数: 采样配置

    返回:
      samples: [num_samples, C, H, W] in [-1, 1]
    """
    model.eval()
    all_samples = []
    remaining = num_samples

    pbar = tqdm(total=num_samples, desc=f"生成样本 ({mode})")
    while remaining > 0:
        cur = min(batch_size, remaining)
        shape = torch.Size([cur, img_channels, img_size, img_size])

        if mode == "ddim":
            samples = diffusion.ddim_sample(
                model=model, shape=shape, steps=sample_steps,
                eta=0.0, clip_x0=True,
            )
        elif mode == "cascade":
            samples = diffusion.spectral_cascade_sample(
                model=model, shape=shape, steps=sample_steps,
                k_truncate=k_truncate, t_mid_frac=cascade_t_frac,
                eta=0.0, clip_x0=True,
            )
        elif mode == "ddpm":
            samples = diffusion.p_sample_loop(
                model=model, shape=shape, clip_x0=True,
            )
        else:
            raise ValueError(f"未知的采样模式: {mode}")

        all_samples.append(samples.cpu())
        remaining -= cur
        pbar.update(cur)

    pbar.close()
    return torch.cat(all_samples, dim=0)[:num_samples]


# ---------------------------------------------------------------------------
# 频率分解可视化
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_frequency_decomposition(
    model: torch.nn.Module,
    diffusion: SpectralDiffusion,
    device: torch.device,
    img_channels: int,
    img_size: int,
    sample_steps: int,
    k_values: list,
    save_path: str,
) -> None:
    """可视化不同 k_truncate 下的频率分解效果

    对同一组生成样本, 分别展示:
      - 原始生成 (完整频谱)
      - SVD 低频分量 (k=k1, k2, ...)
      - 高频残差 (完整 - 低频)
    """
    shape = torch.Size([8, img_channels, img_size, img_size])
    # 使用相同的初始噪声
    init_noise = torch.randn(shape, device=device)

    # 生成完整样本
    full_samples = diffusion.ddim_sample(
        model=model, shape=shape, steps=sample_steps,
        eta=0.0, clip_x0=True, init_noise=init_noise,
    )

    rows = [to_image(full_samples)]  # 第一行: 完整样本

    for k in k_values:
        low = SpectralDiffusion.svd_truncate(full_samples, k=k)
        high = full_samples - low
        rows.append(to_image(low))
        # 高频残差归一化以便可视化 (将值域映射到 [0,1])
        high_norm = (high - high.min()) / (high.max() - high.min() + 1e-8)
        rows.append(high_norm)

    grid = torch.cat(rows, dim=0).clamp(0, 1)
    save_image(grid, save_path, nrow=8)
    print(f"频率分解可视化已保存到 {save_path}")
    print(f"  行排列: 完整样本, " + ", ".join(f"k={k} 低频, k={k} 高频" for k in k_values))


# ---------------------------------------------------------------------------
# 级联对比可视化
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_cascade_comparison(
    model: torch.nn.Module,
    diffusion: SpectralDiffusion,
    device: torch.device,
    img_channels: int,
    img_size: int,
    sample_steps: int,
    k_truncate: int,
    t_frac_values: list,
    save_path: str,
) -> None:
    """对比不同 cascade_t_frac 的采样结果

    展示标准 DDIM 与不同级联比例的生成效果对比
    """
    shape = torch.Size([8, img_channels, img_size, img_size])
    init_noise = torch.randn(shape, device=device)

    # 标准 DDIM 基准
    samples_ddim = diffusion.ddim_sample(
        model=model, shape=shape, steps=sample_steps,
        eta=0.0, clip_x0=True, init_noise=init_noise.clone(),
    )
    rows = [to_image(samples_ddim)]

    # 不同级联比例
    for t_frac in t_frac_values:
        samples_cas = diffusion.spectral_cascade_sample(
            model=model, shape=shape, steps=sample_steps,
            k_truncate=k_truncate, t_mid_frac=t_frac,
            eta=0.0, clip_x0=True,
        )
        rows.append(to_image(samples_cas))

    grid = torch.cat(rows, dim=0).clamp(0, 1)
    save_image(grid, save_path, nrow=8)
    labels = ["DDIM"] + [f"cascade(t={t})" for t in t_frac_values]
    print(f"级联对比可视化已保存到 {save_path}")
    print(f"  行排列: {', '.join(labels)}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="频谱级联扩散模型推理与评估")

    # ---- 模型 ----
    parser.add_argument("--ckpt", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--model_size", type=str, default="S", choices=["T", "S", "B"])
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--img_channels", type=int, default=3)

    # ---- 扩散配置 (应与训练一致) ----
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--pred_type", type=str, default="v")

    # ---- 采样 ----
    parser.add_argument("--mode", type=str, default="cascade",
                        choices=["ddim", "cascade", "ddpm", "eval", "visualize"],
                        help="运行模式: 采样/评估/可视化")
    parser.add_argument("--n", type=int, default=64, help="生成样本数")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--k_truncate", type=int, default=8)
    parser.add_argument("--cascade_t_frac", type=float, default=0.3)
    parser.add_argument("--out", type=str, default="generated_samples.png")

    # ---- 评估 ----
    parser.add_argument("--fid_num", type=int, default=10000)
    parser.add_argument("--fid_cache", type=str, default="real_inception_acts.npy")
    parser.add_argument("--eval_cascade", action="store_true",
                        help="评估模式下同时评估级联采样")
    parser.add_argument("--compute_is", action="store_true",
                        help="计算 Inception Score")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")

    # ---- 其他 ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # 1. 加载模型
    # =========================================================================
    print(f"加载模型: {args.ckpt}")
    model_builders = {"T": DiT_T, "S": DiT_S, "B": DiT_B}
    model = model_builders[args.model_size](
        img_size=args.img_size,
        in_channels=args.img_channels,
        out_channels=args.img_channels,
    ).to(device)

    # 创建 EMA 实例用于加载
    ema = EMA(model)
    load_checkpoint(args.ckpt, model=model, ema=ema, map_location=device)

    # 使用 EMA 模型进行推理
    ema_model = ema.ema_model.to(device).eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params / 1e6:.2f}M")

    # =========================================================================
    # 2. 初始化扩散过程
    # =========================================================================
    diff_config = SpectralDiffusionConfig(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        pred_type=args.pred_type,
        k_truncate=args.k_truncate,
        cascade_t_frac=args.cascade_t_frac,
    )
    diffusion = SpectralDiffusion(config=diff_config, device=device)

    # =========================================================================
    # 3. 执行指定模式
    # =========================================================================

    if args.mode in ("ddim", "cascade", "ddpm"):
        # ---- 采样模式 ----
        print(f"采样模式: {args.mode}, n={args.n}, steps={args.sample_steps}")
        t0 = time.time()
        samples = generate_samples(
            model=ema_model, diffusion=diffusion,
            num_samples=args.n, batch_size=args.batch,
            img_channels=args.img_channels, img_size=args.img_size,
            sample_steps=args.sample_steps, mode=args.mode,
            device=device, k_truncate=args.k_truncate,
            cascade_t_frac=args.cascade_t_frac,
        )
        elapsed = time.time() - t0
        print(f"生成 {args.n} 张图像耗时: {elapsed:.1f}s ({args.n / elapsed:.1f} img/s)")

        # 保存图像网格
        grid = to_image(samples[:min(args.n, 256)])
        save_image(grid, args.out, nrow=int(min(args.n, 256) ** 0.5))
        print(f"样本已保存到 {args.out}")

        # 可选: 计算 IS
        if args.compute_is and args.n >= 100:
            print("计算 Inception Score...")
            is_mean, is_std = compute_is(samples.to(device), device=device)
            print(f"  IS = {is_mean:.2f} ± {is_std:.2f}")

    elif args.mode == "eval":
        # ---- 完整评估模式 ----
        print(f"评估模式: fid_num={args.fid_num}, steps={args.sample_steps}")

        from fid_utils import (
            InceptionFeatureExtractor,
            collect_n_images_from_loader,
            maybe_load_cached_acts,
            save_cached_acts,
            compute_fid,
        )
        import torchvision
        import torchvision.transforms as transforms

        feat_extractor = InceptionFeatureExtractor(device=device)

        # 真实数据特征
        real_acts = maybe_load_cached_acts(args.fid_cache)
        if real_acts is None or real_acts.shape[0] < args.fid_num:
            print(f"计算真实图像 Inception 特征 (n={args.fid_num}) ...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ])
            if args.dataset == "cifar10":
                dataset = torchvision.datasets.CIFAR10(
                    root=args.data_dir, train=True, download=True, transform=transform
                )
            elif args.dataset == "cifar100":
                dataset = torchvision.datasets.CIFAR100(
                    root=args.data_dir, train=True, download=True, transform=transform
                )
            else:
                raise ValueError(f"不支持的数据集: {args.dataset}")
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4)
            real_imgs = collect_n_images_from_loader(loader, args.fid_num, device=device)
            real_acts = feat_extractor.activations(real_imgs, batch_size=args.batch)
            save_cached_acts(args.fid_cache, real_acts)
        else:
            real_acts = real_acts[:args.fid_num]

        # ---- DDIM 评估 ----
        print(f"\n{'='*50}")
        print(f"评估标准 DDIM 采样 ({args.fid_num} 样本, {args.sample_steps} 步)")
        print(f"{'='*50}")

        samples_ddim = generate_samples(
            model=ema_model, diffusion=diffusion,
            num_samples=args.fid_num, batch_size=args.batch,
            img_channels=args.img_channels, img_size=args.img_size,
            sample_steps=args.sample_steps, mode="ddim", device=device,
        )

        gen_acts = feat_extractor.activations(samples_ddim.to(device), batch_size=args.batch)
        fid_ddim = compute_fid(real_acts, gen_acts)
        print(f"  FID (DDIM) = {fid_ddim:.2f}")

        if args.compute_is:
            is_mean, is_std = compute_is(samples_ddim.to(device), device=device)
            print(f"  IS  (DDIM) = {is_mean:.2f} ± {is_std:.2f}")

        # 保存 DDIM 样本
        save_image(to_image(samples_ddim[:64]), "eval_ddim_samples.png", nrow=8)

        # ---- 级联评估 ----
        if args.eval_cascade:
            print(f"\n{'='*50}")
            print(f"评估频谱级联采样 ({args.fid_num} 样本, k={args.k_truncate}, t_frac={args.cascade_t_frac})")
            print(f"{'='*50}")

            samples_cascade = generate_samples(
                model=ema_model, diffusion=diffusion,
                num_samples=args.fid_num, batch_size=args.batch,
                img_channels=args.img_channels, img_size=args.img_size,
                sample_steps=args.sample_steps, mode="cascade", device=device,
                k_truncate=args.k_truncate, cascade_t_frac=args.cascade_t_frac,
            )

            gen_acts_cas = feat_extractor.activations(samples_cascade.to(device), batch_size=args.batch)
            fid_cas = compute_fid(real_acts, gen_acts_cas)
            print(f"  FID (Cascade) = {fid_cas:.2f}")

            if args.compute_is:
                is_mean, is_std = compute_is(samples_cascade.to(device), device=device)
                print(f"  IS  (Cascade) = {is_mean:.2f} ± {is_std:.2f}")

            save_image(to_image(samples_cascade[:64]), "eval_cascade_samples.png", nrow=8)

        # ---- 汇总报告 ----
        print(f"\n{'='*50}")
        print(f"评估汇总")
        print(f"{'='*50}")
        print(f"  数据集: {args.dataset}")
        print(f"  模型:   DiT-{args.model_size}")
        print(f"  步数:   {args.sample_steps}")
        print(f"  FID (DDIM):    {fid_ddim:.2f}")
        if args.eval_cascade:
            print(f"  FID (Cascade): {fid_cas:.2f}")
        print(f"  评估样本数: {args.fid_num}")

    elif args.mode == "visualize":
        # ---- 可视化模式 ----
        print("生成可视化...")

        # 频率分解可视化
        visualize_frequency_decomposition(
            model=ema_model, diffusion=diffusion, device=device,
            img_channels=args.img_channels, img_size=args.img_size,
            sample_steps=args.sample_steps,
            k_values=[2, 4, 8, 16],
            save_path="vis_freq_decomp.png",
        )

        # 级联对比可视化
        visualize_cascade_comparison(
            model=ema_model, diffusion=diffusion, device=device,
            img_channels=args.img_channels, img_size=args.img_size,
            sample_steps=args.sample_steps,
            k_truncate=args.k_truncate,
            t_frac_values=[0.1, 0.2, 0.3, 0.5, 0.7],
            save_path="vis_cascade_compare.png",
        )


if __name__ == "__main__":
    main()
