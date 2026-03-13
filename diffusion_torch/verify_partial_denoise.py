"""verify_partial_denoise.py
验证 Stage-1 (I_L 生成) 中部分去噪 vs. 完全去噪的效果差异。

实验设计：
  从 *相同* 初始高斯噪声出发，分别以不同的 t_stop 运行 DDIM，
  得到一组成对的 I_L；共享该 I_L 经过 Stage-2 (b-diffusion) 重建出 I_out。
  最终计算各组与完全去噪(ratio=1.0)之间的 PSNR / SSIM，并保存对比网格。

用法示例：
  python3 verify_partial_denoise.py \\
      --il_ckpt il_ddpm_ckpt.pt \\
      --b_ckpt  b_ddpm_ckpt.pt \\
      --n 16 \\
      --sample_steps 50 \\
      --il_denoise_ratios 1.0,0.75,0.5,0.25 \\
      --out verify_partial.jpg

输出：
  - verify_partial.jpg       : 对比网格
  - verify_partial_il.jpg    : 仅 I_L 的对比网格
  - 终端打印每个 ratio vs. ratio=1.0 的 PSNR / SSIM
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.unet import UNet
from image_diffusion import ImageDDPM, DDPMParams as ImgDDPMParams
from b_diffusion import ConditionalBDDPM, DDPMParams as BDDPMParams
from ema import EMA, load_checkpoint


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def reconstruct(I_L: torch.Tensor, b_sample: torch.Tensor) -> torch.Tensor:
    """I_out = I_L + U_L @ b_sample"""
    # 输入:
    #   I_L      : [B, 3, H, W]，低频/低秩图
    #   b_sample : [B, 1, r, W]，在 U_L 基上的高频系数
    # 输出:
    #   I_out    : [B, 3, H, W]，重建结果
    b, c, h, w = I_L.shape
    I_L_flat = I_L.reshape(b, c * h, w)
    U_l, S_l, _ = torch.linalg.svd(I_L_flat)
    r_l = S_l.shape[1]
    b_flat = b_sample.squeeze(1)          # [B, r, W]
    out_flat = I_L_flat + U_l[:, :, :r_l] @ b_flat
    return out_flat.reshape(b, c, h, w)


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> float:
    """pixel range assumed [-1, 1] => data_range=2."""
    # 该函数当前未在主流程中使用，保留作参考。
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return float("inf")
    return 10.0 * (2.0 * (data_range ** 2) / mse).__class__(
        torch.tensor(data_range ** 2 / mse).log10().item() * 10
    )


def _psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> float:
    """PSNR (dB), images in [-1,1]."""
    # 实际用于主流程的 PSNR 计算函数。
    import math
    mse = F.mse_loss(a.float(), b.float()).item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse)


def _ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11) -> float:
    """Simplified mean-SSIM (luminance + contrast + structure).
    Both tensors: [B, C, H, W] in [-1, 1].
    Returns scalar float.
    """
    a = a.float()
    b = b.float()
    C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2  # data_range=2

    # 为每个通道构造高斯窗口。
    channels = a.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, device=a.device, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
    kernel = kernel.expand(channels, 1, window_size, window_size)
    pad = window_size // 2

    def conv(x):
        # 分组卷积：每个通道独立做局部统计。
        return F.conv2d(x, kernel, padding=pad, groups=channels)

    mu_a, mu_b = conv(a), conv(b)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sigma_a2 = conv(a * a) - mu_a2
    sigma_b2 = conv(b * b) - mu_b2
    sigma_ab = conv(a * b) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / (
        (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    # -----------------------------
    # 1) 参数与实验设置
    # -----------------------------
    p = argparse.ArgumentParser(description="Verify partial vs. full denoising in Stage-1")
    p.add_argument("--il_ckpt", type=str, required=True, help="Stage-1 checkpoint")
    p.add_argument("--b_ckpt", type=str, required=True, help="Stage-2 checkpoint")
    p.add_argument("--n", type=int, default=16, help="Number of sample pairs")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--sample_steps", type=int, default=50,
                   help="DDIM steps used for each denoising (same for all ratios)")
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--il_denoise_ratios", type=str, default="1.0,0.75,0.5,0.25",
                   help="Comma-separated list of Stage-1 denoising ratios (1.0=full)")
    p.add_argument("--skip_b", action="store_true",
                   help="Skip Stage-2; only compare I_L quality")
    p.add_argument("--out", type=str, default="verify_partial.jpg")
    p.add_argument("--nrow", type=int, default=8, help="Images per row in grid")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ratios = [float(r.strip()) for r in args.il_denoise_ratios.split(",")]
    ratios = sorted(set(ratios), reverse=True)  # 1.0 first (reference)
    T = args.timesteps

    # 将“去噪比例 ratio”映射到“停止时刻 t_stop”：
    # ratio=1.0 -> t_stop=0（完全去噪）
    # ratio=0.5 -> t_stop≈T/2（只去噪一半）
    t_stops = {r: max(0, int(round(T * (1.0 - r)))) for r in ratios}
    print(f"Ratios → t_stop mapping: { {r: t_stops[r] for r in ratios} }")

    # ------------------------------------------------------------------
    # 加载 Stage-1：无条件 I_L 生成模型
    # ------------------------------------------------------------------
    il_model = UNet(t_emb_dim=128, ch=128, out_ch=3, in_ch=3).to(device)
    il_opt = torch.optim.AdamW(il_model.parameters(), lr=1e-4)
    il_ema = EMA(il_model)
    load_checkpoint(args.il_ckpt, model=il_model, opt=il_opt, ema=il_ema, map_location=device)
    il_model = il_ema.ema_model.to(device).eval()

    il_ddpm = ImageDDPM(
        params=ImgDDPMParams(timesteps=T, beta_schedule=args.beta_schedule),
        device=device, dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # 加载 Stage-2（可选）：条件 b 生成模型
    # ------------------------------------------------------------------
    b_model = None
    b_ddpm = None
    if not args.skip_b:
        b_model = UNet(t_emb_dim=128, ch=64, out_ch=1, in_ch=4).to(device)
        b_opt = torch.optim.AdamW(b_model.parameters(), lr=1e-4)
        b_ema = EMA(b_model)
        load_checkpoint(args.b_ckpt, model=b_model, opt=b_opt, ema=b_ema, map_location=device)
        b_model = b_ema.ema_model.to(device).eval()
        b_ddpm = ConditionalBDDPM(
            params=BDDPMParams(timesteps=T, beta_schedule=args.beta_schedule),
            device=device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # 2) 以“同一份初始噪声”对不同 ratio 做配对对比
    # ------------------------------------------------------------------
    H, W = 32, 32
    shape = torch.Size([args.n, 3, H, W])
    init_noise = torch.randn(shape, device=device, dtype=torch.float32)

    il_results: dict[float, torch.Tensor] = {}   # ratio → I_L  [N,3,H,W]
    out_results: dict[float, torch.Tensor] = {}  # ratio → I_out [N,3,H,W]

    for ratio in ratios:
        t_stop = t_stops[ratio]
        print(f"\n=== ratio={ratio:.2f}  t_stop={t_stop} ===")

        # Stage-1：从同一个 init_noise 出发，跑到对应 t_stop
        I_L = il_ddpm.ddim_sample_loop(
            model=il_model,
            shape=shape,
            steps=args.sample_steps,
            eta=args.ddim_eta,
            clip_x0=True,
            t_stop=t_stop,
            init_noise=init_noise.clone(),
        )
        il_results[ratio] = I_L.cpu()
        print(f"  I_L range: [{I_L.min():.3f}, {I_L.max():.3f}]")

        if b_model is not None:
            # Stage-2：给定 I_L，采样 b 并重建 I_out
            I_L_flat = I_L.reshape(args.n, 3 * H, W)
            _, S_l, _ = torch.linalg.svd(I_L_flat)
            r_l = S_l.shape[1]

            b_sample = b_ddpm.ddim_sample_loop(
                model=b_model,
                shape=torch.Size([args.n, 1, r_l, W]),
                cond=I_L,
                steps=args.sample_steps,
                eta=args.ddim_eta,
                clip_x0=False,
            )
            I_out = reconstruct(I_L, b_sample)
            out_results[ratio] = I_out.cpu()
            print(f"  I_out range: [{I_out.min():.3f}, {I_out.max():.3f}]")

    # ------------------------------------------------------------------
    # 3) 定量指标：各 ratio 与 ratio=1.0（完全去噪）做比较
    # ------------------------------------------------------------------
    ref_ratio = max(ratios)   # 1.0
    ref_il = il_results[ref_ratio]
    ref_out = out_results.get(ref_ratio)

    print("\n" + "=" * 60)
    print(f"{'Ratio':>6}  {'IL-PSNR':>9}  {'IL-SSIM':>8}", end="")
    if ref_out is not None:
        print(f"  {'Out-PSNR':>9}  {'Out-SSIM':>9}", end="")
    print()
    print("-" * 60)

    for ratio in ratios:
        il_p = _psnr(il_results[ratio], ref_il)
        il_s = _ssim(il_results[ratio], ref_il)
        row = f"{ratio:>6.2f}  {il_p:>9.2f}  {il_s:>8.4f}"
        if ref_out is not None and ratio in out_results:
            out_p = _psnr(out_results[ratio], ref_out)
            out_s = _ssim(out_results[ratio], ref_out)
            row += f"  {out_p:>9.2f}  {out_s:>9.4f}"
        print(row)
    print("=" * 60)
    print("(指标均相对 ratio=1.0；inf 表示两者完全一致)")

    # ------------------------------------------------------------------
    # 4) 可视化：保存网格图，便于肉眼比较
    # ------------------------------------------------------------------
    nrow = min(args.nrow, args.n)

    # I_L 网格：每行对应一个 ratio
    il_grid_rows = torch.cat([il_results[r] for r in ratios], dim=0)
    out_path_il = args.out.replace(".jpg", "_il.jpg").replace(".png", "_il.png")
    if not (out_path_il.endswith(".jpg") or out_path_il.endswith(".png")):
        out_path_il += "_il.jpg"
    save_image(il_grid_rows, out_path_il, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"\nSaved I_L comparison grid → {out_path_il}")

    if out_results:
        out_grid_rows = torch.cat([out_results[r] for r in ratios if r in out_results], dim=0)
        save_image(out_grid_rows, args.out, nrow=nrow, normalize=True, value_range=(-1, 1))
        print(f"Saved I_out comparison grid → {args.out}")
    else:
        save_image(il_grid_rows, args.out, nrow=nrow, normalize=True, value_range=(-1, 1))

    # 交错网格：同一个样本在不同 ratio 下并排展示
    if args.n >= 1 and len(ratios) > 1:
        # 形状: [n * len(ratios), C, H, W]
        # 排序: img0_r0, img0_r1, ..., img1_r0, ...
        interleaved_il = torch.stack(
            [il_results[r][i] for i in range(min(args.n, nrow)) for r in ratios], dim=0
        )
        save_image(
            interleaved_il,
            args.out.replace(".jpg", "_il_interleaved.jpg").replace(".png", "_il_interleaved.png"),
            nrow=len(ratios), normalize=True, value_range=(-1, 1),
        )
        print(f"Saved interleaved I_L grid (col=ratio {ratios}) → "
              f"{args.out.replace('.jpg', '_il_interleaved.jpg')}")

        if out_results:
            interleaved_out = torch.stack(
                [out_results[r][i] for i in range(min(args.n, nrow))
                 for r in ratios if r in out_results], dim=0
            )
            save_image(
                interleaved_out,
                args.out.replace(".jpg", "_interleaved.jpg").replace(".png", "_interleaved.png"),
                nrow=len(ratios), normalize=True, value_range=(-1, 1),
            )
            print(f"Saved interleaved I_out grid → "
                  f"{args.out.replace('.jpg', '_interleaved.jpg')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
