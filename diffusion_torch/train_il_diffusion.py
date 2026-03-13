import argparse
import os
import sys

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
from image_diffusion import ImageDDPM, DDPMParams
from ema import EMA, save_checkpoint, load_checkpoint


def grad_clip(params, mode: str = "norm", value: float = 1.0, **kwargs) -> None:
    # 梯度裁剪：防止训练时梯度爆炸，提高优化稳定性。
    assert mode in ["value", "norm"], "mode should be value or norm"
    if mode == "norm":
        nn.utils.clip_grad.clip_grad_norm_(parameters=params, max_norm=value, **kwargs)
    else:
        nn.utils.clip_grad.clip_grad_value_(parameters=params, clip_value=value)


def pad_to_square(x: torch.Tensor) -> torch.Tensor:
    # 将输入补齐为方形，便于后续按 [B, C*H, W] 做 SVD。
    b, c, h, w = x.shape
    if h == w:
        return x
    if h > w:
        diff = h - w
        return F.pad(x, (diff // 2, diff - diff // 2, 0, 0), "constant", 0)
    diff = w - h
    return F.pad(x, (0, 0, diff // 2, diff - diff // 2), "constant", 0)


@torch.no_grad()
def make_low_rank(images: torch.Tensor, *, k_truncate: int, do_pad: bool) -> torch.Tensor:
    # 通过截断奇异值构造低秩图 I_L：保留前 k 个奇异值，其他置零。
    if do_pad:
        images = pad_to_square(images)

    b, c, h, w = images.shape
    # 将每张图重排成二维矩阵，形状 [C*H, W]，以便做批量 SVD。
    images_flat = torch.reshape(images, [b, c * h, w])
    U, S, Vt = torch.linalg.svd(images_flat)
    r = S.shape[1]

    k = min(k_truncate, r)
    S_truncated = torch.zeros_like(S)
    S_truncated[:, :k] = S[:, :k]
    S_L_diag = torch.diag_embed(S_truncated)
    # I_L = U * S_truncated * V^T
    I_L_flat = U[:, :, :r] @ S_L_diag @ Vt
    I_L = torch.reshape(I_L_flat, [b, c, h, w])
    return I_L


def main():
    # -----------------------------
    # 1) 参数与环境
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--k_truncate", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ckpt", type=str, default="il_ddpm_ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview", type=str, default="il_preview.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # CIFAR10 预处理：归一化到 [-1, 1]，匹配扩散模型训练分布。
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 训练集与数据加载器。
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    # Stage-1 模型：学习低秩图 I_L 的无条件分布。
    model = UNet(t_emb_dim=128, ch=128, out_ch=3, in_ch=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay)

    # 扩散过程配置（beta schedule + timesteps）。
    ddpm = ImageDDPM(
        params=DDPMParams(timesteps=args.timesteps, beta_schedule=args.beta_schedule),
        device=device,
        dtype=torch.float32,
    )

    step = 0
    # 可选断点续训：恢复 model / optimizer / EMA / step。
    if args.resume and os.path.exists(args.ckpt):
        step = load_checkpoint(args.ckpt, model=model, opt=opt, ema=ema, map_location=device)
        print(f"Resumed from {args.ckpt} at step {step}")

    # CIFAR10 本身是方图，这里仍保留 pad 开关，便于迁移到非方形数据集。
    do_pad = True

    # -----------------------------
    # 2) 训练循环
    # -----------------------------
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
        for images, _ in pbar:
            images = images.to(device)
            with torch.no_grad():
                # 从原图 I_H 构造低秩目标 I_L（训练监督信号）。
                I_L = make_low_rank(images, k_truncate=args.k_truncate, do_pad=do_pad)

            # 随机采样时间步 t，进行噪声预测训练。
            t = torch.randint(0, ddpm.params.timesteps, (I_L.shape[0],), device=device).long()
            loss = ddpm.training_loss(model=model, x_start=I_L, t=t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip(model.parameters(), mode="norm", value=1.0)
            opt.step()
            # EMA 权重用于更稳定的采样与评估。
            ema.update(model)

            step += 1
            pbar.set_postfix(loss=float(loss.item()), step=int(step))

        # -----------------------------
        # 3) 周期性保存与可视化预览
        # -----------------------------
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.ckpt, model=model, opt=opt, ema=ema, step=step)

            ema_model = ema.ema_model.to(device)
            ema_model.eval()
            with torch.no_grad():
                # 用 EMA 模型做 DDIM 采样，得到生成的低秩图样本。
                samples = ddpm.ddim_sample_loop(
                    model=ema_model,
                    shape=torch.Size([64, 3, 32, 32]),
                    steps=args.sample_steps,
                    eta=args.ddim_eta,
                    clip_x0=True,
                )
                # 可视化：左侧真实低秩目标，右侧模型生成样本。
                images, _ = next(iter(trainloader))
                images = images.to(device)[:64]
                low = make_low_rank(images, k_truncate=args.k_truncate, do_pad=do_pad)

                pad_low = F.pad(low, (2, 2, 0, 0), "constant", 1)
                pad_sam = F.pad(samples, (2, 2, 0, 0), "constant", 1)
                grid = torch.cat([pad_low, pad_sam], dim=3)
                save_image(grid, args.preview, normalize=True)
                print(f"Saved {args.preview} and checkpoint {args.ckpt}")


if __name__ == "__main__":
    main()
