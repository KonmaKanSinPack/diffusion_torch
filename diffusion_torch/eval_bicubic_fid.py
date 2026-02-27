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
from torchvision.models import inception_v3

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import classifier_metrics_numpy


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def psnr_from_batch(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-10) -> float:
    x_true_01 = denorm_to_unit(x_true)
    x_pred_01 = denorm_to_unit(x_pred)
    mse = torch.mean((x_true_01 - x_pred_01) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return float(psnr.mean().item())


def build_inception_feature_extractor(device: torch.device) -> nn.Module:
    try:
        from torchvision.models import Inception_V3_Weights

        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    except Exception:
        model = inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def inception_activations(x_01: torch.Tensor, inception: nn.Module) -> np.ndarray:
    x_299 = F.interpolate(x_01, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_299.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_299.device).view(1, 3, 1, 1)
    x_norm = (x_299 - mean) / std
    acts = inception(x_norm)
    if isinstance(acts, tuple):
        acts = acts[0]
    return acts.detach().cpu().numpy()


@torch.no_grad()
def bicubic_reconstruct(images: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return images

    h, w = images.shape[-2], images.shape[-1]
    lr_h = max(h // scale, 1)
    lr_w = max(w // scale, 1)

    lr = F.interpolate(images, size=(lr_h, lr_w), mode="bicubic", align_corners=False)
    recon = F.interpolate(lr, size=(h, w), mode="bicubic", align_corners=False)
    return recon


@torch.no_grad()
def evaluate_bicubic_fid(
    testloader: DataLoader,
    inception: nn.Module,
    device: torch.device,
    scale: int,
    eval_batches: int,
):
    all_real_acts = []
    all_fake_acts = []
    psnr_values = []

    for batch_idx, (images, _) in enumerate(testloader):
        if eval_batches > 0 and batch_idx >= eval_batches:
            break

        images = images.to(device)
        recon = bicubic_reconstruct(images, scale=scale)
        psnr_values.append(psnr_from_batch(images, recon))

        real_acts = inception_activations(denorm_to_unit(images), inception)
        fake_acts = inception_activations(denorm_to_unit(recon), inception)
        all_real_acts.append(real_acts)
        all_fake_acts.append(fake_acts)

    if len(all_real_acts) == 0:
        raise RuntimeError("No evaluation batches were processed. Please set --eval_batches > 0.")

    real_acts_np = np.concatenate(all_real_acts, axis=0)
    fake_acts_np = np.concatenate(all_fake_acts, axis=0)

    fid = float(
        classifier_metrics_numpy.frechet_classifier_distance_from_activations(
            real_acts_np, fake_acts_np
        )
    )
    psnr = float(np.mean(psnr_values))
    return fid, psnr, real_acts_np.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--scale", type=int, default=2, help="Downsample scale before bicubic upsampling")
    parser.add_argument("--eval_batches", type=int, default=100, help="Number of test batches to evaluate")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    inception = build_inception_feature_extractor(device)

    fid, psnr, n_images = evaluate_bicubic_fid(
        testloader=testloader,
        inception=inception,
        device=device,
        scale=args.scale,
        eval_batches=args.eval_batches,
    )

    print(
        f"[Bicubic Eval] FID: {fid:.4f} | PSNR: {psnr:.4f} dB | "
        f"scale: x{args.scale} | images: {n_images}"
    )


if __name__ == "__main__":
    main()
