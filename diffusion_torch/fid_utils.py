from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

import classifier_metrics_numpy


@dataclass
class FIDConfig:
    device: str = "cuda"
    batch_size: int = 64


class InceptionFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()

        model = inception_v3(weights=weights)
        model.eval()
        model.to(device)

        # Grab pooled features.
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "feat"}).to(device)
        self.extractor.eval()

    @torch.no_grad()
    def activations(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """Compute 2048-d Inception activations for images in [-1, 1]."""
        images = images.detach()
        n = images.shape[0]
        feats = []
        for i in range(0, n, batch_size):
            x = images[i : i + batch_size]
            # to [0, 1]
            x = (x + 1.0) / 2.0
            x = x.clamp(0.0, 1.0)
            # preprocess to inception format (resize+normalize)
            x = self.preprocess(x)
            out = self.extractor(x.to(self.device))["feat"]
            out = out.flatten(1)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


def compute_fid(real_acts: np.ndarray, gen_acts: np.ndarray) -> float:
    return float(classifier_metrics_numpy.frechet_classifier_distance_from_activations(real_acts, gen_acts))


def maybe_load_cached_acts(path: str) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    return arr


def save_cached_acts(path: str, acts: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, acts)


def iter_dataloader_images(loader) -> Iterable[torch.Tensor]:
    for x, _ in loader:
        yield x


@torch.no_grad()
def collect_n_images_from_loader(loader, n: int, device: torch.device) -> torch.Tensor:
    xs = []
    total = 0
    for x in iter_dataloader_images(loader):
        x = x.to(device)
        take = min(x.shape[0], n - total)
        xs.append(x[:take])
        total += take
        if total >= n:
            break
    if total < n:
        raise RuntimeError(f"Loader did not provide enough images: got {total}, need {n}")
    return torch.cat(xs, dim=0)
