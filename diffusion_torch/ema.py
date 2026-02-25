from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v in esd.items():
            if k not in msd:
                continue
            src = msd[k]
            if not torch.is_floating_point(v):
                esd[k] = src
            else:
                esd[k].mul_(self.decay).add_(src, alpha=1.0 - self.decay)
        self.ema_model.load_state_dict(esd, strict=False)

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "ema": self.ema_model.state_dict()}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.ema_model.load_state_dict(state["ema"], strict=False)


def save_checkpoint(path: str, *, model: torch.nn.Module, opt: torch.optim.Optimizer, ema: Optional[EMA], step: int) -> None:
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, *, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer] = None, ema: Optional[EMA] = None, map_location=None) -> int:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=False)
    if opt is not None and "opt" in payload:
        opt.load_state_dict(payload["opt"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    return int(payload.get("step", 0))
