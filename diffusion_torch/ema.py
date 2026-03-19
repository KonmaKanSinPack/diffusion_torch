from __future__ import annotations

from typing import Dict, Optional

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = decay
        # 使用 state_dict 方式保存 EMA 权重, 避免 deepcopy 兼容性问题
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: Optional[torch.nn.Module] = None) -> None:
        if model is None:
            return
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.clone().detach()
                continue
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply_shadow(self, model: Optional[torch.nn.Module] = None) -> None:
        """将 EMA 权重应用到模型 (评估时调用), 原始权重保存到 backup"""
        if model is None:
            return
        self.backup = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: Optional[torch.nn.Module] = None) -> None:
        """恢复原始权重 (评估后调用)"""
        if model is None or not self.backup:
            return
        model.load_state_dict(self.backup, strict=False)
        self.backup = {}

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "ema": self.shadow}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {k: v.clone().detach() for k, v in state["ema"].items()}


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    extra: Optional[Dict] = None,
    **kwargs,
) -> None:
    """保存训练 checkpoint"""
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if ema is not None:
        payload["ema_state_dict"] = ema.shadow  # 直接保存 EMA 权重
        payload["ema_decay"] = ema.decay
    if extra is not None:
        payload.update(extra)
    # 兼容旧接口
    if 'opt' in kwargs and kwargs['opt'] is not None:
        payload["optimizer_state_dict"] = kwargs['opt'].state_dict()
    if 'step' in kwargs:
        payload["step"] = int(kwargs['step'])
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location=None,
) -> Dict:
    """加载训练 checkpoint"""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    
    # 加载模型权重
    if 'model_state_dict' in payload:
        model.load_state_dict(payload['model_state_dict'], strict=False)
    elif 'model' in payload:
        model.load_state_dict(payload['model'], strict=False)
    
    # 加载优化器
    if optimizer is not None:
        if 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        elif 'opt' in payload:
            optimizer.load_state_dict(payload['opt'])
    
    # 加载 EMA
    if ema is not None:
        if 'ema_state_dict' in payload:
            ema.shadow = {k: v.clone().detach() for k, v in payload['ema_state_dict'].items()}
        elif 'ema' in payload:
            ema.load_state_dict(payload['ema'])
    
    return payload
