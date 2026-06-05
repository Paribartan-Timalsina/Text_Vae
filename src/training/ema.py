"""Exponential Moving Average (EMA) manager for model parameters."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMAManager:
    """Maintains a shadow copy of model parameters using EMA updates.

    Parameters
    ----------
    model : nn.Module
    decay : float
        EMA decay rate (e.g. 0.9999). Shadow = decay * shadow + (1-decay) * param.
    start_step : int
        Step at which EMA updates begin.
    """

    def __init__(self, model: nn.Module, decay: float, start_step: int) -> None:
        self.model = model
        self.decay = decay
        self.start_step = start_step
        # Shadow copy: dict of param name -> shadow tensor
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: Optional[dict[str, torch.Tensor]] = None

    def update(self, step: int) -> None:
        """Update shadow parameters. Only applied if step >= start_step."""
        if step < self.start_step:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                shadow = self.shadow[name]
                shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self) -> None:
        """Copy shadow params into model, saving originals for restore()."""
        self._backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model parameters saved by apply()."""
        if self._backup is None:
            raise RuntimeError("restore() called before apply()")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                param.data.copy_(self._backup[name])
        self._backup = None

    def state_dict(self) -> dict:
        """Return serializable state dict for checkpointing."""
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "decay": self.decay,
            "start_step": self.start_step,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from a previously saved state dict."""
        self.decay = state["decay"]
        self.start_step = state["start_step"]
        device = next(self.model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}
