"""Tests for optimizer and scheduler factories."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.training.optimizer import create_optimizer, create_scheduler


def _params():
    model = nn.Linear(8, 4)
    return model.parameters()


def test_creates_adamw():
    """create_optimizer should return an AdamW instance."""
    params = _params()
    opt = create_optimizer(params, lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, AdamW), f"Expected AdamW, got {type(opt)}"


def test_scheduler_warmup():
    """LR should start near 0 at step 0 and increase during warmup."""
    model = nn.Linear(8, 4)
    opt = create_optimizer(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = create_scheduler(opt, warmup_steps=10, total_steps=100)

    # At step 0 the multiplier is 0/10 = 0 → lr ≈ 0
    lr_at_0 = scheduler.get_last_lr()[0]
    assert lr_at_0 < 1e-10, f"LR at step 0 should be near 0, got {lr_at_0}"

    # After 5 steps the multiplier is 5/10 = 0.5 → lr = 0.5 * base_lr
    for _ in range(5):
        scheduler.step()
    lr_at_5 = scheduler.get_last_lr()[0]
    assert lr_at_5 > lr_at_0, f"LR should increase during warmup"

    # After warmup is complete the LR should be at its peak
    for _ in range(5):
        scheduler.step()
    lr_at_10 = scheduler.get_last_lr()[0]
    assert lr_at_10 > lr_at_5, f"LR should keep rising through warmup"
