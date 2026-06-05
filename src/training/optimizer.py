"""Optimizer and learning rate scheduler factories."""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    params,
    lr: float,
    weight_decay: float,
) -> AdamW:
    """Create an AdamW optimizer.

    Parameters
    ----------
    params : iterable
        Model parameters or parameter groups.
    lr : float
        Peak learning rate.
    weight_decay : float
        L2 regularisation weight.

    Returns
    -------
    AdamW
    """
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Create a linear-warmup + cosine-decay LR scheduler.

    The learning rate:
    - Linearly increases from 0 to 1.0 over the first ``warmup_steps``.
    - Then cosine-decays from 1.0 to ``min_lr_ratio`` over the remaining steps.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_steps : int
    total_steps : int
    min_lr_ratio : float
        Minimum LR as a fraction of peak LR (default 0.01 = 1%).
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)
