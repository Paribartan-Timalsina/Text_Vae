"""Gradient utilities: clipping and accumulation helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip model gradients by global norm and return the pre-clip norm.

    Parameters
    ----------
    model : nn.Module
    max_norm : float
        Maximum gradient norm threshold.

    Returns
    -------
    float
        The actual (pre-clip) global gradient norm.
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # clip_grad_norm_ returns a tensor; convert to float
    return float(grad_norm)


def accumulation_step(step: int, accum_steps: int) -> bool:
    """Return True if this step should trigger an optimizer update.

    Steps are 1-indexed: triggers when step is a multiple of accum_steps.

    Parameters
    ----------
    step : int
        Current step (1-indexed).
    accum_steps : int
        Number of gradient accumulation steps.

    Returns
    -------
    bool
    """
    if accum_steps <= 1:
        return True
    return step % accum_steps == 0
