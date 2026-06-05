"""Tests for gradient utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.grad_utils import clip_gradients, accumulation_step


def _model_with_large_grad(norm_value: float = 100.0) -> nn.Module:
    """Return a model whose parameter grad has L2 norm == norm_value."""
    model = nn.Linear(10, 1, bias=False)
    # manually set gradient
    grad = torch.ones(1, 10)
    grad = grad / grad.norm() * norm_value
    model.weight.grad = grad
    return model


def test_clip_large_gradient():
    """A large gradient (norm=100) should be clipped to max_norm=1."""
    model = _model_with_large_grad(norm_value=100.0)
    returned_norm = clip_gradients(model, max_norm=1.0)

    # Returned value is the pre-clip norm
    assert abs(returned_norm - 100.0) < 1.0, \
        f"Expected returned norm ~100, got {returned_norm}"

    # Post-clip the actual gradient norm should be at or below max_norm
    post_norm = model.weight.grad.norm().item()
    assert post_norm <= 1.0 + 1e-5, f"Post-clip norm should be ≤ 1, got {post_norm}"


def test_accumulation_logic():
    """accumulation_step should return True only on multiples of accum_steps."""
    accum = 4
    assert accumulation_step(3, accum) is False, "step=3, accum=4 → False"
    assert accumulation_step(4, accum) is True,  "step=4, accum=4 → True"
    assert accumulation_step(8, accum) is True,  "step=8, accum=4 → True"
    assert accumulation_step(5, accum) is False, "step=5, accum=4 → False"


def test_accumulation_logic_single_step():
    """With accum_steps=1 every step should trigger."""
    for step in range(1, 6):
        assert accumulation_step(step, 1) is True, \
            f"accum=1: step={step} should always be True"
