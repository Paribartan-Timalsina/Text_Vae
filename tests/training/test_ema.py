"""Tests for EMAManager."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import pytest

from src.training.ema import EMAManager


def _simple_model() -> nn.Module:
    """Return a tiny linear model with known parameters."""
    model = nn.Linear(4, 4, bias=False)
    nn.init.constant_(model.weight, 1.0)
    return model


def test_ema_diverges():
    """After N updates shadow params should differ from original model params."""
    model = _simple_model()
    ema = EMAManager(model, decay=0.99, start_step=0)

    # Modify model weights and do several updates
    with torch.no_grad():
        model.weight.fill_(10.0)

    for step in range(1, 20):
        ema.update(step)

    # Shadow should differ from both original (1.0) and current model (10.0)
    shadow_val = list(ema.shadow.values())[0]
    assert not torch.allclose(shadow_val, torch.ones_like(shadow_val)), \
        "Shadow should have moved from initial 1.0"
    assert not torch.allclose(shadow_val, torch.full_like(shadow_val, 10.0)), \
        "Shadow should not immediately match model"


def test_apply_restore_roundtrip():
    """apply() copies shadow into model; restore() returns to originals."""
    model = _simple_model()
    ema = EMAManager(model, decay=0.9, start_step=0)

    original_weight = model.weight.data.clone()

    # Update shadow so it differs from model
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(step=1)

    ema.apply()
    # Model weights should now equal the shadow
    shadow_val = list(ema.shadow.values())[0]
    assert torch.allclose(model.weight.data, shadow_val), \
        "apply() should copy shadow into model"

    ema.restore()
    # Model weights should be back to the value before apply()
    assert torch.allclose(model.weight.data, torch.full_like(model.weight.data, 5.0)), \
        "restore() should recover the pre-apply model params"


def test_start_step_respected():
    """Updates before start_step should leave shadow unchanged."""
    model = _simple_model()
    start_step = 10
    ema = EMAManager(model, decay=0.99, start_step=start_step)

    initial_shadow = list(ema.shadow.values())[0].clone()

    with torch.no_grad():
        model.weight.fill_(99.0)

    for step in range(1, start_step):
        ema.update(step)

    shadow_after = list(ema.shadow.values())[0]
    assert torch.allclose(shadow_after, initial_shadow), \
        "Shadow must not change before start_step"


def test_state_dict_round_trip():
    """save → load state dict: shadow tensors should match."""
    model = _simple_model()
    ema = EMAManager(model, decay=0.99, start_step=5)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(step=5)

    state = ema.state_dict()

    # Create a fresh EMA and load
    model2 = _simple_model()
    ema2 = EMAManager(model2, decay=0.0, start_step=0)
    ema2.load_state_dict(state)

    for name in ema.shadow:
        assert torch.allclose(ema.shadow[name], ema2.shadow[name]), \
            f"Shadow mismatch for param '{name}' after round-trip"
