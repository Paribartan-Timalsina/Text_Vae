"""Tests for checkpoint save/load utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointMissingKeyError,
)
from src.training.ema import EMAManager
from src.training.optimizer import create_optimizer, create_scheduler
from src.config.schema import Config


class _DummyScheduler:
    """Minimal scheduler stub with state_dict / load_state_dict."""

    def state_dict(self) -> dict:
        return {"lr_lambda": "dummy"}

    def load_state_dict(self, state: dict) -> None:
        pass


def _build_components():
    model = nn.Linear(4, 4)
    optimizer = create_optimizer(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = _DummyScheduler()
    ema = EMAManager(model, decay=0.99, start_step=0)
    config = Config()
    return model, optimizer, scheduler, ema, config


def test_save_load_roundtrip(tmp_path):
    """Saved checkpoint should load back with all required keys intact."""
    model, optimizer, scheduler, ema, config = _build_components()
    path = tmp_path / "ckpt.pt"
    metrics = {"val_loss": 0.5, "step": 10}

    save_checkpoint(path, model, optimizer, scheduler, ema, config, step=10,
                    metrics=metrics)

    ckpt = load_checkpoint(path)

    assert "step" in ckpt
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "scheduler_state_dict" in ckpt
    assert "ema_state_dict" in ckpt
    assert "config" in ckpt
    assert "metrics" in ckpt
    assert ckpt["step"] == 10
    assert ckpt["metrics"]["val_loss"] == pytest.approx(0.5)


def test_missing_key_raises(tmp_path):
    """Loading a checkpoint with a missing key should raise CheckpointMissingKeyError."""
    # Manually save an incomplete dict
    path = tmp_path / "broken.pt"
    torch.save({"step": 1, "model_state_dict": {}}, path)

    with pytest.raises(CheckpointMissingKeyError):
        load_checkpoint(path)
