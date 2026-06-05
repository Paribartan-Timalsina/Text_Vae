"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_REQUIRED_KEYS = {
    "step",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "ema_state_dict",
    "config",
    "metrics",
}


class CheckpointMissingKeyError(Exception):
    """Raised when a checkpoint is missing one or more required keys."""


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema,
    config,
    step: int,
    metrics: dict[str, Any],
) -> None:
    """Save a training checkpoint to *path*.

    Parameters
    ----------
    path : str or Path
        Output ``.pt`` file path.
    model : nn.Module
    optimizer : Optimizer
    scheduler : LambdaLR (or any scheduler with state_dict)
    ema : EMAManager
    config : Config
        The full config dataclass (serialised via ``to_dict()``).
    step : int
        Current training step.
    metrics : dict
        Arbitrary metrics dict to store alongside the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "config": config.to_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint and validate that all required keys are present.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pt`` file.

    Returns
    -------
    dict
        The full checkpoint dict.

    Raises
    ------
    CheckpointMissingKeyError
        If any expected key is absent from the checkpoint.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    missing = _REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise CheckpointMissingKeyError(
            f"Checkpoint at '{path}' is missing keys: {sorted(missing)}"
        )

    return ckpt
