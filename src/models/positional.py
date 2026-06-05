"""Shared positional encoding utilities for the VAE encoder and decoder."""

from __future__ import annotations

import math

import torch


def sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Return a ``(max_len, dim)`` sinusoidal positional-encoding table."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_encoding requires even dim, got {dim}")
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
