"""Reparameterization trick and KL divergence for the VAE."""

from __future__ import annotations

import torch


def reparameterize(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor:
    """Sample z via the reparameterization trick.

    When *deterministic* is True the mean is returned directly.
    """
    if deterministic:
        return mu
    log_var = log_var.clamp(-10.0, 6.0)  # guard against overflow before exp
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_divergence(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), summed over latent dims, averaged over batch.

    Parameters
    ----------
    mu, log_var : Tensor (B, ...D) — latent parameters with arbitrary
        non-batch shape (e.g. (B, D) for pooled, (B, K, D) for sequence).
    mask : Tensor, optional
        Ignored (kept for backward compatibility).
    """
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.mean(dim=0).sum()
