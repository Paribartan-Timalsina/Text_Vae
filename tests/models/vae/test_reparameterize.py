"""Tests for reparameterize and kl_divergence."""

import torch
import pytest

from src.models.vae.reparameterize import reparameterize, kl_divergence

BATCH_SIZE = 4
SEQ_LEN = 10
LATENT_DIM = 16


def test_deterministic_returns_mu():
    mu = torch.randn(BATCH_SIZE, LATENT_DIM)
    log_var = torch.randn(BATCH_SIZE, LATENT_DIM)
    z = reparameterize(mu, log_var, deterministic=True)
    assert torch.allclose(z, mu)


def test_stochastic_differs():
    mu = torch.randn(BATCH_SIZE, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, LATENT_DIM)
    z1 = reparameterize(mu, log_var, deterministic=False)
    z2 = reparameterize(mu, log_var, deterministic=False)
    assert not torch.allclose(z1, z2)


def test_kl_zero_for_standard_normal():
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, LATENT_DIM)
    kl = kl_divergence(mu, log_var)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_positive():
    mu = torch.randn(BATCH_SIZE, LATENT_DIM)
    log_var = torch.randn(BATCH_SIZE, LATENT_DIM)
    kl = kl_divergence(mu, log_var)
    assert kl.item() > 0
