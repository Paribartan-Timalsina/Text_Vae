"""Tests for the quality gate pipeline."""

from __future__ import annotations

from dataclasses import replace as _replace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.config.schema import Config, QualityGateConfig
from src.pipelines.quality_gate import run_quality_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_CHECKS = {
    "recon_accuracy",
    "mean_kl",
    "active_dims",
    "dead_slots",
    "min_active_in_any_slot",
    "centroid_distance",
}


def _make_config_with_low_thresholds(tiny_config: Config) -> Config:
    """Return a config whose quality gate thresholds are very easy to pass."""
    K = tiny_config.vae_arch.num_latent_tokens
    qg = QualityGateConfig(
        min_recon_accuracy=0.0,
        min_mean_kl=0.0,
        min_active_dims=0,
        min_centroid_distance=0.0,
        active_dim_variance_threshold=1e9,  # nothing will be "active" — but threshold is 0
        max_dead_slots=K,  # untrained model collapses every slot — accept all
        min_active_in_any_slot=0,
    )
    return _replace(tiny_config, quality_gate=qg)


def _make_fake_loader(config: Config, n: int = 4) -> DataLoader:
    """DataLoader yielding tiny random batches shaped like SQuAD outputs."""
    B = config.vae_training.batch_size
    L = config.vae_arch.max_answer_len
    V = 100

    answer_ids = torch.randint(0, V, (n * B, L))
    answer_mask = torch.ones(n * B, L, dtype=torch.long)
    is_answerable = torch.cat([
        torch.ones(n * B // 2, dtype=torch.bool),
        torch.zeros(n * B - n * B // 2, dtype=torch.bool),
    ])

    ds = TensorDataset(answer_ids, answer_mask, is_answerable)

    def _collate(items):
        ids, mask, ans = zip(*items)
        return {
            "answer_ids": torch.stack(ids),
            "answer_mask": torch.stack(mask),
            "is_answerable": torch.stack(ans),
        }

    return DataLoader(ds, batch_size=B, collate_fn=_collate, drop_last=True)


def _make_vae(config: Config) -> nn.Module:
    """SequenceVAE with tiny random embeddings (no BERT)."""
    from src.models.vae.vae import SequenceVAE

    V = 100
    emb = torch.randn(V, config.vae_arch.embed_dim) * 0.02
    return SequenceVAE(config.vae_arch, pretrained_embeddings=emb)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_good_vae_passes(tiny_config: Config):
    """A VAE evaluated against very low thresholds should pass the gate."""
    cfg = _make_config_with_low_thresholds(tiny_config)
    vae = _make_vae(cfg)
    loader = _make_fake_loader(cfg, n=4)
    device = torch.device("cpu")

    passed, report = run_quality_gate(vae, loader, cfg, device)

    assert passed is True, (
        f"Expected quality gate to pass with low thresholds, got report={report}"
    )


def test_report_always_complete(tiny_config: Config):
    """Report should always contain all expected check names regardless of outcome."""
    cfg = tiny_config  # default thresholds (may or may not pass)
    vae = _make_vae(cfg)
    loader = _make_fake_loader(cfg, n=4)
    device = torch.device("cpu")

    _, report = run_quality_gate(vae, loader, cfg, device)

    missing = _EXPECTED_CHECKS - set(report.keys())
    assert not missing, f"Report missing checks: {missing}"

    # Each check must have value, passed, threshold keys
    for name, entry in report.items():
        assert "value" in entry, f"Check '{name}' missing 'value'"
        assert "passed" in entry, f"Check '{name}' missing 'passed'"
        assert "threshold" in entry, f"Check '{name}' missing 'threshold'"
