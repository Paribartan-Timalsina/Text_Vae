"""Tests for the train_vae pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.config.schema import Config, VAETrainingConfig
from src.pipelines.train_vae import train_vae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_loader(config: Config, n_batches: int = 4) -> DataLoader:
    """Return a DataLoader yielding random batches shaped like SQuAD batches."""
    B = config.vae_training.batch_size
    L = config.vae_arch.max_answer_len
    V = 100  # tiny vocab

    answer_ids_list = []
    answer_mask_list = []
    is_ans_list = []

    for _ in range(n_batches):
        answer_ids_list.append(torch.randint(0, V, (B, L)))
        answer_mask_list.append(torch.ones(B, L, dtype=torch.long))
        is_ans_list.append(torch.ones(B, dtype=torch.bool))

    answer_ids = torch.cat(answer_ids_list, dim=0)
    answer_mask = torch.cat(answer_mask_list, dim=0)
    is_ans = torch.cat(is_ans_list, dim=0)

    ds = TensorDataset(answer_ids, answer_mask, is_ans)

    def _collate(batch):
        ids, mask, ans = zip(*batch)
        return {
            "answer_ids": torch.stack(ids),
            "answer_mask": torch.stack(mask),
            "is_answerable": torch.stack(ans),
        }

    return DataLoader(ds, batch_size=B, collate_fn=_collate, drop_last=True)


def _make_tiny_vae(config: Config) -> nn.Module:
    """Build a SequenceVAE with tiny random embeddings (no BERT)."""
    from src.models.vae.vae import SequenceVAE

    V = 100
    pretrained_emb = torch.randn(V, config.vae_arch.embed_dim) * 0.02
    return SequenceVAE(config.vae_arch, pretrained_embeddings=pretrained_emb)


def _patch_pretrained_loader(monkeypatch) -> None:
    """Bypass the real BERT embedding loader so tiny test configs don't trip
    the embed_dim sanity check (BERT is 768-d; tests use 64-d)."""
    import src.utils.pretrained_embeddings as pe_mod

    def _fake_loader(model_name, target_vocab_size, target_embed_dim):
        return torch.randn(target_vocab_size, target_embed_dim) * 0.02

    monkeypatch.setattr(pe_mod, "load_pretrained_token_embeddings", _fake_loader)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_epoch_runs(tiny_config: Config, monkeypatch):
    """Training for 2 steps should complete without raising."""
    _patch_pretrained_loader(monkeypatch)
    tc = tiny_config.vae_training
    from dataclasses import replace as _replace

    new_tc = _replace(tc, epochs=1, patience=10, val_every_n_steps=2)
    cfg = _replace(tiny_config, vae_training=new_tc)

    device = torch.device("cpu")
    train_loader = _make_fake_loader(cfg, n_batches=2)
    val_loader = _make_fake_loader(cfg, n_batches=1)

    # Monkey-patch train_vae to use our fake VAE (no BERT)
    import src.pipelines.train_vae as tv_mod

    original_seq_vae = None
    vae_instance = _make_tiny_vae(cfg)

    class _FakeVAECls:
        def __new__(cls, arch_cfg, pretrained_embeddings=None):
            return vae_instance

    old_cls = tv_mod.SequenceVAE
    tv_mod.SequenceVAE = _FakeVAECls

    try:
        metrics = train_vae(
            cfg, device=device, train_loader=train_loader, val_loader=val_loader
        )
    finally:
        tv_mod.SequenceVAE = old_cls

    assert isinstance(metrics, dict), "train_vae should return a dict"


def test_loss_decreases(tiny_config: Config, monkeypatch):
    """Training for 20 steps on the same batch should reduce loss."""
    _patch_pretrained_loader(monkeypatch)
    from dataclasses import replace as _replace

    new_tc = _replace(
        tiny_config.vae_training,
        epochs=1,
        patience=100,
        learning_rate=1e-2,
        beta_start=0.0,
        beta_end=0.0,  # zero KL makes recon easier to minimise
        beta_warmup_steps=1,
        val_every_n_steps=1000,  # skip val during loss measurement run
    )
    cfg = _replace(tiny_config, vae_training=new_tc)
    device = torch.device("cpu")

    # Repeat the SAME batch so the model can overfit
    B = cfg.vae_training.batch_size
    L = cfg.vae_arch.max_answer_len
    V = 100

    # Use a fixed seed so the batch + model init are deterministic *before*
    # train_vae resets the seed via seed_everything(config.seed).
    torch.manual_seed(0)
    fixed_ids = torch.randint(0, V, (B, L))
    fixed_mask = torch.ones(B, L, dtype=torch.long)
    fixed_ans = torch.ones(B, dtype=torch.bool)

    class _RepeatDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 40 * B  # more steps → reliable convergence under any global seed

        def __getitem__(self, idx):
            return fixed_ids[idx % B], fixed_mask[idx % B], fixed_ans[idx % B]

    def _collate(batch):
        ids, mask, ans = zip(*batch)
        return {
            "answer_ids": torch.stack(ids),
            "answer_mask": torch.stack(mask),
            "is_answerable": torch.stack(ans),
        }

    train_loader = DataLoader(
        _RepeatDataset(), batch_size=B, collate_fn=_collate, drop_last=True
    )
    val_loader = DataLoader(
        _RepeatDataset(), batch_size=B, collate_fn=_collate, drop_last=True
    )

    import src.pipelines.train_vae as tv_mod

    torch.manual_seed(0)
    vae_instance = _make_tiny_vae(cfg)

    class _FakeVAECls:
        def __new__(cls, arch_cfg, pretrained_embeddings=None):
            return vae_instance

    old_cls = tv_mod.SequenceVAE
    tv_mod.SequenceVAE = _FakeVAECls

    # Record losses manually
    losses: list[float] = []
    original_forward = vae_instance.forward

    def _tracked_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        losses.append(out[4]["total"].item())
        return out

    vae_instance.forward = _tracked_forward

    try:
        train_vae(cfg, device=device, train_loader=train_loader, val_loader=val_loader)
    finally:
        tv_mod.SequenceVAE = old_cls
        vae_instance.forward = original_forward

    assert len(losses) >= 5, "Expected at least 5 recorded loss values"
    first_avg = sum(losses[:3]) / 3
    last_avg = sum(losses[-3:]) / 3
    assert (
        last_avg < first_avg
    ), f"Loss should decrease: first_avg={first_avg:.4f} last_avg={last_avg:.4f}"
