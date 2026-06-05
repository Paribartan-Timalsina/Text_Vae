"""Shared test fixtures for the entire test suite."""

import pytest
import torch
from transformers import AutoTokenizer

from src.config.schema import (
    Config,
    PathConfig,
    EncoderConfig,
    VAEArchConfig,
    VAETrainingConfig,
    QualityGateConfig,
)


@pytest.fixture
def tiny_config() -> Config:
    """Minimal config with small dims for fast CPU tests."""
    return Config(
        seed=42,
        paths=PathConfig(),
        encoder=EncoderConfig(
            model_name="bert-base-uncased",
            hidden_dim=64,
            max_context_len=32,
            max_question_len=16,
            unfreeze_top_n=0,
        ),
        vae_arch=VAEArchConfig(
            latent_dim=16,
            embed_dim=64,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_answer_len=10,
            num_latent_tokens=2,
        ),
        vae_training=VAETrainingConfig(
            learning_rate=1e-3,
            batch_size=4,
            epochs=2,
            patience=2,
            warmup_steps=10,
            weight_decay=0.01,
            grad_clip_max_norm=1.0,
            grad_accum_steps=1,
            beta_start=0.0,
            beta_end=1.0,
            beta_warmup_steps=50,
            free_bits=0.0,
            target_kl=None,
            val_every_n_steps=10,
        ),
        quality_gate=QualityGateConfig(),
    )


@pytest.fixture
def dummy_tokenizer():
    """Pretrained tokenizer with [NULL_ANS] special token added."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[NULL_ANS]"]})
    return tokenizer


@pytest.fixture
def small_batch():
    """Small random batch (B=4) for shape tests."""
    B, seq_len, dim = 4, 10, 16
    return {
        "input_ids": torch.randint(0, 1000, (B, seq_len)),
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
        "latent": torch.randn(B, seq_len, dim),
    }
