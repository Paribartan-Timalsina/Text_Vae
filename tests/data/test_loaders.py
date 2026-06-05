"""Tests for DataLoader factory functions."""

import pytest

from src.config.schema import Config, EncoderConfig, VAEArchConfig, VAETrainingConfig
from src.data.tokenization import create_tokenizer
from src.data.loaders import create_squad_dataloaders


@pytest.fixture(scope="module")
def loader_config():
    """Small config for loader tests."""
    return Config(
        encoder=EncoderConfig(
            max_context_len=32,
            max_question_len=16,
        ),
        vae_arch=VAEArchConfig(max_answer_len=10),
        vae_training=VAETrainingConfig(batch_size=4),
    )


@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer("bert-base-uncased")


@pytest.fixture(scope="module")
def loaders(loader_config, tokenizer):
    return create_squad_dataloaders(loader_config, tokenizer)


def test_train_loader_returns_batches(loaders):
    """Training loader should yield at least one batch with expected keys."""
    train_loader, _ = loaders
    batch = next(iter(train_loader))
    expected_keys = {
        "context_ids", "context_mask",
        "question_ids", "question_mask",
        "answer_ids", "answer_mask",
        "is_answerable",
    }
    assert expected_keys.issubset(set(batch.keys()))


def test_val_loader_no_sampler(loaders):
    """Validation loader should exist and yield batches (no sampler)."""
    _, val_loader = loaders
    assert val_loader.sampler is not None  # sequential sampler auto-created
    batch = next(iter(val_loader))
    assert "context_ids" in batch
