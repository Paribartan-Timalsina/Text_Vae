"""Tests for OutputProjection."""

import torch
import pytest

from src.models.vae.output_head import OutputProjection

EMBED_DIM = 64
VOCAB_SIZE = 100
BATCH_SIZE = 4
SEQ_LEN = 10


@pytest.fixture
def head():
    return OutputProjection(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE)


def test_logit_shape(head):
    h = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    logits = head(h)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


def test_temperature_clamped():
    head = OutputProjection(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE)
    # Set log_tau to a very large value
    with torch.no_grad():
        head.log_tau.fill_(100.0)
    h = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    logits = head(h)
    # Cosine similarity is in [-1, 1], clamped tau <= exp(4.6) ~ 100
    # So logits should be bounded by ~100
    assert logits.abs().max().item() < 200.0
    assert torch.isfinite(logits).all()
