"""Tests for decoding strategies."""

import torch
import pytest

from src.models.vae.decoding import greedy_decode, beam_search_decode, nucleus_decode

BATCH_SIZE = 4
SEQ_LEN = 10
VOCAB_SIZE = 100


@pytest.fixture
def logits():
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


def test_greedy_matches_argmax(logits):
    result = greedy_decode(logits)
    expected = logits.argmax(dim=-1)
    assert torch.equal(result, expected)


def test_beam_width_1_equals_greedy(logits):
    greedy = greedy_decode(logits)
    # eos_id=VOCAB_SIZE is out of the vocab range so beam search never hits EOS
    # and doesn't truncate, making it equivalent to greedy decode.
    beam = beam_search_decode(logits, beam_width=1, pad_id=0, eos_id=VOCAB_SIZE)
    assert torch.equal(greedy, beam)


def test_nucleus_diversity(logits):
    """Multiple nucleus samples should not all be identical."""
    # Use uniform-ish logits to encourage diversity
    uniform_logits = torch.zeros(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    samples = []
    for _ in range(10):
        s = nucleus_decode(uniform_logits, top_p=0.9, temperature=1.0)
        samples.append(s)
    # At least two samples should differ
    all_same = all(torch.equal(samples[0], s) for s in samples[1:])
    assert not all_same
