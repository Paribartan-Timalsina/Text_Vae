"""Tests for tokenization utilities."""

from src.data.tokenization import create_tokenizer, get_null_token_id, NULL_TOKEN


def test_null_token_added():
    """[NULL_ANS] should be in the tokenizer vocabulary after creation."""
    tokenizer = create_tokenizer("bert-base-uncased")
    assert NULL_TOKEN in tokenizer.get_vocab()


def test_null_token_not_split():
    """Tokenizing '[NULL_ANS]' should produce a single token ID (not sub-words)."""
    tokenizer = create_tokenizer("bert-base-uncased")
    ids = tokenizer.encode(NULL_TOKEN, add_special_tokens=False)
    assert len(ids) == 1


def test_tokenizer_deterministic():
    """Same input text should always yield the same token IDs."""
    tokenizer = create_tokenizer("bert-base-uncased")
    text = "The quick brown fox jumps over the lazy dog."
    ids_a = tokenizer.encode(text)
    ids_b = tokenizer.encode(text)
    assert ids_a == ids_b
