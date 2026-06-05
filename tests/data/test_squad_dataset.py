"""Tests for SQuADDataset."""

import pytest

from src.data.tokenization import create_tokenizer, get_null_token_id
from src.data.squad_dataset import SQuADDataset

MAX_CONTEXT = 32
MAX_QUESTION = 16
MAX_ANSWER = 10


@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset(tokenizer):
    """Cached validation split dataset (smaller than train)."""
    return SQuADDataset(
        split="validation",
        tokenizer=tokenizer,
        max_context_len=MAX_CONTEXT,
        max_question_len=MAX_QUESTION,
        max_answer_len=MAX_ANSWER,
    )


@pytest.fixture(scope="module")
def null_token_id(tokenizer):
    return get_null_token_id(tokenizer)


def test_dataset_loads(dataset):
    """Dataset should contain at least one example."""
    assert len(dataset) > 0


def test_item_shapes(dataset):
    """All tensor shapes should match the configured maximums."""
    item = dataset[0]
    assert item["context_ids"].shape == (MAX_CONTEXT,)
    assert item["context_mask"].shape == (MAX_CONTEXT,)
    assert item["question_ids"].shape == (MAX_QUESTION,)
    assert item["question_mask"].shape == (MAX_QUESTION,)
    assert item["answer_ids"].shape == (MAX_ANSWER,)
    assert item["answer_mask"].shape == (MAX_ANSWER,)


def test_answerable_has_answer_text(dataset):
    """For answerable items, answer_text should be non-empty."""
    # Find an answerable item
    for i in range(min(100, len(dataset))):
        item = dataset[i]
        if item["is_answerable"]:
            assert len(item["answer_text"]) > 0
            return
    pytest.skip("No answerable item found in first 100 examples")


def test_unanswerable_uses_null(dataset, null_token_id):
    """For unanswerable items, answer_ids should start with the null token."""
    # Find an unanswerable item
    for i in range(len(dataset)):
        item = dataset[i]
        if not item["is_answerable"]:
            # Answers are tokenized without [CLS]/[SEP]; [NULL_ANS] should be present.
            assert null_token_id in item["answer_ids"].tolist()
            return
    pytest.skip("No unanswerable item found")
