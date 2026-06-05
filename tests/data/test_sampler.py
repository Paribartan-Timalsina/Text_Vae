"""Tests for the balanced sampler."""

import pytest

from src.data.tokenization import create_tokenizer
from src.data.squad_dataset import SQuADDataset
from src.data.sampler import create_balanced_sampler
from torch.utils.data import DataLoader


@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return SQuADDataset(
        split="validation",
        tokenizer=tokenizer,
        max_context_len=32,
        max_question_len=16,
        max_answer_len=10,
    )


def test_balanced_ratio(dataset):
    """After sampling several batches the answerable ratio should be 30-70%."""
    from src.data.loaders import _squad_collate

    sampler = create_balanced_sampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0,
                        collate_fn=_squad_collate)

    total = 0
    answerable = 0
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        flags = batch["is_answerable"]
        total += len(flags)
        answerable += sum(flags).item()

    ratio = answerable / total
    assert 0.30 <= ratio <= 0.70, f"Answerable ratio {ratio:.2f} outside 30-70%"
