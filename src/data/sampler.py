"""Balanced sampling for answerable / unanswerable classes."""

from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from src.data.squad_dataset import SQuADDataset


def create_balanced_sampler(dataset: SQuADDataset) -> WeightedRandomSampler:
    """Return a ``WeightedRandomSampler`` that balances answerable vs unanswerable.

    Each sample is weighted by ``1 / class_count`` so that both classes are
    sampled with roughly equal probability over an epoch.
    """
    answerable_flags = [
        len(dataset.data[i]["answers"]["text"]) > 0
        for i in range(len(dataset))
    ]
    n_answerable = sum(answerable_flags)
    n_unanswerable = len(dataset) - n_answerable

    weight_answerable = 1.0 / max(n_answerable, 1)
    weight_unanswerable = 1.0 / max(n_unanswerable, 1)

    weights = torch.tensor(
        [weight_answerable if a else weight_unanswerable for a in answerable_flags],
        dtype=torch.float64,
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )
