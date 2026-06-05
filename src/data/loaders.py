"""DataLoader factory functions (VAE / SQuAD only)."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, default_collate

from src.config.schema import Config
from src.data.squad_dataset import SQuADDataset
from src.data.sampler import create_balanced_sampler

# Fields that contain plain strings or variable-length lists of strings —
# default_collate cannot handle these (it requires equal-length sequences).
_SQUAD_STR_KEYS = {"answer_text", "all_answer_texts"}


def _squad_collate(batch):
    """Collate SQuAD samples, keeping string fields as plain Python lists."""
    str_batch = {k: [sample[k] for sample in batch] for k in _SQUAD_STR_KEYS if k in batch[0]}
    tensor_batch = default_collate(
        [{k: v for k, v in sample.items() if k not in _SQUAD_STR_KEYS} for sample in batch]
    )
    tensor_batch.update(str_batch)
    return tensor_batch


def _subsample_to_null_fraction(data, null_fraction: float, seed: int = 42):
    """Return a copy of *data* whose unanswerable (NULL) examples are subsampled
    so that ``n_null / (n_ans + n_null) ≈ null_fraction``.

    All answerable examples are kept; only nulls are dropped. Deterministic
    given *seed*. SQuAD v2 is ~33% null and the answerability-balanced sampler
    inflates that to 50%, which starves the VAE's real-answer reconstruction.
    """
    import random
    answers = data["answers"]  # column access: list of {"text": [...], ...}
    ans_idx = [i for i, a in enumerate(answers) if len(a["text"]) > 0]
    null_idx = [i for i, a in enumerate(answers) if len(a["text"]) == 0]

    if null_fraction <= 0.0:
        target_null = 0
    else:
        target_null = int(round(null_fraction / (1.0 - null_fraction) * len(ans_idx)))
    target_null = min(target_null, len(null_idx))

    rng = random.Random(seed)
    rng.shuffle(null_idx)
    keep = sorted(ans_idx + null_idx[:target_null])
    return data.select(keep)


def create_squad_dataloaders(
    config: Config,
    tokenizer,
    null_train_fraction: float | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for SQuAD v2.

    Splits the SQuAD v2 training set 90/10 (reproducible seed=42) so that
    validation reflects in-distribution performance on held-out training
    examples rather than the official test partition.

    Parameters
    ----------
    null_train_fraction : float, optional
        If set, the *training* set's NULL (unanswerable) examples are
        subsampled to this fraction and the training loader uses plain
        shuffling instead of the answerability-balanced sampler (which would
        re-inflate nulls to 50%). The validation set is left at its natural
        distribution.
    """
    from datasets import load_dataset

    raw_train = load_dataset("squad_v2", split="train")
    splits = raw_train.train_test_split(test_size=0.1, seed=42)

    ds_kwargs = dict(
        split="train",  # unused when data= is provided; kept for interface compat
        tokenizer=tokenizer,
        max_context_len=config.encoder.max_context_len,
        max_question_len=config.encoder.max_question_len,
        max_answer_len=config.vae_arch.max_answer_len,
    )

    train_split = splits["train"]
    if null_train_fraction is not None:
        train_split = _subsample_to_null_fraction(train_split, null_train_fraction)

    train_ds = SQuADDataset(**ds_kwargs, data=train_split)
    val_ds = SQuADDataset(**ds_kwargs, data=splits["test"])

    if null_train_fraction is not None:
        # Composition is already rebalanced to the requested null fraction, so
        # iterate it directly with plain shuffling. Do NOT use the balanced
        # sampler here — it equalises the classes and would undo the subsample.
        train_loader = DataLoader(
            train_ds,
            batch_size=config.vae_training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=_squad_collate,
        )
    else:
        train_sampler = create_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.vae_training.batch_size,
            sampler=train_sampler,
            num_workers=0,
            drop_last=True,
            collate_fn=_squad_collate,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.vae_training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=_squad_collate,
    )
    return train_loader, val_loader
