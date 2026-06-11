"""DataLoader factory functions."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, default_collate

from src.config.schema import Config
from src.data.tokenization import create_tokenizer, create_decoder_tokenizer
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
        distribution. When ``None`` (e.g. for ``export_latents``) the original
        balanced-sampler behaviour over the full dataset is preserved.
    """
    from datasets import load_dataset

    raw_train = load_dataset("squad_v2", split="train")
    splits = raw_train.train_test_split(test_size=0.1, seed=42)

    decoder_tokenizer = create_decoder_tokenizer(config.decoder.model_name)
    ds_kwargs = dict(
        split="train",  # unused when data= is provided; kept for interface compat
        tokenizer=tokenizer,
        max_context_len=config.encoder.max_context_len,
        max_question_len=config.encoder.max_question_len,
        max_answer_len=config.vae_arch.max_answer_len,
        decoder_tokenizer=decoder_tokenizer,
        dec_max_answer_len=config.decoder.max_answer_len,
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


def create_entailment_dataloaders(
    config: Config,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """Train/val DataLoaders over EntailmentBank explanatory sentences.

    Follows the LangVAE paper (arXiv:2505.00004): all explanatory sentences,
    deduplicated, 99/1 split (seed=42). No NULLs, so no balanced sampler and no
    null subsampling — plain shuffling. Emits the SQuAD batch schema so the rest
    of the VAE pipeline is unchanged.
    """
    from src.data.entailment_dataset import (
        EntailmentBankDataset,
        collect_explanatory_sentences,
    )

    sentences = collect_explanatory_sentences()

    # 99/1 split, deterministic. Shuffle once with a fixed seed so the held-out
    # 1% is a random slice rather than the tail of the dedup-ordered pool.
    import random

    rng = random.Random(42)
    order = list(range(len(sentences)))
    rng.shuffle(order)
    shuffled = [sentences[i] for i in order]
    n_val = max(1, round(len(shuffled) * 0.01))
    val_sents = shuffled[:n_val]
    train_sents = shuffled[n_val:]

    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_answer_len=config.vae_arch.max_answer_len,
        max_context_len=config.encoder.max_context_len,
        max_question_len=config.encoder.max_question_len,
        decoder_tokenizer=create_decoder_tokenizer(config.decoder.model_name),
        dec_max_answer_len=config.decoder.max_answer_len,
    )
    train_ds = EntailmentBankDataset(sentences=train_sents, **ds_kwargs)
    val_ds = EntailmentBankDataset(sentences=val_sents, **ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.vae_training.batch_size,
        shuffle=True,
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


def create_vae_dataloaders(
    config: Config,
    tokenizer,
    null_train_fraction: float | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Dispatch to the VAE training corpus selected by ``vae_training.dataset``.

    ``"squad_v2"`` → :func:`create_squad_dataloaders` (honours
    ``null_train_fraction``). ``"entailment_bank"`` →
    :func:`create_entailment_dataloaders` (no NULLs; ``null_train_fraction`` is
    ignored).
    """
    name = config.vae_training.dataset
    if name == "squad_v2":
        return create_squad_dataloaders(
            config, tokenizer, null_train_fraction=null_train_fraction
        )
    if name == "entailment_bank":
        return create_entailment_dataloaders(config, tokenizer)
    raise ValueError(
        f"Unknown vae_training.dataset={name!r}; expected 'squad_v2' or 'entailment_bank'"
    )
