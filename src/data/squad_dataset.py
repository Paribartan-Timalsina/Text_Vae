"""SQuAD v2 dataset with on-the-fly tokenization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from src.data.tokenization import NULL_TOKEN, get_null_token_id


@dataclass
class SQuADItem:
    """Single tokenized SQuAD v2 example."""

    context_ids: torch.Tensor
    context_mask: torch.Tensor
    question_ids: torch.Tensor
    question_mask: torch.Tensor
    answer_ids: torch.Tensor
    answer_mask: torch.Tensor
    is_answerable: bool
    answer_text: str
    all_answer_texts: List[str] = field(default_factory=list)


def _tokenize_and_pad(
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    max_len: int,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize *text*, truncate/pad to *max_len*, return (ids, mask)."""
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


class SQuADDataset(Dataset):
    """Wraps HuggingFace SQuAD v2 with on-the-fly tokenization.

    Parameters
    ----------
    split:
        ``"train"`` or ``"validation"``.
    tokenizer:
        A tokenizer that already has ``[NULL_ANS]`` registered.
    max_context_len, max_question_len, max_answer_len:
        Maximum token lengths for each field.
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerFast,
        max_context_len: int,
        max_question_len: int,
        max_answer_len: int,
        data=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.null_token_id = get_null_token_id(tokenizer)

        self.data = data if data is not None else load_dataset("squad_v2", split=split)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.data[idx]

        # Context & question
        context_ids, context_mask = _tokenize_and_pad(
            self.tokenizer,
            example["context"],
            self.max_context_len,
        )
        question_ids, question_mask = _tokenize_and_pad(
            self.tokenizer,
            example["question"],
            self.max_question_len,
        )

        # Answer handling
        answers = example["answers"]
        all_answer_texts: List[str] = answers["text"] if answers["text"] else []

        if len(all_answer_texts) > 0:
            answer_text = all_answer_texts[0]
            is_answerable = True
        else:
            answer_text = NULL_TOKEN
            is_answerable = False

        # Skip [CLS] (constant prefix wastes a position), but DO append [SEP] as
        # an end-of-sequence marker so the decoder learns where to stop. Without
        # it the autoregressive decoder always emits max_answer_len tokens and
        # trailing junk pollutes EM/F1.
        answer_ids, answer_mask = _tokenize_and_pad(
            self.tokenizer,
            answer_text,
            self.max_answer_len,
            add_special_tokens=False,
        )
        sep_id = self.tokenizer.sep_token_id
        if isinstance(sep_id, int):
            real_len = int(answer_mask.sum().item())
            if real_len < self.max_answer_len:
                answer_ids[real_len] = int(sep_id)
                answer_mask[real_len] = 1

        return {
            "context_ids": context_ids,
            "context_mask": context_mask,
            "question_ids": question_ids,
            "question_mask": question_mask,
            "answer_ids": answer_ids,
            "answer_mask": answer_mask,
            "is_answerable": torch.tensor(is_answerable, dtype=torch.bool),
            "answer_text": answer_text,
            "all_answer_texts": all_answer_texts,
        }
