"""EntailmentBank explanatory-sentence dataset for VAE training.

Follows the LangVAE paper (arXiv:2505.00004): the sentence VAE is trained to
reconstruct individual *explanatory sentences* from EntailmentBank (the
entailment-tree facts and intermediate conclusions), one sentence per example,
deduplicated. There is no notion of an unanswerable / NULL example here, so
``is_answerable`` is always True and the null-handling machinery
(``null_loss_weight``, balanced sampler) is a no-op for this corpus.

Source: the ``nguyen-brat/entailment_bank`` HF mirror, whose ``cot`` field holds
the chain-of-thought explanatory sentences. We flatten ``cot`` across all rows
into a single deduplicated sentence pool, matching the paper's "subset of all
explanatory sentences" preprocessing.
"""

from __future__ import annotations

from typing import Any, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.data.squad_dataset import _tokenize_and_pad, make_decoder_target

# HF mirror of EntailmentBank and the field holding explanatory sentences.
_HF_NAME = "nguyen-brat/entailment_bank"
_EXPLANATION_FIELD = "cot"


def collect_explanatory_sentences() -> list[str]:
    """Load EntailmentBank and return the deduplicated explanatory-sentence pool.

    Flattens the ``cot`` (chain-of-thought) lists across every row into one list,
    strips whitespace, drops empties, and deduplicates while preserving first-seen
    order (deterministic — no shuffle here; the split is seeded downstream).
    """
    from datasets import load_dataset

    ds = load_dataset(_HF_NAME, split="train")

    seen: set[str] = set()
    sentences: list[str] = []
    for row in ds[_EXPLANATION_FIELD]:
        for sent in row or []:
            s = sent.strip()
            if s and s not in seen:
                seen.add(s)
                sentences.append(s)
    return sentences


class EntailmentBankDataset(Dataset):
    """Wraps a list of explanatory sentences with on-the-fly tokenization.

    Emits the SAME batch schema as :class:`SQuADDataset` so the existing collate,
    training loop, and ``_validate`` work unchanged. The sentence is the
    reconstruction target (``answer_ids``); ``context``/``question`` are filled
    with the same sentence purely to satisfy the shared schema — the VAE only
    consumes ``answer_ids``/``answer_mask``.

    Parameters
    ----------
    sentences:
        The explanatory-sentence pool (or a split of it).
    tokenizer:
        Tokenizer with ``[NULL_ANS]`` registered (shared with SQuAD path).
    max_answer_len:
        Max token length for the reconstructed sentence.
    max_context_len, max_question_len:
        Lengths for the (unused) context/question fields kept for schema parity.
    """

    def __init__(
        self,
        sentences: List[str],
        tokenizer: PreTrainedTokenizerFast,
        max_answer_len: int,
        max_context_len: int,
        max_question_len: int,
        decoder_tokenizer: PreTrainedTokenizerFast | None = None,
        dec_max_answer_len: int | None = None,
    ) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_answer_len = max_answer_len
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.decoder_tokenizer = decoder_tokenizer
        self.dec_max_answer_len = dec_max_answer_len or max_answer_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sentence = self.sentences[idx]

        # Context/question are unused by the VAE but kept for schema parity with
        # the SQuAD path (collate + balanced sampler expect these keys).
        context_ids, context_mask = _tokenize_and_pad(
            self.tokenizer, sentence, self.max_context_len
        )
        question_ids, question_mask = _tokenize_and_pad(
            self.tokenizer, sentence, self.max_question_len
        )

        # Reconstruction target. Mirror SQuADDataset: skip [CLS], append [SEP] as
        # the end-of-sequence marker so the decoder learns where to stop.
        answer_ids, answer_mask = _tokenize_and_pad(
            self.tokenizer, sentence, self.max_answer_len, add_special_tokens=False
        )
        sep_id = self.tokenizer.sep_token_id
        if isinstance(sep_id, int):
            real_len = int(answer_mask.sum().item())
            if real_len < self.max_answer_len:
                answer_ids[real_len] = int(sep_id)
                answer_mask[real_len] = 1

        item = {
            "context_ids": context_ids,
            "context_mask": context_mask,
            "question_ids": question_ids,
            "question_mask": question_mask,
            "answer_ids": answer_ids,
            "answer_mask": answer_mask,
            # EntailmentBank has no unanswerable examples.
            "is_answerable": torch.tensor(True, dtype=torch.bool),
            "answer_text": sentence,
            "all_answer_texts": [sentence],
        }

        if self.decoder_tokenizer is not None:
            dec_ids, dec_mask = make_decoder_target(
                self.decoder_tokenizer, sentence, self.dec_max_answer_len
            )
            item["dec_answer_ids"] = dec_ids
            item["dec_answer_mask"] = dec_mask

        return item
