"""Tokenizer creation with special token support."""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerFast

NULL_TOKEN = "[NULL_ANS]"


def create_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Load a pretrained tokenizer and add the [NULL_ANS] special token.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"bert-base-uncased"``).

    Returns
    -------
    PreTrainedTokenizerFast
        Tokenizer with ``[NULL_ANS]`` registered as an additional special
        token so it is never split by the sub-word algorithm.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Only add if not already present
    if NULL_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [NULL_TOKEN]}
        )

    return tokenizer


def create_decoder_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Load the *decoder* (causal-LM) tokenizer used to tokenize the
    reconstruction target and to detokenize generated ids.

    This is separate from the encoder tokenizer: the encoder reads the answer in
    BERT's vocabulary, while the decoder generates in its own (e.g. GPT-2 BPE).
    Two adjustments are needed for GPT-2-family models:

    * GPT-2 has no pad token — set ``pad_token = eos_token`` so batched
      teacher-forced inputs can be padded. Padding is masked out of the loss.
    * ``add_prefix_space=True`` so a leading word is tokenized the same whether
      or not it follows a space — this keeps SQuAD answers round-tripping
      cleanly (matches LangVAE's decoder tokenizer setup).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_null_token_id(tokenizer: PreTrainedTokenizerFast) -> int:
    """Return the integer ID of the ``[NULL_ANS]`` token.

    Raises
    ------
    ValueError
        If the token has not been added to *tokenizer*.
    """
    token_id = tokenizer.convert_tokens_to_ids(NULL_TOKEN)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"{NULL_TOKEN} is not in the tokenizer vocabulary. "
            "Call create_tokenizer() first."
        )
    return token_id
