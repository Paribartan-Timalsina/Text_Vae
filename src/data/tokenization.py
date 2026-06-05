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
