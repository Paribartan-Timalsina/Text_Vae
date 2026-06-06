"""Load input-token embeddings from a pretrained HuggingFace model.

The VAE needs a meaningful starting point for its token embeddings; random
init forces the encoder/decoder to relearn the entire vocabulary subspace
from scratch. Loading the pretrained model's input embedding matrix gives
both the encoder and the OutputProjection (which ties to it) a head start.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def load_pretrained_encoder_model(model_name: str, freeze: bool = True):
    """Load a pretrained ``AutoModel`` (e.g. BERT) for use as a frozen encoder.

    Returns the HuggingFace model in eval mode with ``requires_grad_(False)``
    when *freeze* is True. The VAE encoder runs the answer through this frozen
    backbone and only trains the Perceiver pool + variational heads on top —
    the LangVAE approach. The model's input embeddings are resized lazily by the
    caller (``resize_token_embeddings``) to cover added special tokens like
    ``[NULL_ANS]``.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    if freeze:
        model.eval()
        model.requires_grad_(False)
    return model


def load_pretrained_token_embeddings(
    model_name: str,
    target_vocab_size: int,
    target_embed_dim: int,
) -> torch.Tensor:
    """Return an embedding matrix initialized from *model_name*'s input embeddings.

    Rows beyond the pretrained vocab (added special tokens like [NULL_ANS])
    are filled with small random Gaussian noise so they start out in the
    same scale as the pretrained rows.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
    target_vocab_size : int
        Final vocab size — typically ``len(tokenizer)`` after adding special
        tokens. Must be ≥ the pretrained model's vocab size.
    target_embed_dim : int
        Required embedding dimension. Must match the pretrained model.

    Returns
    -------
    Tensor of shape ``(target_vocab_size, target_embed_dim)`` (CPU, float32).

    If the pretrained model cannot be loaded (e.g. no network in a test
    environment), falls back to a scaled random init so the caller doesn't
    crash; emits a warning so this isn't silent in production.
    """
    from transformers import AutoModel

    try:
        src = AutoModel.from_pretrained(model_name)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not load pretrained embeddings from %s (%s); "
            "falling back to scaled random init.",
            model_name,
            exc,
        )
        return torch.randn(target_vocab_size, target_embed_dim) * 0.02
    src_emb = src.get_input_embeddings().weight.detach().clone().float()

    src_vocab, src_dim = src_emb.shape
    if src_dim != target_embed_dim:
        raise ValueError(
            f"Pretrained embed_dim ({src_dim}) != target embed_dim "
            f"({target_embed_dim}). Set vae_arch.embed_dim to {src_dim} or "
            f"choose a different encoder model."
        )
    if target_vocab_size < src_vocab:
        raise ValueError(
            f"target_vocab_size ({target_vocab_size}) < pretrained vocab "
            f"({src_vocab}); cannot truncate."
        )

    n_added = target_vocab_size - src_vocab
    if n_added == 0:
        return src_emb

    # Match the scale of the pretrained rows so added rows aren't outliers.
    scale = src_emb.std().item()
    extra = torch.randn(n_added, src_dim) * scale
    logger.info(
        "Loaded pretrained embeddings from %s; padded %d extra rows at scale %.4f",
        model_name,
        n_added,
        scale,
    )
    return torch.cat([src_emb, extra], dim=0)
