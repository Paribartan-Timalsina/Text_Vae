"""Output projection: plain linear over decoder hidden states.

Previously this module used cosine-similarity logits with a learned
temperature. That was non-standard, made weight decay ineffective (since
the projection is L2-normalised before use), and required tuning
``log_tau`` to avoid vanishing logits at init. Replaced with a standard
``nn.Linear`` whose weight is intended to be tied to the decoder token
embedding (handled in :class:`SequenceVAE`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OutputProjection(nn.Module):
    """Linear projection from decoder hidden state to vocabulary logits.

    Parameters
    ----------
    embed_dim : int
    vocab_size : int
    pretrained_embeddings : Tensor, optional
        Shape ``(vocab_size, embed_dim)``. When supplied the projection
        weight is initialised from it (a detached copy). The weight is
        intended to be tied to the decoder token embedding by the parent
        :class:`SequenceVAE`.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_embeddings.detach())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hidden ``(B, L, embed_dim)`` → logits ``(B, L, vocab_size)``."""
        return self.linear(hidden_states)
