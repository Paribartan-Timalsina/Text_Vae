"""VAE encoder: embeds token ids and produces a *sequence* of latent vectors.

Earlier versions of this module pooled the entire input sequence into a
single ``(B, D)`` vector. That was a punishing bottleneck for text and is
also incompatible with a sequence-aware diffusion denoiser. The encoder
now emits ``num_latent_tokens`` query-pooled vectors of shape ``(B, K, D)``
via Perceiver-style cross-attention from K learnable queries onto the
transformer output, preserving positional / sub-segment structure that
diffusion can later attend over.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.positional import sinusoidal_encoding as _sinusoidal_encoding


class VAEEncoder(nn.Module):
    """Bidirectional transformer + Perceiver-style query pool to a sequence
    of latent parameters ``(μ, log_var)`` each of shape ``(B, K, latent_dim)``.

    The K learnable query tokens are randomly initialised parameters; each
    query attends to the full encoder output and produces one latent vector.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
        num_latent_tokens: int,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.num_latent_tokens = num_latent_tokens

        # --- Embedding ---
        if pretrained_embeddings is not None:
            vocab_size, emb_d = pretrained_embeddings.shape
            self.embedding = nn.Embedding(vocab_size, emb_d)
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
        else:
            raise ValueError(
                "vocab_size must be inferred from pretrained_embeddings or set explicitly"
            )

        # --- Positional encoding (sinusoidal, non-learnable) ---
        pe = _sinusoidal_encoding(max_answer_len, embed_dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Perceiver-style latent queries ---
        # K learnable query tokens that cross-attend to the encoder output.
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latent_tokens, embed_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm_q = nn.LayerNorm(embed_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(embed_dim)

        # --- Projection to latent space (per-token) ---
        self.proj = nn.Linear(embed_dim, latent_dim)
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    @classmethod
    def from_vocab_size(
        cls,
        vocab_size: int,
        embed_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
        num_latent_tokens: int,
    ) -> "VAEEncoder":
        emb = torch.randn(vocab_size, embed_dim)
        return cls(
            embed_dim,
            latent_dim,
            num_layers,
            num_heads,
            dropout,
            max_answer_len,
            num_latent_tokens,
            emb,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to a sequence of latent parameters.

        Parameters
        ----------
        token_ids : Tensor (B, L)
        mask : Tensor (B, L)  — 1 for real tokens, 0 for padding.

        Returns
        -------
        (μ, log_var) each of shape (B, K, latent_dim)
        """
        B, L = token_ids.shape
        x = self.embedding(token_ids) + self.pe[:, :L, :]

        pad_mask = mask == 0  # True = ignore
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Cross-attend K queries to the encoder output.
        queries = self.latent_queries.expand(B, -1, -1)  # (B, K, D)
        q = self.cross_attn_norm_q(queries)
        kv = self.cross_attn_norm_kv(x)
        pooled, _ = self.cross_attn(q, kv, kv, key_padding_mask=pad_mask)
        # Residual on the queries so an untrained cross-attn doesn't zero out
        # the signal at init.
        pooled = queries + pooled  # (B, K, D)

        h = self.proj(pooled)  # (B, K, latent_dim)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-6.0, 4.0)
        return mu, log_var
