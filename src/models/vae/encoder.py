"""VAE encoder: a FROZEN pretrained transformer (e.g. BERT) + Perceiver-style
query pool to a *sequence* of latent parameters.

This follows the LangVAE design (arXiv:2505.00004): the heavy language
understanding is done by a frozen pretrained encoder, and only a small pooling
head + variational projection are trained. The encoder emits
``num_latent_tokens`` query-pooled vectors of shape ``(B, K, latent_dim)`` via
Perceiver-style cross-attention from K learnable queries onto the frozen
encoder's hidden states — the "sequence latent" that preserves sub-segment
structure for a downstream diffusion denoiser (vs LangVAE's single vector).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.pretrained_embeddings import load_pretrained_encoder_model


class _PerceiverRefineBlock(nn.Module):
    """One refinement layer: the K queries re-attend to the frozen encoder's
    hidden states, then a position-wise FFN. Pre-norm with residuals. Stacking
    these lets the pool iteratively pull finer detail (e.g. exact entity
    identity) out of the backbone than a single cross-attention can.
    """

    def __init__(self, hidden: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        attn, _ = self.cross_attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), key_padding_mask=pad_mask
        )
        q = q + attn
        q = q + self.ff(self.norm_ff(q))
        return q


class VAEEncoder(nn.Module):
    """Frozen pretrained backbone + Perceiver query pool → ``(μ, log_var)``.

    Parameters
    ----------
    model_name : str
        HuggingFace ``AutoModel`` id (e.g. ``"bert-base-uncased"``).
    latent_dim : int
        Per-token latent dimensionality.
    num_latent_tokens : int
        Number of Perceiver query tokens K (latent sequence length).
    num_heads : int
        Attention heads for the cross-attention pool.
    dropout : float
        Dropout for the cross-attention pool.
    vocab_size : int
        Encoder-tokenizer vocab size (``len(tokenizer)``) — the backbone's input
        embeddings are resized to this so added special tokens like
        ``[NULL_ANS]`` are in range.
    unfreeze_top_n : int
        Number of top backbone encoder layers to leave trainable (0 = fully
        frozen, the default / LangVAE setting).
    """

    def __init__(
        self,
        model_name: str,
        latent_dim: int,
        num_latent_tokens: int,
        num_heads: int,
        dropout: float,
        vocab_size: int,
        unfreeze_top_n: int = 0,
        pool_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens

        # --- Frozen pretrained backbone ---
        self.backbone = load_pretrained_encoder_model(model_name, freeze=True)
        # Cover special tokens added to the tokenizer (e.g. [NULL_ANS]).
        if self.backbone.get_input_embeddings().weight.size(0) != vocab_size:
            self.backbone.resize_token_embeddings(vocab_size)
            self.backbone.requires_grad_(False)  # re-freeze any new rows
        hidden = self.backbone.config.hidden_size

        # Optionally unfreeze the top-N transformer layers of the backbone.
        if unfreeze_top_n > 0:
            self._unfreeze_top_layers(unfreeze_top_n)

        # --- Perceiver-style latent queries (trained) ---
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latent_tokens, hidden) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm_q = nn.LayerNorm(hidden)
        self.cross_attn_norm_kv = nn.LayerNorm(hidden)

        # --- Optional deeper pool: refinement blocks beyond the first cross-attn.
        # pool_num_layers=1 leaves behavior identical to the original single
        # cross-attention; >1 adds (pool_num_layers - 1) refine blocks.
        self.refine_layers = nn.ModuleList(
            _PerceiverRefineBlock(hidden, num_heads, dropout)
            for _ in range(max(0, pool_num_layers - 1))
        )

        # --- Variational projection (trained) ---
        self.proj = nn.Linear(hidden, latent_dim)
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def _unfreeze_top_layers(self, n: int) -> None:
        """Set ``requires_grad=True`` on the top *n* backbone encoder layers."""
        enc = getattr(self.backbone, "encoder", None)
        layers = getattr(enc, "layer", None) if enc is not None else None
        if layers is None:
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)

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
        B = token_ids.size(0)
        pad_mask = mask == 0  # True = ignore

        out = self.backbone(input_ids=token_ids, attention_mask=mask.long())
        hidden = out.last_hidden_state  # (B, L, H)

        # Cross-attend K queries to the frozen encoder output.
        queries = self.latent_queries.expand(B, -1, -1)  # (B, K, H)
        q = self.cross_attn_norm_q(queries)
        kv = self.cross_attn_norm_kv(hidden)
        pooled, _ = self.cross_attn(q, kv, kv, key_padding_mask=pad_mask)
        pooled = queries + pooled  # residual so init cross-attn doesn't zero out

        # Optional refinement: re-attend the K queries to the frozen hidden
        # states (each block uses its own pre-norm on the raw backbone output).
        for layer in self.refine_layers:
            pooled = layer(pooled, hidden, pad_mask)

        h = self.proj(pooled)  # (B, K, latent_dim)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-6.0, 4.0)
        return mu, log_var
