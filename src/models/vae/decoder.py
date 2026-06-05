"""VAE decoder: causal transformer with latent KV-prefix injection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional import sinusoidal_encoding as _sinusoidal_encoding


# ----------------------------------------------------------------------
# Incremental KV-cached attention helpers
#
# nn.TransformerEncoder exposes no internal KV cache, so naive
# autoregressive generation reprocesses the full ``[prefix | decoder]``
# stack every step (O(N²)). These helpers reproduce the forward pass of
# a ``nn.TransformerEncoderLayer`` (default ``norm_first=False`` post-LN
# variant) while caching the per-layer keys and values, taking generation
# down to O(N). Weights are read directly off the trained layer so no
# parameters are duplicated.

def _self_attn_cached(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
    kv_cache: dict,
) -> torch.Tensor:
    """Run self-attention with a growing K/V cache.

    Parameters
    ----------
    attn : nn.MultiheadAttention
        Module whose weights are reused (assumed to be a standard
        combined-QKV projection — the only variant our decoder uses).
    x : Tensor (B, T_new, D)
        Hidden states for the *new* positions only. During prefix
        initialisation, T_new == K (whole prefix); during stepping,
        T_new == 1.
    kv_cache : dict
        Mutated in place. Keys ``"k"`` and ``"v"`` hold the running
        cache of shape ``(B, num_heads, T_total, head_dim)``.
    """
    B, T, D = x.shape
    H = attn.num_heads
    dh = D // H

    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)  # (B, T, 3D)
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(B, T, H, dh).transpose(1, 2)
    k = k.view(B, T, H, dh).transpose(1, 2)
    v = v.view(B, T, H, dh).transpose(1, 2)

    if kv_cache.get("k") is None:
        full_k, full_v = k, v
    else:
        full_k = torch.cat([kv_cache["k"], k], dim=2)
        full_v = torch.cat([kv_cache["v"], v], dim=2)
    kv_cache["k"] = full_k
    kv_cache["v"] = full_v

    # is_causal=False: the cache is already strictly past relative to the
    # new queries, so causality is preserved without an explicit mask.
    out = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)


def _layer_forward_cached(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    kv_cache: dict,
) -> torch.Tensor:
    """One ``TransformerEncoderLayer`` step with KV cache (post-LN variant).

    Mirrors ``layer.forward`` for ``norm_first=False`` and skips the
    train-only dropouts (this code path is used in ``torch.no_grad``
    generation, where ``layer.eval()`` makes them no-ops anyway).
    """
    attn_out = _self_attn_cached(layer.self_attn, x, kv_cache)
    x = layer.norm1(x + attn_out)
    ffn_out = layer.linear2(layer.activation(layer.linear1(x)))
    x = layer.norm2(x + ffn_out)
    return x


class VAEDecoder(nn.Module):
    """Causal transformer decoder with latent prefix (KV cache injection).

    The latent *z* is a sequence of ``num_latent_tokens`` vectors that are
    each projected into ``embed_dim`` and **prepended** to the decoder
    sequence.  A custom attention mask ensures:

    * All positions can attend to the latent prefix tokens.
    * Decoder token positions obey causal (autoregressive) ordering.

    During training the decoder receives the target tokens shifted right
    (a learned ``<start>`` embedding is prepended, the last token is dropped).
    This forces each position to predict the next token using only previous
    tokens **and** the latent — the decoder cannot bypass *z*.

    Output shape: ``(B, L, embed_dim)``.  Does **not** produce logits; that
    is handled by :class:`OutputProjection`.
    """

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
        vocab_size: int,
        num_latent_tokens: int = 8,
        latent_pos_inject: bool = False,
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.embed_dim = embed_dim
        self.latent_pos_inject = latent_pos_inject

        # Decoder's own token embedding (separate from encoder — avoids
        # conflicting gradient streams, see Bug 8).
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Learned start-of-sequence embedding
        self.start_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding — applied only to decoder token positions,
        # NOT to latent prefix tokens.  This prevents PE from competing
        # with the latent signal (Bug 3).
        pe = _sinusoidal_encoding(max_answer_len, embed_dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

        # Project per-position latent (latent_dim → embed_dim). The K dimension
        # of z already matches num_latent_tokens — the encoder produces
        # exactly one latent vector per prefix slot.
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Learnable positional embedding for the K latent prefix slots so the
        # attention layers can distinguish slot order. Without this, slots are
        # only differentiated by content from the encoder's learned queries.
        self.prefix_pos_embed = nn.Parameter(
            torch.randn(1, num_latent_tokens, embed_dim) * 0.02
        )

        # Per-position latent injection. The KV-prefix alone lets a strong
        # causal decoder bypass z entirely (it can copy the teacher-forced
        # previous token), which drives posterior collapse on short answers.
        # Adding a K-pooled projection of z to *every* decoder token input
        # makes z reachable at each step regardless of attention, so the
        # decoder cannot ignore it. Separate projection from ``latent_proj``
        # (different role: a single global context vector, not per-slot KV).
        if self.latent_pos_inject:
            self.latent_context_proj = nn.Linear(latent_dim, embed_dim)
        else:
            self.latent_context_proj = None

        # Transformer encoder used with a custom causal+prefix mask so that
        # the architecture is effectively a causal decoder with KV injection.
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_attn_mask(self, K: int, L: int, device: torch.device) -> torch.Tensor:
        """Additive attention mask of shape ``(K+L, K+L)``.

        * Latent prefix rows (0 … K-1): attend to all prefix tokens, but
          **not** to decoder tokens (keeps prefix representations clean).
        * Decoder rows (K … K+L-1): attend to **all** prefix tokens plus
          causally to previous decoder tokens (position i sees K … K+i).

        Convention: 0 → attend, ``-inf`` → block.
        """
        total = K + L
        mask = torch.zeros(total, total, device=device)
        # Prefix tokens should not attend to decoder tokens
        mask[:K, K:] = float("-inf")
        # Causal mask for decoder positions — use bool then where to avoid
        # 0 * -inf = nan from triu multiplication.
        causal_bool = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
        )
        mask[K:, K:] = torch.where(causal_bool, float("-inf"), 0.0)
        return mask

    def _project_latent(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, K, latent_dim)`` → ``(B, K, embed_dim)``."""
        if z.dim() != 3 or z.size(1) != self.num_latent_tokens:
            raise ValueError(
                f"Decoder expects z of shape (B, {self.num_latent_tokens}, latent_dim); "
                f"got {tuple(z.shape)}"
            )
        return self.latent_proj(z)

    # ------------------------------------------------------------------
    # Forward (teacher-forced training)
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        word_dropout: float = 0.0,
        mask_token_id: int | None = None,
    ) -> torch.Tensor:
        """Teacher-forced decode: predict each token from previous tokens + z.

        Parameters
        ----------
        token_ids : Tensor (B, L) — target answer tokens.
        z         : Tensor (B, K, latent_dim) — sequence of latent vectors.
        mask      : Tensor (B, L) — 1 for real tokens, 0 for padding.
        word_dropout : float
            Probability of corrupting each teacher-forced *input* token by
            replacing it with ``mask_token_id`` (training only). The decoder
            then cannot lean on the previous ground-truth token and is forced
            to read the latent z to predict the target — the standard cure for
            latent bypass / posterior collapse in text VAEs (Bowman et al. 2016).
            Targets are unchanged, so the reconstruction objective is identical;
            only the decoder's *inputs* are degraded. ``0.0`` disables it.
        mask_token_id : int or None
            Token id used as the corruption symbol. Required when
            ``word_dropout > 0``.

        Returns
        -------
        Tensor (B, L, embed_dim)
        """
        B, L = token_ids.shape
        K = self.num_latent_tokens

        # Shift right: [<start>, tok_0, …, tok_{L-2}]
        in_ids = token_ids[:, :-1]
        if self.training and word_dropout > 0.0 and mask_token_id is not None:
            # Corrupt a random subset of the input tokens (NOT the targets).
            # Only real (non-pad) positions are eligible so we don't waste
            # corruption budget on padding the decoder already ignores.
            in_mask = mask[:, :-1] > 0
            drop = (torch.rand_like(in_ids, dtype=torch.float) < word_dropout) & in_mask
            in_ids = torch.where(drop, torch.full_like(in_ids, mask_token_id), in_ids)
        tok_emb = self.token_embedding(in_ids)  # (B, L-1, D)
        start = self.start_embed.expand(B, -1, -1)  # (B, 1, D)
        dec_input = torch.cat([start, tok_emb], dim=1)  # (B, L, D)
        dec_input = dec_input + self.pe[:, :L, :]

        # Per-position latent injection: add a single K-pooled latent context
        # vector to every decoder token input (broadcasts over L).
        if self.latent_context_proj is not None:
            ctx = self.latent_context_proj(z.mean(dim=1, keepdim=True))  # (B, 1, D)
            dec_input = dec_input + ctx

        # Latent prefix (+ learnable slot positional embedding)
        prefix = self._project_latent(z) + self.prefix_pos_embed  # (B, K, D)

        # Concatenate: [prefix | decoder_tokens]
        x = torch.cat([prefix, dec_input], dim=1)  # (B, K+L, D)

        # Attention mask (float additive): causal + prefix blocking
        attn_mask = self._build_attn_mask(K, L, x.device)
        # Padding mask (bool): True = ignore. Latent prefix is never padded.
        latent_pad = torch.zeros(B, K, device=mask.device, dtype=torch.bool)
        # Shift mask to align with dec_input: <start> is always real, and
        # dec_input[i] = token_ids[i-1], so we take mask[:, :-1] as the tail.
        shifted_mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask[:, :-1]],
            dim=1,
        )
        dec_pad = shifted_mask == 0
        full_pad_mask = torch.cat([latent_pad, dec_pad], dim=1)

        x = self.transformer(
            x, mask=attn_mask, src_key_padding_mask=full_pad_mask, is_causal=False
        )

        # Return only the decoder positions
        return x[:, K:, :]  # (B, L, D)

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int,
        output_head: nn.Module,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate token ids from latent *z* with KV cache.

        Parameters
        ----------
        z          : (B, K, latent_dim) — sequence of latent vectors.
        max_len    : int — maximum tokens to generate.
        output_head: OutputProjection module (hidden → logits).
        strategy   : ``"greedy"`` or ``"nucleus"``.
        temperature: softmax temperature (nucleus only).
        top_p      : nucleus probability mass (nucleus only).
        eos_token_id: optional token id that terminates generation early. Once
            every batch element has emitted this id, the loop exits. Positions
            after the first emission are filled with the eos id.

        Returns
        -------
        Tensor (B, max_len)

        Notes
        -----
        The K latent prefix is processed once and its per-layer K/V are
        cached, then each decoder step only pushes the single new token
        through the network — overall complexity drops from O(N²) to O(N).
        Weights are read directly off ``self.transformer.layers`` so this
        path shares the trained parameters with the teacher-forced
        ``forward`` path.
        """
        B = z.size(0)
        device = z.device

        # ---- 1. Initialise the per-layer K/V cache from the prefix.
        prefix = self._project_latent(z) + self.prefix_pos_embed  # (B, K, D)
        layers = list(self.transformer.layers)
        caches: list[dict] = [{} for _ in layers]
        x = prefix
        for layer, kv in zip(layers, caches):
            x = _layer_forward_cached(layer, x, kv)
        # x is the prefix hidden state at the final layer — we don't need it.

        # Per-position latent context (mirrors teacher-forced ``forward``):
        # a single K-pooled vector added to every decoder token input.
        ctx = (
            self.latent_context_proj(z.mean(dim=1, keepdim=True))
            if self.latent_context_proj is not None
            else None
        )  # (B, 1, D) or None

        # ---- 2. First decoder token: <start> + PE[0].
        dec_input = self.start_embed.expand(B, -1, -1) + self.pe[:, :1, :]  # (B, 1, D)
        if ctx is not None:
            dec_input = dec_input + ctx

        generated: list[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            # Push only the new token through each layer; the cache supplies
            # all earlier keys/values (prefix + previously generated tokens).
            x = dec_input
            for layer, kv in zip(layers, caches):
                x = _layer_forward_cached(layer, x, kv)
            logits = output_head(x).squeeze(1)  # (B, V)

            if strategy == "greedy":
                next_id = logits.argmax(dim=-1)  # (B,)
            else:  # nucleus
                scaled = logits / max(temperature, 1e-8)
                probs = F.softmax(scaled, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                cutoff = (cumsum - sorted_probs) > top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                sampled = torch.multinomial(sorted_probs, 1).squeeze(-1)
                next_id = sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            if eos_token_id is not None:
                next_id = torch.where(
                    finished, torch.full_like(next_id, eos_token_id), next_id
                )
                finished = finished | (next_id == eos_token_id)

            generated.append(next_id)

            if eos_token_id is not None and bool(finished.all().item()):
                remaining = max_len - (step + 1)
                if remaining > 0:
                    pad = torch.full(
                        (B,), eos_token_id, dtype=next_id.dtype, device=device
                    )
                    generated.extend([pad] * remaining)
                break

            # Prepare the next decoder token: token embedding + positional
            # (+ the same per-position latent context as the prefix step).
            next_pos = step + 1
            dec_input = (
                self.token_embedding(next_id).unsqueeze(1) + self.pe[:, next_pos : next_pos + 1, :]
            )
            if ctx is not None:
                dec_input = dec_input + ctx

        return torch.stack(generated, dim=1)  # (B, max_len)
