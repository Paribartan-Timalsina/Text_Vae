"""VAE decoder: a FROZEN pretrained causal LM with latent prefix injection.

Replaces the old from-scratch transformer (which could not produce fluent text
on a small corpus). Following LangVAE (arXiv:2505.00004), the generator is a
pretrained ``AutoModelForCausalLM`` (e.g. GPT-2) whose weights are FROZEN — only
the latent-injection projections (and optional LoRA adapters) are trained. The
latent ``z`` (a sequence of ``num_latent_tokens`` vectors) is projected into K
soft-prompt prefix embeddings prepended to the decoder input, and (optionally) a
K-pooled context vector is added to every token embedding so ``z`` is reachable
at every step and cannot be bypassed.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _lora_target_modules(model_name: str) -> list[str]:
    """Reasonable default LoRA target modules per decoder family."""
    name = model_name.lower()
    if "gpt2" in name or "gpt-2" in name:
        # Attention QKV + output proj + MLP (c_proj matches both attn.c_proj
        # and mlp.c_proj). Wider coverage than attention-only so the decoder
        # has enough trainable capacity to integrate the injected latent.
        return ["c_attn", "c_proj", "c_fc"]
    # Llama/Qwen/Mistral-style projection names.
    return ["q_proj", "v_proj"]


def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
    """Rotate the last dim by half (RoPE convention, matches transformers)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _find_rotary_emb(module: nn.Module):
    """Recursively find the first ``rotary_emb`` submodule (works through the
    PEFT/LoRA wrapper; all layers share one rotary in modern transformers).
    Returns None if not found."""
    rot = getattr(module, "rotary_emb", None)
    if rot is not None:
        return rot
    for child in module.children():
        found = _find_rotary_emb(child)
        if found is not None:
            return found
    return None


def _decoder_uses_rope(cfg) -> bool:
    """True if the decoder uses rotary position embeddings (RoPE).

    RoPE decoders (Llama/Qwen/Mistral/…) rotate keys and queries by position, so
    raw un-rotated KV injected via ``past_key_values`` is misaligned with the
    rotated queries and the latent becomes unreadable. Such decoders need the
    prefix-fan-out path instead. GPT-2 (absolute positions) returns False.
    """
    if getattr(cfg, "rope_theta", None) is not None:
        return True
    if getattr(cfg, "rope_scaling", None) is not None:
        return True
    mt = (getattr(cfg, "model_type", "") or "").lower()
    return mt in {
        "llama", "qwen2", "qwen2_moe", "qwen3", "mistral", "mixtral",
        "gemma", "gemma2", "phi", "phi3", "gptneox", "falcon",
    }


def _build_cache(layers: tuple) -> object:
    """Wrap legacy ``((k, v), ...)`` layers in a ``Cache`` object when the
    installed transformers requires it; fall back to the raw tuple."""
    try:
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(layers)
    except (ImportError, AttributeError):
        return layers


class VAEDecoder(nn.Module):
    """Frozen causal-LM decoder with K-token latent prefix injection.

    Output of :meth:`forward` is logits ``(B, L, vocab_size)`` over the decoder
    tokenizer's vocabulary (the pretrained LM head produces them directly — no
    separate output projection).
    """

    def __init__(
        self,
        model_name: str,
        latent_dim: int,
        num_latent_tokens: int,
        max_answer_len: int,
        latent_pos_inject: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        deep_inject: bool = False,
        kv_fanout_len: int = 0,
        torch_dtype: str = "float32",
        load_in_4bit: bool = False,
        device_map: str | None = None,
        fanout_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.max_answer_len = max_answer_len
        self.latent_pos_inject = latent_pos_inject

        # Compute dtype for the LM (and the dtype the injected latent tensors are
        # cast to). bfloat16/float16 keep large decoders (Mistral/Llama) in memory;
        # the trained heads stay fp32 and are cast at injection time.
        self.compute_dtype = getattr(torch, torch_dtype)

        # --- Frozen pretrained causal LM (optionally 4-bit / reduced precision) ---
        load_kwargs: dict = {"torch_dtype": self.compute_dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )
        lm = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        lm.eval()
        lm.requires_grad_(False)

        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            if load_in_4bit:
                from peft import prepare_model_for_kbit_training

                lm = prepare_model_for_kbit_training(lm)
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=_lora_target_modules(model_name),
            )
            lm = get_peft_model(lm, lora_cfg)
            logger.info("Attached LoRA adapters to decoder %s", model_name)

        self.lm = lm
        cfg = lm.config
        self.hidden = cfg.hidden_size
        self.vocab_size = cfg.vocab_size

        # --- Trained latent-injection layers ---
        # K latent vectors → K prefix embeddings at the LM hidden size.
        self.latent_proj = nn.Linear(latent_dim, self.hidden)
        # Learnable positional embedding distinguishing the K prefix slots.
        self.prefix_pos_embed = nn.Parameter(
            torch.randn(1, num_latent_tokens, self.hidden) * 0.02
        )
        # Learned "blank" embedding used to replace word-dropped INPUT tokens
        # (cleaner than random vocab ids: a neutral mask, not misleading noise).
        # At high word_dropout this makes teacher forcing behave like a masked
        # autoencoder so the decoder must reconstruct from z.
        self.mask_embed = nn.Parameter(torch.randn(1, 1, self.hidden) * 0.02)
        # Per-position injection: a single K-pooled context vector added to
        # every token embedding so z is reachable at each decode step.
        if latent_pos_inject:
            self.latent_context_proj = nn.Linear(latent_dim, self.hidden)
        else:
            self.latent_context_proj = None

        # Deep injection (Optimus / LangVAE style): fan the latent into extra
        # read-points the decoder attends to. Two routes, chosen by fanout_mode:
        #   * "kv"     — raw per-layer key/value memory via past_key_values.
        #                Correct for ABSOLUTE-position decoders (GPT-2).
        #   * "prefix" — the fan-out is projected into soft-prompt PREFIX
        #                embeddings prepended to the input, so the decoder applies
        #                its OWN (rotary) position encoding to them. Required for
        #                ROTARY (RoPE) decoders, where raw un-rotated injected KV
        #                is misaligned with the rotated queries → latent unreadable.
        #   * "auto"   — "prefix" if the decoder uses RoPE, else "kv".
        resolved_mode = fanout_mode
        if fanout_mode == "auto":
            resolved_mode = "prefix" if _decoder_uses_rope(cfg) else "kv"
        elif fanout_mode not in ("kv", "prefix"):
            raise ValueError(f"fanout_mode must be auto/kv/prefix; got {fanout_mode!r}")

        # RoPE + kv mode: the injected keys must be rotated to their cache
        # positions to align with the decoder's rotated queries (raw un-rotated
        # keys are unreadable under RoPE). Fetch the model's own rotary module
        # (handles rope_scaling correctly); if it can't be found, fall back to the
        # safe prefix path. Stored in a list so nn.Module doesn't re-register it.
        self.rope_kv = False
        self._rotary_emb: list = []
        if resolved_mode == "kv" and _decoder_uses_rope(cfg):
            rot = _find_rotary_emb(self.lm)
            if rot is not None:
                self.rope_kv = True
                self._rotary_emb = [rot]
                logger.info("RoPE-aware KV fan-out enabled for %s", model_name)
            else:
                logger.warning(
                    "RoPE decoder %s: rotary_emb not found; falling back to prefix "
                    "fan-out (KV injection would be unreadable).", model_name,
                )
                resolved_mode = "prefix"

        self.kv_fanout = deep_inject and kv_fanout_len > 0 and resolved_mode == "kv"
        self.prefix_fanout = deep_inject and kv_fanout_len > 0 and resolved_mode == "prefix"
        # KV memory length (only meaningful in kv mode); prefix length includes
        # the K latent tokens plus (in prefix mode) the M fan-out tokens.
        self.kv_mem_len = kv_fanout_len if self.kv_fanout else num_latent_tokens
        self.fanout_prefix_len = kv_fanout_len if self.prefix_fanout else 0
        self.prefix_len = num_latent_tokens + self.fanout_prefix_len

        self.kv_proj = None
        self.kv_dropout = None
        self.fanout_prefix_proj = None
        self.fanout_dropout = None

        if self.prefix_fanout:
            # RoPE-safe fan-out: pool z → M prefix embeddings (M×hidden — tiny).
            self.fanout_prefix_proj = nn.Linear(latent_dim, kv_fanout_len * self.hidden)
            self.fanout_dropout = nn.Dropout(0.1)
            logger.info(
                "Fan-out via PREFIX (%d soft-prompt tokens) for RoPE decoder %s",
                kv_fanout_len, model_name,
            )
        elif deep_inject:
            # KV-mode: per-layer key/value memory. Fan-out (kv_fanout_len>0) →
            # one pooled latent → kv_mem_len KV slots/layer; else per-token (1/token).
            self.n_layer = getattr(cfg, "n_layer", None) or cfg.num_hidden_layers
            n_head = getattr(cfg, "n_head", None) or cfg.num_attention_heads
            n_kv_head = getattr(cfg, "num_key_value_heads", None) or n_head
            self.n_kv_head = n_kv_head
            self.kv_head_dim = self.hidden // n_head
            self.kv_proj = nn.Linear(
                latent_dim,
                self.n_layer * 2 * n_kv_head * self.kv_head_dim * self.kv_mem_len
                if self.kv_fanout
                else self.n_layer * 2 * n_kv_head * self.kv_head_dim,
            )
            self.kv_dropout = nn.Dropout(0.1) if self.kv_fanout else None
            kv_params = self.kv_proj.weight.numel()
            if kv_params > 1_500_000_000:
                logger.warning(
                    "kv_proj has %.1fB params (kv_fanout_len=%d on %s). That is very "
                    "large — consider lowering decoder.kv_fanout_len.",
                    kv_params / 1e9, self.kv_mem_len, model_name,
                )
            else:
                logger.info("kv_proj: %.0fM params (kv_fanout_len=%d)", kv_params / 1e6, self.kv_mem_len)

    # ------------------------------------------------------------------
    def _input_embeddings(self) -> nn.Module:
        return self.lm.get_input_embeddings()

    def _prefix(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, K, latent_dim)`` → ``(B, prefix_len, hidden)`` prefix embeddings.

        Always emits the K latent soft-prompt tokens. In prefix-fan-out mode it
        additionally fans the pooled latent into ``fanout_prefix_len`` extra
        soft-prompt tokens (RoPE-safe: real input positions the decoder rotates
        itself), so ``prefix_len = K + fanout_prefix_len``.
        """
        if z.dim() != 3 or z.size(1) != self.num_latent_tokens:
            raise ValueError(
                f"Decoder expects z of shape (B, {self.num_latent_tokens}, latent_dim); "
                f"got {tuple(z.shape)}"
            )
        prefix = self.latent_proj(z) + self.prefix_pos_embed  # (B, K, H)
        if self.prefix_fanout:
            pooled = z.mean(dim=1)  # (B, latent_dim)
            fan = self.fanout_dropout(self.fanout_prefix_proj(pooled))  # (B, M*H)
            fan = fan.view(z.size(0), self.fanout_prefix_len, self.hidden)  # (B, M, H)
            prefix = torch.cat([prefix, fan], dim=1)  # (B, K+M, H)
        return prefix.to(self.compute_dtype)  # match the (possibly bf16) LM

    def _context(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.latent_context_proj is None:
            return None
        ctx = self.latent_context_proj(z.mean(dim=1, keepdim=True))  # (B, 1, H)
        return ctx.to(self.compute_dtype)

    def _past_kv(self, z: torch.Tensor) -> object | None:
        """``(B, K, latent_dim)`` → per-layer K-slot key/value memory.

        Returns a fresh cache each call (``Cache.update`` mutates, so it cannot
        be reused across forward passes), or ``None`` when deep injection is off.
        """
        if self.kv_proj is None:
            return None
        B, K, _ = z.shape
        if self.kv_fanout:
            # Pool the latent to a SINGLE vector and fan it into M=kv_mem_len KV
            # slots per layer (LangVAE-style). At K=1 the mean is just the vector.
            M = self.kv_mem_len
            pooled = z.mean(dim=1)  # (B, latent_dim)
            kv = self.kv_dropout(self.kv_proj(pooled))  # (B, n_layer*2*n_kv_head*M*hd)
            kv = kv.view(B, self.n_layer, 2, self.n_kv_head, M, self.kv_head_dim)
            kv = kv.permute(1, 2, 0, 3, 4, 5)  # (n_layer, 2, B, n_kv_head, M, hd)
        else:
            kv = self.kv_proj(z)  # (B, K, n_layer * 2 * n_kv_head * head_dim)
            kv = kv.view(B, K, self.n_layer, 2, self.n_kv_head, self.kv_head_dim)
            kv = kv.permute(2, 3, 0, 4, 1, 5)  # (n_layer, 2, B, n_kv_head, K, hd)
        kv = kv.to(self.compute_dtype)  # match the (possibly bf16) LM attention
        if self.rope_kv:
            # RoPE decoders rotate keys/queries by position. Our injected keys are
            # raw; rotate them to their cache positions 0..M-1 (the decoder auto-
            # positions the real tokens at M..) so they align with the rotated
            # queries. Only KEYS are rotated (RoPE never touches values). cos/sin
            # come from the model's own rotary module (handles rope_scaling).
            Mlen = kv.shape[4]
            pos = torch.arange(Mlen, device=kv.device).unsqueeze(0)  # (1, M)
            cos, sin = self._rotary_emb[0](kv, pos)  # (1, M, head_dim)
            cos = cos[0].to(kv.dtype)  # (M, head_dim) — broadcasts over layer/B/head
            sin = sin[0].to(kv.dtype)
            keys = kv[:, 0]  # (n_layer, B, n_kv_head, M, head_dim)
            keys = keys * cos + _rotate_half(keys) * sin
            kv = torch.stack([keys, kv[:, 1]], dim=1)  # rebuild (n_layer, 2, ...)
        layers = tuple(
            (kv[layer][0].contiguous(), kv[layer][1].contiguous())
            for layer in range(self.n_layer)
        )
        return _build_cache(layers)

    # ------------------------------------------------------------------ training
    def forward(
        self,
        token_ids: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        word_dropout: float = 0.0,
        mask_token_id: int | None = None,
    ) -> torch.Tensor:
        """Teacher-forced decode → logits ``(B, L, vocab_size)``.

        The K latent prefix is prepended to the L target-token embeddings; the
        causal LM predicts each token from the prefix + previous tokens. Logits
        are sliced so position ``i`` of the output predicts ``token_ids[:, i]``.

        ``word_dropout`` optionally replaces a random subset of *input* tokens
        with a **random vocab id** (Bowman et al. 2016) to discourage latent
        bypass; targets are unchanged. A random token (not a fixed mask id) is
        used because GPT-2 BPE has no ``[MASK]`` and ``eos`` would mislead the
        decoder into stopping. At ``word_dropout=1.0`` (the z-forcing pass) the
        entire input becomes noise, so the decoder must reconstruct from the
        z-prefix alone. ``mask_token_id`` is kept only as the on/off gate.
        """
        B, L = token_ids.shape
        P = self.prefix_len  # K (kv mode) or K + fan-out tokens (prefix mode)

        tok_emb = self._input_embeddings()(token_ids)  # (B, L, H)
        if self.training and word_dropout > 0.0 and mask_token_id is not None:
            # Blank a fraction of real input tokens with a LEARNED mask embedding
            # (not random vocab ids) so the decoder must reconstruct from z.
            real = mask > 0
            drop = (torch.rand_like(token_ids, dtype=torch.float) < word_dropout) & real
            tok_emb = torch.where(
                drop.unsqueeze(-1), self.mask_embed.to(tok_emb.dtype), tok_emb
            )
        ctx = self._context(z)
        if ctx is not None:
            tok_emb = tok_emb + ctx

        prefix = self._prefix(z)  # (B, P, H)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # (B, P+L, H)

        attn = torch.cat(
            [torch.ones(B, P, dtype=mask.dtype, device=mask.device), mask], dim=1
        )
        past = self._past_kv(z)  # None in prefix-fan-out mode
        if past is not None:
            # kv_mem_len extra always-visible memory slots per layer (= K for
            # per-token injection, = kv_fanout_len for KV fan-out); extend the mask.
            M = self.kv_mem_len
            attn = torch.cat(
                [torch.ones(B, M, dtype=mask.dtype, device=mask.device), attn], dim=1
            )
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            past_key_values=past,
            use_cache=past is not None,
        )
        logits = out.logits  # (B, P+L, V)
        # Position P-1 (last prefix token) predicts token 0; position P+i-1
        # predicts token i. Slice the L positions that predict tokens 0..L-1.
        return logits[:, P - 1 : P - 1 + L, :]

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate token ids from latent *z*.

        Recomputes ``[prefix | generated-so-far]`` each step (no KV-cache
        threading) so it is robust across transformers versions. At step 0 the
        input is just the ``prefix_len``-token prefix (K latent tokens, plus the
        fan-out tokens in prefix mode), and the logits at the last prefix position
        predict the first token — exactly mirroring the teacher-forced slice in
        :meth:`forward`. In prefix-fan-out mode ``_past_kv`` returns None, so
        ``attn_len`` reduces to ``inp.size(1)``.
        """
        B = z.size(0)
        device = z.device
        ctx = self._context(z)
        prefix = self._prefix(z)  # (B, prefix_len, H)
        embed = self._input_embeddings()

        generated: list[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cur_ids: torch.Tensor | None = None  # (B, t) tokens emitted so far

        for step in range(max_len):
            if cur_ids is None:
                inp = prefix
            else:
                tok_emb = embed(cur_ids)  # (B, t, H)
                if ctx is not None:
                    tok_emb = tok_emb + ctx
                inp = torch.cat([prefix, tok_emb], dim=1)  # (B, K+t, H)
            # Fresh cache each step (Cache.update mutates) — mirrors forward(),
            # where the K memory slots always precede the recomputed sequence.
            past = self._past_kv(z)
            attn_len = inp.size(1) + (self.kv_mem_len if past is not None else 0)
            attn = torch.ones(inp.size(0), attn_len, dtype=torch.long, device=device)
            out = self.lm(
                inputs_embeds=inp,
                attention_mask=attn,
                past_key_values=past,
                use_cache=past is not None,
            )
            logits = out.logits[:, -1, :]

            next_id = self._sample(logits, strategy, temperature, top_p)
            if eos_token_id is not None:
                next_id = torch.where(
                    finished, torch.full_like(next_id, eos_token_id), next_id
                )
                finished = finished | (next_id == eos_token_id)
            generated.append(next_id)
            if eos_token_id is not None and bool(finished.all().item()):
                remaining = max_len - (step + 1)
                if remaining > 0:
                    pad = torch.full((B,), eos_token_id, dtype=next_id.dtype, device=device)
                    generated.extend([pad] * remaining)
                break

            nxt = next_id.unsqueeze(1)
            cur_ids = nxt if cur_ids is None else torch.cat([cur_ids, nxt], dim=1)

        return torch.stack(generated, dim=1)  # (B, max_len)

    @staticmethod
    def _sample(
        logits: torch.Tensor, strategy: str, temperature: float, top_p: float
    ) -> torch.Tensor:
        if strategy == "greedy":
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        cutoff = (cumsum - sorted_probs) > top_p
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sampled = torch.multinomial(sorted_probs, 1).squeeze(-1)
        return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
