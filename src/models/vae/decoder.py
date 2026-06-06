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
        return ["c_attn"]
    # Llama/Qwen/Mistral-style projection names.
    return ["q_proj", "v_proj"]


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
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.max_answer_len = max_answer_len
        self.latent_pos_inject = latent_pos_inject

        # --- Frozen pretrained causal LM ---
        lm = AutoModelForCausalLM.from_pretrained(model_name)
        lm.eval()
        lm.requires_grad_(False)

        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

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
        # Per-position injection: a single K-pooled context vector added to
        # every token embedding so z is reachable at each decode step.
        if latent_pos_inject:
            self.latent_context_proj = nn.Linear(latent_dim, self.hidden)
        else:
            self.latent_context_proj = None

    # ------------------------------------------------------------------
    def _input_embeddings(self) -> nn.Module:
        return self.lm.get_input_embeddings()

    def _prefix(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, K, latent_dim)`` → ``(B, K, hidden)`` prefix embeddings."""
        if z.dim() != 3 or z.size(1) != self.num_latent_tokens:
            raise ValueError(
                f"Decoder expects z of shape (B, {self.num_latent_tokens}, latent_dim); "
                f"got {tuple(z.shape)}"
            )
        return self.latent_proj(z) + self.prefix_pos_embed

    def _context(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.latent_context_proj is None:
            return None
        return self.latent_context_proj(z.mean(dim=1, keepdim=True))  # (B, 1, H)

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
        K = self.num_latent_tokens

        in_ids = token_ids
        if self.training and word_dropout > 0.0 and mask_token_id is not None:
            real = mask > 0
            drop = (torch.rand_like(in_ids, dtype=torch.float) < word_dropout) & real
            rand_ids = torch.randint_like(in_ids, low=0, high=self.vocab_size)
            in_ids = torch.where(drop, rand_ids, in_ids)

        tok_emb = self._input_embeddings()(in_ids)  # (B, L, H)
        ctx = self._context(z)
        if ctx is not None:
            tok_emb = tok_emb + ctx

        prefix = self._prefix(z)  # (B, K, H)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # (B, K+L, H)

        attn = torch.cat(
            [torch.ones(B, K, dtype=mask.dtype, device=mask.device), mask], dim=1
        )
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
        logits = out.logits  # (B, K+L, V)
        # Position K-1 (last prefix token) predicts token 0; position K+i-1
        # predicts token i. Slice the L positions that predict tokens 0..L-1.
        return logits[:, K - 1 : K - 1 + L, :]

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
        input is just the K-token prefix, and the logits at the last prefix
        position predict the first token — exactly mirroring the teacher-forced
        slice in :meth:`forward`.
        """
        B = z.size(0)
        device = z.device
        ctx = self._context(z)
        prefix = self._prefix(z)  # (B, K, H)
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
            attn = torch.ones(inp.size(0), inp.size(1), dtype=torch.long, device=device)
            out = self.lm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
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
            logits = out.logits[:, -1, :]

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
