"""Shared test fixtures for the entire test suite."""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.config.schema import (
    Config,
    PathConfig,
    EncoderConfig,
    DecoderConfig,
    VAEArchConfig,
    VAETrainingConfig,
    QualityGateConfig,
)

# Tiny vocab used by the synthetic fake VAE / fake loaders below.
FAKE_VOCAB = 100


class FakeVAE(nn.Module):
    """Lightweight stand-in for ``SequenceVAE`` implementing the post-rewrite
    interface (no BERT/GPT-2 downloads). Pipeline tests inject this so they can
    exercise the real ``train_vae`` / ``quality_gate`` plumbing offline.

    Matches the real signatures:
      * ``encode(ids, mask, deterministic)`` → ``(z, mu, log_var)`` ``(B, K, D)``
      * ``forward(enc_ids, enc_mask, dec_ids, dec_mask, **kw)`` →
        ``(logits, z, mu, log_var, loss_dict)`` with logits over the decoder vocab
      * ``decode_to_tokens(z, ...)`` → ``(B, max_len)`` ids
    """

    def __init__(self, config: Config, vocab: int = FAKE_VOCAB) -> None:
        super().__init__()
        self.config = config
        arch = config.vae_arch
        self.K = arch.num_latent_tokens
        self.D = arch.latent_dim
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, self.D)
        self.queries = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.mu_head = nn.Linear(self.D, self.D)
        self.logvar_head = nn.Linear(self.D, self.D)
        self.dec_emb = nn.Embedding(vocab, self.D)
        self.dec = nn.Linear(self.D, vocab)

    def encode(self, ids, mask, deterministic: bool = False):
        h = self.emb(ids).mean(dim=1, keepdim=True) + self.queries.unsqueeze(0)  # (B,K,D)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-6.0, 4.0)
        if deterministic:
            z = mu
        else:
            z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        return z, mu, log_var

    def _logits(self, dec_ids, z):
        # Teacher-forced: predict each token from the shifted previous token +
        # K-pooled latent context (so loss can actually decrease in tests).
        shifted = torch.cat([torch.zeros_like(dec_ids[:, :1]), dec_ids[:, :-1]], dim=1)
        h = self.dec_emb(shifted) + z.mean(dim=1, keepdim=True)
        return self.dec(h)

    def forward(
        self, enc_ids, enc_mask, dec_ids, dec_mask, beta: float = 1.0,
        free_bits: float = 0.0, target_kl=None, recon_weights=None,
        bow_weight: float = 0.0, zforce_weight: float = 0.0,
        word_dropout: float = 0.0, mask_token_id=None, noise_aug_sigma: float = 0.0,
    ):
        from src.models.vae.loss import compute_vae_loss

        z, mu, log_var = self.encode(enc_ids, enc_mask)
        logits = self._logits(dec_ids, z)
        total, recon, kl = compute_vae_loss(
            logits, dec_ids, dec_mask, mu, log_var, beta, free_bits, target_kl,
            recon_weights=recon_weights,
        )
        zero = total.new_zeros(())
        return logits, z, mu, log_var, {
            "total": total, "recon": recon, "kl": kl,
            "bow": zero, "recon_zonly": zero,
        }

    def decode_to_tokens(self, z, strategy: str = "greedy", max_len=None,
                         eos_token_id=None, **kw):
        if max_len is None:
            max_len = self.config.decoder.max_answer_len
        B = z.size(0)
        ids = torch.zeros(B, 1, dtype=torch.long, device=z.device)
        for _ in range(max_len):
            logits = self._logits(ids, z)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids[:, 1:]  # drop the seed token


def make_fake_vae(config: Config, vocab: int = FAKE_VOCAB) -> FakeVAE:
    return FakeVAE(config, vocab)


@pytest.fixture
def tiny_config() -> Config:
    """Minimal config with small dims for fast CPU tests."""
    return Config(
        seed=42,
        paths=PathConfig(),
        encoder=EncoderConfig(
            model_name="bert-base-uncased",
            hidden_dim=64,
            max_context_len=32,
            max_question_len=16,
            unfreeze_top_n=0,
        ),
        decoder=DecoderConfig(
            model_name="gpt2",
            max_answer_len=10,
            latent_pos_inject=True,
            use_lora=False,
        ),
        vae_arch=VAEArchConfig(
            latent_dim=16,
            embed_dim=64,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_answer_len=10,
            num_latent_tokens=2,
        ),
        vae_training=VAETrainingConfig(
            learning_rate=1e-3,
            batch_size=4,
            epochs=2,
            patience=2,
            warmup_steps=10,
            weight_decay=0.01,
            grad_clip_max_norm=1.0,
            grad_accum_steps=1,
            beta_start=0.0,
            beta_end=1.0,
            beta_warmup_steps=50,
            free_bits=0.0,
            target_kl=None,
            val_every_n_steps=10,
        ),
        quality_gate=QualityGateConfig(),
    )


@pytest.fixture
def dummy_tokenizer():
    """Pretrained tokenizer with [NULL_ANS] special token added."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[NULL_ANS]"]})
    return tokenizer


@pytest.fixture
def small_batch():
    """Small random batch (B=4) for shape tests."""
    B, seq_len, dim = 4, 10, 16
    return {
        "input_ids": torch.randint(0, 1000, (B, seq_len)),
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
        "latent": torch.randn(B, seq_len, dim),
    }
