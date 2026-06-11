"""SequenceVAE — frozen pretrained encoder + frozen pretrained causal-LM decoder.

The encoder (e.g. BERT) and decoder (e.g. GPT-2) are pretrained and frozen; only
the Perceiver pool, the variational heads, the latent-injection projections, and
(optionally) LoRA adapters are trained. This is the LangVAE backbone, with this
repo's additions: a K-token *sequence* latent and per-position latent injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schema import Config
from .encoder import VAEEncoder
from .decoder import VAEDecoder
from .reparameterize import reparameterize
from .loss import compute_vae_loss, compute_bow_loss


class SequenceVAE(nn.Module):
    """Full sequence VAE over frozen pretrained backbones.

    Parameters
    ----------
    config : Config
        Full config (uses ``encoder``, ``decoder``, ``vae_arch`` sections).
    encoder_vocab_size : int
        ``len(encoder_tokenizer)`` — used to resize the frozen encoder's input
        embeddings for added special tokens like ``[NULL_ANS]``.
    """

    def __init__(self, config: Config, encoder_vocab_size: int) -> None:
        super().__init__()
        self.config = config
        arch = config.vae_arch

        self.encoder = VAEEncoder(
            model_name=config.encoder.model_name,
            latent_dim=arch.latent_dim,
            num_latent_tokens=arch.num_latent_tokens,
            num_heads=arch.num_heads,
            dropout=arch.dropout,
            vocab_size=encoder_vocab_size,
            unfreeze_top_n=config.encoder.unfreeze_top_n,
            pool_num_layers=arch.pool_num_layers,
        )
        self.decoder = VAEDecoder(
            model_name=config.decoder.model_name,
            latent_dim=arch.latent_dim,
            num_latent_tokens=arch.num_latent_tokens,
            max_answer_len=config.decoder.max_answer_len,
            latent_pos_inject=config.decoder.latent_pos_inject,
            use_lora=config.decoder.use_lora,
            lora_r=config.decoder.lora_r,
            lora_alpha=config.decoder.lora_alpha,
            lora_dropout=config.decoder.lora_dropout,
            deep_inject=config.decoder.deep_inject,
        )

        # Optional bag-of-words head over the DECODER vocabulary, predicted from
        # the K-pooled latent (anti-collapse signal independent of the decoder).
        if arch.use_bow_head:
            self.bow_head = nn.Linear(arch.latent_dim, self.decoder.vocab_size)
        else:
            self.bow_head = None

    # ------------------------------------------------------------------
    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode encoder-tokenized text → ``(z, μ, log_var)`` each ``(B, K, D)``."""
        mu, log_var = self.encoder(token_ids, mask)
        z = reparameterize(mu, log_var, deterministic=deterministic)
        return z, mu, log_var

    def decode(
        self,
        dec_token_ids: torch.Tensor,
        z: torch.Tensor,
        dec_mask: torch.Tensor,
        word_dropout: float = 0.0,
        mask_token_id: int | None = None,
    ) -> torch.Tensor:
        """Teacher-forced decode → logits ``(B, L, vocab_size)``."""
        return self.decoder(
            dec_token_ids, z, dec_mask,
            word_dropout=word_dropout, mask_token_id=mask_token_id,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        enc_ids: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_ids: torch.Tensor,
        dec_mask: torch.Tensor,
        beta: float = 1.0,
        free_bits: float = 0.0,
        target_kl: float | None = None,
        noise_aug_sigma: float = 0.0,
        recon_weights: torch.Tensor | None = None,
        word_dropout: float = 0.0,
        mask_token_id: int | None = None,
        bow_weight: float = 0.0,
        zforce_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full forward pass (teacher-forced).

        ``enc_ids``/``enc_mask`` are encoder-tokenized (BERT); ``dec_ids``/
        ``dec_mask`` are decoder-tokenized (GPT-2) and are the reconstruction
        target. Returns ``(logits, z, μ, log_var, loss_dict)``.
        """
        z, mu, log_var = self.encode(enc_ids, enc_mask)

        z_decode = z
        if noise_aug_sigma > 0.0 and self.training:
            z_decode = z + torch.randn_like(z) * noise_aug_sigma

        logits = self.decode(
            dec_ids, z_decode, dec_mask,
            word_dropout=word_dropout, mask_token_id=mask_token_id,
        )
        total, recon, kl = compute_vae_loss(
            logits, dec_ids, dec_mask, mu, log_var, beta, free_bits, target_kl,
            recon_weights=recon_weights,
        )

        bow = total.new_zeros(())
        if self.bow_head is not None and bow_weight > 0.0:
            bow_logits = self.bow_head(z.mean(dim=1))  # (B, V_dec)
            bow = compute_bow_loss(bow_logits, dec_ids, dec_mask)
            total = total + bow_weight * bow

        recon_zonly = total.new_zeros(())
        if zforce_weight > 0.0 and self.training and mask_token_id is not None:
            zonly_logits = self.decode(
                dec_ids, z_decode, dec_mask,
                word_dropout=1.0, mask_token_id=mask_token_id,
            )
            _, recon_zonly, _ = compute_vae_loss(
                zonly_logits, dec_ids, dec_mask, mu, log_var,
                beta=0.0, free_bits=0.0, target_kl=None,
                recon_weights=recon_weights,
            )
            total = total + zforce_weight * recon_zonly

        loss_dict = {
            "total": total, "recon": recon, "kl": kl,
            "bow": bow, "recon_zonly": recon_zonly,
        }
        return logits, z, mu, log_var, loss_dict

    # ------------------------------------------------------------------
    def decode_to_tokens(
        self,
        z: torch.Tensor,
        strategy: str = "greedy",
        max_len: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressively decode latent z → decoder token ids ``(B, max_len)``."""
        if max_len is None:
            max_len = self.config.decoder.max_answer_len
        return self.decoder.generate(
            z, max_len=max_len, strategy=strategy, eos_token_id=eos_token_id, **kwargs,
        )
