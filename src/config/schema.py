"""Frozen dataclass definitions for the VAE configuration.

Trimmed from the parent project to contain only the sections the VAE
training path uses (paths, encoder, vae_arch, vae_training). Diffusion,
null-classifier, sampler and inference sections were removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for data, checkpoints, and outputs."""

    data_dir: str = "data"  # Root data directory
    checkpoint_dir: str = "checkpoints"  # Where to save model checkpoints
    output_dir: str = "outputs"  # Generation outputs and logs


@dataclass(frozen=True)
class EncoderConfig:
    """Pretrained encoder / tokenizer settings."""

    model_name: str = "bert-base-uncased"  # HuggingFace model identifier
    hidden_dim: int = 768  # Encoder hidden dimension
    max_context_len: int = 384  # Max context token length
    max_question_len: int = 64  # Max question token length
    unfreeze_top_n: int = 0  # Number of top layers to unfreeze


@dataclass(frozen=True)
class VAEArchConfig:
    """VAE architecture hyperparameters."""

    latent_dim: int = 128  # Latent space dimensionality
    embed_dim: int = 768  # Internal embedding dimension
    num_layers: int = 4  # Transformer layers in encoder/decoder
    num_heads: int = 8  # Attention heads
    dropout: float = 0.1  # Dropout rate
    max_answer_len: int = 50  # Max answer token length
    num_latent_tokens: int = 8  # Pseudo-tokens for latent KV injection
    latent_pos_inject: bool = False  # Add a K-pooled projection of z to every
    # decoder token input (not just the KV prefix). Stops the causal decoder
    # from bypassing z via teacher forcing — the main posterior-collapse cure.
    use_bow_head: bool = False  # Build a bag-of-words head that predicts the
    # answer's token set from z alone (Zhao et al. 2017). Trained via
    # ``vae_training.bow_loss_weight``; forces z to stay informative.


@dataclass(frozen=True)
class VAETrainingConfig:
    """VAE training hyperparameters."""

    learning_rate: float = 5e-4  # Peak learning rate
    batch_size: int = 64  # Training batch size
    epochs: int = 30  # Maximum training epochs
    patience: int = 5  # Early stopping patience (val checks)
    warmup_steps: int = 500  # LR scheduler warmup
    weight_decay: float = 0.01  # AdamW weight decay
    grad_clip_max_norm: float = 5.0  # Gradient clipping threshold
    grad_accum_steps: int = 1  # Gradient accumulation steps
    beta_start: float = 0.01  # KL weight at start
    beta_end: float = 1.0  # KL weight at end
    beta_warmup_steps: int = 10000  # Steps to ramp beta (monotonic schedule)
    beta_schedule: str = "cyclical"  # "monotonic" or "cyclical"
    beta_cycles: int = 40  # Number of cycles (cyclical only)
    target_kl: float | None = None  # KL hinge target (None = disabled)
    beta_cycle_ratio: float = 0.5  # Fraction of cycle spent ramping
    free_bits: float = 0.02  # Per-dim KL allowance (no penalty below this)
    ema_decay: float = 0.999  # EMA decay rate for validation weights
    val_every_n_steps: int = 500  # Validation frequency (steps)
    noise_aug_sigma: float = 0.0  # Extra Gaussian noise std added to z before decode
    noise_aug_prob: float = 0.0  # Per-step probability of applying noise aug
    null_train_fraction: float = 0.10  # Target fraction of NULL examples in train
    null_loss_weight: float = 0.1  # Reconstruction-loss weight for NULL examples
    word_dropout: float = 0.4  # Prob of replacing a teacher-forced decoder INPUT
    # token with [MASK] (Bowman 2016). Forces the decoder to read z. 0.0 disables.
    bow_loss_weight: float = 0.0  # Weight on the bag-of-words auxiliary loss
    # (requires vae_arch.use_bow_head). 0.0 disables. ~0.3-1.0 is typical.


@dataclass(frozen=True)
class Config:
    """Top-level configuration (VAE-only)."""

    seed: int = 42  # Global random seed
    paths: PathConfig = field(default_factory=PathConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vae_arch: VAEArchConfig = field(default_factory=VAEArchConfig)
    vae_training: VAETrainingConfig = field(default_factory=VAETrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        def _checked(dc_cls, data: dict) -> dict:
            valid = {f.name for f in fields(dc_cls)}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(
                    f"Unknown keys for {dc_cls.__name__}: {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid)}"
                )
            return data

        return cls(
            seed=d.get("seed", 42),
            paths=PathConfig(**_checked(PathConfig, d.get("paths", {}))),
            encoder=EncoderConfig(**_checked(EncoderConfig, d.get("encoder", {}))),
            vae_arch=VAEArchConfig(**_checked(VAEArchConfig, d.get("vae_arch", {}))),
            vae_training=VAETrainingConfig(
                **_checked(VAETrainingConfig, d.get("vae_training", {}))
            ),
        )
