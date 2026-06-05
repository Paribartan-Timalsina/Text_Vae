"""Cross-field validation for the Config object."""

from __future__ import annotations

from src.config.schema import Config


def validate_config(config: Config) -> None:
    """Validate cross-field constraints. Raises ValueError on failure."""
    # Positive dimensions
    if config.vae_arch.latent_dim <= 0:
        raise ValueError(f"latent_dim must be > 0, got {config.vae_arch.latent_dim}")
    if config.vae_arch.max_answer_len <= 0:
        raise ValueError(f"max_answer_len must be > 0, got {config.vae_arch.max_answer_len}")

    # Beta schedule ordering
    if config.vae_training.beta_start >= config.vae_training.beta_end:
        raise ValueError(
            f"beta_start ({config.vae_training.beta_start}) must be < "
            f"beta_end ({config.vae_training.beta_end})"
        )

    # EMA decay range
    ema = config.vae_training.ema_decay
    if not (0.0 < ema < 1.0):
        raise ValueError(f"ema_decay must be in (0, 1), got {ema}")
