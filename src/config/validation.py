"""Cross-field validation for the VAE Config object."""

from __future__ import annotations

from src.config.schema import Config


def validate_config(config: Config) -> None:
    """Validate cross-field constraints. Raises ValueError on failure."""
    # Positive dimensions
    if config.vae_arch.latent_dim <= 0:
        raise ValueError(f"latent_dim must be > 0, got {config.vae_arch.latent_dim}")
    if config.vae_arch.max_answer_len <= 0:
        raise ValueError(
            f"max_answer_len must be > 0, got {config.vae_arch.max_answer_len}"
        )
    if config.vae_arch.num_latent_tokens <= 0:
        raise ValueError(
            f"num_latent_tokens must be > 0, got {config.vae_arch.num_latent_tokens}"
        )

    # Beta schedule ordering
    if config.vae_training.beta_start >= config.vae_training.beta_end:
        raise ValueError(
            f"beta_start ({config.vae_training.beta_start}) must be < "
            f"beta_end ({config.vae_training.beta_end})"
        )

    # Beta schedule type
    valid_schedules = ("monotonic", "cyclical")
    if config.vae_training.beta_schedule not in valid_schedules:
        raise ValueError(
            f"beta_schedule must be one of {valid_schedules}, "
            f"got '{config.vae_training.beta_schedule}'"
        )
