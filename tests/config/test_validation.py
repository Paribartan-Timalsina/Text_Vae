"""Tests for config validation."""

import pytest

from src.config.schema import Config, VAEArchConfig, VAETrainingConfig
from src.config.validation import validate_config


class TestValidConfig:
    def test_valid_config_passes(self, tiny_config):
        validate_config(tiny_config)  # Should not raise

    def test_default_config_passes(self):
        validate_config(Config())


class TestDimensionValidation:
    def test_negative_latent_dim_fails(self):
        config = Config(vae_arch=VAEArchConfig(latent_dim=-1))
        with pytest.raises(ValueError, match="latent_dim"):
            validate_config(config)

    def test_zero_max_answer_len_fails(self):
        config = Config(vae_arch=VAEArchConfig(max_answer_len=0))
        with pytest.raises(ValueError, match="max_answer_len"):
            validate_config(config)


class TestBetaValidation:
    def test_beta_start_ge_end_fails(self):
        config = Config(
            vae_training=VAETrainingConfig(beta_start=1.0, beta_end=0.5)
        )
        with pytest.raises(ValueError, match="beta_start"):
            validate_config(config)


class TestEMAValidation:
    def test_ema_decay_out_of_range_fails(self):
        config = Config(
            vae_training=VAETrainingConfig(ema_decay=1.0)
        )
        with pytest.raises(ValueError, match="ema_decay"):
            validate_config(config)
