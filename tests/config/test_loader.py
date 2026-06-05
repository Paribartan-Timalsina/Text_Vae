"""Tests for config loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_yaml, merge_dicts, load_config, _apply_dot_notation


class TestLoadYaml:
    def test_load_single_yaml(self):
        data = load_yaml("configs/base.yaml")
        assert "seed" in data
        assert "encoder" in data

    def test_load_empty_yaml(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        data = load_yaml(empty)
        assert data == {}


class TestMergeDicts:
    def test_merge_override(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        result = merge_dicts(base, override)
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["a"] == 1

    def test_merge_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 2}


class TestDotNotation:
    def test_cli_dot_notation(self):
        d = {"vae_arch": {"latent_dim": 64}}
        _apply_dot_notation(d, "vae_arch.latent_dim", "128")
        assert d["vae_arch"]["latent_dim"] == 128

    def test_creates_nested_path(self):
        d = {}
        _apply_dot_notation(d, "a.b.c", "42")
        assert d["a"]["b"]["c"] == 42


class TestLoadConfig:
    def test_multi_yaml_merge(self):
        config = load_config(
            [
                "configs/base.yaml",
                "configs/vae/default.yaml",
            ]
        )
        assert config.vae_arch.latent_dim == 128

    def test_cli_overrides_applied(self):
        config = load_config(
            [
                "configs/base.yaml",
                "configs/vae/default.yaml",
            ],
            cli_overrides={"vae_arch.latent_dim": "256"},
        )
        assert config.vae_arch.latent_dim == 256
