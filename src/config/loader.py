"""YAML config loading, merging, and CLI override support."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import Config
from src.config.validation import validate_config


def load_yaml(path: str | Path) -> dict:
    """Read a single YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively deep-merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_dot_notation(d: dict, key: str, value: Any) -> dict:
    """Apply a dot-notation key (e.g. 'vae_arch.latent_dim') to a nested dict."""
    parts = key.split(".")
    target = d
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    # Try to cast to int/float/bool if possible
    target[parts[-1]] = _cast_value(value)
    return d


def _cast_value(value: str) -> Any:
    """Attempt to cast a string CLI value to int, float, or bool."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    yaml_paths: list[str | Path],
    cli_overrides: dict[str, str] | None = None,
) -> Config:
    """Load multiple YAMLs in order, merge, apply CLI overrides, validate."""
    merged: dict = {}
    for path in yaml_paths:
        data = load_yaml(path)
        merged = merge_dicts(merged, data)

    if cli_overrides:
        for key, value in cli_overrides.items():
            merged = _apply_dot_notation(merged, key, value)

    config = Config.from_dict(merged)
    validate_config(config)
    return config


def create_config_from_cli(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments and return a validated Config."""
    parser = argparse.ArgumentParser(description="Latent Diffusion Text Model")
    parser.add_argument(
        "--config", nargs="+", required=True,
        help="One or more YAML config files to load (merged in order)",
    )
    args, unknown = parser.parse_known_args(argv)

    # Parse --key value pairs from unknown args
    cli_overrides: dict[str, str] = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i].lstrip("-")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                cli_overrides[key] = unknown[i + 1]
                i += 2
            else:
                cli_overrides[key] = "true"
                i += 1
        else:
            i += 1

    return load_config(args.config, cli_overrides)
