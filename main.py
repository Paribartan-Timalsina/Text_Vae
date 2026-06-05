"""Standalone VAE training entry point.

Run from inside this folder:

    python main.py --config configs/vae.yaml

Any config field can be overridden on the CLI with dot notation, e.g.:

    python main.py --config configs/vae.yaml --vae_training.bow_loss_weight 0
    python main.py --config configs/vae.yaml --vae_arch.num_latent_tokens 4
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure THIS folder (containing the trimmed ``src/`` package) is searched
# first, so ``import src...`` resolves to the standalone copy regardless of
# where the parent project lives on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.loader import create_config_from_cli  # noqa: E402
from src.pipelines.train_vae import train_vae  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = create_config_from_cli()
    print("Loaded config:")
    print(config)
    metrics = train_vae(config)
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
