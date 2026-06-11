"""Eval-only pipeline: load a trained VAE checkpoint and run validation.

Reuses train_vae._validate so the reported metrics (loss terms, EM/F1, BLEU,
collapse probe) are computed exactly as during training — same μ-deterministic
encode, same free-running autoregressive decode.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.config.schema import Config
from src.models.vae.vae import SequenceVAE
from src.training.checkpoint import load_checkpoint
from src.pipelines.train_vae import _validate

logger = logging.getLogger(__name__)


def eval_vae(
    config: Config,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate a saved checkpoint on the validation split.

    Parameters
    ----------
    config : Config
        Must select the same dataset/arch the checkpoint was trained with.
    checkpoint_path : optional
        Defaults to ``<paths.checkpoint_dir>/vae_best.pt``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.config.seed import seed_everything

    seed_everything(config.seed)

    from src.data.tokenization import create_decoder_tokenizer, create_tokenizer
    from src.data.loaders import create_vae_dataloaders

    tokenizer = create_tokenizer(config.encoder.model_name)
    dec_tokenizer = create_decoder_tokenizer(config.decoder.model_name)
    _, val_loader = create_vae_dataloaders(
        config,
        tokenizer,
        null_train_fraction=config.vae_training.null_train_fraction,
    )

    vae = SequenceVAE(config, encoder_vocab_size=len(tokenizer)).to(device)

    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.checkpoint_dir) / "vae_best.pt"
    ckpt = load_checkpoint(checkpoint_path)
    # strict=False: the frozen pretrained backbones are reloaded from HF and
    # are not part of the trained state dict. The saved weights are already
    # the EMA weights (ema.apply() runs before save_checkpoint in train_vae).
    vae.load_state_dict(ckpt["model_state_dict"], strict=False)
    vae.eval()

    logger.info(
        "Loaded checkpoint %s (step=%s, saved metrics=%s)",
        checkpoint_path, ckpt.get("step"), ckpt.get("metrics"),
    )

    tc = config.vae_training
    metrics = _validate(
        vae, val_loader, device,
        beta=tc.beta_end, free_bits=tc.free_bits,
        target_kl=tc.target_kl, bow_weight=tc.bow_loss_weight,
        tokenizer=tokenizer,
        dec_tokenizer=dec_tokenizer,
    )
    metrics.pop("_samples", None)
    return metrics


if __name__ == "__main__":
    from src.config.loader import create_config_from_cli

    logging.basicConfig(level=logging.INFO)
    cfg = create_config_from_cli()
    metrics = eval_vae(cfg)
    print("Eval metrics:", metrics)
