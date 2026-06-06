"""Export precomputed latents from a trained VAE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.config.schema import Config

logger = logging.getLogger(__name__)


def export_latents(
    config: Config,
    vae_checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> None:
    """Encode the full SQuAD dataset with a frozen VAE and save latent files.

    Encoding is deterministic (uses mu only, no sampling).

    Steps
    -----
    1. Load frozen VAE from checkpoint.
    2. Encode train split, compute per-position per-dim normalisation stats.
    3. Encode val split.
    4. Run quality gate on the val split; raise RuntimeError if it fails.
    5. Save ``latent_dataset_train.pt``, ``latent_dataset_val.pt``,
       and ``normalization_stats.pt`` into ``config.paths.latent_dir``.

    Parameters
    ----------
    config : Config
    vae_checkpoint_path : str
        Path to a ``.pt`` checkpoint (SequenceVAE) or a directory (LangVAE).
    device : torch.device, optional
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _export_latents_sequence_vae(config, str(vae_checkpoint_path), device)


# ============================================================= SequenceVAE path


def _export_latents_sequence_vae(
    config: Config,
    vae_checkpoint_path: str,
    device: torch.device,
) -> None:
    from src.models.vae.vae import SequenceVAE
    from src.training.checkpoint import load_checkpoint
    from src.config.schema import Config as CfgCls

    ckpt = load_checkpoint(vae_checkpoint_path)
    saved_cfg = CfgCls.from_dict(ckpt["config"])

    from src.data.tokenization import create_tokenizer
    from src.data.loaders import create_squad_dataloaders

    tokenizer = create_tokenizer(saved_cfg.encoder.model_name)
    vocab_size = len(tokenizer)

    vae = SequenceVAE(saved_cfg, encoder_vocab_size=vocab_size).to(device)
    # strict=False: the frozen pretrained backbones are reloaded from HF at
    # construction, so the checkpoint only needs to supply the trained adapters
    # (Perceiver pool, variational heads, latent projections, LoRA). Any frozen
    # backbone keys present in the checkpoint are also accepted.
    vae.load_state_dict(ckpt["model_state_dict"], strict=False)
    vae.eval()

    train_loader, val_loader = create_squad_dataloaders(saved_cfg, tokenizer)

    def _encode_split(loader) -> dict:
        latents_list, context_ids_list, context_mask_list = [], [], []
        question_ids_list, question_mask_list, is_ans_list = [], [], []

        with torch.no_grad():
            for batch in loader:
                answer_ids = batch["answer_ids"].to(device)
                answer_mask = batch["answer_mask"].to(device)

                _, mu, _ = vae.encode(answer_ids, answer_mask, deterministic=True)
                latents_list.append(mu.cpu())

                context_ids_list.append(batch["context_ids"])
                context_mask_list.append(batch["context_mask"])
                question_ids_list.append(batch["question_ids"])
                question_mask_list.append(batch["question_mask"])
                is_ans = batch["is_answerable"]
                if not isinstance(is_ans, torch.Tensor):
                    is_ans = torch.tensor(is_ans, dtype=torch.bool)
                is_ans_list.append(is_ans.cpu())

        return {
            "latents_raw": torch.cat(latents_list, dim=0),
            "context_ids": torch.cat(context_ids_list, dim=0),
            "context_mask": torch.cat(context_mask_list, dim=0),
            "question_ids": torch.cat(question_ids_list, dim=0),
            "question_mask": torch.cat(question_mask_list, dim=0),
            "is_answerable": torch.cat(is_ans_list, dim=0),
        }

    logger.info("Encoding train split…")
    train_data = _encode_split(train_loader)

    logger.info("Encoding val split…")
    val_data = _encode_split(val_loader)

    _normalise_and_save(config, train_data, val_data)

    # Quality gate uses SequenceVAE-specific logit format
    from src.pipelines.quality_gate import run_quality_gate

    passed, report = run_quality_gate(vae, val_loader, saved_cfg, device)
    if not passed:
        failed = [k for k, v in report.items() if not v["passed"]]
        raise RuntimeError(f"Quality gate failed on checks: {failed}. Report: {report}")
    logger.info("Quality gate passed.")


# ================================================================ shared helpers


def _normalise_and_save(config: Config, train_data: dict, val_data: dict) -> None:
    """Compute normalisation stats from train, apply to both splits, save.

    Handles both 2D (N, D) and 3D (N, K, D) latent tensors.
    For 3D, computes stats across (N, K) samples, preserving (D,) shape for broadcasting.
    """
    train_latents = train_data["latents_raw"]
    # Handle both 2D and 3D shapes: always normalize across first axis (samples)
    if train_latents.ndim == 3:
        # (N, K, D) -> reshape to (N*K, D), compute stats, reshape back for broadcasting
        N, K, D = train_latents.shape
        train_flat = train_latents.view(N * K, D)
        norm_mean = train_flat.mean(dim=0)  # (D,)
        norm_std = train_flat.std(dim=0).clamp(min=1e-6)  # (D,)
    else:
        # (N, D) case
        norm_mean = train_latents.mean(dim=0)  # (D,)
        norm_std = train_latents.std(dim=0).clamp(min=1e-6)  # (D,)
    norm_stats = {"mean": norm_mean, "std": norm_std}

    def _normalize(data: dict) -> dict:
        z_norm = (data["latents_raw"] - norm_mean) / norm_std
        result = {"z_normalized": z_norm}
        result.update({k: v for k, v in data.items() if k != "latents_raw"})
        return result

    train_data = _normalize(train_data)
    val_data = _normalize(val_data)

    out_dir = Path(config.paths.latent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, out_dir / "latent_dataset_train.pt")
    torch.save(val_data, out_dir / "latent_dataset_val.pt")
    torch.save(norm_stats, out_dir / "normalization_stats.pt")
    logger.info("Saved latents to %s", out_dir)


if __name__ == "__main__":
    import argparse

    from src.config.loader import load_config

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export VAE latents")
    parser.add_argument(
        "--config", nargs="+", required=True,
        help="One or more YAML config files to load (merged in order)",
    )
    parser.add_argument(
        "--vae_checkpoint", required=True,
        help="Path to a trained SequenceVAE .pt checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_latents(cfg, args.vae_checkpoint)
