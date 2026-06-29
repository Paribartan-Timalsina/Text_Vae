"""Compute LangVAE-comparable disentanglement metrics for a trained VAE.

Loads a trained ``SequenceVAE`` checkpoint, encodes the SRL-annotated
EntailmentBank explanatory sentences (from ``saf-datasets``) to deterministic
latent means, and scores z-diff / z-min-var / informativeness via
:mod:`src.evaluation.disentanglement` — a faithful port of LangSpace's
disentanglement probe (the metrics in Table 1 of arXiv:2505.00004).

Usage (uses the same CLI/config plumbing as the other pipelines):

    .venv/bin/python -m scripts.eval_disentanglement \
        --config configs/base.yaml --config configs/vae/default.yaml \
        [--checkpoint checkpoints/vae_best.pt] [--sample-size 10000] [--seed 42]

The model's latent is ``(B, K, D)``; we mean-pool the K slots to one vector per
sentence (at K=1 this is a no-op — exactly LangVAE's single-vector setup).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# LangVAE's reported BERT→GPT-2 (no annotation) row, for side-by-side reference.
LANGVAE_BERT_GPT2 = {"z_diff": 0.46, "z_min_var": 0.68, "informativeness": 0.36}


def load_srl_sentences(sample_size: int | None) -> tuple[list[str], list[list[str]]]:
    """Load SRL-annotated EntailmentBank → (surfaces, per-token core-role lists).

    Uses the same explanatory-sentence corpus as LangVAE. Returns at most
    ``sample_size`` sentences.
    """
    from saf_datasets import EntailmentBankDataSet

    from src.evaluation.disentanglement import token_srl_role

    ds = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")
    surfaces: list[str] = []
    roles: list[list[str]] = []
    for sent in ds:
        surfaces.append(sent.surface)
        roles.append([token_srl_role(tok.annotations.get("srl", ["O"])) for tok in sent.tokens])
        if sample_size is not None and len(surfaces) >= sample_size:
            break
    return surfaces, roles


@torch.no_grad()
def encode_latents(
    vae,
    sentences: list[str],
    tokenizer,
    max_len: int,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Encode sentences to pooled deterministic latents ``(N, D)``.

    Tokenizes each sentence the same way training fed the encoder (BERT side,
    ``add_special_tokens=False`` + appended ``[SEP]``, padded to ``max_len``),
    takes the posterior mean ``mu`` ``(B, K, D)``, and mean-pools over K.
    """
    from src.data.squad_dataset import _tokenize_and_pad

    sep_id = tokenizer.sep_token_id
    out: list[torch.Tensor] = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        ids_list, mask_list = [], []
        for s in chunk:
            ids, mask = _tokenize_and_pad(tokenizer, s, max_len, add_special_tokens=False)
            if isinstance(sep_id, int):
                real = int(mask.sum().item())
                if real < max_len:
                    ids[real] = int(sep_id)
                    mask[real] = 1
            ids_list.append(ids)
            mask_list.append(mask)
        ids_b = torch.stack(ids_list).to(device)
        mask_b = torch.stack(mask_list).to(device)
        _, mu, _ = vae.encode(ids_b, mask_b, deterministic=True)  # (B, K, D)
        out.append(mu.mean(dim=1).cpu())  # pool K → (B, D)
    return torch.cat(out, dim=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help="Config YAML (repeatable).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: <ckpt_dir>/vae_best.pt).")
    parser.add_argument("--sample-size", type=int, default=10000, help="Max sentences to score.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/disentanglement.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    from src.config.loader import load_config
    from src.data.tokenization import create_tokenizer
    from src.models.vae.vae import SequenceVAE
    from src.training.checkpoint import load_checkpoint
    from src.evaluation.disentanglement import compute_disentanglement

    configs = args.config or ["configs/base.yaml", "configs/vae/default.yaml"]
    config = load_config(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model (mirror eval_vae.py loading) ---
    tokenizer = create_tokenizer(config.encoder.model_name)
    vae = SequenceVAE(config, encoder_vocab_size=len(tokenizer)).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(config.paths.checkpoint_dir) / "vae_best.pt"
    ckpt = load_checkpoint(ckpt_path)
    vae.load_state_dict(ckpt["model_state_dict"], strict=False)
    vae.eval()
    logger.info("Loaded checkpoint %s (step=%s)", ckpt_path, ckpt.get("step"))

    # --- data + encode ---
    surfaces, roles = load_srl_sentences(args.sample_size)
    logger.info("Loaded %d SRL-annotated sentences", len(surfaces))
    Z = encode_latents(vae, surfaces, tokenizer, config.encoder.max_context_len, device)
    logger.info("Encoded latents: %s (K pooled to one vector)", tuple(Z.shape))

    # --- metrics ---
    results = compute_disentanglement(Z, roles, seed=args.seed)

    # --- report ---
    print("\n=== Disentanglement (ours, faithful LangSpace port) ===")
    print(f"{'metric':<18}{'ours (mean±std)':<22}{'LangVAE BERT→GPT-2':<20}{'better'}")
    rows = [
        ("z_diff", "↑"),
        ("z_min_var", "↓"),
        ("informativeness", "↑"),
        ("disentanglement", "↑"),
        ("completeness", "↑"),
    ]
    for name, arrow in rows:
        if name not in results:
            continue
        mean, std = results[name]
        ref = LANGVAE_BERT_GPT2.get(name)
        ref_s = f"{ref:.2f}" if ref is not None else "—"
        print(f"{name:<18}{f'{mean:.3f} ± {std:.3f}':<22}{ref_s:<20}{arrow}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("metric,mean,std,langvae_bert_gpt2\n")
        for name, _ in rows:
            if name not in results:
                continue
            mean, std = results[name]
            ref = LANGVAE_BERT_GPT2.get(name, "")
            f.write(f"{name},{mean},{std},{ref}\n")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
