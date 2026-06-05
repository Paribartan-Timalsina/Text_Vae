"""Quality gate: checks latent space quality before exporting latents."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from src.config.schema import Config


def run_quality_gate(
    vae,
    dataloader,
    config: Config,
    device: torch.device,
) -> Tuple[bool, dict]:
    """Evaluate latent quality and return a pass/fail report.

    Checks performed
    ----------------
    1. ``recon_accuracy`` — fraction of tokens correctly reconstructed.
    2. ``mean_kl`` — mean KL divergence across all positions/dims.
    3. ``active_dims`` — number of latent dims with variance > threshold.
    4. ``centroid_distance`` — L2 distance between answerable / unanswerable
       latent centroids (mean over sequence positions).

    Parameters
    ----------
    vae : SequenceVAE
        Model in eval mode.
    dataloader : DataLoader
        Provides batches with keys ``answer_ids``, ``answer_mask``,
        ``is_answerable``.
    config : Config
    device : torch.device

    Returns
    -------
    (all_passed, report)
        ``report`` maps check name -> ``{value, passed, threshold}``.
    """
    qg = config.quality_gate
    vae.eval()

    all_token_correct = 0
    all_token_total = 0
    all_kl: list[torch.Tensor] = []
    all_mu: list[torch.Tensor] = []  # (B, K, D) — sequence latent
    all_answerable: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            is_answerable = batch["is_answerable"].to(device)

            logits, z, mu, log_var, loss_dict = vae(answer_ids, answer_mask)

            # --- Token reconstruction accuracy ---
            pred_ids = logits.argmax(dim=-1)  # (B, L)
            correct = ((pred_ids == answer_ids) * answer_mask).sum()
            total = answer_mask.sum()
            all_token_correct += correct.item()
            all_token_total += total.item()

            # --- KL ---
            # mu, log_var are sequence-shaped: (B, K, D). mean over batch
            # then sum over (K, D) gives a single scalar per batch.
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, K, D)
            kl_batch = kl_per_dim.mean(dim=0).sum()  # scalar
            all_kl.append(kl_batch.cpu())

            # --- Collect mu for active dims + centroid distance ---
            all_mu.append(mu.cpu())
            all_answerable.append(is_answerable.cpu())

    recon_accuracy = all_token_correct / max(all_token_total, 1)
    mean_kl = float(torch.stack(all_kl).mean())

    mu_cat = torch.cat(all_mu, dim=0)  # (N, K, D)
    # Flatten K and D into a single feature axis for active-dim and centroid
    # statistics. A "dim" here is one (k, d) coordinate of the latent.
    mu_flat = mu_cat.reshape(mu_cat.size(0), -1)  # (N, K*D)
    ans_cat = torch.cat(all_answerable, dim=0)  # (N,)

    # Active dims: variance across samples
    variances = mu_flat.var(dim=0)  # (K*D,)
    active_dims = int((variances > qg.active_dim_variance_threshold).sum().item())

    # Per-slot active dims: catches single-slot collapse (some queries die while
    # others stay alive — invisible to the flattened count above).
    variances_per_slot = mu_cat.var(dim=0)  # (K, D)
    active_per_slot = (variances_per_slot > qg.active_dim_variance_threshold).sum(
        dim=-1
    )  # (K,)
    dead_slots = int((active_per_slot == 0).sum().item())
    min_active_in_any_slot = int(active_per_slot.min().item())

    # Centroid distance
    ans_mask = ans_cat.bool()
    unans_mask = ~ans_mask
    if ans_mask.sum() > 0 and unans_mask.sum() > 0:
        centroid_ans = mu_flat[ans_mask].mean(dim=0)
        centroid_unans = mu_flat[unans_mask].mean(dim=0)
        centroid_distance = float(torch.norm(centroid_ans - centroid_unans).item())
    else:
        centroid_distance = 0.0

    report = {
        "recon_accuracy": {
            "value": recon_accuracy,
            "passed": recon_accuracy >= qg.min_recon_accuracy,
            "threshold": qg.min_recon_accuracy,
        },
        "mean_kl": {
            "value": mean_kl,
            "passed": mean_kl >= qg.min_mean_kl,
            "threshold": qg.min_mean_kl,
        },
        "active_dims": {
            "value": active_dims,
            "passed": active_dims >= qg.min_active_dims,
            "threshold": qg.min_active_dims,
        },
        "dead_slots": {
            "value": dead_slots,
            "passed": dead_slots <= qg.max_dead_slots,
            "threshold": qg.max_dead_slots,
        },
        "min_active_in_any_slot": {
            "value": min_active_in_any_slot,
            "passed": min_active_in_any_slot >= qg.min_active_in_any_slot,
            "threshold": qg.min_active_in_any_slot,
        },
        "centroid_distance": {
            "value": centroid_distance,
            "passed": centroid_distance >= qg.min_centroid_distance,
            "threshold": qg.min_centroid_distance,
        },
    }

    all_passed = all(v["passed"] for v in report.values())
    return all_passed, report
