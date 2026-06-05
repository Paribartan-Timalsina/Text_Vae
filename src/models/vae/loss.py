"""VAE loss utilities: reconstruction + KL with beta warmup."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_vae_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
    free_bits: float = 0.0,
    target_kl: Optional[float] = None,
    recon_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss = recon + beta * kl.

    Parameters
    ----------
    logits : (B, L, V)
    target_ids : (B, L)
    mask : (B, L) — 1 for real tokens, 0 for padding.
    mu, log_var : (B, K, latent_dim) — sequence of latent parameters.
    beta : float
    free_bits : float
        Minimum KL per (K, D) position (prevents full posterior collapse).
    target_kl : float or None
        KL ceiling.  The KL contribution to the loss is clamped to this
        value, so gradients stop flowing once KL exceeds the target.
        Set to None to disable (no ceiling, full KL is always penalized).
    recon_weights : (B,) or None
        Optional per-sequence weight on the reconstruction term. Used to
        downweight NULL (unanswerable) examples so the decoder's gradient
        focuses on real answer text. ``None`` weights every sequence equally.

    Returns
    -------
    (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction: cross-entropy ignoring padding.
    # Sum the token NLL per sequence and average over the BATCH (not over
    # tokens). This puts recon on the same scale as the KL term below, which
    # is summed over the (K, D) latent dims and averaged over the batch.
    # A per-token mean instead leaves recon ~100x smaller than the KL sum,
    # so once beta reaches 1.0 the cheapest minimum is to crush the posterior
    # to the prior (posterior collapse). Matching scales makes beta=1 the
    # true sequence-ELBO.
    B, _, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    target_flat = target_ids.reshape(-1)

    # Per-sequence NLL (sum over real tokens), then averaged over the batch.
    ce = F.cross_entropy(logits_flat, target_flat, reduction="none").view(B, -1)
    per_seq = (ce * mask.float()).sum(dim=1)  # (B,)
    if recon_weights is not None:
        # Downweight selected sequences (e.g. NULL examples) before averaging.
        per_seq = per_seq * recon_weights.to(per_seq.dtype)
    recon = per_seq.sum() / B

    # KL: latent is (B, K, D). Clamp per-sample per-(K,D) BEFORE averaging
    # over the batch so individual samples cannot collapse a dimension
    # while the batch mean stays above the free-bits floor (Kingma 2016).
    # mean(dim=0) → (K, D); .sum() then collapses K and D into one scalar.
    kl_raw = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, K, D)
    if free_bits > 0.0:
        # Per-SAMPLE per-dim clamp BEFORE the batch mean (Kingma 2016 free bits).
        # The batch-mean-then-clamp variant let individual samples/dims fall
        # below the floor (their gradient is only zeroed at the aggregate), so
        # the optimizer drove the *true* KL toward zero whenever the decoder
        # ignored z — the partial posterior collapse observed here (true_kl
        # falling 73→24 while recon stayed floored). Clamping per sample enforces
        # a hard ≥ free_bits floor on every (b,k,d): true_kl cannot collapse
        # below free_bits * K * D, keeping the latent alive long enough for the
        # decoder to learn to use it.
        kl = kl_raw.clamp(min=free_bits).mean(dim=0).sum()
    else:
        kl = kl_raw.mean(dim=0).sum()

    # Target KL behaviour: previously this clamped KL contribution at
    # ``target_kl`` so gradients went to zero above the ceiling. That silently
    # disabled KL regularisation for the entire run once exceeded. Switched to
    # a soft hinge: penalise only the *excess* over the target, with full
    # gradient flow everywhere. Below the target the KL term vanishes
    # (free-budget); above it, gradient direction pushes KL back down.
    if target_kl is not None:
        kl_for_loss = torch.relu(kl - target_kl)
    else:
        kl_for_loss = kl
    total = recon + beta * kl_for_loss
    return total, recon, kl


def compute_bow_loss(
    bow_logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Bag-of-words auxiliary loss (Zhao et al. 2017).

    Forces the latent *z* to encode *which* tokens appear in the answer,
    independent of their order. ``bow_logits`` is a single per-sequence
    distribution over the vocabulary predicted from *z* alone (no decoder,
    no teacher forcing). Each real target token must be assigned high
    probability under that distribution, so the encoder cannot collapse the
    posterior without paying here — the standard, reliable cure for latent
    bypass / posterior collapse in text VAEs.

    Parameters
    ----------
    bow_logits : (B, V) — order-agnostic vocab logits predicted from z.
    target_ids : (B, L) — answer token ids.
    mask : (B, L) — 1 for real tokens, 0 for padding.

    Returns
    -------
    Scalar loss, summed over real tokens per sequence and averaged over the
    batch — the same reduction as the reconstruction term, so the BoW weight
    is comparable to ``beta``.
    """
    B = target_ids.size(0)
    log_probs = F.log_softmax(bow_logits, dim=-1)  # (B, V)
    # Gather the log-prob assigned to each target token: (B, L).
    tok_log_probs = log_probs.gather(1, target_ids)
    nll = -(tok_log_probs * mask.float()).sum() / B
    return nll


def compute_beta(
    step: int,
    start: float,
    end: float,
    warmup_steps: int,
) -> float:
    """Linear beta warmup from *start* to *end* over *warmup_steps*.

    Returns *start* at step 0 and *end* at step >= warmup_steps.
    """
    if warmup_steps <= 0:
        return end
    t = min(step / warmup_steps, 1.0)
    return start + (end - start) * t


def compute_cyclical_beta(
    step: int,
    total_steps: int,
    start: float,
    end: float,
    n_cycles: int = 4,
    ratio: float = 0.5,
) -> float:
    """Cyclical beta annealing (Fu et al. 2019).

    Within each cycle, beta linearly ramps from *start* to *end* over
    the first *ratio* fraction, then stays at *end* for the remainder.

    Parameters
    ----------
    step : int
        Current training step (0-indexed).
    total_steps : int
        Total number of training steps.
    start : float
        Beta value at the beginning of each cycle.
    end : float
        Beta value at the end of the ramp.
    n_cycles : int
        Number of annealing cycles over the full run.
    ratio : float
        Fraction of each cycle spent ramping (0 < ratio <= 1).
    """
    if total_steps <= 0 or n_cycles <= 0:
        return end
    cycle_len = total_steps / n_cycles
    tau = (step % cycle_len) / cycle_len
    if tau < ratio:
        t = tau / ratio
        return start + (end - start) * t
    return end
