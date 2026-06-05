"""Decoding strategies for VAE output logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """Argmax decoding per position.

    Parameters
    ----------
    logits : (B, L, V)

    Returns
    -------
    token_ids : (B, L)
    """
    return logits.argmax(dim=-1)


def beam_search_decode(
    logits: torch.Tensor,
    beam_width: int,
    pad_id: int,
    eos_id: int,
) -> torch.Tensor:
    """Position-independent beam search over softmax probabilities.

    For each sample, maintains *beam_width* candidate sequences and keeps the
    highest-scoring complete beam.

    Parameters
    ----------
    logits : (B, L, V)
    beam_width : int
    pad_id : int
    eos_id : int

    Returns
    -------
    token_ids : (B, L)
    """
    B, L, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)

    results = []
    for b in range(B):
        # beam_seqs: (beam, positions_so_far)
        # beam_scores: (beam,)
        beam_seqs = torch.zeros(1, 0, dtype=torch.long, device=logits.device)
        beam_scores = torch.zeros(1, device=logits.device)

        for pos in range(L):
            lp = log_probs[b, pos]  # (V,)
            # Expand each beam with top-k tokens
            topk_scores, topk_ids = lp.topk(beam_width)  # (beam_width,)

            num_beams = beam_seqs.size(0)
            # All combinations: existing beams x new tokens
            expanded_scores = beam_scores.unsqueeze(1) + topk_scores.unsqueeze(0)  # (num_beams, beam_width)
            expanded_scores = expanded_scores.reshape(-1)

            # Indices
            beam_idx = torch.arange(num_beams, device=logits.device).unsqueeze(1).expand(-1, beam_width).reshape(-1)
            token_idx = topk_ids.unsqueeze(0).expand(num_beams, -1).reshape(-1)

            # Keep top beam_width
            k = min(beam_width, expanded_scores.size(0))
            top_scores, top_indices = expanded_scores.topk(k)

            prev_seqs = beam_seqs[beam_idx[top_indices]]
            new_tokens = token_idx[top_indices].unsqueeze(1)
            beam_seqs = torch.cat([prev_seqs, new_tokens], dim=1)
            beam_scores = top_scores

        # Return the best beam, truncated at first eos_id with remainder padded
        best = beam_scores.argmax()
        best_seq = beam_seqs[best].clone()
        eos_positions = (best_seq == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            best_seq[eos_positions[0].item() + 1:] = pad_id
        results.append(best_seq)

    return torch.stack(results, dim=0)


def nucleus_decode(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Top-p (nucleus) sampling.

    Parameters
    ----------
    logits : (B, L, V)
    top_p : float
    temperature : float

    Returns
    -------
    token_ids : (B, L)
    """
    B, L, V = logits.shape
    scaled = logits / max(temperature, 1e-8)

    results = []
    for pos in range(L):
        pos_logits = scaled[:, pos, :]  # (B, V)
        probs = F.softmax(pos_logits, dim=-1)

        # Sort descending
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        # Mask tokens whose cumulative prob exceeds top_p
        # Keep at least one token (the most probable)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0

        # Re-normalise
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
        # Map back to original vocab indices
        token_ids = sorted_indices.gather(dim=-1, index=sampled_idx).squeeze(-1)
        results.append(token_ids)

    return torch.stack(results, dim=1)  # (B, L)
