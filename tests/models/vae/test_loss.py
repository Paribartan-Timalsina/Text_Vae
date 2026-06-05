"""Tests for VAE loss and beta schedule."""

import torch
import pytest

from src.models.vae.loss import compute_vae_loss, compute_beta, compute_bow_loss

BATCH_SIZE = 4
SEQ_LEN = 10
VOCAB_SIZE = 100
LATENT_DIM = 16


def test_padding_ignored():
    """Changing logits/targets at padding positions should not affect loss."""
    torch.manual_seed(0)
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, LATENT_DIM)

    # Mask: first 5 positions real, last 5 padding
    mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mask[:, :5] = 1

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    total1, recon1, _ = compute_vae_loss(logits, target, mask, mu, log_var, beta=1.0)

    # Change padding positions
    logits2 = logits.clone()
    logits2[:, 5:, :] = torch.randn(BATCH_SIZE, 5, VOCAB_SIZE) * 100
    target2 = target.clone()
    target2[:, 5:] = 0

    total2, recon2, _ = compute_vae_loss(logits2, target2, mask, mu, log_var, beta=1.0)

    assert torch.allclose(recon1, recon2)


def test_beta_warmup_schedule():
    assert compute_beta(0, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(0.0)
    assert compute_beta(50, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(0.5)
    assert compute_beta(100, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(1.0)
    assert compute_beta(200, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(1.0)


def test_perfect_prediction_low_loss():
    """One-hot logits matching targets should give near-zero recon loss."""
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, LATENT_DIM)

    # Create very confident logits
    logits = torch.full((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), -100.0)
    for b in range(BATCH_SIZE):
        for l in range(SEQ_LEN):
            logits[b, l, target[b, l]] = 100.0

    _, recon, _ = compute_vae_loss(logits, target, mask, mu, log_var, beta=1.0)
    assert recon.item() < 0.01


def test_free_bits_raises_kl_floor():
    """With free_bits > 0, the KL in the total loss should be at
    least free_bits * LATENT_DIM even when the posterior has collapsed.
    """
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, LATENT_DIM)
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    _, _, kl_vanilla = compute_vae_loss(
        logits, target, mask, mu, log_var, beta=1.0, free_bits=0.0
    )
    assert kl_vanilla.item() == pytest.approx(0.0, abs=1e-5)

    fb = 0.25
    _, _, kl_fb = compute_vae_loss(
        logits, target, mask, mu, log_var, beta=1.0, free_bits=fb
    )
    assert kl_fb.item() >= fb * LATENT_DIM - 1e-5


def test_bow_loss_padding_ignored():
    """Changing target ids at padding positions must not affect the BoW loss."""
    torch.manual_seed(0)
    bow_logits = torch.randn(BATCH_SIZE, VOCAB_SIZE)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mask[:, :5] = 1

    loss1 = compute_bow_loss(bow_logits, target, mask)

    target2 = target.clone()
    target2[:, 5:] = (target2[:, 5:] + 7) % VOCAB_SIZE  # perturb only padding
    loss2 = compute_bow_loss(bow_logits, target2, mask)

    assert torch.allclose(loss1, loss2)


def test_bow_loss_rewards_correct_tokens():
    """Putting probability mass on the target tokens lowers the BoW loss."""
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

    # Confident logits on each sequence's target tokens vs. uniform logits.
    confident = torch.full((BATCH_SIZE, VOCAB_SIZE), -10.0)
    for b in range(BATCH_SIZE):
        confident[b, target[b]] = 10.0
    uniform = torch.zeros(BATCH_SIZE, VOCAB_SIZE)

    good = compute_bow_loss(confident, target, mask)
    bad = compute_bow_loss(uniform, target, mask)
    assert good.item() < bad.item()


def test_bow_loss_gradient_flows_to_logits():
    """BoW loss must produce a gradient on the logits (i.e. on z upstream)."""
    bow_logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, requires_grad=True)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

    compute_bow_loss(bow_logits, target, mask).backward()
    assert bow_logits.grad is not None and bow_logits.grad.abs().sum() > 0
