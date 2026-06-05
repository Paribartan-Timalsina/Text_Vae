"""Tests for VAEDecoder (causal with latent prefix injection)."""

import torch
import pytest

from src.models.vae.decoder import VAEDecoder

EMBED_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 1
NUM_HEADS = 2
DROPOUT = 0.0
MAX_ANSWER_LEN = 10
VOCAB_SIZE = 100
NUM_LATENT_TOKENS = 2
BATCH_SIZE = 4


@pytest.fixture
def decoder():
    return VAEDecoder(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        vocab_size=VOCAB_SIZE,
        num_latent_tokens=NUM_LATENT_TOKENS,
    )


@pytest.fixture
def sample_inputs():
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_ANSWER_LEN))
    z = torch.randn(BATCH_SIZE, NUM_LATENT_TOKENS, LATENT_DIM)
    mask = torch.ones(BATCH_SIZE, MAX_ANSWER_LEN, dtype=torch.long)
    return token_ids, z, mask


def test_output_shape(decoder, sample_inputs):
    token_ids, z, mask = sample_inputs
    out = decoder(token_ids, z, mask)
    assert out.shape == (BATCH_SIZE, MAX_ANSWER_LEN, EMBED_DIM)


def test_gradient_flows(decoder, sample_inputs):
    token_ids, z, mask = sample_inputs
    z = z.clone().requires_grad_(True)
    out = decoder(token_ids, z, mask)
    loss = out.sum()
    loss.backward()
    # Check that at least one decoder parameter has gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
    )
    assert has_grad


def test_causal_masking(decoder, sample_inputs):
    """Changing a future token should not affect earlier hidden states."""
    token_ids, z, mask = sample_inputs
    out1 = decoder(token_ids, z, mask)

    # Modify last token
    token_ids_mod = token_ids.clone()
    token_ids_mod[:, -1] = (token_ids[:, -1] + 1) % VOCAB_SIZE
    out2 = decoder(token_ids_mod, z, mask)

    # All positions except the last should be identical (causal masking)
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)


@pytest.fixture
def inject_decoder():
    return VAEDecoder(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        vocab_size=VOCAB_SIZE,
        num_latent_tokens=NUM_LATENT_TOKENS,
        latent_pos_inject=True,
    )


def test_inject_disabled_by_default(decoder):
    assert decoder.latent_context_proj is None


def test_inject_preserves_shape_and_causality(inject_decoder, sample_inputs):
    token_ids, z, mask = sample_inputs
    out = inject_decoder(token_ids, z, mask)
    assert out.shape == (BATCH_SIZE, MAX_ANSWER_LEN, EMBED_DIM)

    # Causal masking must still hold with per-position injection.
    token_ids_mod = token_ids.clone()
    token_ids_mod[:, -1] = (token_ids[:, -1] + 1) % VOCAB_SIZE
    out2 = inject_decoder(token_ids_mod, z, mask)
    assert torch.allclose(out[:, :-1, :], out2[:, :-1, :], atol=1e-5)


def test_inject_latent_reaches_every_position(inject_decoder, sample_inputs):
    """The injected context must give EVERY decoder position a gradient path
    to z — including position 0 (the <start> token), which otherwise reaches
    z only via the prefix attention."""
    token_ids, _, mask = sample_inputs
    z = torch.randn(BATCH_SIZE, NUM_LATENT_TOKENS, LATENT_DIM, requires_grad=True)
    out = inject_decoder(token_ids, z, mask)
    # Backprop from the first position only.
    out[:, 0, :].sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert inject_decoder.latent_context_proj.weight.grad is not None


def test_inject_generate_runs(inject_decoder, sample_inputs):
    """Autoregressive generation must work with per-position injection."""
    from src.models.vae.output_head import OutputProjection

    _, z, _ = sample_inputs
    head = OutputProjection(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE)
    inject_decoder.eval()
    out = inject_decoder.generate(z, max_len=MAX_ANSWER_LEN, output_head=head)
    assert out.shape == (BATCH_SIZE, MAX_ANSWER_LEN)
