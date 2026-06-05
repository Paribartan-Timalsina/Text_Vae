"""Integration tests for SequenceVAE."""

import torch
import pytest

from src.config.schema import VAEArchConfig
from src.models.vae.vae import SequenceVAE

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
def config():
    return VAEArchConfig(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        num_latent_tokens=NUM_LATENT_TOKENS,
    )


@pytest.fixture
def pretrained_emb():
    return torch.randn(VOCAB_SIZE, EMBED_DIM)


@pytest.fixture
def vae(config, pretrained_emb):
    return SequenceVAE(config, pretrained_embeddings=pretrained_emb)


@pytest.fixture
def sample_batch():
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_ANSWER_LEN))
    mask = torch.ones(BATCH_SIZE, MAX_ANSWER_LEN, dtype=torch.long)
    return ids, mask


def test_full_forward(vae, sample_batch):
    ids, mask = sample_batch
    logits, z, mu, log_var, loss_dict = vae(ids, mask, beta=0.5)

    assert logits.shape == (BATCH_SIZE, MAX_ANSWER_LEN, VOCAB_SIZE)
    assert z.shape == (BATCH_SIZE, NUM_LATENT_TOKENS, LATENT_DIM)
    assert mu.shape == (BATCH_SIZE, NUM_LATENT_TOKENS, LATENT_DIM)
    assert log_var.shape == (BATCH_SIZE, NUM_LATENT_TOKENS, LATENT_DIM)
    assert "total" in loss_dict
    assert "recon" in loss_dict
    assert "kl" in loss_dict
    # BoW key is always present; it is a zero scalar when the head is disabled.
    assert "bow" in loss_dict
    assert loss_dict["bow"].item() == pytest.approx(0.0)


def test_bow_head_contributes_and_flows_to_encoder(pretrained_emb, sample_batch):
    """With use_bow_head + bow_weight, the BoW term is non-zero and its
    gradient reaches the encoder's variational heads through z."""
    cfg = VAEArchConfig(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        num_latent_tokens=NUM_LATENT_TOKENS,
        use_bow_head=True,
    )
    vae = SequenceVAE(cfg, pretrained_embeddings=pretrained_emb)
    assert vae.bow_head is not None

    ids, mask = sample_batch
    _, _, _, _, loss_dict = vae(ids, mask, beta=0.5, bow_weight=1.0)
    assert loss_dict["bow"].item() > 0.0

    # Gradient from BoW alone must reach the encoder mu_head (it pools z = f(mu)).
    vae.zero_grad()
    loss_dict["bow"].backward()
    assert vae.encoder.mu_head.weight.grad is not None
    assert vae.encoder.mu_head.weight.grad.abs().sum() > 0


def test_latent_pos_inject_builds_projection(pretrained_emb):
    cfg = VAEArchConfig(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        num_latent_tokens=NUM_LATENT_TOKENS,
        latent_pos_inject=True,
    )
    vae = SequenceVAE(cfg, pretrained_embeddings=pretrained_emb)
    assert vae.decoder.latent_context_proj is not None


def test_encode_deterministic(vae, sample_batch):
    ids, mask = sample_batch
    z1, _, _ = vae.encode(ids, mask, deterministic=True)
    z2, _, _ = vae.encode(ids, mask, deterministic=True)
    assert torch.equal(z1, z2)


def test_decode_to_tokens(vae, sample_batch):
    ids, mask = sample_batch
    z, _, _ = vae.encode(ids, mask, deterministic=True)
    tokens = vae.decode_to_tokens(z, strategy="greedy")
    assert tokens.dtype == torch.long or tokens.dtype == torch.int64
    assert tokens.shape == (BATCH_SIZE, MAX_ANSWER_LEN)
