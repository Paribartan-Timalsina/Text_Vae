"""Tests for VAEDecoder latent injection — prefix and per-layer KV (deep_inject).

Uses a tiny randomly-initialized GPT-2 (monkeypatched ``from_pretrained``) so the
suite runs offline with no model downloads.
"""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import src.models.vae.decoder as decoder_mod
from src.models.vae.decoder import VAEDecoder

VOCAB = 101
HIDDEN = 32
N_LAYER = 2
N_HEAD = 2
LATENT_DIM = 16
K = 4
B, L = 3, 7


@pytest.fixture(autouse=True)
def tiny_lm(monkeypatch):
    """Replace the pretrained-LM download with a tiny random GPT-2."""

    def fake_from_pretrained(name, *args, **kwargs):
        cfg = GPT2Config(
            vocab_size=VOCAB,
            n_embd=HIDDEN,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            n_positions=64,
        )
        return GPT2LMHeadModel(cfg)

    monkeypatch.setattr(
        decoder_mod.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )


def make_decoder(deep_inject: bool) -> VAEDecoder:
    torch.manual_seed(0)
    return VAEDecoder(
        model_name="gpt2",
        latent_dim=LATENT_DIM,
        num_latent_tokens=K,
        max_answer_len=L,
        latent_pos_inject=True,
        use_lora=False,
        deep_inject=deep_inject,
    )


def make_batch():
    torch.manual_seed(1)
    z = torch.randn(B, K, LATENT_DIM)
    ids = torch.randint(0, VOCAB, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    return z, ids, mask


@pytest.mark.parametrize("deep_inject", [False, True])
def test_forward_shape(deep_inject):
    dec = make_decoder(deep_inject)
    z, ids, mask = make_batch()
    logits = dec(ids, z, mask)
    assert logits.shape == (B, L, VOCAB)


@pytest.mark.parametrize("deep_inject", [False, True])
def test_generate_shape(deep_inject):
    dec = make_decoder(deep_inject).eval()
    z, _, _ = make_batch()
    out = dec.generate(z, max_len=L, strategy="greedy")
    assert out.shape == (B, L)
    assert out.dtype == torch.long


def test_kv_proj_built_only_when_deep_inject():
    assert make_decoder(False).kv_proj is None
    dec = make_decoder(True)
    assert dec.kv_proj is not None
    # One key + one value vector per layer per latent token.
    assert dec.kv_proj.out_features == N_LAYER * 2 * HIDDEN


def test_gradient_flows_to_kv_proj():
    dec = make_decoder(True)
    z, ids, mask = make_batch()
    logits = dec(ids, z, mask)
    logits.sum().backward()
    grad = dec.kv_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_deep_inject_changes_logits():
    """The KV memory must actually reach the attention computation."""
    dec = make_decoder(True).eval()
    z, ids, mask = make_batch()
    with torch.no_grad():
        logits = dec(ids, z, mask)
        # Zero the projection → memory slots become zero vectors → different
        # attention output. If past_key_values were silently dropped, logits
        # would be identical.
        dec.kv_proj.weight.zero_()
        dec.kv_proj.bias.zero_()
        logits_zeroed = dec(ids, z, mask)
    assert not torch.allclose(logits, logits_zeroed, atol=1e-5)


@pytest.mark.parametrize("deep_inject", [False, True])
def test_forward_generate_first_token_parity(deep_inject):
    """Teacher-forced position 0 and generate() step 0 see identical inputs, so
    greedy generation's first token must equal the argmax of forward's first
    logit slice."""
    dec = make_decoder(deep_inject).eval()
    z, ids, mask = make_batch()
    with torch.no_grad():
        fwd_logits = dec(ids, z, mask)  # (B, L, V); position 0 predicts token 0
        gen = dec.generate(z, max_len=1, strategy="greedy")  # (B, 1)
    assert torch.equal(fwd_logits[:, 0, :].argmax(dim=-1), gen[:, 0])


def test_prefix_inject_false_severs_z_but_keeps_shape():
    """prefix_inject=False must keep the K prefix positions (shape unchanged)
    but make them ignore z, so this channel can be ablated independently of
    latent_pos_inject/deep_inject."""
    torch.manual_seed(0)
    dec = VAEDecoder(
        model_name="gpt2",
        latent_dim=LATENT_DIM,
        num_latent_tokens=K,
        max_answer_len=L,
        prefix_inject=False,
        latent_pos_inject=False,
        use_lora=False,
        deep_inject=False,
    ).eval()
    z, ids, mask = make_batch()
    z2 = torch.randn_like(z)
    with torch.no_grad():
        logits_a = dec(ids, z, mask)
        logits_b = dec(ids, z2, mask)
    assert logits_a.shape == (B, L, VOCAB)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)


def test_generate_respects_eos():
    dec = make_decoder(True).eval()
    z, _, _ = make_batch()
    eos = 5
    out = dec.generate(z, max_len=L, strategy="greedy", eos_token_id=eos)
    assert out.shape == (B, L)
    for row in out:
        hits = (row == eos).nonzero()
        if len(hits) > 0:
            # Everything after the first EOS is EOS padding.
            first = hits[0].item()
            assert (row[first:] == eos).all()
