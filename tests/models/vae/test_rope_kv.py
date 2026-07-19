"""Correctness of the RoPE-aware KV fan-out.

The decoder rotates the injected keys itself (a family/version-robust reimplement
of RoPE). These tests pin that reimplementation to transformers' own
``apply_rotary_pos_emb`` so the two can never drift, check the RoPE-detection +
rotary-fetch helpers, and verify end-to-end that ``_past_kv`` injects correctly
rotated keys (and un-rotated values) on a real RoPE decoder.
"""

from __future__ import annotations

import pytest
import torch

from src.models.vae.decoder import _rotate_half, _find_rotary_emb, _decoder_uses_rope


def test_rotate_half_matches_transformers_apply_rope():
    """Our ``k*cos + rotate_half(k)*sin`` must equal transformers'
    ``apply_rotary_pos_emb`` on the same (cos, sin, k)."""
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    lm = LlamaForCausalLM(cfg)
    rot = _find_rotary_emb(lm)
    assert rot is not None, "rotary_emb should be reachable on a Llama model"

    M = 6
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    pos = torch.arange(M).unsqueeze(0)
    cos, sin = rot(torch.zeros(1, M, cfg.hidden_size), pos)  # (1, M, head_dim)

    k = torch.randn(1, cfg.num_key_value_heads, M, head_dim)
    q = torch.zeros_like(k)
    _, k_ref = llama.apply_rotary_pos_emb(q, k, cos, sin)  # library

    c, s = cos[0], sin[0]  # (M, head_dim) broadcast over (…, M, head_dim)
    k_ours = k * c + _rotate_half(k) * s

    assert torch.allclose(k_ours, k_ref, atol=1e-5), "rotation drifted from transformers"


def test_rope_detection():
    from transformers import LlamaConfig, GPT2Config

    assert _decoder_uses_rope(LlamaConfig()) is True
    assert _decoder_uses_rope(GPT2Config()) is False


def test_past_kv_injected_keys_are_correctly_rotated(tmp_path):
    """End-to-end: the keys our ``_past_kv`` injects on a RoPE decoder must equal
    the model's own ``apply_rotary_pos_emb`` of the same raw keys at positions
    0..M-1, and the values must be left un-rotated. (Eval mode → kv_dropout off.)"""
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.models.vae.decoder import VAEDecoder

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )
    d = str(tmp_path / "tiny_llama")
    LlamaForCausalLM(cfg).save_pretrained(d)

    M = 5
    dec = VAEDecoder(d, latent_dim=16, num_latent_tokens=1, max_answer_len=10,
                     use_lora=False, deep_inject=True, kv_fanout_len=M, fanout_mode="kv")
    assert dec.rope_kv, "kv mode on a Llama should enable rope_kv"
    dec.eval()  # dropout off so the reconstruction is deterministic

    z = torch.randn(1, 1, 16)
    with torch.no_grad():
        pooled = z.mean(1)
        raw = (dec.kv_proj(pooled)
               .view(1, dec.n_layer, 2, dec.n_kv_head, M, dec.kv_head_dim)
               .permute(1, 2, 0, 3, 4, 5).to(dec.compute_dtype))
        raw_k0, raw_v0 = raw[0, 0], raw[0, 1]  # layer-0 key/value (1, n_kv_head, M, hd)
        rot = _find_rotary_emb(dec.lm)
        cos, sin = rot(raw_k0, torch.arange(M).unsqueeze(0))
        _, ref_keys = llama.apply_rotary_pos_emb(torch.zeros_like(raw_k0), raw_k0, cos, sin)

        past = dec._past_kv(z)
        legacy = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past

    assert torch.allclose(legacy[0][0], ref_keys, atol=1e-4), "injected keys mis-rotated"
    assert torch.allclose(legacy[0][1], raw_v0, atol=1e-4), "values must not be rotated"
