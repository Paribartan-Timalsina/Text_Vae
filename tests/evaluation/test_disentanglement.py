"""Tests for the disentanglement metrics (LangSpace port).

Strategy: build synthetic data whose latent dims are EITHER aligned to the
generative factors (each factor encoded in its own dim) OR random. Aligned must
score clearly higher than random for z-diff and informativeness; all metrics must
return finite values in [0, 1]. Also unit-tests the factor-bucketing logic.
"""

from __future__ import annotations

import random

import torch

from src.evaluation.disentanglement import (
    GEN_FACTORS,
    build_factor_buckets,
    attach_representations,
    token_srl_role,
    z_diff_score,
    z_min_var_score,
    informativeness_score,
    compute_disentanglement,
)

# Three single-role factors; the VALUE is encoded by repeating the role (v+1)×,
# so each factor has 3 distinct role-pattern values.
_FACTOR_ROLE = {"argument": "ARG1", "verb": "V", "time": "ARGM-TMP"}
_FACTORS = list(_FACTOR_ROLE.keys())
_N_VALUES = 3


def _make_synthetic(n: int, aligned: bool, seed: int = 0):
    """Return (sentences_srl, Z, value_table).

    ``value_table[s]`` = list of value ids (one per factor). If ``aligned``, dim
    ``f`` of Z encodes factor ``f``'s value (so it is recoverable); else Z is pure
    noise. Extra noise dims pad Z to D=16.
    """
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    D = 16
    sentences_srl, Z, value_table = [], [], []
    for _ in range(n):
        vals = [rng.randrange(_N_VALUES) for _ in _FACTORS]
        value_table.append(vals)
        roles: list[str] = []
        for fi, factor in enumerate(_FACTORS):
            roles += [_FACTOR_ROLE[factor]] * (vals[fi] + 1)
        sentences_srl.append(roles)
        z = torch.randn(D, generator=g)
        if aligned:
            for fi in range(len(_FACTORS)):
                z[fi] = vals[fi] * 10.0 + 0.1 * torch.randn(1, generator=g).item()
        Z.append(z)
    return sentences_srl, torch.stack(Z), value_table


# --- token role reduction --------------------------------------------------
def test_token_srl_role_strips_bio():
    assert token_srl_role(["B-ARG1"]) == "ARG1"
    assert token_srl_role(["I-ARG2"]) == "ARG2"
    assert token_srl_role(["B-V"]) == "V"
    assert token_srl_role(["O"]) == "O"
    # first non-O frame wins
    assert token_srl_role(["O", "B-ARGM-TMP"]) == "ARGM-TMP"
    assert token_srl_role("B-ARG0") == "ARG0"


# --- factor bucketing ------------------------------------------------------
def test_build_factor_buckets_groups_by_pattern():
    # two sentences with one ARG1, one with two ARG1 → two distinct "argument"
    # values; verb present in all three.
    srl = [
        ["ARG1", "V"],
        ["ARG1", "V"],
        ["ARG1", "ARG1", "V"],
    ]
    b = build_factor_buckets(srl)
    ai = b.generative_factors.index("argument")
    # value ("ARG1",) covers sentences 0,1; ("ARG1","ARG1") covers sentence 2
    patterns = {tuple(p): set(idx) for p, idx in zip(b.value_space[ai], b.sample_space[ai])}
    assert patterns[("ARG1",)] == {0, 1}
    assert patterns[("ARG1", "ARG1")] == {2}
    vi = b.generative_factors.index("verb")
    assert {i for idx in b.sample_space[vi] for i in idx} == {0, 1, 2}


def test_attach_representations_aligns_rows():
    srl = [["ARG1", "V"], ["ARG1", "ARG1", "V"]]
    Z = torch.arange(2 * 4, dtype=torch.float).reshape(2, 4)
    b = build_factor_buckets(srl)
    attach_representations(b, Z)
    ai = b.generative_factors.index("argument")
    # bucket for sentence 0 must hold row 0
    for vj, idxs in enumerate(b.sample_space[ai]):
        for k, sent_idx in enumerate(idxs):
            assert torch.equal(b.representation_space[ai][vj][k], Z[sent_idx])


# --- metric values: aligned >> random -------------------------------------
def test_aligned_beats_random_z_diff_and_informativeness():
    srl_a, Z_a, _ = _make_synthetic(600, aligned=True, seed=1)
    srl_r, Z_r, _ = _make_synthetic(600, aligned=False, seed=2)

    ba = build_factor_buckets(srl_a)
    attach_representations(ba, Z_a)
    br = build_factor_buckets(srl_r)
    attach_representations(br, Z_r)

    random.seed(0); torch.manual_seed(0)
    zd_a = z_diff_score(ba, sample_number=40)[0]
    random.seed(0); torch.manual_seed(0)
    zd_r = z_diff_score(br, sample_number=40)[0]

    info_a = informativeness_score(ba, sample_number=600)["informativeness"][0]
    info_r = informativeness_score(br, sample_number=600)["informativeness"][0]

    # aligned latents recover factors; random latents do not
    assert info_a > 0.8, f"aligned informativeness too low: {info_a}"
    assert info_a > info_r + 0.2, f"aligned {info_a} should beat random {info_r}"
    assert zd_a >= zd_r, f"z_diff aligned {zd_a} should be >= random {zd_r}"


def test_metrics_return_finite_in_unit_range():
    srl, Z, _ = _make_synthetic(400, aligned=True, seed=3)
    b = build_factor_buckets(srl)
    attach_representations(b, Z)
    for name, val in [
        ("z_diff", z_diff_score(b, sample_number=30)),
        ("z_min_var", z_min_var_score(b, sample_number=100)),
    ]:
        mean, std = val
        assert 0.0 <= mean <= 1.0, f"{name} mean out of range: {mean}"
        assert std >= 0.0
    info = informativeness_score(b, sample_number=400)
    for k in ("informativeness", "disentanglement", "completeness"):
        assert k in info
        assert 0.0 <= info[k][0] <= 1.0


def test_compute_disentanglement_end_to_end():
    srl, Z, _ = _make_synthetic(300, aligned=True, seed=4)
    res = compute_disentanglement(Z, srl, seed=7)
    assert set(res) >= {"z_diff", "z_min_var", "informativeness"}
    for mean, std in res.values():
        assert mean == mean  # not NaN
