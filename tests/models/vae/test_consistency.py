"""Correctness of the self-distillation consistency loss.

``compute_consistency_loss`` distills the teacher (main) pass into the student
(z-only) pass via KL(teacher || student). These tests pin its three defining
properties: it is zero when the passes already agree, positive when they differ,
and its gradient flows ONLY into the student (the teacher is detached).
"""

from __future__ import annotations

import torch

from src.models.vae.loss import compute_consistency_loss
from src.models.vae.vae import SequenceVAE

B, L, V = 3, 5, 17


def _mask():
    m = torch.ones(B, L)
    m[0, -2:] = 0  # a little padding, to check the mask is honoured
    return m


def test_zero_when_teacher_equals_student():
    """Identical logits -> KL is (numerically) zero."""
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    loss = compute_consistency_loss(logits, logits.clone(), _mask())
    # KL of a distribution with itself is 0 up to float error (the log clamp can
    # nudge it a hair negative).
    assert abs(loss.item()) < 1e-5, f"KL should vanish when passes agree, got {loss.item()}"


def test_positive_when_divergent():
    """Different distributions -> strictly positive KL."""
    torch.manual_seed(1)
    teacher = torch.randn(B, L, V) * 3.0  # peaky teacher
    student = torch.zeros(B, L, V)        # ~uniform student
    loss = compute_consistency_loss(teacher, student, _mask())
    assert loss.item() > 0.1, f"divergent passes should give positive KL, got {loss.item()}"


def test_gradient_flows_only_to_student():
    """Teacher is detached: only the student logits receive gradient."""
    torch.manual_seed(2)
    teacher = torch.randn(B, L, V, requires_grad=True)
    student = torch.randn(B, L, V, requires_grad=True)
    loss = compute_consistency_loss(teacher, student, _mask())
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0, "student must get grad"
    assert teacher.grad is None, "teacher must be detached (no grad)"


def test_masked_tokens_excluded():
    """Padding positions must not contribute to the loss."""
    torch.manual_seed(3)
    teacher = torch.randn(B, L, V) * 3.0
    student = torch.zeros(B, L, V)
    full = torch.ones(B, L)
    partial = full.clone()
    partial[:, -1] = 0  # drop the last column
    loss_full = compute_consistency_loss(teacher, student, full)
    loss_partial = compute_consistency_loss(teacher, student, partial)
    assert loss_partial.item() < loss_full.item(), "masking tokens should lower the loss"


def test_temperature_scaling_runs():
    """T != 1 stays finite and non-negative (the T^2 scaling is applied)."""
    torch.manual_seed(4)
    teacher = torch.randn(B, L, V) * 2.0
    student = torch.randn(B, L, V)
    for T in (0.5, 1.0, 2.0):
        loss = compute_consistency_loss(teacher, student, _mask(), temperature=T)
        assert torch.isfinite(loss) and loss.item() >= 0.0


# --- integration: the real SequenceVAE.forward guard, exercised offline ------
# forward() only touches self.encode / self.decode / self.bow_head / self.training,
# so we build a bare instance and stub those — no model downloads — to verify the
# z-only pass is reused (one decode for both zforce and consistency) and the
# loss_dict wiring / back-compat behave.
K, D = 2, 4


def _make_vae(decode_calls):
    vae = object.__new__(SequenceVAE)  # skip __init__ (no model loading)
    vae.bow_head = None
    vae.training = True

    torch.manual_seed(7)
    mu = torch.zeros(B, K, D)
    log_var = torch.zeros(B, K, D)
    z = torch.zeros(B, K, D)
    vae.encode = lambda e, m: (z, mu, log_var)

    def fake_decode(dec_ids, z_, dec_mask, word_dropout=0.0, mask_token_id=None):
        decode_calls.append(word_dropout)
        # main pass (wd<1) is confident; z-only pass (wd==1) is ~uniform → they differ
        scale = 0.0 if word_dropout >= 1.0 else 5.0
        g = torch.Generator().manual_seed(int(word_dropout * 10) + 1)
        return torch.randn(B, L, V, generator=g) * scale

    vae.decode = fake_decode
    return vae


def _run(vae, **kw):
    ids = torch.randint(0, V, (B, L))
    mask = torch.ones(B, L)
    return vae.forward(ids, mask, ids, mask, mask_token_id=0, **kw)


def test_forward_backcompat_no_aux_passes():
    """No zforce, no consistency → decode runs ONCE; consistency stays 0."""
    calls = []
    _, _, _, _, ld = _run(_make_vae(calls), zforce_weight=0.0, consistency_weight=0.0)
    assert calls == [0.0], f"expected a single main decode, got word_dropouts {calls}"
    assert ld["consistency"].item() == 0.0


def test_forward_consistency_reuses_single_zonly_pass():
    """zforce AND consistency both on → exactly TWO decodes (main + ONE z-only)."""
    calls = []
    _, _, _, _, ld = _run(_make_vae(calls), zforce_weight=1.0, consistency_weight=1.0)
    assert calls == [0.0, 1.0], f"z-only pass must be reused once, got {calls}"
    assert ld["consistency"].item() > 0.0
    assert ld["recon_zonly"].item() > 0.0


def test_forward_consistency_only_still_runs_zonly():
    """consistency on but zforce off → z-only pass still runs (for the teacher/student
    KL), recon_zonly stays 0."""
    calls = []
    _, _, _, _, ld = _run(_make_vae(calls), zforce_weight=0.0, consistency_weight=1.0)
    assert calls == [0.0, 1.0], f"z-only pass needed for consistency, got {calls}"
    assert ld["consistency"].item() > 0.0
    assert ld["recon_zonly"].item() == 0.0
