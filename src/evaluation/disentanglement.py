"""Disentanglement metrics for the VAE latent space (LangVAE-comparable).

Ports the three metrics LangVAE reports in Table 1 of arXiv:2505.00004, computed
by LangSpace's disentanglement probe:

  * **z-diff**       — Higgins et al. 2017 (β-VAE metric).      ↑ higher = better
  * **z-min-var**    — Kim & Mnih 2018 (FactorVAE metric).      ↓ lower  = better
  * **informativeness** — Eastwood & Williams 2018 (DCI).       ↑ higher = better

These are SUPERVISED metrics: they score how well individual latent dimensions
isolate known *generative factors*. LangVAE uses semantic-role-labeling (SRL)
roles as the factors. This module is a faithful reimplementation of LangSpace's
``DisentanglementProbe`` (langspace/probe/disentanglement/__init__.py:312/383/511)
operating directly on our own encoder's latents — so the numbers are *comparable*
to the paper's, though not byte-identical to their unreleased harness.

The latent is a single pooled vector per sentence ``(N, D)`` (at K=1, exactly
LangVAE's setup; for K>1 the caller mean-pools the K slots → one vector). The
metrics only see that matrix plus the per-sentence factor labels, so they are
self-contained pure functions of ``(representations, factor buckets)``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- LangVAE's SRL role → generative-factor grouping -----------------------
# Copied from LangSpace examples/definition_vae.py (gen_factors). Keys are the
# 15 factor names; values are the SRL core roles (B-/I- prefixes stripped before
# matching, see ``token_srl_role``) that belong to each factor. Kept verbatim so
# the factor structure matches the paper (incl. the "empty"/"O" and rare C-ARG
# groups). "C-AGR2" is a typo in the upstream code; harmless (role ~never occurs).
GEN_FACTORS: dict[str, list[str]] = {
    "direction": ["ARGM-DIR"],
    "because": ["ARGM-CAU"],
    "purpose": ["ARGM-PRP", "ARGM-PNC", "ARGM-GOL"],
    "more": ["ARGM-EXT"],
    "location": ["ARGM-LOC"],
    "argument": ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4"],
    "manner": ["ARGM-MNR"],
    "can": ["ARGM-MOD"],
    "argm-prd": ["ARGM-PRD"],
    "empty": ["O"],
    "negation": ["ARGM-NEG"],
    "verb": ["V"],
    "if-then": ["ARGM-ADV", "ARGM-DIS"],
    "time": ["ARGM-TMP"],
    "C-ARG": ["C-ARG1", "C-ARG0", "C-AGR2"],
}


def token_srl_role(srl: list[str] | str) -> str:
    """Reduce a token's SRL annotation to a single core role label.

    saf-datasets stores SRL as a list (one entry per predicate frame), each
    prefixed ``B-``/``I-``/``O`` (e.g. ``['B-ARG1']``, ``['I-ARG2']``). We take
    the first non-``O`` frame (else ``O``) and strip the BIO prefix so ``B-ARG1``
    and ``I-ARG1`` both map to the core role ``ARG1`` — a cleaner grouping than
    LangSpace's I-only matching, and it keeps the begin tokens too.
    """
    if isinstance(srl, str):
        srl = [srl]
    label = "O"
    for frame in srl:
        if frame and frame != "O":
            label = frame
            break
    # Strip a leading BIO prefix ("B-" / "I-").
    if len(label) > 2 and label[1] == "-" and label[0] in ("B", "I"):
        label = label[2:]
    return label


# ---------------------------------------------------------------------------
@dataclass
class FactorBuckets:
    """Sentences grouped by their generative-factor values (LangSpace's
    ``SRLFactorDataset``).

    For factor ``i`` (``generative_factors[i]``), ``value_space[i]`` is the list
    of distinct role-patterns observed for that factor, ``sample_space[i][j]`` is
    the list of sentence indices exhibiting value ``j``, and (once attached)
    ``representation_space[i][j]`` is the ``(n_j, D)`` latent matrix for them.
    """

    generative_factors: list[str]
    value_space: list[list[tuple]]
    sample_space: list[list[list[int]]]
    representation_space: list[list[torch.Tensor]] = field(default_factory=list)


def build_factor_buckets(
    sentences_srl: list[list[str]],
    gen_factors: dict[str, list[str]] = GEN_FACTORS,
) -> FactorBuckets:
    """Bucket sentences by generative-factor value.

    Parameters
    ----------
    sentences_srl:
        One entry per sentence: the list of per-token core SRL roles (already
        reduced via :func:`token_srl_role`), in token order.
    gen_factors:
        Factor-name → list of core roles belonging to that factor.

    Returns a :class:`FactorBuckets` with ``generative_factors`` /
    ``value_space`` / ``sample_space`` populated (no representations yet). Mirrors
    ``SRLFactorDataset.__init__`` (LangSpace disentanglement/__init__.py:120-150):
    a sentence's *value* for a factor is the ordered tuple of that factor's roles
    present in the sentence.
    """
    factors = list(gen_factors.keys())
    # role label -> factor name
    role_to_factor: dict[str, str] = {}
    for factor in factors:
        for role in gen_factors[factor]:
            role_to_factor[role] = factor

    value_space: list[list[tuple]] = [[] for _ in factors]
    sample_space: list[list[list[int]]] = [[] for _ in factors]

    for idx, roles in enumerate(sentences_srl):
        # roles of this sentence that belong to some factor, in order
        tagged = [r for r in roles if r in role_to_factor]
        present = {role_to_factor[r] for r in tagged}
        for factor in present:
            fi = factors.index(factor)
            pattern = tuple(r for r in tagged if role_to_factor[r] == factor)
            if pattern not in value_space[fi]:
                value_space[fi].append(pattern)
                sample_space[fi].append([idx])
            else:
                sample_space[fi][value_space[fi].index(pattern)].append(idx)

    return FactorBuckets(factors, value_space, sample_space)


def attach_representations(buckets: FactorBuckets, Z: torch.Tensor) -> None:
    """Fill ``buckets.representation_space[i][j] = Z[sample_space[i][j]]``.

    ``Z`` is ``(N, D)`` and must be aligned (row ``k`` = sentence ``k``) with the
    ``sentences_srl`` list used to build ``buckets``.
    """
    Z = Z.detach().cpu()
    buckets.representation_space = [
        [Z[torch.tensor(idxs, dtype=torch.long)] for idxs in factor_buckets]
        for factor_buckets in buckets.sample_space
    ]


# --- sampling helpers (ported from LangSpace) ------------------------------
def _group_sampling(buckets: FactorBuckets, fi: int, vj: int, batch_size: int) -> torch.Tensor:
    """Sample ``min(batch_size, n)`` latents from bucket (factor ``fi``, value
    ``vj``). Mirrors ``DisentanglementProbe.group_sampling``."""
    space = buckets.representation_space[fi][vj]
    n = space.shape[0]
    rows = random.sample(range(n), min(batch_size, n))
    return space[rows, :]


def _stratified_sampling(
    buckets: FactorBuckets, fi: int, sample_number: int
) -> list[torch.Tensor]:
    """Draw ``sample_number`` latents for factor ``fi``, split across its value
    buckets in proportion to bucket size. Mirrors ``stratified_sampling`` (we
    return only the per-value sample list; the probabilities aren't needed by the
    metrics we expose)."""
    sizes = [len(b) for b in buckets.sample_space[fi]]
    total = sum(sizes)
    samples: list[torch.Tensor] = []
    for vj, space in enumerate(buckets.representation_space[fi]):
        p = sizes[vj] / total if total else 0.0
        take = round(sample_number * p)
        n = space.shape[0]
        rows = random.sample(range(n), min(take, n))
        samples.append(space[rows, :])
    return samples


def _entropy(p: torch.Tensor) -> torch.Tensor:
    flat = p.flatten()
    flat = flat[flat > 0]
    return torch.sum(-flat * torch.log(flat))


# --- the three metrics -----------------------------------------------------
def z_diff_score(
    buckets: FactorBuckets, batch_size: int = 64, sample_number: int = 50,
    n_classifiers: int = 10, n_epochs: int = 10,
) -> tuple[float, float]:
    """β-VAE metric (Higgins 2017) — "z-diff". ↑ higher is better.

    For each factor, repeatedly fix its value and take two batches sharing that
    value; the mean ``|z1 - z2|`` over the batch is the feature, labelled by the
    factor index. A linear classifier that recovers the factor from this feature
    scores high when one latent dim cleanly tracks each factor. Returns
    ``(mean_accuracy, std)`` over ``n_classifiers`` runs. Port of
    ``beta_vae_metric`` (LangSpace :312-381).
    """
    feats: list[torch.Tensor] = []
    labels: list[int] = []
    for fi in range(len(buckets.generative_factors)):
        # flat list of sentence indices that have ANY value for this factor
        flat = [i for b in buckets.sample_space[fi] for i in b]
        if not flat:
            continue
        usable = [vj for vj, b in enumerate(buckets.sample_space[fi]) if len(b) >= 2]
        if not usable:
            continue
        for _ in range(sample_number):
            vj = random.choice(usable)
            z1 = _group_sampling(buckets, fi, vj, batch_size)
            z2 = _group_sampling(buckets, fi, vj, batch_size)
            feats.append(torch.abs(z1.mean(0) - z2.mean(0)).unsqueeze(0))
            labels.append(fi)

    if not feats:
        return float("nan"), float("nan")

    x = torch.cat(feats, dim=0)
    y = F.one_hot(torch.tensor(labels), num_classes=len(buckets.generative_factors)).float()

    accs = torch.zeros(n_classifiers)
    for c in range(n_classifiers):
        perm = torch.randperm(x.shape[0])
        xs, ys = x[perm], y[perm]
        split = int(0.8 * xs.shape[0])
        xtr, xte, ytr, yte = xs[:split], xs[split:], ys[:split], ys[split:]
        if xte.shape[0] == 0:
            continue
        clf = nn.Sequential(nn.Linear(x.shape[1], y.shape[1]), nn.Softmax(dim=-1))
        opt = torch.optim.Adam(clf.parameters(), lr=0.01)
        xl = DataLoader(xtr, batch_size=64)
        yl = DataLoader(ytr, batch_size=64)
        for _ in range(n_epochs):
            clf.train()
            for bx, by in zip(xl, yl):
                opt.zero_grad()
                loss = nn.NLLLoss()(torch.log(clf(bx) + 1e-12), by.argmax(-1))
                loss.backward()
                opt.step()
        clf.eval()
        with torch.no_grad():
            correct = (clf(xte).argmax(-1) == yte.argmax(-1)).int().sum()
        accs[c] = correct / xte.shape[0]
    return accs.mean().item(), accs.std().item()


def z_min_var_score(
    buckets: FactorBuckets, batch_size: int = 64, sample_number: int = 1000,
    n_classifiers: int = 10,
) -> tuple[float, float]:
    """FactorVAE metric (Kim & Mnih 2018) — "z-min-var". ↓ lower is better.

    Normalise each latent dim by its global std; for a batch sharing one factor
    value, the argmin-variance dim is the feature, labelled by the factor index;
    a majority-vote classifier maps dim → factor. Returns ``(mean_acc, std)``.
    Port of ``factor_vae_metric`` (LangSpace :383-444).

    NOTE on direction: LangSpace reports this raw classifier accuracy and marks
    the column ``↓`` in the paper, so we return the same quantity unchanged.
    """
    # global per-dim std over all representations (any factor's union works; use
    # the first non-empty bucket set concatenated)
    all_rows = torch.cat(
        [b for fb in buckets.representation_space for b in fb if b.shape[0] > 0], dim=0
    )
    scale = all_rows.std(dim=0).clamp(min=1e-8)

    xs: list[int] = []
    ys: list[int] = []
    for fi in range(len(buckets.generative_factors)):
        usable = [vj for vj, b in enumerate(buckets.sample_space[fi]) if len(b) >= 2]
        if not usable:
            continue
        for _ in range(sample_number):
            vj = random.choice(usable)
            z = _group_sampling(buckets, fi, vj, batch_size)
            z_var = (z / scale).var(dim=0)
            xs.append(int(z_var.argmin().item()))
            ys.append(fi)

    if not xs:
        return float("nan"), float("nan")

    x = torch.tensor(xs)
    y = torch.tensor(ys)
    n_dim = all_rows.shape[1]
    n_fac = len(buckets.generative_factors)
    accs = []
    for _ in range(n_classifiers):
        perm = torch.randperm(x.shape[0])
        xs_, ys_ = x[perm], y[perm]
        split = int(0.8 * xs_.shape[0])
        xtr, xte, ytr, yte = xs_[:split], xs_[split:], ys_[:split], ys_[split:]
        if xte.shape[0] == 0:
            continue
        V = torch.zeros((n_dim, n_fac))
        for d, f in zip(xtr.tolist(), ytr.tolist()):
            V[d, f] += 1
        correct = sum(int(V[d].argmax().item() == f) for d, f in zip(xte.tolist(), yte.tolist()))
        accs.append(correct / xte.shape[0])
    accs_t = torch.tensor(accs)
    return accs_t.mean().item(), accs_t.std().item()


def informativeness_score(
    buckets: FactorBuckets, sample_number: int = 10000, n_estimators: int = 10,
) -> dict[str, tuple[float, float]]:
    """DCI metrics (Eastwood & Williams 2018). Returns ``informativeness``
    (↑ better) plus the cheap by-products ``disentanglement`` / ``completeness``
    derived from the same random-forest importances. Port of
    ``disentanglement_completeness_informativeness`` (LangSpace :511-582).

    For each factor, a RandomForest predicts the factor's value from the latent;
    informativeness = test accuracy (mean over factors). Disentanglement /
    completeness come from the entropy of the importance matrix.
    """
    from sklearn.ensemble import RandomForestClassifier

    informativeness: list[float] = []
    importances: list[np.ndarray] = []
    for fi in range(len(buckets.generative_factors)):
        samples = _stratified_sampling(buckets, fi, sample_number)
        xtr_l, xte_l, ytr_l, yte_l = [], [], [], []
        for vj, space in enumerate(samples):
            n = space.shape[0]
            if n == 0:
                continue
            cut = int(np.ceil(0.8 * n))
            xtr_l.append(space[:cut])
            xte_l.append(space[cut:])
            ytr_l.append(torch.full((cut,), vj, dtype=torch.long))
            yte_l.append(torch.full((n - cut,), vj, dtype=torch.long))
        if not xtr_l:
            continue
        xtr = torch.cat(xtr_l).numpy()
        ytr = torch.cat(ytr_l).numpy()
        xte = torch.cat(xte_l)
        yte = torch.cat(yte_l)
        # need ≥2 classes with samples and a non-empty test set
        if xte.shape[0] == 0 or len(set(ytr.tolist())) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(xtr, ytr)
        informativeness.append(float(clf.score(xte.numpy(), yte.numpy())))
        importances.append(clf.feature_importances_)

    if not informativeness:
        nan = (float("nan"), float("nan"))
        return {"informativeness": nan, "disentanglement": nan, "completeness": nan}

    info_t = torch.tensor(informativeness)
    out: dict[str, tuple[float, float]] = {
        "informativeness": (info_t.mean().item(), info_t.std().item()),
    }

    if len(importances) >= 1:
        r = torch.tensor(np.stack(importances))  # (n_factors, n_dim)
        # disentanglement: per latent dim, 1 - normalised entropy over factors
        dis = []
        for d in range(r.shape[1]):
            p = r[:, d]
            p = p / p.sum() if p.sum() > 1e-7 else torch.zeros_like(p)
            dis.append(1 - _entropy(p) / math.log(max(r.shape[0], 2)))
        dis_t = torch.tensor(dis)
        # completeness: per factor, 1 - normalised entropy over dims
        comp = []
        for f in range(r.shape[0]):
            p = r[f, :]
            p = p / p.sum() if p.sum() > 1e-7 else torch.zeros_like(p)
            comp.append(1 - _entropy(p) / math.log(max(r.shape[1], 2)))
        comp_t = torch.tensor(comp)
        out["disentanglement"] = (dis_t.mean().item(), dis_t.std().item())
        out["completeness"] = (comp_t.mean().item(), comp_t.std().item())

    return out


# ---------------------------------------------------------------------------
def compute_disentanglement(
    Z: torch.Tensor,
    sentences_srl: list[list[str]],
    gen_factors: dict[str, list[str]] = GEN_FACTORS,
    metrics: tuple[str, ...] = ("z_diff", "z_min_var", "informativeness"),
    seed: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute the requested disentanglement metrics on latent matrix ``Z``.

    Parameters
    ----------
    Z:
        ``(N, D)`` pooled latent vectors, row-aligned with ``sentences_srl``.
    sentences_srl:
        Per-sentence list of per-token core SRL roles (see
        :func:`token_srl_role`).
    metrics:
        Any of ``"z_diff"``, ``"z_min_var"``, ``"informativeness"``.

    Returns ``{metric_name: (mean, std)}`` (informativeness also adds
    ``disentanglement``/``completeness``).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    buckets = build_factor_buckets(sentences_srl, gen_factors)
    attach_representations(buckets, Z)

    results: dict[str, tuple[float, float]] = {}
    if "z_diff" in metrics:
        results["z_diff"] = z_diff_score(buckets)
    if "z_min_var" in metrics:
        results["z_min_var"] = z_min_var_score(buckets)
    if "informativeness" in metrics:
        results.update(informativeness_score(buckets))
    return results
