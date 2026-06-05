"""ROUGE and BLEU metrics for text generation evaluation."""

from __future__ import annotations

import warnings


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Requires the ``rouge_score`` package. Returns an empty dict with a
    warning if the package is not installed.

    Parameters
    ----------
    predictions : list[str]
        Model-generated texts.
    references : list[str]
        Reference texts (one per prediction).

    Returns
    -------
    dict
        Keys: rouge1, rouge2, rougeL (each as F-measure float), or {} on
        import failure.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        warnings.warn(
            "rouge_score package not found. Install with: uv add rouge-score",
            ImportWarning,
            stacklevel=2,
        )
        return {}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )

    totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    return {key: val / n if n else 0.0 for key, val in totals.items()}


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute corpus-level BLEU score using sacrebleu.

    Requires the ``sacrebleu`` package. Returns an empty dict with a
    warning if the package is not installed.

    Parameters
    ----------
    predictions : list[str]
        Model-generated texts.
    references : list[str]
        Reference texts (one per prediction).

    Returns
    -------
    dict
        Keys: bleu (float 0-100), or {} on import failure.
    """
    try:
        import sacrebleu
    except ImportError:
        warnings.warn(
            "sacrebleu package not found. Install with: uv add sacrebleu",
            ImportWarning,
            stacklevel=2,
        )
        return {}

    if not predictions:
        return {"bleu": 0.0}

    result = sacrebleu.corpus_bleu(predictions, [references])
    return {"bleu": result.score}


def compute_bleu_multiref(
    predictions: list[str],
    references: list[list[str]],
) -> dict:
    """Corpus-level multi-reference BLEU-4 via nltk (no sacrebleu dependency).

    Tokenizes with the SQuAD ``normalize_answer`` (lowercase, strip articles /
    punctuation) so the score is consistent with the EM/F1 reported alongside
    it. Uses corpus-level n-gram aggregation with method-1 smoothing — SQuAD
    answers are 1-5 words, so per-sentence BLEU-4 is almost always 0 (no
    4-grams); aggregating counts across the whole answerable set is the only
    way the metric carries signal.

    Parameters
    ----------
    predictions : list[str]
        Model-generated texts.
    references : list[list[str]]
        Per-prediction list of acceptable gold answers (SQuAD multi-ref).

    Returns
    -------
    dict
        Keys: bleu (float 0-100), or {} on import failure / no data.
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError:
        warnings.warn(
            "nltk package not found. Install with: uv add nltk",
            ImportWarning,
            stacklevel=2,
        )
        return {}

    from .normalize import normalize_answer

    hyps: list[list[str]] = []
    refs: list[list[list[str]]] = []
    for pred, golds in zip(predictions, references):
        pred_toks = normalize_answer(pred).split()
        gold_toks = [normalize_answer(g).split() for g in golds if g.strip()]
        if not gold_toks:
            continue
        hyps.append(pred_toks)
        refs.append(gold_toks)

    if not hyps:
        return {}

    score = corpus_bleu(
        refs, hyps, smoothing_function=SmoothingFunction().method1
    )
    return {"bleu": score * 100.0}
