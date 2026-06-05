"""SQuAD-style evaluation metrics: Exact Match and Token F1."""

from __future__ import annotations

from collections import Counter

from .normalize import normalize_answer


def exact_match(pred: str, gold: str) -> float:
    """Return 1.0 if normalized strings are identical, else 0.0.

    Parameters
    ----------
    pred : str
        Model prediction.
    gold : str
        Ground truth answer.

    Returns
    -------
    float
        1.0 for match, 0.0 otherwise.
    """
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer.

    Both strings are normalized before splitting into tokens.

    Parameters
    ----------
    pred : str
        Model prediction.
    gold : str
        Ground truth answer.

    Returns
    -------
    float
        Token F1 in [0, 1].
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_squad_metrics(
    predictions: list[str],
    references: list[list[str]],
) -> dict:
    """Compute SQuAD metrics over a dataset.

    Parameters
    ----------
    predictions : list[str]
        Model predictions. Empty string means "unanswerable".
    references : list[list[str]]
        Per-example list of valid answers. An empty list (or list containing
        only "") means the example is unanswerable.

    Returns
    -------
    dict
        Keys: em, f1, has_ans_em, has_ans_f1, no_ans_accuracy,
              total, has_ans_count, no_ans_count.
    """
    total = len(predictions)
    em_scores: list[float] = []
    f1_scores: list[float] = []
    has_ans_em: list[float] = []
    has_ans_f1: list[float] = []
    no_ans_correct: list[float] = []

    for pred, golds in zip(predictions, references):
        # Determine if example is answerable
        is_answerable = bool(golds) and any(g.strip() for g in golds)

        if is_answerable:
            best_em = max(exact_match(pred, g) for g in golds)
            best_f1 = max(token_f1(pred, g) for g in golds)
            em_scores.append(best_em)
            f1_scores.append(best_f1)
            has_ans_em.append(best_em)
            has_ans_f1.append(best_f1)
        else:
            # Unanswerable: prediction of "" is correct
            correct = float(pred.strip() == "")
            em_scores.append(correct)
            f1_scores.append(correct)
            no_ans_correct.append(correct)

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "em": _mean(em_scores),
        "f1": _mean(f1_scores),
        "has_ans_em": _mean(has_ans_em),
        "has_ans_f1": _mean(has_ans_f1),
        "no_ans_accuracy": _mean(no_ans_correct),
        "total": total,
        "has_ans_count": len(has_ans_em),
        "no_ans_count": len(no_ans_correct),
    }
