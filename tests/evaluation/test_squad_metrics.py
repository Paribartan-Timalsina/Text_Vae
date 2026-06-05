"""Tests for SQuAD evaluation metrics."""

from src.evaluation.squad_metrics import exact_match, token_f1, compute_squad_metrics


class TestExactMatch:
    def test_identical(self):
        assert exact_match("the quick brown fox", "the quick brown fox") == 1.0

    def test_different(self):
        assert exact_match("cat", "dog") == 0.0

    def test_normalized_match(self):
        assert exact_match("The Cat", "the cat") == 1.0

    def test_both_empty(self):
        assert exact_match("", "") == 1.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1("quick fox", "quick fox") == 1.0

    def test_no_overlap(self):
        assert token_f1("cat sits", "dog runs") == 0.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_known_value(self):
        # pred="quick fox", gold="the quick brown fox" → overlap=2, P=1.0, R=0.5
        f1 = token_f1("quick fox", "the quick brown fox")
        assert 0.0 < f1 < 1.0


class TestComputeSquadMetrics:
    def test_perfect_answerable(self):
        preds = ["Paris", "London"]
        refs = [["Paris"], ["London"]]
        m = compute_squad_metrics(preds, refs)
        assert m["em"] == 1.0
        assert m["f1"] == 1.0

    def test_multi_reference_takes_max(self):
        preds = ["Paris"]
        refs = [["Lyon", "Paris"]]
        m = compute_squad_metrics(preds, refs)
        assert m["em"] == 1.0

    def test_empty_prediction_unanswerable(self):
        preds = [""]
        refs = [[""]]
        m = compute_squad_metrics(preds, refs)
        assert m["em"] == 1.0

    def test_has_ans_no_ans_split(self):
        preds = ["Paris", ""]
        refs = [["Paris"], [""]]
        m = compute_squad_metrics(preds, refs)
        assert m["has_ans_count"] == 1
        assert m["no_ans_count"] == 1
        assert m["total"] == 2
