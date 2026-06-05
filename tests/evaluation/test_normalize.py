"""Tests for answer normalization."""

from src.evaluation.normalize import normalize_answer


def test_lowercase():
    assert normalize_answer("The Cat") == "cat"


def test_strip_articles():
    assert normalize_answer("a dog") == "dog"
    assert normalize_answer("an apple") == "apple"
    assert normalize_answer("the quick fox") == "quick fox"


def test_strip_punctuation():
    assert normalize_answer("hello!") == "hello"
    assert normalize_answer("what's up?") == "whats up"


def test_collapse_whitespace():
    assert normalize_answer("the  cat") == "cat"
    assert normalize_answer("  hello   world  ") == "hello world"


def test_empty_string():
    assert normalize_answer("") == ""


def test_already_normalized():
    assert normalize_answer("quick brown fox") == "quick brown fox"
