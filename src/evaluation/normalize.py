"""Text normalization matching the official SQuAD evaluation script."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize an answer string for evaluation.

    Steps (matching official SQuAD eval):
    1. Lowercase
    2. Strip articles (a, an, the) at word boundaries
    3. Strip punctuation
    4. Collapse whitespace

    Parameters
    ----------
    text : str
        Raw answer text.

    Returns
    -------
    str
        Normalized text.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Strip leading/trailing articles at word boundaries
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # 3. Strip punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # 4. Collapse whitespace
    text = " ".join(text.split())

    return text
