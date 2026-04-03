"""Plain Python str lists for sklearn (PyArrow-safe)."""

from __future__ import annotations

import pandas as pd

from src.utils.stratified_split import stratified_train_test_indices

# Backward-compatible name
stratified_split_indices = stratified_train_test_indices


def corpus_plain_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Materialize ``text`` / ``label`` as Python ``str`` via records (not Arrow slices)."""
    records = df.to_dict(orient="records")
    texts: list[str] = []
    labels: list[str] = []
    for r in records:
        texts.append(str(r.get("text", "")))
        labels.append(str(r.get("label", "")))
    return texts, labels
