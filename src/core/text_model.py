"""TF–IDF + logistic regression with explanations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processing.corpus_utils import corpus_plain_lists
from src.utils.stratified_split import stratified_train_test_indices

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = _PROJECT_ROOT / "data" / "social_signal_corpus.jsonl"


def load_corpus(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or CORPUS_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Missing corpus file at {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["text", "label"])
    out = pd.DataFrame([{"text": str(r["text"]), "label": str(r["label"])} for r in rows])
    for col in ("text", "label"):
        out[col] = out[col].astype(object)
    return out


def build_pipeline(max_features: int = 3000) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words=None,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
            ),
        ]
    )


def train_classifier(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = df if df is not None else load_corpus()
    texts, labels = corpus_plain_lists(df)
    idx_train, idx_test = stratified_train_test_indices(labels, test_size, random_state)
    X_train = [texts[int(i)] for i in idx_train]
    X_test = [texts[int(i)] for i in idx_test]
    y_train = [labels[int(i)] for i in idx_train]
    y_test = [labels[int(i)] for i in idx_test]
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


def top_explanatory_terms(pipe: Pipeline, label: str, top_k: int = 12) -> list[tuple[str, float]]:
    clf: LogisticRegression = pipe.named_steps["clf"]
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    names = np.array(vec.get_feature_names_out())
    classes = list(clf.classes_)
    coef = clf.coef_[classes.index(label)]
    top = np.argsort(coef)[::-1][:top_k]
    return [(str(names[i]), float(coef[i])) for i in top]


def token_highlight_explanation(pipe: Pipeline, text: str, top_k: int = 8) -> list[tuple[str, float]]:
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    X = vec.transform([text])
    inv_vocab = {i: w for w, i in vec.vocabulary_.items()}
    coo = X.tocoo()
    contributions: dict[str, float] = {}
    for _, col, val in zip(coo.row, coo.col, coo.data):
        tok = inv_vocab.get(int(col))
        if tok is None:
            continue
        impact = float(np.max(np.abs(clf.coef_[:, int(col)] * val)))
        contributions[tok] = max(contributions.get(tok, 0.0), impact)
    return sorted(contributions.items(), key=lambda x: -x[1])[:top_k]
