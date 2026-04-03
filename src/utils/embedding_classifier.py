"""Sentence embeddings + logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingClassifier:
    def __init__(self, model_name: str = MODEL_NAME):
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers.")
        self._encoder = SentenceTransformer(model_name)
        self._clf = Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)),
            ]
        )
        self._classes: np.ndarray | None = None

    def fit(self, texts: list[str], y) -> "EmbeddingClassifier":
        X = self._encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        self._clf.fit(X, y)
        self._classes = self._clf.named_steps["lr"].classes_
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return self._clf.predict_proba(X)

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self._encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return self._clf.predict(X)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    @property
    def classes_(self):
        if self._classes is None:
            raise RuntimeError("Call fit first.")
        return self._classes
