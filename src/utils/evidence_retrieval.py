"""TF–IDF cosine similarity to labeled corpus lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfEvidenceIndex:
    def __init__(self, max_features: int = 4000):
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        self._texts: list[str] = []
        self._labels: list[str] = []
        self._mat = None

    def fit(self, df: pd.DataFrame, text_col: str = "text", label_col: str = "label"):
        n = len(df)
        self._texts = [str(df[text_col].iloc[i]) for i in range(n)]
        self._labels = [str(df[label_col].iloc[i]) for i in range(n)]
        self._mat = self._vec.fit_transform(self._texts)
        return self

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        if self._mat is None:
            raise RuntimeError("Call fit first.")
        q = self._vec.transform([text])
        sim = cosine_similarity(q, self._mat).ravel()
        idx = np.argsort(-sim)[:top_k]
        return [
            {"text": self._texts[int(i)], "label": self._labels[int(i)], "similarity": float(sim[int(i)])}
            for i in idx
        ]
