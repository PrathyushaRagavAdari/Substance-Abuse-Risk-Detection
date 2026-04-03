"""Compare rule / TF–IDF / embedding classifiers on one stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_processing.corpus_utils import corpus_plain_lists
from src.utils.embedding_classifier import EmbeddingClassifier
from src.utils.rule_signals import score_rules
from src.utils.stratified_split import stratified_train_test_indices
from src.core.text_model import build_pipeline


def _rule_predict(texts: list[str]) -> np.ndarray:
    return np.array([score_rules(t).prediction() for t in texts])


def compare_on_corpus(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    texts, labels = corpus_plain_lists(df)
    idx_tr, idx_te = stratified_train_test_indices(labels, test_size, random_state)

    X_tr = [str(texts[int(i)]) for i in idx_tr]
    X_te = [str(texts[int(i)]) for i in idx_te]
    y_tr = [str(labels[int(i)]) for i in idx_tr]
    y_te = [str(labels[int(i)]) for i in idx_te]

    rule_pred = _rule_predict(X_te)
    tf_pipe = build_pipeline()
    tf_pipe.fit(X_tr, y_tr)
    tf_pred = tf_pipe.predict(X_te)

    enc = EmbeddingClassifier()
    enc.fit(X_tr, y_tr)
    enc_pred = enc.predict(X_te)

    # Random Forest with TF-IDF
    rf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    rf_pipe.fit(X_tr, y_tr)
    rf_pred = rf_pipe.predict(X_te)

    # SVM with TF-IDF
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ("clf", SVC(probability=True, kernel="linear", random_state=random_state))
    ])
    svm_pipe.fit(X_tr, y_tr)
    svm_pred = svm_pipe.predict(X_te)

    y_te_arr = np.asarray(y_te, dtype=object)
    rows = [
        {
            "method": "rule_lexicon",
            "accuracy": accuracy_score(y_te, rule_pred),
            "macro_f1": f1_score(y_te, rule_pred, average="macro", zero_division=0),
        },
        {
            "method": "tfidf_logistic",
            "accuracy": accuracy_score(y_te, tf_pred),
            "macro_f1": f1_score(y_te, tf_pred, average="macro", zero_division=0),
        },
        {
            "method": "minilm_embedding_lr",
            "accuracy": accuracy_score(y_te, enc_pred),
            "macro_f1": f1_score(y_te, enc_pred, average="macro", zero_division=0),
        },
        {
            "method": "tfidf_random_forest",
            "accuracy": accuracy_score(y_te, rf_pred),
            "macro_f1": f1_score(y_te, rf_pred, average="macro", zero_division=0),
        },
        {
            "method": "tfidf_svm",
            "accuracy": accuracy_score(y_te, svm_pred),
            "macro_f1": f1_score(y_te, svm_pred, average="macro", zero_division=0),
        },
    ]
    return pd.DataFrame(rows), {
        "tfidf_pipe": tf_pipe,
        "embedding_clf": enc,
        "X_test": X_te,
        "y_test": y_te_arr,
    }
