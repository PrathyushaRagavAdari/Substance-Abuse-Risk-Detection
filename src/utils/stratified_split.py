"""
Stratified train/test index split using only NumPy.

Does not import sklearn — avoids PyArrow issues when sklearn indexes pandas-backed arrays.
"""

from __future__ import annotations

import numpy as np


def stratified_train_test_indices(
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    y = [str(x) for x in labels]
    by_class: dict[str, list[int]] = {}
    for i, lab in enumerate(y):
        by_class.setdefault(lab, []).append(i)

    train_list: list[int] = []
    test_list: list[int] = []
    for _lab in sorted(by_class.keys()):
        idxs = np.asarray(by_class[_lab], dtype=np.int64)
        rng.shuffle(idxs)
        k = len(idxs)
        if k == 1:
            train_list.append(int(idxs[0]))
            continue
        n_test = int(round(k * test_size))
        n_test = max(1, min(n_test, k - 1))
        test_list.extend(int(j) for j in idxs[:n_test])
        train_list.extend(int(j) for j in idxs[n_test:])

    return np.asarray(train_list, dtype=np.int64), np.asarray(test_list, dtype=np.int64)
