"""Synthetic timestamps for time-aware demos."""

from __future__ import annotations

import pandas as pd

from src.core.text_model import load_corpus


def load_corpus_timeline(start: str = "2025-06-01", hours_step: int = 17) -> pd.DataFrame:
    df = load_corpus().reset_index(drop=True)
    t0 = pd.Timestamp(start, tz="UTC")
    df["posted_at"] = [t0 + pd.Timedelta(hours=hours_step * i) for i in range(len(df))]
    return df
