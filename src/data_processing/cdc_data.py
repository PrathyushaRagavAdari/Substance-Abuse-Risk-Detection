"""CDC provisional drug-involved mortality (public aggregate only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

CDC_RESOURCE = "https://data.cdc.gov/resource/8hzs-zshh.json"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def fetch_cdc_provisional_drugs(
    limit: int = 12_000,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if cache_path is None:
        cache_path = CACHE_DIR / f"cdc_8hzs_zshh_limit_{limit}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    r = requests.get(CDC_RESOURCE, params={"$limit": limit}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def load_sample_frame_if_needed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "death_year": "2021",
                "death_month": "6",
                "jurisdiction_occurrence": "United States",
                "drug_involved": "Fentanyl",
                "time_period": "12 month-ending",
                "drug_overdose_deaths": "52000",
            },
        ]
    )


def prepare_trend_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["deaths"] = pd.to_numeric(out["drug_overdose_deaths"], errors="coerce")
    out["year"] = pd.to_numeric(out["death_year"], errors="coerce")
    out["month"] = pd.to_numeric(out["death_month"], errors="coerce")
    out = out.dropna(subset=["deaths", "year", "month"])
    out["period"] = pd.to_datetime(
        dict(year=out["year"].astype(int), month=out["month"].astype(int), day=1)
    )
    return out


def jurisdiction_us_total(df: pd.DataFrame) -> pd.DataFrame:
    m = df["jurisdiction_occurrence"].astype(str).str.strip().eq("United States")
    return df.loc[m].copy()
