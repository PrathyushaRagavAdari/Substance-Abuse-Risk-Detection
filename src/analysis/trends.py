"""Plot helpers for CDC drug mortality trends."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def figure_deaths_by_drug(
    df: pd.DataFrame,
    drugs: list[str],
    title: str = "Provisional drug-involved overdose deaths (CDC)",
) -> go.Figure:
    d = df[df["drug_involved"].isin(drugs)].copy().sort_values("period")
    fig = px.line(
        d,
        x="period",
        y="deaths",
        color="drug_involved",
        markers=True,
        title=title,
        labels={"period": "Month", "deaths": "Deaths (12 month-ending)", "drug_involved": "Drug"},
    )
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


def figure_yoy_spike(df: pd.DataFrame, drug: str = "Fentanyl") -> go.Figure:
    sub = df[df["drug_involved"] == drug].copy().sort_values("period")
    sub["cal_year"] = sub["period"].dt.year
    last_per_year = sub.groupby("cal_year", as_index=False).last()
    fig = px.bar(
        last_per_year,
        x="period",
        y="deaths",
        title=f"Latest month in dataset per calendar year — {drug}",
        labels={"deaths": "Deaths (12 month-ending)", "period": "Month ending"},
    )
    fig.update_layout(template="plotly_white")
    return fig
