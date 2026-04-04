"""
Early Warning & Intervention Signals Module.
Transforms raw patient data into actionable clinical intelligence for drug & alcohol abuse intervention.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "processed_drug_reviews.parquet"

# Abuse-related conditions for filtering
ABUSE_CONDITIONS = [
    "Opiate Dependence", "Alcohol Dependence", "Opiate Withdrawal",
    "Drug Withdrawal", "Smoking Cessation", "Cocaine Dependence",
    "Methamphetamine Addiction", "Cannabis Use Disorder",
    "Benzodiazepine Dependence", "Polysubstance Abuse"
]


def load_abuse_reviews() -> pd.DataFrame:
    """Load and filter reviews to abuse-related conditions only."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_PATH)
    return df[df["condition"].isin(ABUSE_CONDITIONS)]


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the 4 headline KPI metrics."""
    if df.empty:
        return {"total_reports": 0, "negative_pct": 0, "avg_rating": 0, "unique_substances": 0}
    return {
        "total_reports": len(df),
        "negative_pct": round((df["sentiment"] == "negative").mean() * 100, 1),
        "avg_rating": round(df["rating"].mean(), 1),
        "unique_substances": df["drugName"].nunique(),
    }


def figure_condition_volume(df: pd.DataFrame) -> go.Figure:
    """Abuse Condition Volume — ranks the most reported conditions."""
    counts = df["condition"].value_counts().reset_index()
    counts.columns = ["Condition", "Reports"]
    fig = px.bar(counts, x="Reports", y="Condition", orientation="h",
                 template="plotly_white", color="Reports",
                 color_continuous_scale="Reds",
                 title="Abuse Condition Volume: Most Reported Conditions")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      font_color="black", showlegend=False,
                      coloraxis_showscale=False)
    return fig


def figure_sentiment_by_condition(df: pd.DataFrame) -> go.Figure:
    """Sentiment by Abuse Category — positive/negative/neutral breakdown."""
    ct = pd.crosstab(df["condition"], df["sentiment"], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars="condition", var_name="Sentiment", value_name="Percentage")
    color_map = {"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"}
    fig = px.bar(ct, x="Percentage", y="condition", color="Sentiment", orientation="h",
                 template="plotly_white", barmode="stack", color_discrete_map=color_map,
                 title="Sentiment by Abuse Category: Where Patients Struggle Most")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, font_color="black")
    return fig


def figure_treatment_landscape(df: pd.DataFrame) -> go.Figure:
    """Treatment Drug Landscape — most commonly discussed medications."""
    counts = df["drugName"].value_counts().reset_index()
    counts.columns = ["Drug", "Mentions"]
    fig = px.bar(counts, x="Mentions", y="Drug", orientation="h",
                 template="plotly_white", color="Mentions",
                 color_continuous_scale="Blues",
                 title="Treatment Drug Landscape: Most Discussed Medications")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      font_color="black", showlegend=False,
                      coloraxis_showscale=False)
    return fig


def figure_satisfaction_distribution(df: pd.DataFrame) -> go.Figure:
    """Patient Satisfaction Distribution — histogram of ratings colored by sentiment."""
    color_map = {"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"}
    fig = px.histogram(df, x="rating", color="sentiment", nbins=10,
                       template="plotly_white", barmode="overlay",
                       color_discrete_map=color_map,
                       title="Patient Satisfaction Distribution: Treatment Experience")
    fig.update_layout(font_color="black",
                      xaxis_title="Patient Rating (1-10)",
                      yaxis_title="Number of Reports")
    return fig


def get_critical_negative_reports(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Surface the most community-validated negative reviews (highest usefulCount)."""
    neg = df[df["sentiment"] == "negative"].copy()
    if neg.empty:
        return pd.DataFrame()
    return neg.nlargest(top_n, "usefulCount")[
        ["drugName", "condition", "review", "rating", "usefulCount"]
    ].reset_index(drop=True)
