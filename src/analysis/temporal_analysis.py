"""Weekly aggregates, spike flags, embedding clusters, PCA plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def weekly_counts(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    d = df.copy()
    d["week"] = d["posted_at"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    pivot = d.groupby(["week", pred_col]).size().unstack(fill_value=0).sort_index()
    return pivot


def spike_flags(series: pd.Series, z: float = 1.5) -> pd.Series:
    sigma = series.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(False, index=series.index)
    return ((series - series.mean()) / sigma).abs() >= z


def figure_weekly_stacked(counts: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in counts.columns:
        fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts[col], name=str(col)))
    fig.update_layout(barmode="stack", title=title, template="plotly_white", hovermode="x unified")
    return fig


def figure_spike_line(series: pd.Series, spikes: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index.astype(str), y=series.values, name="weekly total"))
    spike_idx = spikes[spikes].index
    fig.add_trace(
        go.Scatter(
            x=spike_idx.astype(str),
            y=series.reindex(spike_idx).values,
            mode="markers",
            name="spike",
            marker=dict(size=12, color="firebrick"),
        )
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig


def cluster_texts_kmeans(embeddings: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(embeddings)


def figure_embedding_pca(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    hover_text: list[str],
    title: str = "Embeddings (PCA-2) by KMeans cluster",
) -> go.Figure:
    xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    short = [t[:160] + "…" if len(t) > 160 else t for t in hover_text]
    fig = go.Figure(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(size=8, color=cluster_ids, colorscale="Viridis", showscale=True),
            text=short,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig
