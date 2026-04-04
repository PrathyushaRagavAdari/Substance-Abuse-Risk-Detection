"""
Temporal & Behavioral Analysis Module.
Professional-grade time-series visualizations tracking substance abuse risk patterns,
narrative evolution, and risk intensity across drug & alcohol categories.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Professional Chart Styling ──
FONT = dict(family="Inter, Arial, sans-serif", size=14, color="#1e293b")
TITLE_FONT = dict(family="Inter, Arial, sans-serif", size=18, color="#0f172a")
AXIS_FONT = dict(family="Inter, Arial, sans-serif", size=13, color="#334155")


def generate_spike_detection_data() -> pd.DataFrame:
    """Monthly risk signal data with Z-score spike detection."""
    rng = np.random.default_rng(42)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    
    base = 120 + rng.normal(0, 15, 24)
    # Inject spikes at specific months (winter holidays, summer)
    base[2] += 45   # March 2023
    base[7] += 60   # August 2023
    base[11] += 55  # December 2023
    base[14] += 40  # March 2024
    base[19] += 50  # August 2024
    
    z_scores = (base - base.mean()) / base.std()
    
    return pd.DataFrame({
        "Month": months,
        "Risk Reports": np.maximum(base, 50).astype(int),
        "Z-Score": np.round(z_scores, 2),
        "Is Spike": z_scores > 1.5,
        "Historical Avg": int(base.mean()),
    })


def figure_spike_detection(df: pd.DataFrame) -> go.Figure:
    """Spike Detection: months where risk significantly exceeded historical averages."""
    fig = go.Figure()
    
    # Baseline area
    fig.add_trace(go.Scatter(
        x=df["Month"], y=[df["Historical Avg"].iloc[0]] * len(df),
        mode="lines", name="Historical Average",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        fill=None,
    ))
    
    # Risk reports line
    normal = df[~df["Is Spike"]]
    spike = df[df["Is Spike"]]
    
    fig.add_trace(go.Scatter(
        x=df["Month"], y=df["Risk Reports"],
        mode="lines+markers", name="Monthly Risk Reports",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=8, color="#3b82f6"),
        text=[f"Z: {z:.2f}" for z in df["Z-Score"]],
        hovertemplate="<b>%{x|%b %Y}</b><br>Reports: %{y}<br>%{text}<extra></extra>"
    ))
    
    # Spike markers
    if not spike.empty:
        fig.add_trace(go.Scatter(
            x=spike["Month"], y=spike["Risk Reports"],
            mode="markers", name="⚠ Spike Detected",
            marker=dict(size=18, color="#ef4444", symbol="triangle-up",
                        line=dict(width=2, color="#991b1b")),
            hovertemplate="<b>⚠ SPIKE: %{x|%b %Y}</b><br>Reports: %{y}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Spike Detection: Monthly Drug & Alcohol Risk Signals", font=TITLE_FONT),
        xaxis=dict(title=dict(text="Month", font=AXIS_FONT), tickfont=dict(size=12), dtick="M2", tickformat="%b %Y"),
        yaxis=dict(title=dict(text="Number of Risk Reports", font=AXIS_FONT), tickfont=dict(size=12)),
        template="plotly_white", font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13)),
        hovermode="x unified",
    )
    return fig


def generate_narrative_evolution_data() -> pd.DataFrame:
    """Theme composition evolving over quarters."""
    rng = np.random.default_rng(42)
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    themes = ["Withdrawal & Crisis", "Treatment Progress", "Relapse & Triggers", "Recovery & Support"]
    
    records = []
    for q in quarters:
        vals = rng.dirichlet([3, 4, 2, 3]) * 100
        for t, v in zip(themes, vals):
            records.append({"Quarter": q, "Theme": t, "Percentage": round(v, 1)})
    return pd.DataFrame(records)


def figure_narrative_evolution(df: pd.DataFrame) -> go.Figure:
    """Narrative Evolution: how patient themes change over time."""
    theme_colors = {
        "Withdrawal & Crisis": "#ef4444",
        "Treatment Progress": "#3b82f6",
        "Relapse & Triggers": "#f59e0b",
        "Recovery & Support": "#22c55e",
    }
    fig = px.area(df, x="Quarter", y="Percentage", color="Theme",
                  color_discrete_map=theme_colors,
                  template="plotly_white",
                  title="Narrative Evolution: Patient Theme Composition Over Time")
    fig.update_layout(
        title=dict(font=TITLE_FONT),
        xaxis=dict(title=dict(text="Quarter", font=AXIS_FONT), tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="Percentage of Patient Narratives (%)", font=AXIS_FONT), 
                   tickfont=dict(size=12), ticksuffix="%"),
        font=FONT,
        legend=dict(font=dict(size=14), title=dict(text="Theme", font=dict(size=14))),
    )
    return fig


def generate_risk_intensity_data() -> pd.DataFrame:
    """Average risk score per substance class over time."""
    rng = np.random.default_rng(42)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    
    substances = {
        "Opioids": 7.2 + rng.normal(0, 0.5, 24).cumsum() * 0.05,
        "Alcohol": 5.8 + rng.normal(0, 0.4, 24).cumsum() * 0.03,
        "Stimulants": 6.5 + rng.normal(0, 0.6, 24).cumsum() * 0.04,
        "Tobacco/Nicotine": 4.2 + rng.normal(0, 0.3, 24).cumsum() * 0.02,
    }
    
    records = []
    for sub, scores in substances.items():
        for m, s in zip(months, scores):
            records.append({"Month": m, "Substance Class": sub, "Avg Risk Score": round(np.clip(s, 1, 10), 2)})
    return pd.DataFrame(records)


def figure_risk_intensity(df: pd.DataFrame) -> go.Figure:
    """Risk Intensity: average risk score per substance class over time."""
    sub_colors = {
        "Opioids": "#ef4444",
        "Alcohol": "#f59e0b",
        "Stimulants": "#8b5cf6",
        "Tobacco/Nicotine": "#64748b",
    }
    fig = px.line(df, x="Month", y="Avg Risk Score", color="Substance Class",
                  color_discrete_map=sub_colors,
                  template="plotly_white",
                  title="Risk Intensity: Average Risk Score by Substance Class")
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        title=dict(font=TITLE_FONT),
        xaxis=dict(title=dict(text="Month", font=AXIS_FONT), tickfont=dict(size=12), 
                   dtick="M3", tickformat="%b %Y"),
        yaxis=dict(title=dict(text="Average Risk Score (1-10)", font=AXIS_FONT), 
                   tickfont=dict(size=12), range=[0, 10]),
        font=FONT,
        legend=dict(font=dict(size=14), title=dict(text="Substance", font=dict(size=14))),
        hovermode="x unified",
    )
    return fig
