"""
Substance Deep-Dive Analyzer.
AI-driven intelligence exploring the 'Depth of Distress' (DoD) and 
Substance Vulnerability Index (SVI) for Alcohol and Drug abuse.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "processed_drug_reviews.parquet"

# ── Professional Chart Styling ──
PROF_FONT = dict(family="Inter, Arial, sans-serif", size=14, color="#1e293b")
PROF_TITLE = dict(family="Inter, Arial, sans-serif", size=18, color="#0f172a")

def calculate_substance_vulnerability() -> pd.DataFrame:
    """Computes the Substance Vulnerability Index (SVI)."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Group by condition/substance
    sub_stats = df.groupby('condition').agg({
        'rating': 'mean',
        'usefulCount': 'mean',
        'sentiment': lambda x: (x == 'negative').mean() * 100
    }).reset_index()
    
    sub_stats.columns = ['Substance', 'Avg Rating', 'Helpfulness Utility', 'Negative Sentiment %']
    
    # 2. Normalize and compute SVI: (Negative % + (10 - Rating) + Utility/10) / 3
    # Higher = More Vulnerable/Distressing
    sub_stats['Distress Score'] = (sub_stats['Negative Sentiment %'] / 10 + (10 - sub_stats['Avg Rating']) + sub_stats['Helpfulness Utility'] / 10) / 3
    
    # 3. Crisis Probability (Synthetic AI Logic based on High distress keywords in reviews)
    crisis_keywords = ['suicide', 'overdose', 'relapse', 'crashing', 'withdrawal', 'shame', 'hopeless']
    df['is_crisis'] = df['review'].str.contains('|'.join(crisis_keywords), case=False).astype(int)
    
    crisis_stats = df.groupby('condition')['is_crisis'].mean().reset_index()
    crisis_stats.columns = ['Substance', 'Crisis Probability']
    
    final_df = pd.merge(sub_stats, crisis_stats, on='Substance')
    return final_df

def figure_vulnerability_radar(df: pd.DataFrame) -> go.Figure:
    """Vulnerability Radar chart: Multi-dimensional distress profiling."""
    fig = go.Figure()
    
    # Select top 5 substances for comparative clarity
    plot_df = df.sort_values('Distress Score', ascending=False).head(5)
    
    for _, row in plot_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Avg Rating']*10, row['Helpfulness Utility'], row['Negative Sentiment %'], row['Crisis Probability']*100, row['Distress Score']*10],
            theta=['Efficacy (Ratingx10)', 'Community Utility', 'Negative Sentiment %', 'Crisis Prob %', 'SVI Intensity'],
            fill='toself',
            name=row['Substance']
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            angularaxis=dict(tickfont=dict(size=12, family="Inter, Arial, sans-serif"))
        ),
        title=dict(text="Substance Vulnerability Radar: Depth of Distress Analysis", font=PROF_TITLE),
        font=PROF_FONT,
        legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5, font=dict(size=13)),
        template="plotly_white",
        margin=dict(t=100, b=100)
    )
    return fig

def figure_distress_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap correlating Efficacy vs Distress."""
    fig = px.density_heatmap(df, x="Avg Rating", y="Negative Sentiment %", 
                            z="Crisis Probability", 
                            marginal_x="rug", marginal_y="histogram",
                            color_continuous_scale="Reds",
                            title="Distress Heatmap: Efficacy vs. Negative Intensity Correlation",
                            labels={'Avg Rating': 'Clinical Efficacy', 'Negative Sentiment %': 'Distress Intensity'})
    
    fig.update_layout(font=PROF_FONT, title=dict(font=PROF_TITLE))
    return fig

if __name__ == "__main__":
    svi_df = calculate_substance_vulnerability()
    print("Substance Vulnerability Intelligence Ready.")
    print(svi_df.sort_values('Distress Score', ascending=False))
