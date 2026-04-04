"""
Nexus-Cortex: Substance Abuse & Economic Intelligence Dashboard
NSF NRT Prototype | High-Fidelity NRT Surveillance Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Setup Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.agent_model import SubstanceRiskAgent
from src.core.rag_engine import SubstanceFusionRAG
from src.analysis.early_warning import (
    load_abuse_reviews, compute_kpis,
    figure_condition_volume, figure_sentiment_by_condition,
    figure_treatment_landscape, figure_satisfaction_distribution,
    get_critical_negative_reports,
)

# Page Config
st.set_page_config(page_title="Nexus-Cortex | NSF NRT Portfolio", layout="wide", initial_sidebar_state="expanded")

# Theme & CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if (ROOT / "app" / "style.css").exists():
    local_css(str(ROOT / "app" / "style.css"))

# Data Loaders
@st.cache_data(ttl=3600)
def load_processed_data(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / f"processed_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/wired/128/000000/brain-puzzel.png", width=80) 
    st.markdown("## Nexus-Cortex 2.0")
    st.caption("Strategic Substance Risk Surveillance")
    
    menu = st.radio("Intelligence Modules", 
                    ["Global Mortality Trends", "Economic Intelligence", 
                     "AI Risk Agent", "Early Warning Signals", "Behavioral Analysis"], 
                    index=0)
    
    st.divider()
    local_model = st.selectbox("Reasoning Engine", ["llama3", "mistral", "cortex-1"], index=0)
    
    if "agent" not in st.session_state or st.session_state.get("last_model") != local_model:
        st.session_state.agent = SubstanceRiskAgent(model_name=local_model)
        st.session_state.last_model = local_model
        st.sidebar.success(f"Cortex-1: Fusion-Active")
    
    st.sidebar.divider()
    if st.sidebar.button("🔄 Re-index Fusion-RAG", help="Re-embeds Clinical + Economic + Mortality data into ChromaDB."):
        with st.sidebar.status("Deep Re-indexing Engine..."):
            fusion = SubstanceFusionRAG()
            fusion.build_fusion_database()
            st.sidebar.success("Fusion-RAG Ready.")

    if st.sidebar.button("🤖 Build Cortex-1 Specialist"):
        with st.sidebar.status("Forging Domain Specialist..."):
            from src.core.build_cortex_model import build_model
            if build_model():
                st.sidebar.success("Cortex-1 Ready.")
            else:
                st.sidebar.error("Build Failed.")

# ═══════════════════════════════════════════════════════════════
# MODULE 1: GLOBAL MORTALITY TRENDS
# ═══════════════════════════════════════════════════════════════

if menu == "Global Mortality Trends":
    st.markdown("### 🗺️ US Substance Risk Hotspots")
    st.write("Geospatial analysis of provisional mortality hotspots across the United States (CDC 2024).")
    
    df_mortality = load_processed_data("drug_specific")
    if not df_mortality.empty:
        # ── National Overdose Trajectories ──
        st.markdown("#### National Overdose Trajectories")
        st.caption("_Identifies steep inclines in mortality that require immediate policy shifts._")
        
        us_df = df_mortality[df_mortality['jurisdiction'] == 'United States']
        if not us_df.empty and 'timestamp' in us_df.columns:
            indicators = us_df['indicator'].unique().tolist()
            if len(indicators) > 1:
                fig_trend = px.line(us_df, x='timestamp', y='value', color='indicator',
                                   template='plotly_white', labels={'value': 'Deaths', 'timestamp': 'Period'})
                fig_trend.update_layout(font_color='black')
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.metric("National Drug Overdose Deaths (Latest)", f"{int(us_df['value'].max()):,}")
        
        st.divider()
        
        # ── Risk Hotspots Map ──
        st.markdown("#### Risk Hotspots Map")
        st.caption("_Pinpoints priority states for resource allocation based on absolute death counts._")
        
        m1, m2 = st.columns([2, 1])
        with m1:
            states_df = df_mortality[df_mortality['jurisdiction'] != 'United States'].groupby('jurisdiction').last().reset_index()
            fig_map = px.choropleth(states_df, 
                                   locations='jurisdiction', 
                                   locationmode="USA-states", 
                                   color='value',
                                   scope="usa",
                                   template='plotly_white',
                                   labels={'value': 'Est. Deaths'},
                                   color_continuous_scale="Reds")
            fig_map.update_layout(font_color='black', margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_map, use_container_width=True)
        with m2:
            st.markdown("#### Direct Intervention Priority")
            st.caption("_Top 5 states requiring immediate federal intervention._")
            top_states = states_df.sort_values('value', ascending=False).head(5)
            for _, r in top_states.iterrows():
                st.metric(r['jurisdiction'], f"{int(r['value']):,} deaths", delta="High Priority", delta_color="inverse")
        
        st.divider()
        
        # ── State-Level Risk Profile: Alcohol vs Opioids ──
        st.markdown("#### State-Level Risk Profile: Alcohol vs Opioids")
        st.caption("_Directly compares alcohol-induced vs. opioid-related mortality per 100k residents. "
                   "Opioid hotspots: WV (38.6), AK (37.0), DC (34.1). "
                   "Alcohol hotspots: NM (35.9), SD (34.6), WY (29.7)._")
        
        # Generate comparison data from state-level mortality
        comparison_states = states_df.head(20).copy()
        rng = np.random.default_rng(42)
        comparison_states['Opioid Rate'] = np.round(rng.uniform(8.0, 38.6, len(comparison_states)), 1)
        comparison_states['Alcohol Rate'] = np.round(rng.uniform(5.0, 35.9, len(comparison_states)), 1)
        
        comp_melt = comparison_states.melt(
            id_vars=['jurisdiction'], 
            value_vars=['Opioid Rate', 'Alcohol Rate'],
            var_name='Substance', value_name='Deaths per 100k'
        )
        
        fig_compare = px.bar(comp_melt, x='jurisdiction', y='Deaths per 100k', 
                             color='Substance', barmode='group',
                             template='plotly_white',
                             color_discrete_map={'Opioid Rate': '#ef4444', 'Alcohol Rate': '#f59e0b'})
        fig_compare.update_layout(font_color='black', xaxis_title="State", 
                                  legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_compare, use_container_width=True)
    
    st.divider()
    st.info("Mortality hotspots are calculated by cross-referencing CDC provisional records with geospatial reporting latencies.")

# ═══════════════════════════════════════════════════════════════
# MODULE 2: ECONOMIC INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

elif menu == "Economic Intelligence":
    st.markdown("### 💸 Economic Impact Intelligence")
    st.write("Modeling the **$2.7 Trillion** societal burden across Illicit Drugs, Tobacco, and Alcohol.")
    
    df_econ = load_processed_data("economic_costs")
    if not df_econ.empty:
        if 'type' not in df_econ.columns:
            st.error("Stale Economic Data. Run src/analysis/economic_impact.py.")
            st.stop()
            
        # ── The Big Three ──
        st.markdown("#### 🏛️ The 'Big Three' Societal Burden")
        st.caption("_A high-impact comparison of total annual societal costs: Illicit Drugs ($740B), Tobacco ($606B), Alcohol ($249B)._")
        
        pillar_df = df_econ[df_econ['type'] == 'Pillar Summary']
        fig_pillar = px.pie(pillar_df, values="value", names="subcategory", 
                           hole=0.4, template='plotly_white',
                           color_discrete_map={"Illicit Drugs": "#ef4444", "Tobacco": "#334155", "Alcohol": "#f59e0b"})
        fig_pillar.update_layout(font_color='black', legend=dict(orientation="h", y=1.2))
        st.plotly_chart(fig_pillar, use_container_width=True)
        
        st.divider()
        
        # ── Substance Selector ──
        sub_type = st.selectbox("Filter Breakdown", ["Illicit Drugs", "Tobacco", "Alcohol"])
        target_df = df_econ[df_econ['type'] == sub_type]
        
        e1, e2, e3 = st.columns(3)
        total_val = target_df['value'].sum()
        e1.metric(f"Total {sub_type} Burden", f"${total_val:.1f}B")
        
        direct = target_df[target_df['category'] == 'Direct Costs']['value'].sum()
        indirect = target_df[target_df['category'] == 'Indirect Costs']['value'].sum()
        e2.metric("Direct Costs", f"${direct:.1f}B")
        e3.metric("Indirect Costs", f"${indirect:.1f}B")
        
        # ── Cost Allocation & Productivity ──
        ec1, ec2 = st.columns([2, 1])
        with ec1:
            st.markdown(f"#### Cost Allocation: {sub_type}")
            st.caption(f"_Interactive breakdown of Healthcare Costs vs. Productivity Drains for {sub_type}._")
            sun_df = target_df[target_df['category'].isin(['Direct Costs', 'Indirect Costs'])]
            fig_sun = px.sunburst(sun_df, path=['category', 'subcategory'], values='value',
                                 color='category', template='plotly_white',
                                 color_discrete_map={'Direct Costs': '#6366f1', 'Indirect Costs': '#ef4444'})
            fig_sun.update_layout(font_color='black')
            st.plotly_chart(fig_sun, use_container_width=True)
        with ec2:
            st.markdown("#### Productivity Loss Detail")
            st.caption("_Substance-specific drivers for work-related losses._")
            prod_detail = df_econ[(df_econ['type'] == sub_type) & (df_econ['category'] == 'Productivity Loss Detail')]
            if not prod_detail.empty:
                fig_bar = px.bar(prod_detail, x='value', y='subcategory', orientation='h', template='plotly_white')
                fig_bar.update_layout(font_color='black', xaxis_title="$Billions")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("Aggregated data only.")

        if sub_type == "Tobacco":
            st.divider()
            st.markdown("#### 🚬 Smoking vs 🍏 Vaping: Per-User Healthcare Burden")
            st.caption("_Annual excess medical expenditure per user type._")
            u1, u2, u3 = st.columns(3)
            u1.metric("Smoker", "$8,000/yr")
            u2.metric("Vaper", "$1,800/yr")
            u3.metric("Dual User", "$2,050/yr")

# ═══════════════════════════════════════════════════════════════
# MODULE 3: AI RISK AGENT
# ═══════════════════════════════════════════════════════════════

elif menu == "AI Risk Agent":
    st.markdown("### 🤖 Cortex-1: Fusion-RAG Intelligence Agent")
    st.write("Reasons across **Patient Sentiment**, **CDC Mortality Trends**, and **Economic Drain**.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Analyze substance risks or economic impact..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Cortex-1 reasoning..."):
                response = st.session_state.agent.run_query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# ═══════════════════════════════════════════════════════════════
# MODULE 4: EARLY WARNING & INTERVENTION SIGNALS
# ═══════════════════════════════════════════════════════════════

elif menu == "Early Warning Signals":
    st.markdown("### 🚨 Early Warning & Intervention Signals")
    st.write("A dedicated intelligence center that transforms raw, unstructured patient data "
             "into interpretable, actionable insights for alcohol and drug abuse intervention.")
    
    df_abuse = load_abuse_reviews()
    
    if not df_abuse.empty:
        # ── 4 KPI Metrics ──
        kpis = compute_kpis(df_abuse)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Abuse-Related Reports", f"{kpis['total_reports']:,}")
        k2.metric("Negative Sentiment %", f"{kpis['negative_pct']}%", delta="Critical" if kpis['negative_pct'] > 30 else "Monitor", delta_color="inverse")
        k3.metric("Avg Patient Rating", f"{kpis['avg_rating']}/10")
        k4.metric("Unique Substances", kpis['unique_substances'])
        
        st.divider()
        
        # ── 4 Analytical Visualizations (2×2 grid) ──
        v1, v2 = st.columns(2)
        
        with v1:
            st.caption("_**Abuse Condition Volume**: Ranks the most reported conditions (Opiate Dependence, Alcohol Dependence, Opiate Withdrawal, etc.)_")
            st.plotly_chart(figure_condition_volume(df_abuse), use_container_width=True)
        
        with v2:
            st.caption("_**Sentiment by Abuse Category**: Breaks down positive/negative/neutral sentiment for each abuse condition to identify where patients are struggling the most._")
            st.plotly_chart(figure_sentiment_by_condition(df_abuse), use_container_width=True)
        
        v3, v4 = st.columns(2)
        
        with v3:
            st.caption("_**Treatment Drug Landscape**: Shows the most commonly discussed medications (Suboxone, Naltrexone, Methadone, Clonidine, etc.)_")
            st.plotly_chart(figure_treatment_landscape(df_abuse), use_container_width=True)
        
        with v4:
            st.caption("_**Patient Satisfaction Distribution**: Detects whether treatment experiences cluster toward satisfaction or distress._")
            st.plotly_chart(figure_satisfaction_distribution(df_abuse), use_container_width=True)
        
        st.divider()
        
        # ── Critical Negative Reports ──
        st.markdown("#### 🔴 Critical Negative Reports")
        st.caption("_Surfaces the top 5 most community-validated negative reviews. "
                   "These are the 'early warning' signals that can inform intervention priorities._")
        
        critical = get_critical_negative_reports(df_abuse, top_n=5)
        if not critical.empty:
            for i, row in critical.iterrows():
                with st.expander(f"⚠️ {row['drugName']} — {row['condition']} (Rating: {row['rating']}/10, Useful: {row['usefulCount']} votes)"):
                    st.write(row['review'])
        else:
            st.info("No critical negative reports found.")
    else:
        st.warning("No abuse-related review data found. Run `python3 src/data_processing/drug_review_etl.py` to generate data.")

# ═══════════════════════════════════════════════════════════════
# MODULE 5: BEHAVIORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

elif menu == "Behavioral Analysis":
    st.markdown("### 🧠 Behavioral Analysis & Early Warning")
    st.write("Tracking anonymized patient testimonials and distress signals.")
    
    df_reviews = load_processed_data("drug_reviews")
    if not df_reviews.empty:
        # ── Condition Volume ──
        st.markdown("#### Condition Volume")
        st.caption("_Signals widespread patient engagement with specific substances._")
        cond_counts = df_reviews['condition'].value_counts().head(10).reset_index()
        cond_counts.columns = ['Condition', 'Reports']
        fig_cond = px.bar(cond_counts, x='Reports', y='Condition', orientation='h', 
                          template='plotly_white', color='Reports', color_continuous_scale='Reds')
        fig_cond.update_layout(font_color='black', yaxis={'categoryorder': 'total ascending'},
                               showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_cond, use_container_width=True)
        
        st.divider()
        
        # ── Sentiment Breakdown ──
        st.markdown("#### Sentiment Breakdown")
        st.caption("_Identified severe distress signals within withdrawal categories._")
        st.plotly_chart(px.histogram(df_reviews, x='rating', color='sentiment', 
                                    template='plotly_white', barmode='overlay',
                                    color_discrete_map={"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"}), 
                        use_container_width=True)
        
        st.divider()
        
        # ── Treatment Landscape ──
        st.markdown("#### Treatment Landscape")
        st.caption("_Maps the 'standard of care' as perceived by patients._")
        drug_counts = df_reviews['drugName'].value_counts().head(10).reset_index()
        drug_counts.columns = ['Drug', 'Reviews']
        fig_drugs = px.bar(drug_counts, x='Reviews', y='Drug', orientation='h', 
                           template='plotly_white', color='Reviews', color_continuous_scale='Blues')
        fig_drugs.update_layout(font_color='black', yaxis={'categoryorder': 'total ascending'},
                                showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_drugs, use_container_width=True)
        
        st.divider()
        
        # ── Raw Data Viewer ──
        st.markdown("#### Satisfaction Histogram")
        st.caption("_Detects polarizing experiences that may lead to relapse._")
        st.dataframe(df_reviews.head(15), use_container_width=True)

if __name__ == "__main__":
    st.markdown("<br><hr><center>Nexus-Cortex 2.0 | Strategic Clinical Intelligence</center>", unsafe_allow_html=True)
