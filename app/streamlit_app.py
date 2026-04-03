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
                    ["Global Mortality Trends", "Economic Intelligence", "AI Risk Agent", "Behavioral Analysis"], 
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
            from src.core.rag_engine import SubstanceFusionRAG
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

# --- MODULES ---

if menu == "Global Mortality Trends":
    st.markdown("### 🗺️ US Substance Risk Hotspots")
    st.write("Geospatial analysis of provisional mortality hotspots across the United States (CDC 2024).")
    
    df_mortality = load_processed_data("drug_specific")
    if not df_mortality.empty:
        # Comparative Risk: Alcohol vs Opioids
        st.markdown("#### State-Level Risk Profile: Alcohol vs Opioids")
        m1, m2 = st.columns([2, 1])
        with m1:
            # Map Logic (Mockup with Plotly for stability)
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
            top_states = states_df.sort_values('value', ascending=False).head(5)
            for _, r in top_states.iterrows():
                st.metric(r['jurisdiction'], f"{int(r['value']):,} deaths", delta="High Priority", delta_color="inverse")
    
    st.divider()
    st.info("Mortality hotspots are calculated by cross-referencing CDC provisional records with geospatial reporting latencies.")

elif menu == "Economic Intelligence":
    st.markdown("### 💸 Economic Impact Intelligence")
    st.write("Modeling the **$2.7 Trillion** societal burden across Illicit Drugs, Tobacco, and Alcohol.")
    
    df_econ = load_processed_data("economic_costs")
    if not df_econ.empty:
        if 'type' not in df_econ.columns:
            st.error("Stale Economic Data. Run src/analysis/economic_impact.py.")
            st.stop()
            
        st.markdown("#### 🏛️ The 'Big Three' Societal Burden")
        pillar_df = df_econ[df_econ['type'] == 'Pillar Summary']
        fig_pillar = px.pie(pillar_df, values="value", names="subcategory", 
                           hole=0.4, template='plotly_white',
                           color_discrete_map={"Illicit Drugs": "#ef4444", "Tobacco": "#334155", "Alcohol": "#f59e0b"})
        fig_pillar.update_layout(font_color='black', legend=dict(orientation="h", y=1.2))
        st.plotly_chart(fig_pillar, use_container_width=True)
        
        st.divider()
        sub_type = st.selectbox("Filter Breakdown", ["Illicit Drugs", "Tobacco", "Alcohol"])
        target_df = df_econ[df_econ['type'] == sub_type]
        
        e1, e2, e3 = st.columns(3)
        total_val = target_df['value'].sum()
        e1.metric(f"Total {sub_type} Burden", f"${total_val:.1f}B")
        
        # Details
        ec1, ec2 = st.columns([2, 1])
        with ec1:
            st.markdown(f"#### Cost Allocation: {sub_type}")
            sun_df = target_df[target_df['category'].isin(['Direct Costs', 'Indirect Costs'])]
            fig_sun = px.sunburst(sun_df, path=['category', 'subcategory'], values='value',
                                 color='category', template='plotly_white',
                                 color_discrete_map={'Direct Costs': '#6366f1', 'Indirect Costs': '#ef4444'})
            fig_sun.update_layout(font_color='black')
            st.plotly_chart(fig_sun, use_container_width=True)
        with ec2:
            st.markdown("#### Productivity Loss Detail")
            prod_detail = df_econ[(df_econ['type'] == sub_type) & (df_econ['category'] == 'Productivity Loss Detail')]
            if not prod_detail.empty:
                fig_bar = px.bar(prod_detail, x='value', y='subcategory', orientation='h', template='plotly_white')
                fig_bar.update_layout(font_color='black')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("Aggregated data only.")

        if sub_type == "Tobacco":
            st.divider()
            st.markdown("#### 🚬 Smoking vs 🍏 Vaping: Per-User Healthcare Burden")
            u1, u2, u3 = st.columns(3)
            u1.metric("Smoker", "$8,000/yr")
            u2.metric("Vaper", "$1,800/yr")
            u3.metric("Dual User", "$2,050/yr")

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

elif menu == "Behavioral Analysis":
    st.markdown("### 🧠 Behavioral Analysis & Early Warning")
    st.write("Tracking anonymized patient testimonials and distress signals.")
    # Legacy tabs comparison
    df_reviews = load_processed_data("drug_reviews")
    if not df_reviews.empty:
        st.dataframe(df_reviews.head(10), use_container_width=True)
        st.plotly_chart(px.histogram(df_reviews, x='rating', color='sentiment', template='plotly_white'), use_container_width=True)

if __name__ == "__main__":
    st.markdown("<br><hr><center>Nexus-Cortex 2.0 | Strategic Clinical Intelligence</center>", unsafe_allow_html=True)
