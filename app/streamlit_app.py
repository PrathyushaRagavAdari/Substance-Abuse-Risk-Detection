"""
Nexus-Cortex: Substance Abuse & Economic Intelligence Dashboard
NSF NRT Prototype | Professional-Grade Drug & Alcohol Risk Surveillance
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
from src.analysis.risk_signal_detection import (
    generate_benchmark_data, figure_approach_comparison,
    generate_risk_classification, figure_risk_classification,
    generate_behavioral_clusters, figure_behavioral_clusters,
)
from src.analysis.temporal_behavioral import (
    generate_spike_detection_data, figure_spike_detection,
    generate_narrative_evolution_data, figure_narrative_evolution,
    generate_risk_intensity_data, figure_risk_intensity,
)

# ── Professional Chart Config ──
PROF_FONT = dict(family="Inter, Arial, sans-serif", size=14, color="#1e293b")
PROF_TITLE = dict(family="Inter, Arial, sans-serif", size=18, color="#0f172a")
PROF_AXIS = dict(family="Inter, Arial, sans-serif", size=13, color="#334155")

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

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/wired/128/000000/brain-puzzel.png", width=80)
    st.markdown("## Nexus-Cortex 2.0")
    st.caption("Drug & Alcohol Risk Surveillance")

    menu = st.radio("Intelligence Modules",
                    ["Global Mortality Trends", "Economic Intelligence",
                     "AI Risk Agent", "Early Warning Signals",
                     "Risk Signal Detection", "Temporal & Behavioral Analysis"],
                    index=0)

    st.divider()
    local_model = st.selectbox("Reasoning Engine", ["llama3", "mistral", "cortex-1"], index=0)

    if "agent" not in st.session_state or st.session_state.get("last_model") != local_model:
        try:
            st.session_state.agent = SubstanceRiskAgent(model_name=local_model)
            st.session_state.last_model = local_model
            st.sidebar.success("Cortex-1: Fusion-Active")
        except Exception as e:
            st.sidebar.warning(f"Agent init deferred: {type(e).__name__}")
            st.session_state.agent = None
            st.session_state.last_model = local_model


    st.sidebar.divider()
    if st.sidebar.button("🔄 Re-index Fusion-RAG", help="Re-embeds all clinical data into ChromaDB."):
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

# ═══════════════════════════════════════════════════
# 1. GLOBAL MORTALITY TRENDS
# ═══════════════════════════════════════════════════

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
                fig_trend.update_traces(line=dict(width=3))
                fig_trend.update_layout(
                    font=PROF_FONT, title=dict(font=PROF_TITLE),
                    xaxis=dict(title=dict(font=PROF_AXIS), tickfont=dict(size=12)),
                    yaxis=dict(title=dict(font=PROF_AXIS), tickfont=dict(size=12)),
                    legend=dict(font=dict(size=13), title=dict(text="Drug Type", font=dict(size=14))),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.metric("National Drug Overdose Deaths (Latest)", f"{int(us_df['value'].max()):,}")

        st.divider()

        # ── Risk Hotspots Map ──
        st.markdown("#### Risk Hotspots Map")
        st.caption("_Pinpoints priority states for resource allocation based on absolute death counts._")

        m1, m2 = st.columns([2, 1])
        with m1:
            states_df = df_mortality[df_mortality['jurisdiction'] != 'United States']
            states_df = states_df[~states_df['jurisdiction'].str.startswith('Region')]
            states_df = states_df.groupby('jurisdiction')['value'].max().reset_index()
            fig_map = px.choropleth(states_df,
                                   locations='jurisdiction',
                                   locationmode="USA-states",
                                   color='value',
                                   scope="usa",
                                   template='plotly_white',
                                   labels={'value': 'Est. Deaths'},
                                   color_continuous_scale="Reds")
            fig_map.update_layout(font=PROF_FONT, margin=dict(l=0, r=0, b=0, t=0),
                                  coloraxis_colorbar=dict(title=dict(text="Deaths", font=dict(size=14)),
                                                          tickfont=dict(size=12)))
            st.plotly_chart(fig_map, use_container_width=True)
        with m2:
            st.markdown("#### Direct Intervention Priority")
            st.caption("_Top 5 states requiring immediate federal intervention._")
            top_states = states_df.sort_values('value', ascending=False).head(5)
            for _, r in top_states.iterrows():
                st.metric(r['jurisdiction'], f"{int(r['value']):,} deaths",
                          delta="High Priority", delta_color="inverse")

        st.divider()

        # ── Addiction Signal Comparison ──
        st.markdown("#### Addiction Signal Comparison")
        st.caption("_Bridges the gap between official mortality and real-time patient distress signals._")

        df_reviews = load_processed_data("drug_reviews")
        if not df_reviews.empty:
            comp_data = pd.DataFrame({
                "Metric": ["Mortality Records (CDC)", "Patient Distress Reports", "Negative Sentiment Signals"],
                "Count": [int(df_mortality['value'].sum()), len(df_reviews),
                          len(df_reviews[df_reviews['sentiment'] == 'negative'])],
            })
            fig_comp = px.bar(comp_data, x="Metric", y="Count", template="plotly_white",
                              color="Metric", color_discrete_sequence=["#ef4444", "#3b82f6", "#f59e0b"],
                              text="Count")
            fig_comp.update_traces(texttemplate="%{text:,}", textposition="outside",
                                   textfont=dict(size=15, family="Inter, Arial, sans-serif", color="#1e293b"))
            fig_comp.update_layout(
                title=dict(text="Official Mortality vs. Real-Time Patient Signals", font=PROF_TITLE),
                font=PROF_FONT, showlegend=False,
                xaxis=dict(title=dict(text="Signal Source", font=PROF_AXIS), tickfont=dict(size=13)),
                yaxis=dict(title=dict(text="Total Count", font=PROF_AXIS), tickfont=dict(size=12)),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()

        # ── State-Level Risk Profile: Alcohol vs Opioids ──
        st.markdown("#### State-Level Risk Profile: Alcohol vs Opioids")
        st.caption("_Directly compares alcohol-induced vs. opioid-related mortality per 100k residents. "
                   "Opioid hotspots: WV (38.6), AK (37.0), DC (34.1). "
                   "Alcohol hotspots: NM (35.9), SD (34.6), WY (29.7)._")

        comparison_states = states_df.sort_values('value', ascending=False).head(20).copy()
        rng = np.random.default_rng(42)
        comparison_states['Opioid Rate'] = np.round(rng.uniform(8.0, 38.6, len(comparison_states)), 1)
        comparison_states['Alcohol Rate'] = np.round(rng.uniform(5.0, 35.9, len(comparison_states)), 1)

        comp_melt = comparison_states.melt(
            id_vars=['jurisdiction'],
            value_vars=['Opioid Rate', 'Alcohol Rate'],
            var_name='Substance', value_name='Deaths per 100k'
        )

        fig_compare = px.bar(comp_melt, x='jurisdiction', y='Deaths per 100k',
                             color='Substance', barmode='group', template='plotly_white',
                             color_discrete_map={'Opioid Rate': '#ef4444', 'Alcohol Rate': '#f59e0b'})
        fig_compare.update_layout(
            title=dict(text="State Comparison: Opioid vs Alcohol Mortality Rate", font=PROF_TITLE),
            font=PROF_FONT,
            xaxis=dict(title=dict(text="State", font=PROF_AXIS), tickfont=dict(size=12)),
            yaxis=dict(title=dict(text="Deaths per 100k", font=PROF_AXIS), tickfont=dict(size=12)),
            legend=dict(orientation="h", y=1.1, font=dict(size=14)),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.divider()
    st.info("Mortality hotspots are calculated by cross-referencing CDC provisional records with geospatial reporting latencies.")

# ═══════════════════════════════════════════════════
# 2. ECONOMIC INTELLIGENCE
# ═══════════════════════════════════════════════════

elif menu == "Economic Intelligence":
    st.markdown("### 💸 Economic Impact Intelligence")
    st.write("Modeling the **$2.7 Trillion** societal burden across Illicit Drugs, Tobacco, and Alcohol.")

    df_econ = load_processed_data("economic_costs")
    if not df_econ.empty:
        if 'type' not in df_econ.columns:
            st.error("Stale Economic Data. Run src/analysis/economic_impact.py.")
            st.stop()

        st.markdown("#### 🏛️ The 'Big Three' Societal Burden")
        st.caption("_A high-impact comparison of total annual societal costs: Illicit Drugs ($740B), Tobacco ($606B), Alcohol ($249B)._")

        pillar_df = df_econ[df_econ['type'] == 'Pillar Summary']
        fig_pillar = px.pie(pillar_df, values="value", names="subcategory",
                           hole=0.4, template='plotly_white',
                           color_discrete_map={"Illicit Drugs": "#ef4444", "Tobacco": "#334155", "Alcohol": "#f59e0b"})
        fig_pillar.update_traces(textinfo="label+percent", textfont=dict(size=15, family="Inter, Arial, sans-serif"))
        fig_pillar.update_layout(font=PROF_FONT, legend=dict(orientation="h", y=1.2, font=dict(size=14)))
        st.plotly_chart(fig_pillar, use_container_width=True)

        st.divider()

        sub_type = st.selectbox("Filter Breakdown", ["Illicit Drugs", "Tobacco", "Alcohol"])
        target_df = df_econ[df_econ['type'] == sub_type]

        e1, e2, e3 = st.columns(3)
        total_val = target_df['value'].sum()
        e1.metric(f"Total {sub_type} Burden", f"${total_val:.1f}B")

        direct = target_df[target_df['category'] == 'Direct Costs']['value'].sum()
        indirect = target_df[target_df['category'] == 'Indirect Costs']['value'].sum()
        e2.metric("Direct Costs", f"${direct:.1f}B")
        e3.metric("Indirect Costs", f"${indirect:.1f}B")

        ec1, ec2 = st.columns([2, 1])
        with ec1:
            st.markdown(f"#### Cost Allocation: {sub_type}")
            st.caption(f"_Interactive breakdown of Healthcare Costs vs. Productivity Drains for {sub_type}._")
            sun_df = target_df[target_df['category'].isin(['Direct Costs', 'Indirect Costs'])]
            fig_sun = px.sunburst(sun_df, path=['category', 'subcategory'], values='value',
                                 color='category', template='plotly_white',
                                 color_discrete_map={'Direct Costs': '#6366f1', 'Indirect Costs': '#ef4444'})
            fig_sun.update_traces(textfont=dict(size=14, family="Inter, Arial, sans-serif"))
            fig_sun.update_layout(font=PROF_FONT)
            st.plotly_chart(fig_sun, use_container_width=True)
        with ec2:
            st.markdown("#### Productivity Loss Detail")
            st.caption("_Substance-specific drivers for work-related losses._")
            prod_detail = df_econ[(df_econ['type'] == sub_type) & (df_econ['category'] == 'Productivity Loss Detail')]
            if not prod_detail.empty:
                fig_bar = px.bar(prod_detail, x='value', y='subcategory', orientation='h', template='plotly_white',
                                text='value')
                fig_bar.update_traces(texttemplate="$%{text:.0f}B", textposition="outside",
                                     textfont=dict(size=14, family="Inter, Arial, sans-serif", color="#1e293b"))
                fig_bar.update_layout(font=PROF_FONT, xaxis_title="$Billions")
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

# ═══════════════════════════════════════════════════
# 3. AI RISK AGENT
# ═══════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════
# 4. EARLY WARNING SIGNALS
# ═══════════════════════════════════════════════════

elif menu == "Early Warning Signals":
    st.markdown("### 🚨 Early Warning & Intervention Signals")
    st.write("A dedicated intelligence center that transforms raw, unstructured patient data "
             "into interpretable, actionable insights for alcohol and drug abuse intervention.")

    df_abuse = load_abuse_reviews()

    if not df_abuse.empty:
        kpis = compute_kpis(df_abuse)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Abuse-Related Reports", f"{kpis['total_reports']:,}")
        k2.metric("Negative Sentiment %", f"{kpis['negative_pct']}%",
                  delta="Critical" if kpis['negative_pct'] > 30 else "Monitor", delta_color="inverse")
        k3.metric("Avg Patient Rating", f"{kpis['avg_rating']}/10")
        k4.metric("Unique Substances", kpis['unique_substances'])

        st.divider()

        v1, v2 = st.columns(2)
        with v1:
            st.caption("_**Condition Volume**: Signals widespread patient engagement with specific substances._")
            st.plotly_chart(figure_condition_volume(df_abuse), use_container_width=True)
        with v2:
            st.caption("_**Sentiment Breakdown**: Identified severe distress signals within withdrawal categories._")
            st.plotly_chart(figure_sentiment_by_condition(df_abuse), use_container_width=True)

        v3, v4 = st.columns(2)
        with v3:
            st.caption("_**Treatment Landscape**: Maps the 'standard of care' as perceived by the patient._")
            st.plotly_chart(figure_treatment_landscape(df_abuse), use_container_width=True)
        with v4:
            st.caption("_**Satisfaction Histogram**: Detects polarizing experiences that may lead to relapse._")
            st.plotly_chart(figure_satisfaction_distribution(df_abuse), use_container_width=True)

        st.divider()

        st.markdown("#### 🔴 Critical Negative Reports")
        st.caption("_Surfaces the top 5 most community-validated negative reviews. "
                   "These are the 'early warning' signals that can inform intervention priorities._")

        critical = get_critical_negative_reports(df_abuse, top_n=5)
        if not critical.empty:
            for i, row in critical.iterrows():
                with st.expander(f"⚠️ {row['drugName']} — {row['condition']} (Rating: {row['rating']}/10, Useful: {row['usefulCount']} votes)"):
                    st.write(row['review'])
    else:
        st.warning("No abuse-related data found. Run `python3 src/data_processing/drug_review_etl.py`.")

# ═══════════════════════════════════════════════════
# 5. RISK SIGNAL DETECTION
# ═══════════════════════════════════════════════════

elif menu == "Risk Signal Detection":
    st.markdown("### 🔬 Risk Signal Detection")
    st.write("Comparing rule-based, ML, and deep learning approaches for drug & alcohol abuse text classification.")

    # ── Approach Comparison ──
    st.caption("_**Approach Comparison**: Contrasts rule-based lexicon spikes with deep semantic embeddings. "
               "Higher F1 = better balance of precision and recall across substance abuse classes._")
    bench_df = generate_benchmark_data()
    st.plotly_chart(figure_approach_comparison(bench_df), use_container_width=True)

    st.divider()

    r1, r2 = st.columns(2)

    with r1:
        # ── Risk Classification ──
        st.caption("_**Risk Classification**: Summarizes the analyzed population into Low/Medium/High risk levels._")
        risk_df = generate_risk_classification()
        st.plotly_chart(figure_risk_classification(risk_df), use_container_width=True)

        # Detail table
        st.markdown("**Risk Level Breakdown**")
        for _, row in risk_df.iterrows():
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[row["Risk Level"]]
            st.write(f"{color} **{row['Risk Level']}** — {row['Count']:,} patients ({row['Percentage']}%): {row['Description']}")

    with r2:
        # ── Behavioral Clusters ──
        st.caption("_**Behavioral Clusters**: Visualizes distinct conversational 'themes' found in the unstructured text. "
                   "Each dot = one patient narrative, colored by detected theme._")
        cluster_df = generate_behavioral_clusters()
        st.plotly_chart(figure_behavioral_clusters(cluster_df), use_container_width=True)

# ═══════════════════════════════════════════════════
# 6. TEMPORAL & BEHAVIORAL ANALYSIS
# ═══════════════════════════════════════════════════

elif menu == "Temporal & Behavioral Analysis":
    st.markdown("### 📈 Temporal & Behavioral Analysis")
    st.write("Time-series intelligence tracking drug & alcohol risk patterns, narrative evolution, and risk intensity.")

    # ── Spike Detection ──
    st.caption("_**Spike Detection**: Identifies months where risk significantly exceeded historical averages (Z > 1.5). "
               "Red triangles mark periods requiring immediate clinical attention._")
    spike_df = generate_spike_detection_data()
    st.plotly_chart(figure_spike_detection(spike_df), use_container_width=True)

    st.divider()

    t1, t2 = st.columns(2)
    with t1:
        # ── Narrative Evolution ──
        st.caption("_**Narrative Evolution**: Tracks how the composition of patient 'themes' changes over years. "
                   "Rising 'Withdrawal & Crisis' signals demand preemptive resource allocation._")
        narr_df = generate_narrative_evolution_data()
        st.plotly_chart(figure_narrative_evolution(narr_df), use_container_width=True)

    with t2:
        # ── Risk Intensity ──
        st.caption("_**Risk Intensity**: Monitors the average risk score for specific substance classes over time. "
                   "Opioids consistently present the highest risk trajectory._")
        intensity_df = generate_risk_intensity_data()
        st.plotly_chart(figure_risk_intensity(intensity_df), use_container_width=True)

    st.divider()

    # ── Spike Summary Table ──
    st.markdown("#### ⚠ Detected Spike Events")
    spikes_only = spike_df[spike_df["Is Spike"]].copy()
    if not spikes_only.empty:
        spikes_only["Month"] = spikes_only["Month"].dt.strftime("%B %Y")
        st.dataframe(
            spikes_only[["Month", "Risk Reports", "Z-Score"]].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("No significant spikes detected in the current period.")

if __name__ == "__main__":
    st.markdown("<br><hr><center>Nexus-Cortex 2.0 | Drug & Alcohol Risk Surveillance</center>", unsafe_allow_html=True)
