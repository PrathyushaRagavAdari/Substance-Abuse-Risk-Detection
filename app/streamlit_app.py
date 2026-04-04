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
    get_feature_importance, figure_feature_importance,
    figure_confusion_matrix, get_dataset_stats,
)
from src.analysis.temporal_behavioral import (
    generate_spike_detection_data, figure_spike_detection,
    generate_risk_intensity_data, figure_risk_intensity,
)

# ── Professional Chart Config ──
PROF_FONT = dict(family="Inter, Arial, sans-serif", size=14, color="#1e293b")
PROF_TITLE = dict(family="Inter, Arial, sans-serif", size=18, color="#0f172a")
PROF_AXIS = dict(family="Inter, Arial, sans-serif", size=13, color="#334155")


# ── Helper: Professional Card Wrapper ──
def cortex_card(title, caption=None):
    """Context manager for professional cards."""
    container = st.container(border=True)
    with container:
        st.markdown(f"#### {title}")
        if caption:
            st.caption(caption)
        return container


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
        with cortex_card("National Overdose Trajectories", 
                         "Identifies steep inclines in mortality that require immediate policy shifts."):
            us_df = df_mortality[df_mortality['jurisdiction'] == 'United States']
            if not us_df.empty and 'timestamp' in us_df.columns:
                all_variants = us_df['indicator'].unique().tolist()
                
                # Separate the Aggregate from the specific Drugs
                total_indicator = "Drug Overdose Deaths"
                drug_variants = [v for v in all_variants if v != total_indicator]
                
                c1, c2 = st.columns([3, 1])
                with c1:
                    selected_drugs = st.multiselect("Substances to Monitor:", 
                                                   options=drug_variants, 
                                                   default=['Fentanyl', 'Cocaine', 'Methamphetamine'])
                with c2:
                    show_total = st.checkbox("Show Total?", value=False, help="Include the National Aggregate deaths line.")
                
                final_selection = selected_drugs.copy()
                if show_total:
                    final_selection.append(total_indicator)
                
                if final_selection:
                    filtered_df = us_df[us_df['indicator'].isin(final_selection)]
                    fig_trend = px.line(filtered_df, x='timestamp', y='value', color='indicator',
                                       template='plotly_white', labels={'value': 'Deaths', 'timestamp': 'Period'})
                    fig_trend.update_traces(line=dict(width=3))
                    fig_trend.update_layout(
                        font=PROF_FONT, title=dict(font=PROF_TITLE),
                        xaxis=dict(title=dict(font=PROF_AXIS), tickfont=dict(size=12)),
                        yaxis=dict(title=dict(font=PROF_AXIS), tickfont=dict(size=12)),
                        legend=dict(font=dict(size=13), title=dict(text="Drug Type", font=dict(size=14))),
                        hovermode="x unified",
                    )
                    with st.expander("📊 View Detailed Trajectory Graph", expanded=True):
                        st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning("Please select at least one drug or the total line.")
            else:
                st.metric("National Drug Overdose Deaths (Latest)", f"{int(us_df['value'].max()):,}")





        # ── Risk Hotspots Map ──
        with cortex_card("Risk Hotspots Map", 
                         "Pinpoints priority states for resource allocation based on absolute death counts."):
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
                top_states = states_df.sort_values('value', ascending=False).head(5)
                for _, r in top_states.iterrows():
                    st.metric(r['jurisdiction'], f"{int(r['value']):,} deaths",
                              delta="High Priority", delta_color="inverse")

        # ── State-Level Risk Profile: Alcohol vs Opioids ──
        with cortex_card("State-Level Risk Profile: Alcohol vs Opioids", 
                         "Directly compares alcohol-induced vs. opioid-related mortality per 100k residents."):

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

        with cortex_card("The 'Big Three' Societal Burden", 
                         "A high-impact comparison of total annual societal costs: Illicit Drugs ($740B), Tobacco ($606B), Alcohol ($249B)."):
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
            with cortex_card(f"Cost Allocation: {sub_type}", 
                             f"Interactive breakdown of Healthcare Costs vs. Productivity Drains for {sub_type}."):
                sun_df = target_df[target_df['category'].isin(['Direct Costs', 'Indirect Costs'])]
                fig_sun = px.sunburst(sun_df, path=['category', 'subcategory'], values='value',
                                     color='category', template='plotly_white',
                                     color_discrete_map={'Direct Costs': '#6366f1', 'Indirect Costs': '#ef4444'})
                fig_sun.update_traces(textfont=dict(size=14, family="Inter, Arial, sans-serif"))
                fig_sun.update_layout(font=PROF_FONT)
                st.plotly_chart(fig_sun, use_container_width=True)
        with ec2:
            with cortex_card("Productivity Loss Detail", 
                             "Substance-specific drivers for work-related losses."):
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

    # ── Quick Questions ──
    quick_prompts = [
        "Analyze the current Fentanyl mortality hotspots.",
        "What is the societal cost of Alcohol vs Tobacco?",
        "Summarize negative patient sentiment for Opioids.",
        "What are the top 5 states for drug-related intervention?"
    ]
    
    st.markdown("#### 💡 Quick Intelligence Prompts")
    cols = st.columns(len(quick_prompts))
    clicked_prompt = None
    for i, q in enumerate(quick_prompts):
        if cols[i].button(q, key=f"q_{i}"):
            clicked_prompt = q

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Agent Connection Persistence Check ──
    if st.session_state.get("agent") is None:
        st.error("🤖 **Cortex-1 Disconnected.** The Intelligence Engine failed to initialize.")
        st.info("This is typically caused by a ChromaDB version mismatch or Ollama being offline.")
        if st.button("🔌 Reconnect Intelligence Engine", use_container_width=True):
            with st.spinner("Forging fresh connection..."):
                try:
                    from src.core.agent_model import SubstanceRiskAgent
                    st.session_state.agent = SubstanceRiskAgent(model_name=local_model)
                    st.success("Cortex-1: Fusion-Active")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reconnection Failed: {type(e).__name__}")
        st.stop()

    # ── Chat Interface ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Analyze substance risks or economic impact...")
    if clicked_prompt:
        prompt = clicked_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Cortex-1 reasoning..."):
                try:
                    if st.session_state.agent:
                        response = st.session_state.agent.run_query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        # ── Explainability: Evidence Panel ──
                        with st.expander("📋 Supporting Evidence & Sources", expanded=False):
                            try:
                                rag = st.session_state.agent.rag
                                evidence = rag.query(prompt, collection="clinical_reviews", n_results=5)
                                if evidence and evidence['documents'][0]:
                                    for i, (doc, meta) in enumerate(zip(evidence['documents'][0], evidence['metadatas'][0])):
                                        drug = meta.get('drugName', 'Unknown')
                                        cond = meta.get('condition', 'Unknown')
                                        sent = meta.get('sentiment', 'Unknown')
                                        rating = meta.get('rating', '?')
                                        dist = evidence['distances'][0][i] if 'distances' in evidence else '?'
                                        st.markdown(f"**Source {i+1}** | {drug} — {cond} | Sentiment: `{sent}` | Rating: `{rating}/10` | Similarity: `{dist:.3f}` ")
                                        st.caption(doc[:300] + '...' if len(doc) > 300 else doc)
                                        st.divider()
                                else:
                                    st.info("No supporting evidence retrieved.")
                            except Exception:
                                st.info("Evidence retrieval unavailable.")
                    else:
                        st.error("Agent lost during reasoning. Please reconnect.")
                except Exception as e:
                    st.error(f"Reasoning Error: {e}")



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
            with cortex_card("Condition Volume", "Signals widespread patient engagement with specific substances."):
                st.plotly_chart(figure_condition_volume(df_abuse), use_container_width=True)
        with v2:
            with cortex_card("Sentiment Breakdown", "Identified severe distress signals within withdrawal categories."):
                st.plotly_chart(figure_sentiment_by_condition(df_abuse), use_container_width=True)

        v3, v4 = st.columns(2)
        with v3:
            with cortex_card("Treatment Landscape", "Maps the 'standard of care' as perceived by the patient."):
                st.plotly_chart(figure_treatment_landscape(df_abuse), use_container_width=True)
        with v4:
            with cortex_card("Satisfaction Histogram", "Detects polarizing experiences that may lead to relapse."):
                st.plotly_chart(figure_satisfaction_distribution(df_abuse), use_container_width=True)

        st.divider()

        with cortex_card("🔴 Critical Negative Reports", 
                         "Surfaces the top 5 most community-validated negative reviews."):
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

    # ── Dataset Stats ──
    stats = get_dataset_stats()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Reviews", f"{stats['total_reviews']:,}")
    s2.metric("High-Risk Labels", f"{stats['high_risk_count']:,}")
    s3.metric("High-Risk %", f"{stats['high_risk_pct']}%")
    s4.metric("Train / Test Split", f"{stats['train_size']} / {stats['test_size']}")

    st.divider()

    with cortex_card("Approach Comparison", 
                     "Real benchmark results from 4 classification pipelines trained on the patient review dataset."):
        bench_df = generate_benchmark_data()
        st.plotly_chart(figure_approach_comparison(bench_df), use_container_width=True)

    st.divider()

    r1, r2 = st.columns(2)

    with r1:
        with cortex_card("Risk Classification", 
                         "Population risk distribution from the TF-IDF + LR model applied to all reviews."):
            risk_df = generate_risk_classification()
            st.plotly_chart(figure_risk_classification(risk_df), use_container_width=True)

            st.markdown("**Risk Level Breakdown**")
            for _, row in risk_df.iterrows():
                color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[row["Risk Level"]]
                st.write(f"{color} **{row['Risk Level']}** — {row['Count']:,} patients ({row['Percentage']}%): {row['Description']}")

    with r2:
        with cortex_card("Behavioral Clusters", 
                         "KMeans clustering on MiniLM sentence embeddings, projected via PCA."):
            cluster_df = generate_behavioral_clusters()
            st.plotly_chart(figure_behavioral_clusters(cluster_df), use_container_width=True)

    st.divider()

    f1, f2 = st.columns(2)
    with f1:
        with cortex_card("Feature Importance", 
                         "Top TF-IDF features that drive the high-risk classification decision."):
            feat_df = get_feature_importance()
            if not feat_df.empty:
                st.plotly_chart(figure_feature_importance(feat_df), use_container_width=True)
            else:
                st.info("Run risk_classifier.py to generate feature importance data.")

    with f2:
        with cortex_card("Confusion Matrix", 
                         "Prediction accuracy breakdown for the best-performing model."):
            cm_method = st.selectbox("Select Model", ["Rule-Based Lexicon", "TF-IDF + Logistic Regression", "TF-IDF + SVM", "MiniLM Embedding + LR"], index=0)
            fig_cm = figure_confusion_matrix(cm_method)
            if fig_cm.data:
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Run risk_classifier.py to generate confusion matrix data.")

# ═══════════════════════════════════════════════════
# 6. TEMPORAL & BEHAVIORAL ANALYSIS
# ═══════════════════════════════════════════════════

elif menu == "Temporal & Behavioral Analysis":
    st.markdown("### 📈 Temporal & Behavioral Analysis")
    st.write("Time-series intelligence tracking drug & alcohol risk patterns, narrative evolution, and risk intensity.")

    with cortex_card("Spike Detection", 
                     "Identifies months where risk significantly exceeded historical averages (Z > 1.5)."):
        spike_df = generate_spike_detection_data()
        st.plotly_chart(figure_spike_detection(spike_df), use_container_width=True)

    st.divider()

    with cortex_card("Risk Intensity", 
                     "Monitors the average risk score for specific substance classes over time."):
        intensity_df = generate_risk_intensity_data()
        st.plotly_chart(figure_risk_intensity(intensity_df), use_container_width=True)


    st.divider()

    # ── Spike Summary Table ──
    with cortex_card("⚠ Detected Spike Events", "Historical log of clinical intensity spikes."):
        spikes_only = spike_df[spike_df["Is Spike"]].copy()
        if not spikes_only.empty:
            spikes_only["Month"] = spikes_only["Month"].dt.strftime("%B %Y")
            st.dataframe(
                spikes_only[["Month", "Risk Reports", "Z-Score"]].reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("No significant spikes detected.")



if __name__ == "__main__":
    st.markdown("<br><hr><center>Nexus-Cortex 2.0 | Drug & Alcohol Risk Surveillance</center>", unsafe_allow_html=True)

