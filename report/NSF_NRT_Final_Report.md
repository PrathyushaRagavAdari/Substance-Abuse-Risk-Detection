# Nexus-Cortex: AI-Driven Substance Abuse Risk Detection and Surveillance Platform

**NSF NRT — Substance Abuse Risk Detection Project**
**Author:** Krutarth Lad
**Date:** April 2025

---

## 1. Introduction

Substance abuse remains one of the most pressing public health crises in the United States. The Centers for Disease Control and Prevention (CDC) reports over 100,000 drug overdose deaths annually, with synthetic opioids such as Fentanyl driving the majority of fatalities. The economic burden exceeds **$2.7 trillion** across illicit drugs ($740B), tobacco ($606B), and alcohol ($249B). Traditional surveillance methods—relying on lagging mortality statistics and manual case reporting—fail to capture the real-time dynamics of this crisis.

This project presents **Nexus-Cortex**, an AI-powered clinical intelligence platform that fuses multi-source data (CDC mortality records, patient-reported drug reviews, and federal economic reports) to deliver actionable, explainable risk signals for public health decision-makers. The system implements three core capabilities aligned with the NSF NRT research objectives:

1. **Risk Signal Detection** — Comparing rule-based, machine learning, and deep learning approaches for identifying high-risk substance abuse narratives in unstructured patient text.
2. **Temporal and Behavioral Analysis** — Modeling temporal dynamics of risk signals, detecting anomalous spikes, and tracking substance-specific risk intensity over time.
3. **Explainability and Reasoning** — Providing transparent, evidence-backed intelligence through Retrieval-Augmented Generation (RAG) with source attribution.

---

## 2. Data Sources and Processing

### 2.1 Datasets

The platform integrates three primary data sources:

| Dataset | Source | Records | Description |
|---------|--------|---------|-------------|
| Patient Drug Reviews | UCI/Kaggle | 2,838 | Real patient testimonials covering 12 medications across 7 substance abuse conditions |
| CDC Provisional Mortality | data.cdc.gov | ~50 states × 8 indicators | Drug-specific overdose deaths (Fentanyl, Cocaine, Methamphetamine, Heroin, etc.) |
| Federal Economic Impact | NIDA/CDC Reports | 17 signals | Societal cost breakdowns for Illicit Drugs, Tobacco, and Alcohol |

### 2.2 Patient Review Dataset Profile

The clinical review dataset spans **January 2023 – December 2024** and covers seven substance abuse conditions. For risk detection, we identified **340 high-risk reviews** (12.0%) based on strictly severe patient feedback (Rating ≤ 2).

| Condition | Reviews | Avg Rating | Baseline Risk % |
|-----------|---------|------------|-----------------|
| Alcohol Dependence | 811 | 4.74 | 11.2% |
| Opiate Dependence | 599 | 5.27 | 9.8% |
| Smoking Cessation | 477 | 4.76 | 13.5% |
| Opiate Withdrawal | 343 | 5.34 | 10.2% |
| Drug Withdrawal | 246 | 5.60 | 8.9% |
| Cocaine Dependence | 231 | 4.23 | 18.2% |
| Methamphetamine Addiction | 131 | 4.75 | 14.9% |

### 2.3 ETL Pipeline

All raw data is processed through a standardized ETL pipeline (`src/data_processing/`) that:
- Normalizes jurisdiction names for CDC geospatial mapping
- Engineers temporal features and grounds risk labels in numerical patient feedback
- Stores processed outputs as Parquet files for efficient columnar access

---

## 3. Methodology

### 3.1 Risk Signal Detection: Multi-Approach Classification

We formulated risk detection as a **binary classification** task. Unlike previous heuristic approaches that relied on sentiment metadata (which creates circular label leakage), we engineered a robust ground truth based solely on numerical patient ratings (Rating ≤ 2 = High Risk). This ensures the model learns real linguistic markers of distress rather than just reconstructing a secondary sentiment model.

**Pipeline A — Rule-Based Lexicon:**
A keyword-counting heuristic using 40 crisis-related terms (e.g., "overdose," "relapse," "withdrawal," "suicidal"). A review is flagged as high-risk if it contains ≥3 keyword matches.

**Pipeline B — TF-IDF + Logistic Regression:**
A scikit-learn pipeline using TF-IDF (max 5,000 features, unigrams + bigrams) with balanced class weights to handle the 12% minority class.

**Pipeline C — TF-IDF + SVM:**
A LinearSVC classifier trained on TF-IDF features with L2 regularization and 5-fold cross-validation.

**Pipeline D — MiniLM Semantic Embeddings + LR:**
Uses `all-MiniLM-L6-v2` to transform reviews into 384-dimensional dense vectors, capturing semantic context (e.g., "struggling" vs. "battling") that keyword models miss.

---

## 4. Results

### 4.1 Classification Benchmark (Re-evaluated)

All ML/DL models were evaluated using **5-Fold Stratified Cross-Validation**.

| Method | Accuracy | Precision | Recall | F1 Score | Type |
|--------|----------|-----------|--------|----------|------|
| Rule-Based Lexicon | 88.4% | 62.5% | 7.3% | 13.2% | Heuristic |
| TF-IDF + Logistic Regression | 87.3% | 48.6% | 100.0% | 65.4% | ML |
| TF-IDF + SVM | 87.2% | 48.1% | 91.2% | 62.9% | ML |
| MiniLM Embedding + LR | 87.3% | 48.6% | 100.0% | 65.4% | Deep Learning |

**Analysis**: The transition to a rating-based ground truth reveals a significantly more challenging and realistic task. The **65.4% F1-score** achieved by the TF-IDF and Embedding models represents a 5x improvement over the Rule-Based baseline (13.2% F1). The high recall (100%) indicates that these models are excellent at identifying *all* potential risk signals, which is the primary requirement for clinical surveillance, even at the cost of some precision.

### 4.2 Feature Importance: Driving Signals

| Feature | Coefficient | Clinical Context |
|---------|-------------|------------------|
| "worse" | +2.830 | Post-treatment deterioration |
| "stop" | +2.080 | Crisis of control / inability to cease use |
| "battling" | +2.080 | Prolonged struggle / relapse indicators |
| "unbearable" | +2.080 | Severe withdrawal symptoms / physical pain |
| "great" | -1.956 | Positive recovery indicator (Low Risk) |

---

## 5. System Architecture

The platform follows a modular, multi-layer architecture, with the **Cortex-1 Agent** serving as the reasoning interface for all underlying intelligence modules.

```
┌────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                     │
│           Streamlit Dashboard (6 Modules)               │
├────────────────────────────────────────────────────────┤
│                  INTELLIGENCE LAYER                     │
│   Fusion-RAG Engine  │  Risk Classifier  │  Temporal   │
│   (ChromaDB + Llama3)│  (CV Benchmarks)  │  (Z-Spikes) │
├────────────────────────────────────────────────────────┤
│                     DATA LAYER                          │
│  CDC Mortality  │  Patient Reviews  │  Economic Impact  │
│  (Parquet)      │  (Ground Truth)   │  (Parquet)        │
└────────────────────────────────────────────────────────┘
```

---

## 6. Discussion and Innovation

Nexus-Cortex validates that **learned semantic representations** are vastly superior to **rule-based lexicons** for substance abuse surveillance. Even after removing the simplistic "sentiment" signal, deep learning models (MiniLM) and specialized ML pipelines (TF-IDF+LR) were able to capture subtle linguistic markers of withdrawal and crisis (F1 ~65%).

**Innovation Alignment:**
- **Track B (Data Intelligence)**: The project delivers actionable intelligence through behavioral clustering (KMeans) and anomaly detection (Z-score spikes).
- **Explainability**: By integrating a **Fusion-RAG** engine, the system ensures that every AI-generated risk assessment is grounded in the top-5 most relevant clinical or economic documents, providing a transparent audit trail for public health officials.

---

## 7. Conclusion

This project successfully transitioned from a synthetic-data prototype to a mathematically rigorous clinical intelligence platform. The re-evaluation of classification benchmarks confirms that modern NLP pipelines can effectively detect High-Risk substance abuse signals with a 65% F1-score baseline, providing a powerful tool for real-time surveillance and early warning in the context of the national overdose crisis.

---

## References

1. CDC. *Provisional Drug Overdose Death Counts.* 2024.
2. NIDA. *Economic Impact of Substance Overdose.* 2023.
3. Reimers, N. *Sentence-BERT: Embeddings for semantic search.* 2019.
4. Lewis, P. *Retrieval-Augmented Generation (RAG).* 2020.
