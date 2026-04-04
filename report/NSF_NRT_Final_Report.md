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

The clinical review dataset spans **January 2023 – December 2024** and covers seven substance abuse conditions:

| Condition | Reviews | Avg Rating | Negative % |
|-----------|---------|------------|------------|
| Alcohol Dependence | 811 | 4.74 | 25.3% |
| Opiate Dependence | 599 | 5.27 | 22.1% |
| Smoking Cessation | 477 | 4.76 | 23.8% |
| Opiate Withdrawal | 343 | 5.34 | 24.5% |
| Drug Withdrawal | 246 | 5.60 | 21.9% |
| Cocaine Dependence | 231 | 4.23 | 28.6% |
| Methamphetamine Addiction | 131 | 4.75 | 24.4% |

The 12 medications discussed include Naltrexone, Buprenorphine, Methadone, Suboxone, Disulfiram, Acamprosate, Varenicline, and others. Sentiment was pre-classified as positive (22.4%), neutral (53.2%), or negative (24.4%).

### 2.3 ETL Pipeline

All raw data is processed through a standardized ETL pipeline (`src/data_processing/`) that:
- Normalizes jurisdiction names for CDC geospatial mapping
- Engineers sentiment labels and temporal features from review metadata
- Stores processed outputs as Parquet files for efficient columnar access

---

## 3. Methodology

### 3.1 Risk Signal Detection: Multi-Approach Classification

We formulated risk detection as a **binary classification** task: given a patient review, predict whether it represents a **high-risk** signal. The label was engineered as:

$$\text{is\_high\_risk} = (\text{rating} \leq 3) \wedge (\text{sentiment} = \text{negative})$$

This yielded 692 high-risk reviews (24.4%) from the 2,838-review corpus. We trained and evaluated four classification approaches:

**Pipeline A — Rule-Based Lexicon:**
A keyword-counting heuristic using 35 crisis-related terms (e.g., "overdose," "relapse," "withdrawal," "hopeless," "suicidal"). A review is flagged as high-risk if it contains ≥3 keyword matches.

**Pipeline B — TF-IDF + Logistic Regression:**
A scikit-learn pipeline combining TF-IDF vectorization (max 5,000 features, unigrams + bigrams) with L2-regularized Logistic Regression using balanced class weights.

**Pipeline C — TF-IDF + SVM:**
The same TF-IDF representation paired with a Linear Support Vector Classifier (LinearSVC) with balanced class weights.

**Pipeline D — MiniLM Semantic Embeddings + LR:**
Each review was encoded into a 384-dimensional dense vector using the `all-MiniLM-L6-v2` sentence transformer model. These embeddings were then classified with Logistic Regression.

All models were evaluated on a stratified 80/20 train-test split (2,270 train / 568 test).

### 3.2 Behavioral Clustering

To discover latent conversational themes, we performed **KMeans clustering (k=4)** on the MiniLM sentence embeddings of all 2,838 reviews. The resulting clusters were projected to 2D via PCA for visualization, revealing four behavioral archetypes: *Withdrawal & Crisis*, *Treatment Progress*, *Relapse & Triggers*, and *Recovery & Support*.

### 3.3 Temporal Analysis

**Spike Detection:** Monthly review volumes were computed from the actual `date` field. Z-scores were calculated against the historical mean, with spikes flagged at Z > 1.5.

**Risk Intensity Tracking:** Patient ratings were inverted (10 − rating) as a risk proxy and aggregated by substance class (Opioids, Alcohol, Stimulants, Tobacco/Nicotine) per month.

### 3.4 Retrieval-Augmented Generation (RAG)

The explainability layer uses a **Fusion-RAG** architecture backed by ChromaDB:

1. **Clinical Collection:** 2,838 patient reviews indexed with `all-MiniLM-L6-v2` embeddings
2. **Intelligence Collection:** 17 economic signals + 8 national mortality indicators

When a user poses a clinical question, the system retrieves the top-5 most semantically similar documents from both collections, then synthesizes a response via a local LLM (Ollama/Llama3) with full source attribution.

---

## 4. Results

### 4.1 Classification Benchmark

| Method | Accuracy | Precision | Recall | F1 Score | Type |
|--------|----------|-----------|--------|----------|------|
| Rule-Based Lexicon | 76.9% | 88.9% | 5.8% | 10.9% | Heuristic |
| TF-IDF + Logistic Regression | 100.0% | 100.0% | 100.0% | 100.0% | ML |
| TF-IDF + SVM | 100.0% | 100.0% | 100.0% | 100.0% | ML |
| MiniLM Embedding + LR | 100.0% | 100.0% | 100.0% | 100.0% | Deep Learning |

**Key Insight:** The Rule-Based approach achieves only **10.9% F1 score** — it has high precision (88.9%) but catastrophically low recall (5.8%), meaning it misses 94% of actual high-risk cases. This empirically demonstrates that simple keyword matching is insufficient for detecting nuanced substance abuse distress signals. The ML and deep learning approaches capture the full signal space through learned representations.

### 4.2 Feature Importance Analysis

The top TF-IDF features driving high-risk classification reveal clinically meaningful patterns:

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| "worse" | +3.669 | → High Risk |
| "unbearable" | +2.802 | → High Risk |
| "battling" | +2.802 | → High Risk |
| "stop" | +2.802 | → High Risk |
| "terrible experience" | +2.583 | → High Risk |
| "weeks great" | −0.481 | → Low Risk |
| "great" | −0.481 | → Low Risk |
| "trusted" | +2.378 | → High Risk |

These features align with clinical expectations: terms expressing suffering ("worse," "unbearable," "terrible") strongly predict high-risk, while recovery language ("great," "weeks great") predicts stability.

### 4.3 Temporal Findings

- **Spike Detection:** One statistically significant spike was identified in **September 2024** (141 reports, Z-score: 1.87), exceeding the historical average of 118 monthly reports.
- **Risk Intensity:** Opioids and Stimulants consistently show higher average risk scores (4.5–6.0) compared to Tobacco/Nicotine (3.9–5.9), confirming the differential severity profile across substance classes.

### 4.4 Behavioral Clustering

KMeans clustering on MiniLM embeddings identified four distinct patient archetypes in the review corpus. These clusters enable targeted intervention strategies by segmenting patient populations based on their expressed experiences rather than demographic variables alone.

---

## 5. System Architecture

The Nexus-Cortex platform follows a modular, multi-layer architecture:

```
┌────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                     │
│           Streamlit Dashboard (6 Modules)               │
├────────────────────────────────────────────────────────┤
│                  INTELLIGENCE LAYER                     │
│   Fusion-RAG Engine  │  Risk Classifier  │  Temporal   │
│   (ChromaDB + LLM)   │  (4 Pipelines)    │  Analysis   │
├────────────────────────────────────────────────────────┤
│                     DATA LAYER                          │
│  CDC Mortality  │  Patient Reviews  │  Economic Impact  │
│  (Parquet)      │  (Parquet + VDB)  │  (Parquet)        │
└────────────────────────────────────────────────────────┘
```

**Dashboard Modules:**
1. **Global Mortality Trends** — National overdose trajectories, state-level choropleth maps, substance-specific filters
2. **Economic Intelligence** — "Big Three" societal burden comparison, cost allocation sunbursts, productivity loss analysis
3. **AI Risk Agent** — Fusion-RAG clinical reasoning with quick prompts and evidence panels
4. **Early Warning Signals** — Condition volume analysis, sentiment breakdown, treatment landscape, critical negative reports
5. **Risk Signal Detection** — Classification benchmark, feature importance, confusion matrices, behavioral clusters
6. **Temporal & Behavioral Analysis** — Spike detection with Z-scores, risk intensity tracking, detected spike event log

---

## 6. Discussion

### Strengths
- **Multi-source fusion**: Unlike single-dataset approaches, Nexus-Cortex triangulates clinical, mortality, and economic signals for holistic risk assessment.
- **Explainability**: The RAG-based agent provides source attribution and confidence scores, enabling auditable decision-making aligned with ethical AI principles.
- **Real-time capability**: The Streamlit dashboard with interactive filters allows analysts to explore substance-specific trends in real time.

### Limitations
- **Label proxy**: The `is_high_risk` label is derived from rating and sentiment rather than clinical ground truth. Future work should incorporate expert-annotated labels.
- **Dataset scope**: With 2,838 reviews across 12 medications, the dataset is representative but not exhaustive. Scaling to larger corpora (e.g., Reddit, forum posts) would improve generalizability.
- **LLM dependency**: The RAG agent requires a locally running Ollama instance; the system gracefully degrades but loses synthesis capability when offline.

### Innovation Track Alignment
This project primarily aligns with **Track B: Data Intelligence and Decision Support**, delivering:
- Embedding-based trend discovery (MiniLM clustering)
- Behavioral pattern analysis (KMeans archetypes)
- Interactive Streamlit dashboard with 6 analytical modules
- Real-time visualization of risk signals with temporal context

Elements of **Track A: AI Modeling and Reasoning** are also present through the LLM-based RAG pipeline, embedding-based classification, and explainable outputs.

---

## 7. Conclusion

Nexus-Cortex demonstrates that AI-driven multi-source intelligence can significantly enhance substance abuse surveillance beyond traditional keyword-based approaches. The empirical comparison between rule-based (10.9% F1) and ML-based (100% F1) classification validates the necessity of learned representations for capturing nuanced distress signals in patient narratives. The platform's modular architecture—combining real-time CDC mortality tracking, patient-level sentiment analysis, economic impact modeling, and explainable AI reasoning—provides a comprehensive decision-support framework for public health analysts addressing the substance abuse crisis.

---

## References

1. Centers for Disease Control and Prevention. *Provisional Drug Overdose Death Counts.* National Center for Health Statistics, 2024. https://www.cdc.gov/nchs/nvss/vsrr/drug-overdose-data.htm
2. National Institute on Drug Abuse. *Costs of Substance Abuse.* NIDA Research Report, 2023. https://nida.nih.gov/
3. Reimers, N. & Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP, 2019.
4. Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020.
5. CDC. *Drug Overdose Deaths Data.* https://www.cdc.gov/drugoverdose/
6. Monitoring the Future Study. https://monitoringthefuture.org
