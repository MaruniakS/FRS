# FRS — Feature Relevance Score for BGP Anomaly Classification

FRS (Feature Relevance Score) is a SHAP-based feature relevance analysis framework designed to **quantify, compare, and rank the influence of input features** in deep learning models for **BGP anomaly classification**.

The repository extends an LSTM(GRU)-based BGP anomaly classification pipeline by introducing **post-hoc, class-aware and direction-aware feature relevance analysis**, enabling systematic interpretation of model behavior and supporting feature selection, reduction, and model refinement.

This work is built on top of the open-source BGP anomaly classification framework by Thales Paiva:
https://github.com/thalespaiva/bgp-anomaly-classification

---

## Motivation

Most BGP anomaly detection studies focus on:
- proposing new features, or
- improving classification accuracy,

while **feature influence itself is rarely quantified in a structured and comparable manner**.

FRS addresses this gap by:
- extracting SHAP values from a trained LSTM/GRU classifier,
- aggregating them into multiple relevance views (global, per-class, directional),
- combining these views into a single **Feature Relevance Score (FRS)**.

This enables:
- interpretability of BGP anomaly classifiers,
- comparison of classical vs. newly proposed features,
- informed feature reduction without retraining from scratch.

---

## Repository overview

The repository contains:
- the original **LSTM classifier** used for BGP anomaly detection,
- a dedicated **evaluation module** implementing SHAP-based analysis and FRS computation.

```text
FRS/
│
├── classifier.py
│   Main LSTM-based BGP anomaly classifier
│
├── data/
│   Input datasets (features, labels, time windows)
│
├── evaluation/
│   SHAP and FRS analysis scripts
│   ├── calculate_shap_values.py
│   ├── class_importance.py
│   ├── class_direction.py
│   ├── calculate_frs.py
│   ├── shap_analysis.py
│   └── draw_shap_plots.py
│
├── Pipfile
├── Pipfile.lock
└── README.md
```

---

## Environment and dependencies

The project uses **Python 3.8.20** and relies on TensorFlow-based deep learning and SHAP for explainability.

### Pipfile

```toml
[packages]
tensorflow = "==2.4.3"
shap = "==0.39.0"
matplotlib = "*"
seaborn = "*"

[requires]
python_version = "3.8"
python_full_version = "3.8.20"
```

### Setup

Using `pipenv` (recommended):

```bash
pip install pipenv
pipenv install
pipenv shell
```

---

## Workflow

The full workflow consists of **two logical stages**:

### 1. BGP anomaly classification

The LSTM classifier is executed via:

```bash
python classifier.py
```

This step:
- loads BGP-derived feature datasets,
- trains or loads an LSTM model,
- produces class predictions for BGP anomalies.

Anomaly classes follow the original repository design (e.g., direct, indirect, outage).

---

### 2. Feature relevance analysis (FRS)

All interpretability logic is located in the `evaluation/` directory.

#### Step 2.1 — SHAP value computation

```bash
python evaluation/calculate_shap_values.py
```

This script:
- loads the trained LSTM model,
- computes SHAP values for input features,
- stores raw SHAP outputs for further aggregation.

---

#### Step 2.2 — Class-wise feature importance

```bash
python evaluation/class_importance.py
```

Computes **class-conditional feature relevance**, allowing analysis of how feature influence differs across anomaly types.

---

#### Step 2.3 — Directional analysis

```bash
python evaluation/class_direction.py
```

Analyzes **directional SHAP contributions**, capturing whether features predominantly contribute positively or negatively to class predictions.

---

#### Step 2.4 — Feature Relevance Score (FRS) computation

```bash
python evaluation/calculate_frs.py
```

This script aggregates:
- global relevance,
- per-class relevance,
- directional relevance,

into a single **Feature Relevance Score**.

The output is stored as a JSON file:

```text
frs-1-1-0.2.json
```

where the parameters `(α, β, γ)` correspond to weighting coefficients used in the aggregation process.

---

#### Step 2.5 — Visualization and analysis

```bash
python evaluation/draw_shap_plots.py
python evaluation/shap_analysis.py
```

These scripts generate:
- global SHAP bar plots,
- per-class feature impact visualizations,
- distribution and comparison plots for analysis and reporting.

---

## Output

Typical outputs include:
- SHAP value files (intermediate),
- class-wise and directional relevance tables,
- **FRS JSON files** (final ranking),
- figures suitable for academic publications.

---

## Reproducibility

To ensure reproducibility:
- fix random seeds in the classifier and evaluation scripts,
- keep dataset versions unchanged,
- use pinned dependency versions via `Pipfile.lock`.

---

## Relation to existing work

This repository:
- **does not propose a new classifier**,
- **does not focus on feature extraction**,

but instead introduces a **model-agnostic relevance analysis layer** applicable to existing BGP anomaly detection models.

The approach is suitable for extension to other routing protocols or time-series anomaly classification tasks with minimal adaptation.

---

## License

MIT

