# Sepsis Early Prediction — Capstone Rebuild
**PhysioNet 2019 | ICU Early Warning System**

---

## Overview

This project builds an early-warning model for sepsis onset in ICU patients using the
PhysioNet 2019 Challenge dataset. The goal is to predict sepsis several hours before a
clinician would otherwise identify it — giving care teams more time to intervene.

This is a ground-up rebuild of my original M.S. capstone project. The original version
achieved AUROC 0.9985 — a near-perfect score that turned out to be entirely false, caused
by multiple data leakage issues. This rebuild fixes every one of them and documents the
reasoning at every step.

---

## The Problem

Sepsis is a life-threatening response to infection that kills ~270,000 people in the US
annually. Early detection is critical — outcomes worsen significantly with each hour of
delayed treatment. Current clinical tools often catch sepsis too late.

**The core tension:** Earlier alerts are not automatically better. Too many false alarms
cause alert fatigue — clinicians start ignoring the system and true positives get missed.
The real goal is alerts that are *early enough* and *trustworthy enough* that a doctor
will act on them.

---

## Dataset

**PhysioNet Computing in Cardiology Challenge 2019**
- ~40,336 ICU patients across two sets from different hospital systems (Set A and Set B)
- One file per patient, one row per hour of ICU stay
- 41 features: 8 vital signs, 26 lab values, 5 demographics/admin columns, ICULOS, SepsisLabel
- SepsisLabel fires 6 hours *before* clinical sepsis diagnosis — already engineered for early prediction
- ~6% of patients develop sepsis; positive rows are a much smaller fraction of total ICU-hours

Raw data is not included in this repository (size). Download from
[PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/) and place in `data/raw/`.

---

## What Broke the Original Project

Six confirmed leakage sources were found by auditing the original code:

| # | Issue | Effect |
|---|-------|--------|
| 1 | Post-onset rows never filtered | Model saw outcome data during training |
| 2 | ICULOS kept as a feature | Model learned a time counter, not clinical signal |
| 3 | HospAdmTime kept as a feature | Administrative variable with no clinical signal |
| 4 | Imputation and scaling fit on full dataset | Test set statistics leaked into training |
| 5 | Row-level train/test split | Same patient appeared on both sides |
| 6 | LSTM given fake timestep dimension of 1 | Never saw sequences — functionally a dense layer |

All six are fixed in this rebuild and documented in `lab_notebook/lab_journal.md`.

---

## Project Structure

```
capstone_project/
├── data/
│   ├── raw/                  # Original .psv files — never modified
│   │   ├── training_setA/
│   │   └── training_setB/
│   ├── processed/            # Output of preprocessing pipeline
│   └── splits/               # Train/val/test splits saved as parquet
├── lab_notebook/
│   └── lab_journal.md        # Running log of findings and decisions
├── notebooks/                # One notebook per phase
│   └── 01_eda.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   └── results/
├── src/
│   ├── data/                 # loader.py, preprocess.py, splits.py
│   ├── models/               # classical.py, deep.py
│   ├── evaluation/           # metrics.py, fairness.py
│   └── explainability/       # shap_analysis.py
├── tests/
│   └── test_splits.py
├── archive/                  # Original leaky code — reference only
├── CLAUDE.md                 # Project guide and working rules
├── PRINCIPLES.md             # Core design principles
└── requirements.txt
```

---

## Key Design Decisions

**Features excluded:** `ICULOS` and `HospAdmTime` are excluded from all models.
ICULOS is a time counter that correlates with post-onset rows — not a clinical signal.
The bedside test: *"Would a doctor have access to this, and could they act on it?"*
Features that fail this test stay out.

**Missing data:** Lab values are the most clinically meaningful features and the most
frequently missing. Missingness itself is a signal — a missing lab value may mean the
patient seemed stable. Strategy: forward-fill within each patient's timeline, then
fill remaining gaps with the training-set median. Indicator columns (e.g.
`Lactate_was_missing`) are added alongside imputed values.

**Train/val/test split:** 70/15/15, split at the patient level — never at the row level.
Every hour a patient has goes with them. Stratified by sepsis status to preserve the
~6% positive rate across all three sets. Test set is locked until final evaluation.

**Primary metric:** AUPRC over AUROC. With ~2% positive rows, a model that predicts
"no sepsis" for every patient achieves 98% accuracy and a reasonable-looking AUROC.
AUPRC is harder to game — the random baseline is approximately equal to the positive
class prevalence (~0.02).

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd capstone_project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download data from PhysioNet and place in:
# data/raw/training_setA/
# data/raw/training_setB/
```

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Exploratory Data Analysis | In progress |
| 2 | Preprocessing pipeline | Not started |
| 3 | Classical models (Logistic Regression, Random Forest) | Not started |
| 4 | Deep learning (LSTM, GRU) | Not started |
| 5 | Interpretability (SHAP) | Not started |
| 6 | Fairness — subgroup evaluation | Not started |
| 7 | Write-up | Not started |

---

## Metrics Reported

All phases report the full set — no cherry-picking:

- **AUPRC** — primary metric, most honest for rare events
- **AUROC** — useful context, treated skeptically at low positive rates
- **F1, Precision, Recall** — for threshold-level evaluation
- **False Negative Rate per subgroup** — equal AUROC does not mean equal care

---

## Author

Paige Leeseberg  
M.S. Data Science, Rochester Institute of Technology  
