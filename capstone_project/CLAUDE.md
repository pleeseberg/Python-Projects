# Capstone Project — Sepsis Early Prediction
## Claude Code Project Guide

---

## Read This First

This is a masters capstone rebuild. The original project had serious data leakage
issues that inflated results to AUROC 0.9985 — a near-perfect score that was
entirely false. This rebuild starts from scratch with a focus on understanding
every decision before implementing it.

**How to work with me:**
- Discuss concepts BEFORE writing any code
- I make the decisions — you assist and advise
- Quiz me after major decisions and at the end of each phase
- If I am heading in a wrong direction, say so clearly and explain why
- Never write large blocks of code without first explaining what they do and why
- Always read PRINCIPLES.md and lab_notebook/lab_journal.md at the start of each session
- Update lab_journal.md with findings and decisions at the end of every session

---

## Project Goal

Predict sepsis onset in ICU patients earlier than current clinical tools allow,
using the PhysioNet 2019 dataset. The model should fire an alert hours before
a clinician would otherwise notice — giving doctors more time to intervene.

**The core tension:** Earlier is not always better. Too many false alarms cause
alert fatigue and doctors start ignoring the system. The real goal is alerts
that are early enough AND trustworthy enough that a doctor will act on them.

---

## Dataset

**PhysioNet 2019 Challenge Dataset**
- Location: data/raw/training_setA/ and data/raw/training_setB/
- Format: One .psv (pipe-separated) file per patient
- Structure: Each row = one hour of ICU stay
- Total: ~40,336 patients across both sets
- 41 columns: 8 vitals, 26 lab values, 5 demographics/admin, ICULOS, SepsisLabel
- Sets A and B come from different hospital systems — important for fairness analysis

**SepsisLabel explained:**
- Set to 1 starting 6 hours BEFORE clinical sepsis time (already engineered for early prediction)
- It is a clinical decision, not a biological ground truth — inherently noisy
- 0 for all rows of non-sepsis patients

**Two patient types:**
- Never-sepsis: SepsisLabel stays 0 for entire stay — keep ALL rows as negatives
- Sepsis patients: SepsisLabel flips to 1 — keep ONLY pre-onset rows (before first 1)

---

## What Broke the Original Project

These are the confirmed leakage sources found by auditing the original code.
Every one of these is fixed in this rebuild.

1. Post-onset rows never filtered — model learned to recognise sepsis at diagnosis not before it
2. ICULOS and HospAdmTime kept as features — model learned a time counter not clinical signals
3. Imputation and scaling fit on full dataset before splitting — test statistics leaked into training
4. Row-level not patient-level split — same patient appeared on both sides
5. LSTM given fake timestep dimension of 1 — never saw sequences, functionally a dense layer
6. Fairness only reported AUROC — hid dangerous subgroup disparities in missed sepsis cases

---

## Non-Negotiable Rules

### Data Rules
1. data/raw/ is NEVER modified — raw files are read-only forever
2. For sepsis patients, filter ALL rows from first SepsisLabel=1 onward during processing
3. ICULOS is EXCLUDED as a feature — time counter, not clinically actionable
4. HospAdmTime is EXCLUDED as a feature — administrative, not clinical signal
5. Any AUROC above 0.85 triggers a leakage investigation before celebrating

### The Feature Test
Before adding any feature ask:
"Would a doctor at the bedside have access to this, and could they act on it?"
If it fails this test, it stays out.

### Imputation and Scaling Order
FIT ON TRAIN. TRANSFORM BOTH. ALWAYS.
- Split into train/val/test FIRST
- Fit imputer and scaler on TRAINING data only
- Apply fitted parameters to train, val, and test
- Never fit on the full dataset or on test data

### Missing Data Strategy
- Forward-fill within each patient's timeline first
- For remaining missing values (including hour 1): fill with TRAINING median
- Always add indicator columns (e.g. Lactate_was_missing = 1)
- Remove columns missing >80% — threshold to be confirmed by EDA
- Hour 1 missingness is expected and normal — patient just arrived, labs not drawn yet

### Patient-Level Splitting
- Split the PATIENT LIST not the row list
- Every hour a patient has goes with them — never split across both sets
- 70/15/15 train/val/test split
- Stratified by sepsis status (ever SepsisLabel=1) to preserve 2% rate in all sets
- Train = studying | Val = practice exams you can retake | Test = real exam, taken once

---

## Metrics — Always Report All Of These

- AUPRC — primary metric, most honest for rare events, random baseline ~0.02
- AUROC — useful context but treat skeptically at 2% positive rate
- F1 Score — good single-number summary
- Recall — missing a sepsis case means patient not treated
- Precision — too low means alert fatigue
- False Negative Rate per subgroup — equal AUROC does not mean equal care

False Negative = model predicted no sepsis but sepsis occurred. Most dangerous error.

---

## Project Phases

### Phase 1 — EDA (CURRENT PHASE)
Notebook: notebooks/01_eda.ipynb
- Load sample of 500 patients before loading all 40,000
- Never load post-onset rows into EDA for modeling decisions
- Log all findings to lab_notebook/lab_journal.md

Checklist:
- [ ] Class imbalance — % rows and % patients with SepsisLabel=1
- [ ] Schema — all 41 columns, types, ranges, unexpected values
- [ ] Missingness by column — % missing, flag >80%
- [ ] Missingness vs label — does it correlate with sepsis?
- [ ] Patient-level structure — stay length distribution
- [ ] Hours of pre-onset data — how much signal before onset?
- [ ] Clinical signal check — do vitals differ pre-onset between groups?
- [ ] Set A vs Set B — are the two hospital systems different?

### Phase 2 — Preprocessing
Files: src/data/loader.py, src/data/preprocess.py, src/data/splits.py

loader.py: read all .psv files, add patient_id and training_set columns, sample_size parameter
preprocess.py: filter post-onset rows, drop excluded features, forward-fill, training median fallback, indicator columns
splits.py: split patient ID list 70/15/15 stratified by sepsis status, save as parquet
tests/test_splits.py: verify no patient in both train and test, verify no post-onset rows

### Phase 3 — Classical Models
File: src/models/classical.py
Models: Logistic Regression, Random Forest
Reuse from archive: plot_roc, plot_pr, plot_confusion, patient_level_metrics

### Phase 4 — Deep Learning
File: src/models/deep.py
Models: LSTM, GRU
Feed complete patient sequences — NOT individual rows — NOT np.newaxis fake timestep
Open question: padding vs truncation for variable length sequences (inform from EDA)
Reuse from archive: BaseRNN architecture + training loop + early stopping (fix input first)

### Phase 5 — Interpretability
File: src/explainability/shap_analysis.py
Top SHAP features must make clinical sense
If ICULOS or HospAdmTime appears — stop and investigate leakage
Reuse from archive: batch-wise SHAP for memory efficiency

### Phase 6 — Fairness
File: src/evaluation/fairness.py
Subgroups: Gender, Age groups, ICU unit, Set A vs B
Report AUROC + AUPRC + False Negative Rate + Precision/Recall per subgroup
Do NOT use scaled values for demographic thresholds (original bug)

### Phase 7 — Write-up
Abstract matches what was built. Every reference verified. Methods explains why not just what.

---

## Current Status

- [x] Repo structure created
- [x] Raw data downloaded (training_setA and training_setB)
- [x] PRINCIPLES.md written
- [x] Lab journal started with key insights
- [x] EDA agent instructions written
- [x] tests/ folder created with test_splits.py stub
- [x] Original code audited and leakage sources documented
- [x] All design decisions made and logged
- [ ] requirements.txt — create at start of Phase 1
- [ ] Phase 1 EDA — START HERE NEXT SESSION
- [ ] Phase 2 Preprocessing
- [ ] Phase 3 Classical Models
- [ ] Phase 4 Deep Learning
- [ ] Phase 5 Interpretability
- [ ] Phase 6 Fairness
- [ ] Phase 7 Write-up