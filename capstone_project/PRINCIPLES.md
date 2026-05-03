# Sepsis Prediction — Guiding Principles

## The Core Problem
Detect sepsis earlier than current clinical tools. Alert as early as possible,
but only when confident enough that a doctor will trust it. This tension between
sensitivity and precision is the core tradeoff of this project.

## The Dataset
- PhysioNet 2019 — one .psv file per patient, one row per hour
- SepsisLabel fires 6 hours BEFORE clinical suspicion (already engineered for early prediction)
- Two patient types: never-sepsis (keep all rows) and sepsis (keep only pre-onset rows)

## Leakage Rules
- Raw files in data/raw/ are NEVER modified
- During processing, filter out all rows from first SepsisLabel=1 
  onward for sepsis patients
- EXCLUDE ICULOS and HospAdmTime as features when building model input
- Any AUROC above 0.85 triggers a leakage investigation before celebrating

## Missing Data
- Lab values are most important AND most missing
- Missingness itself is a signal — use indicator columns alongside imputed values
- Remove columns missing >80% of values

## Metrics
- Prioritise AUPRC over AUROC for imbalanced data
- Track both precision and recall — false alarms cause alert fatigue
- AUROC alone is not enough with 2% positive class

## Guiding Principles
1. Predictions must be genuinely early
2. Be skeptical of your own results
3. Precision matters as much as recall
4. Deep learning must earn its complexity
5. Explainability is part of the product
6. Understand before you implement