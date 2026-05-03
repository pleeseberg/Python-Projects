# EDA Agent Instructions

When running EDA on the PhysioNet 2019 sepsis dataset, always check the
following in this order:

## 1. Class Imbalance
- What % of rows have SepsisLabel=1?
- What % of PATIENTS have at least one SepsisLabel=1?
- Plot the distribution

## 2. Schema
- List all 41 columns with their types and value ranges
- Flag any unexpected values (negative ages, impossible vitals etc.)

## 3. Missingness
- Calculate % missing per column on PRE-ONSET rows only — never on full dataset
- Flag any column missing >80% — candidate for removal
- Check if missingness correlates with SepsisLabel

## 4. Patient-level Structure
- How many patients total across set A and B?
- Distribution of ICU stay lengths (rows per patient)
- For sepsis patients: how many hours of pre-onset data exist?

## 5. Clinical Signal Check
- Compare distributions of key vitals (HR, BP, Lactate) between
  sepsis and non-sepsis patients in pre-onset rows only
- If distributions look identical, flag this — the signal may be weak

## Rules
- Never use ICULOS or HospAdmTime as features
- Never include post-onset rows in any analysis intended for modeling
- Log all findings and decisions to lab_notebook/lab_journal.md

---

## Clinical Context for High-Missingness Labs

Before dropping any lab value for high missingness, ask:
"Is this lab ordered only when something serious is happening?"

Labs ordered in extremis (ordered only in severe deterioration):
- TroponinI — cardiac stress in septic shock
- Fibrinogen — DIC, late-stage sepsis complication
- Bilirubin_direct — liver failure, organ dysfunction
- Lactate — tissue hypoperfusion, drawn when patient deteriorating

For these features, missingness itself is a strong signal.
Always keep indicator columns. Drop only if >99.5% missing AND
no meaningful correlation with sepsis label found in EDA.
