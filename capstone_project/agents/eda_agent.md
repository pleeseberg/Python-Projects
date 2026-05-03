\# EDA Agent Instructions



When running EDA on the PhysioNet 2019 sepsis dataset, always check the

following in this order:



\## 1. Class Imbalance

\- What % of rows have SepsisLabel=1?

\- What % of PATIENTS have at least one SepsisLabel=1?

\- Plot the distribution



\## 2. Schema

\- List all 41 columns with their types and value ranges

\- Flag any unexpected values (negative ages, impossible vitals etc.)



\## 3. Missingness

\- Calculate % missing per column across all patients

\- Flag any column missing >80% — candidate for removal

\- Check if missingness correlates with SepsisLabel



\## 4. Patient-level Structure

\- How many patients total across set A and B?

\- Distribution of ICU stay lengths (rows per patient)

\- For sepsis patients: how many hours of pre-onset data exist?



\## 5. Clinical Signal Check

\- Compare distributions of key vitals (HR, BP, Lactate) between

&#x20; sepsis and non-sepsis patients in pre-onset rows only

\- If distributions look identical, flag this — the signal may be weak



\## Rules

\- Never use ICULOS or HospAdmTime as features

\- Never include post-onset rows in any analysis intended for modeling

\- Log all findings and decisions to notebook/lab\_journal.md

