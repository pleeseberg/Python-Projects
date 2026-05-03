# Sepsis Prediction — Lab Journal



## Session 1 — Project Setup

**Date:** 2026-05-02



### What this project is trying to do

(Write this in your own words)



### What went wrong the first time and why

(Write this in your own words)



### Rules I'm holding myself to this time

(Write this in your own words)



### Decisions made today

- Chose PhysioNet 2019 dataset
- Excluded ICULOS and HospAdmTime as features
- Using indicator columns for missingness
- Old code moved to archive/ for reference only
- Repo structure created



### Open questions

- Missingness threshold: 80% or 95%?
- Minimum useful prediction window before onset?
- Should we try SMOTE for class imbalance?



---

## Key Insights from Project Planning

### Why ICULOS was a red flag in my original paper

ICULOS appeared as my top predictor because post-onset rows (SepsisLabel=1)
naturally have higher hour counts. The model learned the counter, not the
clinical signal. This is a textbook example of leakage through a time variable.

### Would ICULOS ever be acceptable as a predictor?

Even without leakage, ICULOS is excluded because:
- It encodes when in the stay you are, not how sick the patient is
- The model could learn "sepsis happens around hour X" rather than
  learning actual clinical warning signs — a subtle form of leakage
- It is not actionable — a doctor cannot intervene based on time alone

### The test for any feature

"Would a doctor at the bedside have access to this, and could they act on it?"
- Lactate rising + BP dropping = yes, actionable
- Patient has been here 18 hours = not actionable

Features that fail this test stay out of the model.

### Why AUROC alone is not enough

With 2% positive cases, a model that predicts "not sepsis" for everyone
gets 98% accuracy. AUROC can still look decent. AUPRC is more honest
because it focuses on how well the model finds the rare positive cases.
A model with high AUROC but low AUPRC is not clinically useful.

### On data leakage — the crystal ball problem

The model should never be in a better position than a doctor at the bedside.
If it sees post-onset rows during training it has a crystal ball — it knows
the outcome before making a prediction. That's not prediction, that's lookup.



---

## On Metrics — What "Good" Actually Means

### AUPRC vs AUROC on imbalanced data

AUROC measures how well the model separates positives from negatives across all possible thresholds. With a 2% positive rate, a model that predicts "not sepsis" for every single patient gets 98% accuracy and still produces a decent-looking AUROC — because 98% of the time it is technically right. AUROC treats every threshold equally and treats the classes as roughly balanced when it averages. It doesn't care that the rare class is the one that matters.

AUPRC focuses only on precision and recall for the positive class. Precision asks: of all the patients I flagged, how many actually had sepsis? Recall asks: of all the patients who had sepsis, how many did I catch? The curve plots the tradeoff between these two as the threshold moves. A model that mostly ignores the positive class collapses fast under this lens — it can't hide behind the majority class.

For a clinical early-warning system, AUPRC is the honest metric.

### What 0.43 AUPRC actually means

The random baseline for AUPRC is approximately equal to the class prevalence — about 0.02 for a 2% positive rate. A random classifier that assigns scores with no real signal would land around there.

A score of 0.43 is roughly 21 times above that baseline. That is genuinely meaningful — the model is doing real work, not just guessing. But it also means 57% of the precision-recall space is still uncaptured. It is a useful result, not a finished one. In a clinical context, 0.43 AUPRC is a model worth investigating further, not one worth deploying.

### The threshold problem

The model outputs a probability between 0 and 1. A doctor sees an alert or no alert — binary. Somewhere in between, a threshold converts the probability into a decision, and that threshold is not a statistical choice, it is a clinical one.

Lowering the threshold catches more sepsis cases (higher recall) but fires more false alarms (lower precision). Raising it reduces false alarms but misses more cases. In practice, alert fatigue is a real problem — if the system cries wolf too often, clinicians start ignoring it, and the true positives get missed anyway.

The right threshold depends on the cost tradeoff the care team is willing to accept. That is a conversation between data scientists and clinicians, not a number a model picks for itself. Default 0.5 is almost certainly wrong for a 2% positive rate.



---

## On Patient-Level Splitting

The train/test split must be done at the PATIENT level, not the ROW level.

If rows from the same patient appear in both train and test, the model
gets tested on patients it has already seen. It learns each patient's
individual patterns during training and recognises them at test time.
This inflates results and doesn't reflect real clinical performance —
in the ICU the model will always face patients it has never seen before.

patient_id is therefore one of the most critical columns in the dataset.
It is derived from the filename during loading and must be preserved
through every stage of processing.



---

## On Imputation and Scaling Order — Fit on Train, Transform Both

A subtle leakage issue in the original code: imputation means and scaling
parameters were calculated on the full dataset before splitting. This means
test patients contributed to the statistics used to prepare training data.

The correct order:
1. Split into train and test FIRST
2. Fit imputer and scaler on TRAINING data only
3. Apply those fitted parameters to BOTH train and test

The test set must be treated as a stranger. You only use what you learned
from training to fill its gaps — never ask it to teach you anything.

Rule: Fit on train. Transform both.



---

## On Patient-Level Splitting — Split Patients Not Rows

Original code passed patient_ids as a parallel array to train_test_split.
This only kept alignment — it did not group by patient. Rows from the same
patient could appear on both sides.

Correct approach:
1. Get list of unique patient IDs
2. Split that ID list into train/val/test
3. Pull all rows belonging to those IDs

Every hour a patient has goes with them — patients are never split in half.



---

## On LSTMs — What They Need and What Went Wrong

An LSTM reads a sequence one step at a time and maintains memory of what
came before. For this project that means feeding each patient's full hourly
timeline as a sequence — not individual rows.

The trajectory matters more than the snapshot. Lactate rising from 1.2 to
3.8 over 5 hours is a very different story from lactate stable at 3.8.

Original code mistake: added a fake timestep dimension of 1 with np.newaxis.
Each row was treated as a sequence of length 1 — the LSTM never saw more than
one hour at a time and had nothing to remember. Functionally equivalent to a
dense layer. No temporal learning happened at all.

Fix in Phase 4: feed complete patient sequences, not individual rows.



---

## On Fairness — Why AUROC Alone Is Not Enough

Equal AUROC across subgroups does not mean equal care. Two groups can have
similar AUROC but very different false negative rates — meaning the model
systematically misses sepsis in certain patient groups.

False Negative = model predicted no sepsis but sepsis occurred anyway.
This is the most dangerous error in a clinical context.

Original code only reported AUROC and accuracy per subgroup. This hid
potential disparities in missed cases.

Rebuild must report FALSE NEGATIVE RATE per subgroup:
- Gender (male/female)
- Age groups
- ICU unit (MICU vs SICU)
- Training set (A vs B — different hospital systems)

Equal AUROC is not the same as equal care.



---

## On Forward-Fill and the First-Row Problem

Forward-fill works within a patient's stay but fails at hour 1 when there
is nothing behind it to carry forward. Hour 1 missingness is expected and
normal — the patient just arrived, labs haven't been drawn yet.

Strategy:
- Forward-fill within each patient's timeline first
- For any remaining missing values (including hour 1), fill with the
  TRAINING SET median — never the full dataset median
- Always keep the indicator column alongside (e.g. Lactate_was_missing)

The median fill gives the model a number. The indicator column tells the
model not to trust it. Both are needed.

Rule: Fit on train. Transform both. Always.



---

## On Train/Val/Test Split Strategy

Using 70/15/15 split rather than 80/20 because:
- Validation set allows tuning decisions without touching the test set
- Test set is locked until final evaluation — opened once, never again

Think of it like this:
- Train  = studying
- Val    = practice exams you can retake
- Test   = the real exam, taken once, no retakes

Stratified by sepsis status (ever SepsisLabel=1) because:
- Only 2% positive cases — random split could skew test set by chance
- ~6,000 test patients at 2% = only ~120 sepsis cases
- Cannot afford random variation at that scale
- Stratifying preserves the true clinical distribution in all three sets



---

## Open Question — Variable Length Sequences for LSTM (Phase 4)

ICU stays range from a few hours to several days. Neural networks require
rectangular batches — all sequences must be the same length.

Two options to resolve:
- Padding: add zeros to shorter sequences, tell LSTM to ignore them
- Truncation: cap at maximum length (e.g. 48 hours), pad shorter ones

Decision needed in Phase 4. Questions to consider:
- What is the distribution of stay lengths? (EDA will tell us)
- What is the maximum clinically meaningful prediction window?
- How much padding is too much — does it hurt training?
