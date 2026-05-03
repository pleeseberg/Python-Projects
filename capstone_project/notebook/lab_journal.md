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
