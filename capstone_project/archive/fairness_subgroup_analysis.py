# ============================================================
# 6_fairness_subgroup_analysis.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

# ============================================================
# 1. PATHS
# ============================================================
PRED_PATH = "../outputs/plots/test_predictions.csv"
OUTPUT_DIR = "../outputs/fairness"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv(PRED_PATH)

# Required columns check
required_cols = [
    "Gender", "Age", "Unit1", "Unit2",
    "y_test", "y_pred_rf", "y_prob_rf"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# ============================================================
# 3. AGE GROUP CREATION
# ============================================================
df["age_group"] = pd.cut(
    df["Age"],
    bins=[0, 40, 65, 120],
    labels=["Young", "Middle", "Elderly"]
)

# ============================================================
# 4. METRIC FUNCTION
# ============================================================
def compute_metrics(sub_df):
    y_true = sub_df["y_test"]
    y_pred = sub_df["y_pred_rf"]
    y_prob = sub_df["y_prob_rf"]

    if y_true.nunique() < 2:
        return None

    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "FN_Count": fn,
        "Group_Size": len(sub_df)
    }

# ============================================================
# 5. GENDER FAIRNESS
# ============================================================
gender_results = []

for g in [0, 1]:
    label = "Female" if g == 0 else "Male"
    sub_df = df[df["Gender"] == g]
    metrics = compute_metrics(sub_df)
    metrics["Group"] = label
    gender_results.append(metrics)

gender_df = pd.DataFrame(gender_results)
gender_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_gender_metrics.csv"), index=False)

# ============================================================
# 6. AGE FAIRNESS
# ============================================================
age_results = []

for group in ["Young", "Middle", "Elderly"]:
    sub_df = df[df["age_group"] == group]
    metrics = compute_metrics(sub_df)
    metrics["Group"] = group
    age_results.append(metrics)

age_df = pd.DataFrame(age_results)
age_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_age_metrics.csv"), index=False)

# ============================================================
# 7. ICU UNIT FAIRNESS
# ============================================================
icu_results = []

for unit in ["Unit1", "Unit2"]:
    sub_df = df[df[unit] == 1]
    metrics = compute_metrics(sub_df)
    metrics["Group"] = unit
    icu_results.append(metrics)

icu_df = pd.DataFrame(icu_results)
icu_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_icu_metrics.csv"), index=False)

# ============================================================
# 8. FALSE NEGATIVE CONCENTRATION
# ============================================================
fn_df = df[df["y_test"] == 1].copy()
fn_df["false_negative"] = fn_df["y_pred_rf"] == 0

fn_summary = (
    fn_df.groupby("age_group")["false_negative"]
    .mean()
    .reset_index()
    .rename(columns={"false_negative": "FN_Rate"})
)

fn_summary.to_csv(os.path.join(OUTPUT_DIR, "false_negative_by_age.csv"), index=False)

# ============================================================
# 9. PLOTTING
# ============================================================

# AUC by Gender
plt.figure()
plt.bar(gender_df["Group"], gender_df["AUC"])
plt.title("AUC by Gender")
plt.ylabel("AUC")
plt.savefig(os.path.join(OUTPUT_DIR, "auc_by_gender.png"))
plt.close()

# AUC by Age
plt.figure()
plt.bar(age_df["Group"], age_df["AUC"])
plt.title("AUC by Age Group")
plt.ylabel("AUC")
plt.savefig(os.path.join(OUTPUT_DIR, "auc_by_age.png"))
plt.close()

# Sensitivity by Age
plt.figure()
plt.bar(age_df["Group"], age_df["Sensitivity"])
plt.title("Sensitivity by Age Group")
plt.ylabel("Sensitivity")
plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_by_age.png"))
plt.close()

# FN Rate by Age
plt.figure()
plt.bar(fn_summary["age_group"], fn_summary["FN_Rate"])
plt.title("False Negative Rate by Age Group")
plt.ylabel("FN Rate")
plt.savefig(os.path.join(OUTPUT_DIR, "fn_rate_by_age.png"))
plt.close()

print("✅ Script 6 completed successfully. Fairness analysis outputs saved.")
