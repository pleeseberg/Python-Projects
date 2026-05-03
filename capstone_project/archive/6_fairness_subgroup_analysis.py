# 6_fairness_subgroup_analysis.py

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 0. setup logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# 1. load test predictions
# ============================================================
logging.info("loading test predictions from combined file...")

pred_file = "../outputs/test_predictions.csv"
if not os.path.exists(pred_file):
    raise FileNotFoundError(f"prediction file not found: {pred_file}")

pred_df = pd.read_csv(pred_file)
logging.info(f"columns in test_predictions.csv: {list(pred_df.columns)}")
logging.info(f"sample data:\n{pred_df.head()}")

# ============================================================
# 2. define subgroups
# ============================================================
subgroups = {
    "age >= 50": pred_df["Age"] >= 0.5,   # adjust threshold to match scaled age
    "male": pred_df["Gender"] > 0,        # -1=female, 1=male after encoding
    "female": pred_df["Gender"] < 0
}

# ============================================================
# 3. compute metrics and plot for each subgroup
# ============================================================
metrics_list = []

output_dir = "../outputs/fairness"
os.makedirs(output_dir, exist_ok=True)

for name, mask in subgroups.items():
    subset_df = pred_df[mask]
    
    if subset_df.empty:
        logging.warning(f"subgroup '{name}' has no samples. skipping.")
        continue

    # compute metrics
    y_true = subset_df["y_test"]
    y_pred = subset_df["y_pred_rf"]
    y_prob = subset_df["y_prob_rf"]

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    logging.info(f"subgroup '{name}': {len(subset_df)} samples, auc={auc:.4f}, accuracy={acc:.4f}")

    metrics_list.append({
        "subgroup": name,
        "n_samples": len(subset_df),
        "auc": auc,
        "accuracy": acc
    })

    # -----------------------
    # confusion matrix plot
    # -----------------------
    disp = ConfusionMatrixDisplay(cm, display_labels=["no sepsis", "sepsis"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} - confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

    # -----------------------
    # predicted probability distribution plot
    # -----------------------
    plt.figure(figsize=(6, 4))
    sns.histplot(y_prob[y_true == 0], color="blue", label="no sepsis", stat="density", bins=50, alpha=0.6)
    sns.histplot(y_prob[y_true == 1], color="red", label="sepsis", stat="density", bins=50, alpha=0.6)
    plt.title(f"{name} - predicted probability distribution")
    plt.xlabel("predicted probability")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_')}_prob_distribution.png"))
    plt.close()

# ============================================================
# 4. save metrics table
# ============================================================
metrics_df = pd.DataFrame(metrics_list)
metrics_file = os.path.join(output_dir, "subgroup_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)
logging.info(f"subgroup metrics saved to {metrics_file}")

# -----------------------
# summary table plot
# -----------------------
plt.figure(figsize=(6, 3))
sns.barplot(x="subgroup", y="auc", data=metrics_df, palette="viridis")
plt.title("subgroup auc comparison")
plt.ylabel("auc")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "subgroup_auc_comparison.png"))
plt.close()

plt.figure(figsize=(6, 3))
sns.barplot(x="subgroup", y="accuracy", data=metrics_df, palette="magma")
plt.title("subgroup accuracy comparison")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "subgroup_accuracy_comparison.png"))
plt.close()
