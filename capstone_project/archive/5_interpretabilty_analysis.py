# 5_interpretabilty_analysis.py

import os
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
import shap
import matplotlib.pyplot as plt

# ============================================================
# 0. setup logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# config: test run or full run
# ============================================================
TEST_RUN = False  # true for small test run, false for full run

if TEST_RUN:
    SAMPLE_SIZE = 10
    BATCH_SIZE = 5
else:
    SAMPLE_SIZE = 1500
    BATCH_SIZE = 100

# ============================================================
# 1. paths
# ============================================================
FEATURE_NAMES_PATH = "../outputs/feature_names.csv"
X_TEST_PATH = "../outputs/X_test.npy"
Y_TEST_PATH = "../outputs/y_test.npy"
LR_MODEL_PATH = "../outputs/lr_model.pkl"
RF_MODEL_PATH = "../outputs/rf_model.pkl"
OUTPUT_DIR = "../outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. load feature names
# ============================================================
feature_names_df = pd.read_csv(FEATURE_NAMES_PATH)
feature_names_all = [str(f) for f in feature_names_df['feature_name'].tolist()]
logging.info(f"total feature names loaded: {len(feature_names_all)}")

# ============================================================
# 3. load test data
# ============================================================
logging.info("loading test data...")
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
X_test_df = pd.DataFrame(X_test, columns=feature_names_all)

# ============================================================
# 4. load models
# ============================================================
logging.info("loading models...")
with open(LR_MODEL_PATH, "rb") as f:
    lr_model = pickle.load(f)
with open(RF_MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)
logging.info("models loaded successfully.")

# ============================================================
# 5. calculate model performance
# ============================================================
X_test_np = X_test_df.values

# logistic regression
y_pred_lr = lr_model.predict(X_test_np)
y_prob_lr = lr_model.predict_proba(X_test_np)[:, 1]
auc_lr = roc_auc_score(y_test, y_prob_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)

# random forest
y_pred_rf = rf_model.predict(X_test_np)
y_prob_rf = rf_model.predict_proba(X_test_np)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)

# confusion metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

logging.info("===== model performance =====")
logging.info("logistic regression auc: %.4f, acc: %.4f", auc_lr, acc_lr)
logging.info("random forest auc:       %.4f, acc: %.4f", auc_rf, acc_rf)
logging.info(f"random forest confusion matrix: tp={tp}, fp={fp}, tn={tn}, fn={fn}")
logging.info(f"sensitivity={sensitivity:.4f}, specificity={specificity:.4f}")

# ============================================================
# 6. shap analysis
# ============================================================

# -----------------------------
# logistic regression shap
# -----------------------------
logging.info("computing shap values for logistic regression...")
explainer_lr = shap.Explainer(lr_model, X_test_np)
shap_values_lr = explainer_lr(X_test_np)

# save lr shap plots
shap.summary_plot(shap_values_lr, X_test_df, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_lr.png"))
plt.close()

shap.summary_plot(shap_values_lr, X_test_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_lr_bar.png"))
plt.close()

# -----------------------------
# random forest shap (batch-wise)
# -----------------------------
logging.info("computing shap values for random forest (sampled, batch-wise)...")
sample_size = min(SAMPLE_SIZE, X_test_df.shape[0])
X_sample = X_test_df.sample(sample_size, random_state=42).reset_index(drop=True)

shap_values_list = []
all_base_values_list = []
num_batches = int(np.ceil(sample_size / BATCH_SIZE))

explainer_rf = shap.TreeExplainer(rf_model)  # create once outside loop

for i in range(num_batches):
    start = i * BATCH_SIZE
    end = min(start + BATCH_SIZE, sample_size)
    batch_X = X_sample.iloc[start:end]
    
    shap_vals_batch = explainer_rf(batch_X)
    shap_values_list.append(shap_vals_batch.values)
    
    # save base values (shape: n_classes,)
    all_base_values_list.append(shap_vals_batch.base_values)
    
    logging.info(f"processed batch {i+1}/{num_batches}, shap shape: {shap_vals_batch.values.shape}")

# concatenate batch shap values
all_values = np.vstack(shap_values_list)  # shape (n_samples, n_features, n_classes)
all_base_values = np.mean(np.vstack(all_base_values_list), axis=0)  # average base values across batches

# use positive class (class 1)
shap_values_rf_pos = all_values[:, :, 1]

# create explanation object for positive class
shap_exp_rf_pos = shap.Explanation(
    values=shap_values_rf_pos,
    base_values=all_base_values[1],
    data=X_sample.values,
    feature_names=feature_names_all
)

# save shap summary plots
shap.summary_plot(shap_exp_rf_pos, X_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_rf.png"))
plt.close()

shap.summary_plot(shap_exp_rf_pos, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_rf_bar.png"))
plt.close()

# ============================================================
# 7. additional plots
# ============================================================
# roc curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,6))
plt.plot(fpr_lr, tpr_lr, label=f"logistic regression auc={auc_lr:.3f}")
plt.plot(fpr_rf, tpr_rf, label=f"random forest auc={auc_rf:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("roc curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# rf predicted vs true
plt.figure(figsize=(6,6))
plt.scatter(y_prob_rf, y_test, alpha=0.3)
plt.xlabel("predicted probability (rf)")
plt.ylabel("true label")
plt.title("random forest predicted vs true")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_predicted_vs_true.png"))
plt.close()

# ============================================================
# 8. export test set with predictions
# ============================================================
export_df = X_test_df.copy()
export_df['y_test'] = y_test
export_df['y_pred_rf'] = y_pred_rf
export_df['y_prob_rf'] = y_prob_rf
export_df['y_pred_lr'] = y_pred_lr
export_df['y_prob_lr'] = y_prob_lr
export_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
logging.info("test set with predictions exported successfully.")
