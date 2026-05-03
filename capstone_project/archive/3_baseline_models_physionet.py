
# 3_baseline_models_physionet.py

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    RocCurveDisplay, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
import pickle

# ---------------------------------------------------
# configuration
# ---------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # reproducible splits and rf randomness

# ---------------------------------------------------
# logging setup
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------
# patient-level aggregation
# ---------------------------------------------------
def patient_level_metrics(df, patient_col="Patient_ID", target="SepsisLabel", pred="prediction"):
    """
    aggregate predictions at the patient level.
    returns tp, fn, fp, tn, recall, precision.
    """
    grouped = df.groupby(patient_col).agg({target: "max", pred: "max"})
    tp = ((grouped[target] == 1) & (grouped[pred] == 1)).sum()
    fn = ((grouped[target] == 1) & (grouped[pred] == 0)).sum()
    fp = ((grouped[target] == 0) & (grouped[pred] == 1)).sum()
    tn = ((grouped[target] == 0) & (grouped[pred] == 0)).sum()

    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Recall": tp / (tp + fn + 1e-6),
        "Precision": tp / (tp + fp + 1e-6)
    }

# ---------------------------------------------------
# plotting utilities
# ---------------------------------------------------
def plot_roc(y_true, y_prob, title, outpath, name="Model"):
    """plot roc curve and return auc, with axis labels."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr, tpr)
    logging.info(f"{title} - roc auc: {auc_val:.4f}")

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"auc={auc_val:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("false positive rate", fontsize=12)
    plt.ylabel("true positive rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return auc_val

def plot_pr(y_true, y_prob, title, outpath):
    """plot precision-recall curve and return auprc, with axis labels."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    logging.info(f"{title} - auprc: {auprc:.4f}")

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f"auprc={auprc:.3f}")
    plt.xlabel("recall", fontsize=12)
    plt.ylabel("precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return auprc

def plot_top_features(importances, cols, title, outpath, k=20):
    """plot top k feature importances for random forest and print values to terminal."""
    idx = np.argsort(importances)[-k:]
    top_features = [(cols[i], importances[i]) for i in idx]
    logging.info(f"{title} (top {k} features):")
    for f, val in reversed(top_features):
        logging.info(f"  {f}: {val:.4f}")

    plt.figure(figsize=(10, 8))
    plt.barh([f for f,_ in top_features], [val for _,val in top_features], color="#2b53e2")
    plt.xlabel("feature importance", fontsize=12)
    plt.ylabel("feature", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return top_features

def plot_lr_coefs(model, cols, title, outpath, k=20):
    """plot top k logistic regression coefficients and print values to terminal."""
    coefs = model.coef_[0]
    idx = np.argsort(np.abs(coefs))[-k:]
    top_coefs = [(cols[i], coefs[i]) for i in idx]
    logging.info(f"{title} (top {k} coefficients):")
    for f, val in reversed(top_coefs):
        logging.info(f"  {f}: {val:.4f}")

    plt.figure(figsize=(10, 8))
    plt.barh([f for f,_ in top_coefs], [val for _,val in top_coefs], color="#319161")
    plt.xlabel("coefficient value", fontsize=12)
    plt.ylabel("feature", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return top_coefs

def plot_confusion(model, X, y, title, outpath):
    """plot confusion matrix for a model with labeled axes and print to terminal."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y)
    plt.figure(figsize=(6,5))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.xlabel("predicted label", fontsize=12)
    plt.ylabel("true label", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    # print numeric confusion matrix to terminal
    cm = disp.confusion_matrix
    logging.info(f"{title}:\n{cm}")

# ---------------------------------------------------
# main
# ---------------------------------------------------
def main():
    logging.info("starting baseline model pipeline (corrected)...")

    data_path = Path("../outputs/clean_physionet.parquet")
    out = Path("../outputs")
    out.mkdir(exist_ok=True)

    # load dataset safely
    try:
        df = pd.read_parquet(data_path)
        logging.info(f"dataset loaded successfully. shape: {df.shape}")
    except Exception as e:
        logging.error(f"failed to load dataset: {e}")
        return

    # preserve patient_id
    patient_ids = df["Patient_ID"].copy()

    # prepare features/target
    drop_cols = ["SepsisLabel", "Patient_ID"]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.median())
    y = df["SepsisLabel"]

    # train/test split (patient safe)
    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, patient_ids, test_size=0.3, stratify=y, random_state=RANDOM_SEED
    )
    logging.info(f"train/test split done. train: {X_train.shape}, test: {X_test.shape}")

    # save feature names
    feature_names = X_train.columns.tolist()
    pd.DataFrame({"feature_id": np.arange(len(feature_names)), "feature_name": feature_names}).to_csv(
        out / "feature_names.csv", index=False
    )

    # save aligned numpy arrays + patient ids
    np.save(out / "X_train.npy", X_train.to_numpy())
    np.save(out / "X_test.npy", X_test.to_numpy())
    np.save(out / "y_train.npy", y_train.to_numpy())
    np.save(out / "y_test.npy", y_test.to_numpy())
    pd.Series(pid_train).to_csv(out / "patient_ids_train.csv", index=False)
    pd.Series(pid_test).to_csv(out / "patient_ids_test.csv", index=False)
    logging.info("numpy arrays + patient ids saved")

    # scale lr only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # logistic regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

    plot_roc(y_test, y_prob_lr, "logistic regression roc", out / "lr_roc.png", "lr")
    plot_pr(y_test, y_prob_lr, "logreg pr curve", out / "lr_pr.png")
    plot_confusion(lr, X_test_s, y_test, "lr confusion", out / "lr_cm.png")
    plot_lr_coefs(lr, X.columns.values, "top lr coefs", out / "lr_coefs.png")

    lr_report = classification_report(y_test, y_pred_lr)
    logging.info(f"lr classification report:\n{lr_report}")
    cv_lr = cross_val_score(lr, X_train_s, y_train, cv=5, scoring="roc_auc")
    logging.info(f"lr 5-fold cv roc auc: {cv_lr.mean():.3f} ± {cv_lr.std():.3f}")

    # random forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    plot_roc(y_test, y_prob_rf, "random forest roc", out / "rf_roc.png", "rf")
    plot_pr(y_test, y_prob_rf, "rf pr curve", out / "rf_pr.png")
    plot_confusion(rf, X_test, y_test, "rf confusion", out / "rf_cm.png")
    plot_top_features(rf.feature_importances_, X.columns.values, "rf importance", out / "rf_importance.png")

    rf_report = classification_report(y_test, y_pred_rf)
    logging.info(f"rf classification report:\n{rf_report}")
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
    logging.info(f"rf 5-fold cv roc auc: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")

    # patient-level evaluation
    df_test = df.iloc[X_test.index].copy()
    df_test["LR_pred"] = y_pred_lr
    df_test["RF_pred"] = y_pred_rf
    logging.info("performing patient-level evaluation")
    for name, col in [("lr", "LR_pred"), ("rf", "RF_pred")]:
        m = patient_level_metrics(df_test, pred=col)
        logging.info(f"{name} patient-level metrics: {m}")

    # save models
    with open(out / "lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(out / "rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    joblib.dump(lr, out / "lr_baseline.pkl")
    joblib.dump(rf, out / "rf_baseline.pkl")
    logging.info("logistic regression + random forest models saved")

    # save text report
    with open(out / "baseline_model_report.txt", "w") as f:
        f.write("=== logistic regression ===\n")
        f.write(lr_report)
        f.write(f"\nroc auc: {auc(y_test, y_prob_lr):.3f}, auprc: {average_precision_score(y_test, y_prob_lr):.3f}\n\n")
        f.write("=== random forest ===\n")
        f.write(rf_report)
        f.write(f"\nroc auc: {auc(y_test, y_prob_rf):.3f}, auprc: {average_precision_score(y_test, y_prob_rf):.3f}\n")

    logging.info("baseline modeling complete ✅")

if __name__ == "__main__":
    main()
