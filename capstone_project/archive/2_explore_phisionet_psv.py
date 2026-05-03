#!/usr/bin/env python3

# 2_explore_physionet_psv_improved.py

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# configuration
# ---------------------------
DATA_A = "../dataset/training_setA/training_setA"
DATA_B = "../dataset/training_setB/training_setB"
OUTPUT = "../outputs"
os.makedirs(OUTPUT, exist_ok=True)

np.random.seed(42)  # reproducible sampling

# ---------------------------
# load psv files
# ---------------------------
def load_psv(folder: str, sample_size=100) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.psv"))
    print(f"[info] found {len(files)} psv files in {folder}")

    if sample_size < len(files):
        files = np.random.choice(files, sample_size, replace=False)

    dfs = []
    for f in tqdm(files, desc=f"loading psv from {os.path.basename(folder)}"):
        df = pd.read_csv(f, sep="|")
        df = df.loc[:, ~df.columns.duplicated()]  # remove duplicated columns
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[info] loaded {len(all_df)} rows from {folder}")
    return all_df

# ---------------------------
# main eda
# ---------------------------
def main():
    print("[info] loading samples...\n")

    df_A = load_psv(DATA_A)
    df_B = load_psv(DATA_B)
    df = pd.concat([df_A, df_B], ignore_index=True)

    print("\n[info] ---------------------------")
    print(f"[info] combined dataset shape: {df.shape}")
    print("[info] ---------------------------\n")

    # define key vitals
    vitals = {
        "HR": "Heart Rate",
        "O2Sat": "Oxygen Saturation",
        "Temp": "Temperature",
        "SBP": "Systolic Blood Pressure",
        "MAP": "Mean Arterial Pressure",
        "Resp": "Respiratory Rate"
    }
    vital_cols = list(vitals.keys())

    # ---------------------------
    # summary statistics
    # ---------------------------
    summary = df[vital_cols].describe().T
    summary["MAD"] = df[vital_cols].apply(lambda x: (x - x.mean()).abs().mean())
    summary = summary[["mean", "std", "50%", "min", "max", "MAD"]].rename(columns={"50%": "median"})
    summary.to_csv(f"{OUTPUT}/summary_stats_combined.csv")
    print("[info] summary stats (first 6 rows):")
    print(summary.head(), "\n")

    # ---------------------------
    # outlier detection
    # ---------------------------
    print("[info] outlier percentages (z-score > 3):")
    for v in vital_cols:
        z_scores = (df[v] - df[v].mean()) / df[v].std()
        outlier_pct = (abs(z_scores) > 3).mean() * 100
        print(f"  - {v} ({vitals[v]}): {outlier_pct:.2f}% outliers")
    print()

    # ---------------------------
    # sepsis label distribution
    # ---------------------------
    sepsis_props = df["SepsisLabel"].value_counts(normalize=True) * 100
    print("[info] sepsis label distribution (%):")
    print(sepsis_props, "\n")

    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="SepsisLabel", palette=['#2b53e2', '#319161'])
    plt.title("Sepsis Label Distribution", fontsize=14)
    plt.xlabel("Sepsis Label (0 = No, 1 = Yes)", fontsize=12)
    plt.ylabel("Number of Observations", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/01_sepsis_label_counts.png", dpi=300)
    plt.close()
    print("[saved] 01_sepsis_label_counts.png")

    # ---------------------------
    # vital sign distribution boxplots
    # ---------------------------
    melted = df[vital_cols].melt(var_name="Vital Sign", value_name="Value")
    plt.figure(figsize=(10,6))
    sns.boxplot(data=melted, x="Vital Sign", y="Value", showfliers=False, palette="Set2")
    plt.title("Distribution of Key Vital Signs (No Outliers Shown)", fontsize=14)
    plt.xlabel("Vital Sign", fontsize=12)
    plt.ylabel("Measurement Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/02_vitals_boxplot.png", dpi=300)
    plt.close()
    print("[saved] 02_vitals_boxplot.png")

    # ---------------------------
    # top 10 missing features
    # ---------------------------
    missing = df.isna().mean().sort_values(ascending=False).head(10) * 100  # percent
    print("[info] top 10 missingness (%):")
    print(missing.round(2), "\n")

    plt.figure(figsize=(8,5))
    sns.barplot(x=missing.index, y=missing.values, palette="viridis")
    plt.title("Top 10 Features by Missing Values", fontsize=14)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Percent Missing (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/03_missing_values_comparison.png", dpi=300)
    plt.close()
    print("[saved] 03_missing_values_comparison.png")

    # ---------------------------
    # correlation matrix
    # ---------------------------
    corr = df[vital_cols].corr()
    print("[info] correlation matrix (numeric):")
    print(corr, "\n")

    print("[info] strongest correlations (> |0.4|):")
    strong_pairs = [
        (i, j, corr.loc[i, j])
        for i in vital_cols
        for j in vital_cols
        if i < j and abs(corr.loc[i, j]) > 0.4
    ]
    if strong_pairs:
        for i, j, c in strong_pairs:
            print(f"  - {i} ~ {j}: {c:.2f}")
    else:
        print("  none > |0.4|")

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation'})
    plt.title("Correlation Matrix of Key Vital Signs", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/04_vitals_correlation.png", dpi=300)
    plt.close()
    print("[saved] 04_vitals_correlation.png")

    # ---------------------------
    # final summary
    # ---------------------------
    print("\n================ final summary ================\n")
    print(f"total rows analyzed: {len(df)}")
    print(f"vitals analyzed: {vital_cols}")
    print(f"sepsis prevalence: {sepsis_props.get(1, 0):.3f}%")
    print("missingness, outliers, and correlations computed.")
    print("all plots saved successfully.")
    print("\n===============================================\n")


if __name__ == "__main__":
    main()
