#!/usr/bin/env python3
# 1_preprocess_physionet_corrected.py

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ---------------------------------------
# configuration
# ---------------------------------------
DATASET_A = "../dataset/training_setA"
DATASET_B = "../dataset/training_setB"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MISSING_THRESHOLD = 0.98  # drop features with >98% missing values

# ---------------------------------------
# loading utilities
# ---------------------------------------
def load_physionet_dataset(path: str, label: str) -> pd.DataFrame:
    """load all psv files in folder and add patient_id"""
    print(f"loading {label}...")
    psv_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files if f.lower().endswith(".psv")
    ]
    print(f"→ found {len(psv_files)} files in {path}")

    df_list = []
    for f in psv_files:
        df = pd.read_csv(f, sep="|")
        df["Patient_ID"] = os.path.basename(f).replace(".psv", "")
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

def preprocess_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """fill missing values per patient then with population mean"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    print(f"logging missing values before imputation...")
    missing_before = df[numeric_cols].isna().sum()
    print(missing_before[missing_before > 0])

    df[numeric_cols] = df.groupby("Patient_ID")[numeric_cols].transform(lambda x: x.ffill().bfill())
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print(f"logging missing values after imputation...")
    missing_after = df[numeric_cols].isna().sum()
    print(missing_after[missing_after > 0])

    return df

def normalize_numeric(df: pd.DataFrame, exclude=("Patient_ID", "SepsisLabel")):
    """standard scale numeric features and log mean/std"""
    numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # log mean/std
    means = df[numeric_cols].mean().to_dict()
    stds = df[numeric_cols].std().to_dict()
    print(f"numeric feature means (after scaling): {means}")
    print(f"numeric feature stds (after scaling): {stds}")

    return df, numeric_cols

# ---------------------------------------
# main
# ---------------------------------------
def main():
    start = time.time()
    print("starting physionet preprocessing (corrected)...\n")

    # load a and b
    df_A = load_physionet_dataset(DATASET_A, "training set a")
    df_B = load_physionet_dataset(DATASET_B, "training set b")
    df = pd.concat([df_A, df_B], ignore_index=True)
    print(f"combined shape: {df.shape}")

    # save raw
    raw_path = os.path.join(OUTPUT_DIR, "raw_physionet.parquet")
    df.to_parquet(raw_path)
    print(f"saved raw dataset → {raw_path}\n")

    # drop high-missing columns
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > MISSING_THRESHOLD].index
    df = df.drop(columns=cols_to_drop)
    print(f"dropped {len(cols_to_drop)} high-missing columns: {list(cols_to_drop)}\n")

    # fill missing values
    print(f"[{datetime.now().strftime('%H:%M:%S')}] filling missing values...")
    df = preprocess_missingness(df)

    # normalize numeric features
    df, numeric_cols = normalize_numeric(df)
    print(f"normalized {len(numeric_cols)} numeric features.\n")

    # save cleaned dataset
    clean_path = os.path.join(OUTPUT_DIR, "clean_physionet.parquet")
    df.to_parquet(clean_path)
    print(f"saved cleaned dataset → {clean_path}\n")

    # summary
    summary = pd.DataFrame({
        "records": [len(df)],
        "patients": [df["Patient_ID"].nunique()],
        "features": [df.shape[1]],
        "avg missing (%)": [df.isna().mean().mean() * 100],
        "septic (%)": [df["SepsisLabel"].mean() * 100],
    })
    print(summary.to_string(index=False))

    # columns and first 5 rows
    print("\ncolumns in cleaned dataset:")
    print(df.columns.tolist())
    print("\nfirst 5 rows:")
    print(df.head())

    print(f"\ntotal preprocessing time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
