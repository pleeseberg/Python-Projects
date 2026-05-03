#!/usr/bin/env python3
# generate_feature_names.py

import pandas as pd
from pathlib import Path

# -----------------------------
# output paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FEATURE_NAMES_FILE = OUTPUT_DIR / "feature_names.csv"

# -----------------------------
# define your feature pattern
# -----------------------------
# example: if you computed min, mean, max for each of ra, la, rv, and other features
measurements = ["RA", "LA", "RV", "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
stats = ["min", "mean", "max"]

features = []

for meas in measurements:
    for stat in stats:
        features.append(f"{meas}_{stat}")

# if there are additional features to include, append them here
# e.g., features.append("age"), features.append("gender")

# pad or trim to match actual number of columns in patient_level_features
# load one row to get column count
import pyarrow.parquet as pq
parquet_file = OUTPUT_DIR / "patient_level_features.parquet"
df = pq.read_table(parquet_file, columns=[]).to_pandas()
n_columns = df.shape[1] - 1  # exclude label

# repeat the pattern if needed
while len(features) < n_columns:
    features = features + features
features = features[:n_columns]

# save to csv
df_features = pd.DataFrame({"feature": features})
df_features.to_csv(FEATURE_NAMES_FILE, index=False)
print(f"saved feature names csv to {FEATURE_NAMES_FILE}")
