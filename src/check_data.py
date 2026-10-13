import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
labels = pd.read_csv(DATA / "labeled_anomalies.csv")
print("columns:", labels.columns.tolist())

id_col = "chan_id" if "chan_id" in labels.columns else "channel_id"
labels["seqs"] = labels["anomaly_sequences"].apply(ast.literal_eval)

print(f"channels: {len(labels)}")
print(f"anomaly sequences: {labels['seqs'].apply(len).sum()}")
print(labels["spacecraft"].value_counts())

missing = [
    c for c in labels[id_col]
    if not (DATA / "train" / f"{c}.npy").exists()
    or not (DATA / "test" / f"{c}.npy").exists()
]
print("channels missing data files:", missing)

ch = labels[id_col].iloc[0]
train = np.load(DATA / "train" / f"{ch}.npy")
test = np.load(DATA / "test" / f"{ch}.npy")
print(f"{ch}: train {train.shape}, test {test.shape}")
print("telemetry (col 0) range:", train[:, 0].min(), train[:, 0].max())
