"""Shared helpers: load the labels and the train/test arrays for each channel."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")


def load_labels(data_dir=DATA):
    """One row per channel. `intervals` = list of (start, end), both inclusive."""
    labels = pd.read_csv(Path(data_dir) / "labeled_anomalies.csv")
    labels["intervals"] = labels["anomaly_sequences"].apply(
        lambda s: [tuple(p) for p in ast.literal_eval(s)]
    )
    return labels


def load_channel(chan_id, data_dir=DATA):
    """Return (train, test) arrays. Column 0 is the telemetry value."""
    train = np.load(Path(data_dir) / "train" / f"{chan_id}.npy")
    test = np.load(Path(data_dir) / "test" / f"{chan_id}.npy")
    return train, test
