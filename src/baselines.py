"""Simple detectors. Each takes (train, test) arrays and returns a boolean mask over test.

Rules shared by every real detector:
- It learns only from the training data. Test labels are never used.
- The alarm threshold is the largest score seen on the (anomaly-free) training data.
- Only column 0, the telemetry value, is used. Commands are ignored here.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------- sanity checks: not real detectors ----------

def flag_nothing(train, test):
    return np.zeros(len(test), dtype=bool)


def flag_everything(train, test):
    return np.ones(len(test), dtype=bool)


def blind_middle(train, test, lo=0.50, hi=0.75):
    """Ignores the data. Flags a fixed slice of every test series.

    Chosen after looking at where the labelled anomalies sit (Phase 1), so it is a
    diagnostic of the benchmark, not a fair detector.
    """
    n = len(test)
    mask = np.zeros(n, dtype=bool)
    mask[int(lo * n) : int(hi * n)] = True
    return mask


# ---------- baseline 1: rolling z-score ----------

def _rolling_z(x, window, floor):
    s = pd.Series(x)
    mean = s.rolling(window, min_periods=window // 2).mean().shift(1)
    std = s.rolling(window, min_periods=window // 2).std().shift(1)
    z = (s - mean).abs() / np.maximum(std, floor)
    return z.fillna(0.0).to_numpy()


def rolling_zscore(train, test, window=100):
    """How far is each value from the recent past, in units of recent spread?"""
    x_train, x_test = train[:, 0], test[:, 0]
    floor = 0.1 * x_train.std() + 1e-3  # stops near-constant channels exploding
    threshold = _rolling_z(x_train, window, floor).max()
    # prepend the end of training so the first test steps have a full window
    joined = np.concatenate([x_train[-window:], x_test])
    score = _rolling_z(joined, window, floor)[window:]
    return score > threshold


# ---------- baseline 2: Isolation Forest on sliding windows ----------

def _windows(x, width, stride=1):
    n = (len(x) - width) // stride + 1
    idx = np.arange(width)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def isolation_forest(train, test, width=50, seed=0):
    """Isolation Forest on sliding windows of the telemetry value."""
    x_train, x_test = train[:, 0], test[:, 0]
    if len(x_train) < width or len(x_test) < width:
        return np.zeros(len(x_test), dtype=bool)
    model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)
    model.fit(_windows(x_train, width, stride=5))
    threshold = (-model.score_samples(_windows(x_train, width))).max()
    score = -model.score_samples(_windows(x_test, width))
    mask = np.zeros(len(x_test), dtype=bool)
    mask[width // 2 : width // 2 + len(score)] = score > threshold  # flag window centre
    return mask


DETECTORS = {
    "flag_nothing": flag_nothing,
    "flag_everything": flag_everything,
    "blind_middle": blind_middle,
    "rolling_zscore": rolling_zscore,
    "isolation_forest": isolation_forest,
}
