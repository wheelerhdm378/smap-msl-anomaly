"""Scoring. Event-level is the main metric; point-wise is shown for contrast.

All intervals are (start, end) with both ends inclusive.
"""
import numpy as np


def mask_to_intervals(mask, merge_gap=0):
    """Turn a boolean array into a list of (start, end) runs of True.

    Runs separated by `merge_gap` steps or fewer are merged into one event.
    """
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    intervals = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > merge_gap:
            intervals.append((start, prev))
            start = i
        prev = i
    intervals.append((start, prev))
    return intervals


def intervals_to_mask(intervals, n):
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        mask[start : end + 1] = True
    return mask


def _overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def event_counts(true_iv, pred_iv):
    """(tp, fp, fn) at the event level.

    tp: true anomalies hit by at least one predicted event
    fn: true anomalies with no predicted event touching them
    fp: predicted events that touch no true anomaly
    """
    tp = sum(any(_overlap(t, p) for p in pred_iv) for t in true_iv)
    fn = len(true_iv) - tp
    fp = sum(not any(_overlap(p, t) for t in true_iv) for p in pred_iv)
    return tp, fp, fn


def point_counts(true_iv, pred_iv, n):
    """(tp, fp, fn) counted per time step."""
    t = intervals_to_mask(true_iv, n)
    p = intervals_to_mask(pred_iv, n)
    return int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())


def prf(tp, fp, fn, beta=0.5):
    """Precision, recall and F-beta. Undefined ratios are reported as 0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = beta**2 * precision + recall
    f = (1 + beta**2) * precision * recall / denom if denom else 0.0
    return precision, recall, f
