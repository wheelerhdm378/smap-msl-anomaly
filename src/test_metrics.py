"""Tiny checks of the scoring code on cases where we know the answer by hand.

Run with:  python src/test_metrics.py
"""
from metrics import event_counts, intervals_to_mask, mask_to_intervals, point_counts, prf

# 1. Turning a mask into events, with and without merging nearby runs
mask = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
assert mask_to_intervals(mask) == [(1, 2), (5, 5), (10, 10)]
assert mask_to_intervals(mask, merge_gap=2) == [(1, 5), (10, 10)]
assert mask_to_intervals([0, 0, 0]) == []

# 2. Event counts: one true anomaly hit, one missed, one false alarm
true_iv = [(10, 20), (50, 60)]
pred_iv = [(18, 25), (80, 85)]
assert event_counts(true_iv, pred_iv) == (1, 1, 1)  # tp, fp, fn

# 3. A prediction touching two true anomalies hits both; extra predictions inside one
#    true anomaly are not false alarms
assert event_counts([(0, 5), (10, 15)], [(4, 11)]) == (2, 0, 0)
assert event_counts([(0, 50)], [(5, 6), (20, 21)]) == (1, 0, 0)

# 4. Point counts
tp, fp, fn = point_counts([(0, 9)], [(5, 14)], n=20)
assert (tp, fp, fn) == (5, 5, 5)

# 5. Precision / recall / F0.5 with a worked example
p, r, f = prf(tp=3, fp=1, fn=1)  # P = 0.75, R = 0.75
assert abs(p - 0.75) < 1e-9 and abs(r - 0.75) < 1e-9 and abs(f - 0.75) < 1e-9
p, r, f = prf(tp=0, fp=0, fn=4)  # flag nothing
assert (p, r, f) == (0.0, 0.0, 0.0)
assert len(intervals_to_mask([(2, 4)], 6)) == 6

print("All metric checks passed.")
