"""Run every detector on all 82 channels and write results/baseline_results.csv.

Run from the project folder with:  python src/run_baselines.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

from baselines import DETECTORS
from common import load_channel, load_labels
from metrics import event_counts, mask_to_intervals, point_counts, prf

MERGE_GAP = 50  # runs of alarms closer than this become one event (fixed, not tuned)


def main():
    labels = load_labels()
    rows = []
    for name, detect in DETECTORS.items():
        start = time.time()
        for _, r in labels.iterrows():
            train, test = load_channel(r["chan_id"])
            pred = mask_to_intervals(detect(train, test), merge_gap=MERGE_GAP)
            etp, efp, efn = event_counts(r["intervals"], pred)
            ptp, pfp, pfn = point_counts(r["intervals"], pred, len(test))
            rows.append({
                "detector": name, "chan_id": r["chan_id"], "spacecraft": r["spacecraft"],
                "ev_tp": etp, "ev_fp": efp, "ev_fn": efn,
                "pt_tp": ptp, "pt_fp": pfp, "pt_fn": pfn,
            })
        print(f"{name}: done in {time.time() - start:.0f}s", file=sys.stderr)

    per_channel = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    per_channel.to_csv("results/baseline_per_channel.csv", index=False)

    out = []
    groups = [("ALL", per_channel)] + list(per_channel.groupby("spacecraft"))
    for scope, df in groups:
        for name, d in df.groupby("detector", sort=False):
            s = d[["ev_tp", "ev_fp", "ev_fn", "pt_tp", "pt_fp", "pt_fn"]].sum()
            ep, er, ef = prf(s.ev_tp, s.ev_fp, s.ev_fn, beta=0.5)
            _, _, f1 = prf(s.ev_tp, s.ev_fp, s.ev_fn, beta=1.0)
            pp, pr, pf = prf(s.pt_tp, s.pt_fp, s.pt_fn, beta=1.0)
            out.append({
                "scope": scope, "detector": name,
                "ev_TP": s.ev_tp, "ev_FP": s.ev_fp, "ev_FN": s.ev_fn,
                "ev_precision": ep, "ev_recall": er, "ev_F0.5": ef, "ev_F1": f1,
                "pt_precision": pp, "pt_recall": pr, "pt_F1": pf,
            })
    summary = pd.DataFrame(out)
    summary.to_csv("results/baseline_results.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary[summary["scope"] == "ALL"].drop(columns="scope").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
