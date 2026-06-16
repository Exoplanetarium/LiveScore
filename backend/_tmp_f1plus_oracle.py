"""Oracle decomposition of cluster-F1 headroom for a candidate JSON.

oracle_timing: matched notes get their GT onset (timing/grouping perfect),
               unmatched predictions keep their onset. Membership unchanged.
oracle_recall: add all missed GT notes to the prediction set (recall perfect),
               timing as-is for existing predictions.
oracle_both:   both of the above.
"""
import json
import sys

sys.path.insert(0, ".")
import numpy as np
from test_experiment import (
    load_midi_notes,
    slice_gt_notes,
    compute_onset_cluster_metrics,
)

cand_path = sys.argv[1]
manifest = json.load(open(r"live_benchmark_replay_json/live_benchmark_replay_auto_v2.json"))
d = json.load(open(cand_path))


def match(pred, gt):
    gt_used = set()
    pairs = []  # (pred_idx, gt_idx)
    for pi, p in enumerate(pred):
        best, berr = None, None
        for i, g in enumerate(gt):
            if i in gt_used or int(g["midi_note"]) != int(p["midi_note"]):
                continue
            e = abs(p["onset_time"] - g["onset_time"])
            if e > 0.05:
                continue
            if berr is None or e < berr:
                best, berr = i, e
        if best is not None:
            gt_used.add(best)
            pairs.append((pi, best))
    return pairs, gt_used


totals = {k: {"exact": 0.0, "pred": 0.0, "gt": 0.0} for k in ("base", "timing", "recall", "both")}

for cid, c in sorted(d["clips"].items()):
    clip = manifest["clips"][cid]
    gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
    pred = [dict(n) for n in c["score_notes"]]
    pairs, gt_used = match(pred, gt)

    timing = [dict(n) for n in pred]
    for pi, gi in pairs:
        timing[pi]["onset_time"] = gt[gi]["onset_time"]

    missed = [dict(g) for i, g in enumerate(gt) if i not in gt_used]
    recall = [dict(n) for n in pred] + missed
    both = [dict(n) for n in timing] + missed

    for key, notes in (("base", pred), ("timing", timing), ("recall", recall), ("both", both)):
        m = compute_onset_cluster_metrics(sorted(notes, key=lambda n: n["onset_time"]), gt)
        totals[key]["exact"] += m["exact_matches"]
        totals[key]["pred"] += m["predicted"]
        totals[key]["gt"] += m["ground_truth"]

for key, t in totals.items():
    p = t["exact"] / t["pred"] if t["pred"] else 0.0
    r = t["exact"] / t["gt"] if t["gt"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print("%-8s cluster_f1=%.4f precision=%.4f recall=%.4f" % (key, f1, p, r))
