"""Onset error stats for matched notes, per clip, from a candidate JSON."""
import json
import sys

sys.path.insert(0, ".")
from test_experiment import load_midi_notes, slice_gt_notes

cand_path = sys.argv[1]
only = sys.argv[2:] or None
manifest = json.load(open(r"live_benchmark_replay_json/live_benchmark_replay_auto_v2.json"))
d = json.load(open(cand_path))

print("%-9s %6s %6s %7s %7s %7s %7s" % ("clip", "match", "n>25ms", "mean", "median", "p10", "p90"))
import numpy as np

for cid, c in sorted(d["clips"].items()):
    if only and cid not in only:
        continue
    clip = manifest["clips"][cid]
    gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
    pred = c["score_notes"]
    gt_used = set()
    errs = []
    for p in pred:
        best, berr = None, None
        for i, g in enumerate(gt):
            if i in gt_used or int(g["midi_note"]) != int(p["midi_note"]):
                continue
            e = p["onset_time"] - g["onset_time"]
            if abs(e) > 0.05:
                continue
            if berr is None or abs(e) < abs(berr):
                best, berr = i, e
        if best is not None:
            gt_used.add(best)
            errs.append(berr)
    if not errs:
        continue
    a = np.array(errs) * 1000
    print("%-9s %6d %6d %+7.1f %+7.1f %+7.1f %+7.1f" % (
        cid, len(a), int(np.sum(np.abs(a) > 25)), a.mean(),
        float(np.median(a)), float(np.percentile(a, 10)), float(np.percentile(a, 90))))
