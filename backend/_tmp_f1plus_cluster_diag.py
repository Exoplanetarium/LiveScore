"""Diagnose cluster-structure failure for one clip from a candidate JSON.

Prints GT clusters vs predicted clusters side by side with onset errors,
to show whether failures are timing jitter, merge/split, or missing pitches.
"""
import json
import sys

sys.path.insert(0, ".")
from test_experiment import (
    cluster_note_onsets,
    load_midi_notes,
    slice_gt_notes,
    compute_onset_cluster_metrics,
)

cand_path = sys.argv[1]
clip_id = sys.argv[2]
manifest = json.load(open(r"live_benchmark_replay_json/live_benchmark_replay_auto_v2.json"))
clip = manifest["clips"][clip_id]
gt_notes = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])

d = json.load(open(cand_path))
pred_notes = d["clips"][clip_id]["score_notes"]

gt_clusters = cluster_note_onsets(gt_notes)
pred_clusters = cluster_note_onsets(pred_notes)

print("clip:", clip_id, clip.get("title"))
print("GT notes:", len(gt_notes), " pred notes:", len(pred_notes))
print("GT clusters:", len(gt_clusters), " pred clusters:", len(pred_clusters))


def sig(cl):
    return sorted(int(n["midi_note"]) for n in cl)


def anchor(cl):
    return sum(float(n["onset_time"]) for n in cl) / len(cl)


def spread(cl):
    on = [float(n["onset_time"]) for n in cl]
    return max(on) - min(on)


# walk both lists in time order
gi, pi = 0, 0
while gi < len(gt_clusters) or pi < len(pred_clusters):
    ga = anchor(gt_clusters[gi]) if gi < len(gt_clusters) else float("inf")
    pa = anchor(pred_clusters[pi]) if pi < len(pred_clusters) else float("inf")
    if ga <= pa + 0.05 and gi < len(gt_clusters):
        # find pred clusters within 50ms
        line = "GT  %7.3f %s" % (ga, sig(gt_clusters[gi]))
        gi += 1
        print(line)
    else:
        line = "PRD %7.3f %s spread=%.0fms" % (pa, sig(pred_clusters[pi]), spread(pred_clusters[pi]) * 1000)
        pi += 1
        print(line)
