"""Trace where GT notes die in the continuous path for one clip.

Replays the clip through ContinuousLiveStreamSession with instrumentation:
- raw observations from each window decode (before continuity gates)
- observations surviving the gates
- hypotheses created / promoted to displayed states
Buckets each missed GT note (vs final score surface) by deepest stage reached.
"""
import json
import sys

sys.path.insert(0, ".")
import numpy as np

import main as live_main
from main import ContinuousLiveStreamSession
from test_experiment import (
    TARGET_SR,
    load_audio_excerpt,
    load_midi_notes,
    slice_gt_notes,
)

clip_id = sys.argv[1]
manifest = json.load(open(r"live_benchmark_replay_json/live_benchmark_replay_auto_v2.json"))
clip = manifest["clips"][clip_id]
audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
audio = np.concatenate([audio, np.zeros(int(0.6 * TARGET_SR), dtype=np.float32)])
gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])

raw_obs = []        # all observations before gates
kept_obs = []       # observations after gates
session = ContinuousLiveStreamSession(
    session_id="diag",
    sample_rate=TARGET_SR,
    context_sec=1.8,
    inference_interval_sec=0.07,
    trusted_delay_sec=0.18,
    commit_delay_sec=0.50,
    lock_delay_sec=2.0,
)

orig_filter = session._filter_stream_continuity

def wrapped_filter(observations, window_start_sec):
    raw_obs.extend(dict(o) for o in observations)
    kept, stats = orig_filter(observations, window_start_sec)
    kept_obs.extend(dict(o) for o in kept)
    return kept, stats

session._filter_stream_continuity = wrapped_filter

packet = int(round(0.04 * TARGET_SR))
displayed = {}
all_hyps = {}
for start in range(0, audio.size, packet):
    session.append_audio(audio[start : start + packet])
    upd = session.maybe_run_inference()
    if upd is None:
        continue
    for k in ("committed_notes", "locked_notes", "active_notes"):
        for p in upd.get(k) or []:
            displayed[p["id"]] = dict(p)
    for h in session.hypotheses:
        all_hyps[h["id"]] = dict(h)
upd = session.maybe_run_inference(force=True)
if upd:
    for k in ("committed_notes", "locked_notes", "active_notes"):
        for p in upd.get(k) or []:
            displayed[p["id"]] = dict(p)
for h in session.hypotheses:
    all_hyps[h["id"]] = dict(h)


def has_match(events, g, key, tol=0.05):
    for e in events:
        if int(e["midi_note"]) == int(g["midi_note"]) and abs(float(e[key]) - g["onset_time"]) <= tol:
            return True
    return False


disp_list = list(displayed.values())
hyp_list = list(all_hyps.values())
buckets = {"displayed": 0, "hyp_not_displayed": 0, "gated": 0, "never_decoded": 0}
gated_examples = []
hyp_examples = []
for g in gt:
    if has_match(disp_list, g, "onset_time"):
        buckets["displayed"] += 1
    elif has_match(hyp_list, g, "onset_time"):
        buckets["hyp_not_displayed"] += 1
        hyp_examples.append(g)
    elif has_match(kept_obs, g, "onset_time"):
        buckets["hyp_not_displayed"] += 1
        hyp_examples.append(g)
    elif has_match(raw_obs, g, "onset_time"):
        buckets["gated"] += 1
        gated_examples.append(g)
    else:
        buckets["never_decoded"] += 1

print("clip:", clip_id, clip.get("title"))
print("GT notes:", len(gt))
for k, v in buckets.items():
    print("  %-18s %d" % (k, v))
print("sample gated GT notes (midi@onset):",
      ["%d@%.2f" % (int(g["midi_note"]), g["onset_time"]) for g in gated_examples[:15]])
print("sample hyp-not-displayed:",
      ["%d@%.2f" % (int(g["midi_note"]), g["onset_time"]) for g in hyp_examples[:15]])
