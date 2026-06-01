"""Offline replay of captured onset times through IncrementalTempoTracker.

No GPU/model needed. Feeds each payload's note+chord onsets (in time order)
through the real tracker and prints the resulting BPM, so we can reproduce and
iterate on the 120/240 doubling without re-running live inference.
"""
import glob
import json
import os

from live_rhythm import IncrementalTempoTracker


def onsets_from_payload(p):
    times = []
    for n in p.get("notes") or []:
        t = n.get("time_seconds")
        if t is not None:
            times.append(float(t))
    for c in p.get("chords") or []:
        t = c.get("time_seconds")
        if t is not None:
            times.append(float(t))
    times.sort()
    # Collapse near-simultaneous onsets (chord members) to one tempo event,
    # matching the intent of feeding distinct onset times to the tracker.
    deduped = []
    for t in times:
        if not deduped or t - deduped[-1] > 0.01:
            deduped.append(t)
    return deduped


def track(onsets):
    tr = IncrementalTempoTracker()
    tr.reset()
    for t in onsets:
        tr.add_onset(t)
    return tr.current_bpm, tr.confidence


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "_tmp_app_payload_*.json"))):
        p = json.load(open(path, encoding="utf-8"))
        onsets = onsets_from_payload(p)
        bpm, conf = track(onsets)
        n = len(onsets)
        # median IOI as a sanity reference
        iois = sorted(onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1))
        med = iois[len(iois) // 2] if iois else 0.0
        print(
            f"{p['clip_id']:>10}  live_bpm={float(p.get('bpm', 0)):6.1f}  "
            f"replay_bpm={bpm:6.1f} conf={conf:.2f}  onsets={n:3d}  "
            f"medIOI={med*1000:5.0f}ms  (60/medIOI={60.0/med if med else 0:5.1f})"
        )


if __name__ == "__main__":
    main()
