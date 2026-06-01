"""Inspect tempo-candidate cost/alignment for the doubled clips, to see why the
natural-range (60-160) half-tempo loses to the doubled tempo."""
import glob
import json
import os

from live_rhythm import IncrementalTempoTracker


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "_tmp_app_payload_*.json"))):
        p = json.load(open(path, encoding="utf-8"))
        cid = p["clip_id"]
        if cid not in ("clip_002", "clip_004", "clip_001"):
            continue
        # rebuild onset stream + tracker state
        times = sorted(
            [float(n["time_seconds"]) for n in (p.get("notes") or []) if n.get("time_seconds") is not None]
            + [float(c["time_seconds"]) for c in (p.get("chords") or []) if c.get("time_seconds") is not None]
        )
        ded = []
        for t in times:
            if not ded or t - ded[-1] > 0.01:
                ded.append(t)
        tr = IncrementalTempoTracker()
        tr.reset()
        for t in ded:
            tr.add_onset(t)
        import numpy as np
        iois = np.array(tr.ioi_buffer)
        iois = iois[np.isfinite(iois) & (iois > 0)]
        print(f"\n{cid}: final_bpm={tr.current_bpm:.1f}  n_iois={len(iois)}")
        half = tr.current_bpm / 2
        for cand in sorted({60, 80, 100, 119, 120, round(half, 1), round(tr.current_bpm, 1),
                            round(tr.current_bpm / 1.5, 1)}):
            cost = IncrementalTempoTracker._tempo_cost(iois, cand)
            align = IncrementalTempoTracker._alignment_score(iois, 60.0 / cand)
            print(f"   bpm={cand:6.1f}  cost={cost:.4f}  alignment={align:.4f}")


if __name__ == "__main__":
    main()
