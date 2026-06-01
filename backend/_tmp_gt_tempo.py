"""Report ground-truth tempo + note density for each captured clip window."""
import glob
import json
import os

import numpy as np
import pretty_midi


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "_tmp_app_payload_*.json"))):
        p = json.load(open(path, encoding="utf-8"))
        start = float(p.get("start_sec") or 0.0)
        end = float(p.get("end_sec") or 0.0)
        midi_path = p.get("gt_midi_path")
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"{p['clip_id']:>10}  GT load failed: {e}")
            continue

        times, tempi = pm.get_tempo_changes()
        # median notated tempo over the clip window
        win_tempi = [t for tt, t in zip(times, tempi) if start <= tt <= end]
        notated = float(np.median(win_tempi)) if win_tempi else (float(tempi[-1]) if len(tempi) else 0.0)

        # GT onsets in window
        onsets = sorted(
            n.start
            for inst in pm.instruments
            for n in inst.notes
            if start <= n.start < end
        )
        iois = sorted(onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1))
        med = iois[len(iois) // 2] if iois else 0.0

        print(
            f"{p['clip_id']:>10}  live_bpm={float(p.get('bpm',0)):6.1f}  "
            f"GT_notated_bpm={notated:6.1f}  ratio={ (float(p.get('bpm',0))/notated) if notated else 0:4.2f}  "
            f"GT_onsets={len(onsets):3d}  GT_medIOI={med*1000:5.0f}ms"
        )


if __name__ == "__main__":
    main()
