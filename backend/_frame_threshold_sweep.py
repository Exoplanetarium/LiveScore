"""Sweep the decoder frame threshold on the MAESTRO v3 test split.

Question: is the production frame threshold (0.50, the value passed at
detect_note.py:8838) on the F1 frontier, or just an inherited default?

In the production decode path frame_threshold is used ONLY for the frame-drop
offset candidate (train_enhanced_mel_transcriber.py:1165): a note ends at the
first frame the sounding-frame head falls below it, whichever is earlier than the
offset-head peak. It does not gate note existence. So, exactly like the offset
sweep, only the onset+offset (Wei) metric responds; onset@50ms must stay flat.

Method mirrors _offset_threshold_sweep.py: one windowed inference per piece
(cached heads), then re-decode at each frame threshold with onset=0.70 and
offset=0.35 held at the live production values.

Usage (from backend/):
  python _frame_threshold_sweep.py --limit 40
  python _frame_threshold_sweep.py --thresholds 0.3,0.4,0.5,0.6,0.7
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa

from gpu_ops import get_gpu_enhanced_mel_transcriber
from rhythm_training.train_enhanced_mel_transcriber import decode_enhanced_note_events

from _maestro_mireval import (MAESTRO_ROOT, OFFSET_MIN_TOL, OFFSET_RATIO,
                              gt_notes_from_midi, load_test_rows, pred_to_arrays,
                              prf)
from _offset_threshold_sweep import infer_heads
from mir_eval.transcription import match_notes

LIVE_ONSET = 0.70
LIVE_OFFSET = 0.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--out", default="benchmark_artifacts/frame_threshold_sweep.json")
    args = ap.parse_args()

    thrs = [float(x) for x in args.thresholds.split(",")]
    rows = load_test_rows(args.limit)
    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")
    print(f"Frame sweep over {thrs}\n  fixed onset={LIVE_ONSET} offset={LIVE_OFFSET}, "
          f"{len(rows)} MAESTRO test pieces\n", flush=True)

    micro = {t: {"onset_offset": [0, 0, 0], "onset50": [0, 0, 0]} for t in thrs}
    t_start = time.perf_counter()

    for i, row in enumerate(rows, 1):
        apath = MAESTRO_ROOT / row["audio_filename"]
        mpath = MAESTRO_ROOT / row["midi_filename"]
        audio, _ = librosa.load(str(apath), sr=16000, mono=True)
        ref_i, ref_p = gt_notes_from_midi(mpath)

        t0 = time.perf_counter()
        on, off, fr, vel, nv, sr, hop = infer_heads(tx, audio)
        infer_dt = time.perf_counter() - t0

        line = []
        for t in thrs:
            events = decode_enhanced_note_events(
                on, off, fr, vel, nv,
                onset_threshold=LIVE_ONSET,
                offset_threshold=LIVE_OFFSET,
                frame_threshold=t,
                sr=sr, hop=hop,
            )
            est_i, est_p = pred_to_arrays(events)
            m_oo = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05,
                               offset_ratio=OFFSET_RATIO, offset_min_tolerance=OFFSET_MIN_TOL)
            m_on = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None)
            for k, m in (("onset_offset", m_oo), ("onset50", m_on)):
                micro[t][k][0] += len(m)
                micro[t][k][1] += len(est_i)
                micro[t][k][2] += len(ref_i)
            _, _, f_oo = prf(len(m_oo), len(est_i), len(ref_i))
            line.append(f"{t:.2f}:{f_oo:.3f}")
        print(f"[{i}/{len(rows)}] {Path(row['midi_filename']).name[:34]:34s} "
              f"infer {infer_dt:.1f}s | onset+offset F1 " + " ".join(line), flush=True)

    print("\n" + "=" * 64)
    print(f"FRAME THRESHOLD SWEEP  ({len(rows)} pieces, "
          f"{(time.perf_counter()-t_start)/60:.1f} min, onset={LIVE_ONSET} offset={LIVE_OFFSET})")
    print("=" * 64)
    print(f"{'frame_thr':>10s} | {'onset+offset (Wei)':^30s} | {'onset@50ms':^12s}")
    print(f"{'':>10s} | {'P':>8s} {'R':>8s} {'F1':>8s}  | {'F1':>10s}")
    best_t, best_f = None, -1.0
    summary = {}
    for t in thrs:
        tp, ne, nr = micro[t]["onset_offset"]
        p, r, f = prf(tp, ne, nr)
        _, _, f_on = prf(*micro[t]["onset50"])
        if f > best_f:
            best_f, best_t = f, t
        summary[f"{t:.2f}"] = {"onset_offset": {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)},
                                "onset50_f1": round(f_on, 4)}
        print(f"{t:>10.2f} | {p:8.4f} {r:8.4f} {f:8.4f}  | {f_on:10.4f}")
    print(f"\nbest onset+offset micro-F1 at frame_threshold = {best_t:.2f} (F1={best_f:.4f})")
    print(f"production frame_threshold = 0.50 -> F1={summary['0.50']['onset_offset']['f1']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "fixed": {"onset_threshold": LIVE_ONSET, "offset_threshold": LIVE_OFFSET},
        "n_pieces": len(rows), "thresholds": thrs,
        "best_frame_threshold": best_t, "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
