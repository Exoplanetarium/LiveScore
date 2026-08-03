"""Sweep the decoder min_velocity gate on the MAESTRO v3 test split.

Question: is the production min_velocity (8) on the F1 frontier, or an inherited
hand-set value? (The older ensemble used 15; the enhanced-mel decode lowered it
to 8 to recover recall.)

UNLIKE the offset/frame sweeps, min_velocity gates note EXISTENCE: a decoded
event with vel_int < min_velocity is dropped (train_...:1182). So it is a genuine
precision/recall dial and DOES move onset@50ms -- which is why this sweep reports
full P/R/F1 on the onset@50ms (existence) metric, not just onset+offset.

Method mirrors the other sweeps: one windowed inference per piece (cached heads),
then re-decode at each min_velocity with onset=0.70, offset=0.35, frame=0.50 held
at the live production values.

Usage (from backend/):
  python _min_velocity_sweep.py --limit 40
  python _min_velocity_sweep.py --values 1,4,8,12,16,20
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
LIVE_FRAME = 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--values", default="1,4,8,12,16,20")
    ap.add_argument("--out", default="benchmark_artifacts/min_velocity_sweep.json")
    args = ap.parse_args()

    vals = [int(x) for x in args.values.split(",")]
    rows = load_test_rows(args.limit)
    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")
    print(f"min_velocity sweep over {vals}\n  fixed onset={LIVE_ONSET} offset={LIVE_OFFSET} "
          f"frame={LIVE_FRAME}, {len(rows)} MAESTRO test pieces\n", flush=True)

    micro = {v: {"onset50": [0, 0, 0], "onset_offset": [0, 0, 0]} for v in vals}
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
        for v in vals:
            events = decode_enhanced_note_events(
                on, off, fr, vel, nv,
                onset_threshold=LIVE_ONSET,
                offset_threshold=LIVE_OFFSET,
                frame_threshold=LIVE_FRAME,
                min_velocity=v,
                sr=sr, hop=hop,
            )
            est_i, est_p = pred_to_arrays(events)
            m_on = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None)
            m_oo = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05,
                               offset_ratio=OFFSET_RATIO, offset_min_tolerance=OFFSET_MIN_TOL)
            for k, m in (("onset50", m_on), ("onset_offset", m_oo)):
                micro[v][k][0] += len(m)
                micro[v][k][1] += len(est_i)
                micro[v][k][2] += len(ref_i)
            _, _, f_on = prf(len(m_on), len(est_i), len(ref_i))
            line.append(f"{v}:{f_on:.3f}")
        print(f"[{i}/{len(rows)}] {Path(row['midi_filename']).name[:34]:34s} "
              f"infer {infer_dt:.1f}s | onset@50 F1 " + " ".join(line), flush=True)

    print("\n" + "=" * 72)
    print(f"MIN_VELOCITY SWEEP  ({len(rows)} pieces, {(time.perf_counter()-t_start)/60:.1f} min, "
          f"onset={LIVE_ONSET} offset={LIVE_OFFSET} frame={LIVE_FRAME})")
    print("=" * 72)
    print(f"{'min_vel':>7s} | {'onset@50ms (existence)':^32s} | {'onset+offset':^10s}")
    print(f"{'':>7s} | {'P':>9s} {'R':>9s} {'F1':>9s}  | {'F1':>10s}")
    best_v, best_f = None, -1.0
    summary = {}
    for v in vals:
        tp, ne, nr = micro[v]["onset50"]
        p, r, f = prf(tp, ne, nr)
        _, _, f_oo = prf(*micro[v]["onset_offset"])
        if f > best_f:
            best_f, best_v = f, v
        summary[str(v)] = {"onset50": {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)},
                           "onset_offset_f1": round(f_oo, 4)}
        print(f"{v:>7d} | {p:9.4f} {r:9.4f} {f:9.4f}  | {f_oo:10.4f}")
    print(f"\nbest onset@50ms micro-F1 at min_velocity = {best_v} (F1={best_f:.4f})")
    print(f"production min_velocity = 8 -> onset@50 F1={summary['8']['onset50']['f1']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "fixed": {"onset_threshold": LIVE_ONSET, "offset_threshold": LIVE_OFFSET,
                  "frame_threshold": LIVE_FRAME},
        "n_pieces": len(rows), "values": vals,
        "best_min_velocity": best_v, "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
