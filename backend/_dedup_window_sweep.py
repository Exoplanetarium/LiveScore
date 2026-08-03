"""Sweep the decoder same-pitch dedup window on the MAESTRO v3 test split.

Question: is the production duplicate_window_sec (0.04, LIVE_ENHANCED_DUPLICATE_
WINDOW_SEC) tuned, or an inherited default? (Only one untested A/B candidate at
0.06 exists in tune_decoder_settings.py; no committed sweep.)

The dedup pass (train_...:1230) collapses two SAME-pitch events whose onsets are
within duplicate_window_sec, keeping the higher-onset_prob copy. It gates note
existence, so it can move onset@50 P/R. Prior: _peak_frames already does same-key
temporal NMS, so few same-pitch near-duplicates survive to this pass -> likely
near-inert.

Method mirrors the other sweeps: one windowed inference per piece (cached heads),
re-decode at each window with onset=0.70, offset=0.35, frame=0.50 (live prod).

Usage (from backend/):
  python _dedup_window_sweep.py --limit 40
  python _dedup_window_sweep.py --values 0.0,0.02,0.04,0.06,0.08,0.10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa

from gpu_ops import get_gpu_enhanced_mel_transcriber
from rhythm_training.train_enhanced_mel_transcriber import decode_enhanced_note_events

from _maestro_mireval import (MAESTRO_ROOT, gt_notes_from_midi, load_test_rows,
                              pred_to_arrays, prf)
from _offset_threshold_sweep import infer_heads
from mir_eval.transcription import match_notes

LIVE_ONSET = 0.70
LIVE_OFFSET = 0.35
LIVE_FRAME = 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--values", default="0.0,0.02,0.04,0.06,0.08,0.10")
    ap.add_argument("--out", default="benchmark_artifacts/dedup_window_sweep.json")
    args = ap.parse_args()

    vals = [float(x) for x in args.values.split(",")]
    rows = load_test_rows(args.limit)
    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")
    print(f"dedup-window sweep over {vals}\n  fixed onset={LIVE_ONSET} offset={LIVE_OFFSET} "
          f"frame={LIVE_FRAME}, {len(rows)} MAESTRO test pieces\n", flush=True)

    micro = {v: {"onset50": [0, 0, 0]} for v in vals}
    n_est = {v: 0 for v in vals}
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
                onset_threshold=LIVE_ONSET, offset_threshold=LIVE_OFFSET,
                frame_threshold=LIVE_FRAME, duplicate_window_sec=v,
                sr=sr, hop=hop,
            )
            est_i, est_p = pred_to_arrays(events)
            m = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None)
            micro[v]["onset50"][0] += len(m)
            micro[v]["onset50"][1] += len(est_i)
            micro[v]["onset50"][2] += len(ref_i)
            n_est[v] += len(est_i)
            _, _, f = prf(len(m), len(est_i), len(ref_i))
            line.append(f"{v:.2f}:{f:.3f}")
        print(f"[{i}/{len(rows)}] {Path(row['midi_filename']).name[:34]:34s} "
              f"infer {infer_dt:.1f}s | onset@50 F1 " + " ".join(line), flush=True)

    print("\n" + "=" * 70)
    print(f"DEDUP WINDOW SWEEP  ({len(rows)} pieces, {(time.perf_counter()-t_start)/60:.1f} min, "
          f"onset={LIVE_ONSET} offset={LIVE_OFFSET} frame={LIVE_FRAME})")
    print("=" * 70)
    print(f"{'dedup_sec':>9s} | {'onset@50ms (existence)':^32s} | {'tot_est':>8s}")
    print(f"{'':>9s} | {'P':>9s} {'R':>9s} {'F1':>9s}  | {'':>8s}")
    best_v, best_f = None, -1.0
    summary = {}
    for v in vals:
        tp, ne, nr = micro[v]["onset50"]
        p, r, f = prf(tp, ne, nr)
        if f > best_f:
            best_f, best_v = f, v
        summary[f"{v:.2f}"] = {"onset50": {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)},
                               "total_est": n_est[v]}
        print(f"{v:>9.2f} | {p:9.4f} {r:9.4f} {f:9.4f}  | {n_est[v]:8d}")
    print(f"\nbest onset@50 micro-F1 at duplicate_window_sec = {best_v:.2f} (F1={best_f:.4f})")
    print(f"production duplicate_window_sec = 0.04 -> F1={summary['0.04']['onset50']['f1']:.4f}, "
          f"est={summary['0.04']['total_est']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "fixed": {"onset_threshold": LIVE_ONSET, "offset_threshold": LIVE_OFFSET,
                  "frame_threshold": LIVE_FRAME},
        "n_pieces": len(rows), "values": vals,
        "best_duplicate_window_sec": best_v, "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
