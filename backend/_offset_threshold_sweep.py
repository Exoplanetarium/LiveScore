"""Sweep the decoder offset threshold on the MAESTRO v3 test split.

Question: is the production offset threshold (0.35, LIVE_ENHANCED_OFFSET_BASE)
actually on the F1 frontier, or just an inherited default?

Method: for each test piece we run the windowed 10s inference EXACTLY ONCE
(replicating gpu_ops.transcribe's head assembly + overlap-averaging), cache all
per-frame heads, then re-run decode_enhanced_note_events at each candidate offset
threshold -- holding onset_threshold=0.70 and frame_threshold=0.5 fixed at the
live production values. Decode is cheap; inference is the cost, so this is N_thr
times faster than calling tx.transcribe() per threshold.

Only the onset+offset (Wei) metric responds to the offset threshold; onset@50ms
is reported as an invariance check (it must stay flat across the sweep).

Usage (from backend/):
  python _offset_threshold_sweep.py --limit 25
  python _offset_threshold_sweep.py --thresholds 0.2,0.25,0.3,0.35,0.4,0.45,0.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from gpu_ops import DEVICE, get_gpu_enhanced_mel_transcriber
from rhythm_training.train_enhanced_mel_transcriber import decode_enhanced_note_events

from _maestro_mireval import (MAESTRO_ROOT, OFFSET_MIN_TOL, OFFSET_RATIO,
                              gt_notes_from_midi, load_test_rows, pred_to_arrays,
                              prf)
from mir_eval.transcription import match_notes

LIVE_ONSET = 0.70
LIVE_FRAME = 0.50


@torch.no_grad()
def infer_heads(tx, audio):
    """One windowed inference pass -> overlap-averaged (onset, offset, frame, vel, nv)."""
    sr = tx.config.get("sample_rate", 16000)
    hop = tx.config.get("hop_length", 256)
    n_keys = tx.config.get("n_keys", 88)
    use_nv = bool(tx.config.get("use_note_value_head", True))
    n_nv = int(tx.config.get("n_note_value_classes", 12)) if use_nv else 0

    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    features = tx.extractor.extract(audio_t)
    n_frames = features.size(1)

    chunk_frames = int(10.0 * sr / hop)
    overlap = chunk_frames // 4
    step = chunk_frames - overlap

    all_onset = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_offset = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_frame = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_vel = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_nv = np.zeros((n_frames, n_keys, n_nv), dtype=np.float32) if n_nv else None
    counts = np.zeros(n_frames, dtype=np.float32)

    for start in range(0, n_frames, step):
        end = min(start + chunk_frames, n_frames)
        out = tx.model(features[:, start:end, :])
        frame_key = "sounding_frame_logits" if "sounding_frame_logits" in out else "frame_logits"
        on = torch.sigmoid(out["onset_logits"][0]).cpu().numpy()
        off = torch.sigmoid(out["offset_logits"][0]).cpu().numpy()
        fr = torch.sigmoid(out[frame_key][0]).cpu().numpy()
        vel = out["velocity"][0].cpu().numpy()
        L = end - start
        all_onset[start:end] += on[:L]
        all_offset[start:end] += off[:L]
        all_frame[start:end] += fr[:L]
        all_vel[start:end] += vel[:L]
        if all_nv is not None and "note_value_logits" in out:
            nv = F.softmax(out["note_value_logits"][0], dim=-1).cpu().numpy()
            if nv.shape[-1] >= n_nv:
                all_nv[start:end] += nv[:L, :, :n_nv]
            else:
                all_nv[start:end, :, :nv.shape[-1]] += nv[:L]
        counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    all_onset /= counts[:, None]
    all_offset /= counts[:, None]
    all_frame /= counts[:, None]
    all_vel /= counts[:, None]
    if all_nv is not None:
        all_nv /= counts[:, None, None]
    return all_onset, all_offset, all_frame, all_vel, all_nv, sr, hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--out", default="benchmark_artifacts/offset_threshold_sweep.json")
    args = ap.parse_args()

    thrs = [float(x) for x in args.thresholds.split(",")]
    rows = load_test_rows(args.limit)
    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")
    print(f"Offset sweep over {thrs}\n  fixed onset={LIVE_ONSET} frame={LIVE_FRAME}, "
          f"{len(rows)} MAESTRO test pieces\n", flush=True)

    # summed (tp, n_est, n_ref) per threshold for onset+offset and onset50
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
                offset_threshold=t,
                frame_threshold=LIVE_FRAME,
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
    print(f"OFFSET THRESHOLD SWEEP  ({len(rows)} pieces, "
          f"{(time.perf_counter()-t_start)/60:.1f} min, onset={LIVE_ONSET} frame={LIVE_FRAME})")
    print("=" * 64)
    print(f"{'offset_thr':>10s} | {'onset+offset (Wei)':^30s} | {'onset@50ms':^12s}")
    print(f"{'':>10s} | {'P':>8s} {'R':>8s} {'F1':>8s}  | {'F1':>10s}")
    best_t, best_f = None, -1.0
    summary = {}
    for t in thrs:
        tp, ne, nr = micro[t]["onset_offset"]
        p, r, f = prf(tp, ne, nr)
        _, _, f_on = prf(*micro[t]["onset50"])
        flag = ""
        if f > best_f:
            best_f, best_t = f, t
        summary[f"{t:.2f}"] = {"onset_offset": {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)},
                                "onset50_f1": round(f_on, 4)}
        print(f"{t:>10.2f} | {p:8.4f} {r:8.4f} {f:8.4f}  | {f_on:10.4f}")
    print(f"\nbest onset+offset micro-F1 at offset_threshold = {best_t:.2f} (F1={best_f:.4f})")
    print(f"production offset_threshold = 0.35 -> F1={summary['0.35']['onset_offset']['f1']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "fixed": {"onset_threshold": LIVE_ONSET, "frame_threshold": LIVE_FRAME},
        "n_pieces": len(rows), "thresholds": thrs,
        "best_offset_threshold": best_t, "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
