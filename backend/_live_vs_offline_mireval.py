"""At-latency onset F1: live/chunked path vs offline path on the SAME clips.

Scores both surfaces with mir_eval (onset@50ms + strict 10/20/30ms) on every
full-length piece in the official 177-piece MAESTRO v3 test split,
so we can quote the live system's literature-comparable onset F1 at its shipped
~100ms latency and isolate exactly what going real-time costs vs the offline
model on identical audio.

Live config = shipped: STREAM_MIN_DISPLAY_OBSERVATIONS=2, interval 50ms,
trusted_delay 100ms (score surface = committed/locked notes shown to the user).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
from mir_eval.transcription import match_notes
from mir_eval.util import midi_to_hz

from gpu_ops import get_gpu_enhanced_mel_transcriber
from test_experiment import TARGET_SR, load_audio_excerpt, load_midi_notes, slice_gt_notes
from tune_continuous_stream_decoder import override_live_attrs, run_continuous_replay

from _maestro_mireval import dataset_metadata, load_test_rows, test_rows_to_full_piece_clips

STRICT_TOLS = (0.010, 0.020, 0.030)
KEYS = ["onset50", "strict_30ms", "strict_20ms", "strict_10ms"]


def to_arrays(notes):
    iv, pi = [], []
    for n in notes:
        on = float(n["onset_time"])
        off = max(float(n.get("offset_time", on)), on + 1e-3)
        iv.append([on, off]); pi.append(midi_to_hz(int(n["midi_note"])))
    if not iv:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.asarray(iv, float), np.asarray(pi, float)


def prf(tp, ne, nr):
    p = tp / ne if ne else 0.0
    r = tp / nr if nr else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score(pred, gt):
    ri, rp = to_arrays(gt)
    ei, ep = to_arrays(pred)
    nr, ne = len(ri), len(ei)
    out = {}
    m = match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    out["onset50"] = (len(m), ne, nr)
    for tol in STRICT_TOLS:
        m = match_notes(ri, rp, ei, ep, onset_tolerance=tol, offset_ratio=None)
        out[f"strict_{int(tol*1000)}ms"] = (len(m), ne, nr)
    return out


def base_args():
    return SimpleNamespace(
        tail_padding_sec=0.6, context_sec=1.8, inference_interval_ms=50.0,
        trusted_delay_ms=100.0, commit_delay_ms=500.0, lock_delay_ms=2000.0,
        packet_ms=40.0, chunk_seconds_for_boundary=0.6, eval_boundary_band_sec=0.10,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None,
        help="development smoke-test subset (marked do-not-publish in output)",
    )
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="benchmark_artifacts/live_vs_offline_mireval.json")
    args = ap.parse_args()

    clips = test_rows_to_full_piece_clips(load_test_rows(args.limit))
    dataset = dataset_metadata(len(clips))

    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable.")
    print(f"Scoring {len(clips)} clips (live obs2/int50/td100 vs offline thr={args.onset_threshold})", flush=True)

    micro = {surf: {k: [0, 0, 0] for k in KEYS} for surf in ("live", "offline")}
    macro = {surf: {k: [] for k in KEYS} for surf in ("live", "offline")}
    rep = base_args()

    for i, (cid, clip) in enumerate(clips.items(), 1):
        gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
        # live path
        with override_live_attrs({"STREAM_MIN_DISPLAY_OBSERVATIONS": 2}):
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                res = run_continuous_replay(clip, rep)
        live_notes = res["score_notes"]
        # offline path on the SAME excerpt; transcribe() assumes 16kHz so resample
        # (TARGET_SR is 44100; the live session resamples internally).
        audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
        audio16 = librosa.resample(audio, orig_sr=TARGET_SR, target_sr=16000)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            off_notes = tx.transcribe(audio16, onset_threshold=args.onset_threshold).get("est_note_events", [])

        for surf, notes in (("live", live_notes), ("offline", off_notes)):
            sc = score(notes, gt)
            for k in KEYS:
                tp, ne, nr = sc[k]
                micro[surf][k][0] += tp; micro[surf][k][1] += ne; micro[surf][k][2] += nr
                macro[surf][k].append(prf(tp, ne, nr)[2])

        lo = prf(*score(live_notes, gt)["onset50"])
        of = prf(*score(off_notes, gt)["onset50"])
        print(f"[{i}/{len(clips)}] {cid}: live f1={lo[2]:.3f} (p={lo[0]:.3f} r={lo[1]:.3f}) | "
              f"offline f1={of[2]:.3f} (gt={len(gt)} live={len(live_notes)} off={len(off_notes)})", flush=True)

    print("\n" + "=" * 64)
    print(f"{len(clips)} full-piece MAESTRO-test benchmark | onset F1 @50ms (+strict band)")
    print("=" * 64)
    summary = {}
    for surf in ("offline", "live"):
        print(f"\n--- {surf} ---")
        print(f"{'metric':12s} {'macro-F1':>9s} {'micro-P':>9s} {'micro-R':>9s} {'micro-F1':>9s}")
        summary[surf] = {}
        for k in KEYS:
            tp, ne, nr = micro[surf][k]
            mp, mr, mf = prf(tp, ne, nr)
            mac = float(np.mean(macro[surf][k]))
            summary[surf][k] = {"macro_f1": round(mac, 4), "micro_p": round(mp, 4),
                                "micro_r": round(mr, 4), "micro_f1": round(mf, 4)}
            print(f"{k:12s} {mac:9.4f} {mp:9.4f} {mr:9.4f} {mf:9.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "dataset": dataset,
        "n_pieces": len(clips),
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
