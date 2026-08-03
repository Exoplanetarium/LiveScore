"""Naive chunked-streaming baseline vs LiveScore on full MAESTRO v3 test.

This is the primary "must-beat" scientific baseline for the paper: the SAME
transcription model run as naive causal chunked streaming -- consecutive
NON-OVERLAPPING windows, each decoded independently and concatenated, with NO
persistence consensus, NO overlap-aware seam repair, and NO retro-correction.
That isolates the value of LiveScore's architecture (sliding-window inference +
obs>=2 persistence consensus + commit/lock state machine + continuity filters)
from the value of the backbone model, which both arms share.

Naive-chunk emit latency is (by construction) ~= the chunk size, because a
strictly causal chunk cannot be decoded until its last sample has arrived. So
the naive arm is given MORE latency than LiveScore (~100 ms) at every chunk
size reported here -- if it still loses on accuracy / boundary / duplicates, the
architecture, not a latency handicap, is what carries the win.

Both arms are scored with identical metrics on identical GT:
  - onset F1 @50 ms + strict 20/30 ms (mir_eval, matches _live_vs_offline figure)
  - note P/R/F1, offset-aware F1 (test_experiment metrics)
  - boundary recall on the naive chunk grid (where naive chunking has seams)
  - duplicate rate per 100 notes (the naive re-onset-across-seam failure mode)

Boundary recall for BOTH arms is evaluated on the same naive chunk grid, so the
comparison asks: on exactly the notes that sit on a naive seam, how many does
each arm keep? LiveScore has no seam there (overlapping windows), so this is a
fair head-to-head on the naive system's worst-case notes.

Usage:
    env/Scripts/python.exe _naive_vs_livescore_mireval.py   # all 177 full pieces
    env/Scripts/python.exe _naive_vs_livescore_mireval.py --limit 2       # smoke
    env/Scripts/python.exe _naive_vs_livescore_mireval.py --chunk-secs 0.5 1.0 2.0
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
from _maestro_mireval import dataset_metadata, load_test_rows, test_rows_to_full_piece_clips
from test_experiment import (
    TARGET_SR,
    compute_boundary_miss_metrics,
    compute_duplicate_metrics,
    compute_note_metrics,
    compute_offset_metrics,
    load_audio_excerpt,
    load_midi_notes,
    slice_gt_notes,
)
from tune_continuous_stream_decoder import override_live_attrs, run_continuous_replay

STRICT_TOLS = (0.020, 0.030)
BOUNDARY_BAND_SEC = 0.10


def normalize_events(events, offset_sec=0.0):
    """Turn raw est_note_events into the note dicts the metric helpers expect."""
    notes = []
    for e in events:
        onset = float(e.get("onset_time", 0.0) or 0.0) + offset_sec
        offset = float(e.get("offset_time", onset) or onset) + offset_sec
        if offset < onset:
            offset = onset
        notes.append({
            "onset_time": onset,
            "offset_time": offset,
            "duration": max(0.0, offset - onset),
            "midi_note": int(e.get("midi_note", 0) or 0),
            "velocity": int(e.get("velocity", 0) or 0),
        })
    notes.sort(key=lambda n: (n["onset_time"], n["midi_note"]))
    return notes


def naive_chunked_transcribe(tx, audio16, sr, chunk_sec, onset_threshold):
    """Naive causal chunked streaming: non-overlapping consecutive windows, each
    decoded independently, note times shifted into global time, concatenated.
    No dedup, no consensus, no seam repair -- the textbook naive baseline."""
    chunk_frames = max(1, int(round(chunk_sec * sr)))
    out = []
    for start in range(0, audio16.size, chunk_frames):
        chunk = audio16[start:start + chunk_frames]
        if chunk.size < int(0.02 * sr):  # skip a sub-20ms trailing sliver
            continue
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            res = tx.transcribe(chunk, onset_threshold=onset_threshold)
        out.extend(normalize_events(res.get("est_note_events", []), offset_sec=start / sr))
    out.sort(key=lambda n: (n["onset_time"], n["midi_note"]))
    return out


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


def evaluate(pred, gt, chunk_grid_sec):
    """Shared metric bundle for one arm on one clip. Returns count tuples so we
    can micro-aggregate across clips before dividing."""
    ri, rp = to_arrays(gt)
    ei, ep = to_arrays(pred)
    nr, ne = len(ri), len(ei)

    m = match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    onset = {"onset50": (len(m), ne, nr)}
    for tol in STRICT_TOLS:
        m = match_notes(ri, rp, ei, ep, onset_tolerance=tol, offset_ratio=None)
        onset[f"strict_{int(tol * 1000)}ms"] = (len(m), ne, nr)

    note = compute_note_metrics(pred, gt)
    offset = compute_offset_metrics(pred, gt)
    boundary = compute_boundary_miss_metrics(pred, gt, chunk_seconds=chunk_grid_sec,
                                             boundary_band_sec=BOUNDARY_BAND_SEC)
    dup = compute_duplicate_metrics(pred)
    return {
        "onset": onset,
        "note_counts": (note["matched"], note["predicted"], note["ground_truth"]),
        "offset_matched": offset["offset_matched"],
        "boundary": (boundary["boundary_missed_notes"], boundary["boundary_gt_notes"]),
        "dup": (dup["duplicates"], len(pred)),
    }


def base_args():
    return SimpleNamespace(
        tail_padding_sec=0.6, context_sec=1.8, inference_interval_ms=50.0,
        trusted_delay_ms=100.0, commit_delay_ms=500.0, lock_delay_ms=2000.0,
        packet_ms=40.0, chunk_seconds_for_boundary=0.6, eval_boundary_band_sec=0.10,
    )


class Accum:
    """Micro-count accumulator for one arm across all clips (at one chunk grid)."""
    def __init__(self):
        self.onset = {k: [0, 0, 0] for k in ("onset50", "strict_20ms", "strict_30ms")}
        self.note = [0.0, 0.0, 0.0]
        self.offset_matched = 0.0
        self.boundary = [0.0, 0.0]
        self.dup = [0.0, 0.0]

    def add(self, ev):
        for k, (tp, ne, nr) in ev["onset"].items():
            self.onset[k][0] += tp; self.onset[k][1] += ne; self.onset[k][2] += nr
        for i in range(3):
            self.note[i] += ev["note_counts"][i]
        self.offset_matched += ev["offset_matched"]
        self.boundary[0] += ev["boundary"][0]; self.boundary[1] += ev["boundary"][1]
        self.dup[0] += ev["dup"][0]; self.dup[1] += ev["dup"][1]

    def summary(self):
        out = {}
        for k, (tp, ne, nr) in self.onset.items():
            p, r, f = prf(tp, ne, nr)
            out[k] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
        m, pr, g = self.note
        p, r, f = prf(m, pr, g)
        out["note"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                       "matched": m, "predicted": pr, "ground_truth": g}
        op = self.offset_matched / pr if pr else 0.0
        orr = self.offset_matched / g if g else 0.0
        out["offset_f1"] = round(2 * op * orr / (op + orr) if (op + orr) else 0.0, 4)
        missed, bgt = self.boundary
        out["boundary_recall"] = round(1.0 - missed / bgt, 4) if bgt else None
        out["boundary_gt_notes"] = bgt
        dups, npred = self.dup
        out["duplicates_per_100"] = round(100.0 * dups / npred, 3) if npred else 0.0
        out["duplicates"] = dups
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None,
        help="development smoke-test subset (marked do-not-publish in output)",
    )
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--chunk-secs", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--out", default="benchmark_artifacts/naive_vs_livescore_mireval.json")
    args = ap.parse_args()

    clips = test_rows_to_full_piece_clips(load_test_rows(args.limit))
    dataset = dataset_metadata(len(clips))

    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable.")

    chunk_secs = list(args.chunk_secs)
    print(f"Scoring {len(clips)} clips | LiveScore (obs2/int50/td100) vs naive chunks "
          f"{chunk_secs}s @ onset_thr={args.onset_threshold}", flush=True)

    # LiveScore is scored once per naive grid (its notes don't change; only the
    # boundary grid used to *evaluate* it changes). Naive arm: one per chunk size.
    live_acc = {c: Accum() for c in chunk_secs}
    naive_acc = {c: Accum() for c in chunk_secs}
    rep = base_args()

    for i, (cid, clip) in enumerate(clips.items(), 1):
        gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])

        with override_live_attrs({"STREAM_MIN_DISPLAY_OBSERVATIONS": 2}):
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                res = run_continuous_replay(clip, rep)
        live_notes = res["score_notes"]

        audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
        audio16 = librosa.resample(audio, orig_sr=TARGET_SR, target_sr=16000)

        line = [f"[{i}/{len(clips)}] {cid} gt={len(gt)} live={len(live_notes)}"]
        for c in chunk_secs:
            naive_notes = naive_chunked_transcribe(tx, audio16, 16000, c, args.onset_threshold)
            live_ev = evaluate(live_notes, gt, c)
            naive_ev = evaluate(naive_notes, gt, c)
            live_acc[c].add(live_ev)
            naive_acc[c].add(naive_ev)
            lf = prf(*live_ev["onset"]["onset50"])[2]
            nf = prf(*naive_ev["onset"]["onset50"])[2]
            line.append(f"| c{c}: live_f1={lf:.3f} naive_f1={nf:.3f} (n={len(naive_notes)})")
        print(" ".join(line), flush=True)

    summary = {}
    for c in chunk_secs:
        summary[f"{c}"] = {
            "chunk_sec": c,
            "naive_emit_latency_ms": round(c * 1000, 1),
            "livescore": live_acc[c].summary(),
            "naive_chunked": naive_acc[c].summary(),
        }

    payload = {
        "dataset": dataset,
        "n_pieces": len(clips),
        "onset_threshold": args.onset_threshold,
        "livescore_emit_latency_ms": 100.0,
        "note": ("Naive = same model, non-overlapping causal chunks, independent "
                 "decode, concatenated; no consensus / seam-repair / retro-correction. "
                 "boundary_recall for both arms is measured on the naive chunk grid."),
        "by_chunk_sec": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- console side-by-side ------------------------------------------------
    print("\n" + "=" * 84)
    print(f"{len(clips)} full-piece MAESTRO-test: LiveScore (~100 ms) vs naive chunked streaming")
    print("=" * 84)
    for c in chunk_secs:
        L = summary[f"{c}"]["livescore"]
        N = summary[f"{c}"]["naive_chunked"]
        print(f"\n--- naive chunk = {c:.1f} s  (naive emit latency ~{c*1000:.0f} ms) ---")
        hdr = f"{'metric':22s} {'LiveScore':>12s} {'naive':>12s} {'d(Live-naive)':>16s}"
        print(hdr); print("-" * len(hdr))
        rows = [
            ("onset F1 @50ms", L["onset50"]["f1"], N["onset50"]["f1"]),
            ("onset P @50ms", L["onset50"]["precision"], N["onset50"]["precision"]),
            ("onset R @50ms", L["onset50"]["recall"], N["onset50"]["recall"]),
            ("strict F1 @30ms", L["strict_30ms"]["f1"], N["strict_30ms"]["f1"]),
            ("strict F1 @20ms", L["strict_20ms"]["f1"], N["strict_20ms"]["f1"]),
            ("offset-aware F1", L["offset_f1"], N["offset_f1"]),
            ("boundary recall", L["boundary_recall"], N["boundary_recall"]),
            ("duplicates/100", L["duplicates_per_100"], N["duplicates_per_100"]),
        ]
        for name, lv, nv in rows:
            if lv is None or nv is None:
                print(f"{name:22s} {str(lv):>12s} {str(nv):>12s}")
                continue
            print(f"{name:22s} {lv:>12.4f} {nv:>12.4f} {lv - nv:>+16.4f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
