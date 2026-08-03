"""Isolate the live STREAMING-regime cost from the velocity-confidence gates.

The shipped comparison (_live_vs_offline_mireval.py) pits offline-raw against
live-WITH-gates, which confounds two effects. This adds the missing configs that
trust all onset probs (no velocity-confidence gating) so the cost of the
chunking+overlap regime can be read on its own.

Four surfaces on all 177 full pieces in the official MAESTRO v3 test split,
onset F1 @50ms:
  A. offline        : tx.transcribe() full excerpt, raw decoder, no gates, full ctx
  B. live_gated     : shipped streaming (obs>=2, all confidence gates ON)
  C. live_nogate2   : streaming, confidence gates OFF, persistence obs>=2
  D. live_nogate1   : streaming, confidence gates OFF, obs>=1  (trust all onsets)

Confidence gates OFF = all LIVE_NOISE_FILTER_PROFILES *_min_confidence -> 0,
STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE -> 0, STREAM_WEAK_BIRTH_HIGH_CONFIDENCE
-> 0 (each gate's predicate is `confidence < X`, so X=0 never fires).

Decomposition:  A->D = pure regime cost (raw streaming vs raw offline);
B->D / B->C = what the velocity gates buy in-regime.

Usage (from backend/):
  python _live_nogate_mireval.py --limit 8     # smoke
  python _live_nogate_mireval.py               # all 177 full pieces
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path

import librosa
import numpy as np
from mir_eval.transcription import match_notes
from mir_eval.util import midi_to_hz

import main as live_main
from gpu_ops import get_gpu_enhanced_mel_transcriber
from test_experiment import TARGET_SR, load_audio_excerpt, load_midi_notes, slice_gt_notes

from _live_vs_offline_mireval import base_args, prf, score
from _maestro_mireval import dataset_metadata, load_test_rows, test_rows_to_full_piece_clips


def zeroed_profiles():
    profiles = copy.deepcopy(live_main.LIVE_NOISE_FILTER_PROFILES)
    for prof in profiles.values():
        for k in list(prof.keys()):
            if k.endswith("_min_confidence"):
                prof[k] = 0.0
    return profiles


GATES_OFF = {
    "LIVE_NOISE_FILTER_PROFILES": zeroed_profiles(),
    "STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE": 0.0,
    "STREAM_WEAK_BIRTH_HIGH_CONFIDENCE": 0.0,
}

SURFACES = ["offline", "live_gated", "live_nogate2", "live_nogate1"]
KEYS = ["onset50", "strict_30ms", "strict_20ms", "strict_10ms"]


def run_live(clip, rep, attrs):
    from tune_continuous_stream_decoder import override_live_attrs, run_continuous_replay
    with override_live_attrs(attrs):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            res = run_continuous_replay(clip, rep)
    return res["score_notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None,
        help="development smoke-test subset (marked do-not-publish in output)",
    )
    ap.add_argument("--onset-threshold", type=float, default=0.70,
                    help="offline decode onset threshold (default = live prod 0.70)")
    ap.add_argument("--out", default="benchmark_artifacts/live_nogate_mireval.json")
    args = ap.parse_args()

    clips = test_rows_to_full_piece_clips(load_test_rows(args.limit))
    dataset = dataset_metadata(len(clips))

    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable.")
    print(f"Scoring {len(clips)} clips across {SURFACES} (offline thr={args.onset_threshold})\n", flush=True)

    micro = {s: {k: [0, 0, 0] for k in KEYS} for s in SURFACES}
    macro = {s: {k: [] for k in KEYS} for s in SURFACES}
    rep = base_args()

    gates_on = {"STREAM_MIN_DISPLAY_OBSERVATIONS": 2}
    gates_off2 = {"STREAM_MIN_DISPLAY_OBSERVATIONS": 2, **GATES_OFF}
    gates_off1 = {"STREAM_MIN_DISPLAY_OBSERVATIONS": 1, **GATES_OFF}

    for i, (cid, clip) in enumerate(clips.items(), 1):
        gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])

        notes_by_surface = {
            "live_gated": run_live(clip, rep, gates_on),
            "live_nogate2": run_live(clip, rep, gates_off2),
            "live_nogate1": run_live(clip, rep, gates_off1),
        }
        audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
        audio16 = librosa.resample(audio, orig_sr=TARGET_SR, target_sr=16000)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            notes_by_surface["offline"] = tx.transcribe(
                audio16, onset_threshold=args.onset_threshold).get("est_note_events", [])

        cells = []
        for surf in SURFACES:
            sc = score(notes_by_surface[surf], gt)
            for k in KEYS:
                tp, ne, nr = sc[k]
                micro[surf][k][0] += tp; micro[surf][k][1] += ne; micro[surf][k][2] += nr
                macro[surf][k].append(prf(tp, ne, nr)[2])
            p, r, f = prf(*sc["onset50"])
            cells.append(f"{surf.split('_')[-1] if '_' in surf else surf[:3]}={f:.3f}")
        print(f"[{i}/{len(clips)}] {cid} gt={len(gt)} | "
              + " ".join(cells), flush=True)

    print("\n" + "=" * 70)
    print(f"{len(clips)} full-piece MAESTRO-test benchmark | onset F1 @50ms | "
          f"offline_thr={args.onset_threshold}")
    print("=" * 70)
    print(f"{'surface':14s} {'micro-P':>9s} {'micro-R':>9s} {'micro-F1':>9s} {'macro-F1':>9s}")
    summary = {}
    for surf in SURFACES:
        summary[surf] = {}
        for k in KEYS:
            tp, ne, nr = micro[surf][k]
            mp, mr, mf = prf(tp, ne, nr)
            summary[surf][k] = {"micro_p": round(mp, 4), "micro_r": round(mr, 4),
                                "micro_f1": round(mf, 4), "macro_f1": round(float(np.mean(macro[surf][k])), 4)}
        s = summary[surf]["onset50"]
        print(f"{surf:14s} {s['micro_p']:9.4f} {s['micro_r']:9.4f} {s['micro_f1']:9.4f} {s['macro_f1']:9.4f}")

    f = lambda s: summary[s]["onset50"]["micro_f1"]
    print("\nDecomposition (onset@50 micro-F1):")
    print(f"  regime cost   A->D  offline {f('offline'):.4f} -> nogate-obs1 {f('live_nogate1'):.4f}  = {f('live_nogate1')-f('offline'):+.4f}")
    print(f"  persistence   D->C  obs1 {f('live_nogate1'):.4f} -> obs2 {f('live_nogate2'):.4f}        = {f('live_nogate2')-f('live_nogate1'):+.4f}")
    print(f"  conf gates    C->B  nogate {f('live_nogate2'):.4f} -> gated {f('live_gated'):.4f}      = {f('live_gated')-f('live_nogate2'):+.4f}")
    print(f"  net regime+gate A->B offline {f('offline'):.4f} -> shipped {f('live_gated'):.4f}     = {f('live_gated')-f('offline'):+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "dataset": dataset,
        "n_pieces": len(clips),
        "offline_threshold": args.onset_threshold,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
