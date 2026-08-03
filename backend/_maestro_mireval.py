"""Literature-comparable note F1 on the MAESTRO v3 test split via mir_eval.

Runs the enhanced-mel model's OFFLINE full-file path (gpu_ops.transcribe, windowed
10s inference with overlap-stitch) over every piece in the official MAESTRO v3.0.0
test split, then scores raw model note events against raw MIDI ground truth using
mir_eval -- the same metric definition used by the streaming-transcription papers
we compare against (Wei et al. 2503.01362; Hu et al. 2509.07586) and by
Onsets & Frames / Kong et al. It also reports a score-duration surface where
offsets are rewritten from the next printed onset group, matching the app's
score-facing duration authority.

Metrics reported (mir_eval.transcription.match_notes):
  - onset@50ms      : onset within 50ms, pitch exact (offset ignored)        [headline]
  - onset+offset    : + offset within max(50ms, 20% of GT duration)          [Wei headline]
  - strict 10/20/30 : onset-only at tighter tolerances                       [Hu band]

Ground truth = raw MIDI note onsets/offsets (NO pedal extension), the O&F convention.

Usage:
  python _maestro_mireval.py --limit 5            # quick validation subset
  python _maestro_mireval.py                      # full 177-piece test split
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from mir_eval.transcription import match_notes
from mir_eval.util import midi_to_hz

from gpu_ops import get_gpu_enhanced_mel_transcriber

MAESTRO_ROOT = Path("rhythm_training/maestro_midi")
CSV_PATH = MAESTRO_ROOT / "maestro-v3.0.0.csv"
MAESTRO_VERSION = "v3.0.0"
MAESTRO_EVAL_SPLIT = "test"
EXPECTED_TEST_PIECES = 177

# mir_eval onset+offset defaults == Wei et al. definition:
#   offset correct if within max(0.05s, 0.2 * ref_duration)
OFFSET_RATIO = 0.2
OFFSET_MIN_TOL = 0.05
STRICT_TOLS = (0.010, 0.020, 0.030)
SCORE_ONSET_GROUP_TOL = 0.030


def load_test_rows(limit=None):
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == MAESTRO_EVAL_SPLIT]
    rows.sort(key=lambda r: r["midi_filename"])  # deterministic order
    if len(rows) != EXPECTED_TEST_PIECES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_PIECES} MAESTRO {MAESTRO_VERSION} test pieces, "
            f"but {CSV_PATH} contains {len(rows)}. Refusing a potentially incomplete benchmark."
        )
    return rows[:limit] if limit else rows


def test_rows_to_full_piece_clips(rows):
    """Convert official CSV rows to full-length clips for streaming benchmarks."""
    clips = {}
    for index, row in enumerate(rows):
        duration = float(row["duration"])
        clips[f"maestro_test_{index:03d}"] = {
            "audio_path": str(MAESTRO_ROOT / row["audio_filename"]),
            "midi_path": str(MAESTRO_ROOT / row["midi_filename"]),
            "start_sec": 0.0,
            "end_sec": duration,
            "duration_sec": duration,
            "maestro_split": row["split"],
            "maestro_midi_filename": row["midi_filename"],
        }
    return clips


def dataset_metadata(n_pieces):
    """Machine-readable provenance; downstream paper figures reject subsets."""
    complete = n_pieces == EXPECTED_TEST_PIECES
    return {
        "dataset": "MAESTRO",
        "version": MAESTRO_VERSION,
        "split": MAESTRO_EVAL_SPLIT,
        "unit": "full_piece",
        "n_pieces": n_pieces,
        "expected_pieces": EXPECTED_TEST_PIECES,
        "complete_official_split": complete,
        "evaluation_scope": (
            "full_official_test_split" if complete else "development_subset_do_not_publish"
        ),
    }


def gt_notes_from_midi(midi_path):
    """Raw MIDI notes (no pedal extension) -> (intervals Nx2, pitches_hz N)."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    intervals, pitches = [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            intervals.append([n.start, n.end])
            pitches.append(n.pitch)
    intervals = np.asarray(intervals, dtype=float)
    pitches = np.asarray([midi_to_hz(p) for p in pitches], dtype=float)
    return intervals, pitches


def pred_to_arrays(events):
    intervals, pitches = [], []
    for e in events:
        on = float(e["onset_time"])
        off = max(float(e.get("offset_time", on)), on + 1e-3)
        intervals.append([on, off])
        pitches.append(midi_to_hz(int(e["midi_note"])))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.asarray(intervals, dtype=float), np.asarray(pitches, dtype=float)


def event_hand(event):
    return "bass" if int(event["midi_note"]) < 60 else "treble"


def onset_time(event):
    return float(event.get("onset_time", event.get("time_seconds", 0.0)))


def score_ioi_same_hand_events(events, group_tolerance=SCORE_ONSET_GROUP_TOL):
    """Approximate PianoSheetMusic's score-facing duration pass.

    Raw decoder offsets measure acoustic/sounding duration. The printed score
    later rewrites durations from onset spacing: when raw events have no voice_id,
    the front-end falls back to same-staff/same-hand IOI. This keeps onsets and
    pitch counts unchanged and only replaces offset_time with the next onset
    group in the same hand. Last notes keep their raw offsets.
    """
    if not events:
        return []

    ordered = sorted(
        (dict(event) for event in events),
        key=lambda event: (onset_time(event), int(event["midi_note"])),
    )
    groups = []
    for event in ordered:
        event_onset = onset_time(event)
        group = next(
            (
                candidate
                for candidate in groups
                if abs(candidate["time"] - event_onset) <= group_tolerance
            ),
            None,
        )
        if group is None:
            group = {"time": event_onset, "events": []}
            groups.append(group)
        else:
            group["time"] = min(group["time"], event_onset)
        group["events"].append(event)

    for group_index, group in enumerate(groups):
        present_hands = {event_hand(event) for event in group["events"]}
        next_by_hand = {}
        for hand in present_hands:
            next_group = next(
                (
                    candidate
                    for candidate in groups[group_index + 1:]
                    if candidate["time"] - group["time"] > group_tolerance
                    and any(event_hand(event) == hand for event in candidate["events"])
                ),
                None,
            )
            if next_group is not None:
                next_by_hand[hand] = next_group["time"]

        for event in group["events"]:
            next_onset = next_by_hand.get(event_hand(event))
            if next_onset is None:
                continue
            event_onset = onset_time(event)
            if next_onset <= event_onset:
                continue
            event["raw_offset_time"] = float(event.get("offset_time", event_onset))
            event["offset_time"] = next_onset
            event["duration_source"] = "score_ioi_same_hand"

    return ordered


def offset_repeat_cap_events(events, epsilon=1e-3):
    """Cap each note's offset at the next onset of the SAME pitch.

    A note of pitch p must physically release before pitch p sounds again, so
    the next same-pitch onset is a hard upper bound on this note's true offset.
    The raw decoder occasionally over-sustains a note past its re-articulation
    (frame-threshold tail), pushing the estimated offset well beyond the real
    key release. Capping to the next same-pitch onset can only pull such an
    over-long offset back toward the truth -- it never extends a note, and it
    never shortens a note that already ended before its re-onset.

    This makes it monotone-safe for the onset+offset metric: it can move a
    failing offset into tolerance but cannot push a passing offset out, because
    the cap only fires when raw_offset already overshot the re-onset (and was
    therefore already failing). Onsets, pitches, and event count are unchanged.
    """
    if not events:
        return []
    ordered = sorted(
        (dict(event) for event in events),
        key=lambda event: (onset_time(event), int(event["midi_note"])),
    )
    by_pitch = {}
    for event in ordered:
        by_pitch.setdefault(int(event["midi_note"]), []).append(event)
    for pitch_events in by_pitch.values():
        pitch_events.sort(key=onset_time)
        for current, nxt in zip(pitch_events, pitch_events[1:]):
            current_onset = onset_time(current)
            raw_offset = float(current.get("offset_time", current_onset))
            capped = min(raw_offset, onset_time(nxt) - epsilon)
            if current_onset < capped < raw_offset:
                current["raw_offset_time"] = raw_offset
                current["offset_time"] = capped
                current["duration_source"] = "offset_repeat_cap"
    return ordered


def prf(tp, n_est, n_ref):
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_piece(ref_i, ref_p, est_i, est_p):
    out = {}
    n_ref, n_est = len(ref_i), len(est_i)
    # onset-only @50ms (headline)
    m = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None)
    out["onset50"] = (len(m), n_est, n_ref)
    # onset + offset (Wei)
    m = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05,
                    offset_ratio=OFFSET_RATIO, offset_min_tolerance=OFFSET_MIN_TOL)
    out["onset_offset"] = (len(m), n_est, n_ref)
    # strict onset-only band (Hu)
    for tol in STRICT_TOLS:
        m = match_notes(ref_i, ref_p, est_i, est_p, onset_tolerance=tol, offset_ratio=None)
        out[f"strict_{int(tol*1000)}ms"] = (len(m), n_est, n_ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None,
        help="development smoke-test subset; omitted means all 177 test pieces",
    )
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.5)
    ap.add_argument("--offset-threshold", type=float, default=0.35)
    ap.add_argument("--out", default="benchmark_artifacts/maestro_test_mireval.json")
    args = ap.parse_args()

    rows = load_test_rows(args.limit)
    dataset = dataset_metadata(len(rows))
    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")
    print(f"Scoring {len(rows)} MAESTRO test pieces "
          f"(onset_thr={args.onset_threshold}, frame_thr={args.frame_threshold})", flush=True)

    keys = ["onset50", "onset_offset", "strict_30ms", "strict_20ms", "strict_10ms"]
    surfaces = ["raw_decoder", "offset_repeat_cap", "score_ioi_same_hand"]
    micro = {s: {k: [0, 0, 0] for k in keys} for s in surfaces}  # summed (tp, n_est, n_ref)
    macro = {s: {k: [] for k in keys} for s in surfaces}         # per-piece f1
    per_piece = []
    audio_sec_total = 0.0
    compute_sec_total = 0.0

    for i, row in enumerate(rows, 1):
        apath = MAESTRO_ROOT / row["audio_filename"]
        mpath = MAESTRO_ROOT / row["midi_filename"]
        audio, _ = librosa.load(str(apath), sr=16000, mono=True)
        ref_i, ref_p = gt_notes_from_midi(mpath)

        t0 = time.perf_counter()
        res = tx.transcribe(audio, onset_threshold=args.onset_threshold,
                            frame_threshold=args.frame_threshold,
                            offset_threshold=args.offset_threshold)
        dt = time.perf_counter() - t0
        raw_events = res.get("est_note_events", [])
        surface_events = {
            "raw_decoder": raw_events,
            "offset_repeat_cap": offset_repeat_cap_events(raw_events),
            "score_ioi_same_hand": score_ioi_same_hand_events(raw_events),
        }

        rec = {"piece": row["midi_filename"], "n_ref": len(ref_i)}
        for surface, events in surface_events.items():
            est_i, est_p = pred_to_arrays(events)
            sc = score_piece(ref_i, ref_p, est_i, est_p)
            rec[surface] = {"n_est": len(est_i)}
            for k in keys:
                tp, ne, nr = sc[k]
                p, r, f = prf(tp, ne, nr)
                micro[surface][k][0] += tp
                micro[surface][k][1] += ne
                micro[surface][k][2] += nr
                macro[surface][k].append(f)
                rec[surface][k] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        per_piece.append(rec)

        audio_sec = len(audio) / 16000.0
        audio_sec_total += audio_sec
        compute_sec_total += dt
        raw_o = rec["raw_decoder"]["onset_offset"]
        cap_o = rec["offset_repeat_cap"]["onset_offset"]
        print(f"[{i}/{len(rows)}] onset+offset raw={raw_o['f1']:.3f} repeat_cap={cap_o['f1']:.3f} "
              f"(ref={len(ref_i)} est={rec['raw_decoder']['n_est']}) {audio_sec:.0f}s/{dt:.1f}s "
              f"rtf={dt/max(audio_sec,1e-9):.3f}  {Path(row['midi_filename']).name[:40]}",
              flush=True)

    print("\n" + "=" * 70)
    print(f"MAESTRO v3 TEST SPLIT  ({len(rows)} pieces, "
          f"{audio_sec_total/60:.0f} min audio, compute {compute_sec_total/60:.1f} min, "
          f"overall RTF {compute_sec_total/max(audio_sec_total,1e-9):.3f})")
    print(f"onset_thr={args.onset_threshold} frame_thr={args.frame_threshold} "
          f"offset_thr={args.offset_threshold}")
    print("=" * 70)
    print(
        f"{'surface':20s} {'metric':14s} {'macro-F1':>9s} "
        f"{'micro-P':>9s} {'micro-R':>9s} {'micro-F1':>9s}"
    )
    summary = {}
    for surface in surfaces:
        summary[surface] = {}
        for k in keys:
            tp, ne, nr = micro[surface][k]
            mp, mr, mf = prf(tp, ne, nr)
            macro_f1 = float(np.mean(macro[surface][k]))
            summary[surface][k] = {
                "macro_f1": round(macro_f1, 4),
                "micro_p": round(mp, 4),
                "micro_r": round(mr, 4),
                "micro_f1": round(mf, 4),
            }
            print(f"{surface:20s} {k:14s} {macro_f1:9.4f} {mp:9.4f} {mr:9.4f} {mf:9.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": vars(args),
        "dataset": dataset,
        "n_pieces": len(rows),
        "audio_min": round(audio_sec_total / 60, 1),
        "compute_min": round(compute_sec_total / 60, 2),
        "overall_rtf": round(compute_sec_total / max(audio_sec_total, 1e-9), 4),
        "summary": summary,
        "per_piece": per_piece,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
