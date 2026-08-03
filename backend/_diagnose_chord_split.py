"""Diagnose the "single 32nd note + rest of the chord" split.

The live grouper (_group_neural_note_events_by_onset) starts a new chord group
when EITHER of two gates trips:
  within_group_span   : onset - group_start    <= span_tolerance   (base 30ms)
  near_previous_attack : onset - previous_onset <= step_tolerance   (span * STEP_RATIO = 15ms)

Hypothesis: the orphaned-first-note symptom is driven by the STEP gate, not the
span. When a chord's notes arrive at e.g. 0 / 18 / 33 ms, the 18ms note fails the
15ms step gate even though it's well within the 30ms span, so the first note is
emitted alone (tiny IOI duration -> 32nd) and the rest of the chord forms the
next group.

This script runs the enhanced-mel model over a few MAESTRO test pieces, replays
the REAL grouping logic instrumented to record which gate caused each boundary,
and counts the orphan-singleton-immediately-before-a-chord pattern (plus the
duration the orphan would be rendered with).

Usage:
  python _diagnose_chord_split.py --limit 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa

import detect_note as dn
from _maestro_mireval import load_test_rows, MAESTRO_ROOT
from gpu_ops import get_gpu_enhanced_mel_transcriber


def instrumented_groups(note_events):
    """Mirror _group_neural_note_events_by_onset but record the boundary cause.

    Returns list of (group, boundary_cause) where boundary_cause explains why
    THIS group was split off from the previous one:
      'span'  : within_group_span failed (cumulative spread > span_tolerance)
      'step'  : within_group_span held but near_previous_attack failed
      None    : first group
    """
    events = sorted(note_events or [], key=lambda e: e["onset_time"])
    groups = []
    causes = []
    current = []
    for event in events:
        if not current:
            current = [event]
            continue
        onset = float(event.get("onset_time", 0.0) or 0.0)
        group_start = float(current[0].get("onset_time", 0.0) or 0.0)
        prev_onset = float(current[-1].get("onset_time", group_start) or group_start)
        span_tol, step_tol = dn._adaptive_neural_group_tolerances(len(current))
        within_span = (onset - group_start) <= span_tol
        near_prev = (onset - prev_onset) <= step_tol
        if within_span and near_prev:
            current.append(event)
            continue
        groups.append(current)
        causes.append("span" if not within_span else "step")
        current = [event]
    if current:
        groups.append(current)
        causes.append(None if not groups else "tail")
    return groups, causes


def analyze(note_events, split_midi=60):
    groups, _ = instrumented_groups(note_events)
    span_base = dn._NEURAL_SIMULTANEOUS_BASE_TOLERANCE_SEC
    stats = {
        "groups": len(groups),
        "singletons": 0,
        "chords": 0,
        # singleton followed within span_base by a chord, but the gap to that
        # chord's first onset exceeded the step gate -> step-induced orphan
        "orphan_step_before_chord": 0,
        "orphan_dur_under_60ms": 0,
        "examples": [],
    }
    starts = [float(g[0]["onset_time"]) for g in groups]
    for i, g in enumerate(groups):
        if len(g) == 1:
            stats["singletons"] += 1
        else:
            stats["chords"] += 1
        if i + 1 >= len(groups):
            continue
        nxt = groups[i + 1]
        gap = starts[i + 1] - starts[i]                  # onset spacing to next group
        last_onset = float(g[-1]["onset_time"])
        step_gap = starts[i + 1] - last_onset            # what the step gate saw
        _, step_tol = dn._adaptive_neural_group_tolerances(len(g))
        same_hand = (min(int(e["midi_note"]) for e in g) < split_midi) == (
            min(int(e["midi_note"]) for e in nxt) < split_midi
        )
        # orphan-first-note-of-chord: this group is a singleton, the next is a
        # chord, they are within the SPAN window (would have merged span-only),
        # but the step gate split them.
        if (
            len(g) == 1
            and len(nxt) >= 2
            and same_hand
            and gap <= span_base
            and step_gap > step_tol
        ):
            stats["orphan_step_before_chord"] += 1
            orphan_dur = gap  # IOI duration the orphan gets = time to next group
            if orphan_dur < 0.060:
                stats["orphan_dur_under_60ms"] += 1
            if len(stats["examples"]) < 8:
                stats["examples"].append({
                    "t": round(starts[i], 3),
                    "orphan_midi": int(g[0]["midi_note"]),
                    "gap_ms": round(gap * 1000, 1),
                    "step_tol_ms": round(step_tol * 1000, 1),
                    "chord_midis": sorted(int(e["midi_note"]) for e in nxt),
                    "orphan_dur_ms": round(orphan_dur * 1000, 1),
                })
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.5)
    ap.add_argument("--offset-threshold", type=float, default=0.35)
    args = ap.parse_args()

    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not tx.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")

    span_base = dn._NEURAL_SIMULTANEOUS_BASE_TOLERANCE_SEC
    ratio = dn._NEURAL_SIMULTANEOUS_STEP_RATIO
    print(f"span_base={span_base*1000:.0f}ms  step_ratio={ratio}  "
          f"=> step_tol(2-note)={span_base*ratio*1000:.0f}ms", flush=True)

    rows = load_test_rows(args.limit)
    agg = {"groups": 0, "singletons": 0, "chords": 0,
           "orphan_step_before_chord": 0, "orphan_dur_under_60ms": 0}
    for i, row in enumerate(rows, 1):
        audio, _ = librosa.load(str(MAESTRO_ROOT / row["audio_filename"]), sr=16000, mono=True)
        res = tx.transcribe(audio, onset_threshold=args.onset_threshold,
                            frame_threshold=args.frame_threshold,
                            offset_threshold=args.offset_threshold)
        events = res.get("est_note_events", [])
        s = analyze(events)
        for k in agg:
            agg[k] += s[k]
        orphan_pct = 100.0 * s["orphan_step_before_chord"] / max(s["chords"], 1)
        print(f"[{i}/{len(rows)}] groups={s['groups']} chords={s['chords']} "
              f"step-orphans-before-chord={s['orphan_step_before_chord']} "
              f"({orphan_pct:.1f}% of chords)  {Path(row['midi_filename']).name[:34]}",
              flush=True)
        for ex in s["examples"][:3]:
            print(f"      t={ex['t']}s orphan={ex['orphan_midi']} gap={ex['gap_ms']}ms "
                  f"(step_tol={ex['step_tol_ms']}ms) -> chord {ex['chord_midis']} "
                  f"orphan_dur={ex['orphan_dur_ms']}ms", flush=True)

    print("\n" + "=" * 64)
    print(f"TOTAL groups={agg['groups']}  chords={agg['chords']}")
    print(f"step-induced orphan-before-chord: {agg['orphan_step_before_chord']} "
          f"({100.0*agg['orphan_step_before_chord']/max(agg['chords'],1):.1f}% of chords)")
    print(f"  of which orphan would render <60ms (32nd-ish): "
          f"{agg['orphan_dur_under_60ms']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
