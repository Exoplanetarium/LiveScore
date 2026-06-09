"""Mine hard-case MAESTRO segments for mel-baseline fine-tuning.

This script scores train/validation/test segments using ground-truth structure
that matches the current mel checkpoint's observed failure modes:
  - dense 3+ note onset clusters
  - pedal-heavy and resonance-heavy passages
  - same-pitch repeats near the live decoder's repeat filter
  - soft/middle-register melody under pedal
  - high note density

Usage:
    python mine_mel_hard_cases.py --split train --target-count 4096
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi


SEGMENT_SECONDS = 10.0
ONSET_CLUSTER_TOLERANCE_SEC = 0.05
LOW_PITCH_CUTOFF = 48
MIDDLE_PITCH_LOW = 48
MIDDLE_PITCH_HIGH = 72
PEDAL_CC = 64
PEDAL_DOWN_THRESHOLD = 64
LONG_SUSTAIN_SEC = 1.0
SAME_PITCH_REPEAT_SEC = 0.35
HARMONIC_LOOKBACK_SEC = 1.2
HARMONIC_INTERVAL_CLASSES = {0, 4, 7}
DEFAULT_TARGET_COUNT = 4096
DEFAULT_MAX_PER_PIECE = 8

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "ensemble_index"
DEFAULT_OUTPUT = ROOT / "mel_hard_case_manifest_train.json"

STRUCTURAL_SCORE_WEIGHTS = {
    "dense_cluster_ratio": 0.35,
    "max_cluster_size": 0.15,
    "low_note_ratio": 0.20,
    "soft_note_ratio": 0.10,
    "low_soft_ratio": 0.10,
    "note_density": 0.10,
}

PEDAL_ONSET_SCORE_WEIGHTS = {
    "pedal_coverage_ratio": 0.16,
    "pedaled_note_ratio": 0.14,
    "middle_pedaled_ratio": 0.14,
    "harmonic_risk_ratio": 0.14,
    "dense_cluster_ratio": 0.13,
    "max_cluster_size": 0.08,
    "same_pitch_repeat_rate": 0.08,
    "soft_note_ratio": 0.06,
    "note_density": 0.05,
    "long_sustain_ratio": 0.02,
}


def load_index(split: str) -> Dict:
    index_path = INDEX_DIR / f"{split}_index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_piece_note_cache(midi_path: str) -> Dict:
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    velocities = []
    control_changes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        control_changes.extend(instrument.control_changes)
        for note in instrument.notes:
            velocity = int(note.velocity)
            notes.append(
                {
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                    "velocity": velocity,
                }
            )
            velocities.append(velocity)
    notes.sort(key=lambda item: (item["start"], item["pitch"]))
    piece_end = max(
        [float(note["end"]) for note in notes] + [float(cc.time) for cc in control_changes] + [0.0]
    )
    if velocities:
        soft_threshold = float(np.quantile(np.asarray(velocities, dtype=np.float64), 1.0 / 3.0))
    else:
        soft_threshold = 48.0
    return {
        "notes": notes,
        "pedal_intervals": build_pedal_intervals(control_changes, piece_end),
        "soft_threshold": soft_threshold,
    }


def build_pedal_intervals(control_changes, piece_end: float) -> List[Tuple[float, float]]:
    changes = [
        (float(cc.time), int(cc.value))
        for cc in control_changes
        if int(cc.number) == PEDAL_CC
    ]
    changes.sort(key=lambda item: item[0])

    intervals: List[Tuple[float, float]] = []
    down_start = None
    for time_sec, value in changes:
        is_down = value >= PEDAL_DOWN_THRESHOLD
        if is_down and down_start is None:
            down_start = time_sec
        elif not is_down and down_start is not None:
            if time_sec > down_start:
                intervals.append((down_start, time_sec))
            down_start = None

    if down_start is not None and piece_end > down_start:
        intervals.append((down_start, piece_end))
    return intervals


def slice_segment_onsets(notes: List[Dict], start_sec: float, end_sec: float) -> List[Dict]:
    return [
        note for note in notes
        if start_sec <= note["start"] < end_sec
    ]


def time_in_intervals(time_sec: float, intervals: List[Tuple[float, float]]) -> bool:
    return any(start <= time_sec < end for start, end in intervals)


def interval_overlap(start_sec: float, end_sec: float, intervals: List[Tuple[float, float]]) -> float:
    overlap = 0.0
    for interval_start, interval_end in intervals:
        overlap += max(0.0, min(end_sec, interval_end) - max(start_sec, interval_start))
    return overlap


def cluster_onsets(notes: List[Dict]) -> List[List[Dict]]:
    if not notes:
        return []

    clusters: List[List[Dict]] = []
    current_cluster = [notes[0]]
    cluster_anchor = notes[0]["start"]
    for note in notes[1:]:
        if note["start"] - cluster_anchor <= ONSET_CLUSTER_TOLERANCE_SEC:
            current_cluster.append(note)
        else:
            clusters.append(current_cluster)
            current_cluster = [note]
            cluster_anchor = note["start"]
    clusters.append(current_cluster)
    return clusters


def count_same_pitch_repeats(notes: List[Dict]) -> int:
    last_onset_by_pitch = {}
    repeats = 0
    for note in notes:
        pitch = int(note["pitch"])
        start_sec = float(note["start"])
        last_onset = last_onset_by_pitch.get(pitch)
        if last_onset is not None and start_sec - last_onset <= SAME_PITCH_REPEAT_SEC:
            repeats += 1
        last_onset_by_pitch[pitch] = start_sec
    return repeats


def count_harmonic_risk_notes(notes: List[Dict], pedal_intervals: List[Tuple[float, float]]) -> int:
    risk_notes = 0
    for idx, note in enumerate(notes):
        note_start = float(note["start"])
        note_pitch = int(note["pitch"])
        if not time_in_intervals(note_start, pedal_intervals):
            continue

        for previous in reversed(notes[:idx]):
            prev_start = float(previous["start"])
            if note_start - prev_start > HARMONIC_LOOKBACK_SEC:
                break
            interval = note_pitch - int(previous["pitch"])
            if interval <= 0:
                continue
            if interval % 12 in HARMONIC_INTERVAL_CLASSES:
                risk_notes += 1
                break
    return risk_notes


def percentile_ranks(values: List[float]) -> List[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]

    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.zeros(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks.tolist()


def build_segment_features(index_payload: Dict) -> List[Dict]:
    pieces = index_payload["pieces"]
    segments = index_payload["segments"]
    piece_cache = {}
    rows = []

    for segment_id, segment in enumerate(segments):
        piece_idx = int(segment["piece_idx"])
        piece = pieces[piece_idx]
        cache_key = piece["midi"]
        if cache_key not in piece_cache:
            piece_cache[cache_key] = load_piece_note_cache(piece["midi"])
        cached = piece_cache[cache_key]
        start_sec = float(segment["start_sec"])
        end_sec = start_sec + SEGMENT_SECONDS
        onset_notes = slice_segment_onsets(cached["notes"], start_sec, end_sec)
        clusters = cluster_onsets(onset_notes)
        pedal_intervals = cached["pedal_intervals"]

        note_count = len(onset_notes)
        dense_cluster_notes = sum(len(cluster) for cluster in clusters if len(cluster) >= 3)
        pedaled_dense_cluster_notes = sum(
            len(cluster)
            for cluster in clusters
            if len(cluster) >= 3 and any(time_in_intervals(float(note["start"]), pedal_intervals) for note in cluster)
        )
        max_cluster_size = max((len(cluster) for cluster in clusters), default=0)
        low_notes = sum(1 for note in onset_notes if note["pitch"] < LOW_PITCH_CUTOFF)
        middle_notes = sum(
            1 for note in onset_notes
            if MIDDLE_PITCH_LOW <= note["pitch"] <= MIDDLE_PITCH_HIGH
        )
        pedaled_notes = sum(1 for note in onset_notes if time_in_intervals(float(note["start"]), pedal_intervals))
        middle_pedaled_notes = sum(
            1
            for note in onset_notes
            if MIDDLE_PITCH_LOW <= note["pitch"] <= MIDDLE_PITCH_HIGH
            and time_in_intervals(float(note["start"]), pedal_intervals)
        )
        soft_notes = sum(1 for note in onset_notes if note["velocity"] <= cached["soft_threshold"])
        low_soft_notes = sum(
            1
            for note in onset_notes
            if note["pitch"] < LOW_PITCH_CUTOFF and note["velocity"] <= cached["soft_threshold"]
        )
        long_sustain_notes = sum(1 for note in onset_notes if float(note["end"]) - float(note["start"]) >= LONG_SUSTAIN_SEC)
        same_pitch_repeats = count_same_pitch_repeats(onset_notes)
        harmonic_risk_notes = count_harmonic_risk_notes(onset_notes, pedal_intervals)
        pedal_coverage = interval_overlap(start_sec, end_sec, pedal_intervals) / SEGMENT_SECONDS

        denom = max(note_count, 1)
        rows.append(
            {
                "segment_id": segment_id,
                "piece_idx": piece_idx,
                "start_sec": start_sec,
                "title": piece.get("title", "Unknown"),
                "composer": piece.get("composer", "Unknown"),
                "feature_file": f"seg_{segment_id:06d}.pt",
                "note_count": note_count,
                "note_density": note_count / SEGMENT_SECONDS,
                "dense_cluster_ratio": dense_cluster_notes / denom,
                "pedaled_dense_cluster_ratio": pedaled_dense_cluster_notes / denom,
                "max_cluster_size": max_cluster_size,
                "low_note_ratio": low_notes / denom,
                "middle_note_ratio": middle_notes / denom,
                "pedal_coverage_ratio": pedal_coverage,
                "pedaled_note_ratio": pedaled_notes / denom,
                "middle_pedaled_ratio": middle_pedaled_notes / denom,
                "soft_note_ratio": soft_notes / denom,
                "low_soft_ratio": low_soft_notes / denom,
                "long_sustain_ratio": long_sustain_notes / denom,
                "same_pitch_repeat_rate": same_pitch_repeats / denom,
                "harmonic_risk_ratio": harmonic_risk_notes / denom,
            }
        )
    return rows


def score_rows(rows: List[Dict], score_weights: Dict[str, float], strategy: str) -> None:
    feature_weights = score_weights
    for feature_name, feature_weight in feature_weights.items():
        ranks = percentile_ranks([float(row[feature_name]) for row in rows])
        for row, rank in zip(rows, ranks):
            row[f"{feature_name}_pct"] = rank

    for row in rows:
        score = 0.0
        for feature_name, feature_weight in feature_weights.items():
            score += feature_weight * float(row[f"{feature_name}_pct"])
        row["hard_case_score"] = score
        flag_candidates = [
            ("dense_3plus", row["dense_cluster_ratio"] >= 0.25 or row["max_cluster_size"] >= 4),
            ("bass_heavy", row["low_note_ratio"] >= 0.35),
            ("soft_heavy", row["soft_note_ratio"] >= 0.45),
            ("soft_bass_overlap", row["low_soft_ratio"] >= 0.20),
            ("high_density", row["note_density"] >= 7.0),
        ]
        if strategy == "pedal_onset_v2":
            flag_candidates.extend(
                [
                    ("pedal_heavy", row["pedal_coverage_ratio"] >= 0.35 or row["pedaled_note_ratio"] >= 0.40),
                    ("pedaled_dense", row["pedaled_dense_cluster_ratio"] >= 0.15),
                    ("middle_pedaled", row["middle_pedaled_ratio"] >= 0.25),
                    ("harmonic_risk", row["harmonic_risk_ratio"] >= 0.15),
                    ("same_pitch_repeats", row["same_pitch_repeat_rate"] >= 0.08),
                    ("long_sustain", row["long_sustain_ratio"] >= 0.35),
                ]
            )
        row["hard_case_flags"] = [flag for flag, enabled in flag_candidates if enabled]


def select_rows(rows: List[Dict], target_count: int, max_per_piece: int) -> List[Dict]:
    selected = []
    per_piece = Counter()
    for row in sorted(rows, key=lambda item: item["hard_case_score"], reverse=True):
        piece_idx = int(row["piece_idx"])
        if per_piece[piece_idx] >= max_per_piece:
            continue
        selected.append(row)
        per_piece[piece_idx] += 1
        if len(selected) >= target_count:
            break
    return selected


def build_manifest(
    split: str,
    selected_rows: List[Dict],
    rows: List[Dict],
    args,
    score_weights: Dict[str, float],
) -> Dict:
    feature_summary = {}
    for feature_name in [
        "dense_cluster_ratio",
        "pedaled_dense_cluster_ratio",
        "max_cluster_size",
        "low_note_ratio",
        "middle_note_ratio",
        "pedal_coverage_ratio",
        "pedaled_note_ratio",
        "middle_pedaled_ratio",
        "soft_note_ratio",
        "low_soft_ratio",
        "long_sustain_ratio",
        "same_pitch_repeat_rate",
        "harmonic_risk_ratio",
        "note_density",
    ]:
        feature_summary[feature_name] = {
            "selected_mean": float(np.mean([row[feature_name] for row in selected_rows])) if selected_rows else 0.0,
            "all_mean": float(np.mean([row[feature_name] for row in rows])) if rows else 0.0,
        }

    return {
        "split": split,
        "segment_ids": [int(row["segment_id"]) for row in selected_rows],
        "selection_strategy": args.strategy,
        "target_count": args.target_count,
        "selected_count": len(selected_rows),
        "max_per_piece": args.max_per_piece,
        "segment_seconds": SEGMENT_SECONDS,
        "onset_cluster_tolerance_sec": ONSET_CLUSTER_TOLERANCE_SEC,
        "low_pitch_cutoff": LOW_PITCH_CUTOFF,
        "middle_pitch_low": MIDDLE_PITCH_LOW,
        "middle_pitch_high": MIDDLE_PITCH_HIGH,
        "pedal_cc": PEDAL_CC,
        "pedal_down_threshold": PEDAL_DOWN_THRESHOLD,
        "same_pitch_repeat_sec": SAME_PITCH_REPEAT_SEC,
        "harmonic_lookback_sec": HARMONIC_LOOKBACK_SEC,
        "harmonic_interval_classes": sorted(HARMONIC_INTERVAL_CLASSES),
        "score_weights": score_weights,
        "feature_summary": feature_summary,
        "selection": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard-case segments for mel-baseline fine-tuning")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--max-per-piece", type=int, default=DEFAULT_MAX_PER_PIECE)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--strategy",
        choices=["pedal_onset_v2", "structural_v1"],
        default="pedal_onset_v2",
    )
    args = parser.parse_args()

    index_payload = load_index(args.split)
    rows = build_segment_features(index_payload)
    score_weights = (
        PEDAL_ONSET_SCORE_WEIGHTS if args.strategy == "pedal_onset_v2" else STRUCTURAL_SCORE_WEIGHTS
    )
    score_rows(rows, score_weights, args.strategy)
    selected_rows = select_rows(rows, args.target_count, args.max_per_piece)
    manifest = build_manifest(args.split, selected_rows, rows, args, score_weights)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote hard-case manifest: {output_path}")
    print(f"Selected {len(selected_rows)} segments from {len(rows)} total")
    for row in selected_rows[:10]:
        print(
            f"  seg={row['segment_id']:06d} score={row['hard_case_score']:.3f} "
            f"density={row['note_density']:.2f}/s pedal={row['pedal_coverage_ratio']:.2f} "
            f"pedaled={row['pedaled_note_ratio']:.2f} middle_pedal={row['middle_pedaled_ratio']:.2f} "
            f"harmonic={row['harmonic_risk_ratio']:.2f} dense={row['dense_cluster_ratio']:.2f} "
            f"title={row['title'][:50]}"
        )


if __name__ == "__main__":
    main()
