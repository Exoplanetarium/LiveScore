"""Mine hard-case MAESTRO segments for mel-baseline fine-tuning.

This script scores train/validation/test segments using ground-truth structure
that matches the current mel checkpoint's observed failure modes:
  - dense 3+ note onset clusters
  - bass-heavy passages
  - soft-note-heavy passages
  - high note density

Usage:
    python mine_mel_hard_cases.py --split train --target-count 4096
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pretty_midi


SEGMENT_SECONDS = 10.0
ONSET_CLUSTER_TOLERANCE_SEC = 0.05
LOW_PITCH_CUTOFF = 48
DEFAULT_TARGET_COUNT = 4096
DEFAULT_MAX_PER_PIECE = 8

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "ensemble_index"
DEFAULT_OUTPUT = ROOT / "mel_hard_case_manifest_train.json"


def load_index(split: str) -> Dict:
    index_path = INDEX_DIR / f"{split}_index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_piece_note_cache(midi_path: str) -> Dict:
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    velocities = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
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
    if velocities:
        soft_threshold = float(np.quantile(np.asarray(velocities, dtype=np.float64), 1.0 / 3.0))
    else:
        soft_threshold = 48.0
    return {
        "notes": notes,
        "soft_threshold": soft_threshold,
    }


def slice_segment_onsets(notes: List[Dict], start_sec: float, end_sec: float) -> List[Dict]:
    return [
        note for note in notes
        if start_sec <= note["start"] < end_sec
    ]


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

        note_count = len(onset_notes)
        dense_cluster_notes = sum(len(cluster) for cluster in clusters if len(cluster) >= 3)
        max_cluster_size = max((len(cluster) for cluster in clusters), default=0)
        low_notes = sum(1 for note in onset_notes if note["pitch"] < LOW_PITCH_CUTOFF)
        soft_notes = sum(1 for note in onset_notes if note["velocity"] <= cached["soft_threshold"])
        low_soft_notes = sum(
            1
            for note in onset_notes
            if note["pitch"] < LOW_PITCH_CUTOFF and note["velocity"] <= cached["soft_threshold"]
        )

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
                "max_cluster_size": max_cluster_size,
                "low_note_ratio": low_notes / denom,
                "soft_note_ratio": soft_notes / denom,
                "low_soft_ratio": low_soft_notes / denom,
            }
        )
    return rows


def score_rows(rows: List[Dict]) -> None:
    feature_weights = {
        "dense_cluster_ratio": 0.35,
        "max_cluster_size": 0.15,
        "low_note_ratio": 0.20,
        "soft_note_ratio": 0.10,
        "low_soft_ratio": 0.10,
        "note_density": 0.10,
    }

    for feature_name, feature_weight in feature_weights.items():
        ranks = percentile_ranks([float(row[feature_name]) for row in rows])
        for row, rank in zip(rows, ranks):
            row[f"{feature_name}_pct"] = rank

    for row in rows:
        score = 0.0
        for feature_name, feature_weight in feature_weights.items():
            score += feature_weight * float(row[f"{feature_name}_pct"])
        row["hard_case_score"] = score
        row["hard_case_flags"] = [
            flag
            for flag, enabled in (
                ("dense_3plus", row["dense_cluster_ratio"] >= 0.25 or row["max_cluster_size"] >= 4),
                ("bass_heavy", row["low_note_ratio"] >= 0.35),
                ("soft_heavy", row["soft_note_ratio"] >= 0.45),
                ("soft_bass_overlap", row["low_soft_ratio"] >= 0.20),
                ("high_density", row["note_density"] >= 7.0),
            )
            if enabled
        ]


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


def build_manifest(split: str, selected_rows: List[Dict], rows: List[Dict], args) -> Dict:
    feature_summary = {}
    for feature_name in [
        "dense_cluster_ratio",
        "max_cluster_size",
        "low_note_ratio",
        "soft_note_ratio",
        "low_soft_ratio",
        "note_density",
    ]:
        feature_summary[feature_name] = {
            "selected_mean": float(np.mean([row[feature_name] for row in selected_rows])) if selected_rows else 0.0,
            "all_mean": float(np.mean([row[feature_name] for row in rows])) if rows else 0.0,
        }

    return {
        "split": split,
        "segment_ids": [int(row["segment_id"]) for row in selected_rows],
        "selection_strategy": "gt_structural_hard_cases_v1",
        "target_count": args.target_count,
        "selected_count": len(selected_rows),
        "max_per_piece": args.max_per_piece,
        "segment_seconds": SEGMENT_SECONDS,
        "onset_cluster_tolerance_sec": ONSET_CLUSTER_TOLERANCE_SEC,
        "low_pitch_cutoff": LOW_PITCH_CUTOFF,
        "score_weights": {
            "dense_cluster_ratio": 0.35,
            "max_cluster_size": 0.15,
            "low_note_ratio": 0.20,
            "soft_note_ratio": 0.10,
            "low_soft_ratio": 0.10,
            "note_density": 0.10,
        },
        "feature_summary": feature_summary,
        "selection": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard-case segments for mel-baseline fine-tuning")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--max-per-piece", type=int, default=DEFAULT_MAX_PER_PIECE)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    index_payload = load_index(args.split)
    rows = build_segment_features(index_payload)
    score_rows(rows)
    selected_rows = select_rows(rows, args.target_count, args.max_per_piece)
    manifest = build_manifest(args.split, selected_rows, rows, args)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote hard-case manifest: {output_path}")
    print(f"Selected {len(selected_rows)} segments from {len(rows)} total")
    for row in selected_rows[:10]:
        print(
            f"  seg={row['segment_id']:06d} score={row['hard_case_score']:.3f} "
            f"density={row['note_density']:.2f}/s low={row['low_note_ratio']:.2f} "
            f"soft={row['soft_note_ratio']:.2f} dense={row['dense_cluster_ratio']:.2f} "
            f"title={row['title'][:50]}"
        )


if __name__ == "__main__":
    main()