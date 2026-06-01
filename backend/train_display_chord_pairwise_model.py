from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from display_chord_pairwise_model import (
    DEFAULT_MODEL_FILENAME, PAIR_FEATURE_NAMES, PITCH_FEATURE_NAMES,
    chord_pitch_tuple, chord_time_seconds, extract_pair_features,
    extract_pitch_vote_features, pitch_set_f1)
from live_rhythm import DISPLAY_CHORD_RECONCILE_TOLERANCE_SEC
from test_experiment import (ONSET_CLUSTER_TOLERANCE_SEC, TARGET_SR,
                             cluster_note_onsets, load_audio_excerpt,
                             load_benchmark_manifest, load_midi_notes,
                             run_live_excerpt, slice_gt_notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pairwise display chord merger with canonical pitch voting.")
    parser.add_argument(
        "--benchmark-manifest",
        default=str(Path(__file__).resolve().with_name("live_benchmark_replay_auto_v2.json")),
        help="Path to the replay benchmark manifest.",
    )
    parser.add_argument(
        "--output-model",
        default=str(Path(__file__).resolve().with_name(DEFAULT_MODEL_FILENAME)),
        help="Where to write the exported pairwise model JSON.",
    )
    parser.add_argument(
        "--clip-ids",
        nargs="*",
        default=None,
        help="Optional subset of manifest clip IDs to train on.",
    )
    parser.add_argument(
        "--exclude-clip-ids",
        nargs="*",
        default=(),
        help="Optional clip IDs to exclude from training.",
    )
    parser.add_argument("--chunk-seconds", type=float, default=1.2)
    parser.add_argument("--noise-profile", choices=["open", "balanced", "clean"], default="balanced")
    parser.add_argument("--pair-c", type=float, default=1.0)
    parser.add_argument("--pitch-c", type=float, default=1.0)
    return parser.parse_args()


def _group_chords_for_display(chords: Sequence[Dict]) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    for chord in sorted(chords or [], key=chord_time_seconds):
        chord_time = chord_time_seconds(chord)
        if (
            not groups
            or (chord_time - chord_time_seconds(groups[-1][0])) > DISPLAY_CHORD_RECONCILE_TOLERANCE_SEC
        ):
            groups.append([dict(chord)])
            continue
        groups[-1].append(dict(chord))
    return groups


def _cluster_anchor_time(cluster: Sequence[Dict]) -> float:
    if not cluster:
        return 0.0
    return float(np.mean([float(note.get("onset_time", 0.0) or 0.0) for note in cluster], dtype=np.float64))


def _cluster_pitch_tuple(cluster: Sequence[Dict]) -> tuple[int, ...]:
    pitches = []
    for note in cluster:
        try:
            pitches.append(int(note.get("midi_note", note.get("pitch", 0)) or 0))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(pitches))


def _assign_chord_to_gt_cluster(chord: Dict, gt_clusters: Sequence[Sequence[Dict]]) -> int | None:
    chord_pitches = chord_pitch_tuple(chord)
    if not chord_pitches:
        return None

    chord_time = chord_time_seconds(chord)
    best_idx = None
    best_key = None
    for cluster_index, cluster in enumerate(gt_clusters):
        onset_error = abs(chord_time - _cluster_anchor_time(cluster))
        if onset_error > ONSET_CLUSTER_TOLERANCE_SEC:
            continue
        cluster_pitches = _cluster_pitch_tuple(cluster)
        overlap_f1 = pitch_set_f1(chord_pitches, cluster_pitches)
        if overlap_f1 <= 0.0:
            continue
        key = (
            overlap_f1,
            -onset_error,
            -abs(len(chord_pitches) - len(cluster_pitches)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = cluster_index
    return best_idx


async def build_dataset(args: argparse.Namespace) -> tuple[List[Dict], List[Dict], Dict[str, object]]:
    selected = load_benchmark_manifest(args.benchmark_manifest, args.clip_ids)
    excluded = {str(clip_id) for clip_id in (args.exclude_clip_ids or [])}
    gt_note_cache: Dict[str, List[Dict]] = {}
    pair_rows: List[Dict] = []
    pitch_rows: List[Dict] = []
    summary = {
        "clips": [],
        "groups": 0,
        "pair_examples": 0,
        "pitch_examples": 0,
    }

    for clip_id, clip in selected.items():
        if clip_id in excluded:
            continue

        audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
        if clip["midi_path"] not in gt_note_cache:
            gt_note_cache[clip["midi_path"]] = load_midi_notes(clip["midi_path"])
        gt_notes = slice_gt_notes(gt_note_cache[clip["midi_path"]], clip["start_sec"], clip["end_sec"])
        gt_clusters = cluster_note_onsets(gt_notes, onset_tolerance_sec=ONSET_CLUSTER_TOLERANCE_SEC)

        run = await run_live_excerpt(
            audio,
            adaptive_onset_threshold=False,
            chunk_seconds=float(args.chunk_seconds),
            noise_profile=str(args.noise_profile),
            capture_display_inputs=True,
        )
        chord_groups = [
            group
            for group in _group_chords_for_display(run.get("final_display_input_chords") or [])
            if len(group) <= 7
        ]

        clip_pair_examples = 0
        clip_pitch_examples = 0
        for group_index, chord_group in enumerate(chord_groups):
            assignments = [_assign_chord_to_gt_cluster(chord, gt_clusters) for chord in chord_group]

            for left_index in range(len(chord_group)):
                for right_index in range(left_index + 1, len(chord_group)):
                    label = 1 if (
                        assignments[left_index] is not None
                        and assignments[left_index] == assignments[right_index]
                    ) else 0
                    pair_rows.append(
                        {
                            "clip_id": clip_id,
                            "group_index": group_index,
                            "label": label,
                            **extract_pair_features(chord_group[left_index], chord_group[right_index]),
                        }
                    )
                    clip_pair_examples += 1

            members_by_cluster: Dict[int, List[Dict]] = defaultdict(list)
            noise_components: List[List[Dict]] = []
            for chord, assignment in zip(chord_group, assignments):
                if assignment is None:
                    noise_components.append([dict(chord)])
                    continue
                members_by_cluster[int(assignment)].append(dict(chord))

            for cluster_index, component in members_by_cluster.items():
                gt_pitch_set = set(_cluster_pitch_tuple(gt_clusters[cluster_index]))
                candidate_pitches = sorted({pitch for chord in component for pitch in chord_pitch_tuple(chord)})
                for pitch in candidate_pitches:
                    pitch_rows.append(
                        {
                            "clip_id": clip_id,
                            "group_index": group_index,
                            "label": 1 if int(pitch) in gt_pitch_set else 0,
                            **extract_pitch_vote_features(component, int(pitch)),
                        }
                    )
                    clip_pitch_examples += 1

            for component in noise_components:
                for pitch in chord_pitch_tuple(component[0]):
                    pitch_rows.append(
                        {
                            "clip_id": clip_id,
                            "group_index": group_index,
                            "label": 0,
                            **extract_pitch_vote_features(component, int(pitch)),
                        }
                    )
                    clip_pitch_examples += 1

        summary["clips"].append(
            {
                "clip_id": clip_id,
                "groups": len(chord_groups),
                "pair_examples": clip_pair_examples,
                "pitch_examples": clip_pitch_examples,
            }
        )
        summary["groups"] += len(chord_groups)
        summary["pair_examples"] += clip_pair_examples
        summary["pitch_examples"] += clip_pitch_examples

    return pair_rows, pitch_rows, summary


def _fit_binary_model(rows: Sequence[Dict], feature_names: Sequence[str], c_value: float) -> tuple[StandardScaler, LogisticRegression, np.ndarray, Dict[str, float]]:
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    if x.size == 0 or len(set(y.tolist())) < 2:
        raise RuntimeError("Training data must contain at least one positive and one negative example.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=float(c_value),
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(x_scaled, y)
    probabilities = model.predict_proba(x_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int32)

    summary = {
        "examples": float(len(rows)),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }
    return scaler, model, probabilities, summary


def _best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = None
    for threshold in np.linspace(0.2, 0.8, 25):
        predictions = (probabilities >= threshold).astype(np.int32)
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        key = (score, -abs(float(threshold) - 0.5))
        if best_score is None or key > best_score:
            best_score = key
            best_threshold = float(threshold)
    return best_threshold


def export_model(
    output_path: str,
    pair_scaler: StandardScaler,
    pair_model: LogisticRegression,
    pair_summary: Dict[str, float],
    pitch_scaler: StandardScaler,
    pitch_model: LogisticRegression,
    pitch_threshold: float,
    pitch_summary: Dict[str, float],
    dataset_summary: Dict[str, object],
    args: argparse.Namespace,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_type": "pairwise_same_event_pitch_vote",
        "pair_model": {
            "feature_names": list(PAIR_FEATURE_NAMES),
            "intercept": float(pair_model.intercept_[0]),
            "coefficients": [float(value) for value in pair_model.coef_[0].tolist()],
            "standardize": {
                "mean": [float(value) for value in pair_scaler.mean_.tolist()],
                "scale": [float(value) if abs(float(value)) > 1e-9 else 1.0 for value in pair_scaler.scale_.tolist()],
            },
            "threshold": 0.5,
        },
        "pitch_model": {
            "feature_names": list(PITCH_FEATURE_NAMES),
            "intercept": float(pitch_model.intercept_[0]),
            "coefficients": [float(value) for value in pitch_model.coef_[0].tolist()],
            "standardize": {
                "mean": [float(value) for value in pitch_scaler.mean_.tolist()],
                "scale": [float(value) if abs(float(value)) > 1e-9 else 1.0 for value in pitch_scaler.scale_.tolist()],
            },
            "threshold": float(pitch_threshold),
        },
        "training_summary": {
            "benchmark_manifest": str(Path(args.benchmark_manifest).expanduser().resolve()),
            "chunk_seconds": float(args.chunk_seconds),
            "noise_profile": str(args.noise_profile),
            "pair_c": float(args.pair_c),
            "pitch_c": float(args.pitch_c),
            "clip_count": len(dataset_summary.get("clips") or []),
            "dataset_groups": int(dataset_summary.get("groups", 0) or 0),
            "pair_examples": int(dataset_summary.get("pair_examples", 0) or 0),
            "pitch_examples": int(dataset_summary.get("pitch_examples", 0) or 0),
            "pair_positive_rate": float(pair_summary["positive_rate"]),
            "pair_train_accuracy": float(pair_summary["accuracy"]),
            "pair_train_roc_auc": float(pair_summary["roc_auc"]),
            "pitch_positive_rate": float(pitch_summary["positive_rate"]),
            "pitch_train_accuracy": float(pitch_summary["accuracy"]),
            "pitch_train_roc_auc": float(pitch_summary["roc_auc"]),
            "pitch_threshold": float(pitch_threshold),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


async def main_async() -> int:
    args = parse_args()
    pair_rows, pitch_rows, dataset_summary = await build_dataset(args)
    if not pair_rows or not pitch_rows:
        print("No usable training rows were produced.")
        return 1

    pair_scaler, pair_model, _, pair_summary = _fit_binary_model(pair_rows, PAIR_FEATURE_NAMES, args.pair_c)
    pitch_scaler, pitch_model, pitch_probabilities, pitch_summary = _fit_binary_model(pitch_rows, PITCH_FEATURE_NAMES, args.pitch_c)
    pitch_labels = np.asarray([int(row["label"]) for row in pitch_rows], dtype=np.int32)
    pitch_threshold = _best_threshold(pitch_probabilities, pitch_labels)

    path = export_model(
        args.output_model,
        pair_scaler,
        pair_model,
        pair_summary,
        pitch_scaler,
        pitch_model,
        pitch_threshold,
        pitch_summary,
        dataset_summary,
        args,
    )

    print(f"Wrote model: {path}")
    print(
        "Dataset: "
        f"clips={len(dataset_summary.get('clips') or [])} "
        f"groups={int(dataset_summary.get('groups', 0) or 0)} "
        f"pair_examples={int(dataset_summary.get('pair_examples', 0) or 0)} "
        f"pitch_examples={int(dataset_summary.get('pitch_examples', 0) or 0)}"
    )
    print(
        "Pair model: "
        f"auc={pair_summary['roc_auc']:.4f} "
        f"accuracy={pair_summary['accuracy']:.4f} "
        f"positive_rate={pair_summary['positive_rate']:.4f}"
    )
    print(
        "Pitch model: "
        f"auc={pitch_summary['roc_auc']:.4f} "
        f"accuracy={pitch_summary['accuracy']:.4f} "
        f"positive_rate={pitch_summary['positive_rate']:.4f} "
        f"threshold={pitch_threshold:.3f}"
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())