"""Train/evaluate a calibrated weak-candidate decoder for inner voices.

This is an offline experiment. It learns P(real note | weak candidate context)
from MAESTRO validation/train segments, then chooses an acceptance threshold from
a held-out precision/recall curve.

Example:

    python train_lattice_candidate_calibrator.py \
      --segment-manifest mel_hard_case_manifest_validation_pedal_onset_v2.json \
      --max-segments 256 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from train_enhanced_mel_transcriber import (
    EnhancedPrecomputedMelDataset,
    HOP_LENGTH,
    MIDI_OFFSET,
    MODEL_PATH,
    PIANO_KEYS,
    SAMPLE_RATE,
    _build_model_from_config,
    _move_batch_to_device,
    decode_enhanced_note_events,
    enhanced_collate,
    load_segment_manifest,
)


FEATURE_NAMES = [
    "onset_peak",
    "frame_peak",
    "velocity_peak",
    "velocity_int",
    "local_onset_delta",
    "prev_onset_level",
    "anchor_dt_abs",
    "anchor_cluster_size",
    "pitch_distance_to_anchor",
    "same_pitch_recent_primary",
    "harmonic_to_anchor",
    "active_frame_before",
    "is_middle_register",
    "normalized_pitch",
]

HARMONIC_INTERVAL_CLASSES = {0, 7, 12, 19, 24, 28, 31, 36}


def _event_f1(pred: Sequence[Dict], gt: Sequence[Dict], onset_tol: float = 0.05) -> Dict[str, float]:
    used_gt = set()
    matched = 0
    for event in pred:
        best_idx = None
        best_error = None
        for idx, ref in enumerate(gt):
            if idx in used_gt or int(event["midi_note"]) != int(ref["midi_note"]):
                continue
            error = abs(float(event["onset_time"]) - float(ref["onset_time"]))
            if error <= onset_tol and (best_error is None or error < best_error):
                best_idx = idx
                best_error = error
        if best_idx is not None:
            used_gt.add(best_idx)
            matched += 1
    precision = matched / max(len(pred), 1)
    recall = matched / max(len(gt), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "matched": matched,
        "predicted": len(pred),
        "ground_truth": len(gt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _cluster_primary_events(events: Sequence[Dict], tolerance_sec: float) -> List[Dict]:
    clusters: List[Dict] = []
    for event in sorted(events, key=lambda item: float(item["onset_time"])):
        onset = float(event["onset_time"])
        midi = int(event["midi_note"])
        if clusters and abs(onset - float(clusters[-1]["time"])) <= tolerance_sec:
            cluster = clusters[-1]
            n = int(cluster["count"])
            cluster["time"] = ((float(cluster["time"]) * n) + onset) / float(n + 1)
            cluster["count"] = n + 1
            cluster["pitches"].append(midi)
        else:
            clusters.append({"time": onset, "count": 1, "pitches": [midi]})
    return clusters


def _nearest_anchor(clusters: Sequence[Dict], time_sec: float) -> Tuple[Optional[Dict], float]:
    best = None
    best_error = float("inf")
    for cluster in clusters:
        error = abs(float(cluster["time"]) - time_sec)
        if error < best_error:
            best = cluster
            best_error = error
    return best, best_error


def _matches_gt(midi_note: int, time_sec: float, gt_events: Sequence[Dict], onset_tol: float) -> bool:
    for event in gt_events:
        if int(event["midi_note"]) != int(midi_note):
            continue
        if abs(float(event["onset_time"]) - float(time_sec)) <= onset_tol:
            return True
    return False


def _primary_match_exists(primary_events: Sequence[Dict], midi_note: int, time_sec: float, tolerance_sec: float) -> bool:
    for event in primary_events:
        if int(event["midi_note"]) != int(midi_note):
            continue
        if abs(float(event["onset_time"]) - float(time_sec)) <= tolerance_sec:
            return True
    return False


def _extract_probs(
    model: torch.nn.Module,
    features: torch.Tensor,
    config: Dict,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    sr = int(config.get("sample_rate", SAMPLE_RATE))
    hop = int(config.get("hop_length", HOP_LENGTH))
    n_frames = int(features.size(0))
    n_keys = int(config.get("n_keys", PIANO_KEYS))
    n_note_value_classes = int(config.get("n_note_value_classes", 12))
    chunk_frames = int(10.0 * sr / hop)
    overlap = max(1, chunk_frames // 4)
    step = max(1, chunk_frames - overlap)

    totals = {
        "onset": np.zeros((n_frames, n_keys), dtype=np.float32),
        "offset": np.zeros((n_frames, n_keys), dtype=np.float32),
        "frame": np.zeros((n_frames, n_keys), dtype=np.float32),
        "velocity": np.zeros((n_frames, n_keys), dtype=np.float32),
        "note_value": np.zeros((n_frames, n_keys, n_note_value_classes), dtype=np.float32),
    }
    counts = np.zeros(n_frames, dtype=np.float32)
    features = features.to(device).unsqueeze(0)
    for start in range(0, n_frames, step):
        end = min(start + chunk_frames, n_frames)
        out = model(features[:, start:end, :])
        onset = torch.sigmoid(out["onset_logits"][0]).float().cpu().numpy()
        offset = torch.sigmoid(out["offset_logits"][0]).float().cpu().numpy()
        frame = torch.sigmoid(out["frame_logits"][0]).float().cpu().numpy()
        velocity = out["velocity"][0].float().cpu().numpy()
        note_value = F.softmax(out["note_value_logits"][0].float(), dim=-1).cpu().numpy()
        if note_value.shape[-1] < n_note_value_classes:
            pad = np.zeros(
                (*note_value.shape[:-1], n_note_value_classes - note_value.shape[-1]),
                dtype=np.float32,
            )
            note_value = np.concatenate([note_value, pad], axis=-1)
        elif note_value.shape[-1] > n_note_value_classes:
            note_value = note_value[..., :n_note_value_classes]
        actual_len = end - start
        totals["onset"][start:end] += onset[:actual_len]
        totals["offset"][start:end] += offset[:actual_len]
        totals["frame"][start:end] += frame[:actual_len]
        totals["velocity"][start:end] += velocity[:actual_len]
        totals["note_value"][start:end] += note_value[:actual_len]
        counts[start:end] += 1.0
    counts = np.maximum(counts, 1.0)
    return {
        "onset": totals["onset"] / counts[:, None],
        "offset": totals["offset"] / counts[:, None],
        "frame": totals["frame"] / counts[:, None],
        "velocity": totals["velocity"] / counts[:, None],
        "note_value": totals["note_value"] / counts[:, None, None],
    }


def _peak_frames(probs: np.ndarray, threshold: float) -> List[int]:
    peaks: List[int] = []
    for idx in range(probs.shape[0]):
        left = probs[idx - 1] if idx > 0 else -np.inf
        right = probs[idx + 1] if idx + 1 < probs.shape[0] else -np.inf
        if probs[idx] >= threshold and probs[idx] >= left and probs[idx] >= right:
            peaks.append(int(idx))
    return peaks


def _frame_drop_offset(
    probs: Dict[str, np.ndarray],
    key: int,
    onset_frame: int,
    frame_threshold: float,
    min_frames: int,
) -> int:
    n_frames = probs["frame"].shape[0]
    min_offset = min(n_frames - 1, onset_frame + min_frames)
    for frame_idx in range(min_offset, n_frames):
        if float(probs["frame"][frame_idx, key]) < frame_threshold:
            return max(frame_idx, min_offset)
    return min(n_frames, onset_frame + int(round(2.0 / (HOP_LENGTH / SAMPLE_RATE))))


def _candidate_features(
    probs: Dict[str, np.ndarray],
    key: int,
    frame_idx: int,
    primary_events: Sequence[Dict],
    clusters: Sequence[Dict],
    sr: int,
    hop: int,
    lookback_frames: int,
) -> Tuple[List[float], Dict]:
    midi_note = key + MIDI_OFFSET
    time_sec = frame_idx * hop / float(sr)
    onset_peak = float(probs["onset"][frame_idx, key])
    win_start = max(0, frame_idx - 2)
    win_end = min(probs["onset"].shape[0], frame_idx + 5)
    frame_peak = float(np.max(probs["frame"][win_start:win_end, key]))
    velocity_peak = float(np.max(probs["velocity"][win_start:win_end, key]))
    velocity_int = int(np.clip(round(velocity_peak * 127), 1, 127))
    prev_start = max(0, frame_idx - lookback_frames)
    prev = probs["onset"][prev_start:frame_idx, key]
    prev_level = float(np.median(prev)) if prev.size else 0.0
    local_delta = onset_peak - prev_level
    anchor, anchor_dt = _nearest_anchor(clusters, time_sec)
    anchor_size = int(anchor["count"]) if anchor else 0
    anchor_pitches = [int(pitch) for pitch in anchor["pitches"]] if anchor else []
    if anchor_pitches:
        pitch_distance = min(abs(midi_note - pitch) for pitch in anchor_pitches)
        harmonic = 1.0 if any(abs(midi_note - pitch) % 12 in HARMONIC_INTERVAL_CLASSES for pitch in anchor_pitches) else 0.0
    else:
        pitch_distance = 88
        harmonic = 0.0
    same_pitch_recent = 0.0
    for event in primary_events:
        if int(event["midi_note"]) != midi_note:
            continue
        if 0.0 < time_sec - float(event["onset_time"]) <= 0.35:
            same_pitch_recent = 1.0
            break
    active_before = float(np.max(probs["frame"][max(0, frame_idx - lookback_frames):frame_idx, key])) if frame_idx > 0 else 0.0
    features = [
        onset_peak,
        frame_peak,
        velocity_peak,
        float(velocity_int),
        local_delta,
        prev_level,
        float(anchor_dt if math.isfinite(anchor_dt) else 9.9),
        float(anchor_size),
        float(pitch_distance),
        same_pitch_recent,
        harmonic,
        active_before,
        1.0 if 52 <= midi_note <= 76 else 0.0,
        (midi_note - MIDI_OFFSET) / 87.0,
    ]
    meta = {
        "midi_note": midi_note,
        "time_sec": time_sec,
        "anchor_time": float(anchor["time"]) if anchor else time_sec,
        "anchor_dt": float(anchor_dt if math.isfinite(anchor_dt) else 9.9),
        "onset_peak": onset_peak,
        "frame_peak": frame_peak,
        "velocity_int": velocity_int,
    }
    return features, meta


def collect_segment_candidates(
    probs: Dict[str, np.ndarray],
    gt_events: Sequence[Dict],
    primary_events: Sequence[Dict],
    args: argparse.Namespace,
    sr: int,
    hop: int,
) -> List[Dict]:
    clusters = _cluster_primary_events(primary_events, args.cluster_tolerance_sec)
    lookback_frames = max(1, int(round(args.lookback_sec * sr / hop)))
    candidates: List[Dict] = []
    for key in range(probs["onset"].shape[1]):
        for frame_idx in _peak_frames(probs["onset"][:, key], args.candidate_onset_threshold):
            onset_peak = float(probs["onset"][frame_idx, key])
            if onset_peak >= args.primary_onset_threshold:
                continue
            midi_note = key + MIDI_OFFSET
            time_sec = frame_idx * hop / float(sr)
            if _primary_match_exists(primary_events, midi_note, time_sec, args.duplicate_tolerance_sec):
                continue
            features, meta = _candidate_features(
                probs, key, frame_idx, primary_events, clusters, sr, hop, lookback_frames
            )
            if meta["frame_peak"] < args.candidate_frame_threshold:
                continue
            if meta["velocity_int"] < args.candidate_min_velocity:
                continue
            label = 1 if _matches_gt(midi_note, time_sec, gt_events, args.match_tolerance_sec) else 0
            candidates.append({"features": features, "label": label, "meta": meta})
    return candidates


def add_lattice_events(
    probs: Dict[str, np.ndarray],
    primary_events: Sequence[Dict],
    candidates: Sequence[Dict],
    probabilities: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
    sr: int,
    hop: int,
) -> List[Dict]:
    events = [dict(event) for event in primary_events]
    min_frames = max(1, int(round(args.min_note_duration / (hop / float(sr)))))
    accepted: List[Tuple[float, Dict]] = []
    per_anchor_counts = Counter()
    for candidate, prob in sorted(zip(candidates, probabilities), key=lambda item: float(item[1]), reverse=True):
        if float(prob) < threshold:
            continue
        meta = candidate["meta"]
        if float(meta["anchor_dt"]) > args.max_anchor_distance_sec:
            continue
        anchor_key = round(float(meta["anchor_time"]), 3)
        if per_anchor_counts[anchor_key] >= args.max_additions_per_anchor:
            continue
        midi_note = int(meta["midi_note"])
        onset_time = float(meta["anchor_time"]) if args.snap_to_anchor else float(meta["time_sec"])
        if _primary_match_exists(events, midi_note, onset_time, args.duplicate_tolerance_sec):
            continue
        key = midi_note - MIDI_OFFSET
        onset_frame = int(round(float(meta["time_sec"]) * sr / hop))
        offset_frame = _frame_drop_offset(probs, key, onset_frame, args.frame_threshold, min_frames)
        offset_time = max(onset_time + args.min_note_duration, offset_frame * hop / float(sr))
        event = {
            "onset_time": onset_time,
            "offset_time": offset_time,
            "midi_note": midi_note,
            "velocity": int(meta["velocity_int"]),
            "onset_prob": float(meta["onset_peak"]),
            "offset_prob": float(probs["offset"][min(offset_frame, probs["offset"].shape[0] - 1), key]),
            "decode_source": "lattice_calibrated",
            "lattice_probability": float(prob),
        }
        accepted.append((float(prob), event))
        per_anchor_counts[anchor_key] += 1
    events.extend(event for _, event in accepted)
    events.sort(key=lambda item: (float(item["onset_time"]), int(item["midi_note"])))
    return events


def _load_ids(args: argparse.Namespace) -> List[int]:
    if args.segment_manifest:
        manifest_path = Path(args.segment_manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path(__file__).parent / manifest_path
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            ids = [int(item) for item in payload]
        else:
            selection = payload.get("selection") or []
            if selection:
                ids = [int(item["segment_id"]) for item in selection if "segment_id" in item]
            else:
                raw_ids = payload.get("segment_ids")
                if raw_ids is None:
                    ids = load_segment_manifest(str(manifest_path), args.split)
                    if ids is None:
                        raise ValueError(f"No ids found in {args.segment_manifest}")
                else:
                    ids = [int(item) for item in raw_ids]
    else:
        dataset = EnhancedPrecomputedMelDataset(args.split, augment=False)
        ids = dataset.segment_ids
    if args.max_segments:
        ids = ids[:args.max_segments]
    return [int(item) for item in ids]


@torch.no_grad()
def build_segment_records(args: argparse.Namespace) -> Tuple[List[Dict], Dict, int, int]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    root = Path(__file__).parent
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    sr = int(config.get("sample_rate", SAMPLE_RATE))
    hop = int(config.get("hop_length", HOP_LENGTH))
    model = _build_model_from_config(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    segment_ids = _load_ids(args)
    dataset = EnhancedPrecomputedMelDataset(args.split, augment=False, segment_ids=segment_ids)
    if dataset.segment_ids != segment_ids:
        file_by_id = {
            int(segment_id): file_path
            for segment_id, file_path in zip(dataset.segment_ids, dataset.files)
        }
        dataset.segment_ids = segment_ids
        dataset.files = [file_by_id[int(segment_id)] for segment_id in segment_ids]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=enhanced_collate,
    )
    records: List[Dict] = []
    for idx, batch in enumerate(loader):
        segment_id = int(batch["segment_id"][0])
        probs = _extract_probs(model, batch["features"][0], config, device)
        primary = decode_enhanced_note_events(
            probs["onset"],
            probs["offset"],
            probs["frame"],
            probs["velocity"],
            probs["note_value"],
            onset_threshold=args.primary_onset_threshold,
            offset_threshold=args.offset_threshold,
            frame_threshold=args.frame_threshold,
            min_velocity=args.min_velocity,
            duplicate_window_sec=args.duplicate_window_sec,
            merge_gap_sec=args.merge_gap_sec,
            sr=sr,
            hop=hop,
        )
        gt_events = batch["gt_events"][0]
        candidates = collect_segment_candidates(probs, gt_events, primary, args, sr, hop)
        records.append({
            "segment_id": segment_id,
            "probs": probs,
            "gt_events": gt_events,
            "primary_events": primary,
            "candidates": candidates,
        })
        if args.progress_every and (idx + 1) % args.progress_every == 0:
            print(f"processed {idx + 1}/{len(dataset)} segments")
    return records, config, sr, hop


def _totals_to_metrics(totals: Dict[str, int]) -> Dict[str, float]:
    precision = totals["matched"] / max(totals["predicted"], 1)
    recall = totals["matched"] / max(totals["ground_truth"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {**totals, "precision": precision, "recall": recall, "f1": f1}


def _sum_event_metrics(records: Sequence[Dict], event_key: str) -> Dict[str, float]:
    totals = defaultdict(int)
    for record in records:
        metrics = _event_f1(record[event_key], record["gt_events"])
        totals["matched"] += int(metrics["matched"])
        totals["predicted"] += int(metrics["predicted"])
        totals["ground_truth"] += int(metrics["ground_truth"])
    return _totals_to_metrics(totals)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best = None
    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if precision < target_precision:
            continue
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        item = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or item["recall"] > best["recall"]:
            best = item
    if best is not None:
        return best
    f1_scores = 2.0 * precisions[:-1] * recalls[:-1] / np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
    idx = int(np.argmax(f1_scores)) if f1_scores.size else 0
    return {
        "threshold": float(thresholds[idx]) if thresholds.size else 1.0,
        "precision": float(precisions[idx]) if precisions.size else 0.0,
        "recall": float(recalls[idx]) if recalls.size else 0.0,
        "f1": float(f1_scores[idx]) if f1_scores.size else 0.0,
        "fallback": "max_f1_no_threshold_met_target_precision",
    }


def run(args: argparse.Namespace) -> Dict:
    records, config, sr, hop = build_segment_records(args)
    if len(records) < 2:
        raise ValueError("Need at least two segments for train/eval split")
    split_idx = max(1, min(len(records) - 1, int(round(len(records) * args.train_fraction))))
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    x_train = np.asarray([c["features"] for r in train_records for c in r["candidates"]], dtype=np.float32)
    y_train = np.asarray([c["label"] for r in train_records for c in r["candidates"]], dtype=np.int64)
    x_eval = np.asarray([c["features"] for r in eval_records for c in r["candidates"]], dtype=np.float32)
    y_eval = np.asarray([c["label"] for r in eval_records for c in r["candidates"]], dtype=np.int64)
    if x_train.size == 0 or x_eval.size == 0:
        raise RuntimeError("No candidate examples collected; lower candidate thresholds")
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("Training candidates have only one class; use more segments")

    classifier = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])
    classifier.fit(x_train, y_train)
    eval_prob = classifier.predict_proba(x_eval)[:, 1]
    threshold_info = choose_threshold(y_eval, eval_prob, args.target_precision)
    threshold = float(threshold_info["threshold"])

    offset = 0
    for record in eval_records:
        n = len(record["candidates"])
        probs = eval_prob[offset:offset + n]
        offset += n
        record["lattice_events"] = add_lattice_events(
            record["probs"],
            record["primary_events"],
            record["candidates"],
            probs,
            threshold,
            args,
            sr,
            hop,
        )

    primary_metrics = _sum_event_metrics(eval_records, "primary_events")
    lattice_metrics = _sum_event_metrics(eval_records, "lattice_events")
    added_counts = [
        max(0, len(record["lattice_events"]) - len(record["primary_events"]))
        for record in eval_records
    ]
    try:
        roc_auc = float(roc_auc_score(y_eval, eval_prob))
    except ValueError:
        roc_auc = 0.0
    summary = {
        "model_path": str(args.model_path),
        "split": args.split,
        "segment_manifest": args.segment_manifest,
        "segments_total": len(records),
        "segments_train": len(train_records),
        "segments_eval": len(eval_records),
        "candidate_counts": {
            "train": int(len(y_train)),
            "eval": int(len(y_eval)),
            "train_positive": int(y_train.sum()),
            "eval_positive": int(y_eval.sum()),
        },
        "classifier": {
            "feature_names": FEATURE_NAMES,
            "average_precision": float(average_precision_score(y_eval, eval_prob)) if len(set(y_eval.tolist())) > 1 else 0.0,
            "roc_auc": roc_auc,
            "threshold_selection": threshold_info,
        },
        "event_metrics": {
            "primary": primary_metrics,
            "lattice": lattice_metrics,
            "delta_f1": lattice_metrics["f1"] - primary_metrics["f1"],
            "delta_recall": lattice_metrics["recall"] - primary_metrics["recall"],
            "delta_precision": lattice_metrics["precision"] - primary_metrics["precision"],
        },
        "lattice_additions": {
            "total": int(sum(added_counts)),
            "mean_per_segment": float(np.mean(added_counts)) if added_counts else 0.0,
            "max_per_segment": int(max(added_counts)) if added_counts else 0,
        },
    }

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lattice_calibrator_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "lattice_candidate_calibrator.pkl").open("wb") as handle:
        pickle.dump({"model": classifier, "threshold": threshold, "feature_names": FEATURE_NAMES, "args": vars(args)}, handle)

    segment_rows = []
    for record in eval_records:
        primary = _event_f1(record["primary_events"], record["gt_events"])
        lattice = _event_f1(record["lattice_events"], record["gt_events"])
        segment_rows.append({
            "segment_id": record["segment_id"],
            "primary_f1": primary["f1"],
            "lattice_f1": lattice["f1"],
            "primary_recall": primary["recall"],
            "lattice_recall": lattice["recall"],
            "primary_precision": primary["precision"],
            "lattice_precision": lattice["precision"],
            "added": len(record["lattice_events"]) - len(record["primary_events"]),
            "candidates": len(record["candidates"]),
            "positive_candidates": int(sum(c["label"] for c in record["candidates"])),
        })
    (out_dir / "lattice_eval_segments.json").write_text(json.dumps(segment_rows, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--segment-manifest", default="mel_hard_case_manifest_validation_pedal_onset_v2.json")
    parser.add_argument("--max-segments", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--output-dir", default="lattice_candidate_calibrator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--primary-onset-threshold", type=float, default=0.75)
    parser.add_argument("--candidate-onset-threshold", type=float, default=0.25)
    parser.add_argument("--candidate-frame-threshold", type=float, default=0.35)
    parser.add_argument("--candidate-min-velocity", type=int, default=8)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--min-velocity", type=int, default=8)
    parser.add_argument("--min-note-duration", type=float, default=0.04)
    parser.add_argument("--duplicate-window-sec", type=float, default=0.04)
    parser.add_argument("--merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--cluster-tolerance-sec", type=float, default=0.04)
    parser.add_argument("--duplicate-tolerance-sec", type=float, default=0.04)
    parser.add_argument("--match-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--lookback-sec", type=float, default=0.08)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--max-anchor-distance-sec", type=float, default=0.06)
    parser.add_argument("--max-additions-per-anchor", type=int, default=3)
    parser.add_argument("--snap-to-anchor", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
