"""Diagnose score-duration failures for the enhanced mel transcriber.

This script focuses on cases where the decoded note matches the ground-truth
pitch and score-grid onset, but the displayed note-value/duration class does
not match. Those are the cases that depress score_f1 while event/onset F1 stays
high.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_enhanced_mel_transcriber import (
    DEFAULT_FINETUNE_MODEL_PATH,
    HOP_LENGTH,
    MIDI_OFFSET,
    MODEL_PATH,
    NOTE_VALUE_NAMES,
    SAMPLE_RATE,
    SCORE_GRID_BEATS,
    EnhancedPrecomputedMelDataset,
    _build_model_from_config,
    _cap_validation_dataset,
    _duration_to_note_value_class,
    decode_enhanced_note_events,
    enhanced_collate,
    load_segment_manifest,
)


def _event_bpm(batch: Dict, sample_idx: int) -> float:
    value = batch["bpm"][sample_idx]
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def _slot(time_sec: float, bpm: float, grid_beats: float) -> int:
    beat_duration = 60.0 / max(float(bpm), 1e-6)
    return int(round((float(time_sec) / beat_duration) / max(float(grid_beats), 1e-6)))


def _class_name(class_idx: int) -> str:
    if 0 <= int(class_idx) < len(NOTE_VALUE_NAMES):
        return NOTE_VALUE_NAMES[int(class_idx)]
    return f"class_{int(class_idx)}"


def _event_class(event: Dict, bpm: float) -> int:
    if "note_value_class" in event:
        return int(event["note_value_class"])
    duration = float(event.get("offset_time", 0.0)) - float(event.get("onset_time", 0.0))
    return _duration_to_note_value_class(max(duration, 1e-6), bpm)


def _duration_class(event: Dict, bpm: float, offset_key: str = "offset_time") -> int:
    duration = float(event.get(offset_key, event.get("offset_time", 0.0))) - float(event.get("onset_time", 0.0))
    return _duration_to_note_value_class(max(duration, 1e-6), bpm)


def _find_onset_matches(
    pred_events: Sequence[Dict],
    gt_events: Sequence[Dict],
    bpm: float,
    grid_beats: float,
    onset_slot_tolerance: int,
) -> List[Tuple[int, int]]:
    used_gt = set()
    matches: List[Tuple[int, int]] = []
    pred_slots = [_slot(event["onset_time"], bpm, grid_beats) for event in pred_events]
    gt_slots = [_slot(event["onset_time"], bpm, grid_beats) for event in gt_events]

    for pred_idx, pred in enumerate(pred_events):
        pred_pitch = int(pred["midi_note"])
        best_idx = None
        best_error = None
        for gt_idx, gt in enumerate(gt_events):
            if gt_idx in used_gt or pred_pitch != int(gt["midi_note"]):
                continue
            slot_error = abs(pred_slots[pred_idx] - gt_slots[gt_idx])
            if slot_error > onset_slot_tolerance:
                continue
            if best_error is None or slot_error < best_error:
                best_idx = gt_idx
                best_error = slot_error
        if best_idx is not None:
            used_gt.add(best_idx)
            matches.append((pred_idx, best_idx))
    return matches


def _hist(counter: Counter, limit: int = 24) -> List[Dict]:
    return [
        {"key": str(key), "count": int(count)}
        for key, count in counter.most_common(limit)
    ]


def diagnose(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    checkpoint_path = Path(args.model_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = _build_model_from_config(config).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}")

    segment_ids = load_segment_manifest(args.validation_segment_manifest, "validation")
    dataset = EnhancedPrecomputedMelDataset("validation", augment=False, segment_ids=segment_ids)
    dataset = _cap_validation_dataset(dataset, args.samples, args.sampling, "duration_diagnosis")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=enhanced_collate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    json_path = output_dir / args.json_name

    frame_key = "sounding_frame_logits" if args.decode_use_sounding_frame else "frame_logits"
    use_amp = device.type == "cuda"
    rows: List[Dict] = []
    totals = Counter()
    confusion = Counter()
    pred_head_hist = Counter()
    gt_hist = Counter()
    gt_policy_match = Counter()
    pred_policy_match = Counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_dev = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch_dev["features"])
            onset_np = torch.sigmoid(out["onset_logits"]).cpu().numpy()
            offset_np = torch.sigmoid(out["offset_logits"]).cpu().numpy()
            frame_np = torch.sigmoid(out[frame_key]).cpu().numpy()
            vel_np = out["velocity"].cpu().numpy()
            nv_np = F.softmax(out["note_value_logits"], dim=-1).cpu().numpy()

            for sample_idx in range(onset_np.shape[0]):
                bpm = _event_bpm(batch, sample_idx)
                pred_events = decode_enhanced_note_events(
                    onset_np[sample_idx],
                    offset_np[sample_idx],
                    frame_np[sample_idx],
                    vel_np[sample_idx],
                    nv_np[sample_idx],
                    onset_threshold=args.onset_threshold,
                    offset_threshold=args.offset_threshold,
                    frame_threshold=args.frame_threshold,
                )
                gt_events = batch["gt_events"][sample_idx]
                matches = _find_onset_matches(
                    pred_events,
                    gt_events,
                    bpm=bpm,
                    grid_beats=args.score_grid_beats,
                    onset_slot_tolerance=args.score_onset_slot_tolerance,
                )

                totals["samples"] += 1
                totals["predicted"] += len(pred_events)
                totals["ground_truth"] += len(gt_events)
                totals["onset_matched"] += len(matches)

                for pred_idx, gt_idx in matches:
                    pred = pred_events[pred_idx]
                    gt = gt_events[gt_idx]
                    pred_class = _event_class(pred, bpm)
                    gt_class = _event_class(gt, bpm)
                    pred_duration_class = _duration_class(pred, bpm, "offset_time")
                    gt_physical_class = _duration_class(gt, bpm, "offset_time")
                    gt_sounding_class = _duration_class(gt, bpm, "sounding_offset_time")

                    gt_hist[gt_class] += 1
                    pred_head_hist[pred_class] += 1
                    confusion[(gt_class, pred_class)] += 1
                    if gt_physical_class == gt_class:
                        gt_policy_match["physical"] += 1
                    if gt_sounding_class == gt_class:
                        gt_policy_match["sounding"] += 1
                    if pred_duration_class == gt_class:
                        pred_policy_match["decoded_duration"] += 1
                    if pred_class == gt_class:
                        totals["duration_matched"] += 1
                    else:
                        if len(rows) < args.max_rows:
                            pred_conf = float(pred.get("note_value_confidence", 0.0))
                            rows.append({
                                "sample_index": int(totals["samples"] - 1),
                                "segment_id": int(batch["segment_id"][sample_idx]),
                                "midi_path": batch["midi_path"][sample_idx],
                                "bpm": round(float(bpm), 6),
                                "midi_note": int(gt["midi_note"]),
                                "pitch_name": f"{int(gt['midi_note'])}",
                                "onset_time": round(float(gt["onset_time"]), 6),
                                "onset_slot": _slot(gt["onset_time"], bpm, args.score_grid_beats),
                                "gt_class": int(gt_class),
                                "gt_name": _class_name(gt_class),
                                "pred_class": int(pred_class),
                                "pred_name": _class_name(pred_class),
                                "pred_confidence": round(pred_conf, 6),
                                "pred_duration_class": int(pred_duration_class),
                                "pred_duration_name": _class_name(pred_duration_class),
                                "gt_physical_class": int(gt_physical_class),
                                "gt_physical_name": _class_name(gt_physical_class),
                                "gt_sounding_class": int(gt_sounding_class),
                                "gt_sounding_name": _class_name(gt_sounding_class),
                                "gt_physical_duration": round(float(gt["offset_time"] - gt["onset_time"]), 6),
                                "gt_sounding_duration": round(float(gt.get("sounding_offset_time", gt["offset_time"]) - gt["onset_time"]), 6),
                                "pred_duration": round(float(pred["offset_time"] - pred["onset_time"]), 6),
                            })

    precision = totals["duration_matched"] / max(totals["predicted"], 1)
    recall = totals["duration_matched"] / max(totals["ground_truth"], 1)
    score_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    onset_precision = totals["onset_matched"] / max(totals["predicted"], 1)
    onset_recall = totals["onset_matched"] / max(totals["ground_truth"], 1)
    onset_f1 = 2 * onset_precision * onset_recall / max(onset_precision + onset_recall, 1e-8)
    duration_accuracy = totals["duration_matched"] / max(totals["onset_matched"], 1)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "model_path": str(checkpoint_path),
        "samples": int(totals["samples"]),
        "predicted": int(totals["predicted"]),
        "ground_truth": int(totals["ground_truth"]),
        "onset_matched": int(totals["onset_matched"]),
        "duration_matched": int(totals["duration_matched"]),
        "score_precision": precision,
        "score_recall": recall,
        "score_f1": score_f1,
        "score_onset_f1": onset_f1,
        "duration_accuracy": duration_accuracy,
        "gt_policy_match_rates": {
            key: gt_policy_match[key] / max(totals["onset_matched"], 1)
            for key in ("physical", "sounding")
        },
        "pred_policy_match_rates": {
            key: pred_policy_match[key] / max(totals["onset_matched"], 1)
            for key in ("decoded_duration",)
        },
        "gt_class_hist": [
            {"class": int(key), "name": _class_name(key), "count": int(count)}
            for key, count in gt_hist.most_common()
        ],
        "pred_class_hist": [
            {"class": int(key), "name": _class_name(key), "count": int(count)}
            for key, count in pred_head_hist.most_common()
        ],
        "confusion_top": [
            {
                "gt_class": int(key[0]),
                "gt_name": _class_name(key[0]),
                "pred_class": int(key[1]),
                "pred_name": _class_name(key[1]),
                "count": int(count),
            }
            for key, count in confusion.most_common(30)
        ],
        "csv_path": str(csv_path),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"score_f1={score_f1:.4f} onset_f1={onset_f1:.4f} "
        f"duration_acc={duration_accuracy:.4f} samples={totals['samples']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return summary


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).parent / "enhanced_mel_transcription_pedal_score_repair_latest.pt"
    if not default_model.exists():
        default_model = Path(__file__).parent / "enhanced_mel_transcription_pedal_score_latest.pt"
    parser = argparse.ArgumentParser(description="Diagnose score-duration mismatches")
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--validation-segment-manifest", type=str, default="mel_hard_case_manifest_validation_pedal_onset_v2.json")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--sampling", choices=["spread", "leading"], default="spread")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--score-grid-beats", type=float, default=SCORE_GRID_BEATS)
    parser.add_argument("--score-onset-slot-tolerance", type=int, default=0)
    parser.add_argument("--decode-use-sounding-frame", action="store_true", default=True)
    parser.add_argument("--decode-use-physical-frame", action="store_false", dest="decode_use_sounding_frame")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="score_duration_diagnostics")
    parser.add_argument("--json-name", type=str, default="score_duration_summary.json")
    parser.add_argument("--csv-name", type=str, default="score_duration_mismatches.csv")
    return parser.parse_args()


def main() -> None:
    diagnose(parse_args())


if __name__ == "__main__":
    main()
