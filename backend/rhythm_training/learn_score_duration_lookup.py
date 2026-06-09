"""Learn and evaluate a tiny lookup-table score-duration chooser.

This is an offline experiment for a low-latency score post-processor. It
collects onset-matched decoded notes, fits lookup tables that map cheap local
features to the most common ground-truth score duration class, and evaluates
them on a held-out split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnose_score_duration_errors import _class_name, _event_bpm
from sweep_score_duration_policies import (
    _duration_class,
    _gt_class,
    _match_onsets,
    _policy_classes,
)
from train_enhanced_mel_transcriber import (
    SCORE_GRID_BEATS,
    EnhancedPrecomputedMelDataset,
    _build_model_from_config,
    _cap_validation_dataset,
    decode_enhanced_note_events,
    enhanced_collate,
    load_segment_manifest,
)


def _pitch_bucket(pitch: int) -> str:
    pitch = int(pitch)
    if pitch < 48:
        return "low"
    if pitch < 72:
        return "mid"
    return "high"


def _conf_bucket(conf: float) -> int:
    conf = float(conf or 0.0)
    if conf >= 0.9:
        return 4
    if conf >= 0.75:
        return 3
    if conf >= 0.6:
        return 2
    if conf >= 0.45:
        return 1
    return 0


def _row_features(row: Dict) -> Dict[str, Tuple]:
    head = row["head"]
    physical = row["physical_duration"]
    sounding = row["sounding_duration"]
    cap = row["sounding_same_pitch_cap"]
    ioi = row["ioi_same_hand"]
    hybrid = row["hybrid_cleanup"]
    pitch_bucket = row["pitch_bucket"]
    conf_bucket = row["conf_bucket"]
    return {
        "ioi": (ioi,),
        "ioi_sound": (ioi, sounding),
        "ioi_cap_sound": (ioi, cap, sounding),
        "ioi_head_sound": (ioi, head, sounding),
        "ioi_head_sound_pitch": (ioi, head, sounding, pitch_bucket),
        "ioi_head_sound_conf": (ioi, head, sounding, conf_bucket),
        "all_candidates": (head, physical, sounding, cap, ioi, hybrid),
        "all_candidates_pitch_conf": (head, physical, sounding, cap, ioi, hybrid, pitch_bucket, conf_bucket),
    }


def _collect_rows(
    args: argparse.Namespace,
    split: str,
    manifest_path: str,
    samples: int,
    label: str,
) -> List[Dict]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    checkpoint_path = Path(args.model_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model_from_config(checkpoint.get("config", {})).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}")

    segment_ids = load_segment_manifest(manifest_path, split)
    dataset = EnhancedPrecomputedMelDataset(split, augment=False, segment_ids=segment_ids)
    dataset = _cap_validation_dataset(dataset, samples, args.sampling, label)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=enhanced_collate,
    )

    rows: List[Dict] = []
    use_amp = device.type == "cuda"
    sample_counter = 0
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch_dev["features"])
            onset_np = torch.sigmoid(out["onset_logits"]).cpu().numpy()
            offset_np = torch.sigmoid(out["offset_logits"]).cpu().numpy()
            physical_np = torch.sigmoid(out["frame_logits"]).cpu().numpy()
            sounding_np = torch.sigmoid(out["sounding_frame_logits"]).cpu().numpy()
            vel_np = out["velocity"].cpu().numpy()
            nv_np = F.softmax(out["note_value_logits"], dim=-1).cpu().numpy()

            for sample_idx in range(onset_np.shape[0]):
                bpm = _event_bpm(batch, sample_idx)
                physical_events = decode_enhanced_note_events(
                    onset_np[sample_idx],
                    offset_np[sample_idx],
                    physical_np[sample_idx],
                    vel_np[sample_idx],
                    nv_np[sample_idx],
                    onset_threshold=args.onset_threshold,
                    offset_threshold=args.offset_threshold,
                    frame_threshold=args.frame_threshold,
                )
                sounding_events = decode_enhanced_note_events(
                    onset_np[sample_idx],
                    offset_np[sample_idx],
                    sounding_np[sample_idx],
                    vel_np[sample_idx],
                    nv_np[sample_idx],
                    onset_threshold=args.onset_threshold,
                    offset_threshold=args.offset_threshold,
                    frame_threshold=args.frame_threshold,
                )
                gt_events = batch["gt_events"][sample_idx]
                matches = _match_onsets(
                    sounding_events,
                    gt_events,
                    bpm,
                    args.score_grid_beats,
                    args.score_onset_slot_tolerance,
                )
                policy_classes = _policy_classes(
                    sounding_events,
                    physical_events,
                    bpm,
                    args.score_grid_beats,
                )
                for pred_idx, gt_idx in matches:
                    pred = sounding_events[pred_idx]
                    row = {
                        "sample_index": sample_counter,
                        "gt": _gt_class(gt_events[gt_idx], bpm),
                        "pitch_bucket": _pitch_bucket(int(pred["midi_note"])),
                        "conf_bucket": _conf_bucket(float(pred.get("note_value_confidence", 0.0) or 0.0)),
                    }
                    for policy in (
                        "head",
                        "physical_duration",
                        "sounding_duration",
                        "sounding_same_pitch_cap",
                        "ioi_same_hand",
                        "hybrid_cleanup",
                    ):
                        row[policy] = policy_classes[policy][pred_idx]
                    rows.append(row)
                sample_counter += 1
    return rows


def _fit_lookup(rows: Sequence[Dict], key_name: str, min_count: int) -> Dict[Tuple, int]:
    buckets = defaultdict(Counter)
    for row in rows:
        key = _row_features(row)[key_name]
        buckets[key][int(row["gt"])] += 1
    return {
        key: counts.most_common(1)[0][0]
        for key, counts in buckets.items()
        if sum(counts.values()) >= min_count
    }


def _eval_rows(rows: Sequence[Dict], lookup: Dict[Tuple, int], key_name: str, fallback: str) -> Dict:
    matched = 0
    pred_hist = Counter()
    confusion = Counter()
    for row in rows:
        key = _row_features(row)[key_name]
        pred = lookup.get(key, row[fallback])
        gt = int(row["gt"])
        pred_hist[pred] += 1
        confusion[(gt, pred)] += 1
        if pred == gt:
            matched += 1
    accuracy = matched / max(len(rows), 1)
    return {
        "duration_accuracy": accuracy,
        "duration_matched": matched,
        "onset_matched": len(rows),
        "pred_hist": [
            {"class": int(cls), "name": _class_name(cls), "count": int(count)}
            for cls, count in pred_hist.most_common()
        ],
        "confusion_top": [
            {
                "gt_class": int(key[0]),
                "gt_name": _class_name(key[0]),
                "pred_class": int(key[1]),
                "pred_name": _class_name(key[1]),
                "count": int(count),
            }
            for key, count in confusion.most_common(20)
        ],
    }


def learn(args: argparse.Namespace) -> Dict:
    if args.eval_manifest:
        train_rows = _collect_rows(args, args.train_split, args.train_manifest, args.train_samples, "lookup_train")
        test_rows = _collect_rows(args, args.eval_split, args.eval_manifest, args.eval_samples, "lookup_eval")
        rows = train_rows + test_rows
    else:
        rows = _collect_rows(args, args.train_split, args.train_manifest, args.train_samples, "lookup_learning")
        train_rows = [row for row in rows if row["sample_index"] % 2 == 0]
        test_rows = [row for row in rows if row["sample_index"] % 2 == 1]
    print(f"rows={len(rows)} train={len(train_rows)} test={len(test_rows)}")

    baselines = {
        name: _eval_rows(test_rows, {}, "ioi", name)
        for name in ("head", "physical_duration", "sounding_duration", "sounding_same_pitch_cap", "ioi_same_hand", "hybrid_cleanup")
    }

    learned = {}
    lookup_tables = {}
    for key_name in (
        "ioi",
        "ioi_sound",
        "ioi_cap_sound",
        "ioi_head_sound",
        "ioi_head_sound_pitch",
        "ioi_head_sound_conf",
        "all_candidates",
        "all_candidates_pitch_conf",
    ):
        for min_count in (4, 8, 16, 32, 64):
            lookup = _fit_lookup(train_rows, key_name, min_count=min_count)
            metrics = _eval_rows(test_rows, lookup, key_name, fallback=args.fallback_policy)
            metrics["lookup_size"] = len(lookup)
            metrics["key_name"] = key_name
            metrics["min_count"] = min_count
            metrics["fallback"] = args.fallback_policy
            learned[f"{key_name}:min{min_count}"] = metrics
            lookup_tables[f"{key_name}:min{min_count}"] = {
                "|".join(str(part) for part in key): int(value)
                for key, value in lookup.items()
            }

    best_name, best_metrics = max(learned.items(), key=lambda item: item[1]["duration_accuracy"])

    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "baselines": baselines,
        "learned": learned,
        "best_policy": best_name,
        "best_lookup": {
            "name": best_name,
            "key_name": best_metrics["key_name"],
            "min_count": best_metrics["min_count"],
            "fallback": best_metrics["fallback"],
            "table": lookup_tables[best_name],
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {output_path}")
    print("Baselines:")
    for name, metrics in sorted(baselines.items(), key=lambda item: item[1]["duration_accuracy"], reverse=True):
        print(f"  {name:28s} durAcc={metrics['duration_accuracy']:.4f}")
    print("Learned:")
    for name, metrics in sorted(learned.items(), key=lambda item: item[1]["duration_accuracy"], reverse=True)[:12]:
        print(
            f"  {name:32s} durAcc={metrics['duration_accuracy']:.4f} "
            f"lookup={metrics['lookup_size']}"
        )
    print(f"Best: {best_name} durAcc={best_metrics['duration_accuracy']:.4f}")
    return summary


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).parent / "enhanced_mel_transcription_pedal_score_repair_latest.pt"
    if not default_model.exists():
        default_model = Path(__file__).parent / "enhanced_mel_transcription_pedal_score_latest.pt"
    parser = argparse.ArgumentParser(description="Learn score-duration lookup policy")
    parser.add_argument("--model-path", type=str, default=str(default_model))
    parser.add_argument("--train-split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--train-manifest", type=str, default="mel_hard_case_manifest_validation_pedal_onset_v2.json")
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--validation-segment-manifest", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--sampling", choices=["spread", "leading"], default="spread")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--score-grid-beats", type=float, default=SCORE_GRID_BEATS)
    parser.add_argument("--score-onset-slot-tolerance", type=int, default=0)
    parser.add_argument("--fallback-policy", type=str, default="ioi_same_hand")
    parser.add_argument("--output-path", type=str, default="score_duration_diagnostics/lookup_policy_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_segment_manifest:
        args.train_manifest = args.validation_segment_manifest
        args.train_split = "validation"
    if args.samples is not None:
        args.train_samples = args.samples
        args.eval_samples = args.samples
    learn(args)


if __name__ == "__main__":
    main()
