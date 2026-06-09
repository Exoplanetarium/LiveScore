"""Diagnostics for the enhanced mel transcriber.

This script intentionally does not train. It answers a few cheap questions:

  - Is event F1 sensitive to using only the first N validation batches?
  - Are the fixed decode thresholds hiding a better operating point?
  - Are the refined onset/offset heads better than the raw heads?

Example:

    python diagnose_enhanced_mel_transcriber.py \
      --model-path enhanced_mel_transcription_finetuned.pt \
      --validation-segment-manifest mel_hard_case_manifest_validation_pedal_onset_v2.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_enhanced_mel_transcriber import (
    EnhancedPrecomputedMelDataset,
    EnhancedTranscriptionLoss,
    _build_model_from_config,
    _frame_f1,
    _move_batch_to_device,
    decode_enhanced_note_events,
    enhanced_collate,
    load_segment_manifest,
    match_note_events,
)


def _parse_float_grid(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _counts_to_metrics(counts: Sequence[int]) -> Dict[str, float]:
    tp, fp, fn = [float(x) for x in counts]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _event_metrics(totals: Dict[str, int]) -> Dict[str, float]:
    precision = totals["matched"] / max(totals["predicted"], 1)
    recall = totals["matched"] / max(totals["ground_truth"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, **dict(totals)}


def _add_counts(target: List[int], counts: Tuple[int, int, int]) -> None:
    target[0] += int(counts[0])
    target[1] += int(counts[1])
    target[2] += int(counts[2])


def _sorted_top(items: Iterable[Dict], metric: str = "f1", n: int = 8) -> List[Dict]:
    return sorted(items, key=lambda item: item[metric], reverse=True)[:n]


@torch.no_grad()
def run_diagnostics(args: argparse.Namespace) -> Dict:
    root = Path(__file__).parent
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = _build_model_from_config(config).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"Loaded with missing={len(missing)} unexpected={len(unexpected)}")

    manifest_path = args.validation_segment_manifest
    if manifest_path and not Path(manifest_path).is_absolute():
        manifest_path = str(root / manifest_path)
    segment_ids = load_segment_manifest(manifest_path, "validation") if manifest_path else None

    dataset = EnhancedPrecomputedMelDataset("validation", augment=False, segment_ids=segment_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=enhanced_collate,
    )

    criterion = EnhancedTranscriptionLoss(
        pos_weight=float(config.get("pos_weight", args.pos_weight)),
        onset_weight=float(config.get("onset_weight", 1.0)),
        offset_weight=float(config.get("offset_weight", 1.0)),
        frame_weight=float(config.get("frame_weight", 0.8)),
        velocity_weight=float(config.get("velocity_weight", 0.3)),
        nv_weight=float(config.get("nv_weight", 0.1)),
    )

    onset_grid = _parse_float_grid(args.onset_grid)
    offset_grid = _parse_float_grid(args.offset_grid)
    frame_grid = _parse_float_grid(args.frame_grid)
    threshold_combos = [(o, off, fr) for o in onset_grid for off in offset_grid for fr in frame_grid]

    refined_onset_counts = [0, 0, 0]
    refined_offset_counts = [0, 0, 0]
    raw_onset_counts = [0, 0, 0]
    raw_offset_counts = [0, 0, 0]
    losses = defaultdict(float)
    n_batches = 0
    n_samples = 0

    default_totals_full = defaultdict(int)
    default_totals_first_window = defaultdict(int)
    raw_default_totals_full = defaultdict(int)
    sweep_totals = {combo: defaultdict(int) for combo in threshold_combos}

    for batch_idx, batch in enumerate(loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        batch_dev = _move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(batch_dev["features"])
            batch_losses = criterion(out, batch_dev)
        for key, value in batch_losses.items():
            losses[key] += float(value.detach().cpu())
        n_batches += 1

        _add_counts(refined_onset_counts, _frame_f1(out["onset_logits"], batch_dev["onset"], args.default_onset))
        _add_counts(refined_offset_counts, _frame_f1(out["offset_logits"], batch_dev["offset"], args.default_offset))
        _add_counts(raw_onset_counts, _frame_f1(out["raw_onset_logits"], batch_dev["onset"], args.default_onset))
        _add_counts(raw_offset_counts, _frame_f1(out["raw_offset_logits"], batch_dev["offset"], args.default_offset))

        onset_np = torch.sigmoid(out["onset_logits"]).float().cpu().numpy()
        offset_np = torch.sigmoid(out["offset_logits"]).float().cpu().numpy()
        raw_onset_np = torch.sigmoid(out["raw_onset_logits"]).float().cpu().numpy()
        raw_offset_np = torch.sigmoid(out["raw_offset_logits"]).float().cpu().numpy()
        frame_np = torch.sigmoid(out["frame_logits"]).float().cpu().numpy()
        vel_np = out["velocity"].float().cpu().numpy()
        nv_np = F.softmax(out["note_value_logits"].float(), dim=-1).cpu().numpy()

        for sample_idx in range(onset_np.shape[0]):
            n_samples += 1
            gt_events = batch["gt_events"][sample_idx]
            pred_default = decode_enhanced_note_events(
                onset_np[sample_idx],
                offset_np[sample_idx],
                frame_np[sample_idx],
                vel_np[sample_idx],
                nv_np[sample_idx],
                onset_threshold=args.default_onset,
                offset_threshold=args.default_offset,
                frame_threshold=args.default_frame,
            )
            metrics = match_note_events(pred_default, gt_events)
            default_totals_full["matched"] += int(metrics["matched"])
            default_totals_full["predicted"] += int(metrics["predicted"])
            default_totals_full["ground_truth"] += int(metrics["ground_truth"])
            if n_samples <= args.first_window_samples:
                default_totals_first_window["matched"] += int(metrics["matched"])
                default_totals_first_window["predicted"] += int(metrics["predicted"])
                default_totals_first_window["ground_truth"] += int(metrics["ground_truth"])

            raw_default = decode_enhanced_note_events(
                raw_onset_np[sample_idx],
                raw_offset_np[sample_idx],
                frame_np[sample_idx],
                vel_np[sample_idx],
                nv_np[sample_idx],
                onset_threshold=args.default_onset,
                offset_threshold=args.default_offset,
                frame_threshold=args.default_frame,
            )
            raw_metrics = match_note_events(raw_default, gt_events)
            raw_default_totals_full["matched"] += int(raw_metrics["matched"])
            raw_default_totals_full["predicted"] += int(raw_metrics["predicted"])
            raw_default_totals_full["ground_truth"] += int(raw_metrics["ground_truth"])

            for combo in threshold_combos:
                onset_t, offset_t, frame_t = combo
                pred_events = decode_enhanced_note_events(
                    onset_np[sample_idx],
                    offset_np[sample_idx],
                    frame_np[sample_idx],
                    vel_np[sample_idx],
                    nv_np[sample_idx],
                    onset_threshold=onset_t,
                    offset_threshold=offset_t,
                    frame_threshold=frame_t,
                )
                combo_metrics = match_note_events(pred_events, gt_events)
                totals = sweep_totals[combo]
                totals["matched"] += int(combo_metrics["matched"])
                totals["predicted"] += int(combo_metrics["predicted"])
                totals["ground_truth"] += int(combo_metrics["ground_truth"])

        if args.progress_every and (batch_idx + 1) % args.progress_every == 0:
            print(f"processed {batch_idx + 1}/{len(loader)} batches, samples={n_samples}")

    sweep = []
    for combo, totals in sweep_totals.items():
        metrics = _event_metrics(totals)
        sweep.append({
            "onset_threshold": combo[0],
            "offset_threshold": combo[1],
            "frame_threshold": combo[2],
            **metrics,
        })

    result = {
        "model_path": str(model_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_event_f1": checkpoint.get("event_f1"),
        "selection_metric_value": checkpoint.get("selection_metric_value"),
        "dataset": {
            "split": "validation",
            "manifest": manifest_path,
            "segments": len(dataset),
            "samples_evaluated": n_samples,
            "batches_evaluated": n_batches,
        },
        "losses": {key: value / max(n_batches, 1) for key, value in losses.items()},
        "framewise_default_threshold": {
            "refined_onset": _counts_to_metrics(refined_onset_counts),
            "raw_onset": _counts_to_metrics(raw_onset_counts),
            "refined_offset": _counts_to_metrics(refined_offset_counts),
            "raw_offset": _counts_to_metrics(raw_offset_counts),
        },
        "event_default_threshold": {
            "refined_full": _event_metrics(default_totals_full),
            "refined_first_window": _event_metrics(default_totals_first_window),
            "raw_full": _event_metrics(raw_default_totals_full),
        },
        "event_threshold_sweep_top": _sorted_top(sweep, n=args.top_n),
        "event_threshold_sweep_all": sorted(
            sweep,
            key=lambda item: (item["onset_threshold"], item["offset_threshold"], item["frame_threshold"]),
        ),
    }

    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = root.parent / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="enhanced_mel_transcription_finetuned.pt")
    parser.add_argument("--validation-segment-manifest", default="mel_hard_case_manifest_validation_pedal_onset_v2.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--first-window-samples", type=int, default=160)
    parser.add_argument("--default-onset", type=float, default=0.5)
    parser.add_argument("--default-offset", type=float, default=0.35)
    parser.add_argument("--default-frame", type=float, default=0.5)
    parser.add_argument("--onset-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--offset-grid", default="0.25,0.35,0.45")
    parser.add_argument("--frame-grid", default="0.35,0.50,0.65")
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--output", default="_tmp_json/diagnose_enhanced_mel_finetuned_hard.json")
    args = parser.parse_args()

    result = run_diagnostics(args)
    default = result["event_default_threshold"]["refined_full"]
    raw = result["event_default_threshold"]["raw_full"]
    first = result["event_default_threshold"]["refined_first_window"]
    best = result["event_threshold_sweep_top"][0]
    print(
        "default refined full: "
        f"P={default['precision']:.4f} R={default['recall']:.4f} F1={default['f1']:.4f} "
        f"matched={default['matched']} pred={default['predicted']} gt={default['ground_truth']}"
    )
    print(
        "default refined first-window: "
        f"P={first['precision']:.4f} R={first['recall']:.4f} F1={first['f1']:.4f}"
    )
    print(
        "default raw full: "
        f"P={raw['precision']:.4f} R={raw['recall']:.4f} F1={raw['f1']:.4f}"
    )
    print(
        "best sweep: "
        f"onset={best['onset_threshold']:.2f} offset={best['offset_threshold']:.2f} "
        f"frame={best['frame_threshold']:.2f} P={best['precision']:.4f} "
        f"R={best['recall']:.4f} F1={best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
