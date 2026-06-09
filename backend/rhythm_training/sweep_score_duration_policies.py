"""Sweep score-duration post-processing policies.

The transcriber already gets pitch/onset mostly right. This script keeps the
decoded notes fixed and compares different ways to assign the visible score
duration class: model note-value head, decoded physical duration, decoded
sounding duration, predicted IOI, and a simple hybrid cleanup.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnose_score_duration_errors import _class_name, _event_bpm, _slot
from train_enhanced_mel_transcriber import (
    HOP_LENGTH,
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


POLICIES = (
    "head",
    "physical_duration",
    "sounding_duration",
    "sounding_same_pitch_cap",
    "ioi_same_hand",
    "hybrid_cleanup",
    "consensus_or_ioi",
    "confident_head_or_ioi",
    "sounding_if_longer_else_ioi",
    "sounding_if_much_longer_else_ioi",
    "bass_sounding_treble_ioi",
    "oracle_any_candidate",
)


def _duration_class(start_sec: float, end_sec: float, bpm: float) -> int:
    return _duration_to_note_value_class(max(float(end_sec) - float(start_sec), 1e-6), bpm)


def _score_slot(event: Dict, bpm: float, grid_beats: float) -> int:
    return _slot(float(event["onset_time"]), bpm, grid_beats)


def _hand(pitch: int) -> str:
    return "bass" if int(pitch) < 60 else "treble"


def _physical_event_lookup(events: Sequence[Dict], bpm: float, grid_beats: float) -> Dict[Tuple[int, int], Dict]:
    lookup = {}
    for event in events:
        key = (int(event["midi_note"]), _score_slot(event, bpm, grid_beats))
        existing = lookup.get(key)
        if existing is None or float(event.get("onset_prob", 0.0)) > float(existing.get("onset_prob", 0.0)):
            lookup[key] = event
    return lookup


def _next_onsets(events: Sequence[Dict]) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, Dict[int, float]]]:
    by_pitch = defaultdict(list)
    by_hand = defaultdict(list)
    for idx, event in enumerate(events):
        pitch = int(event["midi_note"])
        onset = float(event["onset_time"])
        by_pitch[pitch].append((onset, idx))
        by_hand[_hand(pitch)].append((onset, idx))

    next_same_pitch = {}
    for values in by_pitch.values():
        values.sort()
        for pos, (_, idx) in enumerate(values[:-1]):
            next_same_pitch[idx] = values[pos + 1][0]

    next_same_hand = {}
    for values in by_hand.values():
        values.sort()
        for pos, (_, idx) in enumerate(values[:-1]):
            next_same_hand[idx] = values[pos + 1][0]

    return next_same_pitch, next_same_hand, {}


def _policy_classes(
    events: Sequence[Dict],
    physical_events: Sequence[Dict],
    bpm: float,
    grid_beats: float,
) -> Dict[str, List[int]]:
    physical_lookup = _physical_event_lookup(physical_events, bpm, grid_beats)
    next_same_pitch, next_same_hand, _ = _next_onsets(events)
    result = {policy: [] for policy in POLICIES if policy != "oracle_any_candidate"}
    candidate_classes = []

    for idx, event in enumerate(events):
        onset = float(event["onset_time"])
        sounding_end = float(event["offset_time"])
        same_pitch_end = min(sounding_end, float(next_same_pitch.get(idx, sounding_end)))
        same_hand_end = float(next_same_hand.get(idx, same_pitch_end))
        same_hand_end = min(same_hand_end, float(next_same_pitch.get(idx, same_hand_end)))

        key = (int(event["midi_note"]), _score_slot(event, bpm, grid_beats))
        physical_event = physical_lookup.get(key)
        physical_end = float(physical_event["offset_time"]) if physical_event is not None else sounding_end

        head_class = int(event.get("note_value_class", _duration_class(onset, same_pitch_end, bpm)))
        head_conf = float(event.get("note_value_confidence", 0.0) or 0.0)
        physical_class = _duration_class(onset, physical_end, bpm)
        sounding_class = _duration_class(onset, sounding_end, bpm)
        capped_class = _duration_class(onset, same_pitch_end, bpm)
        ioi_class = _duration_class(onset, same_hand_end, bpm)

        # Cleanup policy: trust sounding/cap for normal and long notes, but if it
        # collapses to a tiny value while the same-hand IOI supports a longer
        # score value, promote it. This directly attacks excessive 32nd output.
        hybrid_class = capped_class
        if capped_class <= 1 and ioi_class >= 2:
            hybrid_class = ioi_class
        elif capped_class <= 3 and ioi_class >= capped_class + 2:
            hybrid_class = min(ioi_class, capped_class + 2)

        votes = Counter([head_class, physical_class, sounding_class, capped_class, ioi_class])
        top_vote_class, top_vote_count = votes.most_common(1)[0]
        consensus_class = top_vote_class if top_vote_count >= 2 else ioi_class

        confident_head_class = head_class if head_conf >= 0.85 and abs(head_class - ioi_class) <= 1 else ioi_class
        sounding_longer_class = sounding_class if sounding_class > ioi_class else ioi_class
        sounding_much_longer_class = sounding_class if sounding_class >= ioi_class + 2 else ioi_class
        bass_sounding_class = sounding_class if int(event["midi_note"]) < 60 else ioi_class

        result["head"].append(head_class)
        result["physical_duration"].append(physical_class)
        result["sounding_duration"].append(sounding_class)
        result["sounding_same_pitch_cap"].append(capped_class)
        result["ioi_same_hand"].append(ioi_class)
        result["hybrid_cleanup"].append(hybrid_class)
        result["consensus_or_ioi"].append(consensus_class)
        result["confident_head_or_ioi"].append(confident_head_class)
        result["sounding_if_longer_else_ioi"].append(sounding_longer_class)
        result["sounding_if_much_longer_else_ioi"].append(sounding_much_longer_class)
        result["bass_sounding_treble_ioi"].append(bass_sounding_class)
        candidate_classes.append({
            "head": head_class,
            "physical_duration": physical_class,
            "sounding_duration": sounding_class,
            "sounding_same_pitch_cap": capped_class,
            "ioi_same_hand": ioi_class,
            "hybrid_cleanup": hybrid_class,
            "consensus_or_ioi": consensus_class,
            "confident_head_or_ioi": confident_head_class,
            "sounding_if_longer_else_ioi": sounding_longer_class,
            "sounding_if_much_longer_else_ioi": sounding_much_longer_class,
            "bass_sounding_treble_ioi": bass_sounding_class,
        })

    result["oracle_any_candidate"] = candidate_classes
    return result


def _match_onsets(
    events: Sequence[Dict],
    gt_events: Sequence[Dict],
    bpm: float,
    grid_beats: float,
    slot_tolerance: int,
) -> List[Tuple[int, int]]:
    used_gt = set()
    matches = []
    pred_slots = [_score_slot(event, bpm, grid_beats) for event in events]
    gt_slots = [_score_slot(event, bpm, grid_beats) for event in gt_events]
    for pred_idx, event in enumerate(events):
        pitch = int(event["midi_note"])
        best_idx = None
        best_error = None
        for gt_idx, gt in enumerate(gt_events):
            if gt_idx in used_gt or pitch != int(gt["midi_note"]):
                continue
            error = abs(pred_slots[pred_idx] - gt_slots[gt_idx])
            if error > slot_tolerance:
                continue
            if best_error is None or error < best_error:
                best_idx = gt_idx
                best_error = error
        if best_idx is not None:
            used_gt.add(best_idx)
            matches.append((pred_idx, best_idx))
    return matches


def _gt_class(event: Dict, bpm: float) -> int:
    if "note_value_class" in event:
        return int(event["note_value_class"])
    return _duration_class(float(event["onset_time"]), float(event["offset_time"]), bpm)


def _empty_policy_stats() -> Dict:
    return {
        "duration_matched": 0,
        "onset_matched": 0,
        "predicted": 0,
        "ground_truth": 0,
        "confusion": Counter(),
        "pred_hist": Counter(),
    }


def _finalize_policy_stats(stats: Dict) -> Dict:
    precision = stats["duration_matched"] / max(stats["predicted"], 1)
    recall = stats["duration_matched"] / max(stats["ground_truth"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    onset_precision = stats["onset_matched"] / max(stats["predicted"], 1)
    onset_recall = stats["onset_matched"] / max(stats["ground_truth"], 1)
    onset_f1 = 2 * onset_precision * onset_recall / max(onset_precision + onset_recall, 1e-8)
    duration_accuracy = stats["duration_matched"] / max(stats["onset_matched"], 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onset_f1": onset_f1,
        "duration_accuracy": duration_accuracy,
        "duration_matched": int(stats["duration_matched"]),
        "onset_matched": int(stats["onset_matched"]),
        "predicted": int(stats["predicted"]),
        "ground_truth": int(stats["ground_truth"]),
        "pred_hist": [
            {"class": int(cls), "name": _class_name(cls), "count": int(count)}
            for cls, count in stats["pred_hist"].most_common()
        ],
        "confusion_top": [
            {
                "gt_class": int(key[0]),
                "gt_name": _class_name(key[0]),
                "pred_class": int(key[1]),
                "pred_name": _class_name(key[1]),
                "count": int(count),
            }
            for key, count in stats["confusion"].most_common(20)
        ],
    }


def sweep(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    checkpoint_path = Path(args.model_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model_from_config(checkpoint.get("config", {})).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}")

    segment_ids = load_segment_manifest(args.validation_segment_manifest, "validation")
    dataset = EnhancedPrecomputedMelDataset("validation", augment=False, segment_ids=segment_ids)
    dataset = _cap_validation_dataset(dataset, args.samples, args.sampling, "policy_sweep")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=enhanced_collate,
    )

    policy_stats = {policy: _empty_policy_stats() for policy in POLICIES}
    totals = Counter()
    use_amp = device.type == "cuda"

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
                classes_by_policy = _policy_classes(
                    sounding_events,
                    physical_events,
                    bpm,
                    args.score_grid_beats,
                )
                totals["samples"] += 1

                for policy, stats in policy_stats.items():
                    stats["predicted"] += len(sounding_events)
                    stats["ground_truth"] += len(gt_events)
                    stats["onset_matched"] += len(matches)
                    for pred_idx, gt_idx in matches:
                        gt_class = _gt_class(gt_events[gt_idx], bpm)
                        if policy == "oracle_any_candidate":
                            pred_class = None
                            correct = any(
                                candidate == gt_class
                                for candidate in classes_by_policy[policy][pred_idx].values()
                            )
                            if correct:
                                pred_class = gt_class
                            else:
                                pred_class = classes_by_policy[policy][pred_idx]["hybrid_cleanup"]
                        else:
                            pred_class = classes_by_policy[policy][pred_idx]
                            correct = pred_class == gt_class
                        stats["pred_hist"][pred_class] += 1
                        stats["confusion"][(gt_class, pred_class)] += 1
                        if correct:
                            stats["duration_matched"] += 1

    summary = {
        "model_path": str(checkpoint_path),
        "samples": int(totals["samples"]),
        "policies": {
            policy: _finalize_policy_stats(stats)
            for policy, stats in policy_stats.items()
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {output_path}")
    for policy, stats in sorted(summary["policies"].items(), key=lambda item: item[1]["f1"], reverse=True):
        print(
            f"{policy:24s} f1={stats['f1']:.4f} "
            f"durAcc={stats['duration_accuracy']:.4f} onsetF1={stats['onset_f1']:.4f}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).parent / "enhanced_mel_transcription_pedal_score_repair_latest.pt"
    if not default_model.exists():
        default_model = Path(__file__).parent / "enhanced_mel_transcription_pedal_score_latest.pt"
    parser = argparse.ArgumentParser(description="Sweep score-duration policies")
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
    parser.add_argument("--output-path", type=str, default="score_duration_diagnostics/policy_sweep_summary.json")
    return parser.parse_args()


def main() -> None:
    sweep(parse_args())


if __name__ == "__main__":
    main()
