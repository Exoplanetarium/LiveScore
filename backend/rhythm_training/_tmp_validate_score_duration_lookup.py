"""Held-out validation of the score-duration lookup policy via the real evaluator.

Follows through on the 2026-06-14 change-log decision: regenerate a real
full-size, evaluator-compatible (3-part ioi|head|sound) lookup table on the
TRAIN manifest, then validate it on a broader validation slice through the same
evaluate() path the trainer uses. Temporary diagnostic; not shipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

import learn_score_duration_lookup as lsd
from train_enhanced_mel_transcriber import (
    EnhancedTranscriptionLoss,
    _build_model_from_config,
    _build_validation_loaders,
    evaluate,
    parse_args,
)

REPAIR_CKPT = "enhanced_mel_transcription_pedal_score_repair_latest.pt"
TRAIN_MANIFEST = "mel_hard_case_manifest_train_pedal_onset_v2.json"
VAL_MANIFEST = "mel_hard_case_manifest_validation_pedal_onset_v2.json"
TABLE_OUT = "score_duration_diagnostics/lookup_ioi_head_sound_heldout_full.json"

FIT_SAMPLES = 768       # train-manifest rows collected to fit the table
VAL_SAMPLES = 256       # broader held-out slice (smoke check used only 8)
KEY_NAME = "ioi_head_sound"  # 3-part key the evaluator can load


def fit_table(device: str) -> dict:
    fit_args = SimpleNamespace(
        model_path=REPAIR_CKPT,
        device=device,
        batch_size=8,
        num_workers=4,
        sampling="leading",
        onset_threshold=0.5,
        offset_threshold=0.35,
        frame_threshold=0.5,
        score_grid_beats=lsd.SCORE_GRID_BEATS,
        score_onset_slot_tolerance=0,
    )
    rows = lsd._collect_rows(fit_args, "train", TRAIN_MANIFEST, FIT_SAMPLES, "lookup_fit_train")
    # internal even/odd split to pick min_count honestly on the train manifest
    train_rows = [r for r in rows if r["sample_index"] % 2 == 0]
    held_rows = [r for r in rows if r["sample_index"] % 2 == 1]
    best = None
    for min_count in (4, 8, 16, 32, 64):
        table = lsd._fit_lookup(train_rows, KEY_NAME, min_count=min_count)
        metrics = lsd._eval_rows(held_rows, table, KEY_NAME, fallback="ioi_same_hand")
        acc = metrics["duration_accuracy"]
        print(f"  fit {KEY_NAME}:min{min_count} size={len(table)} heldDurAcc={acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, min_count, table)
    # refit the chosen min_count on ALL collected rows for the shipped table
    _, min_count, _ = best
    full_table = lsd._fit_lookup(rows, KEY_NAME, min_count=min_count)
    print(f"rows={len(rows)} chosen min_count={min_count} final_table_size={len(full_table)}")
    payload = {
        "best_lookup": {
            "name": f"{KEY_NAME}:min{min_count}",
            "key_name": KEY_NAME,
            "min_count": min_count,
            "fallback": "ioi_same_hand",
            "table": {"|".join(str(p) for p in k): int(v) for k, v in full_table.items()},
        },
        "meta": {
            "fit_manifest": TRAIN_MANIFEST,
            "fit_split": "train",
            "fit_rows": len(rows),
            "fit_samples": FIT_SAMPLES,
            "model_path": REPAIR_CKPT,
        },
    }
    out = Path(TABLE_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return payload


def run_eval(policy: str, lookup_path: str | None) -> dict:
    argv = [
        "--validation-segment-manifest", VAL_MANIFEST,
        "--model-path", REPAIR_CKPT,
        "--batch-size", "8",
        "--num-workers", "4",
        "--max-val-samples", str(VAL_SAMPLES),
        "--val-sampling", "spread",
        "--max-score-val-samples", str(VAL_SAMPLES),
        "--score-duration-policy", policy,
    ]
    if lookup_path:
        argv += ["--score-duration-lookup-path", lookup_path]
    import sys
    saved = sys.argv
    sys.argv = ["eval"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = saved

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ckpt = torch.load(Path(args.model_path), map_location=device, weights_only=False)
    model = _build_model_from_config(ckpt.get("config", {})).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    loaders = _build_validation_loaders(args, device)
    name = next(iter(loaders))
    loader = loaders[name]
    criterion = EnhancedTranscriptionLoss()

    result = evaluate(
        model, loader, criterion, device, use_amp,
        args.onset_threshold, args.offset_threshold, args.frame_threshold,
        max_event_samples=0,
        max_score_samples=VAL_SAMPLES,
        score_sampling="spread",
        score_grid_beats=args.score_grid_beats,
        score_onset_slot_tolerance=args.score_onset_slot_tolerance,
        score_duration_class_tolerance=args.score_duration_class_tolerance,
        score_duration_policy=policy,
        score_duration_lookup_path=lookup_path,
        decode_use_sounding_frame=args.decode_use_sounding_frame,
    )
    score = result["score"]
    print(f"[{policy:24s}] samples={score.get('samples')} "
          f"f1={score.get('f1'):.4f} precision={score.get('precision'):.4f} "
          f"recall={score.get('recall'):.4f} durAcc={score.get('duration_accuracy'):.4f}")
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-fit", action="store_true")
    a = ap.parse_args()

    if not a.skip_fit:
        print("== Fitting evaluator-compatible 3-part ioi_head_sound table (held out on train manifest) ==")
        fit_table(a.device)

    print("\n== Held-out validation on", VAL_MANIFEST, f"({VAL_SAMPLES} samples) ==")
    base = run_eval("ioi_same_hand", None)
    lookup = run_eval("lookup_ioi_head_sound", TABLE_OUT)
    print("\n== DELTA (lookup - baseline) ==")
    for k in ("f1", "precision", "recall", "duration_accuracy"):
        if k in base and k in lookup:
            print(f"  {k:18s} {base[k]:.4f} -> {lookup[k]:.4f}  ({lookup[k]-base[k]:+.4f})")


if __name__ == "__main__":
    main()
