"""Export a trained lattice-candidate calibrator pickle to plain JSON.

The runtime live path must not depend on sklearn/pickle (version-coupled and
heavy). This script reads a ``lattice_candidate_calibrator.pkl`` produced by
``train_lattice_candidate_calibrator.py`` and writes a dependency-free JSON that
``lattice_candidate_decoder.py`` can load with numpy only.

The exported JSON reproduces ``Pipeline(StandardScaler, LogisticRegression)``:

    z      = (x - scaler_mean) / scaler_scale
    logit  = z . coef + intercept
    prob   = 1 / (1 + exp(-logit))

Example:

    python export_lattice_calibrator_json.py \
      --pkl lattice_candidate_calibrator_hard256_ordered_p75/lattice_candidate_calibrator.pkl \
      --out lattice_candidate_calibrator.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


# Candidate-generation knobs that must match the runtime decoder. These mirror
# the defaults baked into train_lattice_candidate_calibrator.build_arg_parser so
# the runtime candidate pool reproduces the training-time candidate pool.
CANDIDATE_ARG_KEYS = [
    "candidate_onset_threshold",
    "candidate_frame_threshold",
    "candidate_min_velocity",
    "frame_threshold",
    "cluster_tolerance_sec",
    "duplicate_tolerance_sec",
    "lookback_sec",
    "min_note_duration",
    "max_anchor_distance_sec",
    "max_additions_per_anchor",
    "snap_to_anchor",
    "primary_onset_threshold",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl",
        default="lattice_candidate_calibrator_hard256_ordered_p75/lattice_candidate_calibrator.pkl",
        help="Path to the trained calibrator pickle (relative to this directory).",
    )
    parser.add_argument(
        "--out",
        default="lattice_candidate_calibrator.json",
        help="Output JSON path (relative to this directory).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(__file__).parent
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = root / pkl_path
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)

    pipeline = payload["model"]
    scaler = pipeline.named_steps["scale"]
    clf = pipeline.named_steps["clf"]

    feature_names = list(payload.get("feature_names") or [])
    train_args = dict(payload.get("args") or {})
    candidate_args = {
        key: train_args.get(key)
        for key in CANDIDATE_ARG_KEYS
        if key in train_args
    }

    export = {
        "model_type": "standard_scaler_logistic_regression",
        "feature_names": feature_names,
        "scaler_mean": [float(v) for v in scaler.mean_.tolist()],
        "scaler_scale": [float(v) for v in scaler.scale_.tolist()],
        "coef": [float(v) for v in clf.coef_[0].tolist()],
        "intercept": float(clf.intercept_[0]),
        "threshold": float(payload["threshold"]),
        "candidate_args": candidate_args,
        "source_pkl": str(pkl_path.name),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"  features: {len(feature_names)}")
    print(f"  threshold: {export['threshold']:.4f}")
    print(f"  candidate_args: {json.dumps(candidate_args)}")


if __name__ == "__main__":
    main()
