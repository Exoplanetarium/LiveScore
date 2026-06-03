from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
SCRIPT = BACKEND / "test_experiment.py"
MANIFEST = BACKEND / "live_benchmark_replay_auto_v2.json"
ARTIFACTS = BACKEND / "benchmark_artifacts" / "enhanced_cluster_20260602"

BASE_ENV = {
    "LIVE_ENHANCED_ONSET_BASE": "0.75",
    "LIVE_ENHANCED_OFFSET_BASE": "0.35",
    "LIVE_CONTEXT_SEC": "2.4",
    "LIVE_ENHANCED_DUPLICATE_WINDOW_SEC": "0.04",
    "LIVE_ENHANCED_MERGE_GAP_SEC": "0.0",
    "LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.03",
    "LIVE_NEURAL_GROUP_MIN_TOLERANCE_SEC": "0.012",
    "LIVE_NEURAL_GROUP_SHRINK_SEC": "0.004",
    "LIVE_NEURAL_GROUP_STEP_RATIO": "0.65",
    "LIVE_NEURAL_GROUP_PRUNE_ENABLED": "0",
}

CONFIGS = [
    ("group_base020", {"LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.020"}),
    ("group_base025", {"LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.025"}),
    ("group_base035", {"LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.035"}),
    ("group_base040", {"LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.040"}),
    ("group_shrink002", {"LIVE_NEURAL_GROUP_SHRINK_SEC": "0.002"}),
    ("group_shrink006", {"LIVE_NEURAL_GROUP_SHRINK_SEC": "0.006"}),
    ("group_step050", {"LIVE_NEURAL_GROUP_STEP_RATIO": "0.50"}),
    ("group_step080", {"LIVE_NEURAL_GROUP_STEP_RATIO": "0.80"}),
    (
        "prune055_r055_min3",
        {
            "LIVE_NEURAL_GROUP_PRUNE_ENABLED": "1",
            "LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET": "0.55",
            "LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO": "0.55",
            "LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE": "3",
        },
    ),
    (
        "prune060_r060_min3",
        {
            "LIVE_NEURAL_GROUP_PRUNE_ENABLED": "1",
            "LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET": "0.60",
            "LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO": "0.60",
            "LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE": "3",
        },
    ),
    (
        "prune065_r070_min3",
        {
            "LIVE_NEURAL_GROUP_PRUNE_ENABLED": "1",
            "LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET": "0.65",
            "LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO": "0.70",
            "LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE": "3",
        },
    ),
    (
        "prune060_r060_min4",
        {
            "LIVE_NEURAL_GROUP_PRUNE_ENABLED": "1",
            "LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET": "0.60",
            "LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO": "0.60",
            "LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE": "4",
        },
    ),
]

METRICS = [
    "f1",
    "display_note_f1",
    "display_cluster_f1",
    "display_offset_f1",
    "offset_f1",
    "duplicates_per_100_notes",
    "boundary_miss_rate",
    "p95_chunk_total_ms",
]


def aggregate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips") or {}
    out = {"path": str(path), "clips": len(clips)}
    for run in ("control", "treatment"):
        summary = {}
        for metric in METRICS:
            values = []
            for clip in clips.values():
                run_summary = clip.get(run) or {}
                if metric in run_summary and run_summary[metric] is not None:
                    values.append(float(run_summary[metric]))
            if values:
                summary[metric] = st.mean(values)
        out[run] = summary
    return out


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, overrides in CONFIGS:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(overrides)

        output = BACKEND / f"live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_{name}.json"
        log = ARTIFACTS / f"{name}.log"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--benchmark-manifest",
            str(MANIFEST),
            "--no-run-retro-correction",
            "--output-json",
            str(output),
        ]

        print(f"running {name}: {overrides}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

        summary = aggregate(output)
        summary["name"] = name
        summary["env"] = {**BASE_ENV, **overrides}
        summaries.append(summary)
        control = summary["control"]
        treatment = summary["treatment"]
        print(
            f"done {name}: control cluster={control.get('display_cluster_f1', 0):.4f} "
            f"note={control.get('display_note_f1', 0):.4f} "
            f"offset={control.get('display_offset_f1', 0):.4f}; "
            f"treatment cluster={treatment.get('display_cluster_f1', 0):.4f}",
            flush=True,
        )

    summary_path = BACKEND / "enhanced_mel_cluster_sweep_20260602_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    ranked = sorted(
        summaries,
        key=lambda item: (
            item["control"].get("display_cluster_f1", 0.0),
            item["treatment"].get("display_cluster_f1", 0.0),
            item["control"].get("display_note_f1", 0.0),
        ),
        reverse=True,
    )
    print(f"wrote {summary_path}")
    print("top by control display_cluster_f1:")
    for item in ranked[:8]:
        control = item["control"]
        treatment = item["treatment"]
        print(
            f"  {item['name']}: control_cluster={control.get('display_cluster_f1', 0):.4f} "
            f"treatment_cluster={treatment.get('display_cluster_f1', 0):.4f} "
            f"display_note={control.get('display_note_f1', 0):.4f} "
            f"display_offset={control.get('display_offset_f1', 0):.4f}"
        )


if __name__ == "__main__":
    main()
