from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
MANIFEST = BACKEND / "live_benchmark_replay_auto_v2.json"
SCRIPT = BACKEND / "test_experiment.py"
ARTIFACTS = BACKEND / "benchmark_artifacts" / "enhanced_heuristics_20260602"

BASE_ENV = {
    "LIVE_ENHANCED_ONSET_BASE": "0.75",
    "LIVE_ENHANCED_OFFSET_BASE": "0.35",
    "LIVE_ENHANCED_MIN_VELOCITY": "8",
    "LIVE_ENHANCED_FILTER_HARMONICS": "0",
    "LIVE_ENHANCED_DUPLICATE_WINDOW_SEC": "0.04",
    "LIVE_ENHANCED_MERGE_GAP_SEC": "0.0",
    "LIVE_CONTEXT_SEC": "2.4",
}

CONFIGS = [
    ("context36_on070", {"LIVE_CONTEXT_SEC": "3.6", "LIVE_ENHANCED_ONSET_BASE": "0.70"}),
    ("context36_on080", {"LIVE_CONTEXT_SEC": "3.6", "LIVE_ENHANCED_ONSET_BASE": "0.80"}),
    ("context36_on085", {"LIVE_CONTEXT_SEC": "3.6", "LIVE_ENHANCED_ONSET_BASE": "0.85"}),
]

METRICS = [
    "f1",
    "precision",
    "recall",
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
        run_summary = {}
        for metric in METRICS:
            values = []
            for clip in clips.values():
                summary = clip.get(run) or {}
                if metric in summary and summary[metric] is not None:
                    values.append(float(summary[metric]))
            if values:
                run_summary[metric] = st.mean(values)
        out[run] = run_summary
    return out


def run_config(name: str, overrides: dict[str, str]) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = BACKEND / f"live_benchmark_replay_auto_v2_results_20260602_enhancedheur_{name}.json"
    log = ARTIFACTS / f"{name}.log"

    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(overrides)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--benchmark-manifest",
        str(MANIFEST),
        "--no-run-retro-correction",
        "--output-json",
        str(output),
    ]

    print(f"running {name}: {overrides or 'baseline'}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    summary = aggregate(output)
    summary["name"] = name
    summary["env"] = {**BASE_ENV, **overrides}
    control = summary["control"]
    treatment = summary["treatment"]
    print(
        f"done {name}: "
        f"control cluster={control.get('display_cluster_f1', 0):.4f} "
        f"note={control.get('display_note_f1', 0):.4f} "
        f"offset={control.get('display_offset_f1', 0):.4f} "
        f"dup100={control.get('duplicates_per_100_notes', 0):.2f}; "
        f"treatment cluster={treatment.get('display_cluster_f1', 0):.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    summaries = [run_config(name, overrides) for name, overrides in CONFIGS]
    summary_path = BACKEND / "enhanced_mel_heuristic_sweep_20260602_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    ranked = sorted(
        summaries,
        key=lambda item: (
            item["control"].get("display_cluster_f1", 0.0),
            item["control"].get("display_note_f1", 0.0),
            item["control"].get("display_offset_f1", 0.0),
        ),
        reverse=True,
    )
    print(f"wrote {summary_path}")
    print("top by control display_cluster_f1:")
    for item in ranked[:8]:
        control = item["control"]
        print(
            f"  {item['name']}: cluster={control.get('display_cluster_f1', 0):.4f} "
            f"display_note={control.get('display_note_f1', 0):.4f} "
            f"display_offset={control.get('display_offset_f1', 0):.4f} "
            f"note_f1={control.get('f1', 0):.4f} "
            f"dup100={control.get('duplicates_per_100_notes', 0):.2f} "
            f"p95={control.get('p95_chunk_total_ms', 0):.1f}"
        )


if __name__ == "__main__":
    main()
