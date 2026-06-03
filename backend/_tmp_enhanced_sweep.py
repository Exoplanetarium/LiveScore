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

CONFIGS = [
    {"name": "on085_v08_fh0", "onset": "0.85", "min_velocity": "8", "filter_harmonics": "0"},
    {"name": "on090_v08_fh0", "onset": "0.90", "min_velocity": "8", "filter_harmonics": "0"},
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


def main() -> None:
    summaries = []
    for config in CONFIGS:
        name = config["name"]
        output = BACKEND / f"live_benchmark_replay_auto_v2_results_20260602_enhancedmel_{name}.json"
        log = BACKEND / f"_tmp_enhanced_sweep_{name}.log"

        env = os.environ.copy()
        env.update(
            {
                "LIVE_ENHANCED_ONSET_BASE": config["onset"],
                "LIVE_ENHANCED_MIN_VELOCITY": config["min_velocity"],
                "LIVE_ENHANCED_FILTER_HARMONICS": config["filter_harmonics"],
                "LIVE_ENHANCED_OFFSET_BASE": "0.35",
            }
        )

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--benchmark-manifest",
            str(MANIFEST),
            "--no-run-retro-correction",
            "--output-json",
            str(output),
        ]

        print(f"running {name} onset={config['onset']} min_velocity={config['min_velocity']} filter_harmonics={config['filter_harmonics']}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

        summary = aggregate(output)
        summary["config"] = dict(config)
        summaries.append(summary)
        control = summary["control"]
        print(
            f"done {name}: control_f1={control.get('f1', 0):.4f} "
            f"display_note_f1={control.get('display_note_f1', 0):.4f} "
            f"cluster_f1={control.get('display_cluster_f1', 0):.4f} "
            f"offset_f1={control.get('offset_f1', 0):.4f} "
            f"dup100={control.get('duplicates_per_100_notes', 0):.2f}",
            flush=True,
        )

    summary_path = BACKEND / "enhanced_mel_sweep_20260602_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")

    ranked = sorted(
        summaries,
        key=lambda item: (
            item["control"].get("display_cluster_f1", 0.0),
            item["control"].get("f1", 0.0),
        ),
        reverse=True,
    )
    print("top by control display_cluster_f1:")
    for item in ranked[:5]:
        control = item["control"]
        print(
            f"  {item['config']['name']}: cluster={control.get('display_cluster_f1', 0):.4f} "
            f"f1={control.get('f1', 0):.4f} display_note={control.get('display_note_f1', 0):.4f} "
            f"offset={control.get('offset_f1', 0):.4f} dup100={control.get('duplicates_per_100_notes', 0):.2f}"
        )


if __name__ == "__main__":
    main()
