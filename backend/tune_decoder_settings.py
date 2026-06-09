r"""Sweep zero-latency live decoder settings and rank score-quality submetrics.

This wrapper repeatedly runs ``test_experiment.py`` with different environment
variables that affect decoder thresholds/grouping only. It does not change the
neural model architecture or inference cadence, so useful wins should preserve
latency unless the measured timing says otherwise.

Example:
    .\env\Scripts\python.exe tune_decoder_settings.py --preset quick --clip-ids clip_001 clip_002
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from score_quality_report import aggregate_arm


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "live_benchmark_replay_auto_v2.json"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark_artifacts"


Candidate = Dict[str, Any]


def _candidate(name: str, env: Mapping[str, str], notes: str) -> Candidate:
    return {
        "name": name,
        "env": dict(env),
        "notes": notes,
    }


def build_candidates(preset: str) -> List[Candidate]:
    """Return decoder-only candidates.

    These settings are intentionally conservative. The quick preset tests the
    knobs most likely to trade missing melody against pedal false positives.
    """
    candidates = [
        _candidate(
            "baseline_current",
            {},
            "Current checked-in decoder settings.",
        ),
        _candidate(
            "enhanced_onset_070",
            {"LIVE_ENHANCED_ONSET_BASE": "0.70"},
            "More recall from the enhanced model onset head.",
        ),
        _candidate(
            "enhanced_onset_080",
            {"LIVE_ENHANCED_ONSET_BASE": "0.80"},
            "More precision from the enhanced model onset head.",
        ),
        _candidate(
            "enhanced_offset_025",
            {"LIVE_ENHANCED_OFFSET_BASE": "0.25"},
            "Looser offset detection; may improve duration without changing note births.",
        ),
        _candidate(
            "enhanced_offset_045",
            {"LIVE_ENHANCED_OFFSET_BASE": "0.45"},
            "Stricter offset detection; checks whether duration noise is hurting score display.",
        ),
        _candidate(
            "duplicate_window_060ms",
            {"LIVE_ENHANCED_DUPLICATE_WINDOW_SEC": "0.06"},
            "Merge very close duplicate same-pitch events more aggressively.",
        ),
        _candidate(
            "harmonic_filter_on",
            {"LIVE_ENHANCED_FILTER_HARMONICS": "1"},
            "Use model-wrapper harmonic filtering to test pedal/harmonic false positives.",
        ),
    ]

    if preset == "full":
        candidates.extend(
            [
                _candidate(
                    "enhanced_onset_065",
                    {"LIVE_ENHANCED_ONSET_BASE": "0.65"},
                    "Aggressive recall setting; useful only if precision survives.",
                ),
                _candidate(
                    "enhanced_onset_085",
                    {"LIVE_ENHANCED_ONSET_BASE": "0.85"},
                    "Aggressive precision setting; useful only if recall survives.",
                ),
                _candidate(
                    "group_base_020",
                    {"LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.020"},
                    "Stricter onset grouping to reduce accidental chord gluing.",
                ),
                _candidate(
                    "group_base_040",
                    {"LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC": "0.040"},
                    "Looser onset grouping to reduce broken chords.",
                ),
                _candidate(
                    "group_prune_on",
                    {
                        "LIVE_NEURAL_GROUP_PRUNE_ENABLED": "1",
                        "LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE": "3",
                        "LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET": "0.55",
                        "LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO": "0.55",
                    },
                    "Prune low-onset-confidence outliers inside large same-onset groups.",
                ),
                _candidate(
                    "recall_plus_dup",
                    {
                        "LIVE_ENHANCED_ONSET_BASE": "0.70",
                        "LIVE_ENHANCED_DUPLICATE_WINDOW_SEC": "0.06",
                    },
                    "Recall-oriented onset threshold with duplicate cleanup.",
                ),
                _candidate(
                    "recall_plus_harmonic_filter",
                    {
                        "LIVE_ENHANCED_ONSET_BASE": "0.70",
                        "LIVE_ENHANCED_FILTER_HARMONICS": "1",
                    },
                    "Recall-oriented onset threshold protected by harmonic filtering.",
                ),
            ]
        )

    return candidates


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _strict_f1(aggregate: Mapping[str, Any], label: str) -> float:
    timing = aggregate.get("timing") or {}
    strict = timing.get("strict_onset") or {}
    return _as_float((strict.get(label) or {}).get("f1"))


def flatten_arm_metrics(aggregate: Mapping[str, Any]) -> Dict[str, float]:
    note = aggregate.get("note") or {}
    duration = aggregate.get("duration") or {}
    cluster = aggregate.get("cluster") or {}
    timing = aggregate.get("timing") or {}
    stability = aggregate.get("stability") or {}
    duplicates = aggregate.get("duplicates") or {}
    return {
        "clips": _as_float(aggregate.get("clips")),
        "display_note_precision": _as_float(note.get("precision")),
        "display_note_recall": _as_float(note.get("recall")),
        "display_note_f1": _as_float(note.get("f1")),
        "display_note_missing": _as_float(note.get("missing")),
        "display_note_extra": _as_float(note.get("extra")),
        "display_offset_f1": _as_float(duration.get("f1")),
        "display_cluster_f1": _as_float(cluster.get("f1")),
        "display_cluster_avg_jaccard": _as_float(cluster.get("avg_jaccard")),
        "display_cluster_onset_recall": _as_float(cluster.get("onset_alignment_recall")),
        "display_cluster_overclustered": _as_float(cluster.get("overclustered")),
        "display_cluster_underclustered": _as_float(cluster.get("underclustered")),
        "display_cluster_pitch_conflicts": _as_float(cluster.get("pitch_conflicts")),
        "strict_onset_10ms_f1": _strict_f1(aggregate, "10ms"),
        "strict_onset_20ms_f1": _strict_f1(aggregate, "20ms"),
        "strict_onset_30ms_f1": _strict_f1(aggregate, "30ms"),
        "boundary_recall": _as_float(timing.get("boundary_recall")),
        "time_to_visible_median_ms": _as_float(stability.get("time_to_visible_median_ms_weighted")),
        "stabilization_median_ms": _as_float(stability.get("stabilization_latency_median_ms_weighted")),
        "duplicates_per_100_display_notes": _as_float(duplicates.get("per_100_display_notes")),
    }


def summarize_result_file(result_path: Path) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or {}
    arms: Dict[str, Dict[str, Any]] = {}
    for arm in ("control", "treatment", "retro_correction"):
        if any(isinstance(entry, Mapping) and isinstance(entry.get(arm), Mapping) for entry in clips.values()):
            arms[arm] = flatten_arm_metrics(aggregate_arm(clips, arm))

    for arm_name, metrics in arms.items():
        p95_values = []
        avg_rtf_values = []
        for clip in clips.values():
            arm_payload = (clip or {}).get(arm_name) or {}
            p95_values.append(_as_float(arm_payload.get("p95_chunk_total_ms")))
            avg_rtf_values.append(_as_float(arm_payload.get("avg_real_time_factor")))
        metrics["mean_clip_p95_chunk_total_ms"] = (
            sum(p95_values) / len(p95_values) if p95_values else 0.0
        )
        metrics["mean_clip_avg_real_time_factor"] = (
            sum(avg_rtf_values) / len(avg_rtf_values) if avg_rtf_values else 0.0
        )

    return {
        "result_json": str(result_path),
        "config": payload.get("config") or {},
        "arms": arms,
    }


def metric_delta(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: _as_float(metrics.get(key)) - _as_float(baseline.get(key))
        for key in metrics.keys()
        if isinstance(metrics.get(key), (int, float))
    }


def run_candidate(
    candidate: Candidate,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    result_path = output_dir / f"{candidate['name']}.json"
    log_path = output_dir / f"{candidate['name']}.log"
    command = [
        sys.executable,
        str(ROOT / "test_experiment.py"),
        "--benchmark-manifest",
        str(Path(args.benchmark_manifest).resolve()),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--noise-profile",
        args.noise_profile,
        "--run-retro-correction",
        "false",
        "--output-json",
        str(result_path),
    ]
    if args.no_warmup:
        command.append("--no-warmup")
    if args.cluster_metric_slot_consensus:
        command.append("--cluster-metric-slot-consensus")
    if args.cluster_metric_grid_snap:
        command.append("--cluster-metric-grid-snap")
    if args.clip_ids:
        command.extend(["--clip-ids", *args.clip_ids])

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (candidate.get("env") or {}).items()})

    if args.dry_run:
        return {
            "candidate": candidate,
            "command": command,
            "result_json": str(result_path),
            "log": str(log_path),
            "status": "dry_run",
        }

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# Candidate: {candidate['name']}\n")
        log_handle.write(f"# Env: {json.dumps(candidate.get('env') or {}, sort_keys=True)}\n")
        log_handle.write(f"# Command: {' '.join(command)}\n\n")
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    entry: Dict[str, Any] = {
        "candidate": candidate,
        "command": command,
        "result_json": str(result_path),
        "log": str(log_path),
        "returncode": process.returncode,
        "status": "ok" if process.returncode == 0 and result_path.exists() else "failed",
    }
    if entry["status"] == "ok":
        entry.update(summarize_result_file(result_path))
    return entry


def select_candidates(args: argparse.Namespace) -> List[Candidate]:
    candidates = build_candidates(args.preset)
    if args.only:
        requested = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["name"] in requested]
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    return candidates


def rank_entries(entries: Iterable[Mapping[str, Any]], arm: str, latency_tolerance_ms: float) -> List[Dict[str, Any]]:
    ok_entries = [entry for entry in entries if entry.get("status") == "ok"]
    if not ok_entries:
        return []

    baseline_entry = next(
        (entry for entry in ok_entries if (entry.get("candidate") or {}).get("name") == "baseline_current"),
        ok_entries[0],
    )
    baseline_metrics = ((baseline_entry.get("arms") or {}).get(arm) or {})
    baseline_latency = _as_float(baseline_metrics.get("mean_clip_p95_chunk_total_ms"))

    rows: List[Dict[str, Any]] = []
    for entry in ok_entries:
        metrics = ((entry.get("arms") or {}).get(arm) or {})
        latency = _as_float(metrics.get("mean_clip_p95_chunk_total_ms"))
        row = {
            "name": (entry.get("candidate") or {}).get("name"),
            "env": (entry.get("candidate") or {}).get("env") or {},
            "notes": (entry.get("candidate") or {}).get("notes") or "",
            "metrics": metrics,
            "delta_vs_baseline": metric_delta(metrics, baseline_metrics),
            "latency_ok": latency <= baseline_latency + latency_tolerance_ms,
            "result_json": entry.get("result_json"),
            "log": entry.get("log"),
        }
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            bool(row["latency_ok"]),
            _as_float(row["metrics"].get("display_note_f1")),
            _as_float(row["metrics"].get("display_note_recall")),
            _as_float(row["metrics"].get("display_cluster_avg_jaccard")),
            _as_float(row["metrics"].get("display_cluster_f1")),
            -_as_float(row["metrics"].get("duplicates_per_100_display_notes")),
        ),
        reverse=True,
    )


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    ranked = summary.get("ranked") or []
    lines = [
        "# Decoder Tuning Sweep",
        "",
        f"Generated: {summary.get('generated_at')}",
        "",
        "This ranks decoder-only settings. It uses score submetrics directly; there is no combined score-quality number.",
        "",
        "| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Dup/100 | p95 chunk ms |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked, start=1):
        metrics = row.get("metrics") or {}
        lines.append(
            "| "
            f"{index} | {row.get('name')} | {row.get('latency_ok')} | "
            f"{_as_float(metrics.get('display_note_f1')):.4f} | "
            f"{_as_float(metrics.get('display_note_recall')):.4f} | "
            f"{_as_float(metrics.get('display_note_precision')):.4f} | "
            f"{_as_float(metrics.get('display_cluster_f1')):.4f} | "
            f"{_as_float(metrics.get('display_cluster_avg_jaccard')):.4f} | "
            f"{_as_float(metrics.get('strict_onset_20ms_f1')):.4f} | "
            f"{_as_float(metrics.get('duplicates_per_100_display_notes')):.2f} | "
            f"{_as_float(metrics.get('mean_clip_p95_chunk_total_ms')):.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep zero-latency live decoder settings.")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--benchmark-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--clip-ids", nargs="+", default=[])
    parser.add_argument("--chunk-seconds", type=float, default=0.6)
    parser.add_argument("--noise-profile", choices=["open", "balanced", "clean"], default="balanced")
    parser.add_argument("--cluster-metric-slot-consensus", action="store_true")
    parser.add_argument("--cluster-metric-grid-snap", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--arm", choices=["control", "treatment"], default="treatment")
    parser.add_argument("--latency-tolerance-ms", type=float, default=2.0)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if args.cluster_metric_slot_consensus and args.cluster_metric_grid_snap:
        parser.error("--cluster-metric-slot-consensus and --cluster-metric-grid-snap cannot both be used")
    return args


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / f"decoder_tuning_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = select_candidates(args)
    if not candidates:
        raise SystemExit("No candidates selected.")

    entries = []
    print(f"Running {len(candidates)} decoder candidates -> {output_dir}")
    for candidate in candidates:
        print(f"[decoder-tune] {candidate['name']}")
        entries.append(run_candidate(candidate, args, output_dir))

    ranked = rank_entries(entries, args.arm, args.latency_tolerance_ms)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "arm": args.arm,
        "latency_tolerance_ms": args.latency_tolerance_ms,
        "benchmark_manifest": str(Path(args.benchmark_manifest).resolve()),
        "clip_ids": list(args.clip_ids),
        "entries": entries,
        "ranked": ranked,
    }
    summary_json = output_dir / "decoder_tuning_summary.json"
    summary_md = output_dir / "decoder_tuning_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, summary_md)

    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary Markdown: {summary_md}")
    if ranked:
        best = ranked[0]
        metrics = best.get("metrics") or {}
        print(
            "Best candidate: "
            f"{best.get('name')} "
            f"note_f1={_as_float(metrics.get('display_note_f1')):.4f} "
            f"recall={_as_float(metrics.get('display_note_recall')):.4f} "
            f"cluster_f1={_as_float(metrics.get('display_cluster_f1')):.4f} "
            f"p95_chunk_ms={_as_float(metrics.get('mean_clip_p95_chunk_total_ms')):.2f}"
        )


if __name__ == "__main__":
    main()
