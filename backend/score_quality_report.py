"""Generate a score-quality diagnostic report from a test_experiment result JSON.

This intentionally sits one layer above the individual benchmark metrics. The
goal is to answer: "why does the generated score feel wrong?" rather than only
report one F1 number.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


DEFAULT_ARMS = ("control", "treatment")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _weighted_mean(items: Iterable[Tuple[float, float]]) -> float:
    total_weight = 0.0
    total = 0.0
    for value, weight in items:
        if weight <= 0:
            continue
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _clip_label(clip_id: str, clip_entry: Mapping[str, Any]) -> str:
    clip = clip_entry.get("clip") or {}
    title = clip.get("title") or clip.get("piece_id") or ""
    if title:
        return f"{clip_id} - {title}"
    return clip_id


def _strict_onset_summary(clip_metrics: List[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    labels = sorted(
        {
            str(label)
            for metrics in clip_metrics
            for label in (metrics.get("display_strict_onset_metrics") or {}).keys()
        },
        key=lambda label: int(label.rstrip("ms")) if label.endswith("ms") and label[:-2].isdigit() else 9999,
    )
    result: Dict[str, Dict[str, float]] = {}
    for label in labels:
        matched = predicted = ground_truth = 0.0
        for metrics in clip_metrics:
            strict = (metrics.get("display_strict_onset_metrics") or {}).get(label) or {}
            matched += _num(strict.get("matched"))
            predicted += _num(strict.get("predicted"))
            ground_truth += _num(strict.get("ground_truth"))
        precision = _safe_div(matched, predicted)
        recall = _safe_div(matched, ground_truth)
        result[label] = {
            "matched": matched,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    return result


def _needed_exact_matches_for_f1(target_f1: float, predicted: float, ground_truth: float) -> int | None:
    if predicted <= 0 or ground_truth <= 0 or target_f1 <= 0:
        return None
    # F1 = 2x / (predicted + ground_truth), where x is exact cluster matches.
    needed = math.ceil(target_f1 * (predicted + ground_truth) / 2.0)
    if needed > min(predicted, ground_truth):
        return None
    return int(needed)


def aggregate_arm(clips: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    clip_metrics: List[Mapping[str, Any]] = [
        clip_entry[arm]
        for clip_entry in clips.values()
        if isinstance(clip_entry, Mapping) and isinstance(clip_entry.get(arm), Mapping)
    ]

    matched = sum(_num(m.get("display_note_matched")) for m in clip_metrics)
    predicted = sum(_num(m.get("display_note_predicted")) for m in clip_metrics)
    ground_truth = sum(_num(m.get("display_note_ground_truth")) for m in clip_metrics)
    note_precision = _safe_div(matched, predicted)
    note_recall = _safe_div(matched, ground_truth)

    offset_matched = sum(_num(m.get("display_offset_matched")) for m in clip_metrics)
    offset_precision = _safe_div(offset_matched, predicted)
    offset_recall = _safe_div(offset_matched, ground_truth)

    note_value_evaluable = sum(_num(m.get("display_note_value_matched")) for m in clip_metrics)
    note_value_exact = sum(
        _num(m.get("display_note_value_accuracy")) * _num(m.get("display_note_value_matched"))
        for m in clip_metrics
    )

    cluster_exact = sum(_num(m.get("display_cluster_exact_matches")) for m in clip_metrics)
    cluster_predicted = sum(_num(m.get("display_cluster_predicted")) for m in clip_metrics)
    cluster_ground_truth = sum(_num(m.get("display_cluster_ground_truth")) for m in clip_metrics)
    cluster_precision = _safe_div(cluster_exact, cluster_predicted)
    cluster_recall = _safe_div(cluster_exact, cluster_ground_truth)
    onset_aligned = sum(_num(m.get("display_cluster_onset_aligned_matches")) for m in clip_metrics)
    cluster_jaccard = _weighted_mean(
        (
            _num(m.get("display_cluster_avg_jaccard")),
            max(1.0, _num(m.get("display_cluster_onset_aligned_matches"))),
        )
        for m in clip_metrics
    )

    boundary_gt = sum(_num(m.get("boundary_gt_notes")) for m in clip_metrics)
    boundary_missed = sum(_num(m.get("boundary_missed_notes")) for m in clip_metrics)
    duplicates = sum(_num(m.get("duplicates")) for m in clip_metrics)
    final_display_notes = sum(_num(m.get("display_final_note_event_count")) for m in clip_metrics)

    stability_weighted = [
        (m, max(1.0, _num(m.get("matched_display_notes"))))
        for m in clip_metrics
    ]
    time_to_visible_median = _weighted_mean(
        (_num(m.get("time_to_visible_median_ms")), weight)
        for m, weight in stability_weighted
    )
    stabilization_median = _weighted_mean(
        (_num(m.get("stabilization_latency_median_ms")), weight)
        for m, weight in stability_weighted
    )
    avg_revision_count = _weighted_mean(
        (_num(m.get("avg_revision_count")), weight)
        for m, weight in stability_weighted
    )
    max_revision_count = max((_num(m.get("max_revision_count")) for m in clip_metrics), default=0.0)

    strict_onset = _strict_onset_summary(clip_metrics)
    exact_for_080 = _needed_exact_matches_for_f1(0.80, cluster_predicted, cluster_ground_truth)
    exact_gap_to_080 = None if exact_for_080 is None else max(0, exact_for_080 - int(cluster_exact))

    note_f1 = _f1(note_precision, note_recall)
    offset_f1 = _f1(offset_precision, offset_recall)
    note_value_accuracy = _safe_div(note_value_exact, note_value_evaluable)
    cluster_f1 = _f1(cluster_precision, cluster_recall)
    boundary_recall = 1.0 - _safe_div(boundary_missed, boundary_gt)
    duplicate_rate = _safe_div(duplicates * 100.0, final_display_notes)
    return {
        "clips": len(clip_metrics),
        "note": {
            "matched": matched,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "missing": max(0.0, ground_truth - matched),
            "extra": max(0.0, predicted - matched),
            "precision": note_precision,
            "recall": note_recall,
            "f1": note_f1,
        },
        "duration": {
            "offset_matched": offset_matched,
            "duration_or_offset_errors_among_note_matches": max(0.0, matched - offset_matched),
            "precision": offset_precision,
            "recall": offset_recall,
            "f1": offset_f1,
        },
        "note_value": {
            "evaluable": note_value_evaluable,
            "exact": note_value_exact,
            "errors": max(0.0, note_value_evaluable - note_value_exact),
            "accuracy": note_value_accuracy,
        },
        "cluster": {
            "exact_matches": cluster_exact,
            "predicted": cluster_predicted,
            "ground_truth": cluster_ground_truth,
            "precision": cluster_precision,
            "recall": cluster_recall,
            "f1": cluster_f1,
            "onset_aligned_matches": onset_aligned,
            "onset_alignment_precision": _safe_div(onset_aligned, cluster_predicted),
            "onset_alignment_recall": _safe_div(onset_aligned, cluster_ground_truth),
            "avg_jaccard": cluster_jaccard,
            "overclustered": sum(_num(m.get("display_cluster_overclustered_matches")) for m in clip_metrics),
            "underclustered": sum(_num(m.get("display_cluster_underclustered_matches")) for m in clip_metrics),
            "pitch_conflicts": sum(_num(m.get("display_cluster_pitch_conflict_matches")) for m in clip_metrics),
            "unmatched_predicted": sum(_num(m.get("display_cluster_unmatched_predicted")) for m in clip_metrics),
            "unmatched_ground_truth": sum(_num(m.get("display_cluster_unmatched_ground_truth")) for m in clip_metrics),
            "exact_matches_needed_for_0_80_f1": exact_for_080,
            "additional_exact_matches_needed_for_0_80_f1": exact_gap_to_080,
        },
        "timing": {
            "strict_onset": strict_onset,
            "boundary_gt": boundary_gt,
            "boundary_missed": boundary_missed,
            "boundary_recall": boundary_recall,
        },
        "stability": {
            "time_to_visible_median_ms_weighted": time_to_visible_median,
            "stabilization_latency_median_ms_weighted": stabilization_median,
            "avg_revision_count_weighted": avg_revision_count,
            "max_revision_count": max_revision_count,
        },
        "duplicates": {
            "count": duplicates,
            "per_100_display_notes": duplicate_rate,
        },
    }


def rank_clips(clips: Mapping[str, Any], arm: str, limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for clip_id, clip_entry in clips.items():
        if not isinstance(clip_entry, Mapping) or not isinstance(clip_entry.get(arm), Mapping):
            continue
        m = clip_entry[arm]
        rows.append(
            {
                "clip_id": clip_id,
                "label": _clip_label(str(clip_id), clip_entry),
                "display_note_f1": _num(m.get("display_note_f1")),
                "display_offset_f1": _num(m.get("display_offset_f1")),
                "display_cluster_f1": _num(m.get("display_cluster_f1")),
                "cluster_jaccard": _num(m.get("display_cluster_avg_jaccard")),
                "missing_notes": max(0.0, _num(m.get("display_note_ground_truth")) - _num(m.get("display_note_matched"))),
                "extra_notes": max(0.0, _num(m.get("display_note_predicted")) - _num(m.get("display_note_matched"))),
                "underclustered": _num(m.get("display_cluster_underclustered_matches")),
                "overclustered": _num(m.get("display_cluster_overclustered_matches")),
                "pitch_conflicts": _num(m.get("display_cluster_pitch_conflict_matches")),
                "boundary_missed": _num(m.get("boundary_missed_notes")),
                "avg_revision_count": _num(m.get("avg_revision_count")),
                "max_revision_count": _num(m.get("max_revision_count")),
                "stabilization_latency_median_ms": _num(m.get("stabilization_latency_median_ms")),
            }
        )

    return {
        "worst_note_f1": sorted(rows, key=lambda row: row["display_note_f1"])[:limit],
        "worst_cluster_f1": sorted(rows, key=lambda row: row["display_cluster_f1"])[:limit],
        "most_missing_notes": sorted(rows, key=lambda row: row["missing_notes"], reverse=True)[:limit],
        "most_extra_notes": sorted(rows, key=lambda row: row["extra_notes"], reverse=True)[:limit],
        "most_unstable": sorted(
            rows,
            key=lambda row: (row["avg_revision_count"], row["stabilization_latency_median_ms"]),
            reverse=True,
        )[:limit],
    }


def build_report(results: Mapping[str, Any], arms: Iterable[str], clip_limit: int = 8) -> Dict[str, Any]:
    clips = results.get("clips") or {}
    if not isinstance(clips, Mapping):
        raise ValueError("Result JSON does not contain a clips object")

    arm_reports = {}
    for arm in arms:
        arm_reports[arm] = {
            "aggregate": aggregate_arm(clips, arm),
            "clip_rankings": rank_clips(clips, arm, limit=clip_limit),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_config": results.get("config", {}),
        "metric_note": (
            "display_cluster_f1 is strict exact onset-cluster F1. It measures whether each "
            "onset-aligned pitch set exactly matches ground truth; it is useful for chord "
            "exactness, but it is not by itself a full generated-score accuracy metric."
        ),
        "arms": arm_reports,
    }


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any], source_path: Path) -> str:
    lines: List[str] = []
    lines.append("# Score Quality Report")
    lines.append("")
    lines.append(f"Source: `{source_path}`")
    lines.append("")
    lines.append(str(report["metric_note"]))
    lines.append("")

    summary_rows = []
    for arm, arm_report in report["arms"].items():
        agg = arm_report["aggregate"]
        summary_rows.append(
            [
                arm,
                _fmt(agg["note"]["f1"]),
                _fmt(agg["duration"]["f1"]),
                _pct(agg["note_value"]["accuracy"]),
                _fmt(agg["cluster"]["f1"]),
                _fmt(agg["cluster"]["avg_jaccard"]),
                _fmt(agg["stability"]["avg_revision_count_weighted"]),
            ]
        )
    lines.append("## Summary")
    lines.append("")
    lines.append(
        _md_table(
            [
                "arm",
                "note_f1",
                "offset_f1",
                "note_value_acc",
                "exact_cluster_f1",
                "cluster_jaccard",
                "avg_revisions",
            ],
            summary_rows,
        )
    )
    lines.append("")

    for arm, arm_report in report["arms"].items():
        agg = arm_report["aggregate"]
        lines.append(f"## {arm.title()}")
        lines.append("")
        lines.append("### Counts")
        lines.append("")
        lines.append(
            _md_table(
                ["area", "matched/exact", "predicted/evaluable", "ground truth", "errors"],
                [
                    [
                        "notes",
                        str(int(agg["note"]["matched"])),
                        str(int(agg["note"]["predicted"])),
                        str(int(agg["note"]["ground_truth"])),
                        f"missing {int(agg['note']['missing'])}, extra {int(agg['note']['extra'])}",
                    ],
                    [
                        "durations",
                        str(int(agg["duration"]["offset_matched"])),
                        str(int(agg["note"]["predicted"])),
                        str(int(agg["note"]["ground_truth"])),
                        str(int(agg["duration"]["duration_or_offset_errors_among_note_matches"])),
                    ],
                    [
                        "note values",
                        str(round(agg["note_value"]["exact"], 1)),
                        str(int(agg["note_value"]["evaluable"])),
                        "-",
                        str(round(agg["note_value"]["errors"], 1)),
                    ],
                    [
                        "clusters",
                        str(int(agg["cluster"]["exact_matches"])),
                        str(int(agg["cluster"]["predicted"])),
                        str(int(agg["cluster"]["ground_truth"])),
                        (
                            f"under {int(agg['cluster']['underclustered'])}, "
                            f"over {int(agg['cluster']['overclustered'])}, "
                            f"pitch {int(agg['cluster']['pitch_conflicts'])}"
                        ),
                    ],
                ],
            )
        )
        lines.append("")

        needed = agg["cluster"].get("additional_exact_matches_needed_for_0_80_f1")
        if needed is None:
            lines.append("Exact cluster F1 cannot reach `0.80` with the current predicted/ground-truth cluster counts alone.")
        else:
            lines.append(
                f"Exact cluster F1 needs about `{needed}` additional exact cluster matches to reach `0.80`, "
                "assuming predicted/ground-truth cluster counts stay fixed."
            )
        lines.append("")

        lines.append("### Timing And Stability")
        lines.append("")
        lines.append(
            _md_table(
                ["metric", "value"],
                [
                    ["boundary recall", _pct(agg["timing"]["boundary_recall"])],
                    ["weighted time-to-visible median", f"{agg['stability']['time_to_visible_median_ms_weighted']:.0f} ms"],
                    ["weighted stabilization median", f"{agg['stability']['stabilization_latency_median_ms_weighted']:.0f} ms"],
                    ["weighted avg revisions", _fmt(agg["stability"]["avg_revision_count_weighted"])],
                    ["max revisions", _fmt(agg["stability"]["max_revision_count"])],
                    ["duplicates / 100 notes", _fmt(agg["duplicates"]["per_100_display_notes"])],
                ],
            )
        )
        lines.append("")

        lines.append("### Worst Cluster Clips")
        lines.append("")
        cluster_rows = []
        for row in arm_report["clip_rankings"]["worst_cluster_f1"][:5]:
            cluster_rows.append(
                [
                    row["clip_id"],
                    _fmt(row["display_cluster_f1"]),
                    _fmt(row["cluster_jaccard"]),
                    str(int(row["underclustered"])),
                    str(int(row["overclustered"])),
                    str(int(row["pitch_conflicts"])),
                ]
            )
        lines.append(_md_table(["clip", "cluster_f1", "jaccard", "under", "over", "pitch"], cluster_rows))
        lines.append("")

        lines.append("### Most Unstable Clips")
        lines.append("")
        unstable_rows = []
        for row in arm_report["clip_rankings"]["most_unstable"][:5]:
            unstable_rows.append(
                [
                    row["clip_id"],
                    _fmt(row["avg_revision_count"]),
                    _fmt(row["max_revision_count"]),
                    f"{row['stabilization_latency_median_ms']:.0f} ms",
                    _fmt(row["display_note_f1"]),
                ]
            )
        lines.append(_md_table(["clip", "avg revisions", "max revisions", "stabilization", "note_f1"], unstable_rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--clip-limit", type=int, default=8)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    report = build_report(results, arms=args.arms, clip_limit=args.clip_limit)

    out_json = args.out_json or args.results_json.with_name(args.results_json.stem + "_score_quality_report.json")
    out_md = args.out_md or args.results_json.with_name(args.results_json.stem + "_score_quality_report.md")

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report, args.results_json), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    for arm, arm_report in report["arms"].items():
        agg = arm_report["aggregate"]
        print(
            f"{arm}: note_f1={agg['note']['f1']:.4f} "
            f"offset_f1={agg['duration']['f1']:.4f} "
            f"cluster_f1={agg['cluster']['f1']:.4f} "
            f"cluster_jaccard={agg['cluster']['avg_jaccard']:.4f}"
        )


if __name__ == "__main__":
    main()
