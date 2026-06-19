r"""Sweep the continuous live-stream hypothesis decoder.

Unlike tune_decoder_settings.py, this exercises ContinuousLiveStreamSession:
audio is replayed as small PCM packets, inference runs on rolling context, and
the same candidate/active/committed state machine used by the WebSocket path is
measured directly.

Example:
    .\env\Scripts\python.exe tune_continuous_stream_decoder.py --preset quick --clip-ids clip_001 clip_002
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from contextlib import contextmanager, nullcontext, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np

import main as live_main
from main import ContinuousLiveStreamSession
from test_experiment import (
    TARGET_SR,
    compute_boundary_miss_metrics,
    compute_duplicate_metrics,
    compute_note_metrics,
    compute_offset_metrics,
    compute_onset_cluster_metrics,
    compute_onset_tolerance_sweep,
    get_midi_reference_bpm,
    load_audio_excerpt,
    load_midi_notes,
    safe_percentile,
    slice_gt_notes,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "live_benchmark_replay_auto_v2.json"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark_artifacts"


Candidate = Dict[str, Any]


def _candidate(
    name: str,
    attrs: Mapping[str, Any],
    notes: str,
    env: Mapping[str, str] | None = None,
) -> Candidate:
    return {"name": name, "attrs": dict(attrs), "notes": notes, "env": dict(env or {})}


def build_candidates(preset: str) -> List[Candidate]:
    candidates = [
        _candidate("baseline_current", {}, "Current continuous stream decoder settings."),
        _candidate(
            "gates_on",
            {"STREAM_RMS_BIRTH_GATES": True},
            "Legacy RMS-attack birth gates re-enabled (pre-2026-06-12 behaviour).",
        ),
        _candidate(
            "rescue_020",
            {"STREAM_ATTACK_GROUP_RESCUE_SEC": 0.20},
            "Slightly tighter weak-birth rescue window.",
        ),
        _candidate(
            "rescue_025",
            {"STREAM_ATTACK_GROUP_RESCUE_SEC": 0.25},
            "Current 250ms rescue window.",
        ),
        _candidate(
            "rescue_030",
            {"STREAM_ATTACK_GROUP_RESCUE_SEC": 0.30},
            "Looser rescue window for middle/pedaled melody recall.",
        ),
        _candidate(
            "rescue_035",
            {"STREAM_ATTACK_GROUP_RESCUE_SEC": 0.35},
            "Aggressive rescue window; checks false-positive cost.",
        ),
        _candidate(
            "rescue_min_conf_040",
            {"STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE": 0.40},
            "Let weaker non-chord observations birth notes near attack groups.",
        ),
        _candidate(
            "rescue_min_conf_060",
            {"STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE": 0.60},
            "Require stronger confidence for non-chord rescue births.",
        ),
        _candidate(
            "repeat_120ms",
            {"STREAM_MIN_REPEAT_SEC": 0.12},
            "Less same-pitch repeat suppression for ornaments/rearticulations.",
        ),
        _candidate(
            "repeat_200ms",
            {"STREAM_MIN_REPEAT_SEC": 0.20},
            "More same-pitch repeat suppression for pedal/ring-out false rebirths.",
        ),
        _candidate(
            "onset_055",
            {},
            "Lower enhanced onset threshold; persistence gate handles extra noise.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.55"},
        ),
        _candidate(
            "onset_060",
            {},
            "Explicit current/reference enhanced onset threshold.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.60"},
        ),
        _candidate(
            "onset_065",
            {},
            "Moderately stricter enhanced onset threshold.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.65"},
        ),
        _candidate(
            "onset_070",
            {},
            "Stricter enhanced onset threshold for precision.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.70"},
        ),
        _candidate(
            "onset_075",
            {},
            "Old enhanced-onset operating point used by prior heuristic sweeps.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.75"},
        ),
        _candidate(
            "onset_080",
            {},
            "High-precision enhanced onset threshold.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.80"},
        ),
        _candidate(
            "onset_085",
            {},
            "Aggressive precision threshold; checks recall collapse.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.85"},
        ),
        _candidate(
            "onset_090",
            {},
            "Very strict threshold; likely recall-limited.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.90"},
        ),
        _candidate(
            "onset_050",
            {},
            "Lower enhanced onset threshold further for recall.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.50"},
        ),
        _candidate(
            "onset_045",
            {},
            "Aggressive recall threshold; relies on persistence gate for precision.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.45"},
        ),
        _candidate(
            "birth_gates_off",
            {"STREAM_ATTACK_RATIO_STRONG": 0.0, "STREAM_ATTACK_DELTA_STRONG": -1.0},
            "Treat every observation as strong-attack so RMS birth gates never fire; persistence gate is the only noise filter.",
        ),
        _candidate(
            "display_obs_2",
            {"STREAM_MIN_DISPLAY_OBSERVATIONS": 2},
            "Allow display after 2 observations instead of 3.",
        ),
        _candidate(
            "frame_evidence_50ms",
            {"STREAM_FRAME_EVIDENCE_SEC": 0.05, "STREAM_DISPLAY_FRAME_EVIDENCE_SEC": 0.10},
            "Lower frame-evidence birth/display sustain requirements for short notes.",
        ),
        _candidate(
            "ship_default",
            {},
            "New defaults: RMS birth gates off (persistence display gate only).",
        ),
        _candidate(
            "ship_legacy_gates",
            {"STREAM_RMS_BIRTH_GATES": True},
            "Old behavior reference: RMS birth gates on.",
        ),
        _candidate(
            "ship_obs2",
            {"STREAM_MIN_DISPLAY_OBSERVATIONS": 2},
            "Gates-off default plus 2-observation display gate.",
        ),
        _candidate(
            "ship_disp_frame_100ms",
            {"STREAM_DISPLAY_FRAME_EVIDENCE_SEC": 0.10},
            "Gates-off default plus 100ms display frame-evidence sustain.",
        ),
        _candidate(
            "ship_onset055",
            {},
            "Gates-off default plus lower decode onset threshold.",
            env={"LIVE_ENHANCED_ONSET_BASE": "0.55"},
        ),
    ]

    if preset == "full":
        candidates.extend(
            [
                _candidate(
                    "weak_birth_hi_conf_080",
                    {"STREAM_WEAK_BIRTH_HIGH_CONFIDENCE": 0.80},
                    "Allow more high-confidence weak births outside attack groups.",
                ),
                _candidate(
                    "weak_birth_hi_conf_092",
                    {"STREAM_WEAK_BIRTH_HIGH_CONFIDENCE": 0.92},
                    "Block almost all weak births outside attack groups.",
                ),
                _candidate(
                    "harmonic_conf_070",
                    {"STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE": 0.70},
                    "Suppress fewer harmonics; may recover melody but admit pedal artifacts.",
                ),
                _candidate(
                    "harmonic_conf_085",
                    {"STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE": 0.85},
                    "Suppress more harmonic-looking weak births outside attack groups.",
                ),
                _candidate(
                    "boundary_220ms",
                    {"STREAM_CONTINUITY_BOUNDARY_SEC": 0.22},
                    "Narrow chunk-boundary same-pitch continuity suppression.",
                ),
                _candidate(
                    "boundary_400ms",
                    {"STREAM_CONTINUITY_BOUNDARY_SEC": 0.40},
                    "Widen chunk-boundary same-pitch continuity suppression.",
                ),
                _candidate(
                    "rescue_030_conf_040",
                    {
                        "STREAM_ATTACK_GROUP_RESCUE_SEC": 0.30,
                        "STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE": 0.40,
                    },
                    "Recall-oriented combination for weak melody near attacks.",
                ),
                _candidate(
                    "rescue_030_repeat_120",
                    {
                        "STREAM_ATTACK_GROUP_RESCUE_SEC": 0.30,
                        "STREAM_MIN_REPEAT_SEC": 0.12,
                    },
                    "Wider attack rescue while permitting faster same-pitch repeats.",
                ),
            ]
        )

    return candidates


@contextmanager
def override_live_attrs(attrs: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Iterator[None]:
    originals: Dict[str, Any] = {}
    env_originals: Dict[str, str | None] = {}
    try:
        for key, value in attrs.items():
            if not hasattr(live_main, key):
                raise AttributeError(f"backend.main has no tunable attribute {key!r}")
            originals[key] = getattr(live_main, key)
            setattr(live_main, key, value)
        for key, value in (env or {}).items():
            env_originals[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in originals.items():
            setattr(live_main, key, value)
        for key, value in env_originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_manifest(path: Path, clip_ids: Sequence[str]) -> Dict[str, Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or {}
    if clip_ids:
        wanted = set(clip_ids)
        clips = {clip_id: clip for clip_id, clip in clips.items() if clip_id in wanted}
    if not clips:
        raise RuntimeError("No benchmark clips selected.")
    return dict(sorted(clips.items(), key=lambda item: item[0]))


def _note_key(payload: Mapping[str, Any]) -> str:
    if payload.get("id") is not None:
        return f"id-{int(payload.get('id') or 0)}"
    onset_ms = int(round(float(payload.get("onset_time", 0.0) or 0.0) * 1000.0))
    return f"{int(payload.get('midi_note', 0) or 0)}-{onset_ms}"


def payload_to_note(payload: Mapping[str, Any]) -> Dict[str, Any]:
    onset = float(payload.get("onset_time", 0.0) or 0.0)
    offset = float(payload.get("offset_time", onset) or onset)
    if offset < onset:
        offset = onset
    return {
        "time_seconds": onset,
        "onset_time": onset,
        "offset_seconds": offset,
        "offset_time": offset,
        "duration_seconds": max(0.0, offset - onset),
        "duration": max(0.0, offset - onset),
        "midi_note": int(payload.get("midi_note", 0) or 0),
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "state": str(payload.get("state") or ""),
        "observations": int(payload.get("observations", 0) or 0),
    }


def visible_payloads(update: Mapping[str, Any], include_unstable: bool) -> List[Dict]:
    payloads: List[Dict] = []
    if include_unstable:
        payloads.extend(update.get("heard_notes") or [])
        payloads.extend(update.get("candidate_notes") or [])
    payloads.extend(update.get("committed_notes") or [])
    payloads.extend(update.get("locked_notes") or [])
    payloads.extend(update.get("active_notes") or [])
    return payloads


def update_accumulator(
    note_map: Dict[str, Dict],
    update: Mapping[str, Any],
    include_unstable: bool,
) -> None:
    for payload in visible_payloads(update, include_unstable=include_unstable):
        note_map[_note_key(payload)] = dict(payload)


def notes_from_accumulator(note_map: Mapping[str, Dict]) -> List[Dict]:
    notes = [payload_to_note(payload) for payload in note_map.values()]
    notes.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return notes


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize_notes(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    chunk_seconds: float,
    boundary_band_sec: float,
) -> Dict[str, Any]:
    note = compute_note_metrics(pred_notes, gt_notes)
    offset = compute_offset_metrics(pred_notes, gt_notes)
    cluster = compute_onset_cluster_metrics(pred_notes, gt_notes)
    boundary = compute_boundary_miss_metrics(
        pred_notes,
        gt_notes,
        chunk_seconds=chunk_seconds,
        boundary_band_sec=boundary_band_sec,
    )
    duplicates = compute_duplicate_metrics(pred_notes)
    strict = compute_onset_tolerance_sweep(pred_notes, gt_notes)
    return {
        "note": note,
        "offset": offset,
        "cluster": cluster,
        "boundary": boundary,
        "duplicates": duplicates,
        "strict_onset": strict,
    }


def flatten_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    note = summary.get("note") or {}
    offset = summary.get("offset") or {}
    cluster = summary.get("cluster") or {}
    boundary = summary.get("boundary") or {}
    duplicates = summary.get("duplicates") or {}
    strict = summary.get("strict_onset") or {}
    boundary_gt = float(boundary.get("boundary_gt_notes", 0.0) or 0.0)
    boundary_missed = float(boundary.get("boundary_missed_notes", 0.0) or 0.0)
    return {
        "note_precision": float(note.get("precision", 0.0) or 0.0),
        "note_recall": float(note.get("recall", 0.0) or 0.0),
        "note_f1": float(note.get("f1", 0.0) or 0.0),
        "note_matched": float(note.get("matched", 0.0) or 0.0),
        "note_predicted": float(note.get("predicted", 0.0) or 0.0),
        "note_ground_truth": float(note.get("ground_truth", 0.0) or 0.0),
        "offset_f1": float(offset.get("offset_f1", 0.0) or 0.0),
        "offset_matched": float(offset.get("offset_matched", 0.0) or 0.0),
        "cluster_f1": float(cluster.get("f1", 0.0) or 0.0),
        "cluster_precision": float(cluster.get("precision", 0.0) or 0.0),
        "cluster_recall": float(cluster.get("recall", 0.0) or 0.0),
        "cluster_jaccard": float(cluster.get("avg_jaccard", 0.0) or 0.0),
        "cluster_exact": float(cluster.get("exact_matches", 0.0) or 0.0),
        "cluster_predicted": float(cluster.get("predicted", 0.0) or 0.0),
        "cluster_ground_truth": float(cluster.get("ground_truth", 0.0) or 0.0),
        "cluster_onset_recall": float(cluster.get("onset_alignment_recall", 0.0) or 0.0),
        "cluster_overclustered": float(cluster.get("overclustered_matches", 0.0) or 0.0),
        "cluster_underclustered": float(cluster.get("underclustered_matches", 0.0) or 0.0),
        "cluster_pitch_conflicts": float(cluster.get("pitch_conflict_matches", 0.0) or 0.0),
        "strict_10ms_f1": float((strict.get("10ms") or {}).get("f1", 0.0) or 0.0),
        "strict_20ms_f1": float((strict.get("20ms") or {}).get("f1", 0.0) or 0.0),
        "strict_30ms_f1": float((strict.get("30ms") or {}).get("f1", 0.0) or 0.0),
        "boundary_recall": 1.0 - (boundary_missed / boundary_gt) if boundary_gt else 0.0,
        "duplicates_per_100": float(duplicates.get("duplicates_per_100_notes", 0.0) or 0.0),
    }


def aggregate_clip_summaries(clip_summaries: Iterable[Mapping[str, Any]], surface: str) -> Dict[str, float]:
    totals = {
        "note_matched": 0.0,
        "note_predicted": 0.0,
        "note_ground_truth": 0.0,
        "offset_matched": 0.0,
        "cluster_exact": 0.0,
        "cluster_predicted": 0.0,
        "cluster_ground_truth": 0.0,
        "cluster_onset": 0.0,
        "cluster_over": 0.0,
        "cluster_under": 0.0,
        "cluster_conflict": 0.0,
        "boundary_gt": 0.0,
        "boundary_missed": 0.0,
        "duplicates": 0.0,
        "notes": 0.0,
        "strict": {label: {"matched": 0.0, "predicted": 0.0, "ground_truth": 0.0} for label in ("10ms", "20ms", "30ms")},
        "jaccard_weighted": 0.0,
        "jaccard_weight": 0.0,
    }
    timing_values = []
    inference_values = []
    observation_values = []
    suppression_totals: Dict[str, float] = {}

    for clip in clip_summaries:
        metrics = ((clip.get("surfaces") or {}).get(surface) or {})
        note = metrics.get("note") or {}
        offset = metrics.get("offset") or {}
        cluster = metrics.get("cluster") or {}
        boundary = metrics.get("boundary") or {}
        duplicates = metrics.get("duplicates") or {}
        strict = metrics.get("strict_onset") or {}
        totals["note_matched"] += float(note.get("matched", 0.0) or 0.0)
        totals["note_predicted"] += float(note.get("predicted", 0.0) or 0.0)
        totals["note_ground_truth"] += float(note.get("ground_truth", 0.0) or 0.0)
        totals["offset_matched"] += float(offset.get("offset_matched", 0.0) or 0.0)
        totals["cluster_exact"] += float(cluster.get("exact_matches", 0.0) or 0.0)
        totals["cluster_predicted"] += float(cluster.get("predicted", 0.0) or 0.0)
        totals["cluster_ground_truth"] += float(cluster.get("ground_truth", 0.0) or 0.0)
        onset_aligned = float(cluster.get("onset_aligned_matches", 0.0) or 0.0)
        totals["cluster_onset"] += onset_aligned
        totals["cluster_over"] += float(cluster.get("overclustered_matches", 0.0) or 0.0)
        totals["cluster_under"] += float(cluster.get("underclustered_matches", 0.0) or 0.0)
        totals["cluster_conflict"] += float(cluster.get("pitch_conflict_matches", 0.0) or 0.0)
        totals["jaccard_weighted"] += float(cluster.get("avg_jaccard", 0.0) or 0.0) * max(1.0, onset_aligned)
        totals["jaccard_weight"] += max(1.0, onset_aligned)
        totals["boundary_gt"] += float(boundary.get("boundary_gt_notes", 0.0) or 0.0)
        totals["boundary_missed"] += float(boundary.get("boundary_missed_notes", 0.0) or 0.0)
        totals["duplicates"] += float(duplicates.get("duplicates", 0.0) or 0.0)
        totals["notes"] += float(note.get("predicted", 0.0) or 0.0)
        for label in totals["strict"]:
            strict_metrics = strict.get(label) or {}
            totals["strict"][label]["matched"] += float(strict_metrics.get("matched", 0.0) or 0.0)
            totals["strict"][label]["predicted"] += float(strict_metrics.get("predicted", 0.0) or 0.0)
            totals["strict"][label]["ground_truth"] += float(strict_metrics.get("ground_truth", 0.0) or 0.0)

        timing = clip.get("timing") or {}
        timing_values.extend(timing.get("neural_total_ms") or [])
        inference_values.extend(timing.get("inference_ms") or [])
        observation_values.extend(timing.get("observation_count") or [])
        for key, value in (clip.get("suppression_totals") or {}).items():
            suppression_totals[key] = suppression_totals.get(key, 0.0) + float(value or 0.0)

    note_precision = totals["note_matched"] / totals["note_predicted"] if totals["note_predicted"] else 0.0
    note_recall = totals["note_matched"] / totals["note_ground_truth"] if totals["note_ground_truth"] else 0.0
    offset_precision = totals["offset_matched"] / totals["note_predicted"] if totals["note_predicted"] else 0.0
    offset_recall = totals["offset_matched"] / totals["note_ground_truth"] if totals["note_ground_truth"] else 0.0
    cluster_precision = totals["cluster_exact"] / totals["cluster_predicted"] if totals["cluster_predicted"] else 0.0
    cluster_recall = totals["cluster_exact"] / totals["cluster_ground_truth"] if totals["cluster_ground_truth"] else 0.0
    result = {
        "note_precision": note_precision,
        "note_recall": note_recall,
        "note_f1": _f1(note_precision, note_recall),
        "note_matched": totals["note_matched"],
        "note_predicted": totals["note_predicted"],
        "note_ground_truth": totals["note_ground_truth"],
        "offset_f1": _f1(offset_precision, offset_recall),
        "cluster_f1": _f1(cluster_precision, cluster_recall),
        "cluster_precision": cluster_precision,
        "cluster_recall": cluster_recall,
        "cluster_jaccard": totals["jaccard_weighted"] / totals["jaccard_weight"] if totals["jaccard_weight"] else 0.0,
        "cluster_onset_recall": totals["cluster_onset"] / totals["cluster_ground_truth"] if totals["cluster_ground_truth"] else 0.0,
        "cluster_overclustered": totals["cluster_over"],
        "cluster_underclustered": totals["cluster_under"],
        "cluster_pitch_conflicts": totals["cluster_conflict"],
        "boundary_recall": 1.0 - (totals["boundary_missed"] / totals["boundary_gt"]) if totals["boundary_gt"] else 0.0,
        "duplicates_per_100": 100.0 * totals["duplicates"] / totals["notes"] if totals["notes"] else 0.0,
        "inference_ms_p50": safe_percentile(inference_values, 50),
        "inference_ms_p95": safe_percentile(inference_values, 95),
        "neural_total_ms_p50": safe_percentile(timing_values, 50),
        "neural_total_ms_p95": safe_percentile(timing_values, 95),
        "observations_mean": float(np.mean(np.asarray(observation_values, dtype=np.float64))) if observation_values else 0.0,
        "suppression_totals": dict(sorted(suppression_totals.items())),
    }
    for label, values in totals["strict"].items():
        precision = values["matched"] / values["predicted"] if values["predicted"] else 0.0
        recall = values["matched"] / values["ground_truth"] if values["ground_truth"] else 0.0
        result[f"strict_{label}_f1"] = _f1(precision, recall)
    return result


def run_continuous_replay(
    clip: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
    if args.tail_padding_sec > 0:
        audio = np.concatenate(
            [audio, np.zeros(int(round(args.tail_padding_sec * TARGET_SR)), dtype=np.float32)]
        )

    session = ContinuousLiveStreamSession(
        session_id=f"continuous-bench-{uuid.uuid4().hex}",
        sample_rate=TARGET_SR,
        context_sec=args.context_sec,
        inference_interval_sec=args.inference_interval_ms / 1000.0,
        trusted_delay_sec=args.trusted_delay_ms / 1000.0,
        commit_delay_sec=args.commit_delay_ms / 1000.0,
        lock_delay_sec=args.lock_delay_ms / 1000.0,
    )
    packet_frames = max(1, int(round(args.packet_ms * TARGET_SR / 1000.0)))
    score_payloads: Dict[str, Dict] = {}
    preview_payloads: Dict[str, Dict] = {}
    updates = 0
    inference_updates = 0
    inference_ms: List[float] = []
    neural_total_ms: List[float] = []
    observation_count: List[float] = []
    suppression_totals: Dict[str, int] = {}
    last_update: Dict[str, Any] | None = None

    for start in range(0, audio.size, packet_frames):
        packet = audio[start : start + packet_frames]
        session.append_audio(packet)
        update = session.maybe_run_inference()
        if update is None:
            continue
        updates += 1
        last_update = update
        update_accumulator(score_payloads, update, include_unstable=False)
        update_accumulator(preview_payloads, update, include_unstable=True)
        inference = update.get("inference") or {}
        if inference.get("ran"):
            inference_updates += 1
            inference_ms.append(float(inference.get("inference_ms", 0.0) or 0.0))
            neural = inference.get("neural_timing") or {}
            neural_total_ms.append(float(neural.get("neural_total", 0.0) or 0.0))
            observation_count.append(float(inference.get("observation_count", 0.0) or 0.0))
            continuity = inference.get("continuity_filter") or {}
            for key in (
                "suppressed",
                "same_pitch_boundary",
                "implausible_repeat",
                "harmonic_sustain",
                "weak_birth_outside_attack",
                "registered_attack_groups",
            ):
                suppression_totals[key] = suppression_totals.get(key, 0) + int(continuity.get(key, 0) or 0)

    final_update = session.maybe_run_inference(force=True)
    if final_update is not None:
        updates += 1
        last_update = final_update
        update_accumulator(score_payloads, final_update, include_unstable=False)
        update_accumulator(preview_payloads, final_update, include_unstable=True)
        inference = final_update.get("inference") or {}
        if inference.get("ran"):
            inference_updates += 1
            inference_ms.append(float(inference.get("inference_ms", 0.0) or 0.0))
            neural = inference.get("neural_timing") or {}
            neural_total_ms.append(float(neural.get("neural_total", 0.0) or 0.0))
            observation_count.append(float(inference.get("observation_count", 0.0) or 0.0))

    score_notes = notes_from_accumulator(score_payloads)
    preview_notes = notes_from_accumulator(preview_payloads)
    gt_notes = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
    return {
        "score_notes": score_notes,
        "preview_notes": preview_notes,
        "ground_truth_notes": len(gt_notes),
        "surfaces": {
            "score": summarize_notes(score_notes, gt_notes, args.chunk_seconds_for_boundary, args.eval_boundary_band_sec),
            "preview": summarize_notes(preview_notes, gt_notes, args.chunk_seconds_for_boundary, args.eval_boundary_band_sec),
        },
        "timing": {
            "updates": updates,
            "inference_updates": inference_updates,
            "inference_ms": inference_ms,
            "neural_total_ms": neural_total_ms,
            "observation_count": observation_count,
        },
        "suppression_totals": suppression_totals,
        "final_counts": (last_update or {}).get("counts") or {},
    }


def run_candidate(
    candidate: Candidate,
    clips: Mapping[str, Dict],
    args: argparse.Namespace,
    log_path: Path | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    clip_results: Dict[str, Any] = {}
    with override_live_attrs(candidate.get("attrs") or {}, candidate.get("env") or {}):
        log_handle = None
        try:
            if log_path is not None and not args.show_model_logs:
                log_handle = log_path.open("w", encoding="utf-8")
                log_handle.write(f"# Candidate: {candidate['name']}\n")
                log_handle.write(f"# Attrs: {json.dumps(candidate.get('attrs') or {}, sort_keys=True, default=str)}\n\n")
            for clip_id, clip in clips.items():
                print(f"  [{candidate['name']}] {clip_id}")
                log_context = redirect_stdout(log_handle) if log_handle is not None else nullcontext()
                with log_context:
                    clip_result = run_continuous_replay(clip, args)
                clip_results[clip_id] = {
                    "clip": {
                        "title": clip.get("title"),
                        "start_sec": clip.get("start_sec"),
                        "end_sec": clip.get("end_sec"),
                        "gt_note_count": clip.get("gt_note_count"),
                        "selection_features": clip.get("selection_features") or {},
                    },
                    **clip_result,
                }
        finally:
            if log_handle is not None:
                log_handle.close()

    aggregate = {
        "score": aggregate_clip_summaries(clip_results.values(), "score"),
        "preview": aggregate_clip_summaries(clip_results.values(), "preview"),
    }
    return {
        "candidate": candidate,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "log": str(log_path) if log_path is not None else None,
        "aggregate": aggregate,
        "clips": clip_results,
    }


def _metric_delta(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def rank_results(results: Sequence[Mapping[str, Any]], surface: str, latency_tolerance_ms: float) -> List[Dict[str, Any]]:
    baseline = next(
        (item for item in results if (item.get("candidate") or {}).get("name") == "baseline_current"),
        results[0],
    )
    baseline_metrics = ((baseline.get("aggregate") or {}).get(surface) or {})
    baseline_latency = float(baseline_metrics.get("inference_ms_p95", 0.0) or 0.0)
    rows: List[Dict[str, Any]] = []
    for item in results:
        metrics = ((item.get("aggregate") or {}).get(surface) or {})
        latency = float(metrics.get("inference_ms_p95", 0.0) or 0.0)
        rows.append(
            {
                "name": (item.get("candidate") or {}).get("name"),
                "attrs": (item.get("candidate") or {}).get("attrs") or {},
                "notes": (item.get("candidate") or {}).get("notes") or "",
                "metrics": metrics,
                "delta_vs_baseline": _metric_delta(metrics, baseline_metrics),
                "latency_ok": latency <= baseline_latency + latency_tolerance_ms,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            bool(row["latency_ok"]),
            float(row["metrics"].get("note_f1", 0.0) or 0.0),
            float(row["metrics"].get("note_recall", 0.0) or 0.0),
            float(row["metrics"].get("cluster_jaccard", 0.0) or 0.0),
            float(row["metrics"].get("cluster_f1", 0.0) or 0.0),
            -float(row["metrics"].get("duplicates_per_100", 0.0) or 0.0),
        ),
        reverse=True,
    )


def write_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    ranked = summary.get("ranked") or []
    lines = [
        "# Continuous Stream Decoder Sweep",
        "",
        f"Generated: {summary.get('generated_at')}",
        f"Surface: `{summary.get('surface')}`",
        "",
        "This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.",
        "",
        "| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked, start=1):
        metrics = row.get("metrics") or {}
        lines.append(
            "| "
            f"{index} | {row.get('name')} | {row.get('latency_ok')} | "
            f"{float(metrics.get('note_f1', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('note_recall', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('note_precision', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('cluster_f1', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('cluster_jaccard', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('strict_20ms_f1', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('boundary_recall', 0.0) or 0.0):.4f} | "
            f"{float(metrics.get('duplicates_per_100', 0.0) or 0.0):.2f} | "
            f"{float(metrics.get('inference_ms_p95', 0.0) or 0.0):.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_candidates(args: argparse.Namespace) -> List[Candidate]:
    candidates = build_candidates(args.preset)
    if args.only:
        requested = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["name"] in requested]
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep continuous live stream decoder settings.")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--benchmark-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--clip-ids", nargs="+", default=[])
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--surface", choices=["score", "preview"], default="score")
    parser.add_argument("--latency-tolerance-ms", type=float, default=2.0)
    parser.add_argument("--packet-ms", type=float, default=40.0)
    parser.add_argument("--context-sec", type=float, default=1.8)
    parser.add_argument("--inference-interval-ms", type=float, default=70.0)
    parser.add_argument("--trusted-delay-ms", type=float, default=180.0)
    parser.add_argument("--commit-delay-ms", type=float, default=500.0)
    parser.add_argument("--lock-delay-ms", type=float, default=2000.0)
    parser.add_argument("--tail-padding-sec", type=float, default=0.6)
    parser.add_argument("--chunk-seconds-for-boundary", type=float, default=0.6)
    parser.add_argument("--eval-boundary-band-sec", type=float, default=0.10)
    parser.add_argument("--show-model-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = load_manifest(Path(args.benchmark_manifest), args.clip_ids)
    if args.max_clips:
        clips = dict(list(clips.items())[: args.max_clips])
    candidates = select_candidates(args)
    if not candidates:
        raise SystemExit("No candidates selected.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / f"continuous_decoder_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Running {len(candidates)} continuous decoder candidates on {len(clips)} clips -> {output_dir}")
    for candidate in candidates:
        print(f"[continuous-tune] {candidate['name']}")
        result = run_candidate(candidate, clips, args, output_dir / f"{candidate['name']}.log")
        results.append(result)
        (output_dir / f"{candidate['name']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    ranked = rank_results(results, args.surface, args.latency_tolerance_ms)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "surface": args.surface,
        "benchmark_manifest": str(Path(args.benchmark_manifest).resolve()),
        "clip_ids": list(clips.keys()),
        "stream_config": {
            "packet_ms": args.packet_ms,
            "context_sec": args.context_sec,
            "inference_interval_ms": args.inference_interval_ms,
            "trusted_delay_ms": args.trusted_delay_ms,
            "commit_delay_ms": args.commit_delay_ms,
            "lock_delay_ms": args.lock_delay_ms,
            "tail_padding_sec": args.tail_padding_sec,
        },
        "ranked": ranked,
        "results": results,
    }
    summary_json = output_dir / "continuous_decoder_summary.json"
    summary_md = output_dir / "continuous_decoder_summary.md"
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
            f"note_f1={float(metrics.get('note_f1', 0.0) or 0.0):.4f} "
            f"recall={float(metrics.get('note_recall', 0.0) or 0.0):.4f} "
            f"cluster_f1={float(metrics.get('cluster_f1', 0.0) or 0.0):.4f} "
            f"p95_inference_ms={float(metrics.get('inference_ms_p95', 0.0) or 0.0):.2f}"
        )


if __name__ == "__main__":
    main()
