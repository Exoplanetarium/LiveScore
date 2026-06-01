"""Diagnose how much note accuracy is lost by live chunking.

Runs the mel baseline checkpoint in two modes on the same audio excerpt:
1. Full-audio inference over the whole clip.
2. Simulated live chunking with the same checkpoint, threshold, overlap drop,
   and absolute-time shifting used by the stream path.

Optionally compares both outputs to MIDI ground truth and writes a JSON report
with note-by-note mismatch details, including distance to the nearest chunk
boundary so chunk-edge failure modes are obvious.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(__file__))

from detect_note import note_to_name  # noqa: E402
from gpu_ops import get_gpu_mel_baseline_transcriber  # noqa: E402
from test_ensemble_accuracy import compute_note_metrics, load_midi_notes  # noqa: E402


SAMPLE_RATE = 44100
DEFAULT_MANIFEST = Path(__file__).parent / "live_benchmark_replay_auto_v2.json"
DEFAULT_OUTPUT = Path(__file__).parent / "live_chunk_gap_report.json"
DEFAULT_CHUNK_MS = 600.0
DEFAULT_OVERLAP_SAMPLES = 4096
DEFAULT_ONSET_THRESHOLD = 0.55
DEFAULT_FRAME_THRESHOLD = 0.5
DEFAULT_ONSET_TOL = 0.05
DEFAULT_CHUNK_END_GUARD_SEC = 0.025
DEFAULT_CHUNK_END_MICRO_EVENT_MAX_DURATION_SEC = 0.045


@dataclass
class ClipSpec:
    label: str
    audio_path: Path
    midi_path: Optional[Path]
    start_sec: float
    end_sec: float
    clip_id: Optional[str] = None
    title: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _load_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _auto_select_clip(manifest_payload: Dict) -> Tuple[str, Dict]:
    clips = manifest_payload.get("clips") or {}
    if not clips:
        raise ValueError(f"No clips found in manifest: {manifest_payload}")

    def _score(item: Tuple[str, Dict]) -> Tuple[float, float, float]:
        clip_id, clip = item
        features = clip.get("selection_features") or {}
        boundary_rate = float(features.get("boundary_event_rate") or 0.0)
        density = float(features.get("note_density") or 0.0)
        gt_count = float(clip.get("gt_note_count") or 0.0)
        return (boundary_rate, density, gt_count)

    return max(clips.items(), key=_score)


def resolve_clip(args) -> ClipSpec:
    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if args.end_sec is None:
            raise ValueError("--end-sec is required when using --audio-path")
        return ClipSpec(
            label=audio_path.stem,
            audio_path=audio_path,
            midi_path=Path(args.midi_path) if args.midi_path else None,
            start_sec=float(args.start_sec or 0.0),
            end_sec=float(args.end_sec),
            title=audio_path.stem,
        )

    manifest_path = Path(args.manifest or DEFAULT_MANIFEST)
    manifest_payload = _load_manifest(manifest_path)
    clips = manifest_payload.get("clips") or {}
    if args.clip_id:
        if args.clip_id not in clips:
            raise KeyError(f"Clip {args.clip_id} not found in manifest {manifest_path}")
        clip_id = args.clip_id
        clip = clips[clip_id]
    else:
        clip_id, clip = _auto_select_clip(manifest_payload)

    return ClipSpec(
        label=clip_id,
        clip_id=clip_id,
        title=clip.get("title"),
        audio_path=Path(clip["audio_path"]),
        midi_path=Path(clip["midi_path"]) if clip.get("midi_path") else None,
        start_sec=float(clip.get("start_sec") or 0.0),
        end_sec=float(clip.get("end_sec") or 0.0),
    )


def load_audio_excerpt(path: Path, start_sec: float, end_sec: float, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if end_sec <= start_sec:
        raise ValueError(f"Invalid clip bounds: start={start_sec}, end={end_sec}")

    with sf.SoundFile(str(path)) as handle:
        source_sr = int(handle.samplerate)
        start_frame = max(0, int(round(start_sec * source_sr)))
        end_frame = min(len(handle), int(round(end_sec * source_sr)))
        handle.seek(start_frame)
        frames = max(0, end_frame - start_frame)
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)

    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    if source_sr != target_sr:
        gcd = math.gcd(source_sr, target_sr)
        up, down = target_sr // gcd, source_sr // gcd
        mono = resample_poly(mono, up, down).astype(np.float32, copy=False)
    return mono


def load_gt_excerpt(midi_path: Optional[Path], start_sec: float, end_sec: float) -> List[Dict]:
    if midi_path is None or not midi_path.exists():
        return []

    gt_notes, _ = load_midi_notes(str(midi_path), extend_with_pedal=True)
    clipped = []
    for note in gt_notes:
        onset = float(note["onset_time"])
        if onset < start_sec or onset >= end_sec:
            continue
        offset = min(float(note["offset_time"]), end_sec)
        local_note = {
            "onset_time": onset - start_sec,
            "offset_time": max(onset - start_sec, offset - start_sec),
            "duration": max(0.0, offset - onset),
            "midi_note": int(note["midi_note"]),
            "velocity": int(note.get("velocity", 64)),
        }
        clipped.append(local_note)
    clipped.sort(key=lambda item: (item["onset_time"], item["midi_note"]))
    return clipped


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(source_sr, target_sr)
    up, down = target_sr // gcd, source_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def _assign_event_ids(events: Sequence[Dict], prefix: str) -> List[Dict]:
    assigned = []
    for index, event in enumerate(events):
        item = dict(event)
        item["event_id"] = f"{prefix}_{index:05d}"
        assigned.append(item)
    return assigned


def _event_rank(event: Dict) -> Tuple[float, float]:
    duration = float(event.get("offset_time", event["onset_time"]) - event["onset_time"])
    velocity = float(event.get("velocity") or 0.0)
    return (velocity, duration)


def dedupe_note_events(events: Sequence[Dict], dedup_window: float = 0.05) -> List[Dict]:
    deduped: List[Dict] = []
    for event in sorted(events, key=lambda item: (item["onset_time"], item["midi_note"])):
        replacement_idx = None
        for idx in range(len(deduped) - 1, -1, -1):
            existing = deduped[idx]
            if existing["midi_note"] != event["midi_note"]:
                continue
            if abs(existing["onset_time"] - event["onset_time"]) <= dedup_window:
                replacement_idx = idx
                break
        if replacement_idx is None:
            deduped.append(dict(event))
            continue
        if _event_rank(event) > _event_rank(deduped[replacement_idx]):
            deduped[replacement_idx] = dict(event)
    return sorted(deduped, key=lambda item: (item["onset_time"], item["midi_note"]))


def drop_chunk_end_micro_events(
    events: Sequence[Dict],
    analysis_window_sec: float,
    guard_sec: float,
    max_duration_sec: float,
) -> Tuple[List[Dict], int]:
    if not events or analysis_window_sec <= 0:
        return list(events or []), 0

    threshold = max(0.0, analysis_window_sec - guard_sec)
    kept: List[Dict] = []
    dropped = 0
    for event in events:
        onset = float(event["onset_time"])
        duration = max(0.0, float(event.get("offset_time", onset)) - onset)
        if onset >= threshold and duration <= max_duration_sec:
            dropped += 1
            continue
        kept.append(dict(event))
    return kept, dropped


def greedy_match_notes(pred_notes: Sequence[Dict], ref_notes: Sequence[Dict], onset_tol: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    ref_used = set()
    matches: List[Dict] = []
    pred_only: List[Dict] = []

    for pred in pred_notes:
        best_idx = None
        best_key = None
        for ref_idx, ref in enumerate(ref_notes):
            if ref_idx in ref_used:
                continue
            if int(pred["midi_note"]) != int(ref["midi_note"]):
                continue
            onset_diff = abs(float(pred["onset_time"]) - float(ref["onset_time"]))
            if onset_diff > onset_tol:
                continue
            offset_diff = abs(float(pred.get("offset_time", pred["onset_time"])) - float(ref.get("offset_time", ref["onset_time"])))
            key = (onset_diff, offset_diff)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = ref_idx
        if best_idx is None:
            pred_only.append(dict(pred))
            continue
        ref_used.add(best_idx)
        ref = ref_notes[best_idx]
        matches.append({
            "pred_id": pred.get("event_id"),
            "ref_id": ref.get("event_id"),
            "pred": dict(pred),
            "ref": dict(ref),
            "onset_diff_sec": abs(float(pred["onset_time"]) - float(ref["onset_time"])),
            "offset_diff_sec": abs(float(pred.get("offset_time", pred["onset_time"])) - float(ref.get("offset_time", ref["onset_time"]))),
        })
    ref_only = [dict(ref) for idx, ref in enumerate(ref_notes) if idx not in ref_used]
    return matches, pred_only, ref_only


def nearest_chunk_boundary_seconds(time_seconds: float, chunk_seconds: float, clip_duration_seconds: float) -> float:
    if chunk_seconds <= 0:
        return 0.0
    n_boundaries = int(math.ceil(clip_duration_seconds / chunk_seconds))
    boundaries = [index * chunk_seconds for index in range(n_boundaries + 1)]
    return min(abs(time_seconds - boundary) for boundary in boundaries)


def summarize_boundary_distances(events: Sequence[Dict], chunk_seconds: float, clip_duration_seconds: float) -> Dict:
    if not events:
        return {
            "count": 0,
            "median_ms": None,
            "within_50ms": 0,
            "within_100ms": 0,
            "within_200ms": 0,
        }

    distances_ms = [
        nearest_chunk_boundary_seconds(float(event["onset_time"]), chunk_seconds, clip_duration_seconds) * 1000.0
        for event in events
    ]
    return {
        "count": len(distances_ms),
        "median_ms": float(np.median(distances_ms)),
        "within_50ms": int(sum(distance <= 50.0 for distance in distances_ms)),
        "within_100ms": int(sum(distance <= 100.0 for distance in distances_ms)),
        "within_200ms": int(sum(distance <= 200.0 for distance in distances_ms)),
    }


def expand_event_for_report(event: Dict, chunk_seconds: float, clip_duration_seconds: float) -> Dict:
    onset_time = float(event["onset_time"])
    offset_time = float(event.get("offset_time", onset_time))
    boundary_ms = nearest_chunk_boundary_seconds(onset_time, chunk_seconds, clip_duration_seconds) * 1000.0
    return {
        "event_id": event.get("event_id"),
        "onset_time": round(onset_time, 4),
        "offset_time": round(offset_time, 4),
        "duration_ms": round(max(0.0, offset_time - onset_time) * 1000.0, 1),
        "midi_note": int(event["midi_note"]),
        "note_name": note_to_name(int(event["midi_note"])),
        "velocity": int(event.get("velocity", 0) or 0),
        "nearest_chunk_boundary_ms": round(boundary_ms, 1),
        "source_chunk_idx": event.get("source_chunk_idx"),
        "chunk_local_onset_sec": round(float(event.get("chunk_local_onset_sec", 0.0)), 4)
        if event.get("chunk_local_onset_sec") is not None else None,
    }


def annotate_support(events: Sequence[Dict], matched_ids: Iterable[str]) -> List[Dict]:
    supported = set(matched_ids)
    annotated = []
    for event in events:
        item = dict(event)
        item["matches_ground_truth"] = item.get("event_id") in supported
        annotated.append(item)
    return annotated


def run_full_inference(audio: np.ndarray, model, onset_threshold: float, frame_threshold: float) -> List[Dict]:
    model_sr = int(model.config.get("sample_rate", 16000))
    model_audio = _resample_audio(audio, SAMPLE_RATE, model_sr)
    result = model.transcribe(
        model_audio,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    return [dict(event) for event in result.get("est_note_events", [])]


def simulate_live_chunking(
    audio: np.ndarray,
    model,
    chunk_ms: float,
    overlap_samples: int,
    onset_threshold: float,
    frame_threshold: float,
    chunk_end_guard_sec: float,
    chunk_end_micro_event_max_duration_sec: float,
) -> Tuple[List[Dict], List[Dict]]:
    chunk_samples = max(1, int(round(chunk_ms / 1000.0 * SAMPLE_RATE)))
    emitted_events: List[Dict] = []
    chunk_reports: List[Dict] = []

    sample_cursor = 0
    tail = np.zeros(0, dtype=np.float32)

    for chunk_idx, chunk_start in enumerate(range(0, len(audio), chunk_samples)):
        chunk_audio = audio[chunk_start:chunk_start + chunk_samples]
        x_full = np.concatenate([tail, chunk_audio]) if tail.size else chunk_audio
        overlap_sec = float(tail.size) / SAMPLE_RATE
        local_events = run_full_inference(x_full, model, onset_threshold, frame_threshold)
        local_events, dropped_boundary_micro = drop_chunk_end_micro_events(
            local_events,
            len(x_full) / SAMPLE_RATE,
            chunk_end_guard_sec,
            chunk_end_micro_event_max_duration_sec,
        )

        raw_count = len(local_events) + dropped_boundary_micro
        dropped_overlap = 0
        kept = 0

        for event in local_events:
            onset_time = float(event["onset_time"])
            if onset_time < overlap_sec:
                dropped_overlap += 1
                continue
            absolute_onset = (sample_cursor / SAMPLE_RATE) + (onset_time - overlap_sec)
            absolute_offset = (sample_cursor / SAMPLE_RATE) + (float(event["offset_time"]) - overlap_sec)
            shifted = dict(event)
            shifted["onset_time"] = absolute_onset
            shifted["offset_time"] = max(absolute_onset, absolute_offset)
            shifted["source_chunk_idx"] = chunk_idx
            shifted["chunk_local_onset_sec"] = onset_time
            emitted_events.append(shifted)
            kept += 1

        chunk_reports.append({
            "chunk_idx": chunk_idx,
            "chunk_start_sec": round(chunk_start / SAMPLE_RATE, 4),
            "chunk_end_sec": round((chunk_start + len(chunk_audio)) / SAMPLE_RATE, 4),
            "chunk_duration_sec": round(len(chunk_audio) / SAMPLE_RATE, 4),
            "analysis_window_sec": round(len(x_full) / SAMPLE_RATE, 4),
            "overlap_sec": round(overlap_sec, 4),
            "raw_events": raw_count,
            "dropped_boundary_micro": dropped_boundary_micro,
            "dropped_overlap": dropped_overlap,
            "kept_events": kept,
        })

        sample_cursor += len(chunk_audio)
        take = min(overlap_samples, len(x_full))
        tail = x_full[-take:].astype(np.float32, copy=False)

    return dedupe_note_events(emitted_events), chunk_reports


def _matched_id_sets(matches: Sequence[Dict], pred_key: str, ref_key: str) -> Tuple[set, set]:
    pred_ids = {match[pred_key] for match in matches if match.get(pred_key)}
    ref_ids = {match[ref_key] for match in matches if match.get(ref_key)}
    return pred_ids, ref_ids


def build_report(args, clip: ClipSpec) -> Dict:
    model = get_gpu_mel_baseline_transcriber()
    if model is None or not model.initialized:
        raise RuntimeError("Mel baseline GPU transcriber is not initialized")

    audio = load_audio_excerpt(clip.audio_path, clip.start_sec, clip.end_sec)
    if audio.size == 0:
        raise RuntimeError(f"Loaded empty audio excerpt from {clip.audio_path}")

    gt_notes = _assign_event_ids(load_gt_excerpt(clip.midi_path, clip.start_sec, clip.end_sec), "gt")
    full_events = _assign_event_ids(
        dedupe_note_events(run_full_inference(audio, model, args.onset_threshold, args.frame_threshold)),
        "full",
    )
    chunked_events, chunk_reports = simulate_live_chunking(
        audio,
        model,
        args.chunk_ms,
        args.overlap_samples,
        args.onset_threshold,
        args.frame_threshold,
        args.chunk_end_guard_sec,
        args.chunk_end_micro_event_max_duration_sec,
    )
    chunked_events = _assign_event_ids(chunked_events, "chunk")

    full_vs_chunk_matches, chunk_only, full_only = greedy_match_notes(
        chunked_events,
        full_events,
        onset_tol=args.onset_tol,
    )
    full_vs_gt_matches, full_gt_only, gt_only_from_full = greedy_match_notes(
        full_events,
        gt_notes,
        onset_tol=args.onset_tol,
    )
    chunk_vs_gt_matches, chunk_gt_only, gt_only_from_chunk = greedy_match_notes(
        chunked_events,
        gt_notes,
        onset_tol=args.onset_tol,
    )

    full_gt_pred_ids, full_gt_ref_ids = _matched_id_sets(full_vs_gt_matches, "pred_id", "ref_id")
    chunk_gt_pred_ids, chunk_gt_ref_ids = _matched_id_sets(chunk_vs_gt_matches, "pred_id", "ref_id")

    full_only_annotated = annotate_support(full_only, full_gt_pred_ids)
    chunk_only_annotated = annotate_support(chunk_only, chunk_gt_pred_ids)

    chunk_vs_full_metrics = compute_note_metrics(chunked_events, full_events, onset_tol=args.onset_tol)
    full_vs_gt_metrics = compute_note_metrics(full_events, gt_notes, onset_tol=args.onset_tol) if gt_notes else None
    chunk_vs_gt_metrics = compute_note_metrics(chunked_events, gt_notes, onset_tol=args.onset_tol) if gt_notes else None

    chunk_seconds = args.chunk_ms / 1000.0
    clip_duration = clip.duration_sec

    full_only_report = [
        expand_event_for_report(event, chunk_seconds, clip_duration)
        | {"matches_ground_truth": bool(event.get("matches_ground_truth"))}
        for event in full_only_annotated
    ]
    chunk_only_report = [
        expand_event_for_report(event, chunk_seconds, clip_duration)
        | {"matches_ground_truth": bool(event.get("matches_ground_truth"))}
        for event in chunk_only_annotated
    ]

    full_only_report.sort(key=lambda item: (item["nearest_chunk_boundary_ms"], item["onset_time"], item["midi_note"]))
    chunk_only_report.sort(key=lambda item: (item["nearest_chunk_boundary_ms"], item["onset_time"], item["midi_note"]))

    report = {
        "clip": {
            "label": clip.label,
            "clip_id": clip.clip_id,
            "title": clip.title,
            "audio_path": str(clip.audio_path),
            "midi_path": str(clip.midi_path) if clip.midi_path else None,
            "start_sec": clip.start_sec,
            "end_sec": clip.end_sec,
            "duration_sec": clip.duration_sec,
        },
        "config": {
            "chunk_ms": args.chunk_ms,
            "overlap_samples": args.overlap_samples,
            "overlap_ms": args.overlap_samples / SAMPLE_RATE * 1000.0,
            "onset_threshold": args.onset_threshold,
            "frame_threshold": args.frame_threshold,
            "onset_tolerance_sec": args.onset_tol,
            "chunk_end_guard_sec": args.chunk_end_guard_sec,
            "chunk_end_micro_event_max_duration_sec": args.chunk_end_micro_event_max_duration_sec,
            "model_path": str(model.config.get("init_from") or model.config.get("save_path") or "mel_baseline_transcription.pt"),
        },
        "counts": {
            "ground_truth_notes": len(gt_notes),
            "full_notes": len(full_events),
            "chunked_notes": len(chunked_events),
        },
        "metrics": {
            "chunked_vs_full": {
                "precision": chunk_vs_full_metrics["precision"],
                "recall": chunk_vs_full_metrics["recall"],
                "f1": chunk_vs_full_metrics["f1"],
                "matched": chunk_vs_full_metrics["matched"],
                "predicted": chunk_vs_full_metrics["predicted"],
                "reference": chunk_vs_full_metrics["ground_truth"],
            },
            "full_vs_gt": full_vs_gt_metrics,
            "chunked_vs_gt": chunk_vs_gt_metrics,
        },
        "boundary_analysis": {
            "full_only": summarize_boundary_distances(full_only_annotated, chunk_seconds, clip_duration),
            "chunk_only": summarize_boundary_distances(chunk_only_annotated, chunk_seconds, clip_duration),
        },
        "difference_summary": {
            "matched_full_notes": len(full_vs_chunk_matches),
            "missed_by_chunking": len(full_only_annotated),
            "chunk_only_extras": len(chunk_only_annotated),
            "missed_by_chunking_gt_supported": int(sum(item["matches_ground_truth"] for item in full_only_annotated)),
            "chunk_only_gt_supported": int(sum(item["matches_ground_truth"] for item in chunk_only_annotated)),
        },
        "chunk_reports": chunk_reports,
        "missed_by_chunking": full_only_report,
        "chunk_only_extras": chunk_only_report,
        "top_matches": full_vs_chunk_matches[:25],
        "full_gt_missing": [expand_event_for_report(item, chunk_seconds, clip_duration) for item in gt_only_from_full[:25]],
        "chunk_gt_missing": [expand_event_for_report(item, chunk_seconds, clip_duration) for item in gt_only_from_chunk[:25]],
    }
    return report


def print_summary(report: Dict, max_events: int) -> None:
    clip = report["clip"]
    metrics = report["metrics"]
    diff = report["difference_summary"]
    boundary = report["boundary_analysis"]

    print("=" * 72)
    print(f"Live chunk-gap diagnostic: {clip['label']} | {clip.get('title') or 'untitled'}")
    print(f"Audio: {clip['audio_path']}")
    print(f"Clip window: {clip['start_sec']:.2f}s -> {clip['end_sec']:.2f}s ({clip['duration_sec']:.2f}s)")
    print("=" * 72)
    print(
        f"Chunked vs full: P={metrics['chunked_vs_full']['precision']:.3f} "
        f"R={metrics['chunked_vs_full']['recall']:.3f} "
        f"F1={metrics['chunked_vs_full']['f1']:.3f} | "
        f"matched={metrics['chunked_vs_full']['matched']} "
        f"chunked={metrics['chunked_vs_full']['predicted']} full={metrics['chunked_vs_full']['reference']}"
    )

    if metrics.get("full_vs_gt"):
        full_gt = metrics["full_vs_gt"]
        chunk_gt = metrics["chunked_vs_gt"]
        print(
            f"Full vs GT:    P={full_gt['precision']:.3f} R={full_gt['recall']:.3f} F1={full_gt['f1']:.3f}"
        )
        print(
            f"Chunked vs GT: P={chunk_gt['precision']:.3f} R={chunk_gt['recall']:.3f} F1={chunk_gt['f1']:.3f}"
        )

    print(
        f"Missed by chunking: {diff['missed_by_chunking']} "
        f"({diff['missed_by_chunking_gt_supported']} GT-supported)"
    )
    print(
        f"Chunk-only extras: {diff['chunk_only_extras']} "
        f"({diff['chunk_only_gt_supported']} GT-supported)"
    )
    print(
        f"Boundary concentration for missed notes: "
        f"<=50ms {boundary['full_only']['within_50ms']}, "
        f"<=100ms {boundary['full_only']['within_100ms']}, "
        f"median {boundary['full_only']['median_ms']}ms"
    )

    if report["missed_by_chunking"]:
        print("\nTop missed-by-chunking notes:")
        for item in report["missed_by_chunking"][:max_events]:
            gt_tag = " gt" if item.get("matches_ground_truth") else ""
            print(
                f"  miss {item['note_name']:>4} @{item['onset_time']:.3f}s "
                f"dur={item['duration_ms']:.0f}ms boundary={item['nearest_chunk_boundary_ms']:.1f}ms{gt_tag}"
            )

    if report["chunk_only_extras"]:
        print("\nTop chunk-only extras:")
        for item in report["chunk_only_extras"][:max_events]:
            gt_tag = " gt" if item.get("matches_ground_truth") else ""
            print(
                f"  extra {item['note_name']:>4} @{item['onset_time']:.3f}s "
                f"dur={item['duration_ms']:.0f}ms boundary={item['nearest_chunk_boundary_ms']:.1f}ms{gt_tag}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full-audio mel inference against simulated live chunking.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Replay benchmark manifest JSON")
    parser.add_argument("--clip-id", type=str, default=None, help="Clip ID from the manifest")
    parser.add_argument("--audio-path", type=str, default=None, help="Explicit audio path instead of manifest clip")
    parser.add_argument("--midi-path", type=str, default=None, help="Explicit MIDI path when using --audio-path")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Clip start time when using --audio-path")
    parser.add_argument("--end-sec", type=float, default=None, help="Clip end time when using --audio-path")
    parser.add_argument("--chunk-ms", type=float, default=DEFAULT_CHUNK_MS, help="Simulated client chunk size in milliseconds")
    parser.add_argument("--overlap-samples", type=int, default=DEFAULT_OVERLAP_SAMPLES, help="Server-side overlap history in 44.1kHz samples")
    parser.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD, help="Fixed onset threshold for both modes")
    parser.add_argument("--frame-threshold", type=float, default=DEFAULT_FRAME_THRESHOLD, help="Fixed frame threshold for both modes")
    parser.add_argument("--onset-tol", type=float, default=DEFAULT_ONSET_TOL, help="Onset tolerance for note matching")
    parser.add_argument("--chunk-end-guard-sec", type=float, default=DEFAULT_CHUNK_END_GUARD_SEC, help="Suppress short events whose onset falls within this many seconds of the chunk end")
    parser.add_argument("--chunk-end-micro-event-max-duration-sec", type=float, default=DEFAULT_CHUNK_END_MICRO_EVENT_MAX_DURATION_SEC, help="Maximum duration treated as a chunk-end micro-event")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="JSON output path")
    parser.add_argument("--max-events", type=int, default=12, help="How many mismatch events to print to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip = resolve_clip(args)
    report = build_report(args, clip)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report, max_events=args.max_events)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()