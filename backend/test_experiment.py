"""Benchmark the live retro-correction path on local MAESTRO clips.

This script reuses the overlap-aware live chunk analyzer from backend/main.py,
so control, adaptive, and retro-correction runs exercise the same chunk-tail
continuity logic as the app's live path.

Example:
    .\\env\\Scripts\\python.exe test_experiment.py

Quick sanity check:
    .\\env\\Scripts\\python.exe test_experiment.py --pieces 1 --scan-seconds 20 --clip-seconds 4 --categories quiet
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, nullcontext
import json
import math
import os
import sys
import uuid
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(__file__))

import detect_note as detect_note_module  # noqa: E402
from main import _analyze_uploaded_stream_chunk, _clear_stream_session, _dedupe_note_events  # noqa: E402


TARGET_SR = 44100
DEFAULT_CATEGORIES = ("quiet", "loud", "mixed")
FIXED_THRESHOLD_EXPERIMENT = "fixed_threshold"
RETRO_CORRECTION_EXPERIMENT = "retro_correction_seam_v1"
_ORIGINAL_LIVE_THRESHOLD_SELECTOR = detect_note_module._select_live_neural_onset_threshold


@contextmanager
def force_live_onset_threshold(
    onset_threshold: float,
    experiment_name: str = FIXED_THRESHOLD_EXPERIMENT,
) -> Iterator[None]:
    original_selector = detect_note_module._select_live_neural_onset_threshold

    def _fixed_selector(audio_chunk, base_onset_threshold, enabled=True):
        _, info = _ORIGINAL_LIVE_THRESHOLD_SELECTOR(audio_chunk, float(onset_threshold), enabled=False)
        debug_info = dict(info)
        debug_info["experiment"] = experiment_name
        debug_info["profile"] = f"fixed_{float(onset_threshold):.2f}"
        return float(onset_threshold), debug_info

    detect_note_module._select_live_neural_onset_threshold = _fixed_selector
    try:
        yield
    finally:
        detect_note_module._select_live_neural_onset_threshold = original_selector


def load_index(split: str) -> List[Dict]:
    index_path = (
        Path(__file__).resolve().parent
        / "rhythm_training"
        / "ensemble_index"
        / f"{split}_index.json"
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pieces = payload.get("pieces") or []
    if not pieces:
        raise RuntimeError(f"No pieces found in {index_path}")
    return pieces


def load_audio_excerpt(audio_path: str, start_sec: float, duration_sec: float, target_sr: int) -> np.ndarray:
    with sf.SoundFile(audio_path) as handle:
        source_sr = int(handle.samplerate)
        start_frame = max(0, int(round(start_sec * source_sr)))
        n_frames = max(0, int(round(duration_sec * source_sr)))
        handle.seek(start_frame)
        audio = handle.read(frames=n_frames, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1).astype(np.float32, copy=False)
    if source_sr == target_sr:
        return mono

    gcd = math.gcd(source_sr, target_sr)
    up = target_sr // gcd
    down = source_sr // gcd
    return resample_poly(mono, up, down).astype(np.float32, copy=False)


def wav_bytes_from_audio(audio: np.ndarray, sample_rate: int) -> bytes:
    payload = BytesIO()
    sf.write(payload, audio.astype(np.float32, copy=False), sample_rate, format="WAV", subtype="PCM_16")
    return payload.getvalue()


def load_midi_notes(midi_path: str) -> List[Dict]:
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes: List[Dict] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onset = float(note.start)
            offset = float(note.end)
            notes.append(
                {
                    "onset_time": onset,
                    "offset_time": offset,
                    "duration": max(0.0, offset - onset),
                    "midi_note": int(note.pitch),
                    "velocity": int(note.velocity),
                }
            )
    notes.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return notes


def get_midi_reference_bpm(midi_path: str) -> float:
    midi = pretty_midi.PrettyMIDI(midi_path)
    _, tempos = midi.get_tempo_changes()
    if len(tempos) == 0:
        return 120.0
    if len(tempos) == 1:
        return float(tempos[0])
    return float(np.median(tempos.astype(np.float64, copy=False)))


def slice_gt_notes(notes: Sequence[Dict], clip_start_sec: float, clip_end_sec: float) -> List[Dict]:
    sliced: List[Dict] = []
    for note in notes:
        onset = float(note.get("onset_time", 0.0) or 0.0)
        if onset < clip_start_sec or onset >= clip_end_sec:
            continue
        clipped = dict(note)
        clipped["onset_time"] = round(onset - clip_start_sec, 6)
        offset = min(float(note.get("offset_time", onset) or onset), clip_end_sec)
        clipped["offset_time"] = round(max(clipped["onset_time"], offset - clip_start_sec), 6)
        clipped["duration"] = round(max(0.0, clipped["offset_time"] - clipped["onset_time"]), 6)
        sliced.append(clipped)
    sliced.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return sliced


def excerpt_statistics(audio: np.ndarray) -> Dict:
    samples = audio.astype(np.float64, copy=False)
    if samples.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "crest_factor": 0.0,
            "rms_std": 0.0,
            "subwindow_count": 0,
        }

    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    crest_factor = peak / rms if rms > 1e-9 else 0.0

    subwindow_frames = max(1, int(round(0.2 * TARGET_SR)))
    subwindow_rms = []
    for start in range(0, samples.size, subwindow_frames):
        window = samples[start : start + subwindow_frames]
        if window.size == 0:
            continue
        subwindow_rms.append(float(np.sqrt(np.mean(window * window))))

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "rms_std": float(np.std(np.asarray(subwindow_rms, dtype=np.float64))) if subwindow_rms else 0.0,
        "subwindow_count": len(subwindow_rms),
    }


def build_candidates(
    pieces: Sequence[Dict],
    num_pieces: int,
    scan_seconds: float,
    clip_seconds: float,
    step_seconds: float,
    min_notes: int,
) -> List[Dict]:
    candidates: List[Dict] = []

    for piece in list(pieces)[: max(1, int(num_pieces))]:
        audio_path = str(piece.get("audio") or piece.get("audio_path") or "")
        midi_path = str(piece.get("midi") or piece.get("midi_path") or "")
        if not audio_path or not midi_path:
            continue

        title = str(piece.get("title") or Path(midi_path).stem)
        piece_duration = float(piece.get("duration", 0.0) or 0.0)
        scan_limit = min(scan_seconds, piece_duration) if piece_duration > 0.0 else scan_seconds
        if scan_limit < clip_seconds:
            continue

        midi_notes = load_midi_notes(midi_path)
        start_sec = 0.0
        while start_sec + clip_seconds <= scan_limit + 1e-9:
            end_sec = start_sec + clip_seconds
            gt_notes = slice_gt_notes(midi_notes, start_sec, end_sec)
            if len(gt_notes) < min_notes:
                start_sec += step_seconds
                continue

            audio_excerpt = load_audio_excerpt(audio_path, start_sec, clip_seconds, TARGET_SR)
            stats = excerpt_statistics(audio_excerpt)
            candidates.append(
                {
                    "title": title,
                    "audio_path": audio_path,
                    "midi_path": midi_path,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "duration_sec": round(clip_seconds, 6),
                    "gt_note_count": len(gt_notes),
                    **stats,
                }
            )
            start_sec += step_seconds

    if not candidates:
        raise RuntimeError("No MAESTRO excerpts matched the requested scan settings")
    return candidates


def choose_excerpts(candidates: Sequence[Dict], categories: Sequence[str]) -> Dict[str, Dict]:
    scoring = {
        "quiet": lambda candidate: (
            float(candidate.get("rms", 0.0) or 0.0),
            float(candidate.get("rms_std", 0.0) or 0.0),
            -int(candidate.get("gt_note_count", 0) or 0),
        ),
        "loud": lambda candidate: (
            -float(candidate.get("rms", 0.0) or 0.0),
            -float(candidate.get("peak", 0.0) or 0.0),
            -int(candidate.get("gt_note_count", 0) or 0),
        ),
        "mixed": lambda candidate: (
            -float(candidate.get("rms_std", 0.0) or 0.0),
            -float(candidate.get("rms", 0.0) or 0.0),
            -int(candidate.get("gt_note_count", 0) or 0),
        ),
    }

    selected: Dict[str, Dict] = {}
    used_keys: set[tuple[str, float]] = set()

    for category in categories:
        ranked = sorted(candidates, key=scoring[category])
        chosen = next(
            (
                candidate
                for candidate in ranked
                if (str(candidate["audio_path"]), float(candidate["start_sec"])) not in used_keys
            ),
            ranked[0] if ranked else None,
        )
        if chosen is None:
            continue
        selected[category] = dict(chosen)
        used_keys.add((str(chosen["audio_path"]), float(chosen["start_sec"])))

    return selected


def normalize_predicted_notes(notes: Iterable[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for note in notes:
        normalized_note = dict(note)
        onset = float(note.get("time_seconds", note.get("onset_time", 0.0)) or 0.0)
        duration = note.get("duration_seconds")
        if duration is None:
            offset = note.get("offset_seconds")
            if offset is not None:
                duration = float(offset) - onset
            else:
                duration = note.get("duration", 0.0)
        duration = max(0.0, float(duration or 0.0))
        normalized_note.update(
            {
                "onset_time": onset,
                "offset_time": onset + duration,
                "duration": duration,
                "midi_note": int(note.get("midi_note", note.get("pitch", 0)) or 0),
                "confidence": float(note.get("confidence", 0.0) or 0.0),
            }
        )
        if "note_divisions" in note and note.get("note_divisions") is not None:
            normalized_note["note_divisions"] = float(note.get("note_divisions") or 0.0)
        if "note_value_confidence" in note and note.get("note_value_confidence") is not None:
            normalized_note["note_value_confidence"] = float(note.get("note_value_confidence") or 0.0)
        if "dotted" in note:
            normalized_note["dotted"] = bool(note.get("dotted"))
        normalized.append(normalized_note)
    normalized.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return normalized


def compute_note_metrics(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    onset_tol: float = 0.05,
    offset_tol_sec: float = 0.10,
    duration_tol: float = 0.20,
) -> Dict:
    matched = 0
    rhythm_matched = 0
    gt_matched: set[int] = set()

    for pred in pred_notes:
        best_idx = None
        best_onset_error = None
        for idx, gt in enumerate(gt_notes):
            if idx in gt_matched:
                continue
            if int(pred["midi_note"]) != int(gt["midi_note"]):
                continue
            onset_error = abs(float(pred["onset_time"]) - float(gt["onset_time"]))
            if onset_error > onset_tol:
                continue
            if best_onset_error is None or onset_error < best_onset_error:
                best_idx = idx
                best_onset_error = onset_error

        if best_idx is None:
            continue

        gt_matched.add(best_idx)
        matched += 1
        gt = gt_notes[best_idx]
        gt_duration = max(0.0, float(gt.get("duration", 0.0) or 0.0))
        offset_tol = max(offset_tol_sec, gt_duration * duration_tol)
        pred_offset = float(pred.get("offset_time", pred.get("onset_time", 0.0)) or 0.0)
        gt_offset = float(gt.get("offset_time", gt.get("onset_time", 0.0)) or 0.0)
        if abs(pred_offset - gt_offset) <= offset_tol:
            rhythm_matched += 1

    precision = matched / len(pred_notes) if pred_notes else 0.0
    recall = matched / len(gt_notes) if gt_notes else 0.0
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "predicted": len(pred_notes),
        "ground_truth": len(gt_notes),
        "rhythm_precision": (rhythm_matched / matched) if matched else 0.0,
    }


def compute_offset_metrics(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    onset_tol: float = 0.05,
    offset_tol_sec: float = 0.10,
    duration_tol: float = 0.20,
) -> Dict:
    offset_matched = 0
    gt_matched: set[int] = set()

    for pred in pred_notes:
        pred_offset = float(pred.get("offset_time", pred.get("onset_time", 0.0)) or 0.0)
        for idx, gt in enumerate(gt_notes):
            if idx in gt_matched:
                continue
            gt_offset = float(gt.get("offset_time", gt.get("onset_time", 0.0)) or 0.0)
            gt_duration = max(0.0, float(gt.get("duration", gt_offset - float(gt.get("onset_time", 0.0) or 0.0)) or 0.0))
            max_offset_error = max(offset_tol_sec, gt_duration * duration_tol)
            if (
                int(pred["midi_note"]) == int(gt["midi_note"])
                and abs(float(pred["onset_time"]) - float(gt["onset_time"])) <= onset_tol
                and abs(pred_offset - gt_offset) <= max_offset_error
            ):
                offset_matched += 1
                gt_matched.add(idx)
                break

    offset_precision = offset_matched / len(pred_notes) if pred_notes else 0.0
    offset_recall = offset_matched / len(gt_notes) if gt_notes else 0.0
    offset_f1 = 0.0
    if offset_precision + offset_recall > 0.0:
        offset_f1 = 2.0 * offset_precision * offset_recall / (offset_precision + offset_recall)

    return {
        "offset_precision": offset_precision,
        "offset_recall": offset_recall,
        "offset_f1": offset_f1,
        "offset_matched": offset_matched,
    }


def note_beats_from_prediction(note: Dict) -> float | None:
    if note.get("note_divisions") is not None:
        beats = float(note.get("note_divisions") or 0.0)
        return beats if beats > 0.0 else None

    note_value = note.get("note_value")
    if not note_value:
        return None

    beats = detect_note_module.NOTE_VALUE_BEATS.get(str(note_value))
    if beats is None:
        return None
    beats = float(beats)
    if bool(note.get("dotted")):
        beats *= 1.5
    return beats


def note_beats_from_ground_truth(note: Dict, bpm: float) -> float | None:
    duration = max(0.0, float(note.get("duration", 0.0) or 0.0))
    if duration <= 0.0 or bpm <= 0.0:
        return None
    note_value = detect_note_module.duration_to_note_value(duration, bpm=bpm)
    base = note_value.get("type")
    beats = detect_note_module.NOTE_VALUE_BEATS.get(str(base))
    if beats is None:
        return None
    beats = float(beats)
    if bool(note_value.get("dotted")):
        beats *= 1.5
    return beats


def compute_note_value_metrics(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    reference_bpm: float,
    onset_tol: float = 0.05,
) -> Dict:
    exact_matches = 0
    gt_matched: set[int] = set()
    n_evaluable = 0
    beat_errors: List[float] = []

    for pred in pred_notes:
        pred_beats = note_beats_from_prediction(pred)
        if pred_beats is None:
            continue

        for idx, gt in enumerate(gt_notes):
            if idx in gt_matched:
                continue
            if (
                int(pred["midi_note"]) == int(gt["midi_note"])
                and abs(float(pred["onset_time"]) - float(gt["onset_time"])) <= onset_tol
            ):
                gt_beats = note_beats_from_ground_truth(gt, reference_bpm)
                gt_matched.add(idx)
                if gt_beats is None:
                    break
                n_evaluable += 1
                beat_error = abs(float(pred_beats) - float(gt_beats))
                beat_errors.append(beat_error)
                if beat_error <= 1e-3:
                    exact_matches += 1
                break

    accuracy = exact_matches / n_evaluable if n_evaluable else 0.0
    avg_beat_error = float(np.mean(beat_errors)) if beat_errors else 0.0
    return {
        "note_value_accuracy": accuracy,
        "note_value_matched": n_evaluable,
        "note_value_avg_beat_error": avg_beat_error,
    }


def compute_duplicate_metrics(notes: Sequence[Dict], duplicate_window_sec: float = 0.08) -> Dict:
    duplicate_count = 0
    last_onset_by_pitch: Dict[int, float] = {}

    for note in sorted(notes, key=lambda event: (event["onset_time"], event["midi_note"])):
        pitch = int(note["midi_note"])
        onset = float(note["onset_time"])
        prev_onset = last_onset_by_pitch.get(pitch)
        if prev_onset is not None and (onset - prev_onset) <= duplicate_window_sec:
            duplicate_count += 1
        last_onset_by_pitch[pitch] = onset

    per_100_notes = (100.0 * duplicate_count / len(notes)) if notes else 0.0
    return {
        "duplicates": duplicate_count,
        "duplicates_per_100_notes": per_100_notes,
    }


def is_boundary_note(onset_time: float, chunk_seconds: float, boundary_band_sec: float) -> bool:
    if chunk_seconds <= 0.0:
        return False
    boundary_index = round(onset_time / chunk_seconds)
    boundary_sec = boundary_index * chunk_seconds
    if boundary_sec <= 0.0:
        return False
    return abs(onset_time - boundary_sec) <= boundary_band_sec


def compute_boundary_miss_metrics(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    chunk_seconds: float,
    boundary_band_sec: float,
    onset_tol: float = 0.05,
) -> Dict:
    boundary_gt = [
        note
        for note in gt_notes
        if is_boundary_note(float(note.get("onset_time", 0.0) or 0.0), chunk_seconds, boundary_band_sec)
    ]
    matched = 0
    used_pred: set[int] = set()
    for gt in boundary_gt:
        for idx, pred in enumerate(pred_notes):
            if idx in used_pred:
                continue
            if (
                int(pred["midi_note"]) == int(gt["midi_note"])
                and abs(float(pred["onset_time"]) - float(gt["onset_time"])) <= onset_tol
            ):
                matched += 1
                used_pred.add(idx)
                break

    missed = len(boundary_gt) - matched
    miss_rate = missed / len(boundary_gt) if boundary_gt else 0.0
    return {
        "boundary_gt_notes": len(boundary_gt),
        "boundary_matched_notes": matched,
        "boundary_missed_notes": missed,
        "boundary_miss_rate": miss_rate,
    }


def safe_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def notes_match(note_a: Dict, note_b: Dict, onset_tol: float = 0.05) -> bool:
    return (
        int(note_a["midi_note"]) == int(note_b["midi_note"])
        and abs(float(note_a["onset_time"]) - float(note_b["onset_time"])) <= onset_tol
    )


def _boundary_note_coverage(notes: Sequence[Dict], boundary_sec: float, retro_band_sec: float) -> Dict:
    pre_notes = []
    post_notes = []

    for note in notes:
        onset = float(note.get("onset_time", 0.0) or 0.0)
        if boundary_sec - retro_band_sec <= onset < boundary_sec:
            pre_notes.append(note)
        elif boundary_sec <= onset <= boundary_sec + retro_band_sec:
            post_notes.append(note)

    return {
        "pre_notes": pre_notes,
        "post_notes": post_notes,
        "pre_count": len(pre_notes),
        "post_count": len(post_notes),
        "post_max_confidence": max((float(note.get("confidence", 0.0) or 0.0) for note in post_notes), default=0.0),
    }


def classify_retro_boundary_candidate(note: Dict, boundary_sec: float, retro_band_sec: float) -> str | None:
    onset = float(note.get("onset_time", 0.0) or 0.0)
    offset = float(note.get("offset_time", onset) or onset)

    if abs(onset - boundary_sec) <= retro_band_sec:
        return "onset_band"
    if onset < boundary_sec < offset:
        return "spans_boundary"
    return None


def _audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    samples = audio.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(samples * samples)))


def should_scan_retro_boundary(
    audio: np.ndarray,
    notes: Sequence[Dict],
    chunks: Sequence[Dict],
    boundary_index: int,
    boundary_sample: int,
    boundary_sec: float,
    retro_band_sec: float,
    retro_gate_post_confidence: float,
    retro_gate_min_activity_ratio: float,
) -> Dict:
    coverage = _boundary_note_coverage(notes, boundary_sec=boundary_sec, retro_band_sec=retro_band_sec)
    band_frames = max(1, int(round(retro_band_sec * TARGET_SR)))
    band_start = max(0, boundary_sample - band_frames)
    band_end = min(audio.size, boundary_sample + band_frames)
    seam_rms = _audio_rms(audio[band_start:band_end])

    prev_chunk_rms = float(chunks[boundary_index - 1].get("chunk_rms", 0.0) or 0.0) if 0 <= boundary_index - 1 < len(chunks) else 0.0
    next_chunk_rms = float(chunks[boundary_index].get("chunk_rms", 0.0) or 0.0) if 0 <= boundary_index < len(chunks) else 0.0
    reference_rms = max(prev_chunk_rms, next_chunk_rms, 1e-6)
    seam_activity_ratio = seam_rms / reference_rms if reference_rms > 0.0 else 0.0

    if coverage["pre_count"] == 0 and coverage["post_count"] == 0 and seam_activity_ratio < retro_gate_min_activity_ratio:
        return {
            "scan": False,
            "reason": "no_boundary_activity",
            "seam_rms": seam_rms,
            "seam_activity_ratio": seam_activity_ratio,
            **coverage,
        }

    if coverage["post_count"] > 0 and coverage["post_max_confidence"] >= retro_gate_post_confidence:
        return {
            "scan": False,
            "reason": "covered_post_boundary",
            "seam_rms": seam_rms,
            "seam_activity_ratio": seam_activity_ratio,
            **coverage,
        }

    return {
        "scan": True,
        "reason": "missing_post_boundary" if coverage["post_count"] == 0 else "weak_post_boundary",
        "seam_rms": seam_rms,
        "seam_activity_ratio": seam_activity_ratio,
        **coverage,
    }


def find_retro_repair_target(
    notes: Sequence[Dict],
    candidate: Dict,
    boundary_sec: float,
    onset_tol: float,
) -> tuple[int, str] | None:
    candidate_pitch = int(candidate["midi_note"])
    candidate_onset = float(candidate["onset_time"])

    best_replace: tuple[float, int, str] | None = None
    best_extend: tuple[float, int, str] | None = None

    for idx, existing in enumerate(notes):
        if int(existing["midi_note"]) != candidate_pitch:
            continue

        existing_onset = float(existing.get("onset_time", 0.0) or 0.0)
        existing_offset = float(existing.get("offset_time", existing_onset) or existing_onset)
        onset_delta = abs(existing_onset - candidate_onset)
        if onset_delta <= onset_tol:
            if best_replace is None or onset_delta < best_replace[0]:
                best_replace = (onset_delta, idx, "replace")
            continue

        gap_to_candidate = candidate_onset - existing_offset
        if existing_offset >= (boundary_sec - onset_tol) and -0.02 <= gap_to_candidate <= onset_tol:
            gap_score = abs(gap_to_candidate)
            if best_extend is None or gap_score < best_extend[0]:
                best_extend = (gap_score, idx, "extend")

    if best_replace is not None:
        return best_replace[1], best_replace[2]
    if best_extend is not None:
        return best_extend[1], best_extend[2]
    return None


def merge_retro_candidate_into_note(existing: Dict, candidate: Dict) -> tuple[Dict, bool]:
    updated = dict(existing)
    existing_onset = float(existing.get("onset_time", 0.0) or 0.0)
    existing_offset = float(existing.get("offset_time", existing_onset) or existing_onset)
    existing_confidence = float(existing.get("confidence", 0.0) or 0.0)

    candidate_onset = float(candidate.get("onset_time", 0.0) or 0.0)
    candidate_offset = float(candidate.get("offset_time", candidate_onset) or candidate_onset)
    candidate_confidence = float(candidate.get("confidence", 0.0) or 0.0)

    merged_onset = min(existing_onset, candidate_onset)
    merged_offset = max(existing_offset, candidate_offset)
    merged_confidence = max(existing_confidence, candidate_confidence)

    updated["onset_time"] = round(merged_onset, 6)
    updated["offset_time"] = round(merged_offset, 6)
    updated["duration"] = round(max(0.0, merged_offset - merged_onset), 6)
    updated["confidence"] = merged_confidence

    if candidate_confidence >= existing_confidence:
        for key in ("note_value", "note_value_confidence", "note_value_source", "note_divisions", "dotted"):
            if key in candidate:
                updated[key] = candidate[key]

    changed = (
        abs(updated["onset_time"] - existing_onset) > 1e-6
        or abs(updated["offset_time"] - existing_offset) > 1e-6
        or abs(float(updated.get("confidence", 0.0) or 0.0) - existing_confidence) > 1e-6
    )
    return updated, changed


async def run_live_excerpt(
    audio: np.ndarray,
    adaptive_onset_threshold: bool,
    chunk_seconds: float,
    noise_profile: str,
    fixed_onset_threshold: float | None = None,
    experiment_name: str = FIXED_THRESHOLD_EXPERIMENT,
) -> Dict:
    chunk_frames = max(1, int(round(chunk_seconds * TARGET_SR)))
    session_id = f"experiment-{uuid.uuid4().hex}"
    chunk_summaries: List[Dict] = []
    predicted_notes: List[Dict] = []
    threshold_context = (
        force_live_onset_threshold(fixed_onset_threshold, experiment_name)
        if fixed_onset_threshold is not None
        else nullcontext()
    )

    try:
        with threshold_context:
            for chunk_index, start in enumerate(range(0, audio.size, chunk_frames)):
                chunk_audio = audio[start : start + chunk_frames]
                payload = wav_bytes_from_audio(chunk_audio, TARGET_SR)
                result = await _analyze_uploaded_stream_chunk(
                    session_id,
                    payload,
                    False,
                    noise_profile,
                    True,
                    adaptive_onset_threshold,
                )
                timing = result.get("_timing_ms") or {}
                summary = result.get("analysis_summary") or {}
                chunk_summaries.append(
                    {
                        "chunk_index": chunk_index,
                        "analysis_path": timing.get("analysis_path") or summary.get("analysis_path"),
                        "chunk_total_ms": float(timing.get("chunk_total", 0.0) or 0.0),
                        "real_time_factor": float(timing.get("real_time_factor", 0.0) or 0.0),
                        "onset_threshold": float(summary.get("live_onset_threshold", 0.0) or 0.0),
                        "profile": str(summary.get("live_onset_threshold_profile") or "unknown"),
                        "experiment": str(summary.get("live_onset_threshold_experiment") or "unknown"),
                        "chunk_rms": float(timing.get("neural_chunk_rms", 0.0) or 0.0),
                        "chunk_peak": float(timing.get("neural_chunk_peak", 0.0) or 0.0),
                        "chunk_crest_factor": float(timing.get("neural_chunk_crest_factor", 0.0) or 0.0),
                        "neural_total_ms": float(timing.get("neural_total", 0.0) or 0.0),
                    }
                )
                predicted_notes.extend(normalize_predicted_notes(result.get("notes") or []))
    finally:
        _clear_stream_session(session_id)

    predicted_notes.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return {
        "notes": predicted_notes,
        "chunks": chunk_summaries,
    }


def run_direct_live_neural_excerpt(
    audio: np.ndarray,
    adaptive_onset_threshold: bool,
    fixed_onset_threshold: float | None = None,
    experiment_name: str = RETRO_CORRECTION_EXPERIMENT,
) -> Dict:
    threshold_context = (
        force_live_onset_threshold(fixed_onset_threshold, experiment_name)
        if fixed_onset_threshold is not None
        else nullcontext()
    )

    with threshold_context:
        result = detect_note_module.analyze_audio_live_neural(
            audio,
            TARGET_SR,
            False,
            60,
            "cuda",
            adaptive_onset_threshold,
        )

    if result.get("error"):
        return {
            "notes": [],
            "analysis_summary": result.get("analysis_summary", {}),
            "timing": result.get("_timing_ms", {}),
            "error": str(result.get("error")),
        }

    normalized_notes = normalize_predicted_notes(_dedupe_note_events(result.get("notes") or []))
    return {
        "notes": normalized_notes,
        "analysis_summary": result.get("analysis_summary", {}),
        "timing": result.get("_timing_ms", {}),
        "error": None,
    }


async def run_retro_correction_excerpt(
    audio: np.ndarray,
    baseline_run: Dict,
    baseline_threshold: float,
    chunk_seconds: float,
    retro_window_sec: float,
    retro_band_sec: float,
    retro_min_confidence: float,
    retro_match_onset_tol: float,
    retro_boundary_gate: bool,
    retro_gate_post_confidence: float,
    retro_gate_min_activity_ratio: float,
    retro_threshold: float | None = None,
) -> Dict:
    rounded_baseline = round(float(baseline_threshold), 3)
    rounded_retro_threshold = round(
        float(retro_threshold if retro_threshold is not None else baseline_threshold),
        3,
    )

    corrected_notes = [dict(note) for note in baseline_run["notes"]]
    corrected_chunks = [dict(chunk) for chunk in baseline_run["chunks"]]
    extra_notes: List[Dict] = []

    chunk_frames = max(1, int(round(chunk_seconds * TARGET_SR)))
    total_duration_sec = float(audio.size) / TARGET_SR if TARGET_SR > 0 else 0.0

    boundaries_considered = 0
    boundaries_scanned = 0
    boundaries_skipped = 0
    seam_errors = 0
    seam_candidates = 0
    spanning_boundary_candidates = 0
    extras_added = 0
    replaced_existing = 0
    extended_existing = 0
    updated_extra_notes = 0
    rejected_existing = 0
    rejected_confidence = 0
    rejected_seam_band = 0
    rejected_duplicate_extra = 0
    skipped_no_boundary_activity = 0
    skipped_covered_post_boundary = 0
    scanned_missing_post_boundary = 0
    scanned_weak_post_boundary = 0
    retro_total_ms = 0.0

    for boundary_index, boundary_sample in enumerate(range(chunk_frames, audio.size, chunk_frames), start=1):
        boundaries_considered += 1
        boundary_sec = float(boundary_sample) / TARGET_SR

        if retro_boundary_gate:
            gate_decision = should_scan_retro_boundary(
                audio,
                notes=[*corrected_notes, *extra_notes],
                chunks=corrected_chunks,
                boundary_index=boundary_index,
                boundary_sample=boundary_sample,
                boundary_sec=boundary_sec,
                retro_band_sec=retro_band_sec,
                retro_gate_post_confidence=retro_gate_post_confidence,
                retro_gate_min_activity_ratio=retro_gate_min_activity_ratio,
            )
            if not gate_decision["scan"]:
                boundaries_skipped += 1
                if gate_decision["reason"] == "no_boundary_activity":
                    skipped_no_boundary_activity += 1
                elif gate_decision["reason"] == "covered_post_boundary":
                    skipped_covered_post_boundary += 1
                continue

            if gate_decision["reason"] == "missing_post_boundary":
                scanned_missing_post_boundary += 1
            elif gate_decision["reason"] == "weak_post_boundary":
                scanned_weak_post_boundary += 1

        seam_start_sec = max(0.0, boundary_sec - retro_window_sec)
        seam_end_sec = min(total_duration_sec, boundary_sec + retro_window_sec)
        seam_start_sample = max(0, int(round(seam_start_sec * TARGET_SR)))
        seam_end_sample = min(audio.size, int(round(seam_end_sec * TARGET_SR)))
        seam_audio = audio[seam_start_sample:seam_end_sample]
        if seam_audio.size == 0:
            continue

        boundaries_scanned += 1
        seam_run = run_direct_live_neural_excerpt(
            seam_audio,
            adaptive_onset_threshold=False,
            fixed_onset_threshold=rounded_retro_threshold,
            experiment_name=RETRO_CORRECTION_EXPERIMENT,
        )
        if seam_run.get("error"):
            seam_errors += 1
            continue

        seam_timing = seam_run.get("timing") or {}
        seam_total_ms = float(seam_timing.get("neural_total", 0.0) or 0.0)
        retro_total_ms += seam_total_ms

        if corrected_chunks:
            target_chunk_index = min(boundary_index, len(corrected_chunks) - 1)
            corrected_chunks[target_chunk_index]["chunk_total_ms"] = float(
                corrected_chunks[target_chunk_index].get("chunk_total_ms", 0.0) or 0.0
            ) + seam_total_ms
            corrected_chunks[target_chunk_index]["real_time_factor"] = float(
                corrected_chunks[target_chunk_index].get("real_time_factor", 0.0) or 0.0
            ) + (seam_total_ms / max(1.0, chunk_seconds * 1000.0))

        for seam_note in seam_run["notes"]:
            candidate = dict(seam_note)
            candidate["onset_time"] = round(float(candidate["onset_time"]) + seam_start_sec, 6)
            candidate["offset_time"] = round(float(candidate["offset_time"]) + seam_start_sec, 6)
            candidate["duration"] = round(
                max(0.0, float(candidate["offset_time"]) - float(candidate["onset_time"])),
                6,
            )

            boundary_candidate_reason = classify_retro_boundary_candidate(
                candidate,
                boundary_sec=boundary_sec,
                retro_band_sec=retro_band_sec,
            )
            if boundary_candidate_reason is None:
                rejected_seam_band += 1
                continue

            seam_candidates += 1
            if boundary_candidate_reason == "spans_boundary":
                spanning_boundary_candidates += 1

            if float(candidate.get("confidence", 0.0) or 0.0) < retro_min_confidence:
                rejected_confidence += 1
                continue

            existing_target = find_retro_repair_target(
                corrected_notes,
                candidate,
                boundary_sec=boundary_sec,
                onset_tol=retro_match_onset_tol,
            )
            if existing_target is not None:
                target_index, repair_mode = existing_target
                merged_note, changed = merge_retro_candidate_into_note(corrected_notes[target_index], candidate)
                if not changed:
                    rejected_existing += 1
                    continue
                merged_note["selection_reason"] = (
                    "retro_seam_replace" if repair_mode == "replace" else "retro_seam_extend"
                )
                merged_note["source_boundary_sec"] = round(boundary_sec, 6)
                corrected_notes[target_index] = merged_note
                if repair_mode == "replace":
                    replaced_existing += 1
                else:
                    extended_existing += 1
                continue

            extra_target = find_retro_repair_target(
                extra_notes,
                candidate,
                boundary_sec=boundary_sec,
                onset_tol=retro_match_onset_tol,
            )
            if extra_target is not None:
                extra_index, repair_mode = extra_target
                merged_extra, changed = merge_retro_candidate_into_note(extra_notes[extra_index], candidate)
                if not changed:
                    rejected_duplicate_extra += 1
                    continue
                merged_extra["selection_reason"] = (
                    "retro_seam_replace" if repair_mode == "replace" else "retro_seam_extend"
                )
                merged_extra["source_boundary_sec"] = round(boundary_sec, 6)
                extra_notes[extra_index] = merged_extra
                updated_extra_notes += 1
                continue

            candidate["selection_reason"] = "retro_seam_extra"
            candidate["source_boundary_sec"] = round(boundary_sec, 6)
            extra_notes.append(candidate)
            extras_added += 1

    corrected_notes = sorted(
        [*corrected_notes, *extra_notes],
        key=lambda event: (event["onset_time"], event["midi_note"]),
    )

    selected_thresholds = sorted({rounded_baseline, rounded_retro_threshold})
    return {
        "notes": corrected_notes,
        "chunks": corrected_chunks,
        "summary_overrides": {
            "profile_counts": {RETRO_CORRECTION_EXPERIMENT: len(corrected_chunks)},
            "profile_hit_rate": 1.0 if corrected_chunks else 0.0,
            "selected_thresholds": selected_thresholds,
            "selection_stats": {
                "strategy": "seam_boundary_reanalysis",
                "baseline_threshold": rounded_baseline,
                "retro_threshold": rounded_retro_threshold,
                "retro_window_sec": float(retro_window_sec),
                "retro_band_sec": float(retro_band_sec),
                "retro_min_confidence": float(retro_min_confidence),
                "retro_match_onset_tol": float(retro_match_onset_tol),
                "retro_boundary_gate": bool(retro_boundary_gate),
                "retro_gate_post_confidence": float(retro_gate_post_confidence),
                "retro_gate_min_activity_ratio": float(retro_gate_min_activity_ratio),
                "boundaries_considered": boundaries_considered,
                "boundaries_scanned": boundaries_scanned,
                "boundaries_skipped": boundaries_skipped,
                "seam_errors": seam_errors,
                "seam_candidates": seam_candidates,
                "spanning_boundary_candidates": spanning_boundary_candidates,
                "extras_added": extras_added,
                "replaced_existing": replaced_existing,
                "extended_existing": extended_existing,
                "updated_extra_notes": updated_extra_notes,
                "rejected_existing": rejected_existing,
                "rejected_confidence": rejected_confidence,
                "rejected_seam_band": rejected_seam_band,
                "rejected_duplicate_extra": rejected_duplicate_extra,
                "skipped_no_boundary_activity": skipped_no_boundary_activity,
                "skipped_covered_post_boundary": skipped_covered_post_boundary,
                "scanned_missing_post_boundary": scanned_missing_post_boundary,
                "scanned_weak_post_boundary": scanned_weak_post_boundary,
                "retro_total_ms": round(retro_total_ms, 3),
            },
        },
    }


def summarize_run(
    run: Dict,
    gt_notes: Sequence[Dict],
    chunk_seconds: float,
    boundary_band_sec: float,
    reference_bpm: float,
) -> Dict:
    note_metrics = compute_note_metrics(run["notes"], gt_notes)
    offset_metrics = compute_offset_metrics(run["notes"], gt_notes)
    note_value_metrics = compute_note_value_metrics(run["notes"], gt_notes, reference_bpm=reference_bpm)
    duplicate_metrics = compute_duplicate_metrics(run["notes"])
    boundary_metrics = compute_boundary_miss_metrics(
        run["notes"],
        gt_notes,
        chunk_seconds=chunk_seconds,
        boundary_band_sec=boundary_band_sec,
    )
    chunk_totals = [chunk["chunk_total_ms"] for chunk in run["chunks"] if chunk["chunk_total_ms"] > 0.0]
    rtfs = [chunk["real_time_factor"] for chunk in run["chunks"] if chunk["real_time_factor"] > 0.0]
    profiles = Counter(chunk["profile"] for chunk in run["chunks"])
    nonbaseline_profiles = sum(
        count
        for profile, count in profiles.items()
        if profile not in {"baseline_nominal", "fixed_baseline", "unknown"}
    )

    summary = {
        **note_metrics,
        **offset_metrics,
        **note_value_metrics,
        **duplicate_metrics,
        **boundary_metrics,
        "reference_bpm": float(reference_bpm),
        "chunk_count": len(run["chunks"]),
        "profile_counts": dict(sorted(profiles.items())),
        "profile_hit_rate": (nonbaseline_profiles / len(run["chunks"])) if run["chunks"] else 0.0,
        "avg_chunk_total_ms": float(np.mean(chunk_totals)) if chunk_totals else 0.0,
        "p95_chunk_total_ms": safe_percentile(chunk_totals, 95),
        "avg_real_time_factor": float(np.mean(rtfs)) if rtfs else 0.0,
        "selected_thresholds": sorted(
            {
                round(float(chunk["onset_threshold"]), 3)
                for chunk in run["chunks"]
                if chunk["onset_threshold"] > 0.0
            }
        ),
    }

    overrides = run.get("summary_overrides") or {}
    for key in ("profile_counts", "profile_hit_rate", "selected_thresholds", "selection_stats"):
        if key in overrides:
            summary[key] = overrides[key]

    return summary


def print_run_summary(label: str, metrics: Dict) -> None:
    print(f"    {label}:")
    print(
        "      note metrics: "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"matched={metrics['matched']}/{metrics['ground_truth']} pred={metrics['predicted']}"
    )
    print(
        "      timing metrics: "
        f"offset_f1={metrics['offset_f1']:.3f} rhythm_precision={metrics['rhythm_precision']:.3f} "
        f"note_value_acc={metrics['note_value_accuracy']:.3f} note_value_n={metrics['note_value_matched']} "
        f"avg_beat_error={metrics['note_value_avg_beat_error']:.3f}"
    )
    print(
        "      duplicate metrics: "
        f"duplicates={metrics['duplicates']} per_100_notes={metrics['duplicates_per_100_notes']:.2f}"
    )
    print(
        "      boundary metrics: "
        f"gt={metrics['boundary_gt_notes']} matched={metrics['boundary_matched_notes']} "
        f"missed={metrics['boundary_missed_notes']} miss_rate={metrics['boundary_miss_rate']:.3f}"
    )
    print(
        "      latency: "
        f"avg_chunk_ms={metrics['avg_chunk_total_ms']:.2f} p95_chunk_ms={metrics['p95_chunk_total_ms']:.2f} "
        f"avg_rtf={metrics['avg_real_time_factor']:.3f}"
    )
    print(
        "      threshold profiles: "
        f"hit_rate={metrics['profile_hit_rate']:.3f} thresholds={metrics['selected_thresholds']} "
        f"profiles={metrics['profile_counts']}"
    )
    if metrics.get("selection_stats"):
        selection = metrics["selection_stats"]
        if selection.get("strategy") == "seam_boundary_reanalysis":
            print(
                "      selection: "
                f"baseline={selection['baseline_threshold']:.3f} retro={selection['retro_threshold']:.3f} "
                f"window={selection['retro_window_sec']:.2f}s band={selection['retro_band_sec']:.2f}s gate={selection['retro_boundary_gate']} "
                f"considered={selection['boundaries_considered']} scanned={selection['boundaries_scanned']} skipped={selection['boundaries_skipped']} "
                f"candidates={selection['seam_candidates']} spans={selection.get('spanning_boundary_candidates', 0)} extras={selection['extras_added']} "
                f"replaced={selection['replaced_existing']} extended={selection['extended_existing']}"
            )


def print_delta(reference_label: str, reference: Dict, candidate_label: str, candidate: Dict) -> None:
    print(
        f"    delta {candidate_label}-{reference_label}: "
        f"precision={candidate['precision'] - reference['precision']:+.3f} "
        f"recall={candidate['recall'] - reference['recall']:+.3f} "
        f"f1={candidate['f1'] - reference['f1']:+.3f} "
        f"offset_f1={candidate['offset_f1'] - reference['offset_f1']:+.3f} "
        f"note_value_acc={candidate['note_value_accuracy'] - reference['note_value_accuracy']:+.3f} "
        f"dup_per_100={candidate['duplicates_per_100_notes'] - reference['duplicates_per_100_notes']:+.2f} "
        f"boundary_miss_rate={candidate['boundary_miss_rate'] - reference['boundary_miss_rate']:+.3f} "
        f"p95_chunk_ms={candidate['p95_chunk_total_ms'] - reference['p95_chunk_total_ms']:+.2f}"
    )


async def warmup_live_path(chunk_seconds: float) -> None:
    warmup_audio = np.zeros(max(1, int(round(chunk_seconds * TARGET_SR))), dtype=np.float32)
    await run_live_excerpt(warmup_audio, adaptive_onset_threshold=False, chunk_seconds=chunk_seconds, noise_profile="balanced")


async def run_experiment(args: argparse.Namespace) -> Dict:
    pieces = load_index(args.split)
    candidates = build_candidates(
        pieces,
        num_pieces=args.pieces,
        scan_seconds=args.scan_seconds,
        clip_seconds=args.clip_seconds,
        step_seconds=args.step_seconds,
        min_notes=args.min_notes,
    )
    selected = choose_excerpts(candidates, args.categories)

    if args.warmup:
        print("Warming live neural path...")
        await warmup_live_path(args.chunk_seconds)

    results = {
        "config": {
            "split": args.split,
            "pieces": args.pieces,
            "scan_seconds": args.scan_seconds,
            "clip_seconds": args.clip_seconds,
            "chunk_seconds": args.chunk_seconds,
            "step_seconds": args.step_seconds,
            "min_notes": args.min_notes,
            "categories": list(args.categories),
            "noise_profile": args.noise_profile,
            "run_retro_correction": bool(args.run_retro_correction),
            "retro_window_sec": float(args.retro_window_sec),
            "retro_band_sec": float(args.retro_band_sec),
            "retro_min_confidence": float(args.retro_min_confidence),
            "retro_match_onset_tol": float(args.retro_match_onset_tol),
            "retro_boundary_gate": bool(args.retro_boundary_gate),
            "retro_gate_post_confidence": float(args.retro_gate_post_confidence),
            "retro_gate_min_activity_ratio": float(args.retro_gate_min_activity_ratio),
            "retro_threshold": args.retro_threshold,
            "eval_boundary_band_sec": float(args.eval_boundary_band_sec),
        },
        "clips": {},
    }

    for category in args.categories:
        clip = selected.get(category)
        if clip is None:
            continue

        audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
        reference_bpm = get_midi_reference_bpm(clip["midi_path"])
        gt_notes = slice_gt_notes(
            load_midi_notes(clip["midi_path"]),
            clip["start_sec"],
            clip["end_sec"],
        )

        control_run = await run_live_excerpt(
            audio,
            adaptive_onset_threshold=False,
            chunk_seconds=args.chunk_seconds,
            noise_profile=args.noise_profile,
        )
        treatment_run = await run_live_excerpt(
            audio,
            adaptive_onset_threshold=True,
            chunk_seconds=args.chunk_seconds,
            noise_profile=args.noise_profile,
        )

        control_summary = summarize_run(
            control_run,
            gt_notes,
            chunk_seconds=args.chunk_seconds,
            boundary_band_sec=args.eval_boundary_band_sec,
            reference_bpm=reference_bpm,
        )
        treatment_summary = summarize_run(
            treatment_run,
            gt_notes,
            chunk_seconds=args.chunk_seconds,
            boundary_band_sec=args.eval_boundary_band_sec,
            reference_bpm=reference_bpm,
        )

        clip_results = {
            "clip": clip,
            "ground_truth_notes": len(gt_notes),
            "reference_bpm": round(float(reference_bpm), 3),
            "control": control_summary,
            "treatment": treatment_summary,
        }

        retro_summary = None
        control_baseline_threshold = 0.38
        observed_control_thresholds = control_summary.get("selected_thresholds") or []
        if observed_control_thresholds:
            control_baseline_threshold = round(float(observed_control_thresholds[-1]), 3)

        if args.run_retro_correction:
            retro_run = await run_retro_correction_excerpt(
                audio,
                baseline_run=control_run,
                baseline_threshold=control_baseline_threshold,
                chunk_seconds=args.chunk_seconds,
                retro_window_sec=args.retro_window_sec,
                retro_band_sec=args.retro_band_sec,
                retro_min_confidence=args.retro_min_confidence,
                retro_match_onset_tol=args.retro_match_onset_tol,
                retro_boundary_gate=args.retro_boundary_gate,
                retro_gate_post_confidence=args.retro_gate_post_confidence,
                retro_gate_min_activity_ratio=args.retro_gate_min_activity_ratio,
                retro_threshold=args.retro_threshold,
            )
            retro_summary = summarize_run(
                retro_run,
                gt_notes,
                chunk_seconds=args.chunk_seconds,
                boundary_band_sec=args.eval_boundary_band_sec,
                reference_bpm=reference_bpm,
            )
            clip_results["retro_correction"] = retro_summary

        results["clips"][category] = clip_results

        print(f"\n[{category.upper()}] {clip['title']}")
        print(
            "  excerpt: "
            f"start={clip['start_sec']:.2f}s end={clip['end_sec']:.2f}s duration={clip['duration_sec']:.2f}s "
            f"gt_notes={len(gt_notes)} rms={clip['rms']:.4f} peak={clip['peak']:.4f} "
            f"crest={clip['crest_factor']:.3f} rms_std={clip['rms_std']:.4f}"
        )
        print_run_summary("control", control_summary)
        print_run_summary("treatment", treatment_summary)
        print_delta("control", control_summary, "treatment", treatment_summary)
        if retro_summary is not None:
            print_run_summary("retro_correction", retro_summary)
            print_delta("control", control_summary, "retro_correction", retro_summary)

    return results


def parse_args() -> argparse.Namespace:
    def parse_cli_bool(value):
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = argparse.ArgumentParser(description="Benchmark the live retro-correction experiment on local MAESTRO clips")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test", help="MAESTRO split index to sample from")
    parser.add_argument("--pieces", type=int, default=2, help="How many indexed pieces to scan for candidate excerpts")
    parser.add_argument("--scan-seconds", type=float, default=45.0, help="How many seconds from the start of each piece to scan")
    parser.add_argument("--clip-seconds", type=float, default=6.0, help="Length of each selected benchmark excerpt")
    parser.add_argument("--chunk-seconds", type=float, default=0.6, help="Chunk size for the live-path simulation")
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Sliding-window step while searching for candidate excerpts")
    parser.add_argument("--min-notes", type=int, default=8, help="Minimum ground-truth note count required for a candidate excerpt")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(DEFAULT_CATEGORIES),
        default=list(DEFAULT_CATEGORIES),
        help="Which excerpt categories to benchmark",
    )
    parser.add_argument("--noise-profile", choices=["open", "balanced", "clean"], default="balanced", help="Noise profile to pass into the live chunk analyzer")
    parser.add_argument("--run-retro-correction", type=parse_cli_bool, nargs="?", const=True, default=True, help="Benchmark a seam re-analysis retro-correction pass on top of the control baseline")
    parser.add_argument("--no-run-retro-correction", dest="run_retro_correction", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--retro-window-sec", type=float, default=0.20, help="Seconds of audio to include on each side of a chunk boundary during retro-correction")
    parser.add_argument("--retro-band-sec", type=float, default=0.10, help="Only admit seam-pass notes whose onset lies within this many seconds of the chunk boundary")
    parser.add_argument("--retro-min-confidence", type=float, default=0.35, help="Minimum note confidence required before a seam-pass note can be added")
    parser.add_argument("--retro-match-onset-tol", type=float, default=0.05, help="Onset tolerance when deciding whether a seam-pass note is already present in the baseline")
    parser.add_argument("--retro-boundary-gate", type=parse_cli_bool, nargs="?", const=True, default=True, help="Skip seam re-analysis when the baseline already has strong post-boundary coverage")
    parser.add_argument("--no-retro-boundary-gate", dest="retro_boundary_gate", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--retro-gate-post-confidence", type=float, default=0.65, help="Post-boundary confidence that counts as already covered and skips seam re-analysis")
    parser.add_argument("--retro-gate-min-activity-ratio", type=float, default=0.35, help="If no boundary notes are present, skip seam re-analysis when local boundary RMS is below this fraction of the adjacent chunk RMS")
    parser.add_argument("--retro-threshold", type=float, default=None, help="Optional fixed onset threshold for seam re-analysis; defaults to the observed control threshold")
    parser.add_argument("--eval-boundary-band-sec", type=float, default=0.10, help="Boundary band used when computing boundary miss rate in the summary metrics")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write the full results JSON")
    parser.add_argument("--warmup", type=parse_cli_bool, nargs="?", const=True, default=True, help="Warm the live neural path before measuring")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_experiment(args))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()