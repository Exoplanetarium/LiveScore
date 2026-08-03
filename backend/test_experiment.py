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
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(__file__))

import detect_note as detect_note_module  # noqa: E402
from live_rhythm import _cluster_live_onset_times
from live_rhythm import delete_session as delete_live_session  # noqa: E402
from live_rhythm import get_or_create_session as get_live_session
from live_rhythm import quantize_batch_coarse
from main import _analyze_uploaded_stream_chunk  # noqa: E402
from main import _clear_stream_session, _dedupe_note_events

TARGET_SR = 44100
DEFAULT_CATEGORIES = ("quiet", "loud", "mixed")
DEFAULT_SELECTION_STRATEGY = "legacy_categories"
DIVERSE_SUITE_SELECTION_STRATEGY = "diverse_suite"
DEFAULT_TARGET_CLIPS = 48
DEFAULT_MAX_CLIPS_PER_PIECE = 2
DEFAULT_MAX_CLIPS_PER_TITLE = 2
ONSET_CLUSTER_TOLERANCE_SEC = 0.05
DEFAULT_STRICT_ONSET_TOLS_MS = (10, 20, 30)
FAILURE_BUCKET_ORDER = (
    "runtime_only_win",
    "boundary_miss_failure",
    "high_revision_slow_stabilization",
    "note_value_offset_failure",
    "retro_regression",
)
BOUNDARY_DIAGNOSTIC_PRINT_LIMIT = 3
SELECTION_FEATURE_KEYS = (
    "rms",
    "rms_std",
    "note_density",
    "onset_density",
    "mean_notes_per_onset",
    "ioi_cv",
    "boundary_event_rate",
)
FIXED_THRESHOLD_EXPERIMENT = "fixed_threshold"
RETRO_CORRECTION_EXPERIMENT = "retro_correction_seam_v1"
# The live loudness-based onset selector was removed (48-clip ablation showed it
# inert, |dF1| <= 0.0004 vs a fixed base threshold). The live base threshold is
# now env-driven, so forcing a fixed threshold overrides the base-threshold env
# vars instead of monkeypatching the (now-deleted) selector.
_LIVE_ONSET_BASE_ENV_VARS = ("LIVE_ENHANCED_ONSET_BASE", "LIVE_ONSET_BASE")
PAIRED_STATS_BOOTSTRAP_SAMPLES = 20000
PAIRED_STATS_RANDOM_SEED = 0
PAIRED_STATS_EPSILON = 1e-12
PAIRED_DISPLAY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("display_cluster_f1", "display_cluster_f1"),
    ("display_note_f1", "display_note_f1"),
    ("display_offset_f1", "display_offset_f1"),
    ("display_score_edit_accuracy", "display_score_edit_accuracy"),
    ("display_score_exact_token_f1", "display_score_exact_token_f1"),
)

REFERENCE_MUSICXML_FIELDS = (
    "reference_musicxml_path",
    "gt_musicxml_path",
    "musicxml_path",
    "score_musicxml_path",
    "score_path",
)
REFERENCE_MUSICXML_EXTENSIONS = (".musicxml", ".xml")


@contextmanager
def force_live_onset_threshold(
    onset_threshold: float,
    experiment_name: str = FIXED_THRESHOLD_EXPERIMENT,
) -> Iterator[None]:
    # ``experiment_name`` is retained for call compatibility; the live decode now
    # reports a fixed profile/experiment label itself, so it is not propagated.
    forced = f"{float(onset_threshold):.6f}"
    previous = {name: os.environ.get(name) for name in _LIVE_ONSET_BASE_ENV_VARS}
    for name in _LIVE_ONSET_BASE_ENV_VARS:
        os.environ[name] = forced
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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


def cluster_note_onsets(
    notes: Sequence[Dict],
    onset_tolerance_sec: float = ONSET_CLUSTER_TOLERANCE_SEC,
) -> List[List[Dict]]:
    sorted_notes = sorted(notes, key=lambda event: (event.get("onset_time", 0.0), event.get("midi_note", 0)))
    if not sorted_notes:
        return []

    clusters: List[List[Dict]] = []
    current_cluster: List[Dict] = []
    current_anchor = None

    for note in sorted_notes:
        onset = float(note.get("onset_time", 0.0) or 0.0)
        if current_anchor is None or (onset - current_anchor) > onset_tolerance_sec:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [note]
            current_anchor = onset
        else:
            current_cluster.append(note)

    if current_cluster:
        clusters.append(current_cluster)
    return clusters


def compute_excerpt_selection_features(
    gt_notes: Sequence[Dict],
    duration_sec: float,
    chunk_seconds: float,
    boundary_band_sec: float,
) -> Dict:
    duration_sec = max(0.0, float(duration_sec or 0.0))
    clusters = cluster_note_onsets(gt_notes)
    onset_times = [float(cluster[0].get("onset_time", 0.0) or 0.0) for cluster in clusters]
    notes_per_onset = [len(cluster) for cluster in clusters]
    iois = np.diff(np.asarray(onset_times, dtype=np.float64)) if len(onset_times) >= 2 else np.asarray([], dtype=np.float64)

    note_density = (len(gt_notes) / duration_sec) if duration_sec > 0.0 else 0.0
    onset_density = (len(onset_times) / duration_sec) if duration_sec > 0.0 else 0.0
    mean_notes_per_onset = float(np.mean(np.asarray(notes_per_onset, dtype=np.float64))) if notes_per_onset else 0.0
    max_notes_per_onset = int(max(notes_per_onset)) if notes_per_onset else 0
    ioi_cv = 0.0
    if iois.size >= 2:
        mean_ioi = float(np.mean(iois))
        if mean_ioi > 1e-9:
            ioi_cv = float(np.std(iois) / mean_ioi)

    boundary_events = sum(
        1
        for onset_time in onset_times
        if is_boundary_note(onset_time, chunk_seconds, boundary_band_sec)
    )
    boundary_event_rate = (boundary_events / len(onset_times)) if onset_times else 0.0
    long_sustain_rate = (
        sum(1 for note in gt_notes if float(note.get("duration", 0.0) or 0.0) >= 0.75) / len(gt_notes)
        if gt_notes
        else 0.0
    )

    return {
        "note_density": note_density,
        "onset_density": onset_density,
        "onset_event_count": len(onset_times),
        "mean_notes_per_onset": mean_notes_per_onset,
        "max_notes_per_onset": max_notes_per_onset,
        "ioi_cv": ioi_cv,
        "boundary_event_count": boundary_events,
        "boundary_event_rate": boundary_event_rate,
        "long_sustain_rate": long_sustain_rate,
    }


def build_candidates(
    pieces: Sequence[Dict],
    num_pieces: int,
    scan_seconds: float,
    clip_seconds: float,
    step_seconds: float,
    min_notes: int,
    chunk_seconds: float,
    boundary_band_sec: float,
) -> List[Dict]:
    candidates: List[Dict] = []

    for piece in list(pieces)[: max(1, int(num_pieces))]:
        audio_path = str(piece.get("audio") or piece.get("audio_path") or "")
        midi_path = str(piece.get("midi") or piece.get("midi_path") or "")
        if not audio_path or not midi_path:
            continue

        title = str(piece.get("title") or Path(midi_path).stem)
        piece_id = f"{Path(midi_path).stem}"
        reference_musicxml_path = ""
        for field in REFERENCE_MUSICXML_FIELDS:
            if piece.get(field):
                reference_musicxml_path = str(piece.get(field) or "")
                break
        piece_duration = float(piece.get("duration", 0.0) or 0.0)
        scan_limit = min(scan_seconds, piece_duration) if piece_duration > 0.0 else scan_seconds
        if scan_limit < clip_seconds:
            continue

        midi_notes = load_midi_notes(midi_path)
        reference_bpm = get_midi_reference_bpm(midi_path)
        start_sec = 0.0
        while start_sec + clip_seconds <= scan_limit + 1e-9:
            end_sec = start_sec + clip_seconds
            gt_notes = slice_gt_notes(midi_notes, start_sec, end_sec)
            if len(gt_notes) < min_notes:
                start_sec += step_seconds
                continue

            audio_excerpt = load_audio_excerpt(audio_path, start_sec, clip_seconds, TARGET_SR)
            stats = excerpt_statistics(audio_excerpt)
            selection_features = compute_excerpt_selection_features(
                gt_notes,
                duration_sec=clip_seconds,
                chunk_seconds=chunk_seconds,
                boundary_band_sec=boundary_band_sec,
            )
            candidates.append(
                {
                    "title": title,
                    "piece_id": piece_id,
                    "audio_path": audio_path,
                    "midi_path": midi_path,
                    "reference_musicxml_path": reference_musicxml_path,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "duration_sec": round(clip_seconds, 6),
                    "gt_note_count": len(gt_notes),
                    "reference_bpm": round(float(reference_bpm), 3),
                    **stats,
                    "selection_features": selection_features,
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


def _candidate_feature_vector(candidate: Dict) -> np.ndarray:
    selection_features = candidate.get("selection_features") or {}
    return np.asarray(
        [
            float(candidate.get("rms", 0.0) or 0.0),
            float(candidate.get("rms_std", 0.0) or 0.0),
            float(selection_features.get("note_density", 0.0) or 0.0),
            float(selection_features.get("onset_density", 0.0) or 0.0),
            float(selection_features.get("mean_notes_per_onset", 0.0) or 0.0),
            float(selection_features.get("ioi_cv", 0.0) or 0.0),
            float(selection_features.get("boundary_event_rate", 0.0) or 0.0),
        ],
        dtype=np.float64,
    )


def _normalize_title_key(title: str) -> str:
    normalized = re.sub(r"\([^)]*\)", " ", str(title or "").lower())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "untitled"


def _clips_overlap_or_touch(existing_clip: Dict, candidate: Dict) -> bool:
    existing_start = float(existing_clip.get("start_sec", 0.0) or 0.0)
    existing_end = float(existing_clip.get("end_sec", existing_start) or existing_start)
    candidate_start = float(candidate.get("start_sec", 0.0) or 0.0)
    candidate_end = float(candidate.get("end_sec", candidate_start) or candidate_start)
    return not (candidate_end <= existing_start or candidate_start >= existing_end)


def choose_diverse_suite(
    candidates: Sequence[Dict],
    target_clips: int,
    max_clips_per_piece: int,
    max_clips_per_title: int,
) -> Dict[str, Dict]:
    ordered_candidates = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            str(candidate.get("piece_id") or ""),
            str(candidate.get("audio_path") or ""),
            float(candidate.get("start_sec", 0.0) or 0.0),
        ),
    )
    if not ordered_candidates:
        raise RuntimeError("No candidates available for suite selection")

    if target_clips <= 0:
        raise ValueError("target_clips must be positive")
    if max_clips_per_piece <= 0:
        raise ValueError("max_clips_per_piece must be positive")
    if max_clips_per_title <= 0:
        raise ValueError("max_clips_per_title must be positive")

    feature_matrix = np.vstack([_candidate_feature_vector(candidate) for candidate in ordered_candidates])
    feature_min = np.min(feature_matrix, axis=0)
    feature_max = np.max(feature_matrix, axis=0)
    feature_span = np.where((feature_max - feature_min) > 1e-9, feature_max - feature_min, 1.0)
    normalized = (feature_matrix - feature_min) / feature_span

    piece_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    selected_indices: List[int] = []
    selected_lookup: set[int] = set()
    selected_by_piece: Dict[str, List[Dict]] = {}

    def can_select(index: int) -> bool:
        if index in selected_lookup:
            return False
        candidate = ordered_candidates[index]
        piece_id = str(candidate.get("piece_id") or candidate.get("audio_path") or index)
        title_key = _normalize_title_key(str(candidate.get("title") or ""))
        if title_counts[title_key] >= max_clips_per_title:
            return False
        for selected_clip in selected_by_piece.get(piece_id, []):
            if _clips_overlap_or_touch(selected_clip, candidate):
                return False
        return piece_counts[piece_id] < max_clips_per_piece

    def add_index(index: int) -> None:
        if not can_select(index):
            return
        candidate = ordered_candidates[index]
        selected_indices.append(index)
        selected_lookup.add(index)
        piece_id = str(candidate.get("piece_id") or candidate.get("audio_path") or index)
        title_key = _normalize_title_key(str(candidate.get("title") or ""))
        piece_counts[piece_id] += 1
        title_counts[title_key] += 1
        selected_by_piece.setdefault(piece_id, []).append(candidate)

    seed_indices: List[int] = []
    for feature_idx in range(normalized.shape[1]):
        seed_indices.append(int(np.argmin(normalized[:, feature_idx])))
        seed_indices.append(int(np.argmax(normalized[:, feature_idx])))

    for index in seed_indices:
        add_index(index)
        if len(selected_indices) >= min(target_clips, len(ordered_candidates)):
            break

    while len(selected_indices) < min(target_clips, len(ordered_candidates)):
        best_index = None
        best_score = None
        for index, candidate in enumerate(ordered_candidates):
            if not can_select(index):
                continue

            if selected_indices:
                diversity_score = min(
                    float(np.linalg.norm(normalized[index] - normalized[selected_index]))
                    for selected_index in selected_indices
                )
            else:
                diversity_score = float(np.linalg.norm(normalized[index] - np.mean(normalized, axis=0)))

            piece_id = str(candidate.get("piece_id") or candidate.get("audio_path") or index)
            piece_bonus = 0.15 if piece_counts[piece_id] == 0 else 0.0
            selection_features = candidate.get("selection_features") or {}
            boundary_bonus = 0.05 * float(selection_features.get("boundary_event_rate", 0.0) or 0.0)
            rubato_bonus = 0.03 * float(selection_features.get("ioi_cv", 0.0) or 0.0)
            gt_bonus = 0.01 * float(candidate.get("gt_note_count", 0) or 0)
            total_score = diversity_score + piece_bonus + boundary_bonus + rubato_bonus + gt_bonus
            candidate_score = (
                total_score,
                diversity_score,
                piece_bonus,
                boundary_bonus,
                rubato_bonus,
                gt_bonus,
                -index,
            )
            if best_score is None or candidate_score > best_score:
                best_index = index
                best_score = candidate_score

        if best_index is None:
            break
        add_index(best_index)

    if len(selected_indices) < target_clips:
        raise RuntimeError(
            f"Could only select {len(selected_indices)} clips with max_clips_per_piece={max_clips_per_piece}; target was {target_clips}"
        )

    selected: Dict[str, Dict] = {}
    for rank, index in enumerate(selected_indices[:target_clips], start=1):
        clip_id = f"clip_{rank:03d}"
        clip = dict(ordered_candidates[index])
        clip["clip_id"] = clip_id
        clip["selection_rank"] = rank
        selected[clip_id] = clip
    return selected


def normalize_manifest_clip(raw_clip: Dict, category: str) -> Dict:
    clip = dict(raw_clip)
    audio_path = Path(str(clip.get("audio_path") or "")).expanduser()
    midi_path = Path(str(clip.get("midi_path") or "")).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Manifest clip '{category}' audio path does not exist: {audio_path}"
        )
    if not midi_path.exists():
        raise FileNotFoundError(
            f"Manifest clip '{category}' MIDI path does not exist: {midi_path}"
        )

    try:
        start_sec = float(clip.get("start_sec", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest clip '{category}' has an invalid start_sec") from exc

    end_sec = clip.get("end_sec")
    duration_sec = clip.get("duration_sec")
    if duration_sec is None and end_sec is None:
        raise ValueError(
            f"Manifest clip '{category}' must define duration_sec or end_sec"
        )

    if duration_sec is None:
        duration_sec = float(end_sec) - start_sec
    else:
        duration_sec = float(duration_sec)

    if duration_sec <= 0.0:
        raise ValueError(
            f"Manifest clip '{category}' must have a positive duration_sec"
        )

    if end_sec is None:
        end_sec = start_sec + duration_sec
    else:
        end_sec = float(end_sec)

    reference_musicxml_path = ""
    for field in REFERENCE_MUSICXML_FIELDS:
        if clip.get(field):
            candidate = Path(str(clip.get(field) or "")).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Manifest clip '{category}' reference MusicXML path does not exist: {candidate}"
                )
            reference_musicxml_path = str(candidate)
            break

    return {
        **clip,
        "title": str(clip.get("title") or audio_path.stem),
        "audio_path": str(audio_path),
        "midi_path": str(midi_path),
        "reference_musicxml_path": reference_musicxml_path,
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "duration_sec": round(duration_sec, 6),
    }


def load_benchmark_manifest(manifest_path: str, clip_ids: Sequence[str] | None = None) -> Dict[str, Dict]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    manifest_clips = payload.get("clips")
    if not isinstance(manifest_clips, dict) or not manifest_clips:
        raise RuntimeError(
            f"Benchmark manifest {path} has no usable 'clips' mapping"
        )

    selected: Dict[str, Dict] = {}
    requested_ids = list(clip_ids) if clip_ids else list(manifest_clips.keys())
    missing_clip_ids: List[str] = []

    for clip_id in requested_ids:
        entry = manifest_clips.get(clip_id)
        if entry is None:
            missing_clip_ids.append(str(clip_id))
            continue

        raw_clip = (
            entry.get("clip")
            if isinstance(entry, dict) and isinstance(entry.get("clip"), dict)
            else entry
        )
        if not isinstance(raw_clip, dict):
            raise ValueError(f"Manifest clip '{clip_id}' must be an object")
        normalized = normalize_manifest_clip(raw_clip, str(clip_id))
        normalized["clip_id"] = str(raw_clip.get("clip_id") or clip_id)
        selected[str(clip_id)] = normalized

    if missing_clip_ids:
        raise RuntimeError(
            f"Benchmark manifest {path} is missing clip IDs: {', '.join(missing_clip_ids)}"
        )

    return selected


def write_benchmark_manifest(
    manifest_path: str,
    selected: Dict[str, Dict],
    args: argparse.Namespace,
) -> Path:
    path = Path(manifest_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "suite_name": path.stem,
        "description": "Frozen replay benchmark clip selection for live chunk experiments.",
        "generated_from": {
            "split": args.split,
            "pieces": args.pieces,
            "scan_seconds": args.scan_seconds,
            "clip_seconds": args.clip_seconds,
            "step_seconds": args.step_seconds,
            "min_notes": args.min_notes,
            "chunk_seconds": args.chunk_seconds,
            "selection_boundary_band_sec": args.eval_boundary_band_sec,
            "selection_strategy": args.selection_strategy,
            "target_clips": args.target_clips,
            "max_clips_per_piece": args.max_clips_per_piece,
            "max_clips_per_title": args.max_clips_per_title,
            "categories": list(args.categories),
        },
        "clips": {category: dict(clip) for category, clip in selected.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def select_benchmark_clips(args: argparse.Namespace) -> tuple[Dict[str, Dict], Path | None, str]:
    manifest_path = Path(args.benchmark_manifest).expanduser().resolve() if args.benchmark_manifest else None
    if manifest_path is not None:
        selected = load_benchmark_manifest(str(manifest_path), args.clip_ids)
        return selected, manifest_path, "manifest"

    pieces = load_index(args.split)
    candidates = build_candidates(
        pieces,
        num_pieces=args.pieces,
        scan_seconds=args.scan_seconds,
        clip_seconds=args.clip_seconds,
        step_seconds=args.step_seconds,
        min_notes=args.min_notes,
        chunk_seconds=args.chunk_seconds,
        boundary_band_sec=args.eval_boundary_band_sec,
    )

    if args.selection_strategy == DIVERSE_SUITE_SELECTION_STRATEGY:
        selected = choose_diverse_suite(
            candidates,
            target_clips=args.target_clips,
            max_clips_per_piece=args.max_clips_per_piece,
            max_clips_per_title=args.max_clips_per_title,
        )
        return selected, None, DIVERSE_SUITE_SELECTION_STRATEGY

    selected = choose_excerpts(candidates, args.categories)
    return selected, None, DEFAULT_SELECTION_STRATEGY


def resolve_reference_musicxml_path(clip_id: str, clip: Dict, reference_dir: str = "") -> str:
    for field in REFERENCE_MUSICXML_FIELDS:
        value = clip.get(field)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            return str(path.resolve())

    candidate_stems = [
        str(clip.get("clip_id") or clip_id),
        Path(str(clip.get("midi_path") or "")).stem,
        Path(str(clip.get("audio_path") or "")).stem,
    ]
    candidate_stems = [stem for index, stem in enumerate(candidate_stems) if stem and stem not in candidate_stems[:index]]

    search_dirs: List[Path] = []
    if reference_dir:
        search_dirs.append(Path(reference_dir).expanduser())
    midi_parent = Path(str(clip.get("midi_path") or "")).expanduser().parent
    audio_parent = Path(str(clip.get("audio_path") or "")).expanduser().parent
    for directory in (midi_parent, audio_parent):
        if directory and directory not in search_dirs:
            search_dirs.append(directory)

    for directory in search_dirs:
        if not directory.exists():
            continue
        for stem in candidate_stems:
            for ext in REFERENCE_MUSICXML_EXTENSIONS:
                candidate = directory / f"{stem}{ext}"
                if candidate.exists():
                    return str(candidate.resolve())

    return ""


def find_missing_reference_musicxml(
    selected: Dict[str, Dict],
    reference_dir: str = "",
) -> List[str]:
    missing = []
    for clip_id, clip in selected.items():
        if not resolve_reference_musicxml_path(clip_id, clip, reference_dir):
            missing.append(str(clip_id))
    return missing


def print_selection_summary(selected: Dict[str, Dict]) -> None:
    clips = list(selected.items())
    unique_pieces = {
        str(clip.get("piece_id") or clip.get("audio_path") or clip_id)
        for clip_id, clip in clips
    }
    print(
        f"Selected {len(clips)} clips across {len(unique_pieces)} unique pieces"
    )

    preview_count = 12
    for clip_id, clip in clips[:preview_count]:
        features = clip.get("selection_features") or {}
        print(
            f"  {clip_id}: {clip['title']} start={clip['start_sec']:.2f}s duration={clip['duration_sec']:.2f}s "
            f"gt_notes={clip['gt_note_count']} density={features.get('note_density', 0.0):.2f}/s "
            f"poly={features.get('mean_notes_per_onset', 0.0):.2f} ioi_cv={features.get('ioi_cv', 0.0):.2f} "
            f"boundary_rate={features.get('boundary_event_rate', 0.0):.2f} rms={clip.get('rms', 0.0):.4f}"
        )

    remaining = len(clips) - preview_count
    if remaining > 0:
        print(f"  ... {remaining} more clips omitted from preview")


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


def normalize_cluster_metric_notes(notes: Iterable[Dict]) -> List[Dict]:
    normalized = normalize_predicted_notes(notes)
    for note in normalized:
        metric_onset = note.get("cluster_metric_time_seconds")
        if metric_onset is None:
            continue
        onset = float(metric_onset or 0.0)
        duration = float(note.get("duration", 0.0) or 0.0)
        note["onset_time"] = onset
        note["offset_time"] = onset + duration
    normalized.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return normalized


def normalize_cluster_metric_slot_consensus_notes(
    notes: Iterable[Dict],
    min_quantization_confidence: float = 0.85,
) -> List[Dict]:
    normalized = normalize_predicted_notes(notes)
    slot_groups: Dict[tuple[int, int], List[Dict]] = {}
    for note in normalized:
        start_grid_idx = note.get("start_grid_idx")
        grid_subdivision = note.get("grid_subdivision")
        if start_grid_idx is None or grid_subdivision is None:
            continue
        try:
            key = (int(start_grid_idx), int(grid_subdivision))
        except (TypeError, ValueError):
            continue
        slot_groups.setdefault(key, []).append(note)

    for group in slot_groups.values():
        if len(group) < 2:
            continue
        quantization_confidences = [float(note.get("quantization_confidence", 0.0) or 0.0) for note in group]
        if min(quantization_confidences, default=0.0) < float(min_quantization_confidence):
            continue
        onset_values = sorted(float(note.get("onset_time", 0.0) or 0.0) for note in group)
        if not onset_values:
            continue
        median_index = len(onset_values) // 2
        if len(onset_values) % 2 == 1:
            consensus_onset = onset_values[median_index]
        else:
            consensus_onset = (onset_values[median_index - 1] + onset_values[median_index]) / 2.0
        for note in group:
            duration = float(note.get("duration", 0.0) or 0.0)
            note["onset_time"] = consensus_onset
            note["offset_time"] = consensus_onset + duration

    normalized.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    return normalized


def _expand_chords_to_note_events(chords: Iterable[Dict]) -> List[Dict]:
    expanded: List[Dict] = []

    for chord in chords or []:
        chord_dict = dict(chord)
        midi_notes = chord_dict.get("midi_notes") or []
        note_names = list(chord_dict.get("note_names") or [])

        for note_index, midi_note in enumerate(midi_notes):
            try:
                midi_value = int(midi_note)
            except (TypeError, ValueError):
                continue

            note_event = dict(chord_dict)
            note_event["midi_note"] = midi_value
            if note_index < len(note_names):
                note_event["note_name"] = note_names[note_index]
            note_event["source_event_type"] = "chord_member"
            expanded.append(note_event)

    return expanded


def normalize_note_level_predictions(notes: Iterable[Dict], chords: Iterable[Dict] | None = None) -> List[Dict]:
    combined_events: List[Dict] = [dict(note) for note in (notes or [])]
    combined_events.extend(_expand_chords_to_note_events(chords or []))
    return normalize_predicted_notes(_dedupe_note_events(combined_events))


def _prefix_metric_keys(metrics: Dict, prefix: str) -> Dict:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _cluster_anchor_time(cluster: Sequence[Dict]) -> float:
    if not cluster:
        return 0.0
    return float(
        np.mean(
            [float(note.get("onset_time", 0.0) or 0.0) for note in cluster],
            dtype=np.float64,
        )
    )


def _cluster_pitch_signature(cluster: Sequence[Dict]) -> tuple[int, ...]:
    pitches: List[int] = []
    for note in cluster:
        try:
            pitches.append(int(note.get("midi_note", note.get("pitch", 0)) or 0))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(pitches))


def compute_onset_cluster_metrics(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    onset_tolerance_sec: float = ONSET_CLUSTER_TOLERANCE_SEC,
) -> Dict:
    pred_clusters = cluster_note_onsets(pred_notes, onset_tolerance_sec=onset_tolerance_sec)
    gt_clusters = cluster_note_onsets(gt_notes, onset_tolerance_sec=onset_tolerance_sec)

    exact_matches = 0
    onset_aligned_matches = 0
    overclustered_matches = 0
    underclustered_matches = 0
    pitch_conflict_matches = 0
    jaccards: List[float] = []
    gt_matched: set[int] = set()

    for pred_cluster in pred_clusters:
        pred_anchor = _cluster_anchor_time(pred_cluster)
        best_idx = None
        best_onset_error = None

        for idx, gt_cluster in enumerate(gt_clusters):
            if idx in gt_matched:
                continue
            onset_error = abs(pred_anchor - _cluster_anchor_time(gt_cluster))
            if onset_error > onset_tolerance_sec:
                continue
            if best_onset_error is None or onset_error < best_onset_error:
                best_idx = idx
                best_onset_error = onset_error

        if best_idx is None:
            continue

        gt_matched.add(best_idx)
        onset_aligned_matches += 1

        pred_signature = _cluster_pitch_signature(pred_cluster)
        gt_signature = _cluster_pitch_signature(gt_clusters[best_idx])
        pred_set = set(pred_signature)
        gt_set = set(gt_signature)
        union = pred_set | gt_set
        if union:
            jaccards.append(len(pred_set & gt_set) / len(union))

        if pred_signature == gt_signature:
            exact_matches += 1
            continue

        if len(pred_signature) > len(gt_signature):
            overclustered_matches += 1
        elif len(pred_signature) < len(gt_signature):
            underclustered_matches += 1
        else:
            pitch_conflict_matches += 1

    precision = exact_matches / len(pred_clusters) if pred_clusters else 0.0
    recall = exact_matches / len(gt_clusters) if gt_clusters else 0.0
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_matches": exact_matches,
        "predicted": len(pred_clusters),
        "ground_truth": len(gt_clusters),
        "onset_aligned_matches": onset_aligned_matches,
        "onset_alignment_precision": (onset_aligned_matches / len(pred_clusters)) if pred_clusters else 0.0,
        "onset_alignment_recall": (onset_aligned_matches / len(gt_clusters)) if gt_clusters else 0.0,
        "avg_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "overclustered_matches": overclustered_matches,
        "underclustered_matches": underclustered_matches,
        "pitch_conflict_matches": pitch_conflict_matches,
        "unmatched_predicted": max(0, len(pred_clusters) - onset_aligned_matches),
        "unmatched_ground_truth": max(0, len(gt_clusters) - onset_aligned_matches),
    }


def compute_pairwise_coonset_metrics(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    onset_tol: float = ONSET_CLUSTER_TOLERANCE_SEC,
    window_sec: float = ONSET_CLUSTER_TOLERANCE_SEC,
) -> Dict:
    """Tempo-free chord-grouping metric: agreement of the 'struck together'
    relation over commonly-matched notes only. For every unordered pair of
    pitch+onset-matched notes, both pred and GT vote together/apart by
    |Δonset| <= window_sec; F1 is computed over the 'together' relation.

    Unlike the single-linkage cluster F1, this has no anchor and no transitive
    chaining, so it does not flip on where a wide (25-79ms) chord's boundary
    falls between pred and GT. It deliberately factors out recall (scores only
    matched notes), isolating grouping quality from missed notes. Proven
    2026-06-15 to show production grouping is ~0.97 vs the misleading 0.688
    single-linkage headline. See gpt_memory/repo/live-change-log.md.
    """
    gt_used: set[int] = set()
    pairs: List[tuple[int, int]] = []
    for pi, pred in enumerate(pred_notes):
        try:
            p_pitch = int(pred.get("midi_note", pred.get("pitch", 0)) or 0)
            p_onset = float(pred.get("onset_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        best_idx = None
        best_err = None
        for gi, gt in enumerate(gt_notes):
            if gi in gt_used:
                continue
            try:
                g_pitch = int(gt.get("midi_note", gt.get("pitch", 0)) or 0)
                g_onset = float(gt.get("onset_time", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if p_pitch != g_pitch:
                continue
            err = abs(p_onset - g_onset)
            if err > onset_tol:
                continue
            if best_err is None or err < best_err:
                best_idx = gi
                best_err = err
        if best_idx is not None:
            gt_used.add(best_idx)
            pairs.append((pi, best_idx))

    matched_notes = len(pairs)
    if matched_notes < 2:
        # No pair to disagree on -> grouping is vacuously correct.
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "matched_notes": matched_notes,
            "true_positive_pairs": 0,
            "false_positive_pairs": 0,
            "false_negative_pairs": 0,
        }

    tp = fp = fn = 0
    for a in range(matched_notes):
        pa, ga = pairs[a]
        pa_on = float(pred_notes[pa].get("onset_time", 0.0) or 0.0)
        ga_on = float(gt_notes[ga].get("onset_time", 0.0) or 0.0)
        for b in range(a + 1, matched_notes):
            pb, gb = pairs[b]
            pred_together = abs(pa_on - float(pred_notes[pb].get("onset_time", 0.0) or 0.0)) <= window_sec
            gt_together = abs(ga_on - float(gt_notes[gb].get("onset_time", 0.0) or 0.0)) <= window_sec
            if gt_together and pred_together:
                tp += 1
            elif pred_together and not gt_together:
                fp += 1
            elif gt_together and not pred_together:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_notes": matched_notes,
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
    }


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


def compute_onset_tolerance_sweep(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    onset_tolerances_ms: Sequence[int] = DEFAULT_STRICT_ONSET_TOLS_MS,
) -> Dict[str, Dict]:
    """Evaluate onset-only note metrics at multiple strict timing tolerances."""
    sweep: Dict[str, Dict] = {}
    for tol_ms in sorted({int(max(1, round(float(value)))) for value in onset_tolerances_ms}):
        label = f"{tol_ms}ms"
        metrics = compute_note_metrics(pred_notes, gt_notes, onset_tol=tol_ms / 1000.0)
        sweep[label] = {
            "onset_tol_ms": tol_ms,
            **metrics,
        }
    return sweep


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


def empty_reference_musicxml_score_metrics() -> Dict:
    return {
        "score_reference_available": False,
        "score_reference_pending": False,
        "score_reference_path": None,
        "score_edit_accuracy": None,
        "score_edit_cost": None,
        "score_reference_cost": None,
        "score_exact_token_f1": None,
        "score_exact_token_precision": None,
        "score_exact_token_recall": None,
        "score_exact_token_matches": 0,
        "score_predicted_tokens": 0,
        "score_reference_tokens": 0,
        "score_edit_exact_ops": 0,
        "score_edit_substitutions": 0,
        "score_edit_deletions": 0,
        "score_edit_insertions": 0,
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


def _compact_note_debug(note: Dict | None) -> Dict | None:
    if note is None:
        return None

    compact = {
        "midi_note": int(note["midi_note"]),
        "onset_time": round(float(note["onset_time"]), 6),
    }
    if "offset_time" in note:
        compact["offset_time"] = round(float(note["offset_time"]), 6)
    if "duration" in note:
        compact["duration"] = round(float(note["duration"]), 6)
    if note.get("confidence") is not None:
        compact["confidence"] = round(float(note.get("confidence", 0.0) or 0.0), 6)
    for key in ("selection_reason", "note_value"):
        if note.get(key) is not None:
            compact[key] = note[key]
    if note.get("source_boundary_sec") is not None:
        compact["source_boundary_sec"] = round(float(note["source_boundary_sec"]), 6)
    return compact


def _summarize_nearby_note_candidates(
    notes: Sequence[Dict],
    gt_note: Dict,
    onset_window_sec: float,
) -> Dict:
    gt_pitch = int(gt_note["midi_note"])
    gt_onset = float(gt_note["onset_time"])
    nearby: List[tuple[float, Dict]] = []

    for note in notes:
        if int(note.get("midi_note", -1)) != gt_pitch:
            continue
        onset = float(note.get("onset_time", 0.0) or 0.0)
        onset_delta_sec = onset - gt_onset
        if abs(onset_delta_sec) > onset_window_sec:
            continue
        nearby.append((abs(onset_delta_sec), dict(note)))

    if not nearby:
        return {
            "present": False,
            "count": 0,
            "best_delta_ms": None,
            "best_note": None,
        }

    nearby.sort(key=lambda item: (item[0], float(item[1].get("onset_time", 0.0) or 0.0)))
    best_note = nearby[0][1]
    best_delta_ms = round((float(best_note.get("onset_time", 0.0) or 0.0) - gt_onset) * 1000.0, 1)
    return {
        "present": True,
        "count": len(nearby),
        "best_delta_ms": best_delta_ms,
        "best_note": _compact_note_debug(best_note),
    }


def _summarize_emitted_note_candidates(
    run: Dict,
    gt_note: Dict,
    onset_window_sec: float,
) -> Dict:
    gt_pitch = int(gt_note["midi_note"])
    gt_onset = float(gt_note["onset_time"])
    nearby: List[tuple[float, Dict, Dict]] = []

    for chunk in run.get("chunks") or []:
        for note in chunk.get("emitted_notes") or []:
            if int(note.get("midi_note", -1)) != gt_pitch:
                continue
            onset = float(note.get("onset_time", 0.0) or 0.0)
            onset_delta_sec = onset - gt_onset
            if abs(onset_delta_sec) > onset_window_sec:
                continue
            nearby.append((abs(onset_delta_sec), dict(note), chunk))

    if not nearby:
        return {
            "present": False,
            "count": 0,
            "best_delta_ms": None,
            "best_chunk_index": None,
            "best_chunk_end_sec": None,
            "best_note": None,
        }

    nearby.sort(key=lambda item: (item[0], float(item[1].get("onset_time", 0.0) or 0.0)))
    best_note = nearby[0][1]
    best_chunk = nearby[0][2]
    best_delta_ms = round((float(best_note.get("onset_time", 0.0) or 0.0) - gt_onset) * 1000.0, 1)
    return {
        "present": True,
        "count": len(nearby),
        "best_delta_ms": best_delta_ms,
        "best_chunk_index": best_chunk.get("chunk_index"),
        "best_chunk_end_sec": round(float(best_chunk.get("chunk_end_sec", 0.0) or 0.0), 6),
        "best_note": _compact_note_debug(best_note),
    }


def _find_retro_boundary_log(boundary_logs: Sequence[Dict], boundary_sec: float) -> Dict | None:
    rounded_boundary_sec = round(float(boundary_sec), 6)
    for boundary_log in boundary_logs:
        if round(float(boundary_log.get("boundary_sec", -1.0) or -1.0), 6) == rounded_boundary_sec:
            return boundary_log
    return None


def _summarize_retro_boundary_candidates(
    boundary_log: Dict | None,
    gt_note: Dict,
    onset_window_sec: float,
) -> Dict:
    if boundary_log is None:
        return {
            "available": False,
            "scanned": None,
            "reason": "unavailable",
            "candidate_count": 0,
            "candidate_decisions": [],
            "best_candidate_delta_ms": None,
            "best_candidate": None,
        }

    gt_pitch = int(gt_note["midi_note"])
    gt_onset = float(gt_note["onset_time"])
    matching_candidates: List[tuple[float, Dict]] = []
    for candidate_log in boundary_log.get("candidate_logs") or []:
        if int(candidate_log.get("midi_note", -1)) != gt_pitch:
            continue
        onset_delta_sec = float(candidate_log.get("onset_time", 0.0) or 0.0) - gt_onset
        if abs(onset_delta_sec) > onset_window_sec:
            continue
        matching_candidates.append((abs(onset_delta_sec), candidate_log))

    matching_candidates.sort(key=lambda item: (item[0], float(item[1].get("onset_time", 0.0) or 0.0)))
    best_candidate = matching_candidates[0][1] if matching_candidates else None
    best_delta_ms = None
    if best_candidate is not None:
        best_delta_ms = round((float(best_candidate.get("onset_time", 0.0) or 0.0) - gt_onset) * 1000.0, 1)

    candidate_decisions = sorted(
        {
            str(candidate_log.get("decision"))
            for _, candidate_log in matching_candidates
            if candidate_log.get("decision")
        }
    )
    return {
        "available": True,
        "scanned": bool(boundary_log.get("scan")),
        "reason": str(boundary_log.get("reason") or "unknown"),
        "candidate_count": len(matching_candidates),
        "candidate_decisions": candidate_decisions,
        "best_candidate_delta_ms": best_delta_ms,
        "best_candidate": _compact_note_debug(best_candidate),
    }


def build_clip_boundary_failure_diagnostics(
    control_run: Dict,
    treatment_run: Dict,
    gt_notes: Sequence[Dict],
    chunk_seconds: float,
    boundary_band_sec: float,
    retro_run: Dict | None = None,
    onset_tol: float = 0.05,
) -> Dict:
    boundary_gt = [
        note
        for note in gt_notes
        if is_boundary_note(float(note.get("onset_time", 0.0) or 0.0), chunk_seconds, boundary_band_sec)
    ]
    diagnostic_window_sec = max(float(boundary_band_sec), float(onset_tol))
    retro_boundary_logs = (retro_run or {}).get("boundary_logs") or []
    missed_diagnostics: List[Dict] = []
    diagnosis_tag_counts: Counter = Counter()
    control_missed = 0
    treatment_missed = 0
    retro_missed = 0

    for gt_note in boundary_gt:
        gt_onset = float(gt_note["onset_time"])
        boundary_sec = round(round(gt_onset / chunk_seconds) * chunk_seconds, 6) if chunk_seconds > 0.0 else 0.0

        control_match = _summarize_nearby_note_candidates(control_run.get("notes") or [], gt_note, onset_tol)
        control_nearby_final = _summarize_nearby_note_candidates(
            control_run.get("notes") or [],
            gt_note,
            diagnostic_window_sec,
        )
        control_coarse = _summarize_emitted_note_candidates(control_run, gt_note, diagnostic_window_sec)

        treatment_match = _summarize_nearby_note_candidates(treatment_run.get("notes") or [], gt_note, onset_tol)
        treatment_nearby_final = _summarize_nearby_note_candidates(
            treatment_run.get("notes") or [],
            gt_note,
            diagnostic_window_sec,
        )
        treatment_coarse = _summarize_emitted_note_candidates(treatment_run, gt_note, diagnostic_window_sec)

        retro_match = None
        retro_nearby_final = None
        retro_boundary = None
        if retro_run is not None:
            retro_match = _summarize_nearby_note_candidates(retro_run.get("notes") or [], gt_note, onset_tol)
            retro_nearby_final = _summarize_nearby_note_candidates(
                retro_run.get("notes") or [],
                gt_note,
                diagnostic_window_sec,
            )
            retro_boundary = _summarize_retro_boundary_candidates(
                _find_retro_boundary_log(retro_boundary_logs, boundary_sec),
                gt_note,
                diagnostic_window_sec,
            )

        tags: List[str] = []
        if not control_match["present"]:
            control_missed += 1
            tags.append("control_boundary_miss")
            if control_nearby_final["present"]:
                tags.append("control_timing_or_quantization_drift")
            elif control_coarse["present"]:
                tags.append("control_emitted_candidate_dropped")
            else:
                tags.append("no_control_coarse_candidate")

        if not treatment_match["present"]:
            treatment_missed += 1
            if treatment_coarse["present"]:
                tags.append("treatment_candidate_not_retained")
        elif not control_match["present"]:
            tags.append("treatment_recovers_boundary_note")

        if retro_run is not None:
            if not retro_match["present"]:
                retro_missed += 1
                tags.append("retro_boundary_miss")
            elif not control_match["present"]:
                tags.append("retro_recovers_boundary_note")

            if retro_boundary["available"]:
                if retro_boundary["scanned"] is False:
                    tags.append(f"retro_skipped_{retro_boundary['reason']}")
                elif retro_boundary["reason"] == "seam_error":
                    tags.append("retro_seam_error")
                elif retro_boundary["candidate_count"] == 0:
                    tags.append("retro_no_matching_candidate")
                else:
                    for decision in retro_boundary["candidate_decisions"]:
                        tags.append(f"retro_{decision}")

        missed_in_any_arm = (
            not control_match["present"]
            or not treatment_match["present"]
            or (retro_run is not None and not retro_match["present"])
        )
        if not missed_in_any_arm:
            continue

        diagnosis_tags = sorted(set(tags))
        diagnosis_tag_counts.update(diagnosis_tags)
        missed_diagnostics.append(
            {
                "midi_note": int(gt_note["midi_note"]),
                "gt_onset_time": round(gt_onset, 6),
                "boundary_sec": boundary_sec,
                "distance_to_boundary_ms": round((gt_onset - boundary_sec) * 1000.0, 1),
                "control": {
                    "matched": bool(control_match["present"]),
                    "matched_note": control_match["best_note"],
                    "nearby_final_candidate": control_nearby_final,
                    "coarse_candidate": control_coarse,
                },
                "treatment": {
                    "matched": bool(treatment_match["present"]),
                    "matched_note": treatment_match["best_note"],
                    "nearby_final_candidate": treatment_nearby_final,
                    "coarse_candidate": treatment_coarse,
                },
                "retro": None
                if retro_run is None
                else {
                    "matched": bool(retro_match["present"]),
                    "matched_note": retro_match["best_note"],
                    "nearby_final_candidate": retro_nearby_final,
                    "boundary_scan": retro_boundary,
                },
                "diagnosis_tags": diagnosis_tags,
            }
        )

    missed_by_arm = {
        "control": control_missed,
        "treatment": treatment_missed,
    }
    if retro_run is not None:
        missed_by_arm["retro"] = retro_missed

    return {
        "boundary_gt_notes": len(boundary_gt),
        "missed_by_arm": missed_by_arm,
        "missed_note_count": len(missed_diagnostics),
        "missed_notes": missed_diagnostics,
        "diagnosis_tag_counts": dict(sorted(diagnosis_tag_counts.items())),
    }


def classify_clip_failure_buckets(control: Dict, treatment: Dict, retro: Dict | None = None) -> List[str]:
    buckets = []
    if (
        treatment["p95_chunk_total_ms"] <= (control["p95_chunk_total_ms"] - 2.0)
        and abs(treatment["f1"] - control["f1"]) <= 0.01
        and abs(treatment["boundary_miss_rate"] - control["boundary_miss_rate"]) <= 0.01
        and abs(
            treatment.get("algorithmic_time_to_visible_median_ms", 0.0)
            - control.get("algorithmic_time_to_visible_median_ms", 0.0)
        )
        <= 25.0
    ):
        buckets.append("runtime_only_win")

    if control["boundary_missed_notes"] > 0:
        buckets.append("boundary_miss_failure")

    if (
        control.get("stabilization_latency_p95_ms", 0.0) >= 1500.0
        or control.get("avg_revision_count", 0.0) >= 1.5
    ):
        buckets.append("high_revision_slow_stabilization")

    if control["offset_f1"] < 0.5 or control["note_value_accuracy"] < 0.5:
        buckets.append("note_value_offset_failure")

    if retro is not None and (
        retro["f1"] < (control["f1"] - 0.02)
        or retro["duplicates_per_100_notes"] > (control["duplicates_per_100_notes"] + 1.0)
    ):
        buckets.append("retro_regression")

    bucket_set = set(buckets)
    return [bucket for bucket in FAILURE_BUCKET_ORDER if bucket in bucket_set]


def safe_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def notes_match(note_a: Dict, note_b: Dict, onset_tol: float = 0.05) -> bool:
    return (
        int(note_a["midi_note"]) == int(note_b["midi_note"])
        and abs(float(note_a["onset_time"]) - float(note_b["onset_time"])) <= onset_tol
    )


def _process_live_session_chunk(
    session,
    notes: Sequence[Dict],
    chords: Sequence[Dict],
    current_time: float,
) -> Dict:
    current_time = float(current_time)
    session.last_update = current_time

    coarse_notes = [dict(note) for note in notes]
    coarse_chords = [dict(chord) for chord in chords]

    timing_events = list(coarse_notes)
    timing_events.extend(coarse_chords)
    for onset in _cluster_live_onset_times(timing_events):
        session.tempo_tracker.add_onset(onset)

    bpm = session.tempo_tracker.current_bpm
    grid = session.tempo_tracker.beat_grid

    quantize_batch_coarse(coarse_notes, bpm, grid=grid)
    session.coarse_notes.extend(coarse_notes)

    if coarse_chords:
        quantize_batch_coarse(coarse_chords, bpm, grid=grid)
        session.coarse_chords.extend(coarse_chords)

    session.refinement_state.add_notes(coarse_notes, current_time)
    refined = session.refinement_state.check_refinement(current_time, bpm, grid=grid)

    needs_refresh = False
    current_version = session.refinement_state.get_refinement_version()
    if current_version > session._last_notified_version:
        needs_refresh = True
        session._last_notified_version = current_version

    return {
        "coarse_notes": coarse_notes,
        "coarse_chords": coarse_chords,
        "bpm": bpm,
        "bpm_confidence": session.tempo_tracker.confidence,
        "beat_grid": session.grid_payload(),
        "needs_refresh": needs_refresh,
        "refined_notes": refined or [],
        "refinement_version": current_version,
        "next_refinement_poll_ms": session.get_next_refinement_delay_ms(current_time),
    }


def _poll_live_session_refinement(session, current_time: float) -> Dict:
    current_time = float(current_time)
    session.last_update = current_time
    bpm, confidence = session.get_current_bpm()
    grid = session.beat_grid
    refined = session.refinement_state.check_refinement(current_time, bpm, grid=grid)
    refinement_version = session.refinement_state.get_refinement_version()
    needs_refresh = refined is not None and len(refined) > 0
    if needs_refresh:
        session._last_notified_version = refinement_version
    return {
        "needs_refresh": needs_refresh,
        "refined_notes": refined or [],
        "refinement_version": refinement_version,
        "bpm": bpm,
        "bpm_confidence": confidence,
        "beat_grid": session.grid_payload(),
        "next_refinement_poll_ms": session.get_next_refinement_delay_ms(current_time),
    }


def _capture_live_display_snapshot(
    session,
    event_type: str,
    time_sec: float,
    chunk_index: int | None,
    audio_time_sec: float | None = None,
) -> Dict:
    get_display_state = getattr(session, "get_display_state", None)
    if callable(get_display_state):
        display_state = get_display_state() or {}
        display_notes = normalize_predicted_notes(display_state.get("notes") or [])
        display_note_events = normalize_predicted_notes(display_state.get("note_events") or [])
    else:
        display_notes = normalize_predicted_notes(session.get_all_notes())
        display_note_events = normalize_note_level_predictions(
            session.get_all_notes(),
            getattr(session, "coarse_chords", []) or [],
        )
    return {
        "event_type": str(event_type),
        "time_sec": round(float(time_sec), 6),
        "audio_time_sec": round(float(audio_time_sec if audio_time_sec is not None else time_sec), 6),
        "chunk_index": chunk_index,
        "refinement_version": int(session.refinement_state.get_refinement_version()),
        "visible_note_count": len(display_notes),
        "display_note_event_count": len(display_note_events),
        "notes": display_notes,
        "display_note_events": display_note_events,
    }


def _drain_live_refinements(
    session,
    current_time: float,
    deadline_time: float | None,
    audio_time_sec: float,
    snapshots: List[Dict],
    chunk_index: int | None,
) -> float:
    effective_time = float(current_time)
    for _ in range(2048):
        next_delay_ms = session.get_next_refinement_delay_ms(effective_time)
        if next_delay_ms is None:
            break

        next_time = effective_time + max(0.0, float(next_delay_ms) / 1000.0)
        if next_time <= effective_time + 1e-9:
            next_time = effective_time + 1e-6
        if deadline_time is not None and next_time + 1e-9 >= float(deadline_time):
            break

        poll_result = _poll_live_session_refinement(session, next_time)
        effective_time = next_time
        if poll_result.get("needs_refresh"):
            snapshots.append(
                _capture_live_display_snapshot(
                    session,
                    event_type="refinement_poll",
                    time_sec=effective_time,
                    chunk_index=chunk_index,
                    audio_time_sec=audio_time_sec,
                )
            )
    return effective_time


def _match_notes_greedy(
    pred_notes: Sequence[Dict],
    gt_notes: Sequence[Dict],
    onset_tol: float = 0.05,
) -> List[tuple[Dict, Dict]]:
    matches: List[tuple[Dict, Dict]] = []
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
        matches.append((pred, gt_notes[best_idx]))

    return matches


def _find_matching_note(target_note: Dict, notes: Sequence[Dict], onset_tol: float = 0.05) -> Dict | None:
    best_note = None
    best_onset_error = None
    for note in notes:
        if int(note["midi_note"]) != int(target_note["midi_note"]):
            continue
        onset_error = abs(float(note["onset_time"]) - float(target_note["onset_time"]))
        if onset_error > onset_tol:
            continue
        if best_onset_error is None or onset_error < best_onset_error:
            best_note = note
            best_onset_error = onset_error
    return best_note


def _notation_signature(note: Dict) -> tuple:
    note_divisions = note.get("note_divisions")
    return (
        str(note.get("note_value") or ""),
        round(float(note_divisions), 4) if note_divisions is not None else None,
        bool(note.get("dotted", False)),
        bool(note.get("is_triplet", note.get("triplet", False))),
        int(note.get("start_grid_idx")) if note.get("start_grid_idx") is not None else None,
        int(note.get("grid_subdivision")) if note.get("grid_subdivision") is not None else None,
    )


def compute_notation_metrics(
    run: Dict,
    gt_notes: Sequence[Dict],
    onset_tol: float = 0.05,
) -> Dict:
    snapshots = list(run.get("display_snapshots") or [])
    final_display_notes = list(run.get("final_display_notes") or [])
    refinement_snapshot_count = sum(1 for snapshot in snapshots if snapshot.get("event_type") == "refinement_poll")

    empty_metrics = {
        "display_snapshot_count": len(snapshots),
        "refinement_snapshot_count": refinement_snapshot_count,
        "final_display_note_count": len(final_display_notes),
        "matched_display_notes": 0,
        "stabilized_display_notes": 0,
        "time_to_first_correct_note_ms": 0.0,
        "algorithmic_time_to_first_correct_note_ms": 0.0,
        "time_to_visible_median_ms": 0.0,
        "time_to_visible_p95_ms": 0.0,
        "algorithmic_time_to_visible_median_ms": 0.0,
        "algorithmic_time_to_visible_p95_ms": 0.0,
        "stabilization_latency_median_ms": 0.0,
        "stabilization_latency_p95_ms": 0.0,
        "algorithmic_stabilization_latency_median_ms": 0.0,
        "algorithmic_stabilization_latency_p95_ms": 0.0,
        "avg_revision_count": 0.0,
        "max_revision_count": 0.0,
    }
    if not snapshots or not final_display_notes:
        return empty_metrics

    matched_pairs = _match_notes_greedy(final_display_notes, gt_notes, onset_tol=onset_tol)
    if not matched_pairs:
        return {
            **empty_metrics,
            "matched_display_notes": 0,
        }

    time_to_visible_ms: List[float] = []
    algorithmic_time_to_visible_ms: List[float] = []
    stabilization_latency_ms: List[float] = []
    algorithmic_stabilization_latency_ms: List[float] = []
    revision_counts: List[float] = []
    time_to_first_correct_note_ms = None
    algorithmic_time_to_first_correct_note_ms = None
    stabilized_display_notes = 0

    for final_note, gt_note in matched_pairs:
        gt_onset = float(gt_note.get("onset_time", 0.0) or 0.0)
        final_signature = _notation_signature(final_note)
        state_history: List[tuple[float, float, tuple]] = []
        first_visible_latency_ms = None

        for snapshot in snapshots:
            snapshot_note = _find_matching_note(final_note, snapshot.get("notes") or [], onset_tol=onset_tol)
            if snapshot_note is None:
                continue

            snapshot_time = float(snapshot.get("time_sec", 0.0) or 0.0)
            snapshot_audio_time = float(snapshot.get("audio_time_sec", snapshot_time) or snapshot_time)
            if first_visible_latency_ms is None:
                first_visible_latency_ms = max(0.0, (snapshot_time - gt_onset) * 1000.0)
                time_to_visible_ms.append(first_visible_latency_ms)
                if time_to_first_correct_note_ms is None or first_visible_latency_ms < time_to_first_correct_note_ms:
                    time_to_first_correct_note_ms = first_visible_latency_ms
                first_visible_algorithmic_latency_ms = max(0.0, (snapshot_audio_time - gt_onset) * 1000.0)
                algorithmic_time_to_visible_ms.append(first_visible_algorithmic_latency_ms)
                if (
                    algorithmic_time_to_first_correct_note_ms is None
                    or first_visible_algorithmic_latency_ms < algorithmic_time_to_first_correct_note_ms
                ):
                    algorithmic_time_to_first_correct_note_ms = first_visible_algorithmic_latency_ms

            signature = _notation_signature(snapshot_note)
            if not state_history or signature != state_history[-1][2]:
                state_history.append((snapshot_time, snapshot_audio_time, signature))

        if not state_history:
            continue

        revision_counts.append(float(max(0, len(state_history) - 1)))

        for state_time, state_audio_time, signature in reversed(state_history):
            if signature == final_signature:
                stabilization_latency_ms.append(max(0.0, (state_time - gt_onset) * 1000.0))
                algorithmic_stabilization_latency_ms.append(max(0.0, (state_audio_time - gt_onset) * 1000.0))
                stabilized_display_notes += 1
                break

    return {
        "display_snapshot_count": len(snapshots),
        "refinement_snapshot_count": refinement_snapshot_count,
        "final_display_note_count": len(final_display_notes),
        "matched_display_notes": len(matched_pairs),
        "stabilized_display_notes": stabilized_display_notes,
        "time_to_first_correct_note_ms": float(time_to_first_correct_note_ms or 0.0),
        "algorithmic_time_to_first_correct_note_ms": float(
            algorithmic_time_to_first_correct_note_ms or 0.0
        ),
        "time_to_visible_median_ms": safe_percentile(time_to_visible_ms, 50),
        "time_to_visible_p95_ms": safe_percentile(time_to_visible_ms, 95),
        "algorithmic_time_to_visible_median_ms": safe_percentile(algorithmic_time_to_visible_ms, 50),
        "algorithmic_time_to_visible_p95_ms": safe_percentile(algorithmic_time_to_visible_ms, 95),
        "stabilization_latency_median_ms": safe_percentile(stabilization_latency_ms, 50),
        "stabilization_latency_p95_ms": safe_percentile(stabilization_latency_ms, 95),
        "algorithmic_stabilization_latency_median_ms": safe_percentile(
            algorithmic_stabilization_latency_ms, 50
        ),
        "algorithmic_stabilization_latency_p95_ms": safe_percentile(
            algorithmic_stabilization_latency_ms, 95
        ),
        "avg_revision_count": float(np.mean(revision_counts)) if revision_counts else 0.0,
        "max_revision_count": float(max(revision_counts)) if revision_counts else 0.0,
    }


def _get_final_display_note_events(run: Dict) -> List[Dict]:
    return normalize_predicted_notes(run.get("final_display_note_events") or run.get("final_display_notes") or [])


def _get_final_display_cluster_note_events(run: Dict, mode: str = "raw") -> List[Dict]:
    source_notes = run.get("final_display_note_events") or run.get("final_display_notes") or []
    if mode == "grid_snap":
        return normalize_cluster_metric_notes(source_notes)
    if mode == "slot_consensus":
        return normalize_cluster_metric_slot_consensus_notes(source_notes)
    return normalize_predicted_notes(source_notes)


def compute_final_display_accuracy_metrics(
    run: Dict,
    gt_notes: Sequence[Dict],
    reference_bpm: float,
    cluster_metric_mode: str = "raw",
) -> Dict:
    display_notes = _get_final_display_note_events(run)
    display_cluster_notes = _get_final_display_cluster_note_events(run, mode=cluster_metric_mode)
    note_metrics = compute_note_metrics(display_notes, gt_notes)
    offset_metrics = compute_offset_metrics(display_notes, gt_notes)
    note_value_metrics = compute_note_value_metrics(display_notes, gt_notes, reference_bpm=reference_bpm)
    cluster_metrics = compute_onset_cluster_metrics(display_cluster_notes, gt_notes)
    pairwise_metrics = compute_pairwise_coonset_metrics(display_cluster_notes, gt_notes)

    return {
        "display_final_note_event_count": len(display_notes),
        **_prefix_metric_keys(note_metrics, "display_note_"),
        **_prefix_metric_keys(offset_metrics, "display_"),
        **_prefix_metric_keys(note_value_metrics, "display_"),
        **_prefix_metric_keys(cluster_metrics, "display_cluster_"),
        **_prefix_metric_keys(pairwise_metrics, "display_pairwise_"),
        **_prefix_metric_keys(empty_reference_musicxml_score_metrics(), "display_"),
        "display_strict_onset_metrics": compute_onset_tolerance_sweep(display_notes, gt_notes),
    }


def _find_matching_note_index(
    target_note: Dict | None,
    notes: Sequence[Dict],
    onset_tol: float = 0.05,
) -> int | None:
    if not target_note:
        return None

    best_index = None
    best_onset_error = None
    for index, note in enumerate(notes):
        if int(note.get("midi_note", 0) or 0) != int(target_note.get("midi_note", 0) or 0):
            continue
        onset_error = abs(
            float(note.get("onset_time", note.get("time_seconds", 0.0)) or 0.0)
            - float(target_note.get("onset_time", target_note.get("time_seconds", 0.0)) or 0.0)
        )
        if onset_error > onset_tol:
            continue
        if best_onset_error is None or onset_error < best_onset_error:
            best_index = index
            best_onset_error = onset_error
    return best_index


def _merge_display_note(existing_note: Dict, updated_note: Dict) -> Dict:
    merged = dict(existing_note)
    merged.update(dict(updated_note))

    onset_time = float(
        merged.get("onset_time", merged.get("time_seconds", updated_note.get("onset_time", 0.0))) or 0.0
    )
    offset_time = float(
        merged.get("offset_time", merged.get("offset_seconds", onset_time)) or onset_time
    )
    if offset_time < onset_time:
        offset_time = onset_time
    duration = max(0.0, offset_time - onset_time)

    merged["time_seconds"] = onset_time
    merged["onset_time"] = onset_time
    merged["offset_seconds"] = offset_time
    merged["offset_time"] = offset_time
    merged["duration_seconds"] = duration
    merged["duration"] = duration
    merged["midi_note"] = int(merged.get("midi_note", merged.get("pitch", 0)) or 0)
    merged["confidence"] = float(merged.get("confidence", 0.0) or 0.0)
    return merged


def _apply_retro_update_to_display_notes(
    display_notes: Sequence[Dict],
    operation: Dict,
    onset_tol: float = 0.05,
) -> List[Dict]:
    updated_notes = [dict(note) for note in display_notes]
    old_note = operation.get("old_note")
    new_note = dict(operation.get("new_note") or {})
    if not new_note:
        return normalize_predicted_notes(updated_notes)

    target_index = _find_matching_note_index(old_note, updated_notes, onset_tol=onset_tol)
    if target_index is None:
        target_index = _find_matching_note_index(new_note, updated_notes, onset_tol=onset_tol)

    if target_index is None:
        updated_notes.append(_merge_display_note({}, new_note))
    else:
        updated_notes[target_index] = _merge_display_note(updated_notes[target_index], new_note)

    return normalize_predicted_notes(updated_notes)


def _apply_retro_update_group(
    display_notes: Sequence[Dict],
    update_group: Dict,
    onset_tol: float = 0.05,
) -> List[Dict]:
    patched_notes = normalize_predicted_notes(display_notes)
    for operation in update_group.get("operations") or []:
        patched_notes = _apply_retro_update_to_display_notes(
            patched_notes,
            operation,
            onset_tol=onset_tol,
        )
    return patched_notes


def _build_retro_display_snapshots(
    baseline_snapshots: Sequence[Dict],
    retro_update_groups: Sequence[Dict],
) -> tuple[List[Dict], List[Dict]]:
    ordered_baseline = sorted(
        (dict(snapshot) for snapshot in baseline_snapshots),
        key=lambda snapshot: (
            float(snapshot.get("time_sec", 0.0) or 0.0),
            int(snapshot.get("chunk_index", -1) if snapshot.get("chunk_index") is not None else -1),
            str(snapshot.get("event_type") or ""),
        ),
    )
    ordered_updates = sorted(
        (dict(update_group) for update_group in retro_update_groups),
        key=lambda update_group: (
            float(update_group.get("time_sec", 0.0) or 0.0),
            int(update_group.get("chunk_index", -1) if update_group.get("chunk_index") is not None else -1),
        ),
    )

    if not ordered_baseline:
        return [], []

    patched_snapshots: List[Dict] = []
    applied_updates: List[Dict] = []
    next_update_index = 0
    current_display_notes: List[Dict] = []
    last_refinement_version = 0

    for baseline_snapshot in ordered_baseline:
        snapshot_time = float(baseline_snapshot.get("time_sec", 0.0) or 0.0)

        while next_update_index < len(ordered_updates):
            update_time = float(ordered_updates[next_update_index].get("time_sec", 0.0) or 0.0)
            if update_time >= (snapshot_time - 1e-9):
                break
            update_group = ordered_updates[next_update_index]
            current_display_notes = _apply_retro_update_group(current_display_notes, update_group)
            patched_snapshots.append(
                {
                    "event_type": "retro_update",
                    "time_sec": round(update_time, 6),
                    "audio_time_sec": round(float(update_group.get("audio_time_sec", update_time) or update_time), 6),
                    "chunk_index": update_group.get("chunk_index"),
                    "refinement_version": last_refinement_version,
                    "visible_note_count": len(current_display_notes),
                    "display_note_event_count": len(current_display_notes),
                    "notes": current_display_notes,
                    "display_note_events": current_display_notes,
                }
            )
            applied_updates.append(update_group)
            next_update_index += 1

        snapshot_notes = normalize_predicted_notes(baseline_snapshot.get("notes") or [])
        for update_group in applied_updates:
            snapshot_notes = _apply_retro_update_group(snapshot_notes, update_group)

        while next_update_index < len(ordered_updates):
            update_time = float(ordered_updates[next_update_index].get("time_sec", 0.0) or 0.0)
            if abs(update_time - snapshot_time) > 1e-9:
                break
            update_group = ordered_updates[next_update_index]
            snapshot_notes = _apply_retro_update_group(snapshot_notes, update_group)
            applied_updates.append(update_group)
            next_update_index += 1

        patched_snapshot = dict(baseline_snapshot)
        patched_snapshot["notes"] = snapshot_notes
        patched_snapshot["visible_note_count"] = len(snapshot_notes)
        patched_snapshot["display_note_event_count"] = len(snapshot_notes)
        patched_snapshot["display_note_events"] = snapshot_notes
        patched_snapshots.append(patched_snapshot)
        current_display_notes = snapshot_notes
        last_refinement_version = int(
            patched_snapshot.get("refinement_version", last_refinement_version) or last_refinement_version
        )

    while next_update_index < len(ordered_updates):
        update_group = ordered_updates[next_update_index]
        update_time = float(update_group.get("time_sec", 0.0) or 0.0)
        current_display_notes = _apply_retro_update_group(current_display_notes, update_group)
        patched_snapshots.append(
            {
                "event_type": "retro_update",
                "time_sec": round(update_time, 6),
                "audio_time_sec": round(float(update_group.get("audio_time_sec", update_time) or update_time), 6),
                "chunk_index": update_group.get("chunk_index"),
                "refinement_version": last_refinement_version,
                "visible_note_count": len(current_display_notes),
                "display_note_event_count": len(current_display_notes),
                "notes": current_display_notes,
                "display_note_events": current_display_notes,
            }
        )
        next_update_index += 1

    final_display_notes = patched_snapshots[-1]["notes"] if patched_snapshots else []
    return patched_snapshots, final_display_notes


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
    capture_display_inputs: bool = False,
) -> Dict:
    chunk_frames = max(1, int(round(chunk_seconds * TARGET_SR)))
    session_id = f"experiment-{uuid.uuid4().hex}"
    chunk_summaries: List[Dict] = []
    predicted_notes: List[Dict] = []
    display_snapshots: List[Dict] = []
    captured_display_inputs: Dict[str, List[Dict]] = {}
    live_session = get_live_session(session_id)
    live_session.reset()
    total_duration_sec = float(audio.size) / TARGET_SR if TARGET_SR > 0 else 0.0
    current_display_time_sec = 0.0
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
                raw_chords = [dict(chord) for chord in (result.get("chords") or [])]
                raw_notes = normalize_note_level_predictions(result.get("notes") or [], raw_chords)
                predicted_notes.extend(raw_notes)

                chunk_end_sec = min(audio.size, start + chunk_frames) / TARGET_SR if TARGET_SR > 0 else 0.0
                chunk_total_ms = float(timing.get("chunk_total", 0.0) or 0.0)
                response_time_sec = chunk_end_sec + (chunk_total_ms / 1000.0)
                live_result = _process_live_session_chunk(
                    live_session,
                    notes=result.get("notes") or [],
                    chords=raw_chords,
                    current_time=response_time_sec,
                )

                chunk_summaries.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_start_sec": round(float(start) / TARGET_SR if TARGET_SR > 0 else 0.0, 6),
                        "chunk_end_sec": round(chunk_end_sec, 6),
                        "analysis_path": timing.get("analysis_path") or summary.get("analysis_path"),
                        "chunk_total_ms": chunk_total_ms,
                        "real_time_factor": float(timing.get("real_time_factor", 0.0) or 0.0),
                        "onset_threshold": float(summary.get("live_onset_threshold", 0.0) or 0.0),
                        "profile": str(summary.get("live_onset_threshold_profile") or "unknown"),
                        "experiment": str(summary.get("live_onset_threshold_experiment") or "unknown"),
                        "chunk_rms": float(timing.get("neural_chunk_rms", 0.0) or 0.0),
                        "chunk_peak": float(timing.get("neural_chunk_peak", 0.0) or 0.0),
                        "chunk_crest_factor": float(timing.get("neural_chunk_crest_factor", 0.0) or 0.0),
                        "neural_total_ms": float(timing.get("neural_total", 0.0) or 0.0),
                        "response_time_sec": round(response_time_sec, 6),
                        "coarse_note_count": len(live_result.get("coarse_notes") or []),
                        "refined_note_count": len(live_result.get("refined_notes") or []),
                        "visible_note_count": len(live_session.get_all_notes()),
                        "refinement_version": int(live_result.get("refinement_version", 0) or 0),
                        "emitted_notes": raw_notes,
                    }
                )
                display_snapshots.append(
                    _capture_live_display_snapshot(
                        live_session,
                        event_type="chunk_response",
                        time_sec=response_time_sec,
                        chunk_index=chunk_index,
                        audio_time_sec=chunk_end_sec,
                    )
                )

                next_chunk_end_sec = min(
                    total_duration_sec,
                    ((chunk_index + 2) * chunk_frames) / TARGET_SR if TARGET_SR > 0 else total_duration_sec,
                )
                current_display_time_sec = _drain_live_refinements(
                    live_session,
                    current_time=response_time_sec,
                    deadline_time=next_chunk_end_sec,
                    audio_time_sec=chunk_end_sec,
                    snapshots=display_snapshots,
                    chunk_index=chunk_index,
                )

            current_display_time_sec = _drain_live_refinements(
                live_session,
                current_time=current_display_time_sec,
                deadline_time=None,
                audio_time_sec=total_duration_sec,
                snapshots=display_snapshots,
                chunk_index=None,
            )
            live_session.force_refinement()
            display_snapshots.append(
                _capture_live_display_snapshot(
                    live_session,
                    event_type="finalize",
                    time_sec=max(current_display_time_sec, total_duration_sec),
                    chunk_index=None,
                    audio_time_sec=total_duration_sec,
                )
            )

            if capture_display_inputs:
                # The app renders the score and exports MIDI from exactly the
                # display-surface payload returned by /live/check-refinement
                # (all_notes/all_chords == display_state notes/chords). Capture
                # that same surface plus the session bpm so an offline harness can
                # reproduce both generateMusicXML(...) and the exported MIDI.
                final_display_state = live_session.get_display_state() or {}
                try:
                    app_bpm = float(live_session.get_current_bpm()[0])
                except Exception:
                    app_bpm = 0.0
                captured_display_inputs = {
                    "final_display_input_notes": [dict(note) for note in live_session.get_all_notes()],
                    "final_display_input_chords": [dict(chord) for chord in live_session.coarse_chords],
                    "app_notes": [dict(note) for note in (final_display_state.get("notes") or [])],
                    "app_chords": [dict(chord) for chord in (final_display_state.get("chords") or [])],
                    "app_bpm": app_bpm,
                }
    finally:
        delete_live_session(session_id)
        _clear_stream_session(session_id)

    predicted_notes.sort(key=lambda event: (event["onset_time"], event["midi_note"]))
    result = {
        "notes": predicted_notes,
        "chunks": chunk_summaries,
        "display_snapshots": display_snapshots,
        "final_display_notes": display_snapshots[-1]["notes"] if display_snapshots else [],
        "final_display_note_events": display_snapshots[-1].get("display_note_events", []) if display_snapshots else [],
    }
    result.update(captured_display_inputs)
    return result


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

    normalized_notes = normalize_note_level_predictions(
        result.get("notes") or [],
        result.get("chords") or [],
    )
    return {
        "notes": normalized_notes,
        "analysis_summary": result.get("analysis_summary", {}),
        "timing": result.get("_timing_ms", {}),
        "error": None,
    }


def build_score_musicxml_payload(
    clip_id: str,
    arm: str,
    run: Dict,
    reference_musicxml_path: str,
    reference_bpm: float,
) -> Dict | None:
    app_notes = run.get("app_notes")
    app_chords = run.get("app_chords")
    if app_notes is None or app_chords is None:
        return None
    try:
        app_bpm = float(run.get("app_bpm", 0.0) or 0.0)
    except (TypeError, ValueError):
        app_bpm = 0.0
    return {
        "clip_id": f"{clip_id}__{arm}",
        "benchmark_clip_id": clip_id,
        "benchmark_arm": arm,
        "bpm": app_bpm if app_bpm > 1.0 else float(reference_bpm),
        "notes": app_notes,
        "chords": app_chords,
        "reference_musicxml_path": reference_musicxml_path,
    }


def _display_score_metric_updates(scorediff_metrics: Dict) -> Dict:
    return {
        "display_score_reference_available": True,
        "display_score_reference_pending": False,
        "display_score_reference_path": scorediff_metrics.get("reference_path"),
        "display_score_edit_accuracy": scorediff_metrics.get("score_edit_accuracy"),
        "display_score_edit_cost": scorediff_metrics.get("score_edit_cost"),
        "display_score_reference_cost": scorediff_metrics.get("score_reference_cost"),
        "display_score_exact_token_f1": scorediff_metrics.get("exact_token_f1"),
        "display_score_exact_token_precision": scorediff_metrics.get("exact_token_precision"),
        "display_score_exact_token_recall": scorediff_metrics.get("exact_token_recall"),
        "display_score_exact_token_matches": int(scorediff_metrics.get("exact_token_matched", 0) or 0),
        "display_score_predicted_tokens": int(scorediff_metrics.get("predicted_tokens", 0) or 0),
        "display_score_reference_tokens": int(scorediff_metrics.get("reference_tokens", 0) or 0),
        "display_score_edit_exact_ops": int(scorediff_metrics.get("score_edit_exact_ops", 0) or 0),
        "display_score_edit_substitutions": int(scorediff_metrics.get("score_edit_substitutions", 0) or 0),
        "display_score_edit_deletions": int(scorediff_metrics.get("score_edit_deletions", 0) or 0),
        "display_score_edit_insertions": int(scorediff_metrics.get("score_edit_insertions", 0) or 0),
    }


def merge_reference_musicxml_score_metrics(
    results: Dict,
    payloads: Sequence[Dict],
    output_dir: Path | None = None,
    ref_payload_path: str = "",
) -> None:
    score_summary = {
        "enabled": bool(payloads),
        "payload_count": len(payloads),
        "available_metric_count": 0,
        "missing_metric_count": 0,
        "error": None,
    }
    results["score_reference_musicxml"] = score_summary
    if not payloads:
        return

    repo_root = Path(__file__).resolve().parent.parent
    scorediff_script = repo_root / "tools" / "scorediff" / "run.js"
    if not scorediff_script.exists():
        score_summary["error"] = f"scorediff script not found: {scorediff_script}"
        return

    with tempfile.TemporaryDirectory(prefix="livescore_musicxml_refs_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        payload_paths = []
        dump_dir = (output_dir / "score_payloads") if output_dir is not None else None
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
        for payload in payloads:
            path = temp_dir / f"{payload['clip_id']}.json"
            payload_text = json.dumps(payload, indent=2)
            path.write_text(payload_text, encoding="utf-8")
            if dump_dir is not None:
                (dump_dir / f"{payload['clip_id']}.json").write_text(payload_text, encoding="utf-8")
            payload_paths.append(path)

        json_out = temp_dir / "score_metrics.json"
        command = [
            "node",
            str(scorediff_script),
            "--limit=0",
            "--json-out",
            str(json_out),
        ]
        if ref_payload_path:
            resolved_ref_payload = Path(ref_payload_path).expanduser().resolve()
            if not resolved_ref_payload.exists():
                score_summary["error"] = f"score-ref-payload not found: {resolved_ref_payload}"
                return
            command += ["--ref-payload", str(resolved_ref_payload)]
        command += [str(path) for path in payload_paths]
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            score_summary["error"] = (
                f"scorediff failed with code {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )
            return
        if not json_out.exists():
            score_summary["error"] = "scorediff did not write score_metrics.json"
            return

        scorediff_payload = json.loads(json_out.read_text(encoding="utf-8"))
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "reference_musicxml_score_metrics.json").write_text(
                json.dumps(scorediff_payload, indent=2),
                encoding="utf-8",
            )

    comparisons = scorediff_payload.get("comparisons") or []
    for row in comparisons:
        label = str(row.get("label") or "")
        if "__" not in label:
            score_summary["missing_metric_count"] += 1
            continue
        clip_id, arm = label.split("__", 1)
        clip_entry = (results.get("clips") or {}).get(clip_id)
        if not isinstance(clip_entry, dict) or not isinstance(clip_entry.get(arm), dict):
            score_summary["missing_metric_count"] += 1
            continue
        clip_entry[arm].update(_display_score_metric_updates(row))
        score_summary["available_metric_count"] += 1

    score_summary["missing_metric_count"] = max(0, len(payloads) - score_summary["available_metric_count"])


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
    retro_update_groups: List[Dict] = []
    retro_boundary_logs: List[Dict] = []

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
        boundary_log = {
            "boundary_index": boundary_index,
            "boundary_sec": round(boundary_sec, 6),
            "scan": True,
            "reason": "scan_pending",
            "candidate_logs": [],
        }

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
            boundary_log.update(
                {
                    "scan": bool(gate_decision["scan"]),
                    "reason": str(gate_decision["reason"]),
                    "pre_count": int(gate_decision.get("pre_count", 0) or 0),
                    "post_count": int(gate_decision.get("post_count", 0) or 0),
                    "post_max_confidence": round(float(gate_decision.get("post_max_confidence", 0.0) or 0.0), 6),
                    "seam_rms": round(float(gate_decision.get("seam_rms", 0.0) or 0.0), 6),
                    "seam_activity_ratio": round(
                        float(gate_decision.get("seam_activity_ratio", 0.0) or 0.0),
                        6,
                    ),
                }
            )
            if not gate_decision["scan"]:
                boundaries_skipped += 1
                if gate_decision["reason"] == "no_boundary_activity":
                    skipped_no_boundary_activity += 1
                elif gate_decision["reason"] == "covered_post_boundary":
                    skipped_covered_post_boundary += 1
                retro_boundary_logs.append(boundary_log)
                continue

            if gate_decision["reason"] == "missing_post_boundary":
                scanned_missing_post_boundary += 1
            elif gate_decision["reason"] == "weak_post_boundary":
                scanned_weak_post_boundary += 1
        else:
            boundary_log["reason"] = "forced_scan"

        seam_start_sec = max(0.0, boundary_sec - retro_window_sec)
        seam_end_sec = min(total_duration_sec, boundary_sec + retro_window_sec)
        seam_start_sample = max(0, int(round(seam_start_sec * TARGET_SR)))
        seam_end_sample = min(audio.size, int(round(seam_end_sec * TARGET_SR)))
        seam_audio = audio[seam_start_sample:seam_end_sample]
        if seam_audio.size == 0:
            boundary_log["scan"] = False
            boundary_log["reason"] = "empty_seam_audio"
            retro_boundary_logs.append(boundary_log)
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
            boundary_log["reason"] = "seam_error"
            boundary_log["error"] = str(seam_run.get("error"))
            retro_boundary_logs.append(boundary_log)
            continue

        seam_timing = seam_run.get("timing") or {}
        seam_total_ms = float(seam_timing.get("neural_total", 0.0) or 0.0)
        retro_total_ms += seam_total_ms

        target_chunk_index = None
        boundary_update_time_sec = None
        if corrected_chunks:
            target_chunk_index = min(boundary_index, len(corrected_chunks) - 1)
            updated_chunk_total_ms = float(corrected_chunks[target_chunk_index].get("chunk_total_ms", 0.0) or 0.0) + seam_total_ms
            corrected_chunks[target_chunk_index]["chunk_total_ms"] = updated_chunk_total_ms
            corrected_chunks[target_chunk_index]["real_time_factor"] = float(
                corrected_chunks[target_chunk_index].get("real_time_factor", 0.0) or 0.0
            ) + (seam_total_ms / max(1.0, chunk_seconds * 1000.0))
            chunk_end_sec = min(total_duration_sec, float(target_chunk_index + 1) * float(chunk_seconds))
            boundary_update_time_sec = chunk_end_sec + (updated_chunk_total_ms / 1000.0)
            corrected_chunks[target_chunk_index]["response_time_sec"] = round(boundary_update_time_sec, 6)

        boundary_operations: List[Dict] = []

        for seam_note in seam_run["notes"]:
            candidate = dict(seam_note)
            candidate["onset_time"] = round(float(candidate["onset_time"]) + seam_start_sec, 6)
            candidate["offset_time"] = round(float(candidate["offset_time"]) + seam_start_sec, 6)
            candidate["duration"] = round(
                max(0.0, float(candidate["offset_time"]) - float(candidate["onset_time"])),
                6,
            )
            candidate_log = {
                "midi_note": int(candidate["midi_note"]),
                "onset_time": round(float(candidate["onset_time"]), 6),
                "offset_time": round(float(candidate["offset_time"]), 6),
                "confidence": round(float(candidate.get("confidence", 0.0) or 0.0), 6),
                "boundary_relation": None,
                "decision": None,
            }

            boundary_candidate_reason = classify_retro_boundary_candidate(
                candidate,
                boundary_sec=boundary_sec,
                retro_band_sec=retro_band_sec,
            )
            if boundary_candidate_reason is None:
                rejected_seam_band += 1
                candidate_log["decision"] = "rejected_seam_band"
                boundary_log["candidate_logs"].append(candidate_log)
                continue

            seam_candidates += 1
            candidate_log["boundary_relation"] = boundary_candidate_reason
            if boundary_candidate_reason == "spans_boundary":
                spanning_boundary_candidates += 1

            if float(candidate.get("confidence", 0.0) or 0.0) < retro_min_confidence:
                rejected_confidence += 1
                candidate_log["decision"] = "rejected_confidence"
                boundary_log["candidate_logs"].append(candidate_log)
                continue

            existing_target = find_retro_repair_target(
                corrected_notes,
                candidate,
                boundary_sec=boundary_sec,
                onset_tol=retro_match_onset_tol,
            )
            if existing_target is not None:
                target_index, repair_mode = existing_target
                original_note = dict(corrected_notes[target_index])
                merged_note, changed = merge_retro_candidate_into_note(corrected_notes[target_index], candidate)
                if not changed:
                    rejected_existing += 1
                    candidate_log["decision"] = "rejected_existing"
                    boundary_log["candidate_logs"].append(candidate_log)
                    continue
                merged_note["selection_reason"] = (
                    "retro_seam_replace" if repair_mode == "replace" else "retro_seam_extend"
                )
                merged_note["source_boundary_sec"] = round(boundary_sec, 6)
                corrected_notes[target_index] = merged_note
                candidate_log["decision"] = repair_mode
                boundary_operations.append(
                    {
                        "mode": repair_mode,
                        "old_note": original_note,
                        "new_note": dict(merged_note),
                    }
                )
                if repair_mode == "replace":
                    replaced_existing += 1
                else:
                    extended_existing += 1
                boundary_log["candidate_logs"].append(candidate_log)
                continue

            extra_target = find_retro_repair_target(
                extra_notes,
                candidate,
                boundary_sec=boundary_sec,
                onset_tol=retro_match_onset_tol,
            )
            if extra_target is not None:
                extra_index, repair_mode = extra_target
                original_extra = dict(extra_notes[extra_index])
                merged_extra, changed = merge_retro_candidate_into_note(extra_notes[extra_index], candidate)
                if not changed:
                    rejected_duplicate_extra += 1
                    candidate_log["decision"] = "rejected_duplicate_extra"
                    boundary_log["candidate_logs"].append(candidate_log)
                    continue
                merged_extra["selection_reason"] = (
                    "retro_seam_replace" if repair_mode == "replace" else "retro_seam_extend"
                )
                merged_extra["source_boundary_sec"] = round(boundary_sec, 6)
                extra_notes[extra_index] = merged_extra
                candidate_log["decision"] = repair_mode
                boundary_operations.append(
                    {
                        "mode": repair_mode,
                        "old_note": original_extra,
                        "new_note": dict(merged_extra),
                    }
                )
                updated_extra_notes += 1
                boundary_log["candidate_logs"].append(candidate_log)
                continue

            candidate["selection_reason"] = "retro_seam_extra"
            candidate["source_boundary_sec"] = round(boundary_sec, 6)
            extra_notes.append(candidate)
            candidate_log["decision"] = "extra"
            boundary_operations.append(
                {
                    "mode": "extra",
                    "old_note": None,
                    "new_note": dict(candidate),
                }
            )
            extras_added += 1
            boundary_log["candidate_logs"].append(candidate_log)

        if boundary_operations:
            retro_update_groups.append(
                {
                    "time_sec": round(
                        float(boundary_update_time_sec if boundary_update_time_sec is not None else boundary_sec),
                        6,
                    ),
                    "audio_time_sec": round(
                        float(chunk_end_sec if target_chunk_index is not None else boundary_sec),
                        6,
                    ),
                    "chunk_index": target_chunk_index,
                    "boundary_sec": round(boundary_sec, 6),
                    "operations": boundary_operations,
                }
            )

            retro_boundary_logs.append(boundary_log)

    corrected_notes = sorted(
        [*corrected_notes, *extra_notes],
        key=lambda event: (event["onset_time"], event["midi_note"]),
    )

    retro_display_snapshots, retro_final_display_notes = _build_retro_display_snapshots(
        baseline_run.get("display_snapshots") or [],
        retro_update_groups,
    )
    if not retro_display_snapshots:
        fallback_final_notes = normalize_predicted_notes(corrected_notes)
        fallback_time_sec = max(
            total_duration_sec,
            max((float(update_group.get("time_sec", 0.0) or 0.0) for update_group in retro_update_groups), default=0.0),
        )
        retro_display_snapshots = [
            {
                "event_type": "retro_finalize",
                "time_sec": round(fallback_time_sec, 6),
                "audio_time_sec": round(float(total_duration_sec), 6),
                "chunk_index": None,
                "refinement_version": 0,
                "visible_note_count": len(fallback_final_notes),
                "display_note_event_count": len(fallback_final_notes),
                "notes": fallback_final_notes,
                "display_note_events": fallback_final_notes,
            }
        ]
        retro_final_display_notes = fallback_final_notes

    selected_thresholds = sorted({rounded_baseline, rounded_retro_threshold})
    return {
        "notes": corrected_notes,
        "chunks": corrected_chunks,
        "display_snapshots": retro_display_snapshots,
        "final_display_notes": retro_final_display_notes,
        "final_display_note_events": retro_display_snapshots[-1].get("display_note_events", retro_final_display_notes)
        if retro_display_snapshots else retro_final_display_notes,
        "boundary_logs": retro_boundary_logs,
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
    cluster_metric_mode: str = "raw",
) -> Dict:
    note_metrics = compute_note_metrics(run["notes"], gt_notes)
    strict_onset_metrics = compute_onset_tolerance_sweep(run["notes"], gt_notes)
    offset_metrics = compute_offset_metrics(run["notes"], gt_notes)
    note_value_metrics = compute_note_value_metrics(run["notes"], gt_notes, reference_bpm=reference_bpm)
    duplicate_metrics = compute_duplicate_metrics(run["notes"])
    final_display_metrics = compute_final_display_accuracy_metrics(
        run,
        gt_notes,
        reference_bpm=reference_bpm,
        cluster_metric_mode=cluster_metric_mode,
    )
    boundary_metrics = compute_boundary_miss_metrics(
        run["notes"],
        gt_notes,
        chunk_seconds=chunk_seconds,
        boundary_band_sec=boundary_band_sec,
    )
    notation_metrics = compute_notation_metrics(run, gt_notes)
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
        **final_display_metrics,
        **boundary_metrics,
        **notation_metrics,
        "reference_bpm": float(reference_bpm),
        "strict_onset_metrics": strict_onset_metrics,
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
    strict_onset_metrics = metrics.get("strict_onset_metrics") or {}
    if strict_onset_metrics:
        strict_parts = " ".join(
            f"{label}={strict_metrics['f1']:.3f}"
            for label, strict_metrics in sorted(
                strict_onset_metrics.items(),
                key=lambda item: item[1].get("onset_tol_ms", 0),
            )
        )
        print(f"      strict onset f1: {strict_parts}")
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
    if metrics.get("display_snapshot_count", 0) > 0:
        print(
            "      notation metrics: "
            f"first_correct_ms={metrics['time_to_first_correct_note_ms']:.1f} "
            f"visible_p50_ms={metrics['time_to_visible_median_ms']:.1f} visible_p95_ms={metrics['time_to_visible_p95_ms']:.1f} "
            f"stabilize_p50_ms={metrics['stabilization_latency_median_ms']:.1f} stabilize_p95_ms={metrics['stabilization_latency_p95_ms']:.1f} "
            f"avg_revisions={metrics['avg_revision_count']:.2f} matched={metrics['matched_display_notes']} "
            f"refined={metrics['stabilized_display_notes']} snapshots={metrics['display_snapshot_count']}"
        )
        print(
            "      algorithmic latency: "
            f"first_correct_ms={metrics['algorithmic_time_to_first_correct_note_ms']:.1f} "
            f"visible_p50_ms={metrics['algorithmic_time_to_visible_median_ms']:.1f} "
            f"visible_p95_ms={metrics['algorithmic_time_to_visible_p95_ms']:.1f} "
            f"stabilize_p50_ms={metrics['algorithmic_stabilization_latency_median_ms']:.1f} "
            f"stabilize_p95_ms={metrics['algorithmic_stabilization_latency_p95_ms']:.1f}"
        )
    if metrics.get("display_final_note_event_count", 0) > 0:
        print(
            "      final display note metrics: "
            f"precision={metrics['display_note_precision']:.3f} recall={metrics['display_note_recall']:.3f} "
            f"f1={metrics['display_note_f1']:.3f} matched={metrics['display_note_matched']}/{metrics['display_note_ground_truth']} "
            f"pred={metrics['display_note_predicted']}"
        )
        display_strict_metrics = metrics.get("display_strict_onset_metrics") or {}
        if display_strict_metrics:
            strict_parts = " ".join(
                f"{label}={strict_metrics['f1']:.3f}"
                for label, strict_metrics in sorted(
                    display_strict_metrics.items(),
                    key=lambda item: item[1].get("onset_tol_ms", 0),
                )
            )
            print(f"      final display strict onset f1: {strict_parts}")
        print(
            "      final display timing: "
            f"offset_f1={metrics['display_offset_f1']:.3f} rhythm_precision={metrics['display_note_rhythm_precision']:.3f} "
            f"note_value_acc={metrics['display_note_value_accuracy']:.3f} "
            f"note_value_n={metrics['display_note_value_matched']} "
            f"avg_beat_error={metrics['display_note_value_avg_beat_error']:.3f}"
        )
        print(
            "      final display structure: "
            f"cluster_f1={metrics['display_cluster_f1']:.3f} "
            f"onset_align_recall={metrics['display_cluster_onset_alignment_recall']:.3f} "
            f"avg_jaccard={metrics['display_cluster_avg_jaccard']:.3f} "
            f"exact={metrics['display_cluster_exact_matches']}/{metrics['display_cluster_ground_truth']} "
            f"pred={metrics['display_cluster_predicted']} over={metrics['display_cluster_overclustered_matches']} "
            f"under={metrics['display_cluster_underclustered_matches']} "
            f"pitch_conflicts={metrics['display_cluster_pitch_conflict_matches']} "
            f"missed_gt={metrics['display_cluster_unmatched_ground_truth']}"
        )
        if metrics.get("display_score_reference_available"):
            print(
                "      final display score edit: "
                f"accuracy={metrics['display_score_edit_accuracy']:.3f} "
                f"exact_token_f1={metrics['display_score_exact_token_f1']:.3f} "
                f"edit={metrics['display_score_edit_cost']:.2f}/{metrics['display_score_reference_cost']:.2f} "
                f"ops exact={metrics['display_score_edit_exact_ops']} sub={metrics['display_score_edit_substitutions']} "
                f"del={metrics['display_score_edit_deletions']} ins={metrics['display_score_edit_insertions']}"
            )
        else:
            state = "pending batch MusicXML comparison" if metrics.get("display_score_reference_pending") else "unavailable (no reference MusicXML)"
            print(f"      final display score edit: {state}")
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
    parts = [
        f"precision={candidate['precision'] - reference['precision']:+.3f}",
        f"recall={candidate['recall'] - reference['recall']:+.3f}",
        f"f1={candidate['f1'] - reference['f1']:+.3f}",
        f"offset_f1={candidate['offset_f1'] - reference['offset_f1']:+.3f}",
        f"note_value_acc={candidate['note_value_accuracy'] - reference['note_value_accuracy']:+.3f}",
        f"dup_per_100={candidate['duplicates_per_100_notes'] - reference['duplicates_per_100_notes']:+.2f}",
        f"boundary_miss_rate={candidate['boundary_miss_rate'] - reference['boundary_miss_rate']:+.3f}",
        f"p95_chunk_ms={candidate['p95_chunk_total_ms'] - reference['p95_chunk_total_ms']:+.2f}",
    ]
    reference_strict = reference.get("strict_onset_metrics") or {}
    candidate_strict = candidate.get("strict_onset_metrics") or {}
    for label, strict_metrics in sorted(
        candidate_strict.items(),
        key=lambda item: item[1].get("onset_tol_ms", 0),
    ):
        if label not in reference_strict:
            continue
        parts.append(f"f1_{label}={strict_metrics['f1'] - reference_strict[label]['f1']:+.3f}")
    if reference.get("display_snapshot_count", 0) > 0 and candidate.get("display_snapshot_count", 0) > 0:
        parts.extend(
            [
                f"visible_p50_ms={candidate['time_to_visible_median_ms'] - reference['time_to_visible_median_ms']:+.1f}",
                f"stabilize_p50_ms={candidate['stabilization_latency_median_ms'] - reference['stabilization_latency_median_ms']:+.1f}",
                f"algo_visible_p50_ms={candidate['algorithmic_time_to_visible_median_ms'] - reference['algorithmic_time_to_visible_median_ms']:+.1f}",
                f"algo_stabilize_p50_ms={candidate['algorithmic_stabilization_latency_median_ms'] - reference['algorithmic_stabilization_latency_median_ms']:+.1f}",
                f"avg_revisions={candidate['avg_revision_count'] - reference['avg_revision_count']:+.2f}",
            ]
        )
    if reference.get("display_final_note_event_count", 0) > 0 and candidate.get("display_final_note_event_count", 0) > 0:
        parts.extend(
            [
                f"display_note_f1={candidate['display_note_f1'] - reference['display_note_f1']:+.3f}",
                f"display_offset_f1={candidate['display_offset_f1'] - reference['display_offset_f1']:+.3f}",
                f"display_cluster_f1={candidate['display_cluster_f1'] - reference['display_cluster_f1']:+.3f}",
                f"display_cluster_jaccard={candidate['display_cluster_avg_jaccard'] - reference['display_cluster_avg_jaccard']:+.3f}",
            ]
        )
        if (
            reference.get("display_score_edit_accuracy") is not None
            and candidate.get("display_score_edit_accuracy") is not None
        ):
            parts.extend(
                [
                    f"display_score_edit_acc={candidate['display_score_edit_accuracy'] - reference['display_score_edit_accuracy']:+.3f}",
                    f"display_score_exact_token_f1={candidate['display_score_exact_token_f1'] - reference['display_score_exact_token_f1']:+.3f}",
                ]
            )
    print(f"    delta {candidate_label}-{reference_label}: {' '.join(parts)}")


def _build_paired_metric_stats(reference_values: Dict[str, float], candidate_values: Dict[str, float]) -> Dict | None:
    shared_clip_ids = sorted(set(reference_values).intersection(candidate_values))
    if not shared_clip_ids:
        return None

    diffs = [candidate_values[clip_id] - reference_values[clip_id] for clip_id in shared_clip_ids]
    reference_mean = sum(reference_values[clip_id] for clip_id in shared_clip_ids) / len(shared_clip_ids)
    candidate_mean = sum(candidate_values[clip_id] for clip_id in shared_clip_ids) / len(shared_clip_ids)
    mean_diff = sum(diffs) / len(diffs)

    bootstrap_rng = random.Random(PAIRED_STATS_RANDOM_SEED)
    bootstrap_means = []
    for _ in range(PAIRED_STATS_BOOTSTRAP_SAMPLES):
        sample_total = 0.0
        for _ in range(len(diffs)):
            sample_total += diffs[bootstrap_rng.randrange(len(diffs))]
        bootstrap_means.append(sample_total / len(diffs))
    bootstrap_means.sort()
    ci_low = bootstrap_means[int(0.025 * len(bootstrap_means))]
    ci_high = bootstrap_means[int(0.975 * len(bootstrap_means))]

    observed_total = abs(sum(diffs))
    permutation_rng = random.Random(PAIRED_STATS_RANDOM_SEED + 1)
    at_least_as_extreme = 0
    for _ in range(PAIRED_STATS_BOOTSTRAP_SAMPLES):
        permuted_total = 0.0
        for diff in diffs:
            permuted_total += diff if permutation_rng.random() < 0.5 else -diff
        if abs(permuted_total) >= observed_total - PAIRED_STATS_EPSILON:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (PAIRED_STATS_BOOTSTRAP_SAMPLES + 1)

    improved = sum(diff > PAIRED_STATS_EPSILON for diff in diffs)
    regressed = sum(diff < -PAIRED_STATS_EPSILON for diff in diffs)
    unchanged = len(diffs) - improved - regressed
    clip_deltas = list(zip(shared_clip_ids, diffs))
    best_clip_id, best_diff = max(clip_deltas, key=lambda item: item[1])
    worst_clip_id, worst_diff = min(clip_deltas, key=lambda item: item[1])

    return {
        "shared_clip_count": len(shared_clip_ids),
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "mean_diff": mean_diff,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "randomization_p_value": p_value,
        "improved_clip_count": improved,
        "regressed_clip_count": regressed,
        "unchanged_clip_count": unchanged,
        "best_clip_id": best_clip_id,
        "best_clip_diff": best_diff,
        "worst_clip_id": worst_clip_id,
        "worst_clip_diff": worst_diff,
    }


def _extract_run_metric_values(results: Dict, run_label: str, metric_name: str) -> Dict[str, float]:
    metric_values: Dict[str, float] = {}
    for clip_id, clip_results in (results.get("clips") or {}).items():
        run_summary = clip_results.get(run_label) or {}
        metric_value = run_summary.get(metric_name)
        if metric_value is None:
            continue
        metric_values[clip_id] = float(metric_value)
    return metric_values


def build_paired_run_stats(results: Dict, reference_label: str, candidate_label: str) -> Dict[str, Dict]:
    paired_stats: Dict[str, Dict] = {}
    for metric_name, _ in PAIRED_DISPLAY_METRICS:
        reference_values = _extract_run_metric_values(results, reference_label, metric_name)
        candidate_values = _extract_run_metric_values(results, candidate_label, metric_name)
        metric_stats = _build_paired_metric_stats(reference_values, candidate_values)
        if metric_stats is not None:
            paired_stats[metric_name] = metric_stats
    return paired_stats


def print_paired_run_stats(reference_label: str, candidate_label: str, paired_stats: Dict[str, Dict]) -> None:
    if not paired_stats:
        return

    print(f"    paired_stats {candidate_label}-{reference_label}:")
    for metric_name, metric_label in PAIRED_DISPLAY_METRICS:
        metric_stats = paired_stats.get(metric_name)
        if metric_stats is None:
            continue
        print(
            "      "
            f"{metric_label}: ref={metric_stats['reference_mean']:.4f} cand={metric_stats['candidate_mean']:.4f} "
            f"diff={metric_stats['mean_diff']:+.4f} "
            f"95%CI=[{metric_stats['bootstrap_ci_low']:+.4f}, {metric_stats['bootstrap_ci_high']:+.4f}] "
            f"p={metric_stats['randomization_p_value']:.4f} "
            f"signs={metric_stats['improved_clip_count']}/{metric_stats['regressed_clip_count']}/{metric_stats['unchanged_clip_count']} "
            f"best={metric_stats['best_clip_id']} {metric_stats['best_clip_diff']:+.4f} "
            f"worst={metric_stats['worst_clip_id']} {metric_stats['worst_clip_diff']:+.4f}"
        )


def compare_saved_result_jsons(reference_path: str, candidate_path: str) -> None:
    reference_file = Path(reference_path)
    candidate_file = Path(candidate_path)
    reference_results = json.loads(reference_file.read_text(encoding="utf-8"))
    candidate_results = json.loads(candidate_file.read_text(encoding="utf-8"))

    print(f"Comparing saved benchmark results: {candidate_file.name} vs {reference_file.name}")
    for run_label in ("control", "treatment", "retro_correction"):
        run_comparison_stats: Dict[str, Dict] = {}
        for metric_name, _ in PAIRED_DISPLAY_METRICS:
            reference_values = _extract_run_metric_values(reference_results, run_label, metric_name)
            candidate_values = _extract_run_metric_values(candidate_results, run_label, metric_name)
            metric_stats = _build_paired_metric_stats(reference_values, candidate_values)
            if metric_stats is not None:
                run_comparison_stats[metric_name] = metric_stats
        if not run_comparison_stats:
            continue

        print(f"  run={run_label}")
        print_paired_run_stats(reference_file.stem, candidate_file.stem, run_comparison_stats)


def print_boundary_failure_diagnostics(
    boundary_diagnostics: Dict,
    limit: int = BOUNDARY_DIAGNOSTIC_PRINT_LIMIT,
) -> None:
    missed_notes = boundary_diagnostics.get("missed_notes") or []
    if not missed_notes:
        return

    print(
        "      boundary diagnostics: "
        f"missed={len(missed_notes)} by_arm={boundary_diagnostics.get('missed_by_arm')} "
        f"tags={boundary_diagnostics.get('diagnosis_tag_counts', {})}"
    )
    for diagnostic in missed_notes[:limit]:
        retro_status = "n/a"
        retro = diagnostic.get("retro") or {}
        retro_boundary = retro.get("boundary_scan") or {}
        if retro_boundary:
            if retro_boundary.get("scanned") is False:
                retro_status = f"skip:{retro_boundary.get('reason')}"
            elif retro_boundary.get("candidate_decisions"):
                retro_status = "/".join(retro_boundary["candidate_decisions"])
            else:
                retro_status = str(retro_boundary.get("reason") or "no_candidate")

        print(
            "        miss: "
            f"pitch={diagnostic['midi_note']} onset={diagnostic['gt_onset_time']:.3f}s "
            f"boundary_delta_ms={diagnostic['distance_to_boundary_ms']:+.1f} "
            f"control_coarse={diagnostic['control']['coarse_candidate']['present']} "
            f"treatment_coarse={diagnostic['treatment']['coarse_candidate']['present']} "
            f"retro={retro_status}"
        )

    if len(missed_notes) > limit:
        print(f"        ... {len(missed_notes) - limit} more missed boundary notes in diagnostics JSON")


def print_failure_bucket_summary(bucket_index: Dict[str, List[str]]) -> None:
    if not bucket_index:
        return

    print("\nFailure buckets:")
    for bucket in FAILURE_BUCKET_ORDER:
        clip_ids = bucket_index.get(bucket) or []
        if not clip_ids:
            continue
        print(f"  {bucket}: {len(clip_ids)} clips -> {', '.join(clip_ids)}")


def print_boundary_tag_summary(tag_counts: Dict[str, int]) -> None:
    if not tag_counts:
        return

    print("\nBoundary failure tags:")
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {tag}: {count}")


async def warmup_live_path(chunk_seconds: float) -> None:
    warmup_audio = np.zeros(max(1, int(round(chunk_seconds * TARGET_SR))), dtype=np.float32)
    await run_live_excerpt(warmup_audio, adaptive_onset_threshold=False, chunk_seconds=chunk_seconds, noise_profile="balanced")


async def run_experiment(args: argparse.Namespace) -> Dict:
    selected, manifest_path, selection_mode = select_benchmark_clips(args)
    missing_reference_musicxml = find_missing_reference_musicxml(selected, args.reference_musicxml_dir)
    if args.require_reference_musicxml and missing_reference_musicxml:
        raise RuntimeError(
            "Missing reference MusicXML for clips: "
            + ", ".join(missing_reference_musicxml[:20])
            + (" ..." if len(missing_reference_musicxml) > 20 else "")
        )
    failure_bucket_index: Dict[str, List[str]] = {bucket: [] for bucket in FAILURE_BUCKET_ORDER}
    boundary_failure_tag_counts: Counter = Counter()
    cluster_metric_mode = "raw"
    if bool(args.cluster_metric_slot_consensus):
        cluster_metric_mode = "slot_consensus"
    elif bool(args.cluster_metric_grid_snap):
        cluster_metric_mode = "grid_snap"

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
            "clip_ids": list(selected.keys()),
            "noise_profile": args.noise_profile,
            "cluster_metric_mode": cluster_metric_mode,
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
            "selection_mode": selection_mode,
            "benchmark_manifest": str(manifest_path) if manifest_path is not None else None,
            "reference_musicxml_dir": args.reference_musicxml_dir,
            "require_reference_musicxml": bool(args.require_reference_musicxml),
            "reference_musicxml_missing_clip_count": len(missing_reference_musicxml),
        },
        "clips": {},
        "failure_buckets": {},
        "boundary_failure_tag_counts": {},
    }
    score_musicxml_payloads: List[Dict] = []

    for clip_id, clip in selected.items():

        audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
        reference_bpm = get_midi_reference_bpm(clip["midi_path"])
        reference_musicxml_path = resolve_reference_musicxml_path(
            clip_id,
            clip,
            args.reference_musicxml_dir,
        )
        capture_score_inputs = bool(reference_musicxml_path)
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
            capture_display_inputs=capture_score_inputs,
        )
        treatment_run = await run_live_excerpt(
            audio,
            adaptive_onset_threshold=True,
            chunk_seconds=args.chunk_seconds,
            noise_profile=args.noise_profile,
            capture_display_inputs=capture_score_inputs,
        )

        control_summary = summarize_run(
            control_run,
            gt_notes,
            chunk_seconds=args.chunk_seconds,
            boundary_band_sec=args.eval_boundary_band_sec,
            reference_bpm=reference_bpm,
            cluster_metric_mode=cluster_metric_mode,
        )
        treatment_summary = summarize_run(
            treatment_run,
            gt_notes,
            chunk_seconds=args.chunk_seconds,
            boundary_band_sec=args.eval_boundary_band_sec,
            reference_bpm=reference_bpm,
            cluster_metric_mode=cluster_metric_mode,
        )
        if reference_musicxml_path:
            for summary in (control_summary, treatment_summary):
                summary["display_score_reference_pending"] = True
                summary["display_score_reference_path"] = reference_musicxml_path

        clip_results = {
            "clip": clip,
            "ground_truth_notes": len(gt_notes),
            "reference_bpm": round(float(reference_bpm), 3),
            "reference_musicxml_path": reference_musicxml_path or None,
            "control": control_summary,
            "treatment": treatment_summary,
        }
        if reference_musicxml_path:
            for arm, run in (("control", control_run), ("treatment", treatment_run)):
                payload = build_score_musicxml_payload(
                    clip_id,
                    arm,
                    run,
                    reference_musicxml_path,
                    reference_bpm,
                )
                if payload is not None:
                    score_musicxml_payloads.append(payload)

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
            if reference_musicxml_path:
                payload = build_score_musicxml_payload(
                    clip_id,
                    "retro_correction",
                    retro_run,
                    reference_musicxml_path,
                    reference_bpm,
                )
                if payload is not None:
                    score_musicxml_payloads.append(payload)
            retro_summary = summarize_run(
                retro_run,
                gt_notes,
                chunk_seconds=args.chunk_seconds,
                boundary_band_sec=args.eval_boundary_band_sec,
                reference_bpm=reference_bpm,
                cluster_metric_mode=cluster_metric_mode,
            )
            if reference_musicxml_path:
                retro_summary["display_score_reference_pending"] = True
                retro_summary["display_score_reference_path"] = reference_musicxml_path
            clip_results["retro_correction"] = retro_summary

        boundary_failure_diagnostics = build_clip_boundary_failure_diagnostics(
            control_run,
            treatment_run,
            gt_notes,
            chunk_seconds=args.chunk_seconds,
            boundary_band_sec=args.eval_boundary_band_sec,
            retro_run=retro_run if args.run_retro_correction else None,
        )
        failure_buckets = classify_clip_failure_buckets(control_summary, treatment_summary, retro_summary)
        clip_results["boundary_failure_diagnostics"] = boundary_failure_diagnostics
        clip_results["failure_buckets"] = failure_buckets
        for bucket in failure_buckets:
            failure_bucket_index[bucket].append(clip_id)
        boundary_failure_tag_counts.update(boundary_failure_diagnostics.get("diagnosis_tag_counts") or {})

        results["clips"][clip_id] = clip_results

        print(f"\n[{clip_id.upper()}] {clip['title']}")
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
        if failure_buckets:
            print(f"    failure buckets: {failure_buckets}")
        print_boundary_failure_diagnostics(boundary_failure_diagnostics)

    results["failure_buckets"] = {
        bucket: clip_ids
        for bucket, clip_ids in failure_bucket_index.items()
        if clip_ids
    }
    results["boundary_failure_tag_counts"] = dict(
        sorted(boundary_failure_tag_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    score_metrics_output_dir = Path(args.output_json).expanduser().resolve().parent if args.output_json else None
    merge_reference_musicxml_score_metrics(
        results,
        score_musicxml_payloads,
        output_dir=score_metrics_output_dir,
        ref_payload_path=getattr(args, "score_ref_payload", "") or "",
    )
    score_ref_summary = results.get("score_reference_musicxml") or {}
    if score_ref_summary.get("enabled"):
        print(
            "Reference MusicXML score metrics: "
            f"{score_ref_summary.get('available_metric_count', 0)}/{score_ref_summary.get('payload_count', 0)} payloads scored"
        )
        if score_ref_summary.get("error"):
            print(f"  error: {score_ref_summary['error']}")
    elif args.reference_musicxml_dir or any(
        any((clip.get(field) for field in REFERENCE_MUSICXML_FIELDS)) for clip in selected.values()
    ):
        print("Reference MusicXML score metrics: no usable reference payloads")
    results["paired_stats"] = {}
    treatment_paired_stats = build_paired_run_stats(results, "control", "treatment")
    if treatment_paired_stats:
        results["paired_stats"]["treatment_vs_control"] = treatment_paired_stats
        print_paired_run_stats("control", "treatment", treatment_paired_stats)
    if args.run_retro_correction:
        retro_paired_stats = build_paired_run_stats(results, "control", "retro_correction")
        if retro_paired_stats:
            results["paired_stats"]["retro_correction_vs_control"] = retro_paired_stats
            print_paired_run_stats("control", "retro_correction", retro_paired_stats)
    print_failure_bucket_summary(results["failure_buckets"])
    print_boundary_tag_summary(results["boundary_failure_tag_counts"])

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
    parser.add_argument("--selection-strategy", choices=[DEFAULT_SELECTION_STRATEGY, DIVERSE_SUITE_SELECTION_STRATEGY], default=DEFAULT_SELECTION_STRATEGY, help="How to choose clips when generating a benchmark suite without a manifest")
    parser.add_argument("--target-clips", type=int, default=DEFAULT_TARGET_CLIPS, help="Target number of clips when using the diverse suite selector")
    parser.add_argument("--max-clips-per-piece", type=int, default=DEFAULT_MAX_CLIPS_PER_PIECE, help="Maximum number of clips selected from any one piece when using the diverse suite selector")
    parser.add_argument("--max-clips-per-title", type=int, default=DEFAULT_MAX_CLIPS_PER_TITLE, help="Maximum number of clips with the same normalized title when using the diverse suite selector")
    parser.add_argument("--benchmark-manifest", type=str, default="", help="Optional path to a fixed benchmark clip manifest; skips dynamic clip selection when provided")
    parser.add_argument("--compare-result-jsons", nargs=2, metavar=("REFERENCE_JSON", "CANDIDATE_JSON"), default=[], help="Skip the experiment and compute paired metric deltas between two saved results JSON files")
    parser.add_argument("--write-selected-manifest", type=str, default="", help="Optional path to save the dynamically selected clips as a fixed benchmark manifest")
    parser.add_argument("--selection-only", action="store_true", help="Only build and optionally write the selected benchmark clips; do not run the live experiment")
    parser.add_argument("--clip-ids", nargs="+", default=[], help="Optional subset of manifest clip IDs to run when using --benchmark-manifest")
    parser.add_argument("--reference-musicxml-dir", type=str, default="", help="Directory containing true reference MusicXML files keyed by clip id, MIDI stem, or audio stem")
    parser.add_argument("--require-reference-musicxml", action="store_true", help="Fail if any selected benchmark clip lacks a true reference MusicXML file")
    parser.add_argument("--score-ref-payload", type=str, default="", help="GT-payload JSON (e.g. oracle_gt_midi_payloads.json); when set, scorediff regenerates each reference score at the predicted clip's bpm (tempo-normalized) instead of using the static reference MusicXML")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(DEFAULT_CATEGORIES),
        default=list(DEFAULT_CATEGORIES),
        help="Which excerpt categories to benchmark",
    )
    parser.add_argument("--noise-profile", choices=["open", "balanced", "clean"], default="balanced", help="Noise profile to pass into the live chunk analyzer")
    parser.add_argument(
        "--cluster-metric-grid-snap",
        action="store_true",
        help="Use snapped grid onsets only for display cluster metrics when notes provide cluster_metric_time_seconds.",
    )
    parser.add_argument(
        "--cluster-metric-slot-consensus",
        action="store_true",
        help="Use a shared raw-onset median for high-confidence multi-note grid slots when computing display cluster metrics.",
    )
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
    args = parser.parse_args()
    if args.compare_result_jsons and args.selection_only:
        parser.error("--compare-result-jsons cannot be combined with --selection-only")
    if args.cluster_metric_grid_snap and args.cluster_metric_slot_consensus:
        parser.error("--cluster-metric-grid-snap and --cluster-metric-slot-consensus cannot be combined")
    if args.benchmark_manifest and args.write_selected_manifest:
        parser.error("--benchmark-manifest and --write-selected-manifest cannot be used together")
    if args.selection_only and args.warmup:
        args.warmup = False
    return args


def main() -> None:
    args = parse_args()

    if args.compare_result_jsons:
        compare_saved_result_jsons(args.compare_result_jsons[0], args.compare_result_jsons[1])
        return

    if args.selection_only:
        selected, manifest_path, selection_mode = select_benchmark_clips(args)
        print(f"Selection mode: {selection_mode}")
        if manifest_path is not None:
            print(f"Loaded benchmark manifest from {manifest_path}")
        if args.write_selected_manifest:
            written_manifest = write_benchmark_manifest(args.write_selected_manifest, selected, args)
            print(f"Saved benchmark manifest to {written_manifest}")
        print_selection_summary(selected)
        missing_reference_musicxml = find_missing_reference_musicxml(selected, args.reference_musicxml_dir)
        if missing_reference_musicxml:
            print(
                f"Reference MusicXML missing for {len(missing_reference_musicxml)} clips"
                + (f": {', '.join(missing_reference_musicxml[:20])}" if len(missing_reference_musicxml) <= 20 else "")
            )
            if args.require_reference_musicxml:
                raise RuntimeError("Selection contains clips without reference MusicXML")
        return

    results = asyncio.run(run_experiment(args))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()
