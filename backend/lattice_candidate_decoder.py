"""Runtime calibrated weak-candidate (inner-voice) decoder.

This is the live counterpart of the offline experiment in
``rhythm_training/train_lattice_candidate_calibrator.py``. Quiet inner voices
played under held outer notes reach the enhanced-mel model as onset peaks just
below the primary onset threshold (frame/velocity evidence present). The primary
decode drops them, and the downstream live continuity gates would drop them
again as "harmonic sustain" / "weak birth outside attack".

This module rescues those candidates with the *calibrated* policy that was
validated offline: for each below-threshold onset peak with frame/velocity
support, build the same 14 features used in training, score them with the
exported logistic model, accept above the calibrated threshold, and snap the
accepted event onto the nearest primary onset cluster.

The feature math, candidate generation, and acceptance/snapping are ported
verbatim from the trainer so the offline calibration transfers. Scoring is pure
numpy (``StandardScaler`` + ``LogisticRegression`` reproduced from exported
weights), so the live path needs no sklearn/pickle dependency.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Mirrors rhythm_training/train_mel_baseline.py.
SAMPLE_RATE = 16000
HOP_LENGTH = 256
PIANO_KEYS = 88
MIDI_OFFSET = 21

# Mirrors train_lattice_candidate_calibrator.HARMONIC_INTERVAL_CLASSES so the
# ``harmonic_to_anchor`` feature is computed identically to training.
HARMONIC_INTERVAL_CLASSES = {0, 7, 12, 19, 24, 28, 31, 36}

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent
    / "rhythm_training"
    / "lattice_candidate_calibrator.json"
)

_MODEL_CACHE: Dict[str, Optional["LatticeCalibrator"]] = {}


class LatticeCalibrator:
    """Loaded calibrator: scaler + logistic weights + candidate knobs."""

    def __init__(self, payload: Dict):
        self.feature_names: List[str] = list(payload.get("feature_names") or [])
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.coef = np.asarray(payload["coef"], dtype=np.float64)
        self.intercept = float(payload["intercept"])
        self.threshold = float(payload["threshold"])
        self.candidate_args: Dict = dict(payload.get("candidate_args") or {})

    def score(self, features: np.ndarray) -> np.ndarray:
        """Reproduce ``Pipeline(StandardScaler, LogisticRegression).predict_proba[:, 1]``."""
        if features.size == 0:
            return np.zeros((0,), dtype=np.float64)
        z = (features.astype(np.float64) - self.scaler_mean) / self.scaler_scale
        logit = z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logit))


def load_calibrator(model_path: Optional[str] = None) -> Optional[LatticeCalibrator]:
    """Load (and cache) the calibrator JSON. Returns None if unavailable."""
    path = model_path or os.environ.get("LIVE_LATTICE_MODEL") or str(_DEFAULT_MODEL_PATH)
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    calibrator: Optional[LatticeCalibrator] = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        calibrator = LatticeCalibrator(payload)
    except (OSError, ValueError, KeyError):
        calibrator = None
    _MODEL_CACHE[path] = calibrator
    return calibrator


# --- ported feature / candidate / acceptance helpers (keep in sync with trainer) ---


def _peak_frames(probs: np.ndarray, threshold: float) -> List[int]:
    peaks: List[int] = []
    for idx in range(probs.shape[0]):
        left = probs[idx - 1] if idx > 0 else -np.inf
        right = probs[idx + 1] if idx + 1 < probs.shape[0] else -np.inf
        if probs[idx] >= threshold and probs[idx] >= left and probs[idx] >= right:
            peaks.append(int(idx))
    return peaks


def _cluster_primary_events(events: Sequence[Dict], tolerance_sec: float) -> List[Dict]:
    clusters: List[Dict] = []
    for event in sorted(events, key=lambda item: float(item["onset_time"])):
        onset = float(event["onset_time"])
        midi = int(event["midi_note"])
        if clusters and abs(onset - float(clusters[-1]["time"])) <= tolerance_sec:
            cluster = clusters[-1]
            n = int(cluster["count"])
            cluster["time"] = ((float(cluster["time"]) * n) + onset) / float(n + 1)
            cluster["count"] = n + 1
            cluster["pitches"].append(midi)
        else:
            clusters.append({"time": onset, "count": 1, "pitches": [midi]})
    return clusters


def _nearest_anchor(clusters: Sequence[Dict], time_sec: float) -> Tuple[Optional[Dict], float]:
    best = None
    best_error = float("inf")
    for cluster in clusters:
        error = abs(float(cluster["time"]) - time_sec)
        if error < best_error:
            best = cluster
            best_error = error
    return best, best_error


def _primary_match_exists(
    primary_events: Sequence[Dict],
    midi_note: int,
    time_sec: float,
    tolerance_sec: float,
) -> bool:
    for event in primary_events:
        if int(event["midi_note"]) != int(midi_note):
            continue
        if abs(float(event["onset_time"]) - float(time_sec)) <= tolerance_sec:
            return True
    return False


def _frame_drop_offset(
    frame_probs: np.ndarray,
    key: int,
    onset_frame: int,
    frame_threshold: float,
    min_frames: int,
    frame_time: float,
) -> int:
    n_frames = frame_probs.shape[0]
    min_offset = min(n_frames - 1, onset_frame + min_frames)
    for frame_idx in range(min_offset, n_frames):
        if float(frame_probs[frame_idx, key]) < frame_threshold:
            return max(frame_idx, min_offset)
    return min(n_frames, onset_frame + int(round(2.0 / frame_time)))


def _candidate_features(
    onset_probs: np.ndarray,
    frame_probs: np.ndarray,
    velocity: np.ndarray,
    key: int,
    frame_idx: int,
    primary_events: Sequence[Dict],
    clusters: Sequence[Dict],
    sr: int,
    hop: int,
    lookback_frames: int,
) -> Tuple[List[float], Dict]:
    midi_note = key + MIDI_OFFSET
    time_sec = frame_idx * hop / float(sr)
    onset_peak = float(onset_probs[frame_idx, key])
    win_start = max(0, frame_idx - 2)
    win_end = min(onset_probs.shape[0], frame_idx + 5)
    frame_peak = float(np.max(frame_probs[win_start:win_end, key]))
    velocity_peak = float(np.max(velocity[win_start:win_end, key]))
    velocity_int = int(np.clip(round(velocity_peak * 127), 1, 127))
    prev_start = max(0, frame_idx - lookback_frames)
    prev = onset_probs[prev_start:frame_idx, key]
    prev_level = float(np.median(prev)) if prev.size else 0.0
    local_delta = onset_peak - prev_level
    anchor, anchor_dt = _nearest_anchor(clusters, time_sec)
    anchor_size = int(anchor["count"]) if anchor else 0
    anchor_pitches = [int(pitch) for pitch in anchor["pitches"]] if anchor else []
    if anchor_pitches:
        pitch_distance = min(abs(midi_note - pitch) for pitch in anchor_pitches)
        harmonic = (
            1.0
            if any(
                abs(midi_note - pitch) % 12 in HARMONIC_INTERVAL_CLASSES
                for pitch in anchor_pitches
            )
            else 0.0
        )
    else:
        pitch_distance = 88
        harmonic = 0.0
    same_pitch_recent = 0.0
    for event in primary_events:
        if int(event["midi_note"]) != midi_note:
            continue
        if 0.0 < time_sec - float(event["onset_time"]) <= 0.35:
            same_pitch_recent = 1.0
            break
    active_before = (
        float(np.max(frame_probs[max(0, frame_idx - lookback_frames):frame_idx, key]))
        if frame_idx > 0
        else 0.0
    )
    features = [
        onset_peak,
        frame_peak,
        velocity_peak,
        float(velocity_int),
        local_delta,
        prev_level,
        float(anchor_dt if math.isfinite(anchor_dt) else 9.9),
        float(anchor_size),
        float(pitch_distance),
        same_pitch_recent,
        harmonic,
        active_before,
        1.0 if 52 <= midi_note <= 76 else 0.0,
        (midi_note - MIDI_OFFSET) / 87.0,
    ]
    meta = {
        "midi_note": midi_note,
        "time_sec": time_sec,
        "anchor_time": float(anchor["time"]) if anchor else time_sec,
        "anchor_dt": float(anchor_dt if math.isfinite(anchor_dt) else 9.9),
        "onset_peak": onset_peak,
        "frame_peak": frame_peak,
        "velocity_int": velocity_int,
    }
    return features, meta


def rescue_lattice_events(
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    frame_probs: np.ndarray,
    velocity: np.ndarray,
    primary_events: Sequence[Dict],
    *,
    calibrator: LatticeCalibrator,
    primary_onset_threshold: float,
    sr: int = SAMPLE_RATE,
    hop: int = HOP_LENGTH,
) -> List[Dict]:
    """Return new calibrated inner-voice events to append to ``primary_events``.

    Candidate generation, scoring and anchor-snapping mirror the offline
    ``train_lattice_candidate_calibrator`` so the calibrated threshold transfers.
    ``primary_onset_threshold`` should match the live primary decode threshold so
    candidates are exactly the sub-threshold peaks the primary pass dropped.
    """
    args = calibrator.candidate_args
    candidate_onset_threshold = float(args.get("candidate_onset_threshold", 0.25))
    candidate_frame_threshold = float(args.get("candidate_frame_threshold", 0.35))
    candidate_min_velocity = int(args.get("candidate_min_velocity", 8))
    frame_threshold = float(args.get("frame_threshold", 0.5))
    cluster_tolerance_sec = float(args.get("cluster_tolerance_sec", 0.04))
    duplicate_tolerance_sec = float(args.get("duplicate_tolerance_sec", 0.04))
    lookback_sec = float(args.get("lookback_sec", 0.08))
    min_note_duration = float(args.get("min_note_duration", 0.04))
    max_anchor_distance_sec = float(args.get("max_anchor_distance_sec", 0.06))
    max_additions_per_anchor = int(args.get("max_additions_per_anchor", 3))
    snap_to_anchor = bool(args.get("snap_to_anchor", True))

    frame_time = hop / float(sr)
    clusters = _cluster_primary_events(primary_events, cluster_tolerance_sec)
    if not clusters:
        return []
    lookback_frames = max(1, int(round(lookback_sec * sr / hop)))

    candidates: List[Dict] = []
    for key in range(min(onset_probs.shape[1], PIANO_KEYS)):
        for frame_idx in _peak_frames(onset_probs[:, key], candidate_onset_threshold):
            onset_peak = float(onset_probs[frame_idx, key])
            if onset_peak >= primary_onset_threshold:
                continue
            midi_note = key + MIDI_OFFSET
            time_sec = frame_idx * hop / float(sr)
            if _primary_match_exists(primary_events, midi_note, time_sec, duplicate_tolerance_sec):
                continue
            features, meta = _candidate_features(
                onset_probs, frame_probs, velocity, key, frame_idx,
                primary_events, clusters, sr, hop, lookback_frames,
            )
            if meta["frame_peak"] < candidate_frame_threshold:
                continue
            if meta["velocity_int"] < candidate_min_velocity:
                continue
            candidates.append({"features": features, "meta": meta})

    if not candidates:
        return []

    feature_matrix = np.asarray([c["features"] for c in candidates], dtype=np.float64)
    probabilities = calibrator.score(feature_matrix)

    min_frames = max(1, int(round(min_note_duration / frame_time)))
    existing = [dict(event) for event in primary_events]
    accepted: List[Dict] = []
    per_anchor_counts: Counter = Counter()
    for candidate, prob in sorted(
        zip(candidates, probabilities), key=lambda item: float(item[1]), reverse=True
    ):
        if float(prob) < calibrator.threshold:
            continue
        meta = candidate["meta"]
        if float(meta["anchor_dt"]) > max_anchor_distance_sec:
            continue
        anchor_key = round(float(meta["anchor_time"]), 3)
        if per_anchor_counts[anchor_key] >= max_additions_per_anchor:
            continue
        midi_note = int(meta["midi_note"])
        onset_time = float(meta["anchor_time"]) if snap_to_anchor else float(meta["time_sec"])
        if _primary_match_exists(existing + accepted, midi_note, onset_time, duplicate_tolerance_sec):
            continue
        key = midi_note - MIDI_OFFSET
        onset_frame = int(round(float(meta["time_sec"]) * sr / hop))
        offset_frame = _frame_drop_offset(
            frame_probs, key, onset_frame, frame_threshold, min_frames, frame_time
        )
        offset_time = max(onset_time + min_note_duration, offset_frame * frame_time)
        event = {
            "onset_time": onset_time,
            "offset_time": offset_time,
            "midi_note": midi_note,
            "velocity": int(meta["velocity_int"]),
            "onset_prob": float(meta["onset_peak"]),
            "offset_prob": float(offset_probs[min(offset_frame, offset_probs.shape[0] - 1), key]),
            "decode_source": "lattice_calibrated",
            "lattice_probability": float(prob),
        }
        accepted.append(event)
        per_anchor_counts[anchor_key] += 1

    return accepted
