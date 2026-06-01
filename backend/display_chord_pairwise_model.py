from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

MODEL_ENV_VAR = "LIVE_DISPLAY_CHORD_PAIRWISE_MODEL"
DEFAULT_MODEL_FILENAME = "display_chord_pairwise_model.json"

PAIR_FEATURE_NAMES: Tuple[str, ...] = (
    "time_delta_ms",
    "duration_delta_ms",
    "confidence_mean",
    "confidence_gap",
    "same_slot",
    "both_slot_known",
    "shared_pitch_count",
    "union_pitch_count",
    "jaccard_overlap",
    "subset_relation",
    "pitch_count_gap",
    "lowest_pitch_gap",
    "gap_after_left_offset_ms",
    "left_has_duration",
)

PITCH_FEATURE_NAMES: Tuple[str, ...] = (
    "component_size",
    "component_confidence_sum",
    "component_union_pitch_count",
    "pitch_support_count",
    "pitch_support_fraction",
    "pitch_confidence_sum",
    "pitch_confidence_fraction",
    "pitch_confidence_max",
    "pitch_confidence_mean",
    "pitch_time_span_ms",
    "pitch_mean_offset_ms",
    "pitch_slot_known_fraction",
    "pitch_slot_mode_fraction",
    "pitch_is_lowest_fraction",
)

MIDI_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return float(default)


def chord_time_seconds(chord: Dict) -> float:
    return _safe_float(chord.get("time_seconds", chord.get("onset_time", 0.0)))


def chord_duration_seconds(chord: Dict) -> float:
    duration = chord.get("duration_seconds")
    if duration is not None:
        return max(0.0, _safe_float(duration))
    onset = chord_time_seconds(chord)
    offset = chord.get("offset_seconds")
    if offset is None:
        return 0.0
    return max(0.0, _safe_float(offset) - onset)


def chord_slot(chord: Dict) -> int | None:
    raw_idx = chord.get("start_grid_idx")
    try:
        if raw_idx is not None:
            return int(raw_idx)
    except (TypeError, ValueError):
        pass

    raw_beat = chord.get("start_beat")
    raw_subdivision = chord.get("grid_subdivision")
    try:
        beat_value = float(raw_beat)
        subdivision = int(raw_subdivision)
    except (TypeError, ValueError):
        return None

    if subdivision <= 0:
        return None

    return int(round(beat_value * subdivision))


def chord_confidence(chord: Dict) -> float:
    return _safe_float(chord.get("confidence", 0.0))


def chord_pitch_tuple(chord: Dict) -> Tuple[int, ...]:
    pitches = []
    for midi_note in chord.get("midi_notes") or []:
        try:
            pitches.append(int(midi_note))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(pitches))


def pitch_set_f1(predicted_pitches: Iterable[int], ground_truth_pitches: Iterable[int]) -> float:
    pred = {int(pitch) for pitch in predicted_pitches}
    gt = {int(pitch) for pitch in ground_truth_pitches}
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    intersect = len(pred & gt)
    return (2.0 * intersect) / (len(pred) + len(gt)) if intersect > 0 else 0.0


def note_name_from_midi(midi_note: int) -> str:
    octave = (int(midi_note) // 12) - 1
    return f"{MIDI_NOTE_NAMES[int(midi_note) % 12]}{octave}"


def extract_pair_features(left: Dict, right: Dict) -> Dict[str, float]:
    left_pitches = set(chord_pitch_tuple(left))
    right_pitches = set(chord_pitch_tuple(right))
    shared = float(len(left_pitches & right_pitches))
    union = float(len(left_pitches | right_pitches))
    lowest_pitch_gap = 0.0
    if left_pitches and right_pitches:
        lowest_pitch_gap = abs(min(left_pitches) - min(right_pitches))

    left_slot = chord_slot(left)
    right_slot = chord_slot(right)
    left_conf = chord_confidence(left)
    right_conf = chord_confidence(right)

    left_duration = chord_duration_seconds(left)
    left_has_duration = 1.0 if left_duration > 0.0 else 0.0
    if left_has_duration:
        left_offset = chord_time_seconds(left) + left_duration
        right_onset = chord_time_seconds(right)
        gap_after_left_offset_ms = (right_onset - left_offset) * 1000.0
    else:
        gap_after_left_offset_ms = 0.0

    return {
        "time_delta_ms": abs(chord_time_seconds(left) - chord_time_seconds(right)) * 1000.0,
        "duration_delta_ms": abs(chord_duration_seconds(left) - chord_duration_seconds(right)) * 1000.0,
        "confidence_mean": (left_conf + right_conf) / 2.0,
        "confidence_gap": abs(left_conf - right_conf),
        "same_slot": 1.0 if (left_slot is not None and right_slot is not None and left_slot == right_slot) else 0.0,
        "both_slot_known": 1.0 if (left_slot is not None and right_slot is not None) else 0.0,
        "shared_pitch_count": shared,
        "union_pitch_count": union,
        "jaccard_overlap": (shared / union) if union > 0.0 else 0.0,
        "subset_relation": 1.0 if (left_pitches <= right_pitches or right_pitches <= left_pitches) else 0.0,
        "pitch_count_gap": abs(len(left_pitches) - len(right_pitches)),
        "lowest_pitch_gap": float(lowest_pitch_gap),
        "gap_after_left_offset_ms": float(gap_after_left_offset_ms),
        "left_has_duration": float(left_has_duration),
    }


def extract_pitch_vote_features(component: Sequence[Dict], pitch: int) -> Dict[str, float]:
    component_size = float(len(component))
    component_confidence_sum = sum(chord_confidence(chord) for chord in component)
    component_union = set().union(*(chord_pitch_tuple(chord) for chord in component)) if component else set()

    supporters = [chord for chord in component if int(pitch) in set(chord_pitch_tuple(chord))]
    supporter_count = float(len(supporters))
    supporter_confidences = [chord_confidence(chord) for chord in supporters]
    pitch_confidence_sum = sum(supporter_confidences)
    supporter_times = [chord_time_seconds(chord) for chord in supporters]
    component_anchor_time = min((chord_time_seconds(chord) for chord in component), default=0.0)
    supporter_offsets_ms = [(time_value - component_anchor_time) * 1000.0 for time_value in supporter_times]

    supporter_slots = [chord_slot(chord) for chord in supporters]
    known_slots = [slot for slot in supporter_slots if slot is not None]
    slot_mode_fraction = 0.0
    if known_slots:
        counts: Dict[int, int] = {}
        for slot in known_slots:
            counts[int(slot)] = counts.get(int(slot), 0) + 1
        slot_mode_fraction = max(counts.values()) / len(known_slots)

    lowest_hits = 0.0
    for chord in supporters:
        pitches = chord_pitch_tuple(chord)
        if pitches and int(pitch) == pitches[0]:
            lowest_hits += 1.0

    return {
        "component_size": component_size,
        "component_confidence_sum": component_confidence_sum,
        "component_union_pitch_count": float(len(component_union)),
        "pitch_support_count": supporter_count,
        "pitch_support_fraction": (supporter_count / component_size) if component_size > 0.0 else 0.0,
        "pitch_confidence_sum": pitch_confidence_sum,
        "pitch_confidence_fraction": (pitch_confidence_sum / component_confidence_sum) if component_confidence_sum > 0.0 else 0.0,
        "pitch_confidence_max": max(supporter_confidences) if supporter_confidences else 0.0,
        "pitch_confidence_mean": (pitch_confidence_sum / supporter_count) if supporter_count > 0.0 else 0.0,
        "pitch_time_span_ms": ((max(supporter_times) - min(supporter_times)) * 1000.0) if len(supporter_times) >= 2 else 0.0,
        "pitch_mean_offset_ms": (sum(supporter_offsets_ms) / supporter_count) if supporter_count > 0.0 else 0.0,
        "pitch_slot_known_fraction": (len(known_slots) / supporter_count) if supporter_count > 0.0 else 0.0,
        "pitch_slot_mode_fraction": slot_mode_fraction,
        "pitch_is_lowest_fraction": (lowest_hits / supporter_count) if supporter_count > 0.0 else 0.0,
    }


@dataclass(frozen=True)
class LinearProbabilityModel:
    feature_names: Tuple[str, ...]
    intercept: float
    coefficients: Tuple[float, ...]
    means: Tuple[float, ...]
    scales: Tuple[float, ...]
    threshold: float

    def predict_logit(self, features: Dict[str, float]) -> float:
        score = float(self.intercept)
        for index, name in enumerate(self.feature_names):
            value = float(features.get(name, 0.0) or 0.0)
            centered = value - self.means[index]
            scale = self.scales[index] if abs(self.scales[index]) > 1e-9 else 1.0
            score += self.coefficients[index] * (centered / scale)
        return score

    def predict_probability(self, features: Dict[str, float]) -> float:
        logit = self.predict_logit(features)
        if logit >= 0.0:
            exp_value = math.exp(-logit)
            return 1.0 / (1.0 + exp_value)
        exp_value = math.exp(logit)
        return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class PairwiseDisplayModel:
    pair_model: LinearProbabilityModel
    pitch_model: LinearProbabilityModel
    metadata: Dict[str, object]


def _load_linear_model(payload: Dict, default_feature_names: Tuple[str, ...]) -> LinearProbabilityModel | None:
    feature_names = tuple(payload.get("feature_names") or default_feature_names)
    if not feature_names:
        return None

    standardize = payload.get("standardize") or {}
    means = tuple(float(value) for value in (standardize.get("mean") or [0.0] * len(feature_names)))
    scales = tuple(
        float(value) if abs(float(value)) > 1e-9 else 1.0
        for value in (standardize.get("scale") or [1.0] * len(feature_names))
    )
    coefficients = tuple(float(value) for value in (payload.get("coefficients") or [0.0] * len(feature_names)))
    if len(coefficients) != len(feature_names):
        return None

    return LinearProbabilityModel(
        feature_names=feature_names,
        intercept=float(payload.get("intercept", 0.0) or 0.0),
        coefficients=coefficients,
        means=means if len(means) == len(feature_names) else tuple([0.0] * len(feature_names)),
        scales=scales if len(scales) == len(feature_names) else tuple([1.0] * len(feature_names)),
        threshold=float(payload.get("threshold", 0.5) or 0.5),
    )


def resolve_model_path(model_path: str | None = None) -> Path:
    raw_path = model_path or os.environ.get(MODEL_ENV_VAR)
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return Path(__file__).resolve().with_name(DEFAULT_MODEL_FILENAME)


@lru_cache(maxsize=4)
def load_pairwise_model(model_path: str | None = None) -> PairwiseDisplayModel | None:
    path = resolve_model_path(model_path)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    pair_model = _load_linear_model(payload.get("pair_model") or {}, PAIR_FEATURE_NAMES)
    pitch_model = _load_linear_model(payload.get("pitch_model") or {}, PITCH_FEATURE_NAMES)
    if pair_model is None or pitch_model is None:
        return None

    return PairwiseDisplayModel(
        pair_model=pair_model,
        pitch_model=pitch_model,
        metadata=dict(payload.get("training_summary") or {}),
    )


def clear_pairwise_model_cache() -> None:
    load_pairwise_model.cache_clear()


@lru_cache(maxsize=None)
def _partition_templates(n_items: int) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    if n_items <= 0:
        return (tuple(),)
    if n_items == 1:
        return ((((0,),)),)

    results = []
    seen = set()
    for partition in _partition_templates(n_items - 1):
        singleton = tuple(sorted(partition + ((n_items - 1,),), key=lambda block: block[0]))
        if singleton not in seen:
            seen.add(singleton)
            results.append(singleton)

        for block_index, block in enumerate(partition):
            merged_block = tuple(sorted(block + (n_items - 1,)))
            new_partition = list(partition)
            new_partition[block_index] = merged_block
            normalized = tuple(sorted(tuple(new_partition), key=lambda item: item[0]))
            if normalized not in seen:
                seen.add(normalized)
                results.append(normalized)

    return tuple(results)


def _pair_probability_matrix(chord_group: Sequence[Dict], model: LinearProbabilityModel) -> Tuple[Tuple[float, ...], ...]:
    n_items = len(chord_group)
    rows = [[1.0 for _ in range(n_items)] for _ in range(n_items)]
    for left_index in range(n_items):
        for right_index in range(left_index + 1, n_items):
            probability = model.predict_probability(extract_pair_features(chord_group[left_index], chord_group[right_index]))
            rows[left_index][right_index] = probability
            rows[right_index][left_index] = probability
    return tuple(tuple(row) for row in rows)


def _score_partition(partition: Tuple[Tuple[int, ...], ...], pair_probabilities: Tuple[Tuple[float, ...], ...]) -> float:
    cluster_by_index: Dict[int, int] = {}
    for cluster_index, block in enumerate(partition):
        for item_index in block:
            cluster_by_index[int(item_index)] = cluster_index

    score = 0.0
    n_items = len(pair_probabilities)
    for left_index in range(n_items):
        for right_index in range(left_index + 1, n_items):
            probability = min(1.0 - 1e-6, max(1e-6, pair_probabilities[left_index][right_index]))
            if cluster_by_index.get(left_index) == cluster_by_index.get(right_index):
                score += math.log(probability)
            else:
                score += math.log(1.0 - probability)
    return score


def _best_partition(chord_group: Sequence[Dict], model: LinearProbabilityModel) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[float, ...], ...]]:
    if len(chord_group) <= 1:
        partition = (tuple(range(len(chord_group))),) if chord_group else tuple()
        return (partition, _pair_probability_matrix(chord_group, model))

    pair_probabilities = _pair_probability_matrix(chord_group, model)
    best_partition = None
    best_score = None
    for partition in _partition_templates(len(chord_group)):
        score = _score_partition(partition, pair_probabilities)
        if best_score is None or score > best_score:
            best_score = score
            best_partition = partition

    return (best_partition or tuple(), pair_probabilities)


def compute_component_pitch_probabilities(
    component: Sequence[Dict],
    pitch_model: LinearProbabilityModel,
) -> Dict[int, float]:
    candidate_pitches = sorted({pitch for chord in component for pitch in chord_pitch_tuple(chord)})
    return {
        int(pitch): float(pitch_model.predict_probability(extract_pitch_vote_features(component, pitch)))
        for pitch in candidate_pitches
    }


def _build_component_base_event(component: Sequence[Dict]) -> Dict[str, object]:
    weights = [max(0.05, chord_confidence(chord)) for chord in component]
    weight_sum = sum(weights) or float(len(component))
    weighted_time = sum(chord_time_seconds(chord) * weight for chord, weight in zip(component, weights)) / weight_sum
    weighted_duration = sum(chord_duration_seconds(chord) * weight for chord, weight in zip(component, weights)) / weight_sum
    representative = max(
        component,
        key=lambda chord: (chord_confidence(chord), len(chord_pitch_tuple(chord))),
    )

    base_event = {
        key: value
        for key, value in dict(representative).items()
        if not str(key).startswith("_")
    }
    base_event["time_seconds"] = round(weighted_time, 6)
    base_event["duration_seconds"] = round(max(0.05, weighted_duration), 6)
    base_event["offset_seconds"] = round(base_event["time_seconds"] + base_event["duration_seconds"], 6)
    base_event["confidence"] = round(sum(chord_confidence(chord) for chord in component) / max(1, len(component)), 6)
    return base_event


def build_canonical_events_from_pitch_set(
    component: Sequence[Dict],
    selected_pitches: Sequence[int],
) -> Dict[str, Tuple[Dict, ...]]:
    normalized_pitches = tuple(sorted({int(pitch) for pitch in selected_pitches}))
    if not normalized_pitches:
        return {"notes": tuple(), "chords": tuple()}

    base_event = _build_component_base_event(component)

    if len(normalized_pitches) == 1:
        midi_note = int(normalized_pitches[0])
        note_event = {
            key: value
            for key, value in base_event.items()
            if key not in {"midi_notes", "note_names", "root", "octave", "inversion", "label"}
        }
        note_event["midi_note"] = midi_note
        note_event["note_name"] = note_name_from_midi(midi_note)
        return {"notes": (note_event,), "chords": tuple()}

    chord_event = dict(base_event)
    chord_event["midi_notes"] = list(normalized_pitches)
    chord_event["note_names"] = [note_name_from_midi(midi_note) for midi_note in normalized_pitches]
    lowest_pitch = int(normalized_pitches[0])
    chord_event["root"] = note_name_from_midi(lowest_pitch)
    chord_event["octave"] = (lowest_pitch // 12) - 1
    chord_event["inversion"] = "root"
    if set(normalized_pitches) != set(chord_pitch_tuple(max(component, key=lambda chord: (chord_confidence(chord), len(chord_pitch_tuple(chord)))))):
        chord_event.pop("label", None)
    return {"notes": tuple(), "chords": (chord_event,)}


def _canonicalize_component(component: Sequence[Dict], pitch_model: LinearProbabilityModel) -> Dict[str, Tuple[Dict, ...]]:
    if not component:
        return {"notes": tuple(), "chords": tuple()}

    pitch_probabilities = compute_component_pitch_probabilities(component, pitch_model)
    selected_pitches = tuple(
        sorted(int(pitch) for pitch, probability in pitch_probabilities.items() if float(probability) >= float(pitch_model.threshold))
    )
    return build_canonical_events_from_pitch_set(component, selected_pitches)


def canonicalize_pairwise_chord_group(
    chord_group: Sequence[Dict],
    model: PairwiseDisplayModel,
) -> Dict[str, Tuple[Dict, ...]]:
    if not chord_group:
        return {"notes": tuple(), "chords": tuple()}

    partition, _ = _best_partition(chord_group, model.pair_model)
    notes = []
    chords = []
    for block in partition:
        component = [dict(chord_group[index]) for index in block]
        canonical = _canonicalize_component(component, model.pitch_model)
        notes.extend(canonical.get("notes") or ())
        chords.extend(canonical.get("chords") or ())

    notes.sort(key=chord_time_seconds)
    chords.sort(key=chord_time_seconds)
    return {"notes": tuple(notes), "chords": tuple(chords)}