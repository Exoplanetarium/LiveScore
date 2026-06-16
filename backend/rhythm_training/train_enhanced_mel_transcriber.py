"""
Enhanced piano transcription model for LiveScore.

This is a clean successor to train_mel_baseline.py, not a replacement for the
current production checkpoint. It keeps the fast 229-bin log-mel frontend but
adds the model/metric pieces needed for a stronger base transcriber:

  - frequency-preserving ConvStack with pitch-local key readout
  - onset, offset, frame, and velocity heads
  - event refinement GRU for onset/offset logits
  - MIDI-derived offset tent labels
  - validation by decoded note-event F1, not only framewise F1

Recommended starting point:

    python train_enhanced_mel_transcriber.py --train --batch-size 8 \
      --d-model 384 --n-layers 10 --n-heads 8 --conv-channels 192 \
      --event-hidden 192 --pos-weight 4 --offset-weight 1.0 \
      --frame-weight 0.8 --sounding-frame-weight 0.3 \
      --pedal-weight 0.1 --velocity-weight 0.3 --nv-weight 0.1

Fine-tuning on the pedal/onset hard-case manifests:

        python train_enhanced_mel_transcriber.py --train --finetune \
            --train-segment-manifest mel_hard_case_manifest_train_pedal_onset_v2.json \
            --validation-segment-manifest mel_hard_case_manifest_validation_pedal_onset_v2.json \
            --epochs 20 --batch-size 8 --finetune-scope decoder

For a larger offline teacher, try d_model=512, n_layers=12, n_heads=8.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)
from train_mel_baseline import (FEATURES_DIR, HOP_LENGTH, INDEX_DIR,
                                MIDI_OFFSET, N_FFT, N_MELS, NOTE_VALUE_BEATS,
                                NOTE_VALUE_CLASSES, NOTE_VALUE_NAMES,
                                ONSET_TENT_SEC, PIANO_KEYS, SAMPLE_RATE,
                                SEGMENT_SECONDS, ConformerBlock,
                                MelFeatureExtractor, ResidualAdapter,
                                SinusoidalPositionalEncoding)

MODEL_PATH = Path(__file__).parent / "enhanced_mel_transcription.pt"
DEFAULT_FINETUNE_MODEL_PATH = Path(__file__).parent / "enhanced_mel_transcription_finetuned.pt"
OFFSET_TENT_SEC = 0.05
PEDAL_CC = 64
PEDAL_DOWN_THRESHOLD = 64
SCORE_GRID_BEATS = 0.25
SCORE_DURATION_POLICIES = (
    "head",
    "decoded_duration",
    "same_pitch_cap",
    "ioi_same_hand",
    "hybrid_cleanup",
    "lookup_ioi_head_sound",
)


def _time_to_frame(time_sec: float, frame_time: float) -> int:
    return int(round(time_sec / frame_time))


def _apply_tent(target: np.ndarray, center_time: float, key: int, tent_sec: float, frame_time: float) -> None:
    if center_time < 0:
        return
    center_f = _time_to_frame(center_time, frame_time)
    radius = max(1, int(math.ceil(tent_sec / frame_time)))
    for frame_idx in range(max(0, center_f - radius), min(target.shape[0], center_f + radius + 1)):
        frame_time_sec = frame_idx * frame_time
        value = max(0.0, 1.0 - abs(frame_time_sec - center_time) / max(tent_sec, 1e-6))
        target[frame_idx, key] = max(target[frame_idx, key], value)


def _extract_control_changes(midi) -> List:
    control_changes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            control_changes.extend(instrument.control_changes)
    return control_changes


def _make_pedal_curve(
    control_changes: Sequence,
    start_sec: float,
    n_frames: int,
    frame_time: float,
    cc_number: int = PEDAL_CC,
) -> np.ndarray:
    changes = []
    for cc in control_changes:
        if isinstance(cc, tuple):
            cc_time, cc_num, cc_value = cc
        else:
            cc_time, cc_num, cc_value = cc.time, cc.number, cc.value
        if int(cc_num) == cc_number:
            changes.append((float(cc_time), int(cc_value)))
    changes.sort(key=lambda item: item[0])
    pedal = np.zeros(n_frames, dtype=np.float32)
    if not changes:
        return pedal

    times = np.asarray([item[0] for item in changes], dtype=np.float64)
    values = np.asarray([item[1] for item in changes], dtype=np.float32) / 127.0
    frame_times = start_sec + np.arange(n_frames, dtype=np.float64) * frame_time
    indices = np.searchsorted(times, frame_times, side="right") - 1
    valid = indices >= 0
    pedal[valid] = values[indices[valid]]
    return pedal


def _make_pedal_intervals(
    control_changes: Sequence,
    piece_end: float,
    cc_number: int = PEDAL_CC,
    down_threshold: int = PEDAL_DOWN_THRESHOLD,
) -> List[Tuple[float, float]]:
    changes = []
    for cc in control_changes:
        if isinstance(cc, tuple):
            cc_time, cc_num, cc_value = cc
        else:
            cc_time, cc_num, cc_value = cc.time, cc.number, cc.value
        if int(cc_num) == cc_number:
            changes.append((float(cc_time), int(cc_value)))
    changes.sort(key=lambda item: item[0])
    intervals: List[Tuple[float, float]] = []
    down_start = None
    for time_sec, value in changes:
        is_down = value >= down_threshold
        if is_down and down_start is None:
            down_start = time_sec
        elif not is_down and down_start is not None:
            if time_sec > down_start:
                intervals.append((down_start, time_sec))
            down_start = None
    if down_start is not None and piece_end > down_start:
        intervals.append((down_start, piece_end))
    return intervals


def _pedal_extended_end(note_end: float, pedal_intervals: Sequence[Tuple[float, float]]) -> float:
    for start_sec, end_sec in pedal_intervals:
        if start_sec <= note_end < end_sec:
            return float(end_sec)
    return float(note_end)


def _duration_to_note_value_class(duration_sec: float, bpm: float) -> int:
    beat_duration = 60.0 / max(float(bpm), 1e-6)
    beats = max(0.0625, min(8.0, float(duration_sec) / beat_duration))
    log_nv = np.log2(np.asarray(NOTE_VALUE_BEATS, dtype=np.float64))
    return int(np.argmin(np.abs(log_nv - math.log2(beats))))


@lru_cache(maxsize=256)
def _load_midi_target_cache(midi_path: str) -> Dict:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(midi_path)
    tempo_times, tempos = midi.get_tempo_changes()
    bpm = float(tempos[0]) if len(tempos) > 0 else 120.0

    control_changes = []
    all_notes = []
    notes_by_key: Dict[int, List[Dict]] = defaultdict(list)
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            control_changes.append((float(cc.time), int(cc.number), int(cc.value)))
        for note in instrument.notes:
            key = int(note.pitch) - MIDI_OFFSET
            if key < 0 or key >= PIANO_KEYS:
                continue
            item = {
                "pitch": int(note.pitch),
                "key": key,
                "start": float(note.start),
                "end": float(note.end),
                "velocity": int(note.velocity),
                "next_same_pitch_start": float("inf"),
                "ioi": None,
            }
            all_notes.append(item)
            notes_by_key[key].append(item)

    all_notes.sort(key=lambda item: (item["start"], item["pitch"], item["end"]))
    hand_notes = {"bass": [], "treble": []}
    for note in all_notes:
        hand_notes["bass" if note["pitch"] < 60 else "treble"].append(note)
    for hand_note_list in hand_notes.values():
        for idx in range(len(hand_note_list) - 1):
            current = hand_note_list[idx]
            current["ioi"] = float(hand_note_list[idx + 1]["start"] - current["start"])

    for key_notes in notes_by_key.values():
        key_notes.sort(key=lambda item: (item["start"], item["end"]))
        for idx, note in enumerate(key_notes[:-1]):
            note["next_same_pitch_start"] = float(key_notes[idx + 1]["start"])

    piece_end = max(
        [float(midi.get_end_time())]
        + [note["end"] for note in all_notes]
        + [cc_time for cc_time, _, _ in control_changes]
        + [0.0]
    )
    return {
        "bpm": bpm,
        "piece_end": piece_end,
        "control_changes": tuple(control_changes),
        "notes": tuple(all_notes),
        "pedal_intervals": tuple(_make_pedal_intervals(control_changes, piece_end)),
    }


@lru_cache(maxsize=1024)
def _load_segment_targets(
    midi_path: str,
    start_sec: float,
    n_frames: int,
    sr: int,
    hop_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    frame_time = hop_length / sr
    end_sec = start_sec + SEGMENT_SECONDS
    offset = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
    sounding_frame = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
    gt_events: List[Dict] = []

    midi_cache = _load_midi_target_cache(midi_path)
    bpm = float(midi_cache["bpm"])
    piece_end = max(float(midi_cache["piece_end"]), end_sec)
    control_changes = midi_cache["control_changes"]
    pedal = _make_pedal_curve(control_changes, start_sec, n_frames, frame_time)
    pedal_intervals = midi_cache["pedal_intervals"]

    for note in midi_cache["notes"]:
        onset_rel = float(note["start"]) - start_sec
        offset_rel = float(note["end"]) - start_sec
        key = int(note["key"])
        sounding_end = min(
            _pedal_extended_end(float(note["end"]), pedal_intervals),
            float(note["next_same_pitch_start"]),
        )

        if sounding_end >= start_sec and float(note["start"]) <= end_sec:
            start_f = max(0, int(math.floor((float(note["start"]) - start_sec) / frame_time)))
            end_f = min(n_frames, int(math.ceil((sounding_end - start_sec) / frame_time)))
            if end_f > start_f:
                sounding_frame[start_f:end_f, key] = 1.0

        if float(note["end"]) < start_sec or float(note["start"]) > end_sec:
            continue

        _apply_tent(offset, offset_rel, key, OFFSET_TENT_SEC, frame_time)
        if 0.0 <= onset_rel < SEGMENT_SECONDS:
            ioi = note["ioi"]
            if ioi is None or ioi < 0.03:
                ioi = float(sounding_end - float(note["start"]))
            note_value_class = _duration_to_note_value_class(float(ioi), bpm)
            gt_events.append({
                "onset_time": float(onset_rel),
                "offset_time": float(max(offset_rel, onset_rel + frame_time)),
                "sounding_offset_time": float(max(sounding_end - start_sec, onset_rel + frame_time)),
                "midi_note": int(note["pitch"]),
                "velocity": int(note["velocity"]),
                "note_value_class": note_value_class,
                "note_value_name": NOTE_VALUE_NAMES[note_value_class],
            })

    offset_t = torch.from_numpy(offset)
    pedal_t = torch.from_numpy(pedal)
    sounding_frame_t = torch.from_numpy(sounding_frame)
    return offset_t, pedal_t, sounding_frame_t, gt_events


class EnhancedPrecomputedMelDataset(Dataset):
    """Precomputed mel features plus MIDI-derived offset/event targets."""

    def __init__(
        self,
        split: str,
        augment: bool = False,
        segment_ids: Optional[Sequence[int]] = None,
        train_window_sec: float = 0.0,
        emit_window_sec: float = 0.0,
        include_teacher_features: bool = False,
    ):
        self.split = split
        self.split_dir = FEATURES_DIR / split
        self.augment = augment
        self.include_teacher_features = bool(include_teacher_features)
        self.train_window_frames = (
            int(round(float(train_window_sec) * SAMPLE_RATE / HOP_LENGTH))
            if train_window_sec and train_window_sec > 0 else 0
        )
        self.emit_window_frames = (
            int(round(float(emit_window_sec) * SAMPLE_RATE / HOP_LENGTH))
            if emit_window_sec and emit_window_sec > 0 else self.train_window_frames
        )
        if self.train_window_frames > 0:
            if self.emit_window_frames <= 0:
                raise ValueError("--emit-window-sec must be positive when --train-window-sec is set")
            if self.emit_window_frames > self.train_window_frames:
                raise ValueError("--emit-window-sec cannot exceed --train-window-sec")

        if not self.split_dir.exists():
            raise RuntimeError(
                f"Precomputed mel features not found at {self.split_dir}. "
                "Run train_mel_baseline.py --precompute first."
            )

        index_path = INDEX_DIR / f"{split}_index.json"
        with index_path.open("r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.segments = self.index["segments"]
        self.pieces = self.index["pieces"]

        file_lookup = {
            int(path.stem.split("_")[-1]): path
            for path in self.split_dir.glob("seg_*.pt")
        }
        if segment_ids is None:
            self.segment_ids = sorted(file_lookup)
        else:
            requested = sorted(set(int(x) for x in segment_ids))
            missing = [x for x in requested if x not in file_lookup]
            if missing:
                raise ValueError(f"Missing precomputed feature files for {split}: {missing[:10]}")
            self.segment_ids = requested

        self.files = [file_lookup[segment_id] for segment_id in self.segment_ids]
        aug = " with augmentation" if augment else ""
        window = ""
        if self.train_window_frames > 0:
            window_sec = self.train_window_frames * HOP_LENGTH / SAMPLE_RATE
            emit_sec = self.emit_window_frames * HOP_LENGTH / SAMPLE_RATE
            teacher = ", teacher full-context" if self.include_teacher_features else ""
            window = f" (live-window {window_sec:.3f}s, emit {emit_sec:.3f}s{teacher})"
        print(f"[EnhancedDataset] {split}: {len(self.files)} segments{aug}{window}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        segment_id = self.segment_ids[idx]
        data = torch.load(self.files[idx], weights_only=True)
        segment = self.segments[segment_id]
        piece = self.pieces[segment["piece_idx"]]
        start_sec = float(segment["start_sec"])
        midi_path = str(piece["midi"])

        features = data["features"].detach().clone().contiguous()
        onset = data["onset"].detach().clone().contiguous()
        frame = data["frame"].detach().clone().contiguous()
        velocity = data["velocity"].detach().clone().contiguous()
        note_value = data["note_value"].detach().clone().contiguous()
        bpm = data.get("bpm", torch.tensor(120.0, dtype=torch.float32))

        n_frames = min(features.size(0), onset.size(0), frame.size(0))
        features = features[:n_frames]
        onset = onset[:n_frames]
        frame = frame[:n_frames]
        velocity = velocity[:n_frames]
        note_value = note_value[:n_frames]

        offset, pedal, sounding_frame, gt_events = _load_segment_targets(
            midi_path,
            start_sec,
            n_frames,
            SAMPLE_RATE,
            HOP_LENGTH,
        )

        if self.augment:
            features, onset, offset, frame, sounding_frame, velocity, note_value = self._augment(
                features, onset, offset, frame, sounding_frame, velocity, note_value
            )

        item = {
            "features": features,
            "onset": onset,
            "offset": offset,
            "frame": frame,
            "sounding_frame": sounding_frame,
            "velocity": velocity,
            "note_value": note_value,
            "pedal": pedal,
            "bpm": bpm.detach().clone() if torch.is_tensor(bpm) else torch.tensor(float(bpm)),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
            "midi_path": midi_path,
            "start_sec": torch.tensor(start_sec, dtype=torch.float32),
            "gt_events": gt_events,
        }
        if self.include_teacher_features:
            item["teacher_features"] = features.detach().clone().contiguous()
        if self.train_window_frames > 0:
            item = self._crop_live_window(item)
        return item

    @staticmethod
    def _augment(
        features: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        frame: torch.Tensor,
        sounding_frame: torch.Tensor,
        velocity: torch.Tensor,
        note_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if np.random.random() < 0.5:
            t_mask_len = int(np.random.randint(1, 8))
            t_start = int(np.random.randint(0, max(1, features.size(0) - t_mask_len)))
            features[t_start:t_start + t_mask_len, :] = 0

        if np.random.random() < 0.5:
            f_mask_len = int(np.random.randint(5, 36))
            f_start = int(np.random.randint(0, max(1, features.size(1) - f_mask_len)))
            features[:, f_start:f_start + f_mask_len] = 0

        features = features * float(np.random.uniform(0.85, 1.15))

        if np.random.random() < 0.25:
            shift = int(np.random.choice([-2, -1, 1, 2]))
            mel_shift = shift * 3
            shifted_features = torch.zeros_like(features)
            if mel_shift > 0:
                shifted_features[:, mel_shift:] = features[:, :-mel_shift]
            else:
                shifted_features[:, :mel_shift] = features[:, -mel_shift:]
            features = shifted_features

            def shift_keys(value: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
                shifted = torch.full_like(value, fill_value)
                if shift > 0:
                    shifted[:, shift:] = value[:, :-shift]
                else:
                    shifted[:, :shift] = value[:, -shift:]
                return shifted

            onset = shift_keys(onset)
            offset = shift_keys(offset)
            frame = shift_keys(frame)
            sounding_frame = shift_keys(sounding_frame)
            velocity = shift_keys(velocity)
            note_value = shift_keys(note_value, fill_value=0)

        return features, onset, offset, frame, sounding_frame, velocity, note_value

    def _crop_live_window(self, item: Dict) -> Dict:
        features = item["features"]
        total_frames = int(features.size(0))
        window_frames = self.train_window_frames
        crop_frames = min(window_frames, total_frames)
        if total_frames > crop_frames:
            if self.split == "train":
                start = int(np.random.randint(0, total_frames - crop_frames + 1))
            else:
                start = int((total_frames - crop_frames) // 2)
        else:
            start = 0
        end = start + crop_frames

        cropped = dict(item)
        tensor_keys = [
            "features", "onset", "offset", "frame", "sounding_frame",
            "velocity", "note_value", "pedal",
        ]
        for key in tensor_keys:
            value = cropped.get(key)
            if torch.is_tensor(value) and value.dim() >= 1:
                cropped[key] = value[start:end].clone()
        for key in tensor_keys:
            value = cropped.get(key)
            if torch.is_tensor(value) and value.dim() >= 1:
                cropped[key] = self._pad_time_dim(value, window_frames)

        loss_mask = torch.zeros(window_frames, dtype=torch.float32)
        emit_frames = min(self.emit_window_frames, crop_frames)
        if emit_frames > 0:
            emit_start = max(crop_frames - emit_frames, 0)
            loss_mask[emit_start:crop_frames] = 1.0
        cropped["loss_mask"] = loss_mask
        cropped["crop_start_frame"] = torch.tensor(start, dtype=torch.long)

        frame_time = HOP_LENGTH / SAMPLE_RATE
        crop_start_sec = start * frame_time
        crop_end_sec = end * frame_time
        adjusted_events = []
        for event in item.get("gt_events", []):
            onset_time = float(event["onset_time"])
            if crop_start_sec <= onset_time < crop_end_sec:
                adjusted = dict(event)
                adjusted["onset_time"] = onset_time - crop_start_sec
                for offset_key in ("offset_time", "sounding_offset_time"):
                    if offset_key in adjusted:
                        adjusted[offset_key] = max(
                            adjusted["onset_time"] + frame_time,
                            float(adjusted[offset_key]) - crop_start_sec,
                        )
                adjusted_events.append(adjusted)
        cropped["gt_events"] = adjusted_events
        return cropped

    @staticmethod
    def _pad_time_dim(value: torch.Tensor, target_frames: int) -> torch.Tensor:
        current_frames = int(value.size(0))
        if current_frames == target_frames:
            return value
        if current_frames > target_frames:
            return value[:target_frames].clone()
        padded_shape = (target_frames, *value.shape[1:])
        padded = torch.zeros(padded_shape, dtype=value.dtype, device=value.device)
        if current_frames > 0:
            padded[:current_frames] = value
        return padded


class SourceTaggedDataset(Dataset):
    """Wrap a dataset and mark whether samples are general replay or hard cases."""

    def __init__(self, dataset: Dataset, source_id: int):
        self.dataset = dataset
        self.source_id = int(source_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = dict(self.dataset[idx])
        item["sample_source"] = torch.tensor(self.source_id, dtype=torch.long)
        return item


class FrequencyConvStack(nn.Module):
    """2D conv frontend that preserves a frequency axis for pitch-local readout."""

    def __init__(self, n_mels: int = N_MELS, channels: int = 192, dropout: float = 0.1):
        super().__init__()
        mid = max(64, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid // 2),
            nn.GELU(),
            nn.Conv2d(mid // 2, mid // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid // 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid // 2, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.freq_bins = n_mels // 4
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))  # (B, C, T, F')
        return h.permute(0, 2, 3, 1).contiguous()  # (B, T, F', C)


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _make_key_frequency_bias(n_freq: int, n_mels: int = N_MELS, sigma_bins: float = 2.5) -> torch.Tensor:
    midi = np.arange(MIDI_OFFSET, MIDI_OFFSET + PIANO_KEYS)
    freqs = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    mel_min = _hz_to_mel(np.array([30.0]))[0]
    mel_max = _hz_to_mel(np.array([SAMPLE_RATE / 2.0]))[0]
    key_pos = ( _hz_to_mel(freqs) - mel_min) / max(mel_max - mel_min, 1e-6)
    key_pos = np.clip(key_pos, 0.0, 1.0) * (n_freq - 1)
    bins = np.arange(n_freq, dtype=np.float32)[None, :]
    centers = key_pos[:, None].astype(np.float32)
    bias = -0.5 * ((bins - centers) / sigma_bins) ** 2
    return torch.from_numpy(bias.astype(np.float32))


class PitchLocalReadout(nn.Module):
    """Learned key queries over the preserved frequency axis, with a pitch prior."""

    def __init__(self, freq_bins: int, channels: int, key_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(PIANO_KEYS, channels) * (channels ** -0.5))
        self.register_buffer("freq_bias", _make_key_frequency_bias(freq_bins), persistent=False)
        self.freq_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Sequential(
            nn.Linear(channels, key_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(key_dim),
        )

    def forward(self, freq_map: torch.Tensor) -> torch.Tensor:
        # freq_map: (B, T, F, C)
        logits = torch.einsum("btfc,kc->btkf", freq_map, self.query)
        logits = logits + self.freq_bias_scale * self.freq_bias.unsqueeze(0).unsqueeze(0)
        weights = torch.softmax(logits, dim=-1)
        local = torch.einsum("btkf,btfc->btkc", weights, freq_map)
        return self.proj(local)


class CrossKeyAttentionBlock(nn.Module):
    """Residual self-attention across piano keys for each frame.

    The residual projections are zero-initialized so enabling this block on an
    existing checkpoint starts as an exact no-op.
    """

    def __init__(self, key_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        n_heads = max(1, min(int(n_heads), key_dim))
        while key_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn_norm = nn.LayerNorm(key_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=key_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(key_dim),
            nn.Linear(key_dim, key_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(key_dim * 2, key_dim),
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, key_h: torch.Tensor) -> torch.Tensor:
        bsz, frames, n_keys, key_dim = key_h.shape
        x = key_h.reshape(bsz * frames, n_keys, key_dim)
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(x)
        return x.reshape(bsz, frames, n_keys, key_dim)


class EnhancedMelTranscriber(nn.Module):
    """Pitch-local Conformer transcriber with explicit offset prediction."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        conv_channels: int = 192,
        d_model: int = 384,
        n_layers: int = 10,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        event_hidden: int = 192,
        event_context: bool = False,
        event_residual: bool = False,
        cross_key_layers: int = 0,
        cross_key_heads: int = 4,
        adapter_bottleneck: int = 0,
        adapter_dropout: float = 0.0,
        n_note_value_classes: int = NOTE_VALUE_CLASSES,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.n_keys = PIANO_KEYS
        self.n_nv = n_note_value_classes
        self.event_residual = bool(event_residual)
        self.freq_stack = FrequencyConvStack(n_mels, conv_channels, dropout)
        self.global_proj = nn.Sequential(
            nn.Linear(self.freq_stack.freq_bins * conv_channels, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=4096, dropout=dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(n_layers)
        ])
        self.conformer_adapters = nn.ModuleList([
            ResidualAdapter(d_model, adapter_bottleneck, adapter_dropout)
            for _ in range(n_layers)
        ]) if adapter_bottleneck > 0 else None

        key_dim = d_model // 4
        self.key_dim = key_dim
        self.global_key_proj = nn.Sequential(
            nn.Linear(d_model, PIANO_KEYS * key_dim),
            nn.GELU(),
        )
        self.local_readout = PitchLocalReadout(self.freq_stack.freq_bins, conv_channels, key_dim, dropout)
        self.key_temporal = nn.Sequential(
            nn.Conv1d(key_dim, key_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(key_dim, key_dim, kernel_size=7, padding=6, dilation=2),
            nn.GELU(),
            nn.Conv1d(key_dim, key_dim, kernel_size=7, padding=12, dilation=4),
            nn.GELU(),
        )
        nn.init.zeros_(self.key_temporal[-2].weight)
        nn.init.zeros_(self.key_temporal[-2].bias)
        self.cross_key_blocks = nn.ModuleList([
            CrossKeyAttentionBlock(key_dim, cross_key_heads, dropout)
            for _ in range(max(0, int(cross_key_layers)))
        ])

        self.onset_head_raw = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, 1)
        )
        self.offset_head_raw = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, 1)
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, 1), nn.Sigmoid()
        )

        refine_features = PIANO_KEYS * 4
        self.event_refine_gru = nn.GRU(
            input_size=refine_features,
            hidden_size=event_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.event_refine_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(event_hidden * 2, PIANO_KEYS * 2),
        )
        self.event_context_head = None
        if event_context:
            context_hidden = max(16, key_dim // 2)
            self.event_context_head = nn.Sequential(
                nn.Linear(key_dim, context_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_hidden, 2),
            )
            nn.init.zeros_(self.event_context_head[-1].weight)
            nn.init.zeros_(self.event_context_head[-1].bias)

        self.frame_head = nn.Sequential(
            nn.Linear(key_dim + 2, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, 1)
        )
        self.pedal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
        self.sounding_frame_head = nn.Sequential(
            nn.Linear(key_dim + 4, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, 1)
        )
        self.note_value_head = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, n_note_value_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, frames, _ = x.shape
        freq_map = self.freq_stack(x)
        global_h = self.global_proj(freq_map.reshape(bsz, frames, -1))
        global_h = self.pos_enc(global_h)
        for idx, block in enumerate(self.conformer_blocks):
            global_h = block(global_h)
            if self.conformer_adapters is not None:
                global_h = self.conformer_adapters[idx](global_h)

        key_global = self.global_key_proj(global_h).reshape(bsz, frames, PIANO_KEYS, self.key_dim)
        key_local = self.local_readout(freq_map)
        key_h = key_global + key_local

        h_t = key_h.permute(0, 2, 3, 1).reshape(bsz * PIANO_KEYS, self.key_dim, frames)
        h_t = self.key_temporal(h_t)
        key_h = key_h + h_t.reshape(bsz, PIANO_KEYS, self.key_dim, frames).permute(0, 3, 1, 2)
        for block in self.cross_key_blocks:
            key_h = block(key_h)

        raw_onset_logits = self.onset_head_raw(key_h).squeeze(-1)
        raw_offset_logits = self.offset_head_raw(key_h).squeeze(-1)
        velocity = self.velocity_head(key_h).squeeze(-1)

        raw_onset = torch.sigmoid(raw_onset_logits.float())
        raw_offset = torch.sigmoid(raw_offset_logits.float())
        vel = velocity.float().detach()
        refine_input = torch.cat([
            raw_onset,
            raw_offset,
            raw_onset.clamp_min(1e-12).sqrt() * vel,
            raw_offset.clamp_min(1e-12).sqrt() * vel,
        ], dim=-1)
        refine_out, _ = self.event_refine_gru(refine_input)
        refined = self.event_refine_fc(refine_out).reshape(bsz, frames, 2, PIANO_KEYS)
        if self.event_context_head is not None:
            context_delta = self.event_context_head(key_h).permute(0, 1, 3, 2)
            refined = refined + context_delta
        if self.event_residual:
            onset_logits = raw_onset_logits + refined[:, :, 0, :]
            offset_logits = raw_offset_logits + refined[:, :, 1, :]
        else:
            onset_logits = refined[:, :, 0, :]
            offset_logits = refined[:, :, 1, :]

        frame_in = torch.cat([
            key_h,
            onset_logits.unsqueeze(-1).detach(),
            offset_logits.unsqueeze(-1).detach(),
        ], dim=-1)
        frame_logits = self.frame_head(frame_in).squeeze(-1)
        pedal_logits = self.pedal_head(global_h).squeeze(-1)
        pedal_context = pedal_logits.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, PIANO_KEYS, -1).detach()
        sounding_frame_in = torch.cat([
            key_h,
            onset_logits.unsqueeze(-1).detach(),
            offset_logits.unsqueeze(-1).detach(),
            frame_logits.unsqueeze(-1).detach(),
            pedal_context,
        ], dim=-1)
        sounding_frame_logits = self.sounding_frame_head(sounding_frame_in).squeeze(-1)
        note_value_logits = self.note_value_head(key_h)

        return {
            "onset_logits": onset_logits,
            "raw_onset_logits": raw_onset_logits,
            "offset_logits": offset_logits,
            "raw_offset_logits": raw_offset_logits,
            "frame_logits": frame_logits,
            "sounding_frame_logits": sounding_frame_logits,
            "pedal_logits": pedal_logits,
            "velocity": velocity,
            "note_value_logits": note_value_logits,
        }


class EnhancedTranscriptionLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 4.0,
        onset_weight: float = 1.0,
        raw_onset_weight: float = 0.25,
        offset_weight: float = 1.0,
        raw_offset_weight: float = 0.25,
        frame_weight: float = 0.8,
        sounding_frame_weight: float = 0.3,
        pedal_weight: float = 0.1,
        velocity_weight: float = 0.3,
        nv_weight: float = 0.1,
        focal_gamma: float = 1.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.onset_weight = onset_weight
        self.raw_onset_weight = raw_onset_weight
        self.offset_weight = offset_weight
        self.raw_offset_weight = raw_offset_weight
        self.frame_weight = frame_weight
        self.sounding_frame_weight = sounding_frame_weight
        self.pedal_weight = pedal_weight
        self.velocity_weight = velocity_weight
        self.nv_weight = nv_weight
        self.focal_gamma = focal_gamma

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return value.mean()
        mask_value = mask.to(device=value.device, dtype=value.dtype)
        if mask_value.shape != value.shape:
            mask_value = mask_value.expand_as(value)
        weighted = value * mask_value
        return weighted.sum() / mask_value.sum().clamp_min(1.0)

    def _event_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        sample_weight = 1.0 + (pos_weight - 1.0) * target
        return self._masked_mean(bce * sample_weight, mask)

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        onset_gt = batch["onset"]
        offset_gt = batch["offset"]
        frame_gt = batch["frame"]
        sounding_frame_gt = batch.get("sounding_frame", frame_gt)
        pedal_gt = batch["pedal"]
        vel_gt = batch["velocity"]
        nv_gt = batch["note_value"]

        time_mask = batch.get("loss_mask")
        event_mask = time_mask.unsqueeze(-1) if torch.is_tensor(time_mask) else None

        onset_loss = self._event_loss(out["onset_logits"], onset_gt, self.pos_weight, event_mask)
        raw_onset_loss = self._event_loss(out["raw_onset_logits"], onset_gt, self.pos_weight, event_mask)
        offset_loss = self._event_loss(out["offset_logits"], offset_gt, self.pos_weight, event_mask)
        raw_offset_loss = self._event_loss(out["raw_offset_logits"], offset_gt, self.pos_weight, event_mask)

        frame_prob = torch.sigmoid(out["frame_logits"])
        p_t = torch.where(frame_gt > 0.5, frame_prob, 1.0 - frame_prob)
        focal = (1.0 - p_t.detach()).pow(self.focal_gamma)
        frame_sample_weight = torch.where(
            frame_gt > 0.5,
            torch.full_like(frame_gt, self.pos_weight),
            torch.ones_like(frame_gt),
        )
        frame_bce = F.binary_cross_entropy_with_logits(out["frame_logits"], frame_gt, reduction="none")
        frame_loss = self._masked_mean(focal * frame_bce * frame_sample_weight, event_mask)

        sounding_frame_prob = torch.sigmoid(out["sounding_frame_logits"])
        sounding_p_t = torch.where(sounding_frame_gt > 0.5, sounding_frame_prob, 1.0 - sounding_frame_prob)
        sounding_focal = (1.0 - sounding_p_t.detach()).pow(self.focal_gamma)
        sounding_frame_sample_weight = torch.where(
            sounding_frame_gt > 0.5,
            torch.full_like(sounding_frame_gt, self.pos_weight),
            torch.ones_like(sounding_frame_gt),
        )
        sounding_frame_bce = F.binary_cross_entropy_with_logits(
            out["sounding_frame_logits"],
            sounding_frame_gt,
            reduction="none",
        )
        sounding_frame_loss = self._masked_mean(
            sounding_focal * sounding_frame_bce * sounding_frame_sample_weight,
            event_mask,
        )

        pedal_bce = F.binary_cross_entropy_with_logits(out["pedal_logits"], pedal_gt, reduction="none")
        pedal_sample_weight = 1.0 + (self.pos_weight - 1.0) * (pedal_gt >= (PEDAL_DOWN_THRESHOLD / 127.0)).float()
        pedal_loss = self._masked_mean(pedal_bce * pedal_sample_weight, time_mask)

        active = frame_gt > 0.5
        if event_mask is not None:
            active = active & (event_mask.to(device=frame_gt.device) > 0.5)
        velocity_loss = (
            F.mse_loss(out["velocity"][active], vel_gt[active])
            if active.any()
            else torch.tensor(0.0, device=frame_gt.device)
        )

        onset_mask = (onset_gt > 0.5) & (frame_gt > 0.5)
        if event_mask is not None:
            onset_mask = onset_mask & (event_mask.to(device=onset_gt.device) > 0.5)
        if onset_mask.any() and self.nv_weight > 0:
            nv_loss = F.cross_entropy(out["note_value_logits"][onset_mask], nv_gt[onset_mask])
        else:
            nv_loss = torch.tensor(0.0, device=frame_gt.device)

        total = (
            self.onset_weight * onset_loss
            + self.raw_onset_weight * raw_onset_loss
            + self.offset_weight * offset_loss
            + self.raw_offset_weight * raw_offset_loss
            + self.frame_weight * frame_loss
            + self.sounding_frame_weight * sounding_frame_loss
            + self.pedal_weight * pedal_loss
            + self.velocity_weight * velocity_loss
            + self.nv_weight * nv_loss
        )
        return {
            "total": total,
            "onset": onset_loss,
            "raw_onset": raw_onset_loss,
            "offset": offset_loss,
            "raw_offset": raw_offset_loss,
            "frame": frame_loss,
            "sounding_frame": sounding_frame_loss,
            "pedal": pedal_loss,
            "velocity": velocity_loss,
            "note_value": nv_loss,
        }


def _peak_frames(probs: np.ndarray, threshold: float) -> np.ndarray:
    if probs.size == 0:
        return np.zeros(0, dtype=np.int64)
    peaks = []
    for idx in range(probs.shape[0]):
        left = probs[idx - 1] if idx > 0 else -np.inf
        right = probs[idx + 1] if idx + 1 < probs.shape[0] else -np.inf
        if probs[idx] > threshold and probs[idx] >= left and probs[idx] >= right:
            peaks.append(idx)
    return np.asarray(peaks, dtype=np.int64)


def _soft_polyphony_rescue_frames(
    onset_key: np.ndarray,
    frame_key: np.ndarray,
    velocity_key: np.ndarray,
    primary_frames: np.ndarray,
    onset_threshold: float,
    rescue_onset_threshold: float,
    rescue_frame_threshold: float,
    rescue_min_velocity: int,
    rescue_min_delta: float,
    rescue_lookback_frames: int,
    rescue_duplicate_frames: int,
) -> List[int]:
    """Recover softer per-key onset peaks that still have local frame evidence."""
    if rescue_onset_threshold >= onset_threshold:
        return []

    primary = [int(frame) for frame in primary_frames.tolist()]
    candidates = _peak_frames(onset_key, rescue_onset_threshold)
    rescued: List[int] = []
    n_frames = int(onset_key.shape[0])
    for frame in candidates.tolist():
        frame = int(frame)
        peak = float(onset_key[frame])
        if peak >= onset_threshold:
            continue
        if any(abs(frame - existing) <= rescue_duplicate_frames for existing in primary):
            continue
        if any(abs(frame - existing) <= rescue_duplicate_frames for existing in rescued):
            continue

        pre_start = max(0, frame - rescue_lookback_frames)
        pre = onset_key[pre_start:frame]
        pre_level = float(np.median(pre)) if pre.size else 0.0
        onset_delta = peak - pre_level
        if onset_delta < rescue_min_delta and peak < onset_threshold * 0.85:
            continue

        evidence_end = min(n_frames, frame + max(2, rescue_duplicate_frames + 1))
        frame_evidence = float(np.max(frame_key[frame:evidence_end])) if evidence_end > frame else float(frame_key[frame])
        velocity_evidence = float(np.max(velocity_key[frame:evidence_end])) if evidence_end > frame else float(velocity_key[frame])
        velocity_int = int(np.clip(round(velocity_evidence * 127), 1, 127))
        if frame_evidence < rescue_frame_threshold or velocity_int < rescue_min_velocity:
            continue

        rescued.append(frame)
    return rescued


def decode_enhanced_note_events(
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    frame_probs: np.ndarray,
    velocity: np.ndarray,
    note_value_probs: Optional[np.ndarray] = None,
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.35,
    frame_threshold: float = 0.5,
    min_note_duration: float = 0.04,
    min_velocity: int = 8,
    duplicate_window_sec: float = 0.04,
    merge_gap_sec: float = 0.0,
    note_value_pool_sec: float = ONSET_TENT_SEC,
    soft_polyphony_rescue: bool = False,
    soft_polyphony_onset_threshold: float = 0.45,
    soft_polyphony_frame_threshold: float = 0.35,
    soft_polyphony_min_velocity: int = 4,
    soft_polyphony_min_delta: float = 0.05,
    soft_polyphony_lookback_sec: float = 0.08,
    lattice_rescue: bool = False,
    lattice_model_path: Optional[str] = None,
    sr: int = SAMPLE_RATE,
    hop: int = HOP_LENGTH,
) -> List[Dict]:
    frame_time = hop / sr
    min_frames = max(1, int(round(min_note_duration / frame_time)))
    duplicate_frames = max(1, int(round(duplicate_window_sec / frame_time)))
    rescue_lookback_frames = max(1, int(round(soft_polyphony_lookback_sec / frame_time)))
    n_frames = onset_probs.shape[0]
    events: List[Dict] = []

    for key in range(PIANO_KEYS):
        primary_frames = _peak_frames(onset_probs[:, key], onset_threshold)
        rescued_frames: List[int] = []
        if soft_polyphony_rescue:
            rescued_frames = _soft_polyphony_rescue_frames(
                onset_probs[:, key],
                frame_probs[:, key],
                velocity[:, key],
                primary_frames,
                onset_threshold,
                soft_polyphony_onset_threshold,
                soft_polyphony_frame_threshold,
                soft_polyphony_min_velocity,
                soft_polyphony_min_delta,
                rescue_lookback_frames,
                duplicate_frames,
            )
        rescued_set = set(rescued_frames)
        onset_frames = np.asarray(
            sorted(set(primary_frames.tolist()).union(rescued_set)),
            dtype=np.int64,
        )
        offset_frames = _peak_frames(offset_probs[:, key], offset_threshold)
        for onset_f in onset_frames:
            is_rescued = int(onset_f) in rescued_set
            min_offset_f = min(n_frames - 1, onset_f + min_frames)
            later_offsets = offset_frames[offset_frames >= min_offset_f]
            frame_drop = None
            for frame_idx in range(min_offset_f, n_frames):
                if frame_probs[frame_idx, key] < frame_threshold:
                    frame_drop = frame_idx
                    break
            candidates = []
            if later_offsets.size:
                candidates.append(int(later_offsets[0]))
            if frame_drop is not None:
                candidates.append(int(frame_drop))
            offset_f = min(candidates) if candidates else min(n_frames, onset_f + int(round(2.0 / frame_time)))
            offset_f = max(offset_f, min_offset_f)

            vel_avg = float(velocity[onset_f:offset_f, key].mean()) if offset_f > onset_f else float(velocity[onset_f, key])
            vel_int = int(np.clip(round(vel_avg * 127), 1, 127))
            event_min_velocity = soft_polyphony_min_velocity if is_rescued else min_velocity
            if vel_int < event_min_velocity:
                continue

            event = {
                "onset_time": float(onset_f * frame_time),
                "offset_time": float(offset_f * frame_time),
                "midi_note": int(key + MIDI_OFFSET),
                "velocity": vel_int,
                "onset_prob": float(onset_probs[onset_f, key]),
                "offset_prob": float(offset_probs[min(offset_f, n_frames - 1), key]),
            }
            if is_rescued:
                pre_start = max(0, int(onset_f) - rescue_lookback_frames)
                pre = onset_probs[pre_start:int(onset_f), key]
                pre_level = float(np.median(pre)) if pre.size else 0.0
                event["decode_source"] = "soft_polyphony_rescue"
                event["rescue_onset_delta"] = float(onset_probs[onset_f, key] - pre_level)
            else:
                event["decode_source"] = "primary_onset"
            if note_value_probs is not None:
                nv_radius = max(1, int(math.ceil(note_value_pool_sec / frame_time)))
                pool_start = onset_f
                pool_end = min(n_frames, onset_f + nv_radius + 1)
                pooled = note_value_probs[pool_start:pool_end, key, :].mean(axis=0)
                nv_class = int(np.argmax(pooled))
                event["note_value_class"] = nv_class
                event["note_value_name"] = NOTE_VALUE_NAMES[nv_class]
                event["note_value_confidence"] = float(pooled[nv_class])
            events.append(event)

    events.sort(key=lambda item: (item["onset_time"], item["midi_note"]))
    filtered: List[Dict] = []
    for event in events:
        duplicate = False
        for previous in filtered[-20:]:
            if event["midi_note"] != previous["midi_note"]:
                continue

            onset_delta = abs(event["onset_time"] - previous["onset_time"])
            gap = float(event["onset_time"]) - float(previous["offset_time"])
            if merge_gap_sec > 0.0 and 0.0 <= gap <= merge_gap_sec:
                previous["offset_time"] = max(float(previous["offset_time"]), float(event["offset_time"]))
                previous["velocity"] = max(int(previous["velocity"]), int(event["velocity"]))
                previous["onset_prob"] = max(float(previous["onset_prob"]), float(event["onset_prob"]))
                previous["offset_prob"] = max(float(previous["offset_prob"]), float(event["offset_prob"]))
                duplicate = True
                break

            if onset_delta < duplicate_window_sec:
                duplicate = True
                if event["onset_prob"] > previous["onset_prob"]:
                    event["offset_time"] = max(float(event["offset_time"]), float(previous["offset_time"]))
                    filtered.remove(previous)
                    filtered.append(event)
                break
        if not duplicate:
            filtered.append(event)

    if lattice_rescue:
        try:
            from lattice_candidate_decoder import (load_calibrator,
                                                   rescue_lattice_events)
        except ImportError:
            from backend.lattice_candidate_decoder import (  # type: ignore
                load_calibrator, rescue_lattice_events)
        calibrator = load_calibrator(lattice_model_path)
        if calibrator is not None:
            lattice_events = rescue_lattice_events(
                onset_probs,
                offset_probs,
                frame_probs,
                velocity,
                filtered,
                calibrator=calibrator,
                primary_onset_threshold=onset_threshold,
                sr=sr,
                hop=hop,
            )
            for event in lattice_events:
                if note_value_probs is not None:
                    nv_radius = max(1, int(math.ceil(note_value_pool_sec / frame_time)))
                    onset_f = int(round(float(event["onset_time"]) / frame_time))
                    onset_f = max(0, min(n_frames - 1, onset_f))
                    key = int(event["midi_note"]) - MIDI_OFFSET
                    pool_end = min(n_frames, onset_f + nv_radius + 1)
                    pooled = note_value_probs[onset_f:pool_end, key, :].mean(axis=0)
                    nv_class = int(np.argmax(pooled))
                    event["note_value_class"] = nv_class
                    event["note_value_name"] = NOTE_VALUE_NAMES[nv_class]
                    event["note_value_confidence"] = float(pooled[nv_class])
                filtered.append(event)
            filtered.sort(key=lambda item: (item["onset_time"], item["midi_note"]))

    return filtered


def match_note_events(pred: Sequence[Dict], gt: Sequence[Dict], onset_tol: float = 0.05) -> Dict[str, float]:
    used_gt = set()
    matched = 0
    for event in pred:
        best_idx = None
        best_error = None
        for idx, ref in enumerate(gt):
            if idx in used_gt or int(event["midi_note"]) != int(ref["midi_note"]):
                continue
            err = abs(float(event["onset_time"]) - float(ref["onset_time"]))
            if err <= onset_tol and (best_error is None or err < best_error):
                best_idx = idx
                best_error = err
        if best_idx is not None:
            used_gt.add(best_idx)
            matched += 1
    precision = matched / max(len(pred), 1)
    recall = matched / max(len(gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "predicted": len(pred),
        "ground_truth": len(gt),
    }


def _event_note_value_class(event: Dict, bpm: float) -> int:
    if "note_value_class" in event:
        return int(event["note_value_class"])
    duration = float(event.get("sounding_offset_time", event.get("offset_time", 0.0))) - float(event.get("onset_time", 0.0))
    return _duration_to_note_value_class(max(duration, 1e-6), bpm)


def _score_event_slot(event: Dict, bpm: float, grid_beats: float) -> int:
    beat_duration = 60.0 / max(float(bpm), 1e-6)
    onset_beats = float(event["onset_time"]) / beat_duration
    onset_slot = int(round(onset_beats / max(float(grid_beats), 1e-6)))
    return onset_slot


def _score_event_signature(event: Dict, bpm: float, grid_beats: float) -> Tuple[int, int, int]:
    return int(event["midi_note"]), _score_event_slot(event, bpm, grid_beats), _event_note_value_class(event, bpm)


def _score_event_hand(event: Dict) -> str:
    return "bass" if int(event["midi_note"]) < 60 else "treble"


def _next_policy_onsets(events: Sequence[Dict]) -> Tuple[Dict[int, float], Dict[int, float]]:
    by_pitch = defaultdict(list)
    by_hand = defaultdict(list)
    for idx, event in enumerate(events):
        onset = float(event["onset_time"])
        by_pitch[int(event["midi_note"])].append((onset, idx))
        by_hand[_score_event_hand(event)].append((onset, idx))

    next_same_pitch = {}
    for values in by_pitch.values():
        values.sort()
        for pos, (_, idx) in enumerate(values[:-1]):
            next_same_pitch[idx] = values[pos + 1][0]

    next_same_hand = {}
    for values in by_hand.values():
        values.sort()
        for pos, (_, idx) in enumerate(values[:-1]):
            next_same_hand[idx] = values[pos + 1][0]
    return next_same_pitch, next_same_hand


def _pred_score_note_value_class(
    events: Sequence[Dict],
    event_idx: int,
    bpm: float,
    duration_policy: str,
    next_same_pitch: Optional[Dict[int, float]] = None,
    next_same_hand: Optional[Dict[int, float]] = None,
    duration_lookup: Optional[Dict[Tuple[int, int, int], int]] = None,
) -> int:
    event = events[event_idx]
    onset = float(event["onset_time"])
    decoded_end = float(event.get("offset_time", onset))
    head_class = _event_note_value_class(event, bpm)
    decoded_class = _duration_to_note_value_class(max(decoded_end - onset, 1e-6), bpm)

    if duration_policy == "head":
        return head_class
    if duration_policy == "decoded_duration":
        return decoded_class

    if next_same_pitch is None or next_same_hand is None:
        next_same_pitch, next_same_hand = _next_policy_onsets(events)

    same_pitch_end = min(decoded_end, float(next_same_pitch.get(event_idx, decoded_end)))
    if duration_policy == "same_pitch_cap":
        return _duration_to_note_value_class(max(same_pitch_end - onset, 1e-6), bpm)

    same_hand_end = float(next_same_hand.get(event_idx, same_pitch_end))
    same_hand_end = min(same_hand_end, float(next_same_pitch.get(event_idx, same_hand_end)))
    ioi_class = _duration_to_note_value_class(max(same_hand_end - onset, 1e-6), bpm)
    if duration_policy == "ioi_same_hand":
        return ioi_class
    if duration_policy == "lookup_ioi_head_sound":
        if duration_lookup is None:
            return ioi_class
        return int(duration_lookup.get((ioi_class, head_class, decoded_class), ioi_class))

    capped_class = _duration_to_note_value_class(max(same_pitch_end - onset, 1e-6), bpm)
    if duration_policy == "hybrid_cleanup":
        if capped_class <= 1 and ioi_class >= 2:
            return ioi_class
        if capped_class <= 3 and ioi_class >= capped_class + 2:
            return min(ioi_class, capped_class + 2)
        return capped_class

    raise ValueError(f"Unknown score duration policy: {duration_policy}")


def match_score_events(
    pred: Sequence[Dict],
    gt: Sequence[Dict],
    bpm: float,
    grid_beats: float = SCORE_GRID_BEATS,
    onset_slot_tolerance: int = 0,
    duration_class_tolerance: int = 0,
    duration_policy: str = "head",
    duration_lookup: Optional[Dict[Tuple[int, int, int], int]] = None,
) -> Dict[str, float]:
    used_gt = set()
    onset_matched = 0
    duration_matched = 0
    matched_pairs = []

    next_same_pitch, next_same_hand = _next_policy_onsets(pred)
    pred_sig = [
        (
            int(event["midi_note"]),
            _score_event_slot(event, bpm, grid_beats),
            _pred_score_note_value_class(
                pred,
                idx,
                bpm,
                duration_policy,
                next_same_pitch,
                next_same_hand,
                duration_lookup,
            ),
        )
        for idx, event in enumerate(pred)
    ]
    gt_sig = [_score_event_signature(event, bpm, grid_beats) for event in gt]

    for pred_idx, (pred_pitch, pred_slot, pred_nv) in enumerate(pred_sig):
        best_idx = None
        best_score = None
        for gt_idx, (gt_pitch, gt_slot, gt_nv) in enumerate(gt_sig):
            if gt_idx in used_gt or pred_pitch != gt_pitch:
                continue
            slot_error = abs(pred_slot - gt_slot)
            if slot_error > onset_slot_tolerance:
                continue
            nv_error = abs(pred_nv - gt_nv)
            score = (slot_error, nv_error)
            if best_score is None or score < best_score:
                best_idx = gt_idx
                best_score = score

        if best_idx is None:
            continue
        used_gt.add(best_idx)
        onset_matched += 1
        matched_pairs.append((pred_idx, best_idx))
        if abs(pred_sig[pred_idx][2] - gt_sig[best_idx][2]) <= duration_class_tolerance:
            duration_matched += 1

    onset_precision = onset_matched / max(len(pred), 1)
    onset_recall = onset_matched / max(len(gt), 1)
    onset_f1 = 2 * onset_precision * onset_recall / max(onset_precision + onset_recall, 1e-8)
    score_precision = duration_matched / max(len(pred), 1)
    score_recall = duration_matched / max(len(gt), 1)
    score_f1 = 2 * score_precision * score_recall / max(score_precision + score_recall, 1e-8)
    duration_accuracy = duration_matched / max(onset_matched, 1)
    return {
        "precision": score_precision,
        "recall": score_recall,
        "f1": score_f1,
        "onset_precision": onset_precision,
        "onset_recall": onset_recall,
        "onset_f1": onset_f1,
        "duration_accuracy": duration_accuracy,
        "matched": duration_matched,
        "onset_matched": onset_matched,
        "predicted": len(pred),
        "ground_truth": len(gt),
    }


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def enhanced_collate(batch: Sequence[Dict]) -> Dict:
    tensor_keys = [
        "features", "onset", "offset", "frame", "sounding_frame", "velocity", "note_value",
        "pedal", "bpm", "segment_id", "start_sec",
    ]
    for optional_key in ("teacher_features", "loss_mask", "crop_start_frame"):
        if optional_key in batch[0]:
            tensor_keys.append(optional_key)
    if "sample_source" in batch[0]:
        tensor_keys.append("sample_source")
    out = {}
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["midi_path"] = [item["midi_path"] for item in batch]
    out["gt_events"] = [item["gt_events"] for item in batch]
    return out


def _frame_f1(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[int, int, int]:
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    return int(tp), int(fp), int(fn)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: EnhancedTranscriptionLoss,
    device: torch.device,
    use_amp: bool,
    onset_threshold: float,
    offset_threshold: float,
    frame_threshold: float,
    max_event_batches: int = 20,
    max_event_samples: int = 0,
    event_sampling: str = "spread",
    max_score_batches: int = 20,
    max_score_samples: int = 0,
    score_sampling: str = "spread",
    score_grid_beats: float = SCORE_GRID_BEATS,
    score_onset_slot_tolerance: int = 0,
    score_duration_class_tolerance: int = 0,
    score_duration_policy: str = "head",
    score_duration_lookup_path: Optional[str] = None,
    max_batches: Optional[int] = None,
    decode_use_sounding_frame: bool = True,
) -> Dict:
    model.eval()
    loss_sums = defaultdict(float)
    n_batches = 0
    onset_counts = [0, 0, 0]
    offset_counts = [0, 0, 0]
    frame_counts = [0, 0, 0]
    sounding_frame_counts = [0, 0, 0]
    pedal_counts = [0, 0, 0]
    event_totals = defaultdict(int)
    score_totals = defaultdict(int)
    event_samples = 0
    score_samples = 0
    seen_samples = 0
    duration_lookup = load_score_duration_lookup(score_duration_lookup_path)

    def make_sample_selection(max_samples: int, max_sample_batches: int, sampling: str) -> Tuple[int, Optional[set]]:
        sample_limit = int(max_samples or 0)
        if sample_limit <= 0 and max_sample_batches > 0:
            sample_limit = int(max_sample_batches) * int(loader.batch_size or 1)
        selected_indices: Optional[set] = None
        if sample_limit <= 0 or sampling != "spread":
            return sample_limit, selected_indices
        total_samples = len(loader.dataset) if hasattr(loader, "dataset") else 0
        if max_batches is not None and loader.batch_size is not None:
            total_samples = min(total_samples, int(max_batches) * int(loader.batch_size))
        if total_samples > 0:
            count = min(sample_limit, total_samples)
            selected_indices = set(int(idx) for idx in np.linspace(0, total_samples - 1, count))
        return sample_limit, selected_indices

    event_sample_limit, selected_event_indices = make_sample_selection(
        max_event_samples,
        max_event_batches,
        event_sampling,
    )
    score_sample_limit, selected_score_indices = make_sample_selection(
        max_score_samples,
        max_score_batches,
        score_sampling,
    )

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_dev = _move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch_dev["features"])
            losses = criterion(out, batch_dev)
        for key, value in losses.items():
            loss_sums[key] += float(value.detach().cpu())
        n_batches += 1

        counts = _frame_f1(out["onset_logits"], batch_dev["onset"])
        onset_counts = [a + b for a, b in zip(onset_counts, counts)]
        counts = _frame_f1(out["offset_logits"], batch_dev["offset"])
        offset_counts = [a + b for a, b in zip(offset_counts, counts)]
        counts = _frame_f1(out["frame_logits"], batch_dev["frame"])
        frame_counts = [a + b for a, b in zip(frame_counts, counts)]
        counts = _frame_f1(out["sounding_frame_logits"], batch_dev["sounding_frame"])
        sounding_frame_counts = [a + b for a, b in zip(sounding_frame_counts, counts)]
        counts = _frame_f1(out["pedal_logits"], batch_dev["pedal"])
        pedal_counts = [a + b for a, b in zip(pedal_counts, counts)]

        current_batch_size = int(batch_dev["features"].size(0))
        if selected_event_indices is not None or selected_score_indices is not None:
            should_prepare_event_arrays = any(
                (
                    (selected_event_indices is not None and (seen_samples + sample_idx) in selected_event_indices)
                    or (selected_score_indices is not None and (seen_samples + sample_idx) in selected_score_indices)
                )
                for sample_idx in range(current_batch_size)
            )
        else:
            should_prepare_event_arrays = (
                (event_sample_limit <= 0 or event_samples < event_sample_limit)
                or (score_sample_limit <= 0 or score_samples < score_sample_limit)
            )
        if should_prepare_event_arrays:
            onset_np = torch.sigmoid(out["onset_logits"]).cpu().numpy()
            offset_np = torch.sigmoid(out["offset_logits"]).cpu().numpy()
            frame_key = "sounding_frame_logits" if decode_use_sounding_frame else "frame_logits"
            frame_np = torch.sigmoid(out[frame_key]).cpu().numpy()
            vel_np = out["velocity"].cpu().numpy()
            nv_np = F.softmax(out["note_value_logits"], dim=-1).cpu().numpy()
            for sample_idx in range(onset_np.shape[0]):
                global_sample_idx = seen_samples + sample_idx
                use_event_sample = True
                if event_sample_limit > 0:
                    if selected_event_indices is not None:
                        use_event_sample = global_sample_idx in selected_event_indices
                    elif event_samples >= event_sample_limit:
                        use_event_sample = False
                use_score_sample = True
                if score_sample_limit > 0:
                    if selected_score_indices is not None:
                        use_score_sample = global_sample_idx in selected_score_indices
                    elif score_samples >= score_sample_limit:
                        use_score_sample = False
                if not use_event_sample and not use_score_sample:
                    continue

                pred_events = decode_enhanced_note_events(
                    onset_np[sample_idx],
                    offset_np[sample_idx],
                    frame_np[sample_idx],
                    vel_np[sample_idx],
                    nv_np[sample_idx],
                    onset_threshold=onset_threshold,
                    offset_threshold=offset_threshold,
                    frame_threshold=frame_threshold,
                )
                gt_events = batch["gt_events"][sample_idx]
                if use_event_sample:
                    metrics = match_note_events(pred_events, gt_events)
                    event_totals["matched"] += int(metrics["matched"])
                    event_totals["predicted"] += int(metrics["predicted"])
                    event_totals["ground_truth"] += int(metrics["ground_truth"])
                    event_samples += 1
                if use_score_sample:
                    bpm = float(batch["bpm"][sample_idx].detach().cpu()) if torch.is_tensor(batch["bpm"][sample_idx]) else float(batch["bpm"][sample_idx])
                    score_metrics = match_score_events(
                        pred_events,
                        gt_events,
                        bpm=bpm,
                        grid_beats=score_grid_beats,
                        onset_slot_tolerance=score_onset_slot_tolerance,
                        duration_class_tolerance=score_duration_class_tolerance,
                        duration_policy=score_duration_policy,
                        duration_lookup=duration_lookup,
                    )
                    for key in (
                        "matched", "onset_matched", "predicted", "ground_truth",
                    ):
                        score_totals[key] += int(score_metrics[key])
                    score_samples += 1

        seen_samples += current_batch_size

    def counts_to_metrics(counts: Sequence[int]) -> Dict[str, float]:
        tp, fp, fn = counts
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        }

    event_p = event_totals["matched"] / max(event_totals["predicted"], 1)
    event_r = event_totals["matched"] / max(event_totals["ground_truth"], 1)
    event_f1 = 2 * event_p * event_r / max(event_p + event_r, 1e-8)
    score_p = score_totals["matched"] / max(score_totals["predicted"], 1)
    score_r = score_totals["matched"] / max(score_totals["ground_truth"], 1)
    score_f1 = 2 * score_p * score_r / max(score_p + score_r, 1e-8)
    score_onset_p = score_totals["onset_matched"] / max(score_totals["predicted"], 1)
    score_onset_r = score_totals["onset_matched"] / max(score_totals["ground_truth"], 1)
    score_onset_f1 = 2 * score_onset_p * score_onset_r / max(score_onset_p + score_onset_r, 1e-8)
    score_duration_acc = score_totals["matched"] / max(score_totals["onset_matched"], 1)
    return {
        "losses": {key: value / max(n_batches, 1) for key, value in loss_sums.items()},
        "onset": counts_to_metrics(onset_counts),
        "offset": counts_to_metrics(offset_counts),
        "frame": counts_to_metrics(frame_counts),
        "sounding_frame": counts_to_metrics(sounding_frame_counts),
        "pedal": counts_to_metrics(pedal_counts),
        "event": {
            "precision": event_p,
            "recall": event_r,
            "f1": event_f1,
            "samples": event_samples,
            **dict(event_totals),
        },
        "score": {
            "precision": score_p,
            "recall": score_r,
            "f1": score_f1,
            "onset_precision": score_onset_p,
            "onset_recall": score_onset_r,
            "onset_f1": score_onset_f1,
            "duration_accuracy": score_duration_acc,
            "samples": score_samples,
            **dict(score_totals),
        },
    }


def _build_model_from_args(args) -> EnhancedMelTranscriber:
    return EnhancedMelTranscriber(
        conv_channels=args.conv_channels,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_expansion=args.ff_expansion,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
        event_hidden=args.event_hidden,
        event_context=getattr(args, "event_context", False),
        event_residual=getattr(args, "event_residual", False),
        cross_key_layers=getattr(args, "cross_key_layers", 0),
        cross_key_heads=getattr(args, "cross_key_heads", 4),
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        use_checkpoint=args.use_checkpoint,
    )


def _build_model_from_config(config: Dict) -> EnhancedMelTranscriber:
    """Build EnhancedMelTranscriber from a saved checkpoint config."""
    return EnhancedMelTranscriber(
        n_mels=int(config.get("n_mels", N_MELS)),
        conv_channels=int(config.get("conv_channels", 192)),
        d_model=int(config.get("d_model", 384)),
        n_layers=int(config.get("n_layers", 10)),
        n_heads=int(config.get("n_heads", 8)),
        ff_expansion=int(config.get("ff_expansion", 4)),
        conv_kernel=int(config.get("conv_kernel", 31)),
        dropout=float(config.get("dropout", 0.1)),
        event_hidden=int(config.get("event_hidden", 192)),
        event_context=bool(config.get("event_context", False)),
        event_residual=bool(config.get("event_residual", False)),
        cross_key_layers=int(config.get("cross_key_layers", 0)),
        cross_key_heads=int(config.get("cross_key_heads", 4)),
        adapter_bottleneck=int(config.get("adapter_bottleneck", 0)),
        adapter_dropout=float(config.get("adapter_dropout", 0.0)),
        n_note_value_classes=int(config.get("n_note_value_classes", NOTE_VALUE_CLASSES)),
        use_checkpoint=bool(config.get("use_checkpoint", False)),
    )


def _load_teacher_model(checkpoint_path: Path, fallback_args, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if isinstance(config, dict) and config.get("model_type") == "EnhancedMelTranscriber":
        model = _build_model_from_config(config).to(device).eval()
    else:
        model = _build_model_from_args(fallback_args).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    for param in model.parameters():
        param.requires_grad = False
    return model


def _checkpoint_config(args) -> Dict:
    return {
        "model_type": "EnhancedMelTranscriber",
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "n_keys": PIANO_KEYS,
        "n_note_value_classes": NOTE_VALUE_CLASSES,
        "onset_tent_sec": ONSET_TENT_SEC,
        "offset_tent_sec": OFFSET_TENT_SEC,
        "pedal_cc": PEDAL_CC,
        "pedal_down_threshold": PEDAL_DOWN_THRESHOLD,
        "pedal_informed_sounding_frame": True,
        "conv_channels": args.conv_channels,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "ff_expansion": args.ff_expansion,
        "conv_kernel": args.conv_kernel,
        "dropout": args.dropout,
        "event_hidden": args.event_hidden,
        "event_context": bool(args.event_context),
        "event_residual": bool(args.event_residual),
        "cross_key_layers": int(args.cross_key_layers),
        "cross_key_heads": int(args.cross_key_heads),
        "adapter_bottleneck": args.adapter_bottleneck,
        "adapter_dropout": args.adapter_dropout,
        "explicit_offset_head": True,
        "pitch_local_readout": True,
        "save_best_on": args.save_best_on,
        "finetune": bool(args.finetune),
        "finetune_scope": args.finetune_scope,
        "finetune_hard_ratio": args.finetune_hard_ratio,
        "train_window_sec": float(args.train_window_sec),
        "emit_window_sec": float(args.emit_window_sec),
        "teacher_preserve_weight": args.teacher_preserve_weight,
        "teacher_from": args.teacher_from,
        "live_window_distill_weight": float(args.live_window_distill_weight),
        "live_distill_temperature": float(args.live_distill_temperature),
        "live_distill_onset_weight": float(args.live_distill_onset_weight),
        "live_distill_offset_weight": float(args.live_distill_offset_weight),
        "live_distill_frame_weight": float(args.live_distill_frame_weight),
        "live_distill_sounding_frame_weight": float(args.live_distill_sounding_frame_weight),
        "live_distill_pedal_weight": float(args.live_distill_pedal_weight),
        "live_distill_velocity_weight": float(args.live_distill_velocity_weight),
        "live_distill_note_value_weight": float(args.live_distill_note_value_weight),
        "lr": args.lr,
        "pos_weight": args.pos_weight,
        "onset_weight": args.onset_weight,
        "raw_onset_weight": args.raw_onset_weight,
        "offset_weight": args.offset_weight,
        "raw_offset_weight": args.raw_offset_weight,
        "frame_weight": args.frame_weight,
        "sounding_frame_weight": args.sounding_frame_weight,
        "pedal_weight": args.pedal_weight,
        "decode_use_sounding_frame": bool(args.decode_use_sounding_frame),
        "max_val_samples": int(args.max_val_samples),
        "val_sampling": args.val_sampling,
        "max_score_val_batches": int(args.max_score_val_batches),
        "max_score_val_samples": int(args.max_score_val_samples),
        "score_val_sampling": args.score_val_sampling,
        "score_grid_beats": float(args.score_grid_beats),
        "score_onset_slot_tolerance": int(args.score_onset_slot_tolerance),
        "score_duration_class_tolerance": int(args.score_duration_class_tolerance),
        "score_duration_policy": args.score_duration_policy,
        "score_duration_lookup_path": args.score_duration_lookup_path,
        "velocity_weight": args.velocity_weight,
        "nv_weight": args.nv_weight,
        "focal_gamma": args.focal_gamma,
    }


def _resolve_training_paths(args) -> Tuple[Optional[Path], Path]:
    default_save_path = DEFAULT_FINETUNE_MODEL_PATH if args.finetune else MODEL_PATH
    save_path = Path(args.model_path) if args.model_path else default_save_path

    init_checkpoint_path = Path(args.init_from) if args.init_from else None
    if args.resume and init_checkpoint_path is None:
        init_checkpoint_path = save_path
    elif args.finetune and init_checkpoint_path is None:
        init_checkpoint_path = MODEL_PATH

    return init_checkpoint_path, save_path


def _build_optimizer_param_groups(model: EnhancedMelTranscriber, args):
    component_groups = {
        "backbone": [
            ("freq_stack", model.freq_stack),
            ("global_proj", model.global_proj),
            ("conformer_blocks", model.conformer_blocks),
        ],
        "decoder": [
            ("global_key_proj", model.global_key_proj),
            ("local_readout", model.local_readout),
            ("key_temporal", model.key_temporal),
            ("cross_key_blocks", model.cross_key_blocks),
        ],
        "heads": [
            ("onset_head_raw", model.onset_head_raw),
            ("offset_head_raw", model.offset_head_raw),
            ("velocity_head", model.velocity_head),
            ("event_refine_gru", model.event_refine_gru),
            ("event_refine_fc", model.event_refine_fc),
            ("frame_head", model.frame_head),
            ("sounding_frame_head", model.sounding_frame_head),
            ("pedal_head", model.pedal_head),
            ("note_value_head", model.note_value_head),
        ],
    }
    if model.event_context_head is not None:
        component_groups["heads"].append(("event_context_head", model.event_context_head))
    if model.conformer_adapters is not None:
        component_groups["backbone"].append(("conformer_adapters", model.conformer_adapters))

    for param in model.parameters():
        param.requires_grad = True

    if not args.finetune:
        all_modules = [name for groups in component_groups.values() for name, _ in groups]
        all_params = [param for param in model.parameters() if param.requires_grad]
        trainable_count = sum(param.numel() for param in all_params)
        return (
            [{"params": all_params, "lr": args.lr, "name": "full"}],
            all_modules,
            [],
            trainable_count,
            0,
        )

    trainable_categories = {
        "heads": {"heads"},
        "decoder": {"decoder", "heads"},
        "full": {"backbone", "decoder", "heads"},
    }[args.finetune_scope]
    lr_scales = {
        "backbone": args.backbone_lr_scale,
        "decoder": args.decoder_lr_scale,
        "heads": args.head_lr_scale,
    }

    optimizer_groups = []
    trainable_modules = []
    frozen_modules = []

    for category, modules in component_groups.items():
        category_params = []
        category_is_trainable = category in trainable_categories
        for module_name, module in modules:
            module_params = list(module.parameters())
            if not module_params:
                continue
            for param in module_params:
                param.requires_grad = category_is_trainable
            if category_is_trainable:
                trainable_modules.append(module_name)
                category_params.extend(module_params)
            else:
                frozen_modules.append(module_name)

        if category_is_trainable and category_params:
            optimizer_groups.append({
                "params": category_params,
                "lr": args.lr * lr_scales[category],
                "name": category,
            })

    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_count = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return optimizer_groups, trainable_modules, frozen_modules, trainable_count, frozen_count


def load_segment_manifest(path: Optional[str], split: str) -> Optional[List[int]]:
    if not path:
        return None

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    manifest_split = payload.get("split")
    if manifest_split is not None and manifest_split != split:
        raise ValueError(
            f"Manifest {manifest_path} is for split={manifest_split!r}, expected {split!r}"
        )

    segment_ids = payload.get("segment_ids")
    if segment_ids is None:
        segment_ids = [row["segment_id"] for row in payload.get("selection", [])]
    if not segment_ids:
        raise ValueError(f"Manifest {manifest_path} does not contain any segment_ids")

    ids = sorted(set(int(segment_id) for segment_id in segment_ids))
    strategy = payload.get("selection_strategy", "unknown")
    print(f"[Manifest] {split}: loaded {len(ids)} segment IDs from {manifest_path} ({strategy})")
    return ids


def load_score_duration_lookup(path: Optional[str]) -> Optional[Dict[Tuple[int, int, int], int]]:
    if not path:
        return None
    lookup_path = Path(path)
    if not lookup_path.exists():
        raise FileNotFoundError(f"Score duration lookup not found: {lookup_path}")
    with lookup_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    table_payload = payload.get("table")
    if table_payload is None and isinstance(payload.get("best_lookup"), dict):
        table_payload = payload["best_lookup"].get("table")
    if table_payload is None:
        raise ValueError(f"Score duration lookup {lookup_path} has no table or best_lookup.table")

    table: Dict[Tuple[int, int, int], int] = {}
    for raw_key, raw_value in table_payload.items():
        parts = tuple(int(part) for part in str(raw_key).split("|"))
        if len(parts) != 3:
            raise ValueError(f"Expected 3-part ioi|head|sound lookup key, got {raw_key!r}")
        table[parts] = int(raw_value)
    print(f"[ScoreDurationLookup] loaded {len(table)} entries from {lookup_path}")
    return table


def _cap_validation_dataset(dataset: Dataset, max_samples: int, sampling: str, name: str) -> Dataset:
    max_samples = int(max_samples or 0)
    total = len(dataset)
    if max_samples <= 0 or max_samples >= total:
        return dataset

    count = max(1, max_samples)
    if sampling == "leading":
        indices = list(range(count))
    else:
        indices = [int(round(idx)) for idx in np.linspace(0, total - 1, count)]
        indices = list(dict.fromkeys(indices))
        if len(indices) < count:
            used = set(indices)
            for idx in range(total):
                if idx not in used:
                    indices.append(idx)
                    used.add(idx)
                    if len(indices) >= count:
                        break
        indices = sorted(indices[:count])

    print(f"[ValSample] {name}: using {len(indices)}/{total} samples ({sampling})")
    return Subset(dataset, indices)


def _build_train_dataset_and_sampler(args) -> Tuple[Dataset, Optional[WeightedRandomSampler], bool]:
    hard_segment_ids = load_segment_manifest(args.train_segment_manifest, "train")
    include_teacher_features = bool(args.live_window_distill_weight > 0)
    if not args.finetune or hard_segment_ids is None or args.finetune_hard_ratio >= 0.999:
        dataset = EnhancedPrecomputedMelDataset(
            "train",
            augment=args.train_augment,
            segment_ids=hard_segment_ids,
            train_window_sec=args.train_window_sec,
            emit_window_sec=args.emit_window_sec,
            include_teacher_features=include_teacher_features,
        )
        if args.finetune and hard_segment_ids is not None:
            dataset = SourceTaggedDataset(dataset, source_id=1)
        return dataset, None, True

    general_dataset = SourceTaggedDataset(
        EnhancedPrecomputedMelDataset(
            "train",
            augment=args.train_augment,
            train_window_sec=args.train_window_sec,
            emit_window_sec=args.emit_window_sec,
            include_teacher_features=include_teacher_features,
        ),
        source_id=0,
    )
    hard_dataset = SourceTaggedDataset(
        EnhancedPrecomputedMelDataset(
            "train",
            augment=args.train_augment,
            segment_ids=hard_segment_ids,
            train_window_sec=args.train_window_sec,
            emit_window_sec=args.emit_window_sec,
            include_teacher_features=include_teacher_features,
        ),
        source_id=1,
    )
    dataset = ConcatDataset([general_dataset, hard_dataset])

    hard_ratio = float(np.clip(args.finetune_hard_ratio, 0.0, 1.0))
    general_ratio = 1.0 - hard_ratio
    weights = (
        [general_ratio / max(len(general_dataset), 1)] * len(general_dataset)
        + [hard_ratio / max(len(hard_dataset), 1)] * len(hard_dataset)
    )
    if args.finetune_samples_per_epoch > 0:
        samples_per_epoch = args.finetune_samples_per_epoch
    else:
        samples_per_epoch = max(len(hard_dataset) * 2, args.batch_size * 100)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(samples_per_epoch),
        replacement=True,
    )
    print(
        f"[FineTuneMix] general={len(general_dataset)} hard={len(hard_dataset)} "
        f"hard_ratio={hard_ratio:.2f} samples_per_epoch={samples_per_epoch}"
    )
    return dataset, sampler, False


def _build_validation_loaders(args, device: torch.device) -> Dict[str, DataLoader]:
    validation_segment_ids = load_segment_manifest(args.validation_segment_manifest, "validation")
    loaders = {}
    primary_dataset = EnhancedPrecomputedMelDataset(
        "validation",
        augment=False,
        segment_ids=validation_segment_ids,
    )
    primary_name = "hard" if validation_segment_ids is not None else "validation"
    primary_dataset = _cap_validation_dataset(
        primary_dataset,
        args.max_val_samples,
        args.val_sampling,
        primary_name,
    )
    loaders[primary_name] = DataLoader(
        primary_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=enhanced_collate,
    )
    if args.finetune and validation_segment_ids is not None and args.finetune_eval_general:
        general_dataset = EnhancedPrecomputedMelDataset("validation", augment=False)
        general_dataset = _cap_validation_dataset(
            general_dataset,
            args.max_val_samples,
            args.val_sampling,
            "general",
        )
        loaders["general"] = DataLoader(
            general_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=enhanced_collate,
        )
    return loaders


def _slice_teacher_time_window(value: torch.Tensor, starts: torch.Tensor, target_frames: int) -> torch.Tensor:
    windows = []
    total_frames = int(value.size(1))
    for batch_idx, start_value in enumerate(starts.detach().cpu().tolist()):
        start = max(0, int(start_value))
        end = min(start + target_frames, total_frames)
        window = value[batch_idx, start:end]
        current_frames = int(window.size(0))
        if current_frames < target_frames:
            padded_shape = (target_frames, *window.shape[1:])
            padded = torch.zeros(padded_shape, dtype=value.dtype, device=value.device)
            if current_frames > 0:
                padded[:current_frames] = window
            window = padded
        windows.append(window)
    return torch.stack(windows, dim=0)


def _maybe_slice_teacher_out(
    teacher_out: Dict[str, torch.Tensor],
    student_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    target_frames = int(student_out["onset_logits"].size(1))
    teacher_frames = int(teacher_out["onset_logits"].size(1))
    if teacher_frames == target_frames:
        return teacher_out
    if "crop_start_frame" not in batch:
        raise ValueError("Teacher output length differs from student output, but batch has no crop_start_frame")
    starts = batch["crop_start_frame"].to(student_out["onset_logits"].device)
    sliced = {}
    for key, value in teacher_out.items():
        if torch.is_tensor(value) and value.dim() >= 2 and int(value.size(1)) == teacher_frames:
            sliced[key] = _slice_teacher_time_window(value, starts, target_frames)
        else:
            sliced[key] = value
    return sliced


def _time_key_mask(
    student_tensor: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    sample_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if "loss_mask" in batch:
        mask = batch["loss_mask"].to(student_tensor.device) > 0.5
        while mask.dim() < student_tensor.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(student_tensor)
    else:
        mask = torch.ones_like(student_tensor, dtype=torch.bool)
    if sample_mask is not None:
        batch_mask = sample_mask.to(student_tensor.device).bool()
        view_shape = [batch_mask.size(0)] + [1] * (student_tensor.dim() - 1)
        mask = mask & batch_mask.view(*view_shape)
    return mask


def _teacher_bce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if not mask.any():
        return torch.tensor(0.0, device=student_logits.device)
    temp = max(float(temperature), 1e-6)
    teacher_prob = torch.sigmoid(teacher_logits.float() / temp).detach()
    return F.binary_cross_entropy_with_logits(
        student_logits.float()[mask] / temp,
        teacher_prob[mask],
    ) * (temp * temp)


def _teacher_preservation_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args,
) -> torch.Tensor:
    if args.teacher_preserve_weight <= 0:
        return torch.tensor(0.0, device=student_out["onset_logits"].device)

    if "sample_source" in batch:
        preserve_mask = batch["sample_source"] == 0
        if not preserve_mask.any():
            return torch.tensor(0.0, device=student_out["onset_logits"].device)
    else:
        preserve_mask = torch.ones(
            student_out["onset_logits"].shape[0],
            dtype=torch.bool,
            device=student_out["onset_logits"].device,
        )

    teacher_out = _maybe_slice_teacher_out(teacher_out, student_out, batch)
    loss = torch.tensor(0.0, device=student_out["onset_logits"].device)
    weighted_terms = 0.0
    for key, weight in (
        ("onset_logits", args.teacher_onset_weight),
        ("offset_logits", args.teacher_offset_weight),
        ("frame_logits", args.teacher_frame_weight),
    ):
        if weight <= 0:
            continue
        mask = _time_key_mask(student_out[key], batch, preserve_mask)
        if not mask.any():
            continue
        teacher_prob = torch.sigmoid(teacher_out[key].float()).detach()
        term = F.binary_cross_entropy_with_logits(student_out[key].float()[mask], teacher_prob[mask])
        loss = loss + weight * term
        weighted_terms += weight

    if args.teacher_velocity_weight > 0:
        mask = _time_key_mask(student_out["velocity"], batch, preserve_mask)
        if mask.any():
            term = F.mse_loss(student_out["velocity"][mask].float(), teacher_out["velocity"][mask].float().detach())
            loss = loss + args.teacher_velocity_weight * term
            weighted_terms += args.teacher_velocity_weight

    if args.teacher_note_value_weight > 0:
        mask = _time_key_mask(student_out["note_value_logits"][..., 0], batch, preserve_mask)
        if mask.any():
            student_log_prob = F.log_softmax(student_out["note_value_logits"].float(), dim=-1)
            teacher_prob = F.softmax(teacher_out["note_value_logits"].float(), dim=-1).detach()
            term = F.kl_div(student_log_prob[mask], teacher_prob[mask], reduction="none").sum(dim=-1).mean()
            loss = loss + args.teacher_note_value_weight * term
            weighted_terms += args.teacher_note_value_weight

    if weighted_terms <= 0:
        return torch.tensor(0.0, device=student_out["onset_logits"].device)
    return args.teacher_preserve_weight * loss / weighted_terms


def _live_window_distillation_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args,
) -> Dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, device=student_out["onset_logits"].device)
    if args.live_window_distill_weight <= 0:
        return {"total": zero}
    if "crop_start_frame" not in batch or "loss_mask" not in batch:
        raise ValueError("Live-window distillation requires --train-window-sec and --emit-window-sec")

    teacher_out = _maybe_slice_teacher_out(teacher_out, student_out, batch)
    temperature = args.live_distill_temperature
    losses: Dict[str, torch.Tensor] = {}
    weighted_total = zero
    total_weight = 0.0
    for name, key, weight in (
        ("onset", "onset_logits", args.live_distill_onset_weight),
        ("offset", "offset_logits", args.live_distill_offset_weight),
        ("frame", "frame_logits", args.live_distill_frame_weight),
        ("sounding_frame", "sounding_frame_logits", args.live_distill_sounding_frame_weight),
    ):
        if weight <= 0:
            continue
        mask = _time_key_mask(student_out[key], batch)
        term = _teacher_bce_loss(student_out[key], teacher_out[key], mask, temperature)
        losses[name] = term
        weighted_total = weighted_total + weight * term
        total_weight += weight

    if args.live_distill_pedal_weight > 0:
        mask = _time_key_mask(student_out["pedal_logits"], batch)
        term = _teacher_bce_loss(student_out["pedal_logits"], teacher_out["pedal_logits"], mask, temperature)
        losses["pedal"] = term
        weighted_total = weighted_total + args.live_distill_pedal_weight * term
        total_weight += args.live_distill_pedal_weight

    if args.live_distill_velocity_weight > 0:
        mask = _time_key_mask(student_out["velocity"], batch)
        if mask.any():
            term = F.mse_loss(student_out["velocity"][mask].float(), teacher_out["velocity"][mask].float().detach())
        else:
            term = zero
        losses["velocity"] = term
        weighted_total = weighted_total + args.live_distill_velocity_weight * term
        total_weight += args.live_distill_velocity_weight

    if args.live_distill_note_value_weight > 0:
        mask = _time_key_mask(student_out["note_value_logits"][..., 0], batch)
        if mask.any():
            student_log_prob = F.log_softmax(student_out["note_value_logits"].float(), dim=-1)
            teacher_prob = F.softmax(teacher_out["note_value_logits"].float(), dim=-1).detach()
            term = F.kl_div(student_log_prob[mask], teacher_prob[mask], reduction="none").sum(dim=-1).mean()
        else:
            term = zero
        losses["note_value"] = term
        weighted_total = weighted_total + args.live_distill_note_value_weight * term
        total_weight += args.live_distill_note_value_weight

    if total_weight <= 0:
        losses["total"] = zero
    else:
        losses["total"] = args.live_window_distill_weight * weighted_total / total_weight
    return losses


def train(args) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.live_window_distill_weight > 0:
        if args.train_window_sec <= 0 or args.emit_window_sec <= 0:
            raise ValueError("Live-window distillation requires --train-window-sec and --emit-window-sec")
        if args.emit_window_sec > args.train_window_sec:
            raise ValueError("--emit-window-sec cannot exceed --train-window-sec")

    init_checkpoint_path, save_path = _resolve_training_paths(args)
    if args.finetune and init_checkpoint_path is None:
        raise ValueError("--finetune requires a checkpoint via --init-from or the default model path")
    if init_checkpoint_path is not None and not init_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {init_checkpoint_path}")

    train_dataset, train_sampler, train_shuffle = _build_train_dataset_and_sampler(args)
    validation_loaders = _build_validation_loaders(args, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=enhanced_collate,
    )

    model = _build_model_from_args(args).to(device)
    checkpoint = None
    if init_checkpoint_path is not None:
        checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"{'Resumed' if args.resume else 'Initialized'} from {init_checkpoint_path}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    teacher_model = None
    teacher_preserve_enabled = bool(args.finetune and args.teacher_preserve_weight > 0)
    if teacher_preserve_enabled or args.live_window_distill_weight > 0:
        teacher_path = Path(args.teacher_from) if args.teacher_from else init_checkpoint_path
        if teacher_path is None:
            raise ValueError("Teacher losses require --teacher-from or an init checkpoint")
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
        teacher_model = _load_teacher_model(teacher_path, args, device)
        print(
            f"Teacher model: {teacher_path} "
            f"preserve={(args.teacher_preserve_weight if teacher_preserve_enabled else 0.0):.3f} "
            f"live_distill={args.live_window_distill_weight:.3f}"
        )

    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = EnhancedTranscriptionLoss(
        pos_weight=args.pos_weight,
        onset_weight=args.onset_weight,
        raw_onset_weight=args.raw_onset_weight,
        offset_weight=args.offset_weight,
        raw_offset_weight=args.raw_offset_weight,
        frame_weight=args.frame_weight,
        sounding_frame_weight=args.sounding_frame_weight,
        pedal_weight=args.pedal_weight,
        velocity_weight=args.velocity_weight,
        nv_weight=args.nv_weight,
        focal_gamma=args.focal_gamma,
    )
    optimizer_groups, trainable_modules, frozen_modules, trainable_count, frozen_count = _build_optimizer_param_groups(
        model,
        args,
    )
    optimizer = optim.AdamW(optimizer_groups, lr=args.lr, weight_decay=args.weight_decay)

    if args.finetune:
        print(f"Fine-tune scope: {args.finetune_scope}")
        print(f"  trainable params={trainable_count:,} frozen params={frozen_count:,}")
        print(f"  trainable modules: {', '.join(trainable_modules)}")
        if frozen_modules:
            print(f"  frozen modules: {', '.join(frozen_modules)}")
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "group")
        group_params = sum(param.numel() for param in param_group["params"])
        print(f"  Optimizer group {group_name}: lr={param_group['lr']:.2e}, params={group_params:,}")

    grad_params = [param for param in model.parameters() if param.requires_grad]

    def schedule(step: int) -> float:
        step = max(step, 1)
        warmup_steps = args.finetune_warmup_steps if args.finetune else args.warmup_steps
        return args.d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_metric = float("-inf")
    global_step = 0
    start_epoch = 0
    latest_save_path = (
        Path(args.latest_model_path)
        if args.latest_model_path
        else save_path.with_name(f"{save_path.stem}_latest{save_path.suffix}")
    )
    if args.resume:
        if checkpoint is None:
            raise ValueError("--resume requires a checkpoint via --model-path or --init-from")
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as exc:
                print(f"  Resume warning: could not load optimizer_state_dict ({exc}); using fresh optimizer")
        else:
            print("  Resume warning: checkpoint has no optimizer_state_dict")
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError as exc:
                print(f"  Resume warning: could not load scheduler_state_dict ({exc}); using fresh scheduler")
        else:
            print("  Resume warning: checkpoint has no scheduler_state_dict")
        global_step = int(checkpoint.get("global_step", 0) or 0)
        start_epoch = int(checkpoint.get("epoch", -1) or -1) + 1
        best_metric = float(
            checkpoint.get("selection_metric_value", checkpoint.get("event_f1", float("-inf")))
        )
        print(
            f"  Resume state: start_epoch={start_epoch + 1}, "
            f"global_step={global_step}, best_metric={best_metric:.4f}"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if start_epoch >= args.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch}; "
            f"requested --epochs {args.epochs}. Increase --epochs to continue."
        )
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sums = defaultdict(float)
        n_batches = 0
        start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch["features"])
                losses = criterion(out, batch)
                if teacher_model is not None:
                    with torch.no_grad():
                        if args.live_window_distill_weight > 0:
                            teacher_out = teacher_model(batch["teacher_features"])
                        else:
                            teacher_out = teacher_model(batch["features"])
                    if teacher_preserve_enabled:
                        teacher_loss = _teacher_preservation_loss(out, teacher_out, batch, args)
                        losses["teacher"] = teacher_loss
                        losses["total"] = losses["total"] + teacher_loss
                    if args.live_window_distill_weight > 0:
                        distill_losses = _live_window_distillation_loss(out, teacher_out, batch, args)
                        distill_total = distill_losses["total"]
                        losses["live_distill"] = distill_total
                        for distill_key, distill_value in distill_losses.items():
                            if distill_key != "total":
                                losses[f"live_distill_{distill_key}"] = distill_value
                        losses["total"] = losses["total"] + distill_total
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            for key, value in losses.items():
                loss_sums[key] += float(value.detach().cpu())
            n_batches += 1
            if batch_idx % args.log_every == 0:
                avg = loss_sums["total"] / max(n_batches, 1)
                print(f"  epoch {epoch + 1} batch {batch_idx}/{len(train_loader)} loss={avg:.4f}")

        train_losses = {key: value / max(n_batches, 1) for key, value in loss_sums.items()}
        val_results = {}
        for val_name, val_loader in validation_loaders.items():
            max_batches = None
            if val_name == "general" and args.finetune_general_val_batches > 0:
                max_batches = args.finetune_general_val_batches
            elif args.max_val_batches > 0:
                max_batches = args.max_val_batches
            val_results[val_name] = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                args.onset_threshold,
                args.offset_threshold,
                args.frame_threshold,
                max_event_batches=args.max_event_val_batches,
                max_event_samples=args.max_event_val_samples,
                event_sampling=args.event_val_sampling,
                max_score_batches=args.max_score_val_batches,
                max_score_samples=args.max_score_val_samples,
                score_sampling=args.score_val_sampling,
                score_grid_beats=args.score_grid_beats,
                score_onset_slot_tolerance=args.score_onset_slot_tolerance,
                score_duration_class_tolerance=args.score_duration_class_tolerance,
                score_duration_policy=args.score_duration_policy,
                score_duration_lookup_path=args.score_duration_lookup_path,
                max_batches=max_batches,
                decode_use_sounding_frame=args.decode_use_sounding_frame,
            )

        primary_val_name = "hard" if "hard" in val_results else next(iter(val_results))
        val = val_results[primary_val_name]
        if args.save_best_on == "event_f1":
            primary_metric = val["event"]["f1"]
        elif args.save_best_on == "score_f1":
            primary_metric = val["score"]["f1"]
        else:
            primary_metric = val["onset"]["f1"]
        metric = primary_metric
        general_metric = None
        if args.finetune and "general" in val_results:
            general = val_results["general"]
            if args.save_best_on == "event_f1":
                general_metric = general["event"]["f1"]
            elif args.save_best_on == "score_f1":
                general_metric = general["score"]["f1"]
            else:
                general_metric = general["onset"]["f1"]
            weight = float(np.clip(args.finetune_general_val_weight, 0.0, 1.0))
            metric = (1.0 - weight) * primary_metric + weight * general_metric

        elapsed = (time.time() - start) / 60
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({elapsed:.1f} min)")
        print(
            f"  Train total={train_losses['total']:.4f} "
            f"onset={train_losses['onset']:.4f} offset={train_losses['offset']:.4f} "
            f"frame={train_losses['frame']:.4f} sound={train_losses['sounding_frame']:.4f} "
            f"pedal={train_losses['pedal']:.4f} vel={train_losses['velocity']:.4f} "
            f"teacher={train_losses.get('teacher', 0.0):.4f}"
        )
        print(
            "  Weighted train terms "
            f"onset={args.onset_weight * train_losses['onset']:.4f} "
            f"raw_onset={args.raw_onset_weight * train_losses['raw_onset']:.4f} "
            f"offset={args.offset_weight * train_losses['offset']:.4f} "
            f"raw_offset={args.raw_offset_weight * train_losses['raw_offset']:.4f} "
            f"frame={args.frame_weight * train_losses['frame']:.4f} "
            f"sound={args.sounding_frame_weight * train_losses['sounding_frame']:.4f} "
            f"pedal={args.pedal_weight * train_losses['pedal']:.4f} "
            f"vel={args.velocity_weight * train_losses['velocity']:.4f} "
            f"nv={args.nv_weight * train_losses['note_value']:.4f}"
        )
        print(
            f"  {primary_val_name.title()} val loss={val['losses']['total']:.4f} "
            f"onset_f1={val['onset']['f1']:.3f} offset_f1={val['offset']['f1']:.3f} "
            f"sound_f1={val['sounding_frame']['f1']:.3f} pedal_f1={val['pedal']['f1']:.3f}"
        )
        print(
            f"  {primary_val_name.title()} event P={val['event']['precision']:.3f} "
            f"R={val['event']['recall']:.3f} F1={val['event']['f1']:.3f} "
            f"({val['event']['matched']}/{val['event']['predicted']} pred, "
            f"{val['event']['ground_truth']} gt)"
        )
        print(
            f"  {primary_val_name.title()} score P={val['score']['precision']:.3f} "
            f"R={val['score']['recall']:.3f} F1={val['score']['f1']:.3f} "
            f"onsetF1={val['score']['onset_f1']:.3f} "
            f"durAcc={val['score']['duration_accuracy']:.3f} "
            f"samples={val['score']['samples']} policy={args.score_duration_policy}"
        )
        if general_metric is not None:
            general = val_results["general"]
            print(
                f"  General event P={general['event']['precision']:.3f} "
                f"R={general['event']['recall']:.3f} F1={general['event']['f1']:.3f} "
                f"scoreF1={general['score']['f1']:.3f} "
                f"selection={metric:.4f}"
            )

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": _checkpoint_config(args),
            "epoch": epoch,
            "global_step": global_step,
            "train_losses": train_losses,
            "val_losses": val["losses"],
            "val_results": val_results,
            "onset_f1": val["onset"]["f1"],
            "offset_f1": val["offset"]["f1"],
            "event_precision": val["event"]["precision"],
            "event_recall": val["event"]["recall"],
            "event_f1": val["event"]["f1"],
            "score_precision": val["score"]["precision"],
            "score_recall": val["score"]["recall"],
            "score_f1": val["score"]["f1"],
            "selection_metric_name": args.save_best_on,
            "selection_metric_value": max(best_metric, metric),
            "selection_primary_val": primary_val_name,
        }
        if args.save_latest:
            latest_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload, latest_save_path)
            print(f"  Saved latest checkpoint to {latest_save_path}")

        if metric > best_metric:
            best_metric = metric
            checkpoint_payload["selection_metric_value"] = best_metric
            torch.save(checkpoint_payload, save_path)
            print(f"  Saved best model to {save_path} ({args.save_best_on}={best_metric:.4f})")


@torch.no_grad()
def benchmark(args) -> None:
    device = torch.device(args.device)
    model = _build_model_from_args(args).to(device).eval()
    model_path = Path(args.model_path) if args.model_path else MODEL_PATH
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    extractor = MelFeatureExtractor(device=device)
    audio = torch.randn(1, int(SAMPLE_RATE * 10), device=device) * 0.01
    features = extractor.extract(audio)
    for _ in range(3):
        model(features)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    runs = max(1, args.benchmark_runs)
    for _ in range(runs):
        model(features)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Enhanced model: {n_params:,} params, {elapsed_ms:.1f} ms / 10s features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhanced mel piano transcriber")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--latest-model-path", type=str, default=None)
    parser.add_argument("--save-latest", action="store_true", default=True)
    parser.add_argument("--no-save-latest", action="store_false", dest="save_latest")
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--train-segment-manifest", type=str, default=None)
    parser.add_argument("--validation-segment-manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=8000)
    parser.add_argument("--finetune-warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--conv-channels", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ff-expansion", type=int, default=4)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--event-hidden", type=int, default=192)
    parser.add_argument("--event-context", action="store_true", default=True)
    parser.add_argument("--no-event-context", action="store_false", dest="event_context")
    parser.add_argument("--event-residual", action="store_true", default=False)
    parser.add_argument("--cross-key-layers", type=int, default=0)
    parser.add_argument("--cross-key-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter-bottleneck", type=int, default=0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--use-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-checkpoint", action="store_false", dest="use_checkpoint")
    parser.add_argument("--train-augment", action="store_true", default=True)
    parser.add_argument("--no-train-augment", action="store_false", dest="train_augment")
    parser.add_argument("--finetune-scope", choices=["heads", "decoder", "full"], default="decoder")
    parser.add_argument("--finetune-hard-ratio", type=float, default=0.25)
    parser.add_argument("--finetune-samples-per-epoch", type=int, default=0)
    parser.add_argument("--finetune-eval-general", action="store_true", default=True)
    parser.add_argument("--no-finetune-eval-general", action="store_false", dest="finetune_eval_general")
    parser.add_argument("--finetune-general-val-weight", type=float, default=0.50)
    parser.add_argument("--finetune-general-val-batches", type=int, default=80)
    parser.add_argument("--head-lr-scale", type=float, default=1.0)
    parser.add_argument("--decoder-lr-scale", type=float, default=0.35)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--train-window-sec", type=float, default=0.0,
                        help="Crop student training examples to this many seconds of live-style context")
    parser.add_argument("--emit-window-sec", type=float, default=0.0,
                        help="Compute supervised/distillation loss only over the final emit window")
    parser.add_argument("--teacher-from", type=str, default=None)
    parser.add_argument("--teacher-preserve-weight", type=float, default=0.20)
    parser.add_argument("--teacher-onset-weight", type=float, default=1.0)
    parser.add_argument("--teacher-offset-weight", type=float, default=0.5)
    parser.add_argument("--teacher-frame-weight", type=float, default=0.5)
    parser.add_argument("--teacher-velocity-weight", type=float, default=0.1)
    parser.add_argument("--teacher-note-value-weight", type=float, default=0.0)
    parser.add_argument("--live-window-distill-weight", type=float, default=0.0,
                        help="Distill cropped live-window student outputs from a full-context teacher")
    parser.add_argument("--live-distill-temperature", type=float, default=2.0)
    parser.add_argument("--live-distill-onset-weight", type=float, default=1.0)
    parser.add_argument("--live-distill-offset-weight", type=float, default=0.5)
    parser.add_argument("--live-distill-frame-weight", type=float, default=0.5)
    parser.add_argument("--live-distill-sounding-frame-weight", type=float, default=0.25)
    parser.add_argument("--live-distill-pedal-weight", type=float, default=0.0)
    parser.add_argument("--live-distill-velocity-weight", type=float, default=0.1)
    parser.add_argument("--live-distill-note-value-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--onset-weight", type=float, default=1.0)
    parser.add_argument("--raw-onset-weight", type=float, default=0.25)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--raw-offset-weight", type=float, default=0.25)
    parser.add_argument("--frame-weight", type=float, default=0.8)
    parser.add_argument("--sounding-frame-weight", type=float, default=0.3)
    parser.add_argument("--pedal-weight", type=float, default=0.1)
    parser.add_argument("--velocity-weight", type=float, default=0.3)
    parser.add_argument("--nv-weight", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--save-best-on", choices=["event_f1", "score_f1", "onset_f1"], default="event_f1")
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--val-sampling", choices=["spread", "leading"], default="spread")
    parser.add_argument("--max-event-val-batches", type=int, default=20)
    parser.add_argument("--max-event-val-samples", type=int, default=0)
    parser.add_argument("--event-val-sampling", choices=["spread", "leading"], default="spread")
    parser.add_argument("--max-score-val-batches", type=int, default=20)
    parser.add_argument("--max-score-val-samples", type=int, default=0)
    parser.add_argument("--score-val-sampling", choices=["spread", "leading"], default="spread")
    parser.add_argument("--score-grid-beats", type=float, default=SCORE_GRID_BEATS)
    parser.add_argument("--score-onset-slot-tolerance", type=int, default=0)
    parser.add_argument("--score-duration-class-tolerance", type=int, default=0)
    # NOTE: lookup_ioi_head_sound is kept opt-in, NOT defaulted. It improves the
    # offline note_value duration_accuracy metric, but the score renderer
    # (generateMeasureXmls in components/PianoSheetMusic.tsx) discards note_value
    # and prints each note's duration as the per-voice start_beat IOI, so this
    # policy does not reach the app or the gold12/scorediff score. Pass
    # --score-duration-policy lookup_ioi_head_sound --score-duration-lookup-path
    # score_duration_lookup.json to measure it. See live-change-log 2026-06-14.
    parser.add_argument("--score-duration-policy", choices=SCORE_DURATION_POLICIES, default="ioi_same_hand")
    parser.add_argument("--score-duration-lookup-path", type=str, default=None)
    parser.add_argument("--decode-use-sounding-frame", action="store_true", default=True)
    parser.add_argument("--decode-use-physical-frame", action="store_false", dest="decode_use_sounding_frame")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lr is None:
        args.lr = 0.08 if args.finetune else 1.0
    if args.train:
        train(args)
    elif args.benchmark:
        benchmark(args)
    else:
        raise SystemExit("Specify --train or --benchmark")


if __name__ == "__main__":
    main()
