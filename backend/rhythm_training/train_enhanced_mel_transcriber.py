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
      --frame-weight 0.8 --velocity-weight 0.3 --nv-weight 0.1

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
from torch.utils.data import DataLoader, Dataset

from train_mel_baseline import (
    FEATURES_DIR,
    HOP_LENGTH,
    INDEX_DIR,
    MIDI_OFFSET,
    N_FFT,
    N_MELS,
    NOTE_VALUE_BEATS,
    NOTE_VALUE_CLASSES,
    NOTE_VALUE_NAMES,
    ONSET_TENT_SEC,
    PIANO_KEYS,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    ConformerBlock,
    MelFeatureExtractor,
    ResidualAdapter,
    SinusoidalPositionalEncoding,
)


MODEL_PATH = Path(__file__).parent / "enhanced_mel_transcription.pt"
OFFSET_TENT_SEC = 0.05


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


@lru_cache(maxsize=4096)
def _load_segment_targets(
    midi_path: str,
    start_sec: float,
    n_frames: int,
    sr: int,
    hop_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    import pretty_midi

    frame_time = hop_length / sr
    end_sec = start_sec + SEGMENT_SECONDS
    offset = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
    gt_events: List[Dict] = []

    midi = pretty_midi.PrettyMIDI(midi_path)
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.end < start_sec or note.start > end_sec:
                continue
            key = note.pitch - MIDI_OFFSET
            if key < 0 or key >= PIANO_KEYS:
                continue

            onset_rel = note.start - start_sec
            offset_rel = note.end - start_sec
            _apply_tent(offset, offset_rel, key, OFFSET_TENT_SEC, frame_time)

            if 0.0 <= onset_rel < SEGMENT_SECONDS:
                gt_events.append({
                    "onset_time": float(onset_rel),
                    "offset_time": float(max(offset_rel, onset_rel + frame_time)),
                    "midi_note": int(note.pitch),
                    "velocity": int(note.velocity),
                })

    offset_t = torch.from_numpy(offset)
    # Reserved for future pedal/sustain target without changing the dataset API.
    pedal_t = torch.zeros(n_frames, dtype=torch.float32)
    return offset_t, pedal_t, gt_events


class EnhancedPrecomputedMelDataset(Dataset):
    """Precomputed mel features plus MIDI-derived offset/event targets."""

    def __init__(
        self,
        split: str,
        augment: bool = False,
        segment_ids: Optional[Sequence[int]] = None,
    ):
        self.split = split
        self.split_dir = FEATURES_DIR / split
        self.augment = augment

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
        print(f"[EnhancedDataset] {split}: {len(self.files)} segments{aug}")

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

        offset, pedal, gt_events = _load_segment_targets(
            midi_path,
            start_sec,
            n_frames,
            SAMPLE_RATE,
            HOP_LENGTH,
        )

        if self.augment:
            features, onset, offset, frame, velocity, note_value = self._augment(
                features, onset, offset, frame, velocity, note_value
            )

        return {
            "features": features,
            "onset": onset,
            "offset": offset,
            "frame": frame,
            "velocity": velocity,
            "note_value": note_value,
            "pedal": pedal,
            "bpm": bpm.detach().clone() if torch.is_tensor(bpm) else torch.tensor(float(bpm)),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
            "midi_path": midi_path,
            "start_sec": torch.tensor(start_sec, dtype=torch.float32),
            "gt_events": gt_events,
        }

    @staticmethod
    def _augment(
        features: torch.Tensor,
        onset: torch.Tensor,
        offset: torch.Tensor,
        frame: torch.Tensor,
        velocity: torch.Tensor,
        note_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            velocity = shift_keys(velocity)
            note_value = shift_keys(note_value, fill_value=0)

        return features, onset, offset, frame, velocity, note_value


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
        adapter_bottleneck: int = 0,
        adapter_dropout: float = 0.0,
        n_note_value_classes: int = NOTE_VALUE_CLASSES,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.n_keys = PIANO_KEYS
        self.n_nv = n_note_value_classes
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

        self.frame_head = nn.Sequential(
            nn.Linear(key_dim + 2, key_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(key_dim, 1)
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
        onset_logits = refined[:, :, 0, :]
        offset_logits = refined[:, :, 1, :]

        frame_in = torch.cat([
            key_h,
            onset_logits.unsqueeze(-1).detach(),
            offset_logits.unsqueeze(-1).detach(),
        ], dim=-1)
        frame_logits = self.frame_head(frame_in).squeeze(-1)
        note_value_logits = self.note_value_head(key_h)

        return {
            "onset_logits": onset_logits,
            "raw_onset_logits": raw_onset_logits,
            "offset_logits": offset_logits,
            "raw_offset_logits": raw_offset_logits,
            "frame_logits": frame_logits,
            "velocity": velocity,
            "note_value_logits": note_value_logits,
        }


class EnhancedTranscriptionLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 4.0,
        onset_weight: float = 1.0,
        offset_weight: float = 1.0,
        frame_weight: float = 0.8,
        velocity_weight: float = 0.3,
        nv_weight: float = 0.1,
        focal_gamma: float = 1.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.onset_weight = onset_weight
        self.offset_weight = offset_weight
        self.frame_weight = frame_weight
        self.velocity_weight = velocity_weight
        self.nv_weight = nv_weight
        self.focal_gamma = focal_gamma

    @staticmethod
    def _event_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        sample_weight = 1.0 + (pos_weight - 1.0) * target
        return (bce * sample_weight).mean()

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        onset_gt = batch["onset"]
        offset_gt = batch["offset"]
        frame_gt = batch["frame"]
        vel_gt = batch["velocity"]
        nv_gt = batch["note_value"]

        onset_loss = self._event_loss(out["onset_logits"], onset_gt, self.pos_weight)
        raw_onset_loss = self._event_loss(out["raw_onset_logits"], onset_gt, self.pos_weight)
        offset_loss = self._event_loss(out["offset_logits"], offset_gt, self.pos_weight)
        raw_offset_loss = self._event_loss(out["raw_offset_logits"], offset_gt, self.pos_weight)

        frame_prob = torch.sigmoid(out["frame_logits"])
        p_t = torch.where(frame_gt > 0.5, frame_prob, 1.0 - frame_prob)
        focal = (1.0 - p_t.detach()).pow(self.focal_gamma)
        frame_sample_weight = torch.where(
            frame_gt > 0.5,
            torch.full_like(frame_gt, self.pos_weight),
            torch.ones_like(frame_gt),
        )
        frame_bce = F.binary_cross_entropy_with_logits(out["frame_logits"], frame_gt, reduction="none")
        frame_loss = (focal * frame_bce * frame_sample_weight).mean()

        active = frame_gt > 0.5
        velocity_loss = (
            F.mse_loss(out["velocity"][active], vel_gt[active])
            if active.any()
            else torch.tensor(0.0, device=frame_gt.device)
        )

        onset_mask = onset_gt > 0.5
        if onset_mask.any() and self.nv_weight > 0:
            nv_loss = F.cross_entropy(out["note_value_logits"][onset_mask], nv_gt[onset_mask])
        else:
            nv_loss = torch.tensor(0.0, device=frame_gt.device)

        total = (
            self.onset_weight * onset_loss
            + 0.25 * raw_onset_loss
            + self.offset_weight * offset_loss
            + 0.25 * raw_offset_loss
            + self.frame_weight * frame_loss
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
    sr: int = SAMPLE_RATE,
    hop: int = HOP_LENGTH,
) -> List[Dict]:
    frame_time = hop / sr
    min_frames = max(1, int(round(min_note_duration / frame_time)))
    n_frames = onset_probs.shape[0]
    events: List[Dict] = []

    for key in range(PIANO_KEYS):
        onset_frames = _peak_frames(onset_probs[:, key], onset_threshold)
        offset_frames = _peak_frames(offset_probs[:, key], offset_threshold)
        for onset_f in onset_frames:
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
            if vel_int < min_velocity:
                continue

            event = {
                "onset_time": float(onset_f * frame_time),
                "offset_time": float(offset_f * frame_time),
                "midi_note": int(key + MIDI_OFFSET),
                "velocity": vel_int,
                "onset_prob": float(onset_probs[onset_f, key]),
                "offset_prob": float(offset_probs[min(offset_f, n_frames - 1), key]),
            }
            if note_value_probs is not None:
                pooled = note_value_probs[onset_f:offset_f, key, :].mean(axis=0)
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
        "features", "onset", "offset", "frame", "velocity", "note_value",
        "pedal", "bpm", "segment_id", "start_sec",
    ]
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
) -> Dict:
    model.eval()
    loss_sums = defaultdict(float)
    n_batches = 0
    onset_counts = [0, 0, 0]
    offset_counts = [0, 0, 0]
    event_totals = defaultdict(int)

    for batch_idx, batch in enumerate(loader):
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

        if batch_idx < max_event_batches:
            onset_np = torch.sigmoid(out["onset_logits"]).cpu().numpy()
            offset_np = torch.sigmoid(out["offset_logits"]).cpu().numpy()
            frame_np = torch.sigmoid(out["frame_logits"]).cpu().numpy()
            vel_np = out["velocity"].cpu().numpy()
            nv_np = F.softmax(out["note_value_logits"], dim=-1).cpu().numpy()
            for sample_idx in range(onset_np.shape[0]):
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
                metrics = match_note_events(pred_events, batch["gt_events"][sample_idx])
                event_totals["matched"] += int(metrics["matched"])
                event_totals["predicted"] += int(metrics["predicted"])
                event_totals["ground_truth"] += int(metrics["ground_truth"])

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
    return {
        "losses": {key: value / max(n_batches, 1) for key, value in loss_sums.items()},
        "onset": counts_to_metrics(onset_counts),
        "offset": counts_to_metrics(offset_counts),
        "event": {
            "precision": event_p,
            "recall": event_r,
            "f1": event_f1,
            **dict(event_totals),
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
        adapter_bottleneck=int(config.get("adapter_bottleneck", 0)),
        adapter_dropout=float(config.get("adapter_dropout", 0.0)),
        n_note_value_classes=int(config.get("n_note_value_classes", NOTE_VALUE_CLASSES)),
        use_checkpoint=bool(config.get("use_checkpoint", False)),
    )


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
        "conv_channels": args.conv_channels,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "ff_expansion": args.ff_expansion,
        "conv_kernel": args.conv_kernel,
        "dropout": args.dropout,
        "event_hidden": args.event_hidden,
        "adapter_bottleneck": args.adapter_bottleneck,
        "adapter_dropout": args.adapter_dropout,
        "explicit_offset_head": True,
        "pitch_local_readout": True,
        "save_best_on": args.save_best_on,
    }


def train(args) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dataset = EnhancedPrecomputedMelDataset("train", augment=args.train_augment)
    val_dataset = EnhancedPrecomputedMelDataset("validation", augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=enhanced_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=enhanced_collate,
    )

    model = _build_model_from_args(args).to(device)
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Initialized from {args.init_from}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = EnhancedTranscriptionLoss(
        pos_weight=args.pos_weight,
        onset_weight=args.onset_weight,
        offset_weight=args.offset_weight,
        frame_weight=args.frame_weight,
        velocity_weight=args.velocity_weight,
        nv_weight=args.nv_weight,
        focal_gamma=args.focal_gamma,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def schedule(step: int) -> float:
        step = max(step, 1)
        return args.d_model ** -0.5 * min(step ** -0.5, step * args.warmup_steps ** -1.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_metric = float("-inf")
    global_step = 0
    save_path = Path(args.model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
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
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        val = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            args.onset_threshold,
            args.offset_threshold,
            args.frame_threshold,
            max_event_batches=args.max_event_val_batches,
        )

        metric = val["event"]["f1"] if args.save_best_on == "event_f1" else val["onset"]["f1"]
        elapsed = (time.time() - start) / 60
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({elapsed:.1f} min)")
        print(
            f"  Train total={train_losses['total']:.4f} "
            f"onset={train_losses['onset']:.4f} offset={train_losses['offset']:.4f} "
            f"frame={train_losses['frame']:.4f} vel={train_losses['velocity']:.4f}"
        )
        print(
            f"  Val loss={val['losses']['total']:.4f} "
            f"onset_f1={val['onset']['f1']:.3f} offset_f1={val['offset']['f1']:.3f}"
        )
        print(
            f"  Event P={val['event']['precision']:.3f} "
            f"R={val['event']['recall']:.3f} F1={val['event']['f1']:.3f} "
            f"({val['event']['matched']}/{val['event']['predicted']} pred, "
            f"{val['event']['ground_truth']} gt)"
        )

        if metric > best_metric:
            best_metric = metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": _checkpoint_config(args),
                "epoch": epoch,
                "global_step": global_step,
                "train_losses": train_losses,
                "val_losses": val["losses"],
                "onset_f1": val["onset"]["f1"],
                "offset_f1": val["offset"]["f1"],
                "event_precision": val["event"]["precision"],
                "event_recall": val["event"]["recall"],
                "event_f1": val["event"]["f1"],
                "selection_metric_name": args.save_best_on,
                "selection_metric_value": best_metric,
            }, save_path)
            print(f"  Saved best model to {save_path} ({args.save_best_on}={best_metric:.4f})")


@torch.no_grad()
def benchmark(args) -> None:
    device = torch.device(args.device)
    model = _build_model_from_args(args).to(device).eval()
    if args.model_path and Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
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
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=8000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--conv-channels", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ff-expansion", type=int, default=4)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--event-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter-bottleneck", type=int, default=0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--use-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-checkpoint", action="store_false", dest="use_checkpoint")
    parser.add_argument("--train-augment", action="store_true", default=True)
    parser.add_argument("--no-train-augment", action="store_false", dest="train_augment")
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--onset-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--frame-weight", type=float, default=0.8)
    parser.add_argument("--velocity-weight", type=float, default=0.3)
    parser.add_argument("--nv-weight", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--save-best-on", choices=["event_f1", "onset_f1"], default="event_f1")
    parser.add_argument("--max-event-val-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        train(args)
    elif args.benchmark:
        benchmark(args)
    else:
        raise SystemExit("Specify --train or --benchmark")


if __name__ == "__main__":
    main()
