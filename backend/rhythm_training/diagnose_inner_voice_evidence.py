"""Inspect enhanced-mel evidence for missing quiet inner voices.

This diagnostic does not modify transcription. It runs the enhanced mel model,
decodes notes with the normal thresholds, then writes evidence tables for:

  - decoded events;
  - expected notes supplied as midi@time_sec;
  - below-threshold per-pitch onset/frame candidates.

Examples:

    python diagnose_inner_voice_evidence.py --audio path/to/passage.wav

    python diagnose_inner_voice_evidence.py --audio path/to/passage.wav \
      --start-sec 12.5 --duration-sec 8 \
      --expected 64@0.82,67@1.31
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from train_enhanced_mel_transcriber import (
    EnhancedPrecomputedMelDataset,
    HOP_LENGTH,
    MIDI_OFFSET,
    MODEL_PATH,
    N_FFT,
    N_MELS,
    PIANO_KEYS,
    SAMPLE_RATE,
    _build_model_from_config,
    _peak_frames,
    decode_enhanced_note_events,
)
from train_mel_baseline import MelFeatureExtractor


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_name(midi_note: int) -> str:
    return f"{NOTE_NAMES[int(midi_note) % 12]}{int(midi_note) // 12 - 1}"


def _load_audio_excerpt(
    path: Path,
    target_sr: int,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    info = sf.info(str(path))
    source_sr = int(info.samplerate)
    start_frame = max(0, int(round(start_sec * source_sr)))
    frames = -1
    if duration_sec is not None and duration_sec > 0:
        frames = max(1, int(round(duration_sec * source_sr)))
    audio, sr = sf.read(
        str(path),
        start=start_frame,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    if int(sr) != int(target_sr):
        gcd = math.gcd(int(sr), int(target_sr))
        mono = resample_poly(mono, target_sr // gcd, int(sr) // gcd).astype(np.float32, copy=False)
    return mono, target_sr


def _parse_expected(value: str) -> List[Dict]:
    expected: List[Dict] = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "@" not in item:
            raise ValueError(f"Expected note must be midi@time_sec, got: {item}")
        midi_raw, time_raw = item.split("@", 1)
        midi_note = int(midi_raw.strip())
        time_sec = float(time_raw.strip())
        expected.append({
            "midi_note": midi_note,
            "note_name": _note_name(midi_note),
            "time_sec": time_sec,
        })
    return expected


def _chunked_probs(
    model: torch.nn.Module,
    extractor: MelFeatureExtractor,
    audio: np.ndarray,
    config: Dict,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    sr = int(config.get("sample_rate", SAMPLE_RATE))
    hop = int(config.get("hop_length", HOP_LENGTH))
    n_keys = int(config.get("n_keys", PIANO_KEYS))
    n_note_value_classes = int(config.get("n_note_value_classes", 12))

    audio_t = torch.from_numpy(audio).float().to(device)
    features = extractor.extract(audio_t)
    n_frames = int(features.size(1))
    chunk_frames = int(10.0 * sr / hop)
    overlap = max(1, chunk_frames // 4)
    step = max(1, chunk_frames - overlap)

    all_onset = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_offset = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_frame = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_velocity = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_note_value = np.zeros((n_frames, n_keys, n_note_value_classes), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)

    for start in range(0, n_frames, step):
        end = min(start + chunk_frames, n_frames)
        out = model(features[:, start:end, :])

        onset_p = torch.sigmoid(out["onset_logits"][0]).float().cpu().numpy()
        offset_p = torch.sigmoid(out["offset_logits"][0]).float().cpu().numpy()
        frame_p = torch.sigmoid(out["frame_logits"][0]).float().cpu().numpy()
        velocity = out["velocity"][0].float().cpu().numpy()
        nv_probs = F.softmax(out["note_value_logits"][0].float(), dim=-1).cpu().numpy()

        if nv_probs.shape[-1] < n_note_value_classes:
            pad = np.zeros(
                (*nv_probs.shape[:-1], n_note_value_classes - nv_probs.shape[-1]),
                dtype=np.float32,
            )
            nv_probs = np.concatenate([nv_probs, pad], axis=-1)
        elif nv_probs.shape[-1] > n_note_value_classes:
            nv_probs = nv_probs[..., :n_note_value_classes]

        actual_len = end - start
        all_onset[start:end] += onset_p[:actual_len]
        all_offset[start:end] += offset_p[:actual_len]
        all_frame[start:end] += frame_p[:actual_len]
        all_velocity[start:end] += velocity[:actual_len]
        all_note_value[start:end] += nv_probs[:actual_len]
        counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    return {
        "onset": all_onset / counts[:, None],
        "offset": all_offset / counts[:, None],
        "frame": all_frame / counts[:, None],
        "velocity": all_velocity / counts[:, None],
        "note_value": all_note_value / counts[:, None, None],
    }


def _chunked_probs_from_features(
    model: torch.nn.Module,
    features: torch.Tensor,
    config: Dict,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    sr = int(config.get("sample_rate", SAMPLE_RATE))
    hop = int(config.get("hop_length", HOP_LENGTH))
    n_frames = int(features.size(0))
    n_keys = int(config.get("n_keys", PIANO_KEYS))
    n_note_value_classes = int(config.get("n_note_value_classes", 12))
    chunk_frames = int(10.0 * sr / hop)
    overlap = max(1, chunk_frames // 4)
    step = max(1, chunk_frames - overlap)

    all_onset = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_offset = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_frame = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_velocity = np.zeros((n_frames, n_keys), dtype=np.float32)
    all_note_value = np.zeros((n_frames, n_keys, n_note_value_classes), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)
    features = features.to(device).unsqueeze(0)

    for start in range(0, n_frames, step):
        end = min(start + chunk_frames, n_frames)
        out = model(features[:, start:end, :])
        onset_p = torch.sigmoid(out["onset_logits"][0]).float().cpu().numpy()
        offset_p = torch.sigmoid(out["offset_logits"][0]).float().cpu().numpy()
        frame_p = torch.sigmoid(out["frame_logits"][0]).float().cpu().numpy()
        velocity = out["velocity"][0].float().cpu().numpy()
        nv_probs = F.softmax(out["note_value_logits"][0].float(), dim=-1).cpu().numpy()

        if nv_probs.shape[-1] < n_note_value_classes:
            pad = np.zeros(
                (*nv_probs.shape[:-1], n_note_value_classes - nv_probs.shape[-1]),
                dtype=np.float32,
            )
            nv_probs = np.concatenate([nv_probs, pad], axis=-1)
        elif nv_probs.shape[-1] > n_note_value_classes:
            nv_probs = nv_probs[..., :n_note_value_classes]

        actual_len = end - start
        all_onset[start:end] += onset_p[:actual_len]
        all_offset[start:end] += offset_p[:actual_len]
        all_frame[start:end] += frame_p[:actual_len]
        all_velocity[start:end] += velocity[:actual_len]
        all_note_value[start:end] += nv_probs[:actual_len]
        counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    return {
        "onset": all_onset / counts[:, None],
        "offset": all_offset / counts[:, None],
        "frame": all_frame / counts[:, None],
        "velocity": all_velocity / counts[:, None],
        "note_value": all_note_value / counts[:, None, None],
    }


def _event_match(events: Sequence[Dict], midi_note: int, time_sec: float, tolerance_sec: float) -> Optional[Dict]:
    best = None
    best_error = None
    for event in events:
        if int(event.get("midi_note", -1)) != int(midi_note):
            continue
        error = abs(float(event.get("onset_time", 0.0)) - float(time_sec))
        if error <= tolerance_sec and (best_error is None or error < best_error):
            best = event
            best_error = error
    return best


def _window_stats(
    probs: Dict[str, np.ndarray],
    midi_note: int,
    center_sec: float,
    sr: int,
    hop: int,
    radius_sec: float,
    onset_threshold: float,
    frame_threshold: float,
    decoded: bool,
) -> Dict:
    key = int(midi_note) - MIDI_OFFSET
    n_frames = probs["onset"].shape[0]
    center_frame = int(round(float(center_sec) * sr / hop))
    radius_frames = max(1, int(round(radius_sec * sr / hop)))
    start = max(0, center_frame - radius_frames)
    end = min(n_frames, center_frame + radius_frames + 1)

    onset_slice = probs["onset"][start:end, key]
    frame_slice = probs["frame"][start:end, key]
    velocity_slice = probs["velocity"][start:end, key]
    local_idx = int(np.argmax(onset_slice)) if onset_slice.size else 0
    peak_frame = start + local_idx
    peak_time = peak_frame * hop / sr
    onset_peak = float(probs["onset"][peak_frame, key])
    frame_peak = float(np.max(frame_slice)) if frame_slice.size else 0.0
    velocity_peak = float(np.max(velocity_slice)) if velocity_slice.size else 0.0
    velocity_int = int(np.clip(round(velocity_peak * 127), 1, 127))

    if decoded:
        diagnosis = "decoded"
    elif onset_peak >= onset_threshold and frame_peak >= frame_threshold:
        diagnosis = "strong_evidence_not_decoded"
    elif onset_peak >= onset_threshold:
        diagnosis = "onset_only_no_frame"
    elif frame_peak >= frame_threshold and onset_peak >= onset_threshold * 0.6:
        diagnosis = "frame_present_weak_onset"
    elif frame_peak >= frame_threshold:
        diagnosis = "frame_present_no_onset"
    elif onset_peak >= onset_threshold * 0.6:
        diagnosis = "weak_onset_no_frame"
    else:
        diagnosis = "absent_or_buried"

    return {
        "midi_note": int(midi_note),
        "note_name": _note_name(midi_note),
        "query_time_sec": round(float(center_sec), 4),
        "peak_time_sec": round(float(peak_time), 4),
        "time_error_sec": round(float(peak_time - center_sec), 4),
        "onset_peak": round(onset_peak, 4),
        "frame_peak": round(frame_peak, 4),
        "velocity_peak": round(velocity_peak, 4),
        "velocity_int": velocity_int,
        "decoded": bool(decoded),
        "diagnosis": diagnosis,
    }


def _near_miss_candidates(
    probs: Dict[str, np.ndarray],
    decoded_events: Sequence[Dict],
    sr: int,
    hop: int,
    onset_threshold: float,
    near_miss_onset: float,
    frame_threshold: float,
    max_candidates: int,
) -> List[Dict]:
    candidates: List[Dict] = []
    decoded_lookup = [
        (int(event["midi_note"]), float(event["onset_time"]))
        for event in decoded_events
    ]
    for key in range(probs["onset"].shape[1]):
        midi_note = key + MIDI_OFFSET
        peak_frames = _peak_frames(probs["onset"][:, key], near_miss_onset)
        for frame_idx in peak_frames.tolist():
            onset_time = frame_idx * hop / sr
            if any(midi == midi_note and abs(time_sec - onset_time) <= 0.05 for midi, time_sec in decoded_lookup):
                continue
            start = max(0, int(frame_idx) - 2)
            end = min(probs["onset"].shape[0], int(frame_idx) + 5)
            onset_peak = float(probs["onset"][frame_idx, key])
            frame_peak = float(np.max(probs["frame"][start:end, key]))
            velocity_peak = float(np.max(probs["velocity"][start:end, key]))
            if onset_peak < onset_threshold and frame_peak < frame_threshold:
                kind = "weak_onset_weak_frame"
            elif onset_peak < onset_threshold:
                kind = "weak_onset_frame_present"
            else:
                kind = "above_onset_not_decoded"
            score = onset_peak + 0.7 * frame_peak + 0.25 * velocity_peak
            candidates.append({
                "midi_note": midi_note,
                "note_name": _note_name(midi_note),
                "time_sec": round(float(onset_time), 4),
                "onset_peak": round(onset_peak, 4),
                "frame_peak": round(frame_peak, 4),
                "velocity_peak": round(velocity_peak, 4),
                "velocity_int": int(np.clip(round(velocity_peak * 127), 1, 127)),
                "candidate_type": kind,
                "score": round(float(score), 4),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_candidates]


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model_path = Path(args.model_path) if args.model_path else MODEL_PATH
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    sr = int(config.get("sample_rate", SAMPLE_RATE))
    hop = int(config.get("hop_length", HOP_LENGTH))

    model = _build_model_from_config(config).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    audio_path = Path(args.audio) if args.audio else None
    piece_info = None
    segment_info = None
    audio_duration_sec = None
    if args.segment_id is not None:
        dataset = EnhancedPrecomputedMelDataset(args.split, augment=False, segment_ids=[int(args.segment_id)])
        sample = dataset[0]
        probs = _chunked_probs_from_features(model, sample["features"], config, device)
        segment_info = dataset.segments[int(args.segment_id)]
        piece_info = dataset.pieces[int(segment_info["piece_idx"])]
        audio_path = Path(piece_info["audio"])
        audio_duration_sec = probs["onset"].shape[0] * hop / float(sr)
    else:
        if audio_path is None:
            raise ValueError("--audio is required unless --segment-id is provided")
        extractor = MelFeatureExtractor(
            sr=sr,
            hop_length=hop,
            n_fft=int(config.get("n_fft", N_FFT)),
            n_mels=int(config.get("n_mels", N_MELS)),
            device=device,
        )
        audio, _ = _load_audio_excerpt(audio_path, sr, args.start_sec, args.duration_sec)
        probs = _chunked_probs(model, extractor, audio, config, device)
        audio_duration_sec = len(audio) / float(sr)
    decoded_events = decode_enhanced_note_events(
        probs["onset"],
        probs["offset"],
        probs["frame"],
        probs["velocity"],
        probs["note_value"],
        onset_threshold=args.onset_threshold,
        offset_threshold=args.offset_threshold,
        frame_threshold=args.frame_threshold,
        min_velocity=args.min_velocity,
        duplicate_window_sec=args.duplicate_window_sec,
        merge_gap_sec=args.merge_gap_sec,
        sr=sr,
        hop=hop,
    )

    expected = _parse_expected(args.expected)
    expected_rows = []
    for item in expected:
        match = _event_match(decoded_events, item["midi_note"], item["time_sec"], args.match_tolerance_sec)
        row = _window_stats(
            probs,
            item["midi_note"],
            item["time_sec"],
            sr,
            hop,
            args.expected_radius_sec,
            args.onset_threshold,
            args.frame_threshold,
            decoded=match is not None,
        )
        if match is not None:
            row["matched_onset_sec"] = round(float(match["onset_time"]), 4)
            row["matched_offset_sec"] = round(float(match["offset_time"]), 4)
            row["matched_onset_prob"] = round(float(match.get("onset_prob", 0.0)), 4)
        expected_rows.append(row)

    near_misses = _near_miss_candidates(
        probs,
        decoded_events,
        sr,
        hop,
        args.onset_threshold,
        args.near_miss_onset,
        args.frame_threshold,
        args.max_candidates,
    )

    decoded_rows = []
    for event in decoded_events:
        decoded_rows.append({
            "onset_time": round(float(event["onset_time"]), 4),
            "offset_time": round(float(event["offset_time"]), 4),
            "duration": round(float(event["offset_time"]) - float(event["onset_time"]), 4),
            "midi_note": int(event["midi_note"]),
            "note_name": _note_name(int(event["midi_note"])),
            "velocity": int(event.get("velocity", 0)),
            "onset_prob": round(float(event.get("onset_prob", 0.0)), 4),
            "offset_prob": round(float(event.get("offset_prob", 0.0)), 4),
            "decode_source": str(event.get("decode_source", "")),
        })

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "audio": str(audio_path),
        "model_path": str(model_path),
        "device": str(device),
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
        "segment_id": args.segment_id,
        "split": args.split,
        "segment_info": segment_info,
        "piece_info": piece_info,
        "audio_duration_sec": audio_duration_sec,
        "sample_rate": sr,
        "hop_length": hop,
        "frame_time_sec": hop / float(sr),
        "thresholds": {
            "onset": args.onset_threshold,
            "offset": args.offset_threshold,
            "frame": args.frame_threshold,
            "near_miss_onset": args.near_miss_onset,
            "min_velocity": args.min_velocity,
        },
        "loaded_missing_keys": len(missing),
        "loaded_unexpected_keys": len(unexpected),
        "decoded_event_count": len(decoded_events),
        "expected_count": len(expected_rows),
        "near_miss_count": len(near_misses),
        "expected": expected_rows,
        "near_misses": near_misses,
    }

    (out_dir / "inner_voice_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "inner_voice_expected.csv", expected_rows)
    _write_csv(out_dir / "inner_voice_near_misses.csv", near_misses)
    _write_csv(out_dir / "inner_voice_decoded_events.csv", decoded_rows)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default=None, help="Path to WAV/FLAC/etc audio.")
    parser.add_argument("--segment-id", type=int, default=None, help="Use precomputed MAESTRO segment features.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Enhanced mel checkpoint.")
    parser.add_argument("--output-dir", default="inner_voice_diagnostics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--expected", default="", help="Comma-separated midi@time_sec entries in excerpt time.")
    parser.add_argument("--expected-radius-sec", type=float, default=0.12)
    parser.add_argument("--match-tolerance-sec", type=float, default=0.06)
    parser.add_argument("--onset-threshold", type=float, default=0.75)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--min-velocity", type=int, default=8)
    parser.add_argument("--duplicate-window-sec", type=float, default=0.04)
    parser.add_argument("--merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--near-miss-onset", type=float, default=0.30)
    parser.add_argument("--max-candidates", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(args)
    print(json.dumps({
        "decoded_event_count": summary["decoded_event_count"],
        "expected_count": summary["expected_count"],
        "near_miss_count": summary["near_miss_count"],
        "output": str(Path(args.output_dir)),
    }, indent=2))


if __name__ == "__main__":
    main()
