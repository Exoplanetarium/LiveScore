"""Transcribe a WAV file to MIDI with the enhanced-mel model (offline, full context).

This is the high-accuracy offline path (whole-file inference, not the chunked
live decoder). Useful for sanity-checking what the model actually detects on a
recording -- e.g. whether quiet inner voices are present.

Example:
    python transcribe_wav_to_midi.py rhythm_training/test_inner_voice.wav \
      --onset-threshold 0.6 --out rhythm_training/test_inner_voice.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from gpu_ops import get_gpu_enhanced_mel_transcriber


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to the input WAV file.")
    parser.add_argument("--out", default=None, help="Output MIDI path (default: alongside input).")
    parser.add_argument("--onset-threshold", type=float, default=0.6)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--offset-threshold", type=float, default=0.35)
    parser.add_argument("--min-velocity", type=int, default=8)
    parser.add_argument("--inner-low", type=int, default=52, help="Low MIDI bound of the inner-voice band to report.")
    parser.add_argument("--inner-high", type=int, default=76, help="High MIDI bound of the inner-voice band to report.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audio_path = Path(args.audio)
    out_path = Path(args.out) if args.out else audio_path.with_suffix(".mid")

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    transcriber = get_gpu_enhanced_mel_transcriber()
    if transcriber is None or not transcriber.initialized:
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")

    result = transcriber.transcribe(
        audio,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        offset_threshold=args.offset_threshold,
        min_velocity=args.min_velocity,
    )
    events = result.get("est_note_events", [])

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for ev in events:
        start = float(ev["onset_time"])
        end = max(start + 0.03, float(ev.get("offset_time", start)))
        inst.notes.append(pretty_midi.Note(
            velocity=int(np.clip(int(ev.get("velocity", 64)), 1, 127)),
            pitch=int(ev["midi_note"]),
            start=start,
            end=end,
        ))
    pm.instruments.append(inst)
    pm.write(str(out_path))

    pitches = sorted(int(ev["midi_note"]) for ev in events)
    inner = [p for p in pitches if args.inner_low <= p <= args.inner_high]
    print(f"wrote {out_path}")
    print(f"  notes: {len(events)}")
    if pitches:
        lo, hi = min(pitches), max(pitches)
        print(f"  pitch range: {lo} ({pretty_midi.note_number_to_name(lo)}) .. {hi} ({pretty_midi.note_number_to_name(hi)})")
    print(f"  notes in inner band [{args.inner_low}..{args.inner_high}]: {len(inner)}")

    # Overlap report: notes that begin while >=2 other notes are already sounding
    # are likely inner/added voices under held outer notes.
    notes = sorted(
        ((float(e["onset_time"]), float(e.get("offset_time", e["onset_time"])), int(e["midi_note"])) for e in events),
        key=lambda x: x[0],
    )
    overlapped = 0
    for onset, _, _ in notes:
        sounding = sum(1 for s, e, _ in notes if s < onset - 1e-3 and e > onset + 1e-3)
        if sounding >= 2:
            overlapped += 1
    print(f"  onsets entering under >=2 held notes (inner-voice-like): {overlapped}")


if __name__ == "__main__":
    main()
