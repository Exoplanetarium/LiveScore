"""Diff offline full-file transcription vs the live chunked+gated path.

Replays a WAV through ContinuousLiveStreamSession (the real live state machine)
and compares the emitted notes against a reference MIDI (e.g. the high-accuracy
offline transcription). Notes present in the reference but missing live are the
notes the live path loses; suppression totals attribute the loss to a gate.

Example:
    python diff_offline_vs_live.py rhythm_training/test_inner_voice.wav \
      --reference-midi rhythm_training/test_inner_voice.mid
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from main import ContinuousLiveStreamSession
from tune_continuous_stream_decoder import (
    notes_from_accumulator,
    override_live_attrs,
    update_accumulator,
)

TARGET_SR = 16000


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def load_reference_notes(path: Path) -> List[Dict]:
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({
                "onset_time": float(n.start),
                "offset_time": float(n.end),
                "midi_note": int(n.pitch),
            })
    notes.sort(key=lambda e: (e["onset_time"], e["midi_note"]))
    return notes


def replay_live(audio: np.ndarray, args: argparse.Namespace) -> Dict:
    if args.tail_padding_sec > 0:
        audio = np.concatenate([audio, np.zeros(int(round(args.tail_padding_sec * TARGET_SR)), dtype=np.float32)])
    session = ContinuousLiveStreamSession(
        session_id=f"diff-{uuid.uuid4().hex}",
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
    suppression: Dict[str, int] = {}

    def absorb(update):
        if update is None:
            return
        update_accumulator(score_payloads, update, include_unstable=False)
        update_accumulator(preview_payloads, update, include_unstable=True)
        inference = update.get("inference") or {}
        if inference.get("ran"):
            cont = inference.get("continuity_filter") or {}
            for key in ("suppressed", "same_pitch_boundary", "implausible_repeat",
                        "harmonic_sustain", "weak_birth_outside_attack", "registered_attack_groups"):
                suppression[key] = suppression.get(key, 0) + int(cont.get(key, 0) or 0)

    for start in range(0, audio.size, packet_frames):
        session.append_audio(audio[start:start + packet_frames])
        absorb(session.maybe_run_inference())
    absorb(session.maybe_run_inference(force=True))

    return {
        "score_notes": notes_from_accumulator(score_payloads),
        "preview_notes": notes_from_accumulator(preview_payloads),
        "suppression": suppression,
    }


def _held_count(notes: Sequence[Dict], onset: float) -> int:
    return sum(1 for n in notes if n["onset_time"] < onset - 1e-3 and n["offset_time"] > onset + 1e-3)


def match_reference(reference: Sequence[Dict], live: Sequence[Dict], onset_tol: float) -> Tuple[List[Dict], List[Dict]]:
    """Greedy pitch+onset match. Returns (matched_refs, missing_refs)."""
    used = [False] * len(live)
    matched: List[Dict] = []
    missing: List[Dict] = []
    for ref in reference:
        best = None
        best_err = None
        for i, ln in enumerate(live):
            if used[i] or ln["midi_note"] != ref["midi_note"]:
                continue
            err = abs(ln["onset_time"] - ref["onset_time"])
            if err <= onset_tol and (best_err is None or err < best_err):
                best, best_err = i, err
        if best is None:
            missing.append(ref)
        else:
            used[best] = True
            matched.append(ref)
    return matched, missing


def report_surface(name: str, reference: Sequence[Dict], live: Sequence[Dict], onset_tol: float,
                   inner_low: int, inner_high: int) -> None:
    matched, missing = match_reference(reference, live, onset_tol)
    inner_missing = [m for m in missing if inner_low <= m["midi_note"] <= inner_high]
    held_missing = [m for m in missing if _held_count(reference, m["onset_time"]) >= 2]
    print(f"\n=== live {name} surface vs reference ===")
    print(f"  reference notes: {len(reference)}   live {name} notes: {len(live)}")
    print(f"  matched: {len(matched)}   missing: {len(missing)}   recall: {len(matched)/max(1,len(reference)):.3f}")
    print(f"  missing in inner band [{inner_low}..{inner_high}]: {len(inner_missing)}")
    print(f"  missing that enter under >=2 held notes (inner-voice-like): {len(held_missing)}")
    if missing:
        print("  sample missing (onset, midi, name, held_under):")
        for m in missing[:15]:
            print(f"    t={m['onset_time']:.2f}  {m['midi_note']} {pretty_midi.note_number_to_name(m['midi_note'])}  held={_held_count(reference, m['onset_time'])}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio")
    p.add_argument("--reference-midi", required=True)
    p.add_argument("--onset-tol", type=float, default=0.20, help="Onset match tolerance (s); live timing drifts so keep generous.")
    p.add_argument("--inner-low", type=int, default=52)
    p.add_argument("--inner-high", type=int, default=76)
    p.add_argument("--packet-ms", type=float, default=40.0)
    p.add_argument("--context-sec", type=float, default=1.8)
    p.add_argument("--inference-interval-ms", type=float, default=70.0)
    p.add_argument("--trusted-delay-ms", type=float, default=180.0)
    p.add_argument("--commit-delay-ms", type=float, default=500.0)
    p.add_argument("--lock-delay-ms", type=float, default=2000.0)
    p.add_argument("--tail-padding-sec", type=float, default=0.3)
    p.add_argument("--relax-gates", action="store_true", help="Disable weak-birth/harmonic gates to isolate their contribution.")
    p.add_argument("--weak-birth-conf", type=float, default=None, help="Override STREAM_WEAK_BIRTH_HIGH_CONFIDENCE.")
    p.add_argument("--harmonic-max-conf", type=float, default=None, help="Override STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    audio = load_audio(Path(args.audio))
    reference = load_reference_notes(Path(args.reference_midi))
    gate_overrides = {}
    if args.relax_gates:
        gate_overrides = {
            "STREAM_WEAK_BIRTH_HIGH_CONFIDENCE": 0.0,
            "STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE": 0.0,
        }
        print("[relax-gates] weak_birth/harmonic suppression disabled for this replay")
    if args.weak_birth_conf is not None:
        gate_overrides["STREAM_WEAK_BIRTH_HIGH_CONFIDENCE"] = args.weak_birth_conf
    if args.harmonic_max_conf is not None:
        gate_overrides["STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE"] = args.harmonic_max_conf
    if gate_overrides:
        print(f"[gate overrides] {gate_overrides}")
    with override_live_attrs(gate_overrides):
        live = replay_live(audio, args)

    print(f"audio: {audio.size / TARGET_SR:.2f}s   reference notes: {len(reference)}")
    print(f"live score notes: {len(live['score_notes'])}   live preview notes: {len(live['preview_notes'])}")
    print("\nlive continuity-gate suppression totals (whole replay):")
    for k, v in sorted(live["suppression"].items()):
        print(f"  {k}: {v}")

    report_surface("score", reference, live["score_notes"], args.onset_tol, args.inner_low, args.inner_high)
    report_surface("preview", reference, live["preview_notes"], args.onset_tol, args.inner_low, args.inner_high)


if __name__ == "__main__":
    main()
