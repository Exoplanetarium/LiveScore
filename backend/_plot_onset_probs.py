"""Plot the model's onset-probability surface for a window of frames from a
MAESTRO test clip.

This reproduces the exact OFFLINE inference path used by
``GpuEnhancedMelTranscriber.transcribe`` (10s windowed chunks with overlap-stitch
averaging) to recover the per-frame, per-key ``onset_probs`` array -- the same
array that ``decode_enhanced_note_events`` consumes -- and visualises it for a
selected frame range. Peaks are marked with the real ``_peak_frames`` picker so
the dots are exactly the onsets the decoder would emit at the given threshold.

Two panels:
  (A) heatmap  : onset prob over (time x pitch) for the window, picked onsets dotted
  (B) line plot: onset prob vs time for the few most active keys in the window,
                 with the threshold line and picked peaks (stars) -- this is the
                 "local maximum above threshold" picture in 1-D.

Usage (run from the backend/ directory):
  python _plot_onset_probs.py                         # first test piece, 4s @ 10s in
  python _plot_onset_probs.py --piece-index 3 --start-sec 30 --window-sec 5
  python _plot_onset_probs.py --onset-threshold 0.75 --top-keys 6
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from gpu_ops import DEVICE, get_gpu_enhanced_mel_transcriber
from rhythm_training.train_enhanced_mel_transcriber import (MIDI_OFFSET,
                                                            _peak_frames)

MAESTRO_ROOT = Path("rhythm_training/maestro_midi")
CSV_PATH = MAESTRO_ROOT / "maestro-v3.0.0.csv"


def live_onset_threshold():
    """The onset threshold the LIVE path uses, read from the same source.

    detect_note.py's enhanced-mel live path (analyze_audio_live_neural) sets its
    onset threshold from os.environ['LIVE_ENHANCED_ONSET_BASE'] with a 0.70
    default. Reading the identical env var here keeps this script locked to live:
    any override the user exports applies to both.
    """
    try:
        return float(os.environ.get("LIVE_ENHANCED_ONSET_BASE", "0.70"))
    except (TypeError, ValueError):
        return 0.70


def load_test_rows(limit=None):
    import csv
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rows.sort(key=lambda r: r["midi_filename"])
    return rows[:limit] if limit else rows


@torch.no_grad()
def onset_probs_for_audio(tx, audio):
    """Replicate transcribe()'s windowed inference, return (onset_probs, frame_time).

    onset_probs: float32 [n_frames, n_keys], overlap-averaged exactly as in
    gpu_ops.transcribe (only the onset head is kept here).
    """
    sr = tx.config.get("sample_rate", 16000)
    hop = tx.config.get("hop_length", 256)
    n_keys = tx.config.get("n_keys", 88)

    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    features = tx.extractor.extract(audio_t)
    n_frames = features.size(1)

    chunk_frames = int(10.0 * sr / hop)
    overlap = chunk_frames // 4
    step = chunk_frames - overlap

    all_onset = np.zeros((n_frames, n_keys), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)

    for start in range(0, n_frames, step):
        end = min(start + chunk_frames, n_frames)
        chunk = features[:, start:end, :]
        out = tx.model(chunk)
        onset_p = torch.sigmoid(out["onset_logits"][0]).cpu().numpy()
        actual_len = end - start
        all_onset[start:end] += onset_p[:actual_len]
        counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    all_onset /= counts[:, None]
    return all_onset, hop / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece-index", type=int, default=0,
                    help="index into the (sorted) MAESTRO test split")
    ap.add_argument("--start-sec", type=float, default=10.0,
                    help="window start in seconds")
    ap.add_argument("--window-sec", type=float, default=4.0,
                    help="window length in seconds")
    ap.add_argument("--onset-threshold", type=float, default=None,
                    help="peak-pick threshold; default = the live path's value "
                         "(LIVE_ENHANCED_ONSET_BASE, 0.70)")
    ap.add_argument("--top-keys", type=int, default=5,
                    help="how many most-active keys to draw in the line panel")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: ./_onset_probs_<piece>.png)")
    args = ap.parse_args()

    if args.onset_threshold is None:
        args.onset_threshold = live_onset_threshold()
        print(f"Using live onset threshold {args.onset_threshold} "
              f"(LIVE_ENHANCED_ONSET_BASE)", flush=True)

    tx = get_gpu_enhanced_mel_transcriber()
    if tx is None or not getattr(tx, "initialized", False):
        raise SystemExit("Enhanced mel transcriber unavailable (CUDA/model not loaded).")

    rows = load_test_rows()
    if not (0 <= args.piece_index < len(rows)):
        raise SystemExit(f"piece-index out of range 0..{len(rows) - 1}")
    row = rows[args.piece_index]
    apath = MAESTRO_ROOT / row["audio_filename"]
    print(f"Piece [{args.piece_index}] {Path(row['midi_filename']).name}", flush=True)

    audio, _ = librosa.load(str(apath), sr=16000, mono=True)
    onset_probs, frame_time = onset_probs_for_audio(tx, audio)
    n_frames, n_keys = onset_probs.shape

    f0 = int(round(args.start_sec / frame_time))
    f1 = int(round((args.start_sec + args.window_sec) / frame_time))
    f0 = max(0, min(f0, n_frames - 1))
    f1 = max(f0 + 1, min(f1, n_frames))
    window = onset_probs[f0:f1]  # [W, n_keys]
    times = (np.arange(f0, f1) * frame_time)
    print(f"Window frames {f0}..{f1} ({times[0]:.2f}-{times[-1]:.2f}s), "
          f"frame_time={frame_time*1000:.1f}ms", flush=True)

    # Picked onsets (full-clip peak-pick, then restrict to the window) -- exactly
    # what decode_enhanced_note_events would emit at this threshold.
    peak_t, peak_pitch, peak_prob = [], [], []
    for key in range(n_keys):
        for fr in _peak_frames(onset_probs[:, key], args.onset_threshold):
            if f0 <= fr < f1:
                peak_t.append(fr * frame_time)
                peak_pitch.append(key + MIDI_OFFSET)
                peak_prob.append(float(onset_probs[fr, key]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed: pip install matplotlib")

    fig, ax_l = plt.subplots(figsize=(13, 5))

    # ---- 1-D onset prob for the most active keys ----
    energy = window.sum(axis=0)
    top = np.argsort(energy)[::-1][:args.top_keys]
    top = [k for k in top if energy[k] > 0]
    for key in top:
        pitch = key + MIDI_OFFSET
        name = librosa.midi_to_note(pitch)
        ax_l.plot(times, window[:, key], lw=1.3, label=f"{name} ({pitch})")
    ax_l.axhline(args.onset_threshold, color="red", ls=":", lw=1.6,
                 label=f"onset threshold {args.onset_threshold}")
    ax_l.set_ylim(-0.02, 1.02)
    ax_l.set_xlim(times[0], times[-1] + frame_time)
    ax_l.set_xlabel("time (s)")
    ax_l.set_ylabel("P(onset)")
    ax_l.set_title(
        f"Onset probability vs time  |  {times[0]:.2f}-{times[-1]:.2f}s")
    ax_l.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    out = args.out or f"_onset_probs_piece{args.piece_index}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}  ({len(peak_t)} onsets picked in window)", flush=True)


if __name__ == "__main__":
    main()
