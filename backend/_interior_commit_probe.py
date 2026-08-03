"""Does a SINGLE decode's interior replace temporal persistence?

User's point: the model already runs on a long context (here 1.8s) but we only
act on the trailing edge (~now), where there is no post-onset audio -> flicker.
Yet for an onset 60ms in the past, THIS window's interior already contains those
60ms of post-onset evidence. Persistence waits for a 2nd decode to confirm what a
single decode could already read from its own interior.

Test: ignore the session/hypothesis machinery. At each step decode the window and
COMMIT (once, first appearance) every onset whose distance from the trailing edge
from_end = window_end - onset is >= D. No multi-observation requirement. Sweep D.
If a single decode's interior reaches the obs>=2 precision (0.972), we can commit
from one window at latency ~= D instead of waiting out the persistence interval.

Compared against the shipped persistence baseline: P0.972 R0.927, emit median 137ms.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

import main as live_main
from main import MIN_STREAM_ANALYSIS_SAMPLES
from tune_continuous_stream_decoder import (
    TARGET_SR, load_audio_excerpt, load_manifest, load_midi_notes, slice_gt_notes,
)

MANIFEST = Path("benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json")
MATCH_TOL = 0.050
DEDUP_TOL = 0.050
PACKET_MS = 40.0
CONTEXT_SEC = 1.8
INTERVAL_MS = 50.0


def _flat_notes(result, window_start_sec):
    out = []
    for n in result.get("notes") or []:
        out.append((int(n.get("midi_note", 0) or 0),
                    window_start_sec + float(n.get("time_seconds", 0.0) or 0.0),
                    float(n.get("confidence", 0.0) or 0.0)))
    for c in result.get("chords") or []:
        onset = window_start_sec + float(c.get("time_seconds", 0.0) or 0.0)
        conf = float(c.get("confidence", 0.0) or 0.0)
        for m in c.get("midi_notes") or []:
            out.append((int(m), onset, conf))
    return out


def decode_window(buffer, cursor_samples):
    ctx = max(MIN_STREAM_ANALYSIS_SAMPLES, int(round(CONTEXT_SEC * TARGET_SR)))
    win = buffer[-ctx:] if buffer.size > ctx else buffer
    if win.size < MIN_STREAM_ANALYSIS_SAMPLES:
        return [], 0.0
    window_start_sec = (cursor_samples - win.size) / TARGET_SR
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        res = live_main.analyze_audio_live_neural(win.astype(np.float32, copy=True), sr=TARGET_SR,
                                                  debug=False, split_midi=60, device="cuda",
                                                  adaptive_onset_threshold=True)
    if res.get("error"):
        return [], window_start_sec
    return _flat_notes(res, window_start_sec), window_start_sec


def run_clip(clip, margins):
    audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
    audio = np.concatenate([audio, np.zeros(int(round(0.6 * TARGET_SR)), dtype=np.float32)])
    packet_frames = max(1, int(round(PACKET_MS * TARGET_SR / 1000.0)))
    interval_samples = max(1, int(round(INTERVAL_MS / 1000.0 * TARGET_SR)))

    # committed[D] -> list of (midi, onset_abs, commit_now) ; dedup by pitch+onset
    committed = {D: [] for D in margins}
    buffer = np.zeros(0, dtype=np.float32)
    cursor = 0
    last_infer = -interval_samples
    for start in range(0, audio.size, packet_frames):
        pkt = audio[start:start + packet_frames]
        buffer = np.concatenate([buffer, pkt])
        cursor += pkt.size
        if cursor - last_infer < interval_samples:
            continue
        last_infer = cursor
        notes, _ = decode_window(buffer, cursor)
        now = cursor / TARGET_SR
        for midi, onset_abs, _conf in notes:
            from_end = now - onset_abs
            for D in margins:
                if from_end < D:
                    continue
                lst = committed[D]
                if any(m == midi and abs(o - onset_abs) <= DEDUP_TOL for m, o, _ in lst):
                    continue
                lst.append((midi, onset_abs, now))

    gt_raw = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
    gt = sorted(((int(n["midi_note"]), float(n.get("onset_time", n.get("time_seconds", 0.0)) or 0.0))
                 for n in gt_raw), key=lambda mp: (mp[1], mp[0]))
    return committed, gt


def match(committed_list, gt):
    """Greedy GT<->committed; returns (matched, n_pred, latencies_ms)."""
    pred = sorted(committed_list, key=lambda x: x[1])
    used = [False] * len(pred)
    matched = 0
    lats = []
    for g_midi, g_onset in gt:
        best_j, best_err = -1, MATCH_TOL + 1e-9
        for j, (p_midi, p_onset, _now) in enumerate(pred):
            if used[j] or p_midi != g_midi:
                continue
            err = abs(p_onset - g_onset)
            if err < best_err:
                best_err, best_j = err, j
        if best_j >= 0:
            used[best_j] = True
            matched += 1
            lats.append((pred[best_j][2] - g_onset) * 1000.0)
    return matched, len(pred), lats


def main():
    margins = [0.0, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    clips = load_manifest(MANIFEST, [])
    print(f"Loaded {len(clips)} clips; context={CONTEXT_SEC}s interval={INTERVAL_MS}ms "
          f"(single-decode interior, NO persistence)\n")
    agg = {D: {"m": 0, "p": 0, "lat": []} for D in margins}
    n_gt = 0
    for cid, clip in clips.items():
        committed, gt = run_clip(clip, margins)
        n_gt += len(gt)
        for D in margins:
            m, p, lats = match(committed[D], gt)
            agg[D]["m"] += m
            agg[D]["p"] += p
            agg[D]["lat"].extend(lats)

    print("SHIPPED persistence baseline:   P=0.972 R=0.927 F1=0.949 | emit med=137 p95=333 ms\n")
    print(f"{'interior D':>10} | {'P':>6} {'R':>6} {'F1':>6} | {'pred':>5} {'match':>5} | "
          f"{'lat_med':>7} {'lat_p95':>7} ms")
    for D in margins:
        m, p = agg[D]["m"], agg[D]["p"]
        prec = m / p if p else float("nan")
        rec = m / n_gt if n_gt else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        lat = np.asarray(agg[D]["lat"], dtype=np.float64)
        med = np.percentile(lat, 50) if lat.size else float("nan")
        p95 = np.percentile(lat, 95) if lat.size else float("nan")
        print(f"{D*1000:8.0f}ms | {prec:6.3f} {rec:6.3f} {f1:6.3f} | {p:5d} {m:5d} | "
              f"{med:7.1f} {p95:7.1f}")


if __name__ == "__main__":
    main()
