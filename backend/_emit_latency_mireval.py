"""Per-note EMIT LATENCY for the continuous live-stream path (literature-comparable).

The streaming-transcription papers (Wei et al. 2503.01362; Hu et al. 2509.07586)
define latency as the algorithmic delay between when an event actually occurs in
the audio and the earliest time the system can output it, *ignoring compute speed*
(Wei: "latency = length of future frames"; Hu decomposes buffering + preprocessing
+ inference + postprocessing). This harness measures that quantity empirically for
our system, which is the most honest version because it captures every source at
once: the multi-observation consensus wait, the STFT lookahead, and chunk-end
deferral.

Method
------
We replay each clip through ContinuousLiveStreamSession exactly as the shipped
/live/stream path does, feeding packets sequentially so that "audio fed so far"
IS the simulated wall-clock. At every inference step we snapshot which notes are
visible on a surface (score = stable; preview = immediate/unstable). For each
ground-truth note that the final score correctly contains (pitch exact, onset
within 50 ms), we find the EARLIEST snapshot whose visible set already contains a
matching payload, and define

    emit_latency = (audio_time_fed_at_that_snapshot) - (ground_truth_onset)

This is compute-independent (algorithmic) latency. We report it as a distribution
(median / p95 / mean), micro-pooled across clips, alongside:
  - the design-time algorithmic latency  max(trusted_delay, min_obs * interval)
    (the back-of-envelope number; the measured distribution is the real thing),
  - the real-time factor (RTF = inference_p95 / interval; must be < 1 to run live),
  - onset F1 / precision / recall @ 50 ms on the same surface.

Best published comparisons (audio->MIDI):
  Wei 2024  : 380 ms @ 96.5% onset F1     Kwon online : 128-320 ms
  Hu  2025  : 10-30 ms target, F1 collapses to ~31-37%
For audio->SCORE (notation) there is NO published latency number: the only
end-to-end A2S system (Zeng 2024) is offline. A number here is the first of its
kind for the notation task.

Usage (from backend/):
  python _emit_latency_mireval.py                       # gold12, shipped configs
  python _emit_latency_mireval.py --manifest <path.json>  # e.g. the 48-clip set
  python _emit_latency_mireval.py --limit 4             # smoke
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tune_continuous_stream_decoder import (
    ContinuousLiveStreamSession,
    TARGET_SR,
    load_audio_excerpt,
    load_manifest,
    load_midi_notes,
    notes_from_accumulator,
    override_live_attrs,
    slice_gt_notes,
    update_accumulator,
    visible_payloads,
)

DEFAULT_MANIFEST = Path(
    "benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json"
)

MATCH_TOL_SEC = 0.050   # onset tolerance for a correct detection (mir_eval standard)
IDENTITY_TOL_SEC = 0.060  # onset tolerance when locating a payload across snapshots

# (label, surface, include_unstable, trusted_delay_ms, min_obs, interval_ms, interior_margin_sec)
# Sweeping the consensus depth (min_obs) on the stable score surface traces the
# latency<->precision curve; the preview point is the zero-consensus extreme.
CONFIGS = [
    ("preview_obs1_int50", "preview", True, 60.0, 1, 50.0, 0.0),  # immediate / zero consensus
    ("score_obs1_int50", "score", False, 60.0, 1, 50.0, 0.0),     # min_obs=1 flicker flood (P~0.59)
    ("score_obs2_int50", "score", False, 60.0, 2, 50.0, 0.0),     # shipped (~100 ms design)
    ("score_obs3_int50", "score", False, 60.0, 3, 50.0, 0.0),     # stable, 3 observations
    # INTERIOR-TRUST validation: drop trailing-edge observations (within D of "now")
    # so flicker is never born, then trust the interior at min_obs=1. Trusted-delay
    # set == D so display fires exactly when the first interior observation lands.
    # Probe (_interior_commit_probe.py) predicted D=50 -> ~F1 0.944 @ ~94 ms; this
    # measures it through the REAL session pipeline (continuity filter, accumulator).
    ("score_obs1_int50_interior30", "score", False, 30.0, 1, 50.0, 0.030),
    ("score_obs1_int50_interior50", "score", False, 50.0, 1, 50.0, 0.050),
]


def base_args(trusted_ms: float, interval_ms: float) -> SimpleNamespace:
    return SimpleNamespace(
        tail_padding_sec=0.6,
        context_sec=1.8,
        inference_interval_ms=interval_ms,
        trusted_delay_ms=trusted_ms,
        commit_delay_ms=500.0,
        lock_delay_ms=2000.0,
        packet_ms=40.0,
        chunk_seconds_for_boundary=0.6,
        eval_boundary_band_sec=0.10,
    )


def _payload_pairs(update: Mapping, include_unstable: bool) -> List[Tuple[int, float]]:
    """(midi, onset_time) of every note visible on the surface at this step."""
    out: List[Tuple[int, float]] = []
    for p in visible_payloads(update, include_unstable=include_unstable):
        out.append((int(p.get("midi_note", 0) or 0), float(p.get("onset_time", 0.0) or 0.0)))
    return out


def replay_with_snapshots(
    clip: Mapping, args: SimpleNamespace, include_unstable: bool
) -> Dict:
    """Run the continuous session; record, per inference step, the audio-time-fed
    and the set of notes visible on the chosen surface. Returns final predicted
    notes, GT notes, the snapshot timeline, and inference timing."""
    audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
    if args.tail_padding_sec > 0:
        audio = np.concatenate(
            [audio, np.zeros(int(round(args.tail_padding_sec * TARGET_SR)), dtype=np.float32)]
        )

    session = ContinuousLiveStreamSession(
        session_id="emit-latency-bench",
        sample_rate=TARGET_SR,
        context_sec=args.context_sec,
        inference_interval_sec=args.inference_interval_ms / 1000.0,
        trusted_delay_sec=args.trusted_delay_ms / 1000.0,
        commit_delay_sec=args.commit_delay_ms / 1000.0,
        lock_delay_sec=args.lock_delay_ms / 1000.0,
    )
    packet_frames = max(1, int(round(args.packet_ms * TARGET_SR / 1000.0)))

    # snapshots: list of (audio_time_fed_sec, [(midi, onset), ...]) for first-visible timing.
    snapshots: List[Tuple[float, List[Tuple[int, float]]]] = []
    # acc: id-deduped final predicted set -- reuses the SHIPPED accumulator so the
    # predicted count (and thus precision) matches the production benchmark exactly.
    acc: Dict[str, Dict] = {}
    inference_ms: List[float] = []

    def _ingest(update: Mapping, audio_fed_sec: float) -> None:
        if update is None:
            return
        snapshots.append((audio_fed_sec, _payload_pairs(update, include_unstable)))
        update_accumulator(acc, update, include_unstable=include_unstable)
        inference = update.get("inference") or {}
        if inference.get("ran"):
            inference_ms.append(float(inference.get("inference_ms", 0.0) or 0.0))

    for start in range(0, audio.size, packet_frames):
        packet = audio[start : start + packet_frames]
        session.append_audio(packet)
        audio_fed_sec = (start + packet.size) / TARGET_SR
        _ingest(session.maybe_run_inference(), audio_fed_sec)

    total_audio_sec = audio.size / TARGET_SR
    _ingest(session.maybe_run_inference(force=True), total_audio_sec)

    pred_notes = sorted(
        ((int(n["midi_note"]), float(n["onset_time"])) for n in notes_from_accumulator(acc)),
        key=lambda mp: (mp[1], mp[0]),
    )
    gt_raw = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
    gt_notes = sorted(
        ((int(n["midi_note"]), float(n.get("onset_time", n.get("time_seconds", 0.0)) or 0.0)) for n in gt_raw),
        key=lambda mp: (mp[1], mp[0]),
    )
    return {
        "pred_notes": pred_notes,
        "gt_notes": gt_notes,
        "snapshots": snapshots,
        "inference_ms": inference_ms,
    }


def first_visible_time(
    snapshots: Sequence[Tuple[float, List[Tuple[int, float]]]],
    midi: int,
    onset: float,
    tol: float = IDENTITY_TOL_SEC,
) -> float | None:
    """Earliest audio_fed time at which a payload of this pitch/onset was visible."""
    for audio_fed_sec, pairs in snapshots:  # snapshots are append-order = chronological
        for p_midi, p_onset in pairs:
            if p_midi == midi and abs(p_onset - onset) <= tol:
                return audio_fed_sec
    return None


def match_and_measure(result: Mapping) -> Dict:
    """Greedy onset-pitch match (GT<->final pred); for each matched pred, attach
    the emit latency = first-visible-time - GT onset."""
    pred = list(result["pred_notes"])
    gt = list(result["gt_notes"])
    snapshots = result["snapshots"]

    used = [False] * len(pred)
    latencies: List[float] = []
    matched = 0
    for g_midi, g_onset in gt:
        best_j, best_err = -1, MATCH_TOL_SEC + 1e-9
        for j, (p_midi, p_onset) in enumerate(pred):
            if used[j] or p_midi != g_midi:
                continue
            err = abs(p_onset - g_onset)
            if err < best_err:
                best_err, best_j = err, j
        if best_j < 0:
            continue
        used[best_j] = True
        matched += 1
        p_midi, p_onset = pred[best_j]
        seen = first_visible_time(snapshots, p_midi, p_onset)
        if seen is not None:
            latencies.append(seen - g_onset)  # signed; ~>=0 by construction

    return {
        "matched": matched,
        "predicted": len(pred),
        "ground_truth": len(gt),
        "latencies": latencies,
    }


def _pct(vals: Sequence[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=0, help="cap clip count for smoke runs")
    parser.add_argument("--out", type=Path, default=Path("benchmark_artifacts/_emit_latency_results.json"))
    cli = parser.parse_args()

    clips = load_manifest(cli.manifest, [])
    if cli.limit > 0:
        clips = dict(list(clips.items())[: cli.limit])
    print(f"Loaded {len(clips)} clips from {cli.manifest.name}")

    rows = []
    for label, surface, include_unstable, trusted_ms, min_obs, interval_ms, interior_sec in CONFIGS:
        args = base_args(trusted_ms, interval_ms)
        all_latencies: List[float] = []
        all_infer: List[float] = []
        m = p = g = 0
        with override_live_attrs({
            "STREAM_MIN_DISPLAY_OBSERVATIONS": min_obs,
            "STREAM_INTERIOR_MARGIN_SEC": interior_sec,
        }):
            for cid, clip in clips.items():
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    result = replay_with_snapshots(clip, args, include_unstable)
                meas = match_and_measure(result)
                all_latencies.extend(meas["latencies"])
                all_infer.extend(result["inference_ms"])
                m += meas["matched"]; p += meas["predicted"]; g += meas["ground_truth"]

        prec = m / p if p else 0.0
        rec = m / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lat_ms = [x * 1000.0 for x in all_latencies]
        design_lat = max(trusted_ms, min_obs * interval_ms)
        infer_p95 = _pct(all_infer, 95)
        rtf = infer_p95 / interval_ms if interval_ms else float("nan")

        row = dict(
            label=label, surface=surface, design_latency_ms=design_lat,
            emit_latency_median_ms=_pct(lat_ms, 50),
            emit_latency_p95_ms=_pct(lat_ms, 95),
            emit_latency_mean_ms=float(np.mean(lat_ms)) if lat_ms else float("nan"),
            precision=prec, recall=rec, f1=f1,
            matched=m, predicted=p, ground_truth=g, n_latency=len(lat_ms),
            infer_p95_ms=infer_p95, rtf=rtf,
            latencies_ms=[round(x, 2) for x in lat_ms],  # raw, for plotting
        )
        rows.append(row)
        print(
            f"{label:20s} surf={surface:7s} design~{design_lat:3.0f}ms | "
            f"emit median={row['emit_latency_median_ms']:6.1f} p95={row['emit_latency_p95_ms']:6.1f} "
            f"mean={row['emit_latency_mean_ms']:6.1f} ms (n={len(lat_ms)}) | "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | "
            f"infer_p95={infer_p95:4.1f}ms rtf={rtf:.2f}"
        )

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {cli.out}")


if __name__ == "__main__":
    main()
