"""Birth-signal separability probe for two floor-shifting ideas.

The emit-latency floor is the persistence wait: a note must survive
STREAM_MIN_DISPLAY_OBSERVATIONS (=2) re-decodes before it is displayed, and that
wait IS the latency. Both candidate fixes try to earn the SAME precision from a
single observation (obs==1), so the note can display ~1 interval sooner:

  Idea 1  confidence-routed depth : trust obs==1 when first-observation
          confidence >= tau.
  Idea 2  multi-context ensemble  : trust obs==1 when the note ALSO appears in a
          parallel decode at a shorter context length at the same instant
          ("instant persistence" instead of temporal persistence).

This probe does NOT change production. It replays gold12 through the real
ContinuousLiveStreamSession, and for every hypothesis BORN it records:
  - first_conf      : model confidence at obs==1
  - ensemble_agree  : did a short-context decode at the birth step contain it
  - max_obs         : peak observation count it ever reached (>=2 => the current
                      rule would display it)
  - is_tp           : does its final (pitch, onset) match a GT note @50 ms

Then it reports, for each rule, the precision/recall over born hypotheses, so we
can see whether obs==1 + signal reaches the obs>=2 precision (~0.97) the shipped
system gets. If a signal cannot separate TP from flicker, that idea is dead.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

import main as live_main
from main import ContinuousLiveStreamSession, MIN_STREAM_ANALYSIS_SAMPLES
from tune_continuous_stream_decoder import (
    TARGET_SR,
    load_audio_excerpt,
    load_manifest,
    load_midi_notes,
    slice_gt_notes,
)

MANIFEST = Path("benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json")
MATCH_TOL = 0.050
ENSEMBLE_TOL = 0.050
PACKET_MS = 40.0
CONTEXT_SEC = 1.8
SHORT_CONTEXT_SEC = 0.9   # the parallel "second view" for idea 2
INTERVAL_MS = 50.0
TRUSTED_MS = 60.0


def _flat_notes(result, window_start_sec):
    """(midi, absolute_onset, confidence) for every note/chord-member in a decode."""
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


def probe_clip(clip):
    audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
    audio = np.concatenate([audio, np.zeros(int(round(0.6 * TARGET_SR)), dtype=np.float32)])

    session = ContinuousLiveStreamSession(
        session_id="probe", sample_rate=TARGET_SR, context_sec=CONTEXT_SEC,
        inference_interval_sec=INTERVAL_MS / 1000.0, trusted_delay_sec=TRUSTED_MS / 1000.0,
        commit_delay_sec=0.5, lock_delay_sec=2.0,
    )
    packet_frames = max(1, int(round(PACKET_MS * TARGET_SR / 1000.0)))

    # per-hypothesis-id record, captured across steps (hyps age out, so we keep our own copy)
    born = {}  # id -> dict(first_conf, max_obs, midi, onset, ensemble_agree)

    def run_short_decode():
        """Parallel shorter-context decode on the same window-end; returns flat notes."""
        end = session.sample_cursor
        short_samples = max(MIN_STREAM_ANALYSIS_SAMPLES, int(round(SHORT_CONTEXT_SEC * TARGET_SR)))
        start = max(session.absolute_start_sample, end - short_samples)
        rel = max(0, int(start - session.absolute_start_sample))
        win = session.audio[rel:].astype(np.float32, copy=True)
        if win.size < MIN_STREAM_ANALYSIS_SAMPLES:
            return []
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            res = live_main.analyze_audio_live_neural(win, sr=TARGET_SR, debug=False,
                                                      split_midi=60, device="cuda",
                                                      adaptive_onset_threshold=True)
        if res.get("error"):
            return []
        return _flat_notes(res, start / TARGET_SR)

    for start in range(0, audio.size, packet_frames):
        session.append_audio(audio[start:start + packet_frames])
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            upd = session.maybe_run_inference()
        if upd is None or not (upd.get("inference") or {}).get("ran"):
            continue
        # a fresh hypothesis is one we have not recorded yet (observations==1 this step)
        new_ids = [h for h in session.hypotheses
                   if int(h.get("observations", 0) or 0) >= 1 and h["id"] not in born]
        short_notes = run_short_decode() if new_ids else []
        for h in new_ids:
            midi = int(h.get("midi_note", 0) or 0)
            onset = float(h.get("onset_time", 0.0) or 0.0)
            agree = any(m == midi and abs(o - onset) <= ENSEMBLE_TOL for m, o, _ in short_notes)
            born[h["id"]] = {
                "first_conf": float(h.get("confidence", 0.0) or 0.0),
                "max_obs": int(h.get("observations", 0) or 0),
                "midi": midi, "onset": onset, "ensemble_agree": agree,
            }
        # update max_obs / latest onset for everyone still alive
        for h in session.hypotheses:
            if h["id"] in born:
                rec = born[h["id"]]
                rec["max_obs"] = max(rec["max_obs"], int(h.get("observations", 0) or 0))
                rec["onset"] = float(h.get("onset_time", rec["onset"]) or rec["onset"])

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        session.maybe_run_inference(force=True)
    for h in session.hypotheses:
        if h["id"] in born:
            born[h["id"]]["max_obs"] = max(born[h["id"]]["max_obs"], int(h.get("observations", 0) or 0))

    gt_raw = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
    gt = sorted(((int(n["midi_note"]), float(n.get("onset_time", n.get("time_seconds", 0.0)) or 0.0))
                 for n in gt_raw), key=lambda mp: (mp[1], mp[0]))
    return list(born.values()), gt


def label_tp(records, gt):
    """Greedy GT<->born match on pitch+onset@50ms; returns is_tp per record and #gt."""
    recs = sorted(records, key=lambda r: r["onset"])
    used = [False] * len(recs)
    for g_midi, g_onset in gt:
        best_j, best_err = -1, MATCH_TOL + 1e-9
        for j, r in enumerate(recs):
            if used[j] or r["midi"] != g_midi:
                continue
            err = abs(r["onset"] - g_onset)
            if err < best_err:
                best_err, best_j = err, j
        if best_j >= 0:
            used[best_j] = True
    for j, r in enumerate(recs):
        r["is_tp"] = used[j]
    return recs


def pr(records, predicate, n_gt):
    sel = [r for r in records if predicate(r)]
    tp = sum(1 for r in sel if r["is_tp"])
    p = tp / len(sel) if sel else float("nan")
    r = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) and not np.isnan(p) and not np.isnan(r) else float("nan")
    return p, r, f1, len(sel), tp


def main():
    clips = load_manifest(MANIFEST, [])
    print(f"Loaded {len(clips)} clips; context={CONTEXT_SEC}s short={SHORT_CONTEXT_SEC}s "
          f"interval={INTERVAL_MS}ms packet={PACKET_MS}ms\n")
    all_recs, n_gt = [], 0
    for cid, clip in clips.items():
        recs, gt = probe_clip(clip)
        recs = label_tp(recs, gt)
        all_recs.extend(recs)
        n_gt += len(gt)
        print(f"  {cid:28s} born={len(recs):4d} tp={sum(r['is_tp'] for r in recs):3d} gt={len(gt):3d}")

    R = all_recs
    print(f"\nTotal: born={len(R)}  GT={n_gt}  born_TP={sum(r['is_tp'] for r in R)}")

    print("\n=== BASELINE rules ===")
    for name, pred in [
        ("obs>=1 (display everything born)", lambda r: True),
        ("obs>=2 (SHIPPED persistence)", lambda r: r["max_obs"] >= 2),
    ]:
        p, rc, f1, n, tp = pr(R, pred, n_gt)
        print(f"  {name:36s} P={p:.3f} R={rc:.3f} F1={f1:.3f}  (n={n}, tp={tp})")

    print("\n=== IDEA 1: obs==1 + first_conf >= tau ===")
    print("  (precision should approach the obs>=2 baseline if confidence separates TP from flicker)")
    confs = np.array([r["first_conf"] for r in R])
    print(f"  first_conf over born: min={confs.min():.3f} p25={np.percentile(confs,25):.3f} "
          f"med={np.percentile(confs,50):.3f} p75={np.percentile(confs,75):.3f} max={confs.max():.3f}")
    print(f"  first_conf | TP  : med={np.median([r['first_conf'] for r in R if r['is_tp']]):.3f}")
    print(f"  first_conf | FP  : med={np.median([r['first_conf'] for r in R if not r['is_tp']]):.3f}")
    for tau in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        p, rc, f1, n, tp = pr(R, lambda r, t=tau: r["first_conf"] >= t, n_gt)
        print(f"  tau={tau:.2f}  P={p:.3f} R={rc:.3f} F1={f1:.3f}  (n={n}, tp={tp})")

    print("\n=== IDEA 2: obs==1 + ensemble_agree (parallel short-context decode) ===")
    for name, pred in [
        ("ensemble_agree only", lambda r: r["ensemble_agree"]),
        ("ensemble_agree AND first_conf>=0.5", lambda r: r["ensemble_agree"] and r["first_conf"] >= 0.5),
    ]:
        p, rc, f1, n, tp = pr(R, pred, n_gt)
        print(f"  {name:36s} P={p:.3f} R={rc:.3f} F1={f1:.3f}  (n={n}, tp={tp})")
    # how much does the short view agree with the eventual survivors?
    surv = [r for r in R if r["max_obs"] >= 2]
    agree_surv = sum(1 for r in surv if r["ensemble_agree"])
    flick = [r for r in R if r["max_obs"] < 2]
    agree_flick = sum(1 for r in flick if r["ensemble_agree"])
    print(f"  ensemble agreement among obs>=2 survivors: {agree_surv}/{len(surv)} "
          f"({100*agree_surv/max(1,len(surv)):.0f}%)")
    print(f"  ensemble agreement among obs<2 flicker   : {agree_flick}/{len(flick)} "
          f"({100*agree_flick/max(1,len(flick)):.0f}%)")


if __name__ == "__main__":
    main()
