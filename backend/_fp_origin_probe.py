"""Idea 3: where in the context window do born notes (esp. reproducible FPs) sit?

Ideas 1/2 died because FPs are reproducible within an instant (same model +
overlapping audio). If those reproducible FPs have a STRUCTURAL origin -- the
edges of the 1.8 s decode window, where attacks are truncated (leading edge) or
post-onset evidence is missing (trailing edge) -- then an edge/interior gate can
remove them at obs==1, which is the single-instant fix ideas 1/2 could not be.

For every born hypothesis this logs, AT ITS BIRTH STEP, the onset's position in
the decode window:
  from_start = onset - window_start   (0 = oldest/leading edge, truncation)
  from_end   = window_end - onset     (0 = newest/trailing edge, = "now")
Then it asks: do FPs cluster at an edge while TPs sit in the interior, and does an
interior gate at obs==1 reach the obs>=2 precision (0.972) at useful recall?
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

import main as live_main
from main import ContinuousLiveStreamSession, MIN_STREAM_ANALYSIS_SAMPLES
from tune_continuous_stream_decoder import (
    TARGET_SR, load_audio_excerpt, load_manifest, load_midi_notes, slice_gt_notes,
)

MANIFEST = Path("benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json")
MATCH_TOL = 0.050
PACKET_MS = 40.0
CONTEXT_SEC = 1.8
INTERVAL_MS = 50.0
TRUSTED_MS = 60.0


def probe_clip(clip):
    audio = load_audio_excerpt(clip["audio_path"], clip["start_sec"], clip["duration_sec"], TARGET_SR)
    audio = np.concatenate([audio, np.zeros(int(round(0.6 * TARGET_SR)), dtype=np.float32)])
    session = ContinuousLiveStreamSession(
        session_id="probe", sample_rate=TARGET_SR, context_sec=CONTEXT_SEC,
        inference_interval_sec=INTERVAL_MS / 1000.0, trusted_delay_sec=TRUSTED_MS / 1000.0,
        commit_delay_sec=0.5, lock_delay_sec=2.0,
    )
    packet_frames = max(1, int(round(PACKET_MS * TARGET_SR / 1000.0)))
    born = {}

    for start in range(0, audio.size, packet_frames):
        session.append_audio(audio[start:start + packet_frames])
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            upd = session.maybe_run_inference()
        if upd is None or not (upd.get("inference") or {}).get("ran"):
            continue
        now = session.current_time_sec
        win_start = max(session.absolute_start_sample / TARGET_SR, now - CONTEXT_SEC)
        for h in session.hypotheses:
            if h["id"] in born:
                rec = born[h["id"]]
                rec["max_obs"] = max(rec["max_obs"], int(h.get("observations", 0) or 0))
                rec["onset"] = float(h.get("onset_time", rec["onset"]) or rec["onset"])
                continue
            onset = float(h.get("onset_time", 0.0) or 0.0)
            born[h["id"]] = {
                "midi": int(h.get("midi_note", 0) or 0), "onset": onset,
                "max_obs": int(h.get("observations", 0) or 0),
                "from_start": onset - win_start, "from_end": now - onset,
            }

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


def stat(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return "  (none)"
    return (f"n={v.size:4d} min={v.min():6.3f} p10={np.percentile(v,10):6.3f} "
            f"med={np.percentile(v,50):6.3f} p90={np.percentile(v,90):6.3f} max={v.max():6.3f}")


def pr(records, predicate, n_gt):
    sel = [r for r in records if predicate(r)]
    tp = sum(1 for r in sel if r["is_tp"])
    p = tp / len(sel) if sel else float("nan")
    rc = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * rc / (p + rc) if (p + rc) and not np.isnan(p) else float("nan")
    return p, rc, f1, len(sel), tp


def main():
    clips = load_manifest(MANIFEST, [])
    print(f"Loaded {len(clips)} clips; context={CONTEXT_SEC}s interval={INTERVAL_MS}ms\n")
    R, n_gt = [], 0
    for cid, clip in clips.items():
        recs, gt = probe_clip(clip)
        recs = label_tp(recs, gt)
        R.extend(recs)
        n_gt += len(gt)
    print(f"Total born={len(R)} GT={n_gt} TP={sum(r['is_tp'] for r in R)}\n")

    tp = [r for r in R if r["is_tp"]]
    fp = [r for r in R if not r["is_tp"]]
    fp_committed = [r for r in fp if r["max_obs"] >= 2]   # FPs that pollute the shipped score
    fp_flicker = [r for r in fp if r["max_obs"] < 2]      # FPs persistence already kills

    print("=== from_start (sec from leading/oldest edge; small => truncated attack) ===")
    print("  TP          ", stat([r["from_start"] for r in tp]))
    print("  FP committed ", stat([r["from_start"] for r in fp_committed]))
    print("  FP flicker  ", stat([r["from_start"] for r in fp_flicker]))
    print("\n=== from_end (sec from trailing/newest edge = 'now'; small => no post-onset evidence) ===")
    print("  TP          ", stat([r["from_end"] for r in tp]))
    print("  FP committed ", stat([r["from_end"] for r in fp_committed]))
    print("  FP flicker  ", stat([r["from_end"] for r in fp_flicker]))

    print("\n=== fraction of each group within an edge band ===")
    for band in [0.05, 0.1, 0.15, 0.2, 0.3]:
        def frac(group, key):
            return 100 * np.mean([r[key] <= band for r in group]) if group else 0.0
        print(f"  band<={band:.2f}s | from_start: TP={frac(tp,'from_start'):4.0f}% "
              f"FPcomm={frac(fp_committed,'from_start'):4.0f}% FPflick={frac(fp_flicker,'from_start'):4.0f}%"
              f"   | from_end: TP={frac(tp,'from_end'):4.0f}% "
              f"FPcomm={frac(fp_committed,'from_end'):4.0f}% FPflick={frac(fp_flicker,'from_end'):4.0f}%")

    print("\n=== INTERIOR GATE at obs==1: keep notes with from_start>=a AND from_end>=b ===")
    print("  (target = obs>=2 precision 0.972; want high P at useful R, beating idea1/2)")
    p, rc, f1, n, t = pr(R, lambda r: True, n_gt)
    print(f"  no gate (obs==1 all)            P={p:.3f} R={rc:.3f} F1={f1:.3f} (n={n})")
    for a in [0.0, 0.1, 0.2, 0.3]:
        for b in [0.0, 0.06, 0.1, 0.15]:
            p, rc, f1, n, t = pr(R, lambda r, a=a, b=b: r["from_start"] >= a and r["from_end"] >= b, n_gt)
            print(f"  a(start)>={a:.2f} b(end)>={b:.2f}   P={p:.3f} R={rc:.3f} F1={f1:.3f} (n={n})")

    print("\n=== for reference: does interior gate clean the SHIPPED obs>=2 set? ===")
    for a in [0.0, 0.1, 0.2]:
        for b in [0.0, 0.06, 0.1]:
            p, rc, f1, n, t = pr(R, lambda r, a=a, b=b: r["max_obs"] >= 2 and r["from_start"] >= a and r["from_end"] >= b, n_gt)
            print(f"  obs>=2 & a>={a:.2f} & b>={b:.2f}   P={p:.3f} R={rc:.3f} F1={f1:.3f} (n={n})")


if __name__ == "__main__":
    main()
