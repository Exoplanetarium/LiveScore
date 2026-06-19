"""Latency/accuracy sweep for the continuous live-stream decoder.

Replays the gold12 suite through ContinuousLiveStreamSession (the shipped
/live/stream path) while varying the two display-latency knobs:
  - trusted_delay_ms (session constructor)
  - STREAM_MIN_DISPLAY_OBSERVATIONS (module attr; binding constraint at obs*70ms)

Latency-to-visible ~= max(trusted_delay_ms, obs * inference_interval_ms) is the
independent variable; F1/precision/recall at 50ms (+ strict 30ms) is measured.
Micro-averaged across clips. Reports both 'score' (stable) and 'preview'
(immediate, includes unstable) surfaces.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from tune_continuous_stream_decoder import (
    load_manifest,
    override_live_attrs,
    run_continuous_replay,
)

MANIFEST = Path(
    "benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json"
)
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


# (label, trusted_delay_ms, min_display_observations, inference_interval_ms)
# trusted_delay held low so the obs*interval gate binds and isolates the interval lever.
CONFIGS = [
    ("int40_obs2", 60, 2, 40.0),
    ("int50_obs2", 60, 2, 50.0),
]


def strict30(summary: dict) -> dict:
    row = (summary.get("strict_onset") or {}).get("30ms") or {}
    return dict(
        matched=float(row.get("matched", 0.0) or 0.0),
        predicted=float(row.get("predicted", 0.0) or 0.0),
        ground_truth=float(row.get("ground_truth", 0.0) or 0.0),
    )


def _f1(m, p, g):
    prec = m / p if p else 0.0
    rec = m / g if g else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)


def micro(results, surface):
    m = p = g = 0.0
    m30 = p30 = g30 = 0.0
    for r in results:
        surf = r["surfaces"][surface]
        n = surf["note"]
        m += float(n["matched"]); p += float(n["predicted"]); g += float(n["ground_truth"])
        s = strict30(surf)
        m30 += s["matched"]; p30 += s["predicted"]; g30 += s["ground_truth"]
    prec, rec, f1 = _f1(m, p, g)
    _, _, f1_30 = _f1(m30, p30, g30)
    return dict(precision=prec, recall=rec, f1=f1, f1_30=f1_30,
                matched=m, predicted=p, ground_truth=g)


def p95(vals):
    s = sorted(vals)
    if not s:
        return 0.0
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[k]


def main() -> None:
    clips = load_manifest(MANIFEST, [])
    print(f"Loaded {len(clips)} clips: {list(clips.keys())}")
    rows = []
    for label, td, obs, interval in CONFIGS:
        args = base_args(td, interval)
        results = []
        infer_ms = []
        with override_live_attrs({"STREAM_MIN_DISPLAY_OBSERVATIONS": obs}):
            for cid, clip in clips.items():
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    res = run_continuous_replay(clip, args)
                results.append(res)
                infer_ms.extend(res["timing"]["inference_ms"])
        latency = int(max(td, obs * interval))
        infer_p95 = p95(infer_ms)
        rtf = infer_p95 / interval if interval else 0.0
        for surface in ("score", "preview"):
            mm = micro(results, surface)
            row = dict(label=label, trusted_delay_ms=td, min_obs=obs,
                       interval_ms=interval, latency_ms=latency,
                       infer_p95_ms=infer_p95, hop_rtf=rtf, surface=surface, **mm)
            rows.append(row)
            print(
                f"{label:12s} int={interval:5.0f} obs={obs} lat~{latency:3d}ms "
                f"{surface:7s} P={mm['precision']:.3f} R={mm['recall']:.3f} "
                f"F1={mm['f1']:.3f} F1@30={mm['f1_30']:.3f} "
                f"infer_p95={infer_p95:5.1f}ms rtf={rtf:.2f} "
                f"({mm['matched']:.0f}/{mm['predicted']:.0f}/{mm['ground_truth']:.0f})"
            )
    out = Path("benchmark_artifacts/_latency_sweep_results.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
