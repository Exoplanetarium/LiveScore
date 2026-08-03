"""Full-metric old-vs-new comparison for the obs2 + 50ms latency ship.

OLD defaults: obs3, inference_interval 70ms, trusted_delay 180ms
NEW defaults: obs2, inference_interval 50ms, trusted_delay 100ms

Runs the 48-clip continuous suite through ContinuousLiveStreamSession and
reports the full micro-averaged metric set per surface, with deltas, so we can
see whether anything other than note F1 regressed.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from tune_continuous_stream_decoder import (
    aggregate_clip_summaries,
    override_live_attrs,
    run_continuous_replay,
)

MANIFEST = Path("live_benchmark_replay_json/live_benchmark_replay_baseline_v1.json")

CONFIGS = {
    "OLD_obs3_int70_td180": dict(obs=3, interval=70.0, trusted=180.0),
    "NEW_obs2_int50_td100": dict(obs=2, interval=50.0, trusted=100.0),
}

METRICS = [
    "note_f1", "note_precision", "note_recall",
    "strict_30ms_f1", "strict_20ms_f1", "strict_10ms_f1",
    "cluster_f1", "cluster_precision", "cluster_recall",
    "offset_f1", "boundary_recall", "duplicates_per_100",
    "inference_ms_p95",
]


def base_args(trusted_ms, interval_ms):
    return SimpleNamespace(
        tail_padding_sec=0.6, context_sec=1.8, inference_interval_ms=interval_ms,
        trusted_delay_ms=trusted_ms, commit_delay_ms=500.0, lock_delay_ms=2000.0,
        packet_ms=40.0, chunk_seconds_for_boundary=0.6, eval_boundary_band_sec=0.10,
    )


def load_clips():
    d = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for cid, v in d["clips"].items():
        out[cid] = v["clip"]  # nested clip metadata has audio_path/midi_path/etc.
    return dict(sorted(out.items()))


def main():
    clips = load_clips()
    print(f"Loaded {len(clips)} clips from {MANIFEST.name}")
    agg = {}
    for name, cfg in CONFIGS.items():
        args = base_args(cfg["trusted"], cfg["interval"])
        results = []
        with override_live_attrs({"STREAM_MIN_DISPLAY_OBSERVATIONS": cfg["obs"]}):
            for cid, clip in clips.items():
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    results.append(run_continuous_replay(clip, args))
        agg[name] = {
            "score": aggregate_clip_summaries(results, "score"),
            "preview": aggregate_clip_summaries(results, "preview"),
        }
        s = agg[name]["score"]
        print(f"[{name}] score note_f1={s['note_f1']:.4f} "
              f"cluster_f1={s['cluster_f1']:.4f} p95={s['inference_ms_p95']:.1f}ms")

    old, new = list(CONFIGS.keys())
    for surface in ("score", "preview"):
        print(f"\n=== {surface} surface ===")
        print(f"{'metric':22s} {'OLD':>9s} {'NEW':>9s} {'delta':>9s}")
        for m in METRICS:
            o = float(agg[old][surface].get(m, 0.0))
            n = float(agg[new][surface].get(m, 0.0))
            flag = "  <== DROP" if (n < o - 0.002 and m not in ("duplicates_per_100", "inference_ms_p95")) else ""
            print(f"{m:22s} {o:9.4f} {n:9.4f} {n - o:+9.4f}{flag}")

    Path("benchmark_artifacts/_config_compare_results.json").write_text(
        json.dumps(agg, indent=2, default=float), encoding="utf-8")
    print("\nSaved benchmark_artifacts/_config_compare_results.json")


if __name__ == "__main__":
    main()
