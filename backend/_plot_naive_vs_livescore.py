"""Figure for the primary must-beat baseline: LiveScore vs naive chunked streaming.

Reads naive_vs_livescore_mireval.json and writes naive_vs_livescore.png:
  - Left panel : grouped bars, LiveScore vs naive (primary chunk size) on the
                 higher-is-better axes (onset F1, strict F1@30, offset F1,
                 boundary recall on the naive seam grid).
  - Right panel: duplicate rate (lower-is-better) for naive across chunk sizes
                 vs LiveScore's flat line -- the re-onset-across-seam failure the
                 architecture removes.

House style matches latency_accuracy_landscape.png (teal = LiveScore, gray = baseline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_LIVE = "#16a085"   # LiveScore (notation, ours)
C_NAIVE = "#5b6770"  # naive baseline (gray)

RESULTS = Path("benchmark_artifacts/naive_vs_livescore_mireval.json")
OUT = Path("benchmark_artifacts/naive_vs_livescore.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--primary-chunk", default="1.0")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    by_chunk = data["by_chunk_sec"]
    n = data["n_clips"]
    P = by_chunk[args.primary_chunk]
    L, N = P["livescore"], P["naive_chunked"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})

    # ---- Panel A: higher-is-better grouped bars -----------------------------
    labels = ["onset F1\n@50 ms", "strict F1\n@30 ms", "offset-aware\nF1",
              "boundary recall\n(naive seam)"]
    live_vals = [L["onset50"]["f1"], L["strict_30ms"]["f1"], L["offset_f1"], L["boundary_recall"]]
    naive_vals = [N["onset50"]["f1"], N["strict_30ms"]["f1"], N["offset_f1"], N["boundary_recall"]]
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax0.bar(x - w / 2, live_vals, w, label="LiveScore (~100 ms)",
                 color=C_LIVE, edgecolor="black", linewidth=0.6)
    b2 = ax0.bar(x + w / 2, naive_vals, w,
                 label=f"Naive chunked ({P['naive_emit_latency_ms']:.0f} ms)",
                 color=C_NAIVE, edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax0.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5)
    # win deltas above each pair
    for xi, lv, nv in zip(x, live_vals, naive_vals):
        ax0.annotate(f"+{lv - nv:.3f}", (xi, max(lv, nv)), xytext=(0, 16),
                     textcoords="offset points", ha="center", fontsize=8,
                     fontweight="bold", color=C_LIVE)
    ax0.set_xticks(x); ax0.set_xticklabels(labels, fontsize=8.5)
    ax0.set_ylabel(f"score (48-clip MAESTRO-test, micro)")
    ax0.set_ylim(0, 1.08)
    ax0.set_title("Accuracy & boundary behavior (higher is better)", fontsize=10)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    # ---- Panel B: duplicate rate across chunk sizes (lower is better) --------
    chunks = sorted(by_chunk.values(), key=lambda d: d["chunk_sec"])
    cx = [c["chunk_sec"] for c in chunks]
    naive_dup = [c["naive_chunked"]["duplicates_per_100"] for c in chunks]
    live_dup = P["livescore"]["duplicates_per_100"]
    ax1.plot(cx, naive_dup, "o-", color=C_NAIVE, lw=2, ms=8,
             label="Naive chunked", markeredgecolor="black")
    for xi, yi in zip(cx, naive_dup):
        ax1.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=8, color=C_NAIVE)
    ax1.axhline(live_dup, ls="--", lw=2, color=C_LIVE,
                label=f"LiveScore ({live_dup:.1f})")
    ax1.set_xlabel("naive chunk size (s)  —  = emit latency")
    ax1.set_ylabel("duplicate onsets / 100 notes")
    ax1.set_title("Boundary duplicates (lower is better)", fontsize=10)
    ax1.set_xticks(cx)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Primary baseline: same model, LiveScore architecture vs naive chunked streaming",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
