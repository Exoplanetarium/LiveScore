"""Figures for the emit-latency study. Reads _emit_latency_results.json and writes:
  - emit_latency_curve.png : latency<->precision/F1 tradeoff (one point per config)
  - emit_latency_ecdf.png  : per-note emit-latency ECDF per config (the distribution)

Literature latency regimes (audio->MIDI, MAESTRO) are drawn as context bands only;
their accuracy is on a different dataset/metric so we do NOT plot it on our P/F1 axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("benchmark_artifacts/_emit_latency_results.json")
OUTDIR = Path("benchmark_artifacts")

LIT = [  # (label, latency_ms, color) -- context only, audio->MIDI
    ("Hu '25 target\n10-30 ms", 20, "#c0392b"),
    ("Wei '24\n380 ms", 380, "#2980b9"),
]


def main() -> None:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))

    # ---- Figure 1: latency <-> precision / F1 curve -------------------------
    med = [r["emit_latency_median_ms"] for r in rows]
    p95 = [r["emit_latency_p95_ms"] for r in rows]
    prec = [r["precision"] for r in rows]
    f1 = [r["f1"] for r in rows]
    labels = [r["label"].replace("_int50", "") for r in rows]
    order = np.argsort(med)
    med, p95, prec, f1, labels = ([np.array(x)[order].tolist() for x in (med, p95, prec, f1, labels)])
    xerr = [[0] * len(med), [p - m for m, p in zip(med, p95)]]  # median..p95 whisker

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lbl, x, color in LIT:
        ax.axvline(x, ls=":", lw=1.2, color=color, alpha=0.7)
        ax.text(x, 0.02, lbl, rotation=90, va="bottom", ha="right",
                fontsize=7.5, color=color, transform=ax.get_xaxis_transform())
    ax.errorbar(med, prec, xerr=xerr, fmt="o-", color="#2c3e50", capsize=3,
                lw=2, ms=7, label="Precision (median..p95 latency)")
    ax.plot(med, f1, "s--", color="#16a085", lw=1.6, ms=6, label="F1")
    for x, y, lbl in zip(med, prec, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 7), fontsize=8)
    ax.set_xlabel("Emit latency (ms)  — median, whisker to p95")
    ax.set_ylabel("Onset accuracy @50 ms (gold12)")
    ax.set_title("Live audio→notation: latency vs accuracy (consensus-depth sweep)")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "emit_latency_curve.png", dpi=150)
    print(f"wrote {OUTDIR / 'emit_latency_curve.png'}")

    # ---- Figure 2: per-note emit-latency ECDF -------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for r in rows:
        vals = np.sort(np.asarray(r.get("latencies_ms") or [], dtype=np.float64))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        ax.step(vals, y, where="post", lw=1.8,
                label=f"{r['label'].replace('_int50','')}  (med {r['emit_latency_median_ms']:.0f} ms)")
    for gx in (50, 100, 150, 200):
        ax.axvline(gx, ls=":", color="gray", alpha=0.35)
    ax.set_xlabel("Per-note emit latency (ms)")
    ax.set_ylabel("Cumulative fraction of notes")
    ax.set_title("Emit-latency distribution per surface / consensus depth (gold12)")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "emit_latency_ecdf.png", dpi=150)
    print(f"wrote {OUTDIR / 'emit_latency_ecdf.png'}")


if __name__ == "__main__":
    main()
