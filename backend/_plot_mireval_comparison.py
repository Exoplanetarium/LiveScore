"""Note-onset F1 comparison: LiveScore vs published piano-transcription systems.

Companion bar chart to latency_accuracy_landscape.png. Same numbers as
_plot_latency_landscape.py so the paper stays internally consistent. Bars are
onset F1 @50 ms on MAESTRO test, sorted, colored by regime (offline MIDI /
finite-latency MIDI / live notation), with each system's latency labeled.

HONESTY (in the caption): cross-system numbers use different splits and are
approximate. Two systems are intentionally NOT drawn as F1 bars because their
headline number is a different metric and a bar would misrepresent them:
  - Hu Causal-AMT '25: reported at strict 10-30 ms tolerance (not @50 ms).
  - Zeng A2S '24: reported as MV2H, not onset F1 (see mv2h_vs_zeng.png).
Both are shown in latency_accuracy_landscape.png where the axes make the
metric/latency differences explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path("benchmark_artifacts/mireval_system_comparison.png")
OURS_RESULTS = Path("benchmark_artifacts/live_vs_offline_mireval.json")

# name, onset-F1 @50ms MAESTRO, latency_ms (None=offline), regime, note
# regime: 'offline_midi' | 'stream_midi' | 'live_notation'
SYS = [
    ("hFT-Transformer '23",   0.9744, None, "offline_midi", ""),
    ("HPPNet-sp '22",         0.9718, None, "offline_midi", ""),
    ("Fernandez O&V '23",     0.9678, 6000, "stream_midi",  ""),
    ("Mobile-AMT '24",        0.9630, 174,  "stream_midi",  ""),
    ("Wei Streaming '24",     0.9652, 380,  "stream_midi",  ""),
    ("Semi-CRFs '21",         0.9611, None, "offline_midi", ""),
    ("Hawthorne Seq2Seq '21", 0.9601, 4088, "stream_midi",  ""),
    ("Kwon multi-state '20",  0.9570, 200,  "stream_midi",  ""),
    ("LiveScore (ours)",      None,   137,  "live_notation", ""),
    ("Onsets & Frames '18",   0.9532, None, "offline_midi", ""),
]

C = {
    "offline_midi": "#95a5a6",
    "stream_midi": "#5b6770",
    "live_notation": "#16a085",
}
REGIME_LABEL = {
    "offline_midi": "Offline MIDI",
    "stream_midi": "Streaming / finite-latency MIDI",
    "live_notation": "Live notation (ours)",
}


def lat_label(latency_ms):
    if latency_ms is None:
        return "offline"
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.1f} s"
    return f"{latency_ms:.0f} ms"


def load_full_maestro_livescore_f1(path: Path) -> float:
    """Load piece-macro F1, matching the averaging used by published tables."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("dataset") or {}
    if not dataset.get("complete_official_split"):
        raise SystemExit(
            f"{path} is not a complete official MAESTRO test-split run "
            f"(scope={dataset.get('evaluation_scope', 'legacy/unknown')!r}). "
            "Run _live_vs_offline_mireval.py without --limit before making the paper figure."
        )
    if dataset.get("unit") != "full_piece":
        raise SystemExit(f"{path} does not contain full-piece evaluation results.")
    try:
        return float(payload["summary"]["live"]["onset50"]["macro_f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{path} is missing live onset50 macro-F1: {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ours-results", default=str(OURS_RESULTS))
    args = ap.parse_args()

    ours_f1 = load_full_maestro_livescore_f1(Path(args.ours_results))
    systems = [
        (name, ours_f1 if name == "LiveScore (ours)" else val, lat, regime, note)
        for name, val, lat, regime, note in SYS
    ]
    rows = sorted(systems, key=lambda r: r[1])  # ascending -> largest on top in barh
    names = [r[0] for r in rows]
    f1 = [r[1] for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for yi, (name, val, lat, regime, note) in zip(y, rows):
        ours = regime == "live_notation"
        ax.barh(yi, val, color=C[regime], edgecolor="black",
                linewidth=1.4 if ours else 0.6, height=0.68, zorder=3)
        # value at bar end
        ax.text(val + 0.0015, yi, f"{val:.4f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold" if ours else "normal",
                color="#111")
        # latency chip inside the bar left edge (blended transform: x in axes
        # fraction so it's always visible regardless of the zoomed data xlim)
        ax.text(0.012, yi, lat_label(lat), transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color="white",
                fontweight="bold", zorder=4,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="#1a1a1a")])

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r[3] == "live_notation":
            tick.set_fontweight("bold")
            tick.set_color("#0e6b57")

    ax.set_xlim(0.90, 0.99)
    ax.set_xlabel("Piece-macro note-onset F1 @50 ms, MAESTRO test  (higher is better)  ·  chip = reported latency")
    ax.set_title("LiveScore vs published piano-transcription systems", fontsize=12, pad=34)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.axvline(ours_f1, ls=":", lw=1.2, color="#16a085", alpha=0.7, zorder=2)

    legend = [Patch(facecolor=C[k], edgecolor="black", label=REGIME_LABEL[k])
              for k in ("offline_midi", "stream_midi", "live_notation")]
    legend.append(Line2D([0], [0], color="#16a085", ls=":", lw=1.2, label="LiveScore F1"))
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, fontsize=8.3, framealpha=0.95, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.3)

    cap = ("LiveScore is evaluated on all 177 full-length pieces in the official MAESTRO v3 test split;\n"
           "published values retain their papers' stated MAESTRO protocols. The chip is reported algorithmic\n"
           "latency. Hu Causal-AMT '25 (strict 10-30 ms tol) and\n"
           "Zeng A2S '24 (MV2H) use non-comparable metrics and are omitted here — see the latency–\n"
           "accuracy landscape and MV2H figures.")
    fig.text(0.015, 0.015, cap, fontsize=7.2, color="#555", ha="left", va="bottom")

    fig.subplots_adjust(left=0.235, right=0.965, top=0.87, bottom=0.28)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
