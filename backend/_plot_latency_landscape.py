"""Latency vs accuracy landscape of published piano transcription systems, with
output-type encoded (MIDI events vs music notation). Shows that every low-latency
system emits MIDI, the only notation system is offline, and ours is the lone
low-latency + notation point.

HONESTY NOTE: accuracy is piece-macro note-onset F1 @50 ms on MAESTRO
test EXCEPT where flagged. Cross-system numbers are approximate -- datasets/splits,
latency definitions, and (Zeng=MV2H, Hu=strict 10-30 ms tol) metrics differ. This
is a landscape, not a controlled benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("benchmark_artifacts/latency_accuracy_landscape.png")
OURS_RESULTS = Path("benchmark_artifacts/live_vs_offline_mireval.json")
OFFLINE_X = 14000.0  # nominal x for systems with no real-time constraint

# name, latency_ms, lat_lo, lat_hi (None=no bar), F1, output('midi'/'notation'),
# offline(bool), note
SYS = [
    # ---- offline MIDI (no latency constraint; placed in offline band) ----
    ("Onsets & Frames '18", OFFLINE_X, None, None, 0.9532, "midi", True, ""),
    ("Semi-CRFs '21",       OFFLINE_X, None, None, 0.9611, "midi", True, ""),
    ("HPPNet-sp '22",       OFFLINE_X, None, None, 0.9718, "midi", True, ""),
    ("hFT-Transformer '23", OFFLINE_X, None, None, 0.9744, "midi", True, ""),
    # ---- finite-latency MIDI ----
    ("Hawthorne Seq2Seq '21", 4088, None, None, 0.9601, "midi", False, ""),
    ("Fernandez O&V '23",     6000, 4000, 9000, 0.9678, "midi", False, ""),
    ("Wei Streaming '24",      380, None, None, 0.9652, "midi", False, ""),
    ("Kwon multi-state '20",   200,  128,  320, 0.957,  "midi", False, ""),
    ("Mobile-AMT '24",         174, None, None, 0.9630, "midi", False, ""),
    ("Hu Causal-AMT '25",       20,   10,   30, 0.374,  "midi", False, ""),
    # ---- notation output ----
    ("Zeng A2S '24",        OFFLINE_X, None, None, 0.82, "notation", True, "MV2H"),
    ("LiveScore",              137, None, None, None,   "notation", False, ""),
]

C_MIDI = "#5b6770"
C_NOTE = "#16a085"
C_OURS = "#c0392b"

# per-system label placement: name -> (x-multiplier, dy, ha, va). None = suppress.
LBL = {
    "Mobile-AMT '24":        (1.0,  +0.022, "center", "bottom"),
    "Kwon multi-state '20":  (1.0,  -0.055, "center", "top"),
    "Wei Streaming '24":     (1.08, +0.020, "left",   "bottom"),
    "Hawthorne Seq2Seq '21": (1.0,  -0.040, "center", "top"),
    "Fernandez O&V '23":     (1.0,  +0.022, "center", "bottom"),
    "Hu Causal-AMT '25":     (1.18, +0.015, "left",   "bottom"),
    "Zeng A2S '24":          (0.60, +0.000, "right",  "center"),
    "LiveScore":             (1.0,  -0.028, "center", "top"),
}
OFFLINE_MIDI = {"Onsets & Frames '18", "Semi-CRFs '21", "HPPNet-sp '22", "hFT-Transformer '23"}


def load_full_maestro_livescore_f1(path: Path) -> float:
    """Only plot a piece-macro score from all 177 full MAESTRO test pieces."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("dataset") or {}
    if not dataset.get("complete_official_split") or dataset.get("unit") != "full_piece":
        raise SystemExit(
            f"{path} is not a complete full-piece MAESTRO test-split result."
        )
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
        (name, lat, lo, hi, ours_f1 if name == "LiveScore" else f1, out, offline, note)
        for name, lat, lo, hi, f1, out, offline, note in SYS
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    # regime shading
    ax.axvspan(8, 30, color="#e74c3c", alpha=0.07)
    ax.text(16, 0.325, "real-time\ntarget <30 ms", color="#c0392b", fontsize=7.5,
            ha="center", va="bottom")
    ax.axvspan(8000, 22000, color="#7f8c8d", alpha=0.08)
    ax.text(OFFLINE_X, 0.325, "offline", color="#555",
            fontsize=7.5, ha="center", va="bottom")

    for name, lat, lo, hi, f1, out, offline, note in systems:
        ours = name == "LiveScore"
        if out == "notation":
            color, marker, size = (C_OURS if ours else C_NOTE), "*", (460 if ours else 320)
        else:
            color, marker, size = C_MIDI, "o", 90
        ax.scatter([lat], [f1], s=size, marker=marker, color=color,
                   edgecolor="black", linewidth=0.8, zorder=5 if ours else 3)
        if name in OFFLINE_MIDI:
            continue  # labelled together in the offline box
        spec = LBL.get(name)
        if spec is None:
            continue
        mult, dy, ha, va = spec
        label = "Kwon '20" if name == "Kwon multi-state '20" else name
        label += f"\n({note})" if note else ""
        ax.annotate(label, (lat, f1), xytext=(lat * mult, f1 + dy),
                    fontsize=9 if ours else 8,
                    fontweight="bold" if ours else "normal", ha=ha, va=va,
                    color=color if (ours or out == "notation") else "#333")

    # combined label for the offline-MIDI cluster (placed in empty mid-right space)
    ax.text(2600, 0.50,
            "Offline MIDI:\n"
            "O&F 95.3  ·  Semi-CRF 96.1\nHPPNet 97.2  ·  hFT 97.4",
            fontsize=8, color="#333", ha="left", va="center",
            bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.9))

    ax.set_xscale("log")
    ax.set_xlim(8, 22000)
    ax.set_ylim(0.30, 1.04)
    ax.set_xlabel("Algorithmic latency (ms, log scale)")
    ax.set_ylabel("Piece-macro note-onset F1 @50 ms, MAESTRO test")
    ax.set_title("Piano transcription: latency vs accuracy, by output type")
    ax.grid(True, which="both", alpha=0.25)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MIDI,
               markeredgecolor="black", markersize=9, label="MIDI events"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_NOTE,
               markeredgecolor="black", markersize=15, label="Music notation"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_OURS,
               markeredgecolor="black", markersize=17, label="LiveScore"),
    ]
    ax.legend(handles=legend, loc="lower center", fontsize=8.5, ncol=3, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} (LiveScore macro-F1={ours_f1:.4f})")


if __name__ == "__main__":
    main()
