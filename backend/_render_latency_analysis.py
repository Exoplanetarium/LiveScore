#!/usr/bin/env python3
"""Analyze OSMD render-latency captures for the audio->score paper.

The app logs one `[RENDER_LATENCY] {json}` line per live OSMD re-engrave
(PianoSheetMusic.tsx -> recordRenderLatency). Because every update is a
whole-score osmd.load()+render(), render latency scales with accumulated
score size. This harness parses those lines (from a raw log OR a .jsonl of
the extracted objects), fits renderMs vs score size, and prints/plots the
O(N) trend so we can quote a real audio->score render-latency figure and,
later, show a windowed-engrave fix flattening it.

Usage:
    python backend/_render_latency_analysis.py backend/_render_latency_capture_55s.jsonl
    python backend/_render_latency_analysis.py path/to/metro_log.txt        # raw log, auto-extracts
    python backend/_render_latency_analysis.py a.jsonl b.jsonl --label-a baseline --label-b windowed
    python backend/_render_latency_analysis.py capture.jsonl --plot backend/_render_latency.png
"""
import argparse
import json
import re
import sys

LINE_RE = re.compile(r"\[RENDER_LATENCY\]\s*(\{.*\})\s*$")


def load_samples(path):
    """Accept either a .jsonl of sample objects or a raw log with [RENDER_LATENCY] lines."""
    samples = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            m = LINE_RE.search(raw)
            blob = m.group(1) if m else (raw if raw.startswith("{") else None)
            if blob is None:
                continue
            try:
                samples.append(json.loads(blob))
            except json.JSONDecodeError:
                continue
    # Add elapsed-seconds (from first live render) so render latency can be
    # plotted against session progress — the axis on which a growing score
    # makes full-render climb while windowed stays flat.
    live = [s for s in samples if not s.get("appLoad") and "sentAt" in s]
    if live:
        t0 = min(s["sentAt"] for s in live)
        for s in samples:
            if "sentAt" in s:
                s["elapsed"] = (s["sentAt"] - t0) / 1000.0
    return samples


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def linfit(xs, ys):
    """Ordinary least squares y = a + b*x. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return my, 0.0, float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def split_freeze_ticks(live):
    """Tag each windowed sample as a freeze tick (frozenChunks increased vs the
    previous sample) or a steady tick. Returns (steady, freeze) sample lists.
    Full-render captures (no frozenChunks field) all count as steady."""
    steady, freeze = [], []
    prev = None
    for s in live:
        fc = s.get("frozenChunks")
        if fc is not None and prev is not None and fc > prev:
            freeze.append(s)
        else:
            steady.append(s)
        if fc is not None:
            prev = fc
    return steady, freeze


def summarize(samples, label, x_key="xmlLength"):
    live = [s for s in samples if not s.get("appLoad")]
    if not live:
        print(f"[{label}] no live samples")
        return None

    steady, freeze = split_freeze_ticks(live)
    windowed = any(s.get("frozenChunks") is not None for s in live)
    if windowed:
        steady_ms = [s["renderMs"] for s in steady]
        print(f"\n--- {label}: windowed engagement ---")
        print(f"  total renders {len(live)}  | steady {len(steady)}  freeze-ticks {len(freeze)}  "
              f"max frozenChunks {max((s.get('frozenChunks') or 0) for s in live)}")
        if steady_ms:
            print(f"  STEADY-STATE renderMs (the bounded tail): "
                  f"median {pct(steady_ms,50):.0f}  p95 {pct(steady_ms,95):.0f}  max {max(steady_ms):.0f}")
        if freeze:
            fz = [s["renderMs"] for s in freeze]
            print(f"  freeze-tick renderMs (occasional spikes): "
                  f"median {pct(fz,50):.0f}  p95 {pct(fz,95):.0f}  max {max(fz):.0f}")
        if max((s.get("frozenChunks") or 0) for s in live) == 0:
            print("  !! windowing NEVER ENGAGED (0 frozen chunks) — clip too short "
                  "or chunk threshold too high; every render was the full score.")

    ms = [s["renderMs"] for s in live]
    xs = [s.get(x_key, 0) for s in live]
    a, b, r2 = linfit(xs, ms)
    x_label = {
        "elapsed": "session time (s)",
        "xmlLength": "score size (MusicXML length, chars)",
        "measures": "rendered measures",
    }.get(x_key, x_key)
    # Slope is more intuitive per-note; xml is ~125-130 chars/note in this schema.
    chars_per_note = 127.0
    print(f"\n=== {label}  (n={len(live)} renders) ===")
    print(f"  renderMs   median {pct(ms,50):6.0f}   p95 {pct(ms,95):6.0f}   "
          f"min {min(ms):.0f}   max {max(ms):.0f}")
    print(f"  vs {x_key}: renderMs ~= {a:.0f} + {b*1000:.1f} ms / 1000 {x_key}   (R^2={r2:.3f})")
    print(f"    -> intercept (fixed cost): {a:.0f} ms")
    if x_key == "xmlLength":
        print(f"    -> marginal: ~{b*chars_per_note:.2f} ms per added note "
              f"(at ~{chars_per_note:.0f} chars/note)")
    elif x_key == "elapsed":
        print(f"    -> slope: ~{b:.1f} ms per second of session "
              f"({'climbing' if b > 5 else 'flat/declining'})")
    print(f"    -> first render {min(xs):.0f} {x_key}: {ms[0]:.0f} ms; "
          f"last {max(xs):.0f} {x_key}: {ms[-1]:.0f} ms  ({ms[-1]/ms[0]:.1f}x growth)")
    slope_str = (f"{b*1000:.1f}ms/1k" if x_key == "xmlLength"
                 else f"{b:.0f}ms/s" if x_key == "elapsed"
                 else f"{b:.2f}/unit")
    return {"label": label, "xs": xs, "ms": ms, "a": a, "b": b, "r2": r2,
            "x_label": x_label, "slope_str": slope_str}


def maybe_plot(fits, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available; skipping plot)")
        return
    plt.figure(figsize=(7, 4.5))
    for fit in fits:
        if fit is None:
            continue
        xs, ms = fit["xs"], fit["ms"]
        # matplotlib ignores legend labels starting with "_" (our filenames do).
        disp = fit["label"].lstrip("_") or fit["label"]
        plt.scatter(xs, ms, s=36, label=f"{disp} (measured)")
        if len(xs) >= 2:
            lo, hi = min(xs), max(xs)
            plt.plot([lo, hi], [fit["a"] + fit["b"] * lo, fit["a"] + fit["b"] * hi],
                     linestyle="--", alpha=0.8,
                     label=f"{disp} fit: {fit['a']:.0f}+{fit.get('slope_str','?')}, R^2={fit['r2']:.2f}")
    plt.xlabel(fits[0].get("x_label", "score size (MusicXML length, chars)")
               if fits and fits[0] else "x")
    plt.ylabel("render latency (ms)")
    plt.title("OSMD live re-engrave latency: full-render vs windowed")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nplot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one or two log/.jsonl files")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--x-key", default="xmlLength")
    ap.add_argument("--plot", default=None, help="output PNG path")
    args = ap.parse_args()

    fits = []
    labels = [args.label_a, args.label_b]
    for i, path in enumerate(args.paths[:2]):
        samples = load_samples(path)
        label = labels[i] or path.split("/")[-1].split("\\")[-1]
        fits.append(summarize(samples, label, args.x_key))

    if args.plot:
        maybe_plot(fits, args.plot)


if __name__ == "__main__":
    main()
