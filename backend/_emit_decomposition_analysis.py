#!/usr/bin/env python3
"""Decide whether idea #4 (targeted trailing-edge re-decode) can help, from a
real device capture.

The streaming session now stamps, for every note at the instant it first
displays (`candidate` -> `active`), which term bound the emit wait:

    binding = "trusted"      -> onset+trusted_delay was the later (gating) term
    binding = "persistence"  -> the 2nd-observation accrual was the later term

The frontend logs one `[EMIT_DECOMP] {json}` line per such promotion. Idea #4
(speed up the 2nd observation via a small extra decode) can ONLY reduce emit
latency for the *persistence*-bound notes; for *trusted*-bound notes the floor
is trusted_delay (the precision knob that interior-trust already proved can't be
lowered for free). So the go/no-go for #4 is simply: what fraction of notes are
persistence-bound, and by how much (term_gap) could a faster 2nd obs move them?

Usage:
    python backend/_emit_decomposition_analysis.py metro_log.txt
    python backend/_emit_decomposition_analysis.py capture.jsonl
"""
import argparse
import json
import re
import sys

LINE_RE = re.compile(r"\[EMIT_DECOMP\]\s*(\{.*\})\s*$")


def load(path):
    out = []
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
                out.append(json.loads(blob))
            except json.JSONDecodeError:
                continue
    return out


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    samples = load(args.path)
    if not samples:
        print("no [EMIT_DECOMP] samples found")
        sys.exit(1)

    n = len(samples)
    trusted = [s for s in samples if s.get("binding") == "trusted"]
    persistence = [s for s in samples if s.get("binding") == "persistence"]
    deltas = [s["delta_active_ms"] for s in samples if "delta_active_ms" in s]
    td = [s.get("trusted_delay_ms") for s in samples if s.get("trusted_delay_ms") is not None]
    trusted_delay_ms = td[0] if td else None

    print(f"=== emit decomposition  (n={n} promotions) ===")
    if trusted_delay_ms is not None:
        print(f"  trusted_delay = {trusted_delay_ms} ms (the precision floor)")
    print(f"  delta_active (onset->first display): "
          f"median {pct(deltas,50):.0f}  p95 {pct(deltas,95):.0f}  "
          f"min {min(deltas):.0f}  max {max(deltas):.0f}  ms")
    print(f"  binding term:  trusted {len(trusted)} ({100*len(trusted)/n:.0f}%)  |  "
          f"persistence {len(persistence)} ({100*len(persistence)/n:.0f}%)")

    # The recoverable headroom for idea #4 = for persistence-bound notes, how far
    # above trusted_delay did they sit (term_gap). That is the absolute max a
    # faster 2nd observation could shave; a small gap means even success is moot.
    gaps = [s.get("term_gap_ms", 0) for s in persistence]
    if gaps:
        print(f"  [persistence-bound] term_gap above trusted_delay: "
              f"median {pct(gaps,50):.0f}  p95 {pct(gaps,95):.0f}  max {max(gaps):.0f}  ms")
        recoverable = sum(gaps) / n
        print(f"  -> ceiling for idea #4 ~= {recoverable:.0f} ms amortized over all notes "
              f"(only persistence-bound notes, only down to trusted_delay)")

    print("\nverdict:")
    frac_p = len(persistence) / n
    if frac_p < 0.2:
        print("  trusted_delay dominates -> idea #4 is MOOT. The emit floor is the")
        print("  precision knob (trusted_delay); a faster 2nd observation can't beat it.")
    elif gaps and pct(gaps, 50) < 25:
        print("  persistence binds often but only by a hair -> idea #4 ceiling is tiny;")
        print("  not worth the poorly-calibrated short-window re-decode risk.")
    else:
        print("  persistence-bound notes carry real headroom -> idea #4 (or simply a")
        print("  higher decode cadence if GPU has slack) is worth a controlled try.")


if __name__ == "__main__":
    main()
