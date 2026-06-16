"""Diagnostic: how much cluster-F1 does a PERFECT-pitch transcriber lose purely to
onset jitter under the current single-linkage 50ms metric? Establishes the
measurement-artifact ceiling and compares against grid-quantized clustering.

Temp / throwaway. Reverted after analysis.
"""
import json
import numpy as np

import test_experiment as te

rng = np.random.default_rng(0)
MANIFEST = "benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json"
N_TRIALS = 40


def jitter(notes, sigma_sec):
    out = []
    for n in notes:
        d = dict(n)
        delta = float(rng.normal(0.0, sigma_sec))
        d["onset_time"] = float(n["onset_time"]) + delta
        d["offset_time"] = float(n.get("offset_time", n["onset_time"])) + delta
        out.append(d)
    out.sort(key=lambda e: (e["onset_time"], e["midi_note"]))
    return out


# --- alternative clustering: quantize onsets to a metrical grid, group by cell ---
def grid_cluster(notes, bpm, subdiv=4):
    """Snap each onset to nearest grid cell (beat/subdiv) and group by cell index.
    Symmetric: both pred and GT use the SAME absolute grid, so a wide chord lands
    in one cell regardless of small per-note timing differences."""
    if not notes:
        return []
    beat_sec = 60.0 / max(1e-6, bpm)
    cell = beat_sec / max(1, subdiv)
    buckets = {}
    for n in notes:
        idx = round(float(n["onset_time"]) / cell)
        buckets.setdefault(idx, []).append(n)
    return [buckets[k] for k in sorted(buckets)]


def cluster_f1_with(pred, gt, cluster_fn):
    pc = cluster_fn(pred)
    gc = cluster_fn(gt)
    exact = 0
    gt_matched = set()
    for p in pc:
        pa = te._cluster_anchor_time(p)
        best = None
        besterr = None
        for i, g in enumerate(gc):
            if i in gt_matched:
                continue
            err = abs(pa - te._cluster_anchor_time(g))
            if err > 0.05:
                continue
            if besterr is None or err < besterr:
                best, besterr = i, err
        if best is None:
            continue
        gt_matched.add(best)
        if te._cluster_pitch_signature(p) == te._cluster_pitch_signature(gc[best]):
            exact += 1
    prec = exact / len(pc) if pc else 0.0
    rec = exact / len(gc) if gc else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1


def grid_cluster_f1(pred, gt, bpm, subdiv):
    # match grid cells by integer cell index (exact-cell), not anchor time
    beat_sec = 60.0 / max(1e-6, bpm)
    cell = beat_sec / max(1, subdiv)

    def cells(notes):
        b = {}
        for n in notes:
            idx = round(float(n["onset_time"]) / cell)
            b.setdefault(idx, []).append(n)
        return b

    pc = cells(pred)
    gc = cells(gt)
    exact = 0
    for idx, g in gc.items():
        if idx in pc and te._cluster_pitch_signature(pc[idx]) == te._cluster_pitch_signature(g):
            exact += 1
    prec = exact / len(pc) if pc else 0.0
    rec = exact / len(gc) if gc else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1


def match_notes(pred, gt, onset_tol=0.05):
    """Greedy pitch+onset match; return list of (pred_idx, gt_idx)."""
    gt_used = set()
    pairs = []
    for pi, p in enumerate(pred):
        best = None
        besterr = None
        for gi, g in enumerate(gt):
            if gi in gt_used:
                continue
            if int(p["midi_note"]) != int(g["midi_note"]):
                continue
            err = abs(float(p["onset_time"]) - float(g["onset_time"]))
            if err > onset_tol:
                continue
            if besterr is None or err < besterr:
                best, besterr = gi, err
        if best is not None:
            gt_used.add(best)
            pairs.append((pi, best))
    return pairs


def pairwise_coonset_f1(pred, gt, W=0.05):
    """Agreement of the 'struck together' relation over commonly-matched notes.
    For every unordered pair of matched notes, both pred and GT vote together/apart
    by |onset_i - onset_j| <= W. F1 over the 'together' relation. Anchor-free,
    no transitive chaining -> robust to where a wide chord's boundary falls."""
    pairs = match_notes(pred, gt)
    if len(pairs) < 2:
        return 1.0
    tp = fp = fn = 0
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            pi_a, gi_a = pairs[a]
            pi_b, gi_b = pairs[b]
            pred_tog = abs(pred[pi_a]["onset_time"] - pred[pi_b]["onset_time"]) <= W
            gt_tog = abs(gt[gi_a]["onset_time"] - gt[gi_b]["onset_time"]) <= W
            if gt_tog and pred_tog:
                tp += 1
            elif pred_tog and not gt_tog:
                fp += 1
            elif gt_tog and not pred_tog:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 1.0


m = json.load(open(MANIFEST))
clips = m["clips"]

hdr = f"{'clip':<9} {'self':>5} {'cur5':>6} {'cur10':>6} {'cur15':>6} {'pair5':>6} {'pair10':>6} {'pair15':>6}"
print(hdr)
agg = {k: [] for k in ["self", "cur5", "cur10", "cur15", "pair5", "pair10", "pair15"]}
for cid, c in clips.items():
    gt = te.load_midi_notes(c["excerpt_midi_path"])
    self_f1 = cluster_f1_with(gt, gt, te.cluster_note_onsets)

    def avg(sigma, fn):
        return float(np.mean([fn(jitter(gt, sigma)) for _ in range(N_TRIALS)]))

    cur5 = avg(0.005, lambda p: cluster_f1_with(p, gt, te.cluster_note_onsets))
    cur10 = avg(0.010, lambda p: cluster_f1_with(p, gt, te.cluster_note_onsets))
    cur15 = avg(0.015, lambda p: cluster_f1_with(p, gt, te.cluster_note_onsets))
    pair5 = avg(0.005, lambda p: pairwise_coonset_f1(p, gt))
    pair10 = avg(0.010, lambda p: pairwise_coonset_f1(p, gt))
    pair15 = avg(0.015, lambda p: pairwise_coonset_f1(p, gt))
    for k, v in zip(agg, [self_f1, cur5, cur10, cur15, pair5, pair10, pair15]):
        agg[k].append(v)
    print(f"{cid:<9} {self_f1:>5.2f} {cur5:>6.3f} {cur10:>6.3f} {cur15:>6.3f} "
          f"{pair5:>6.3f} {pair10:>6.3f} {pair15:>6.3f}")

print("-" * len(hdr))
print(f"{'MEAN':<9} {np.mean(agg['self']):>5.2f} {np.mean(agg['cur5']):>6.3f} "
      f"{np.mean(agg['cur10']):>6.3f} {np.mean(agg['cur15']):>6.3f} "
      f"{np.mean(agg['pair5']):>6.3f} {np.mean(agg['pair10']):>6.3f} {np.mean(agg['pair15']):>6.3f}")
print()
print("self=GT-vs-GT (sanity, =1.0). curN=current 50ms single-linkage cluster-F1, perfect pitch + Nms jitter.")
print("pairN=pairwise co-onset (W=50ms) agreement F1, perfect pitch + Nms jitter. Higher=more jitter-robust.")
