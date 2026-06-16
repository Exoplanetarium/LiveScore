"""Compute current cluster-F1 vs pairwise co-onset F1 on the REAL production
decoder output (idea3 baseline_current.json score_notes) vs gold12 GT."""
import json
import numpy as np
import test_experiment as te

BASE = json.load(open("benchmark_artifacts/idea3_onset_sweep/baseline_current.json"))
MAN = json.load(open("benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json"))


def match_notes(pred, gt, onset_tol=0.05):
    gt_used = set(); pairs = []
    for pi, p in enumerate(pred):
        best = besterr = None
        for gi, g in enumerate(gt):
            if gi in gt_used or int(p["midi_note"]) != int(g["midi_note"]):
                continue
            err = abs(float(p["onset_time"]) - float(g["onset_time"]))
            if err > onset_tol:
                continue
            if besterr is None or err < besterr:
                best, besterr = gi, err
        if best is not None:
            gt_used.add(best); pairs.append((pi, best))
    return pairs


def pairwise_coonset_prf(pred, gt, W=0.05):
    pairs = match_notes(pred, gt)
    if len(pairs) < 2:
        return 1.0, 1.0, 1.0
    tp = fp = fn = 0
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            pa, ga = pairs[a]; pb, gb = pairs[b]
            pt = abs(pred[pa]["onset_time"] - pred[pb]["onset_time"]) <= W
            gtt = abs(gt[ga]["onset_time"] - gt[gb]["onset_time"]) <= W
            if gtt and pt: tp += 1
            elif pt and not gtt: fp += 1
            elif gtt and not pt: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 1.0
    return prec, rec, f1


hdr = f"{'clip':<9} {'npred':>5} {'ngt':>4} {'cur_clF1':>8} {'pair_F1':>8} {'pair_P':>7} {'pair_R':>7}"
print(hdr)
rows = []
for cid, c in MAN["clips"].items():
    gt = te.load_midi_notes(c["excerpt_midi_path"])
    pred = BASE["clips"][cid]["score_notes"]
    cur = te.compute_onset_cluster_metrics(pred, gt)["f1"]
    pp, pr, pf = pairwise_coonset_prf(pred, gt)
    rows.append((cid, len(pred), len(gt), cur, pf, pp, pr))
    print(f"{cid:<9} {len(pred):>5} {len(gt):>4} {cur:>8.3f} {pf:>8.3f} {pp:>7.3f} {pr:>7.3f}")

# weighted by GT note count (dense clips dominate)
arr = np.array([(r[3], r[4], r[5], r[6], r[2]) for r in rows], dtype=float)
w = arr[:, 4]
print("-" * len(hdr))
print(f"{'MEAN':<9} {'':>5} {'':>4} {arr[:,0].mean():>8.3f} {arr[:,1].mean():>8.3f} "
      f"{arr[:,2].mean():>7.3f} {arr[:,3].mean():>7.3f}   (unweighted)")
print(f"{'wMEAN':<9} {'':>5} {'':>4} {np.average(arr[:,0],weights=w):>8.3f} "
      f"{np.average(arr[:,1],weights=w):>8.3f} {np.average(arr[:,2],weights=w):>7.3f} "
      f"{np.average(arr[:,3],weights=w):>7.3f}   (GT-note-weighted)")
print()
print("cur_clF1 = current single-linkage 50ms cluster F1 (per-clip).")
print("pair_*   = pairwise co-onset (W=50ms) on the SAME real predictions.")
