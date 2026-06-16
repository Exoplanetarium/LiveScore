"""Per-clip comparison between two candidate JSONs (cluster F1 / note F1)."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
common = sorted(set(a["clips"]) & set(b["clips"]))
print("%-9s %18s %18s %18s" % ("clip", "clF1 a->b", "noteF1 a->b", "recall a->b"))
for cid in common:
    ma = a["clips"][cid]["surfaces"]["score"]
    mb = b["clips"][cid]["surfaces"]["score"]
    print("%-9s %8.4f->%8.4f %8.4f->%8.4f %8.4f->%8.4f" % (
        cid,
        ma["cluster"]["f1"], mb["cluster"]["f1"],
        ma["note"]["f1"], mb["note"]["f1"],
        ma["note"]["recall"], mb["note"]["recall"]))


def agg(d, keys):
    # aggregate over common clips only (micro-average like the harness)
    t = {"exact": 0.0, "cpred": 0.0, "cgt": 0.0, "m": 0.0, "p": 0.0, "g": 0.0}
    for cid in common:
        m = d["clips"][cid]["surfaces"]["score"]
        t["exact"] += m["cluster"]["exact_matches"]
        t["cpred"] += m["cluster"]["predicted"]
        t["cgt"] += m["cluster"]["ground_truth"]
        t["m"] += m["note"]["matched"]
        t["p"] += m["note"]["predicted"]
        t["g"] += m["note"]["ground_truth"]
    cp = t["exact"] / t["cpred"] if t["cpred"] else 0
    cr = t["exact"] / t["cgt"] if t["cgt"] else 0
    cf = 2 * cp * cr / (cp + cr) if cp + cr else 0
    np_ = t["m"] / t["p"] if t["p"] else 0
    nr = t["m"] / t["g"] if t["g"] else 0
    nf = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0
    return cf, nf, np_, nr


fa = agg(a, None)
fb = agg(b, None)
print("AGG(common) a: clF1=%.4f noteF1=%.4f prec=%.4f rec=%.4f" % fa)
print("AGG(common) b: clF1=%.4f noteF1=%.4f prec=%.4f rec=%.4f" % fb)
