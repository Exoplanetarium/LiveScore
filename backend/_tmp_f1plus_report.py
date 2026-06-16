"""Per-clip report for a continuous-stream benchmark candidate JSON."""
import json
import sys

path = sys.argv[1]
d = json.load(open(path))
print("elapsed_sec:", d.get("elapsed_sec"))
rows = []
for cid, c in d["clips"].items():
    m = c["surfaces"]["score"]
    n = m["note"]
    cl = m["cluster"]
    rows.append((
        cid,
        (c["clip"]["title"] or "")[:38],
        n["recall"], n["precision"], n["f1"],
        cl["f1"], cl["ground_truth"], cl["exact_matches"],
        cl["underclustered_matches"], cl["overclustered_matches"],
        cl["pitch_conflict_matches"], cl["unmatched_ground_truth"],
    ))
rows.sort(key=lambda r: r[5])
hdr = ("clip", "title", "rec", "prec", "nF1", "clF1", "gtCl", "exact", "under", "over", "conf", "unmGT")
print("%-9s%-40s%6s%6s%6s%6s%5s%6s%6s%5s%5s%6s" % hdr)
for r in rows:
    print("%-9s%-40s%6.3f%6.3f%6.3f%6.3f%5d%6d%6d%5d%5d%6d" % r)
