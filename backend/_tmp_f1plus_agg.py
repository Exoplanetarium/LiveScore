"""Aggregate-metrics comparison across candidate JSONs."""
import glob
import json
import sys

paths = []
for arg in sys.argv[1:]:
    paths.extend(sorted(glob.glob(arg)))
print("%-28s %7s %7s %7s %7s %7s %7s %7s" % (
    "candidate", "noteF1", "prec", "recall", "clF1", "clPrec", "clRec", "dup100"))
for p in paths:
    d = json.load(open(p))
    a = d["aggregate"]["score"]
    name = d["candidate"]["name"]
    print("%-28s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f" % (
        name, a["note_f1"], a["note_precision"], a["note_recall"],
        a["cluster_f1"], a["cluster_precision"], a["cluster_recall"],
        a["duplicates_per_100"]))
