import contextlib, io, json
from pathlib import Path
import main as live_main
from tune_continuous_stream_decoder import override_live_attrs, run_continuous_replay
from test_experiment import load_midi_notes, slice_gt_notes
from _live_vs_offline_mireval import MANIFEST, base_args, prf, score
from _live_nogate_mireval import GATES_OFF

d = json.loads(MANIFEST.read_text(encoding="utf-8"))
clips = dict(sorted({k: v["clip"] for k, v in d["clips"].items()}.items()))
cid, clip = list(clips.items())[1]  # clip_002
gt = slice_gt_notes(load_midi_notes(clip["midi_path"]), clip["start_sec"], clip["end_sec"])
rep = base_args()

print("baseline note_min_confidence:", live_main.LIVE_NOISE_FILTER_PROFILES["balanced"]["note_min_confidence"])
print("GATES_OFF note_min_confidence:", GATES_OFF["LIVE_NOISE_FILTER_PROFILES"]["balanced"]["note_min_confidence"])

def run(attrs):
    with override_live_attrs(attrs):
        nm = live_main.LIVE_NOISE_FILTER_PROFILES["balanced"]["note_min_confidence"]
        hs = live_main.STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            res = run_continuous_replay(clip, rep)
        return res["score_notes"], nm, hs

for label, attrs in [
    ("obs1_gated", {"STREAM_MIN_DISPLAY_OBSERVATIONS": 1}),
    ("obs1_nogate", {"STREAM_MIN_DISPLAY_OBSERVATIONS": 1, **GATES_OFF}),
    ("obs2_gated", {"STREAM_MIN_DISPLAY_OBSERVATIONS": 2}),
    ("obs2_nogate", {"STREAM_MIN_DISPLAY_OBSERVATIONS": 2, **GATES_OFF}),
]:
    notes, nm, hs = run(attrs)
    p, r, f = prf(*score(notes, gt)["onset50"])
    print(f"{label:12s} inside-ctx note_min_conf={nm} harm_supp={hs}  n_notes={len(notes)} P={p:.3f} R={r:.3f} F1={f:.3f}")
