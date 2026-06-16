"""Gates-on vs gates-off comparison on the real pedaled recording.

Uses the offline model transcription as reference (same protocol as
diff_offline_vs_live.py). Reports recall AND live note count (precision proxy).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
from diff_offline_vs_live import load_audio, load_reference_notes, replay_live, match_reference
from tune_continuous_stream_decoder import override_live_attrs

WAV = Path("rhythm_training/test_inner_voice.wav")
MID = Path("rhythm_training/test_inner_voice.mid")

args = argparse.Namespace(
    onset_tol=0.20, inner_low=52, inner_high=76, packet_ms=40.0, context_sec=1.8,
    inference_interval_ms=70.0, trusted_delay_ms=180.0, commit_delay_ms=500.0,
    lock_delay_ms=2000.0, tail_padding_sec=0.3,
)

audio = load_audio(WAV)
reference = load_reference_notes(MID)

variants = {
    "gates_on (current)": {},
    "gates_off": {"STREAM_ATTACK_RATIO_STRONG": 0.0, "STREAM_ATTACK_DELTA_STRONG": -1.0},
}
for name, overrides in variants.items():
    with override_live_attrs(overrides):
        live = replay_live(audio, args)
    score = live["score_notes"]
    matched, missing = match_reference(reference, score, args.onset_tol)
    inner_missing = [m for m in missing if 52 <= m["midi_note"] <= 76]
    print("%-22s ref=%d score_notes=%d matched=%d recall=%.3f inner_missing=%d suppression=%s" % (
        name, len(reference), len(score), len(matched),
        len(matched) / max(1, len(reference)), len(inner_missing),
        {k: v for k, v in sorted(live["suppression"].items()) if v}))
