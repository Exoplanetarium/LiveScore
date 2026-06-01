"""Dump the exact display-surface payload the app renders the score from.

The live app builds both the on-screen score (via generateMusicXML) and the
exported MIDI from the SAME payload returned by /live/check-refinement, i.e.
``display_state["notes"]`` / ``display_state["chords"]`` plus the session bpm.
This script replays a handful of benchmark clips and writes that payload to disk
so an offline harness (tools/scorediff) can compare the rendered score XML
against the raw MIDI events and pinpoint where chord membership diverges.

Run from the backend/ directory (needs the model + GPU, same as the benchmark):

    python dump_app_payloads.py                 # first 5 manifest clips
    python dump_app_payloads.py --clips 5
    python dump_app_payloads.py --clip-ids clip_017 clip_031

Writes backend/_tmp_app_payload_<clip_id>.json for each clip.
"""

import argparse
import asyncio
import json
from pathlib import Path

import test_experiment as te


async def _run(args: argparse.Namespace) -> None:
    selected = te.load_benchmark_manifest(args.manifest, args.clip_ids)
    clip_items = list(selected.items())
    if not args.clip_ids:
        clip_items = clip_items[: args.clips]

    if not clip_items:
        print("No clips selected.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm the model once so the first real clip is not penalised / skewed.
    first = clip_items[0][1]
    warm_audio = te.load_audio_excerpt(
        first["audio_path"],
        first["start_sec"],
        min(1.0, float(first["duration_sec"])),
        te.TARGET_SR,
    )
    await te.run_live_excerpt(
        warm_audio,
        adaptive_onset_threshold=False,
        chunk_seconds=args.chunk_seconds,
        noise_profile=args.noise_profile,
    )

    for clip_id, clip in clip_items:
        audio = te.load_audio_excerpt(
            clip["audio_path"],
            clip["start_sec"],
            float(clip["duration_sec"]),
            te.TARGET_SR,
        )
        # Same call the control arm makes, with the app payload captured.
        run = await te.run_live_excerpt(
            audio,
            adaptive_onset_threshold=False,
            chunk_seconds=args.chunk_seconds,
            noise_profile=args.noise_profile,
            capture_display_inputs=True,
        )

        payload = {
            "clip_id": clip_id,
            "title": clip.get("title"),
            "bpm": run.get("app_bpm", 0.0),
            "notes": run.get("app_notes", []),
            "chords": run.get("app_chords", []),
            "gt_midi_path": clip.get("midi_path"),
            "start_sec": clip.get("start_sec"),
            "end_sec": clip.get("end_sec"),
        }
        out_path = out_dir / f"_tmp_app_payload_{clip_id}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"wrote {out_path.name}  "
            f"notes={len(payload['notes'])} chords={len(payload['chords'])} "
            f"bpm={float(payload['bpm']):.1f}"
        )


def main() -> None:
    default_manifest = str(Path(__file__).resolve().parent / "live_benchmark_replay_auto_v2.json")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=default_manifest)
    ap.add_argument("--clips", type=int, default=5, help="Number of leading manifest clips to dump.")
    ap.add_argument("--clip-ids", nargs="*", default=None, help="Explicit clip ids (overrides --clips).")
    ap.add_argument("--chunk-seconds", type=float, default=0.6)
    ap.add_argument("--noise-profile", default="balanced")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
