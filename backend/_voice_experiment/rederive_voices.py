#!/usr/bin/env python3
"""GPU-free A/B harness for voice-assignment strategies.

Loads the dumped gold12 app payloads, re-derives voice_id / voice_ids / voice_index
on notes and chords with a pluggable strategy, and writes new payloads that can be
scored with tools/scorediff/run.js. The renderer's printed duration for each note is
the beat-IOI to the next note in the SAME voice lane, so voice assignment is a direct
lever on duration accuracy (see memory: score_vs_midi_timing_divergence).
"""
import argparse
import glob
import json
import os
import sys


def staff_of(pitch):
    # Mirror PianoSheetMusic.getStaff: octave = floor(midi/12)-1; octave<4 -> bass.
    if pitch is None:
        return "treble"
    return "bass" if pitch < 60 else "treble"


def note_pitch(n):
    p = n.get("midi_note")
    if p is None:
        mn = n.get("midi_notes") or []
        if mn:
            try:
                p = min(int(x) for x in mn)
            except (TypeError, ValueError):
                p = None
    try:
        return int(p) if p is not None else None
    except (TypeError, ValueError):
        return None


def set_voice(ev, hand, index):
    ev["voice_id"] = f"{hand}_voice_{index}"
    ev["voice_index"] = index
    ev["voice_assignment"] = "experiment"


# ─────────────────────────── strategies ───────────────────────────

def strat_pitch_lanes(hand, pitch):
    """Reproduce backend _voice_id_from_pitch (control)."""
    if pitch is None:
        return 1
    if hand == "treble":
        if pitch >= 72:
            return 0
        if pitch >= 60:
            return 1
        return 2
    if pitch < 48:
        return 0
    if pitch < 60:
        return 1
    return 2


def strat_single(hand, pitch):
    """Collapse each hand to a single voice (index 0 -> MusicXML voice 1/2)."""
    return 0


def assign_static(notes, chords, fn):
    for n in notes:
        p = note_pitch(n)
        hand = staff_of(p)
        set_voice(n, hand, fn(hand, p))
    for c in chords:
        mn = c.get("midi_notes") or []
        if mn:
            vids, vidx = [], []
            for m in mn:
                try:
                    mv = int(m)
                except (TypeError, ValueError):
                    continue
                hand = staff_of(mv)
                idx = fn(hand, mv)
                vids.append(f"{hand}_voice_{idx}")
                vidx.append(idx)
            if vids:
                c["voice_ids"] = vids
                c["voice_indices"] = vidx
                c["voice_id"] = vids[0]
                c["voice_index"] = vidx[0]
                c["voice_assignment"] = "experiment"
        else:
            p = note_pitch(c)
            hand = staff_of(p)
            set_voice(c, hand, fn(hand, p))


# ─────────────────── streaming voice separation ───────────────────

def assign_stream(notes, chords, max_voices=2, gap_reset=4.0):
    """Greedy nearest-pitch streaming separation, per hand.

    Process onset events in time order. Each voice tracks (last_pitch, last_time).
    A new note joins the voice whose last pitch is nearest, provided that voice's
    last onset is strictly earlier (an occupied-at-same-time voice forces a new
    lane). Voices that have been silent longer than gap_reset beats are free to be
    reused for any pitch. Capped at max_voices per hand; overflow joins nearest.
    Lower voice index == higher pitch line (so the top melody -> index 0).
    """
    # Build a flat onset list: notes (single) + chord onsets (each chord note is
    # its own onset event but shares the chord's time).
    events = []
    for n in notes:
        p = note_pitch(n)
        events.append({"t": float(n.get("time_seconds") or 0.0), "p": p,
                       "kind": "note", "ref": n})
    for ci, c in enumerate(chords):
        mn = c.get("midi_notes") or []
        t = float(c.get("time_seconds") or 0.0)
        if mn:
            for mi, m in enumerate(mn):
                try:
                    mv = int(m)
                except (TypeError, ValueError):
                    mv = None
                events.append({"t": t, "p": mv, "kind": "chordnote",
                               "ref": c, "ci": ci, "mi": mi, "n_in_chord": len(mn)})
        else:
            events.append({"t": t, "p": note_pitch(c), "kind": "note", "ref": c})

    events.sort(key=lambda e: (e["t"], -(e["p"] if e["p"] is not None else -999)))

    # per-hand voice state: list of dicts {last_pitch,last_time}
    voices = {"treble": [], "bass": []}
    # collect chord assignments
    chord_assign = {}  # ci -> list[idx] aligned with midi_notes

    for e in events:
        p = e["p"]
        hand = staff_of(p)
        vs = voices[hand]
        now = e["t"]
        # candidate voices: those whose last_time < now (not simultaneous)
        best = None
        best_d = None
        for vi, v in enumerate(vs):
            if v["last_time"] >= now - 1e-4:
                continue  # occupied at this onset -> cannot reuse
            d = abs((p if p is not None else 60) - v["last_pitch"])
            # silent-long voices get a discount so they can restart cheaply
            if best is None or d < best_d:
                best, best_d = vi, d
        if best is None:
            if len(vs) < max_voices:
                vs.append({"last_pitch": p if p is not None else 60, "last_time": now})
                idx = len(vs) - 1
            else:
                # forced overflow: pick nearest regardless of occupancy
                idx = min(range(len(vs)),
                          key=lambda i: abs((p if p is not None else 60) - vs[i]["last_pitch"]))
                vs[idx] = {"last_pitch": p if p is not None else 60, "last_time": now}
        else:
            idx = best
            vs[idx] = {"last_pitch": p if p is not None else 60, "last_time": now}

        if e["kind"] == "chordnote":
            chord_assign.setdefault(e["ci"], [None] * e["n_in_chord"])
            chord_assign[e["ci"]][e["mi"]] = (hand, idx)
        else:
            set_voice(e["ref"], hand, idx)

    # Re-rank voice indices per hand so that index 0 = highest mean pitch (top line).
    # Build mapping from raw idx -> ranked idx, per hand, using mean pitch.
    # (skip for simplicity if only 1 voice)
    for ci, alist in chord_assign.items():
        c = chords[ci]
        vids, vidx = [], []
        for a in alist:
            if a is None:
                vids.append("treble_voice_0")
                vidx.append(0)
            else:
                hand, idx = a
                vids.append(f"{hand}_voice_{idx}")
                vidx.append(idx)
        c["voice_ids"] = vids
        c["voice_indices"] = vidx
        c["voice_id"] = vids[0]
        c["voice_index"] = vidx[0]
        c["voice_assignment"] = "experiment"


STRATS = {
    "pitch_lanes": lambda notes, chords: assign_static(notes, chords, strat_pitch_lanes),
    "single": lambda notes, chords: assign_static(notes, chords, strat_single),
    "stream2": lambda notes, chords: assign_stream(notes, chords, max_voices=2),
    "stream3": lambda notes, chords: assign_stream(notes, chords, max_voices=3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=list(STRATS))
    ap.add_argument("--in-glob", default="backend/_tmp_rescore_gold12/_tmp_app_payload_clip_*.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.in_glob))
    if not files:
        print("no input files", file=sys.stderr)
        sys.exit(1)
    for f in files:
        d = json.load(open(f))
        notes = d.get("notes") or []
        chords = d.get("chords") or []
        STRATS[args.strategy](notes, chords)
        out = os.path.join(args.out_dir, os.path.basename(f))
        json.dump(d, open(out, "w"))
    print(f"wrote {len(files)} payloads to {args.out_dir} with strategy={args.strategy}")


if __name__ == "__main__":
    main()
