import * as ScreenOrientation from "expo-screen-orientation";
import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import { Alert, StyleSheet, TouchableOpacity, View } from "react-native";
import { WebView, WebViewMessageEvent } from "react-native-webview";
import { ThemedText } from "./ThemedText";
import { OSMD_HTML } from "./osmdHTML";

interface NoteResult {
  time_seconds: number;
  start_beat?: number;
  end_beat?: number;
  frame_index?: number;
  midi_note: number;
  note_name?: string;
  frequency_hz?: number;
  method?: string;
  confidence?: number;
  offset_seconds?: number;
  duration_seconds?: number;
  offset_frame?: number;
  hand?: "bass" | "treble"; // Neural output: bass/treble hand assignment
  note_value?: "whole" | "half" | "quarter" | "eighth" | "16th" | "32nd";
  note_divisions?: number;
  duration_source?: string;
  timing_authority?: string;
  rest_after_beats?: number;
  local_beat_duration?: number;
  dotted?: boolean;
  // Triplet fields
  triplet?: boolean;
  triplet_position?: "start" | "middle" | "end";
  triplet_type?: "half" | "quarter" | "eighth" | "16th" | "32nd";
  actual_notes?: number; // 3 for triplet
  normal_notes?: number; // 2 for triplet
  // Ornament fields
  ornament?:
    | "trill"
    | "grace"
    | "mordent_upper"
    | "mordent_lower"
    | "turn_upper"
    | "turn_inverted";
  trill_to?: number; // MIDI note to trill to
  trill_interval?: number; // Interval in semitones
  grace_type?: "acciaccatura" | "appoggiatura"; // Slashed vs unslashed grace note
}

interface ChordResult {
  time_seconds: number;
  start_beat?: number;
  end_beat?: number;
  frame_index?: number;
  duration_seconds?: number;
  chord_quality?: string; // Optional - may not be present in neural output
  label: string;
  confidence: number;
  note_score?: number;
  octave?: number;
  inversion?: string; // Optional - neural defaults to 'root'
  offset_frame?: number;
  offset_seconds?: number;
  midi_notes?: number[];
  note_names?: string[]; // Neural output: array of note names like ['C4', 'E4', 'G4']
  root?: string; // Neural output: root note name like 'C4'
  root_midi?: number;
  method?: string; // Detection method ('neural', 'bic', etc.)
  hand?: "bass" | "treble"; // Neural output: bass/treble hand assignment
  note_value?: "whole" | "half" | "quarter" | "eighth" | "16th" | "32nd";
  note_divisions?: number;
  duration_source?: string;
  timing_authority?: string;
  rest_after_beats?: number;
  local_beat_duration?: number;
  dotted?: boolean;
  // Triplet fields
  triplet?: boolean;
  triplet_position?: "start" | "middle" | "end";
  triplet_type?: "half" | "quarter" | "eighth" | "16th" | "32nd";
  actual_notes?: number;
  normal_notes?: number;
}

interface AnalysisResult {
  onsets: {
    duration_seconds?: number;
    frame_index?: number;
    offset_frame?: number;
    offset_seconds?: number;
    time_seconds: number;
  }[];
  notes: NoteResult[];
  chords: ChordResult[];
  analysis_summary: {
    total_onsets: number;
    total_notes: number;
    total_chords: number;
    duration_seconds: number;
    sample_rate: number;
    detected_bpm?: number;
    tempo_confidence?: number;
    beat_interval?: number;
    bass_notes?: number; // Number of bass notes (left hand)
    treble_notes?: number; // Number of treble notes (right hand)
    bass_chords?: number; // Number of bass chords (left hand)
    treble_chords?: number; // Number of treble chords (right hand)
    method?: string; // Analysis method used ('neural', 'bic', etc.)
    device?: string; // Device used for neural inference ('cuda', 'cpu')
  };
  stream_info?: {
    analysis_type?: string;
    duration_seconds?: number;
    original_sample_rate?: number;
    processed_sample_rate?: number;
    processing_method?: string;
    samples_received?: number;
  };
}

interface PianoSheetMusicProps {
  results?: AnalysisResult;
  timeSignature?: "4/4" | "3/4" | "6/8"; // Time signature for measure grouping
  keySignature?: number; // MusicXML fifths value: -7 (Cb) to +7 (C#), default 0 (C)
  compact?: boolean;
  viewportHeight?: number;
  /**
   * Version number for live refinement updates.
   * When this changes, the component will re-render the score even if
   * the note count hasn't changed (useful for rhythm refinement).
   */
  refinementVersion?: number;
  /**
   * Callback when the WebView has finished rendering the score.
   * Useful for tracking when live updates are visible to the user.
   */
  onScoreRendered?: (measureCount: number) => void;
  onScoreScrollActiveChange?: (active: boolean) => void;
}

interface OsmdDebugSnapshot {
  reason?: string;
  requestId?: number | null;
  initCompleted?: boolean;
  initInFlight?: boolean;
  dependenciesReady?: boolean;
  hasOsmd?: boolean;
  currentXmlLength?: number;
  renderedMeasureCount?: number;
  stageChildElementCount?: number;
  stageInnerHtmlLength?: number;
  stageSvgCount?: number;
  stageClientWidth?: number;
  stageClientHeight?: number;
  containerClientWidth?: number;
  containerClientHeight?: number;
  containerScrollWidth?: number;
  containerScrollHeight?: number;
  containerClassName?: string;
  cameraMode?: string;
  cameraX?: number;
  cameraTargetX?: number;
  cameraMaxX?: number;
  cameraViewportWidth?: number;
  cameraContentWidth?: number;
  cameraMeasureCount?: number;
  cameraAverageMeasureWidth?: number;
  cameraPaddingLeft?: number;
  cameraPaddingRight?: number;
  osmdScriptStatus?: string | null;
  marker?: string | null;
  timestamp?: number;
}

type CameraMotionMode = "smooth" | "snap";

// Key-aware MIDI to pitch spelling.
// fifths: MusicXML fifths value (-7 Cb … 0 C … +7 C#).
// Returns the correctly-spelled step letter, alter (-1/0/+1), and octave.
function midiToStepOctaveForKey(
  midi: number,
  fifths: number = 0,
): { step: string; alter: number; octave: number } {
  // Default sharp-only map: pitch-class → [step, alter]
  const map: [string, number][] = [
    ["C", 0],
    ["C", 1],
    ["D", 0],
    ["D", 1],
    ["E", 0],
    ["F", 0],
    ["F", 1],
    ["G", 0],
    ["G", 1],
    ["A", 0],
    ["A", 1],
    ["B", 0],
  ];

  // Flat respellings in order-of-flats (Bb Eb Ab Db Gb Cb Fb)
  const flatOrder: [number, string, number][] = [
    [10, "B", -1], // A# → Bb
    [3, "E", -1], // D# → Eb
    [8, "A", -1], // G# → Ab
    [1, "D", -1], // C# → Db
    [6, "G", -1], // F# → Gb
    [11, "C", -1], // B  → Cb  (octave +1)
    [4, "F", -1], // E  → Fb
  ];

  // Extra sharp respellings for 6-7 sharps (E# and B#)
  const sharpExtra: [number, string, number][] = [
    [5, "E", 1], // F → E#
    [0, "B", 1], // C → B#  (octave -1)
  ];

  if (fifths < 0) {
    const n = Math.min(Math.abs(fifths), 7);
    for (let i = 0; i < n; i++) {
      const [pc, step, alter] = flatOrder[i];
      map[pc] = [step, alter];
    }
  } else if (fifths > 5) {
    const extra = Math.min(fifths - 5, 2);
    for (let i = 0; i < extra; i++) {
      const [pc, step, alter] = sharpExtra[i];
      map[pc] = [step, alter];
    }
  }

  const pc = midi % 12;
  const [step, alter] = map[pc];
  let octave = Math.floor(midi / 12) - 1;

  // Octave corrections for cross-letter respellings
  // Cb is one letter above B, so it belongs to the next octave
  if (step === "C" && alter === -1) octave += 1; // Cb
  // B# is one letter below C, so it belongs to the previous octave
  if (step === "B" && alter === 1) octave -= 1; // B#

  return { step, alter, octave };
}

// Function to generate MusicXML from notes
// timeSignature: "4/4" | "3/4" | "6/8" - controls beats per measure
// bpm: Beats per minute for playback tempo marking
function generateMeasureXmls(
  notes: NoteResult[],
  chords: ChordResult[],
  timeSignature: "4/4" | "3/4" | "6/8" = "4/4",
  bpm: number = 120,
  fifths: number = 0,
): string[] {
  const measures: string[] = [];

  // Parse time signature to get beats per measure
  const getBeatsPerMeasure = (ts: "4/4" | "3/4" | "6/8"): number => {
    switch (ts) {
      case "4/4":
        return 4;
      case "3/4":
        return 3;
      case "6/8":
        return 6 * 0.5; // 6 eighth notes = 3 quarter note beats
      default:
        return 4;
    }
  };
  const BEATS_PER_MEASURE = getBeatsPerMeasure(timeSignature);
  const getAuthoritativeStartBeat = (event: {
    time_seconds?: number;
    start_beat?: number;
  }): number => {
    if (
      typeof event.start_beat === "number" &&
      Number.isFinite(event.start_beat)
    ) {
      return Math.round(event.start_beat * 24) / 24;
    }
    return Math.round(((event.time_seconds ?? 0) / 60) * bpm * 24) / 24;
  };

  // Helper to get beat value for a note type (considers triplets)
  const getNoteBeats = (
    noteType?: string,
    dotted?: boolean,
    triplet?: boolean,
  ): number => {
    let beats = 1;
    switch (noteType) {
      case "whole":
        beats = 4;
        break;
      case "half":
        beats = 2;
        break;
      case "quarter":
        beats = 1;
        break;
      case "eighth":
        beats = 0.5;
        break;
      case "16th":
        beats = 0.25;
        break;
      case "32nd":
        beats = 0.125;
        break;
      default:
        beats = 1;
        break;
    }
    if (dotted) beats *= 1.5;
    // Triplet: 3 notes in time of 2, so each note is 2/3 of normal
    // Round to nearest 1/24 of a beat for triplet-exact precision
    // E.g., 3 eighth note triplets: round(0.5 * 2/3 * 24) / 24 = round(8) / 24 = 1/3 per note
    // Total: 1/3 * 3 = 1.0 beats (exact)
    if (triplet) beats = Math.round(beats * (2 / 3) * 24) / 24;
    return beats;
  };

  // Helper to get MusicXML duration (divisions=24 for triplet-exact arithmetic)
  // With divisions=24, all note values AND triplet values are exact integers:
  //   quarter=24, triplet quarter=16, eighth=12, triplet eighth=8, etc.
  const getNoteDuration = (
    noteType?: string,
    dotted?: boolean,
    triplet?: boolean,
  ): number => {
    let duration = 24;
    switch (noteType) {
      case "whole":
        duration = 96;
        break;
      case "half":
        duration = 48;
        break;
      case "quarter":
        duration = 24;
        break;
      case "eighth":
        duration = 12;
        break;
      case "16th":
        duration = 6;
        break;
      case "32nd":
        duration = 3;
        break;
      default:
        duration = 24;
        break;
    }
    if (dotted) duration = duration * 1.5;
    // Guard: dotted 32nd = 4.5 which is non-integer (invalid MusicXML duration)
    // divisions=24 can't represent it — fall back to plain 32nd
    if (duration !== Math.floor(duration)) duration = Math.floor(duration);
    // Triplet: 3 notes in time of 2 — exact with divisions=24
    // E.g., triplet quarter: 24 * 2/3 = 16, and 16*3 = 48 = 2*24 (exact)
    if (triplet) duration = (duration * 2) / 3;
    return duration;
  };

  type DurationSpec = {
    beats: number;
    duration: number;
    noteType: "whole" | "half" | "quarter" | "eighth" | "16th" | "32nd";
    dotted: boolean;
    triplet: boolean;
  };

  const getFallbackDurationSpec = (
    noteType?: string,
    dotted?: boolean,
    triplet?: boolean,
  ): DurationSpec => {
    const resolvedNoteType =
      (noteType as DurationSpec["noteType"] | undefined) || "quarter";
    const resolvedDotted = dotted || false;
    const resolvedTriplet = triplet || false;
    return {
      beats: getNoteBeats(resolvedNoteType, resolvedDotted, resolvedTriplet),
      duration: getNoteDuration(
        resolvedNoteType,
        resolvedDotted,
        resolvedTriplet,
      ),
      noteType: resolvedNoteType,
      dotted: resolvedDotted,
      triplet: resolvedTriplet,
    };
  };

  const getDurationSpec = (
    preferredBeats?: number,
    noteType?: string,
    dotted?: boolean,
    triplet?: boolean,
  ): DurationSpec => {
    const fallback = getFallbackDurationSpec(noteType, dotted, triplet);
    if (
      preferredBeats === undefined ||
      !Number.isFinite(preferredBeats) ||
      preferredBeats <= 0
    ) {
      return fallback;
    }

    const canonicalSpecs: DurationSpec[] = [
      {
        beats: 4,
        duration: 96,
        noteType: "whole",
        dotted: false,
        triplet: false,
      },
      {
        beats: 3,
        duration: 72,
        noteType: "half",
        dotted: true,
        triplet: false,
      },
      {
        beats: 8 / 3,
        duration: 64,
        noteType: "whole",
        dotted: false,
        triplet: true,
      },
      {
        beats: 2,
        duration: 48,
        noteType: "half",
        dotted: false,
        triplet: false,
      },
      {
        beats: 1.5,
        duration: 36,
        noteType: "quarter",
        dotted: true,
        triplet: false,
      },
      {
        beats: 4 / 3,
        duration: 32,
        noteType: "half",
        dotted: false,
        triplet: true,
      },
      {
        beats: 1,
        duration: 24,
        noteType: "quarter",
        dotted: false,
        triplet: false,
      },
      {
        beats: 0.75,
        duration: 18,
        noteType: "eighth",
        dotted: true,
        triplet: false,
      },
      {
        beats: 2 / 3,
        duration: 16,
        noteType: "quarter",
        dotted: false,
        triplet: true,
      },
      {
        beats: 0.5,
        duration: 12,
        noteType: "eighth",
        dotted: false,
        triplet: false,
      },
      {
        beats: 0.375,
        duration: 9,
        noteType: "16th",
        dotted: true,
        triplet: false,
      },
      {
        beats: 1 / 3,
        duration: 8,
        noteType: "eighth",
        dotted: false,
        triplet: true,
      },
      {
        beats: 0.25,
        duration: 6,
        noteType: "16th",
        dotted: false,
        triplet: false,
      },
      {
        beats: 1 / 6,
        duration: 4,
        noteType: "16th",
        dotted: false,
        triplet: true,
      },
      {
        beats: 0.125,
        duration: 3,
        noteType: "32nd",
        dotted: false,
        triplet: false,
      },
      {
        beats: 1 / 12,
        duration: 2,
        noteType: "32nd",
        dotted: false,
        triplet: true,
      },
    ];

    const matched = canonicalSpecs.find(
      (spec) => Math.abs(spec.beats - preferredBeats) <= 1 / 48,
    );

    return matched || fallback;
  };

  // Helper to split beats into a list of (noteType, duration, dotted, beats) tuples
  // Used for ties that span measure boundaries
  const splitBeatsIntoNoteTypes = (
    beats: number,
  ): {
    noteType: string;
    duration: number;
    beats: number;
    dotted: boolean;
  }[] => {
    const result: {
      noteType: string;
      duration: number;
      beats: number;
      dotted: boolean;
    }[] = [];
    // Note values from largest to smallest
    const noteValues = [
      { beats: 4, noteType: "whole", duration: 96, dotted: false },
      { beats: 3, noteType: "half", duration: 72, dotted: true }, // dotted half
      { beats: 2, noteType: "half", duration: 48, dotted: false },
      { beats: 1.5, noteType: "quarter", duration: 36, dotted: true }, // dotted quarter
      { beats: 1, noteType: "quarter", duration: 24, dotted: false },
      { beats: 0.75, noteType: "eighth", duration: 18, dotted: true }, // dotted eighth
      { beats: 0.5, noteType: "eighth", duration: 12, dotted: false },
      { beats: 0.25, noteType: "16th", duration: 6, dotted: false },
      { beats: 0.125, noteType: "32nd", duration: 3, dotted: false },
    ];

    let remaining = Math.round(beats * 24) / 24; // Round to 1/24 beat (divisions-exact)

    while (remaining >= 0.125 - 0.001) {
      let found = false;
      for (const nv of noteValues) {
        if (remaining >= nv.beats - 0.001) {
          result.push({
            noteType: nv.noteType,
            duration: nv.duration,
            beats: nv.beats,
            dotted: nv.dotted,
          });
          remaining = Math.round((remaining - nv.beats) * 24) / 24;
          found = true;
          break;
        }
      }
      if (!found) break;
    }

    return result;
  };

  // Generate a note XML with optional tie start/stop
  const generateNoteXmlWithTie = (
    pitchXml: string,
    duration: number,
    noteType: string,
    staff: number,
    tieType?: "start" | "stop" | "continue",
    isChord?: boolean,
    dotted?: boolean,
  ): string => {
    const chordTag = isChord ? "<chord/>" : "";
    const dotXml = dotted ? "<dot/>" : "";
    let tieXml = "";
    let notationsXml = "";

    if (tieType === "start") {
      tieXml = '<tie type="start"/>';
      notationsXml = '<notations><tied type="start"/></notations>';
    } else if (tieType === "stop") {
      tieXml = '<tie type="stop"/>';
      notationsXml = '<notations><tied type="stop"/></notations>';
    } else if (tieType === "continue") {
      tieXml = '<tie type="stop"/><tie type="start"/>';
      notationsXml =
        '<notations><tied type="stop"/><tied type="start"/></notations>';
    }

    return `<note>${chordTag}${pitchXml}<duration>${duration}</duration>${tieXml}<voice>${staff}</voice><type>${noteType}</type>${dotXml}<staff>${staff}</staff>${notationsXml}</note>`;
  };

  // Compute per-segment tie type for multi-segment splits.
  // When a tied duration is split into multiple note types (e.g., dotted-half + eighth),
  // each segment needs its own tie type. The outer tieType describes the relationship
  // to notes OUTSIDE this split (previous/next measure). Internal segments need additional ties.
  const getSegmentTieType = (
    outerTieType: "start" | "stop" | "continue",
    segIndex: number,
    totalSegments: number,
  ): "start" | "stop" | "continue" => {
    if (totalSegments <= 1) return outerTieType;
    const isFirst = segIndex === 0;
    const isLast = segIndex === totalSegments - 1;
    if (isFirst) {
      // First segment: keep outer start side, add start to next segment
      return outerTieType === "stop" ? "continue" : outerTieType; // stop→continue, start→start, continue→continue
    } else if (isLast) {
      // Last segment: stop from previous segment, keep outer stop side
      return outerTieType === "start" ? "continue" : outerTieType; // start→continue, stop→stop, continue→continue
    } else {
      // Middle segment: always continue (stop from prev + start to next)
      return "continue";
    }
  };

  // Return the note type (no tempo-based adjustments anymore)
  const getAdjustedNoteType = (noteType?: string): string =>
    noteType || "quarter";

  // Generate triplet notation XML elements
  const getTripletNotations = (
    tripletPosition?: "start" | "middle" | "end",
    actualNotes: number = 3,
    normalNotes: number = 2,
  ): string => {
    if (!tripletPosition) return "";

    if (tripletPosition === "start") {
      return `<notations><tuplet type="start" bracket="yes" number="1"/></notations>`;
    } else if (tripletPosition === "end") {
      return `<notations><tuplet type="stop" number="1"/></notations>`;
    }
    // Middle notes don't need tuplet notation
    return "";
  };

  // Generate ornament notation XML elements
  const getOrnamentNotations = (
    ornament?: string,
    trillTo?: number,
  ): string => {
    if (!ornament) return "";

    switch (ornament) {
      case "trill":
        // MusicXML trill-mark with optional accidental-mark for the auxiliary note
        let trillXml = '<ornaments><trill-mark placement="above"/>';
        if (trillTo !== undefined) {
          // Calculate if trill note needs an accidental
          const trillPitch = trillTo % 12;
          const needsAccidental = [1, 3, 6, 8, 10].includes(trillPitch); // Black keys
          if (needsAccidental) {
            trillXml += "<accidental-mark>sharp</accidental-mark>";
          }
        }
        trillXml += "</ornaments>";
        return trillXml;
      case "mordent_upper":
        return "<ornaments><inverted-mordent/></ornaments>";
      case "mordent_lower":
        return "<ornaments><mordent/></ornaments>";
      case "turn_upper":
        return '<ornaments><turn placement="above"/></ornaments>';
      case "turn_inverted":
        return '<ornaments><inverted-turn placement="above"/></ornaments>';
      case "grace":
        // Grace notes handled separately via note type
        return "";
      default:
        return "";
    }
  };

  // Generate time-modification XML for triplets
  const getTimeModification = (
    triplet?: boolean,
    actualNotes: number = 3,
    normalNotes: number = 2,
  ): string => {
    if (!triplet) return "";
    return `<time-modification><actual-notes>${actualNotes}</actual-notes><normal-notes>${normalNotes}</normal-notes></time-modification>`;
  };

  // Generate rest XML for a given number of beats
  // Decomposes into standard rest values (whole → 32nd), then uses <forward>
  // for any sub-32nd remainder to keep MusicXML duration accounting exact.
  // Returns { xml, beatsEmitted } so callers can track actual emitted duration.
  const generateRestXml = (
    beats: number,
    staff: number,
  ): { xml: string[]; beatsEmitted: number } => {
    const rests: string[] = [];
    // Round to nearest 1/24 beat (divisions-exact) to avoid float drift
    let remaining = Math.round(beats * 24) / 24;
    const totalRounded = remaining;
    const restValues = [
      { beats: 4, type: "whole", duration: 96 },
      { beats: 2, type: "half", duration: 48 },
      { beats: 1, type: "quarter", duration: 24 },
      { beats: 0.5, type: "eighth", duration: 12 },
      { beats: 0.25, type: "16th", duration: 6 },
      { beats: 0.125, type: "32nd", duration: 3 },
    ];
    while (remaining >= 0.125 - 0.001) {
      let found = false;
      for (const rv of restValues) {
        if (remaining >= rv.beats - 0.001) {
          rests.push(
            `<note><rest/><duration>${rv.duration}</duration><voice>${staff}</voice><type>${rv.type}</type><staff>${staff}</staff></note>`,
          );
          remaining = Math.round((remaining - rv.beats) * 24) / 24;
          found = true;
          break;
        }
      }
      if (!found) break;
    }
    // Handle sub-32nd remainder (e.g., after triplet notes: 1/24 or 2/24 beats)
    // Use <forward> to advance time without a visible rest
    if (remaining > 0.001) {
      const fwdDiv = Math.round(remaining * 24);
      if (fwdDiv > 0) {
        rests.push(`<forward><duration>${fwdDiv}</duration></forward>`);
        remaining = Math.round((remaining - fwdDiv / 24) * 24) / 24;
      }
    }
    return { xml: rests, beatsEmitted: totalRounded - remaining };
  };

  // Determine staff for a MIDI note
  const getStaff = (midi: number): number => {
    const octave = Math.floor(midi / 12) - 1;
    return octave < 4 ? 2 : 1;
  };

  // Convert note to XML with voice (supports triplets)
  const noteToXmlWithVoice = (
    n: NoteResult,
    isChordNote: boolean = false,
  ): string => {
    const {
      step: baseStep,
      alter,
      octave,
    } = midiToStepOctaveForKey(n.midi_note, fifths);
    const staff = getStaff(n.midi_note);
    const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
    const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
    const durationSpec = getDurationSpec(
      n.note_divisions,
      n.note_value,
      n.dotted,
      n.triplet,
    );
    const adjustedNoteType = getAdjustedNoteType(durationSpec.noteType);
    const dotXml = durationSpec.dotted ? "<dot/>" : "";
    const chordTag = isChordNote ? "<chord/>" : "";

    // Triplet-specific XML
    const timeModXml = getTimeModification(
      durationSpec.triplet,
      n.actual_notes || 3,
      n.normal_notes || 2,
    );

    // Build notations XML (can contain both tuplet and ornament)
    const tupletXml = getTripletNotations(
      n.triplet_position,
      n.actual_notes || 3,
      n.normal_notes || 2,
    );
    const ornamentXml = getOrnamentNotations(n.ornament, n.trill_to);

    // Combine notations - need to merge into single <notations> tag
    let notationsXml = "";
    if (tupletXml || ornamentXml) {
      let notationsContent = "";
      // Extract content from tuplet notations if present
      if (tupletXml) {
        const tupletMatch = tupletXml.match(/<notations>(.*)<\/notations>/);
        if (tupletMatch) notationsContent += tupletMatch[1];
      }
      // Ornament XML doesn't have <notations> wrapper
      if (ornamentXml) notationsContent += ornamentXml;
      if (notationsContent) {
        notationsXml = `<notations>${notationsContent}</notations>`;
      }
    }

    // Handle grace notes - they have special structure
    if (n.ornament === "grace") {
      const graceType =
        n.grace_type === "appoggiatura" ? "<grace/>" : '<grace slash="yes"/>';
      return `<note>${graceType}${chordTag}${pitchXml}<voice>${staff}</voice><type>eighth</type><staff>${staff}</staff></note>`;
    }

    return `<note>${chordTag}${pitchXml}<duration>${durationSpec.duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${dotXml}${timeModXml}<staff>${staff}</staff>${notationsXml}</note>`;
  };

  // Helper to construct chord note MIDI list from ChordResult
  const chordToMidiList = (c: ChordResult): number[] => {
    if (c.midi_notes && c.midi_notes.length) return c.midi_notes.slice();
    const nameToSemitone: Record<string, number> = {
      C: 0,
      "C#": 1,
      DB: 1,
      D: 2,
      "D#": 3,
      EB: 3,
      E: 4,
      FB: 4,
      F: 5,
      "F#": 6,
      GB: 6,
      G: 7,
      "G#": 8,
      AB: 8,
      A: 9,
      "A#": 10,
      BB: 10,
      B: 11,
      CB: 11,
    };
    const buildFromRoot = (root: number, quality?: string): number[] => {
      const q = (quality || "").toLowerCase();
      if (q.includes("maj7") || q === "maj7")
        return [root, root + 4, root + 7, root + 11];
      if (q === "7" || q.includes("dom7") || q === "dom")
        return [root, root + 4, root + 7, root + 10];
      if (q === "m7" || q === "min7")
        return [root, root + 3, root + 7, root + 10];
      if (q === "m" || q === "min") return [root, root + 3, root + 7];
      if (q === "dim") return [root, root + 3, root + 6];
      if (q === "aug") return [root, root + 4, root + 8];
      if (q === "sus2") return [root, root + 2, root + 7];
      if (q === "sus4") return [root, root + 5, root + 7];
      return [root, root + 4, root + 7];
    };
    if (typeof c.root_midi === "number")
      return buildFromRoot(c.root_midi, c.chord_quality);
    if (c.label) {
      const m = String(c.label)
        .toUpperCase()
        .match(/^([A-G][#B]?)/);
      if (m) {
        const rootName = m[1].replace("B", "B");
        const semitone = nameToSemitone[rootName] ?? 0;
        const octave = typeof c.octave === "number" ? c.octave : 4;
        const rootMidi = (octave + 1) * 12 + semitone;
        return buildFromRoot(rootMidi, c.chord_quality);
      }
    }
    return [];
  };

  // Convert chord MIDI notes to XML, grouped by staff (supports triplets)
  const chordMidiToXml = (
    midiList: number[],
    noteType: string,
    dotted: boolean,
    staff: number,
    preferredBeats?: number,
    triplet?: boolean,
    tripletPosition?: "start" | "middle" | "end",
    actualNotes: number = 3,
    normalNotes: number = 2,
  ): string[] => {
    const durationSpec = getDurationSpec(
      preferredBeats,
      noteType,
      dotted,
      triplet,
    );
    const adjustedNoteType = getAdjustedNoteType(durationSpec.noteType);
    const dotXml = durationSpec.dotted ? "<dot/>" : "";
    const timeModXml = getTimeModification(
      durationSpec.triplet,
      actualNotes,
      normalNotes,
    );

    // Filter to only notes on this staff
    const staffNotes = midiList.filter((m) => getStaff(m) === staff);
    if (staffNotes.length === 0) return [];

    const xmlParts: string[] = [];
    let first = true;
    for (const midi of staffNotes) {
      const {
        step: baseStep,
        alter,
        octave,
      } = midiToStepOctaveForKey(midi, fifths);
      const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
      const pitchInner = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
      const chordTag = first ? "" : "<chord/>";
      // Only first note of chord gets triplet notations
      const tripletNotationsXml = first
        ? getTripletNotations(tripletPosition, actualNotes, normalNotes)
        : "";
      const noteXml = `<note>${chordTag}${pitchInner}<duration>${durationSpec.duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${dotXml}${timeModXml}<staff>${staff}</staff>${tripletNotationsXml}</note>`;
      xmlParts.push(noteXml);
      first = false;
    }
    return xmlParts;
  };

  // Build a timeline of events per staff
  type TimelineEvent = {
    time: number;
    beatStart: number;
    staff: number;
    beats: number;
    xml: string[];
    midiNotes: number[]; // Store MIDI notes for tie generation
    // Triplet metadata for cross-measure handling
    triplet?: boolean;
    tripletPosition?: "start" | "middle" | "end";
    tripletType?: string;
    actualNotes?: number;
    normalNotes?: number;
  };

  const timeline: TimelineEvent[] = [];

  // Build a set of notes that are already covered by chords
  // (to avoid duplicate notes appearing alongside their chords)
  const BEAT_KEY_TOLERANCE = 1 / 48;
  const notesInChords = new Set<string>();
  for (const c of chords) {
    const beatStart = getAuthoritativeStartBeat(c);
    const midiList = chordToMidiList(c);
    for (const midi of midiList) {
      notesInChords.add(
        `${Math.round(beatStart / BEAT_KEY_TOLERANCE)}:${midi}`,
      );
    }
  }

  // Process all notes (filter out notes that are already in chords at same time)
  for (const n of notes) {
    const time = n.time_seconds ?? 0;
    const beatStart = getAuthoritativeStartBeat(n);
    const noteKey = `${Math.round(beatStart / BEAT_KEY_TOLERANCE)}:${n.midi_note}`;

    // Skip notes that are part of a chord at the same time
    if (notesInChords.has(noteKey)) {
      continue;
    }

    const staff = getStaff(n.midi_note);
    // Grace notes have 0 beats - they don't take up time in the measure
    // Grace notes also cannot have triplet markings
    const isGrace = n.ornament === "grace";
    const durationSpec = getDurationSpec(
      n.note_divisions,
      n.note_value,
      n.dotted,
      n.triplet,
    );
    const beats = isGrace ? 0 : durationSpec.beats;
    const xml = [noteToXmlWithVoice(n, false)];
    timeline.push({
      time,
      beatStart,
      staff,
      beats,
      xml,
      midiNotes: [n.midi_note],
      triplet: isGrace ? false : n.triplet,
      tripletPosition: isGrace ? undefined : n.triplet_position,
      tripletType: isGrace ? undefined : n.triplet_type || n.note_value,
      actualNotes: isGrace ? undefined : n.actual_notes,
      normalNotes: isGrace ? undefined : n.normal_notes,
    });
  }

  // Process all chords - split across staves if needed
  for (const c of chords) {
    const time = c.time_seconds ?? 0;
    const beatStart = getAuthoritativeStartBeat(c);
    let midiList = chordToMidiList(c);
    if (midiList.length === 0) continue;

    // Apply inversion
    const inversionToIndex = (inv: any, chordLen: number) => {
      if (typeof inv === "number") return Math.max(0, Math.floor(inv));
      if (!inv) return 0;
      const s = String(inv).toLowerCase();
      if (s === "root") return 0;
      if (s === "first") return 1;
      if (s === "second") return 2;
      if (s === "third") return 3;
      if (s === "slash") return chordLen >= 4 ? 3 : 1;
      return 0;
    };
    const inversion = inversionToIndex(c.inversion, midiList.length);
    for (let i = 0; i < inversion; i++) {
      const n = midiList.shift();
      if (typeof n === "number") midiList.push(n + 12);
    }

    const noteType = c.note_value || "quarter";
    const dotted = c.dotted || false;
    const triplet = c.triplet || false;
    const durationSpec = getDurationSpec(
      c.note_divisions,
      noteType,
      dotted,
      triplet,
    );
    const beats = durationSpec.beats;

    // Split chord by staff (with triplet info)
    const trebleXml = chordMidiToXml(
      midiList,
      noteType,
      dotted,
      1,
      c.note_divisions,
      triplet,
      c.triplet_position,
      c.actual_notes,
      c.normal_notes,
    );
    const bassXml = chordMidiToXml(
      midiList,
      noteType,
      dotted,
      2,
      c.note_divisions,
      triplet,
      c.triplet_position,
      c.actual_notes,
      c.normal_notes,
    );

    if (trebleXml.length > 0) {
      const trebleMidiNotes = midiList.filter((m) => getStaff(m) === 1);
      timeline.push({
        time,
        beatStart,
        staff: 1,
        beats,
        xml: trebleXml,
        midiNotes: trebleMidiNotes,
        triplet,
        tripletPosition: c.triplet_position,
        tripletType: c.triplet_type || noteType,
        actualNotes: c.actual_notes,
        normalNotes: c.normal_notes,
      });
    }
    if (bassXml.length > 0) {
      const bassMidiNotes = midiList.filter((m) => getStaff(m) === 2);
      timeline.push({
        time,
        beatStart,
        staff: 2,
        beats,
        xml: bassXml,
        midiNotes: bassMidiNotes,
        triplet,
        tripletPosition: c.triplet_position,
        tripletType: c.triplet_type || noteType,
        actualNotes: c.actual_notes,
        normalNotes: c.normal_notes,
      });
    }
  }

  // Sort timeline by authoritative beat position first, then by raw time.
  timeline.sort((a, b) => a.beatStart - b.beatStart || a.time - b.time);

  // Group events by backend beat position for rendering. Raw time still protects
  // against merging same-staff sequential notes into chords.
  const CROSS_STAFF_TIME_TOLERANCE = 0.025;
  const SAME_STAFF_TOLERANCE = 0.005; // 5ms — perceptual simultaneity threshold
  const BEAT_GROUP_TOLERANCE = 1 / 48;
  const SAME_STAFF_BEAT_TOLERANCE = 1 / 96;
  type TimeGroup = {
    time: number;
    beatStart: number;
    treble: TimelineEvent[];
    bass: TimelineEvent[];
  };
  const timeGroups: TimeGroup[] = [];

  for (const ev of timeline) {
    let group = timeGroups.find((g) => {
      const beatDelta = Math.abs(g.beatStart - ev.beatStart);
      const timeDelta = Math.abs(g.time - ev.time);
      if (
        beatDelta < BEAT_GROUP_TOLERANCE &&
        timeDelta < CROSS_STAFF_TIME_TOLERANCE
      ) {
        const sameStaffEvents = ev.staff === 1 ? g.treble : g.bass;
        if (sameStaffEvents.length > 0) {
          return (
            beatDelta < SAME_STAFF_BEAT_TOLERANCE &&
            timeDelta < SAME_STAFF_TOLERANCE
          );
        }
        return true;
      }
      return false;
    });
    if (!group) {
      group = { time: ev.time, beatStart: ev.beatStart, treble: [], bass: [] };
      timeGroups.push(group);
    } else {
      group.time = Math.min(group.time, ev.time);
      group.beatStart = Math.min(group.beatStart, ev.beatStart);
    }
    if (ev.staff === 1) {
      group.treble.push(ev);
    } else {
      group.bass.push(ev);
    }
  }

  // Sort groups by authoritative beat position.
  timeGroups.sort((a, b) => a.beatStart - b.beatStart || a.time - b.time);

  // Deduplicate MIDI notes within each time group per staff
  // This prevents the same note from appearing twice at the same position
  for (const group of timeGroups) {
    for (const staffKey of ["treble", "bass"] as const) {
      const events = group[staffKey];
      if (events.length <= 1) continue;

      // Collect all unique MIDI notes across events in this staff/time
      const seenMidi = new Set<number>();
      const deduped: TimelineEvent[] = [];

      for (const ev of events) {
        const uniqueMidi = (ev.midiNotes || []).filter((m) => {
          if (seenMidi.has(m)) return false;
          seenMidi.add(m);
          return true;
        });

        if (uniqueMidi.length === 0) continue;

        if (uniqueMidi.length === ev.midiNotes.length) {
          // No duplicates found in this event, keep as-is
          deduped.push(ev);
        } else {
          // Rebuild XML for only the unique notes
          const filteredXml = ev.xml.filter((xml) => {
            // Check if this XML element's pitch is in the unique set
            const pitchMatch = xml.match(
              /<pitch><step>(\w+)<\/step>(?:<alter>(-?\d)<\/alter>)?<octave>(\d+)<\/octave><\/pitch>/,
            );
            if (!pitchMatch) return true; // Keep non-pitch elements
            const step = pitchMatch[1];
            const alter = pitchMatch[2] ? parseInt(pitchMatch[2]) : 0;
            const octave = parseInt(pitchMatch[3]);
            const noteNames = [
              "C",
              "C#",
              "D",
              "D#",
              "E",
              "F",
              "F#",
              "G",
              "G#",
              "A",
              "A#",
              "B",
            ];
            const stepIdx = noteNames.indexOf(
              step + (alter === 1 ? "#" : alter === -1 ? "b" : ""),
            );
            const midi = stepIdx >= 0 ? (octave + 1) * 12 + stepIdx : -1;
            return midi < 0 || uniqueMidi.includes(midi);
          });

          if (filteredXml.length > 0) {
            deduped.push({
              ...ev,
              midiNotes: uniqueMidi,
              xml: filteredXml,
            });
          }
        }
      }

      group[staffKey] = deduped;
    }
  }

  // Merge simultaneous events on the same staff into chords
  // Events within a time group are considered simultaneous
  const CHORD_MERGE_TOLERANCE = 0;

  for (const group of timeGroups) {
    // For treble: sub-group by actual time, only merge within each sub-group
    if (group.treble.length > 1) {
      // Group treble events by their actual time (stricter tolerance)
      const subGroups: TimelineEvent[][] = [];
      for (const ev of group.treble) {
        let found = false;
        for (const sg of subGroups) {
          if (Math.abs(sg[0].time - ev.time) < CHORD_MERGE_TOLERANCE) {
            sg.push(ev);
            found = true;
            break;
          }
        }
        if (!found) {
          subGroups.push([ev]);
        }
      }

      // Merge each sub-group into a chord (if > 1 event)
      const newTreble: TimelineEvent[] = [];
      for (const sg of subGroups) {
        if (sg.length === 1) {
          newTreble.push(sg[0]);
        } else {
          // Merge into chord
          const mergedXml: string[] = [];
          const mergedMidiNotes: number[] = [];
          let maxBeats = 0;
          let first = true;
          for (const ev of sg) {
            mergedMidiNotes.push(...(ev.midiNotes || []));
            for (const xml of ev.xml) {
              if (first) {
                mergedXml.push(xml);
                first = false;
              } else {
                const chordXml = xml.replace("<note>", "<note><chord/>");
                mergedXml.push(chordXml);
              }
            }
            maxBeats = Math.max(maxBeats, ev.beats);
          }
          newTreble.push({
            time: sg[0].time,
            beatStart: sg[0].beatStart,
            staff: 1,
            beats: maxBeats,
            xml: mergedXml,
            midiNotes: mergedMidiNotes,
            triplet: sg[0].triplet,
            tripletPosition: sg[0].tripletPosition,
            tripletType: sg[0].tripletType,
            actualNotes: sg[0].actualNotes,
            normalNotes: sg[0].normalNotes,
          });
        }
      }
      group.treble = newTreble;
    }

    // For bass: same sub-grouping approach
    if (group.bass.length > 1) {
      const subGroups: TimelineEvent[][] = [];
      for (const ev of group.bass) {
        let found = false;
        for (const sg of subGroups) {
          if (Math.abs(sg[0].time - ev.time) < CHORD_MERGE_TOLERANCE) {
            sg.push(ev);
            found = true;
            break;
          }
        }
        if (!found) {
          subGroups.push([ev]);
        }
      }

      const newBass: TimelineEvent[] = [];
      for (const sg of subGroups) {
        if (sg.length === 1) {
          newBass.push(sg[0]);
        } else {
          const mergedXml: string[] = [];
          const mergedMidiNotes: number[] = [];
          let maxBeats = 0;
          let first = true;
          for (const ev of sg) {
            mergedMidiNotes.push(...(ev.midiNotes || []));
            for (const xml of ev.xml) {
              if (first) {
                mergedXml.push(xml);
                first = false;
              } else {
                const chordXml = xml.replace("<note>", "<note><chord/>");
                mergedXml.push(chordXml);
              }
            }
            maxBeats = Math.max(maxBeats, ev.beats);
          }
          newBass.push({
            time: sg[0].time,
            beatStart: sg[0].beatStart,
            staff: 2,
            beats: maxBeats,
            xml: mergedXml,
            midiNotes: mergedMidiNotes,
            triplet: sg[0].triplet,
            tripletPosition: sg[0].tripletPosition,
            tripletType: sg[0].tripletType,
            actualNotes: sg[0].actualNotes,
            normalNotes: sg[0].normalNotes,
          });
        }
      }
      group.bass = newBass;
    }
  }

  // ============================================================================
  // TRIPLET VALIDATION: Ensure triplets are valid before building measures
  // Rules:
  // 1. Grace notes can never be triplets
  // 2. Triplets must be complete (start -> middle -> end = exactly 3 notes)
  // 3. Triplets must NOT cross measure boundaries
  // 4. All 3 notes in a triplet must have the same note type
  // ============================================================================

  // Helper to strip triplet markers from XML strings
  const stripTripletFromXml = (xmlArr: string[]): string[] => {
    return xmlArr.map((xml) => {
      // Remove time-modification
      let result = xml.replace(
        /<time-modification>.*?<\/time-modification>/g,
        "",
      );
      // Remove tuplet elements (may be inside notations with other content)
      result = result.replace(/<tuplet[^>]*\/>/g, "");
      // Clean up empty notations tags
      result = result.replace(/<notations>\s*<\/notations>/g, "");
      return result;
    });
  };

  // Helper to strip triplet from a single event
  const stripTripletFromEvent = (ev: TimelineEvent) => {
    ev.xml = stripTripletFromXml(ev.xml);
    ev.triplet = false;
    ev.tripletPosition = undefined;
  };

  // Track triplet groups per staff to validate completeness and measure alignment
  for (const staff of [1, 2] as const) {
    const events =
      staff === 1
        ? timeGroups.flatMap((g) => g.treble)
        : timeGroups.flatMap((g) => g.bass);

    let tripletStart: TimelineEvent | null = null;
    let tripletMiddle: TimelineEvent | null = null;

    for (const ev of events) {
      if (ev.tripletPosition === "start") {
        // If there was a previous incomplete triplet, strip it
        if (tripletStart) stripTripletFromEvent(tripletStart);
        if (tripletMiddle) stripTripletFromEvent(tripletMiddle);
        tripletStart = ev;
        tripletMiddle = null;
      } else if (ev.tripletPosition === "middle" && tripletStart) {
        tripletMiddle = ev;
      } else if (ev.tripletPosition === "end") {
        if (tripletStart && tripletMiddle) {
          // All 3 notes present - now validate measure alignment and same type

          // Check all notes have the same triplet type
          const sameType =
            tripletStart.tripletType === tripletMiddle.tripletType &&
            tripletMiddle.tripletType === ev.tripletType;

          // Check all notes fall in the same measure
          const startBeat = tripletStart.beatStart;
          const middleBeat = tripletMiddle.beatStart;
          const endBeat = ev.beatStart;

          const startMeasure = Math.floor(startBeat / BEATS_PER_MEASURE);
          const middleMeasure = Math.floor(middleBeat / BEATS_PER_MEASURE);
          const endMeasure = Math.floor(endBeat / BEATS_PER_MEASURE);

          const sameMeasure =
            startMeasure === middleMeasure && middleMeasure === endMeasure;

          if (!sameType || !sameMeasure) {
            // Invalid - strip all three
            stripTripletFromEvent(tripletStart);
            stripTripletFromEvent(tripletMiddle);
            stripTripletFromEvent(ev);
          }
          // else: valid triplet - keep markers
        } else {
          // Incomplete triplet - strip all present
          if (tripletStart) stripTripletFromEvent(tripletStart);
          if (tripletMiddle) stripTripletFromEvent(tripletMiddle);
          stripTripletFromEvent(ev);
        }
        tripletStart = null;
        tripletMiddle = null;
      }
    }

    // Handle orphaned triplet starts/middles at the end
    if (tripletStart) stripTripletFromEvent(tripletStart);
    if (tripletMiddle) stripTripletFromEvent(tripletMiddle);
  }

  // ============================================================================
  // NEW APPROACH: Build measures by writing ALL treble first, then backup, then ALL bass
  // This ensures treble and bass play simultaneously (not sequentially)
  // ============================================================================

  // Group time groups by measure
  type MeasureEventData = {
    beatPos: number;
    xml: string[];
    beats: number;
    midiNotes: number[];
    staff: number;
    time: number; // original onset time in seconds, for chord merge decisions
  };
  type MeasureData = {
    trebleEvents: MeasureEventData[];
    bassEvents: MeasureEventData[];
  };

  const measuresData: MeasureData[] = [];
  let currentMeasure: MeasureData = { trebleEvents: [], bassEvents: [] };
  let currentBeatPos = 0;

  // Track pending tied notes that need to continue in the next measure
  type PendingTie = {
    midiNotes: number[];
    remainingBeats: number;
    staff: number;
  };
  let pendingTrebleTies: PendingTie[] = [];
  let pendingBassTies: PendingTie[] = [];

  // Helper to add pending ties at the start of a new measure
  const addPendingTiesToMeasure = () => {
    // Add treble ties
    for (const tie of pendingTrebleTies) {
      const beatsThisMeasure = Math.min(tie.remainingBeats, BEATS_PER_MEASURE);
      const tieType =
        tie.remainingBeats > BEATS_PER_MEASURE ? "continue" : "stop";
      const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
      const xml: string[] = [];

      for (let si = 0; si < segments.length; si++) {
        const seg = segments[si];
        const segTie = getSegmentTieType(tieType, si, segments.length);
        let first = xml.length === 0;
        for (const midi of tie.midiNotes) {
          const {
            step: baseStep,
            alter,
            octave,
          } = midiToStepOctaveForKey(midi, fifths);
          const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
          const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
          xml.push(
            generateNoteXmlWithTie(
              pitchXml,
              seg.duration,
              seg.noteType,
              1,
              segTie,
              !first,
              seg.dotted,
            ),
          );
          first = false;
        }
      }

      currentMeasure.trebleEvents.push({
        beatPos: 0,
        xml,
        beats: beatsThisMeasure,
        midiNotes: tie.midiNotes,
        staff: 1,
        time: -1,
      });

      tie.remainingBeats -= beatsThisMeasure;
    }
    // Keep ties that still have remaining beats
    pendingTrebleTies = pendingTrebleTies.filter(
      (t) => t.remainingBeats > 0.001,
    );

    // Add bass ties
    for (const tie of pendingBassTies) {
      const beatsThisMeasure = Math.min(tie.remainingBeats, BEATS_PER_MEASURE);
      const tieType =
        tie.remainingBeats > BEATS_PER_MEASURE ? "continue" : "stop";
      const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
      const xml: string[] = [];

      for (let si = 0; si < segments.length; si++) {
        const seg = segments[si];
        const segTie = getSegmentTieType(tieType, si, segments.length);
        let first = xml.length === 0;
        for (const midi of tie.midiNotes) {
          const {
            step: baseStep,
            alter,
            octave,
          } = midiToStepOctaveForKey(midi, fifths);
          const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
          const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
          xml.push(
            generateNoteXmlWithTie(
              pitchXml,
              seg.duration,
              seg.noteType,
              2,
              segTie,
              !first,
              seg.dotted,
            ),
          );
          first = false;
        }
      }

      currentMeasure.bassEvents.push({
        beatPos: 0,
        xml,
        beats: beatsThisMeasure,
        midiNotes: tie.midiNotes,
        staff: 2,
        time: -1,
      });

      tie.remainingBeats -= beatsThisMeasure;
    }
    pendingBassTies = pendingBassTies.filter((t) => t.remainingBeats > 0.001);
  };

  // Helper to add an event, splitting with ties if it overflows the measure
  const addEventToMeasure = (
    ev: TimelineEvent,
    events: MeasureEventData[],
    pendingTies: PendingTie[],
  ) => {
    const remainingInMeasure = BEATS_PER_MEASURE - currentBeatPos;

    if (ev.beats <= remainingInMeasure + 0.001) {
      // Event fits entirely in this measure
      events.push({
        beatPos: currentBeatPos,
        xml: ev.xml,
        beats: ev.beats,
        midiNotes: ev.midiNotes || [],
        staff: ev.staff,
        time: ev.time,
      });
    } else {
      // Event overflows - split with ties
      const beatsThisMeasure = remainingInMeasure;
      const beatsRemaining = ev.beats - beatsThisMeasure;

      if (beatsThisMeasure > 0.001 && ev.midiNotes && ev.midiNotes.length > 0) {
        // Generate tied note XML for the portion that fits
        const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
        const xml: string[] = [];

        for (let si = 0; si < segments.length; si++) {
          const seg = segments[si];
          const segTie = getSegmentTieType("start", si, segments.length);
          let first = xml.length === 0;
          for (const midi of ev.midiNotes) {
            const {
              step: baseStep,
              alter,
              octave,
            } = midiToStepOctaveForKey(midi, fifths);
            const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
            const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
            xml.push(
              generateNoteXmlWithTie(
                pitchXml,
                seg.duration,
                seg.noteType,
                ev.staff,
                segTie,
                !first,
                seg.dotted,
              ),
            );
            first = false;
          }
        }

        events.push({
          beatPos: currentBeatPos,
          xml,
          beats: beatsThisMeasure,
          midiNotes: ev.midiNotes,
          staff: ev.staff,
          time: ev.time,
        });

        // Queue the remainder for the next measure
        pendingTies.push({
          midiNotes: ev.midiNotes,
          remainingBeats: beatsRemaining,
          staff: ev.staff,
        });
      } else {
        // No room in this measure, just queue the whole thing
        pendingTies.push({
          midiNotes: ev.midiNotes || [],
          remainingBeats: ev.beats,
          staff: ev.staff,
        });
      }
    }
  };

  // Use backend-authored beat positions when available so the renderer does not
  // perform a second quantization pass from wall-clock time.
  let currentMeasureIndex = 0;

  for (const group of timeGroups) {
    const groupMeasureIndex = Math.max(
      0,
      Math.floor(group.beatStart / BEATS_PER_MEASURE),
    );

    // If we've moved to a new measure, finalize previous and create intervening ones
    while (currentMeasureIndex < groupMeasureIndex) {
      if (
        currentMeasure.trebleEvents.length > 0 ||
        currentMeasure.bassEvents.length > 0
      ) {
        measuresData.push(currentMeasure);
      }
      currentMeasure = { trebleEvents: [], bassEvents: [] };
      currentBeatPos = 0;
      currentMeasureIndex++;

      // Add any pending ties to the new measure
      addPendingTiesToMeasure();
    }

    const beatPosFromGrid =
      group.beatStart - currentMeasureIndex * BEATS_PER_MEASURE;

    // Snap to 1/24-beat grid (finest resolution MusicXML can represent
    // with divisions=24). This ensures beat positions are always
    // representable as integer divisions, preventing cumulative drift.
    currentBeatPos = Math.min(
      Math.max(0, Math.round(beatPosFromGrid * 24) / 24),
      BEATS_PER_MEASURE - 1 / 24,
    );

    // Add treble events (with overflow handling)
    for (const ev of group.treble) {
      addEventToMeasure(ev, currentMeasure.trebleEvents, pendingTrebleTies);
    }

    // Add bass events (with overflow handling)
    for (const ev of group.bass) {
      addEventToMeasure(ev, currentMeasure.bassEvents, pendingBassTies);
    }
  }

  // Handle any remaining pending ties
  while (pendingTrebleTies.length > 0 || pendingBassTies.length > 0) {
    if (
      currentMeasure.trebleEvents.length > 0 ||
      currentMeasure.bassEvents.length > 0
    ) {
      measuresData.push(currentMeasure);
    }
    currentMeasure = { trebleEvents: [], bassEvents: [] };
    currentBeatPos = 0;
    addPendingTiesToMeasure();
  }

  // Push final measure if it has content
  if (
    currentMeasure.trebleEvents.length > 0 ||
    currentMeasure.bassEvents.length > 0
  ) {
    measuresData.push(currentMeasure);
  }

  // Now generate XML for each measure
  for (let mIdx = 0; mIdx < measuresData.length; mIdx++) {
    const mData = measuresData[mIdx];
    const measureNum = mIdx + 1;
    let measureContent = "";

    // Attributes only for first measure - with dynamic time signature
    if (measureNum === 1) {
      // Parse time signature for XML
      const [beats, beatType] =
        timeSignature === "6/8"
          ? ["6", "8"]
          : timeSignature === "3/4"
            ? ["3", "4"]
            : ["4", "4"];
      measureContent += `<attributes><divisions>24</divisions><key><fifths>${fifths}</fifths></key><time><beats>${beats}</beats><beat-type>${beatType}</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>`;
      // Add tempo marking for playback
      measureContent += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`;
    }

    // Sort events by beat position
    mData.trebleEvents.sort((a, b) => a.beatPos - b.beatPos);
    mData.bassEvents.sort((a, b) => a.beatPos - b.beatPos);

    // Minimum gap size to insert a rest (must be at least a 32nd note = 0.125 beats)
    // All gaps need rests for correct playback timing — skipping rests causes duration drift
    const MIN_REST_GAP = 0.12; // ~32nd note — smallest representable rest

    // ============================================================================
    // CRITICAL FIX: Ensure total beats in measure don't exceed BEATS_PER_MEASURE
    // Group events by beat position - events at same position are concurrent (chord)
    // Also deduplicate MIDI notes to prevent overlapping noteheads
    // ============================================================================

    // Helper to extract MIDI note from a note XML element
    const midiFromXml = (xml: string): number | null => {
      const m = xml.match(
        /<pitch><step>(\w+)<\/step>(?:<alter>(-?\d)<\/alter>)?<octave>(\d+)<\/octave><\/pitch>/,
      );
      if (!m) return null;
      const stepNames: Record<string, number> = {
        C: 0,
        D: 2,
        E: 4,
        F: 5,
        G: 7,
        A: 9,
        B: 11,
      };
      const base = stepNames[m[1]] ?? 0;
      const alter = m[2] ? parseInt(m[2]) : 0;
      const octave = parseInt(m[3]);
      return (octave + 1) * 12 + base + alter;
    };

    const consolidateEvents = (
      events: MeasureEventData[],
    ): MeasureEventData[] => {
      if (events.length === 0) return events;

      // Sort by beatPos
      events.sort((a, b) => a.beatPos - b.beatPos);

      // Group events by beat position
      const consolidated: MeasureEventData[] = [];
      let currentGroup: MeasureEventData[] = [events[0]];

      const finalizeGroup = (group: MeasureEventData[]) => {
        const maxBeats = Math.max(...group.map((e) => e.beats));
        const canonicalDuration = Math.round(maxBeats * 24);
        const mergedXml: string[] = [];
        const mergedMidi: number[] = [];
        const seenMidi = new Set<number>();
        let first = true;
        for (const groupEv of group) {
          for (let xi = 0; xi < groupEv.xml.length; xi++) {
            let xml = groupEv.xml[xi];
            const midi =
              xi < groupEv.midiNotes.length
                ? groupEv.midiNotes[xi]
                : midiFromXml(xml);

            // Skip duplicate MIDI notes at the same beat position
            if (midi !== null && seenMidi.has(midi)) continue;
            if (midi !== null) seenMidi.add(midi);

            // Normalize duration so all chord notes match
            xml = xml.replace(
              /<duration>\d+<\/duration>/,
              `<duration>${canonicalDuration}</duration>`,
            );

            if (first) {
              mergedXml.push(xml);
              first = false;
            } else if (!xml.includes("<chord/>")) {
              mergedXml.push(xml.replace("<note>", "<note><chord/>"));
            } else {
              mergedXml.push(xml);
            }
            if (midi !== null) mergedMidi.push(midi);
          }
        }
        consolidated.push({
          beatPos: group[0].beatPos,
          xml: mergedXml,
          beats: maxBeats,
          midiNotes: mergedMidi,
          staff: group[0].staff,
          time: group[0].time,
        });
      };

      for (let i = 1; i < events.length; i++) {
        const ev = events[i];
        // Only merge events that are at the same beat position AND from the same
        // original onset time. This prevents sequential notes that ended up at the
        // same beatPos (because they were in the same time group) from being
        // incorrectly merged into chords.
        const sameBeatPos =
          Math.abs(ev.beatPos - currentGroup[0].beatPos) < 0.001;
        const sameTime =
          Math.abs(ev.time - currentGroup[0].time) < SAME_STAFF_TOLERANCE; // 5ms
        if (sameBeatPos && sameTime) {
          // Same beat position - add to current group
          currentGroup.push(ev);
        } else {
          // Different beat position - finalize current group and start new one
          finalizeGroup(currentGroup);
          currentGroup = [ev];
        }
      }

      // Finalize last group
      if (currentGroup.length > 0) {
        finalizeGroup(currentGroup);
      }

      return consolidated;
    };

    // Clamp events to ensure total doesn't exceed measure
    // Preserves gaps where they exist in original positions
    // Returns { events, overflows } — overflows need to be queued as pending ties
    type ClampOverflow = {
      midiNotes: number[];
      remainingBeats: number;
      staff: number;
    };
    const clampEventsToMeasure = (
      events: MeasureEventData[],
    ): { events: MeasureEventData[]; overflows: ClampOverflow[] } => {
      if (events.length === 0) return { events, overflows: [] };

      const result: MeasureEventData[] = [];
      const overflows: ClampOverflow[] = [];
      let currentBeatEnd = 0; // Track where the last note ended

      for (const ev of events) {
        // Start position is either the original beatPos or where the last note ended
        // (use max to not overlap if original position is behind current end)
        const startPos = Math.max(ev.beatPos, currentBeatEnd);

        if (startPos >= BEATS_PER_MEASURE - 0.001) {
          // No room left in this measure. Don't drop the note — carry the
          // whole event into the next measure as an overflow tie so it still
          // appears in the score (matching what the MIDI export contains).
          if (ev.midiNotes.length > 0) {
            overflows.push({
              midiNotes: ev.midiNotes,
              remainingBeats: ev.beats,
              staff: ev.staff,
            });
          }
          continue;
        }

        // Calculate how much room is left from this start position
        const remaining = BEATS_PER_MEASURE - startPos;

        if (ev.beats <= remaining + 0.001) {
          // Event fits - add with updated position
          result.push({ ...ev, beatPos: startPos });
          currentBeatEnd = startPos + ev.beats;
        } else {
          // Event needs to be truncated - use the remaining space
          const truncatedBeats = remaining;
          if (truncatedBeats < 0.125) {
            // Less than a 32nd note of room — too little to notate here.
            // Carry the whole event to the next measure instead of dropping it.
            if (ev.midiNotes.length > 0) {
              overflows.push({
                midiNotes: ev.midiNotes,
                remainingBeats: ev.beats,
                staff: ev.staff,
              });
            }
            continue;
          }
          const segments = splitBeatsIntoNoteTypes(truncatedBeats);
          const xml: string[] = [];

          const outerTie: "start" | "continue" = ev.xml.some((x) =>
            x.includes('tie type="stop"'),
          )
            ? "continue"
            : "start";

          for (let si = 0; si < segments.length; si++) {
            const seg = segments[si];
            const segTie = getSegmentTieType(outerTie, si, segments.length);
            let first = xml.length === 0;
            for (const midi of ev.midiNotes) {
              const {
                step: baseStep,
                alter,
                octave,
              } = midiToStepOctaveForKey(midi, fifths);
              const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
              const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
              xml.push(
                generateNoteXmlWithTie(
                  pitchXml,
                  seg.duration,
                  seg.noteType,
                  ev.staff,
                  segTie,
                  !first,
                  seg.dotted,
                ),
              );
              first = false;
            }
          }

          result.push({
            beatPos: startPos,
            xml,
            beats: truncatedBeats,
            midiNotes: ev.midiNotes,
            staff: ev.staff,
            time: ev.time,
          });

          // Queue the overflow for the next measure as a pending tie
          const overflowBeats = ev.beats - truncatedBeats;
          if (overflowBeats > 0.001 && ev.midiNotes.length > 0) {
            overflows.push({
              midiNotes: ev.midiNotes,
              remainingBeats: overflowBeats,
              staff: ev.staff,
            });
          }

          currentBeatEnd = BEATS_PER_MEASURE;
        }
      }

      return { events: result, overflows };
    };

    // Apply consolidation and clamping to treble events
    const consolidatedTreble = consolidateEvents([...mData.trebleEvents]);
    const trebleClampResult = clampEventsToMeasure(consolidatedTreble);
    const clampedTreble = trebleClampResult.events;
    // Queue any treble overflows as pending ties for the next measure
    for (const ov of trebleClampResult.overflows) {
      pendingTrebleTies.push(ov);
    }

    // Write ALL treble events first (with rests only for significant gaps)
    let trebleBeatPos = 0;
    for (let evIdx = 0; evIdx < clampedTreble.length; evIdx++) {
      const ev = clampedTreble[evIdx];
      const gap = ev.beatPos - trebleBeatPos;

      if (gap < -0.001) {
        // Negative gap: rounding caused trebleBeatPos to overshoot.
        // Snap back to avoid compounding drift.
        trebleBeatPos = ev.beatPos;
      } else if (gap > MIN_REST_GAP) {
        // Add a rest for the gap — use actual emitted beats for tracking
        const restResult = generateRestXml(gap, 1);
        measureContent += restResult.xml.join("");
        trebleBeatPos += restResult.beatsEmitted;
      } else if (gap > 0.001) {
        // Small gap — use <forward> to keep duration accounting correct
        const forwardDivisions = Math.round(gap * 24);
        if (forwardDivisions > 0) {
          measureContent += `<forward><duration>${forwardDivisions}</duration></forward>`;
          trebleBeatPos += forwardDivisions / 24;
        }
      }

      // Add the notes (chord tags already added during merging phase)
      measureContent += ev.xml.join("");
      trebleBeatPos += ev.beats;
      // Re-snap to 1/24 grid to prevent floating-point drift accumulation
      trebleBeatPos = Math.round(trebleBeatPos * 24) / 24;
    }

    // Pad treble to fill measure
    const trebleShortfall =
      Math.round((BEATS_PER_MEASURE - trebleBeatPos) * 24) / 24;
    if (trebleShortfall > 0.001) {
      const padResult = generateRestXml(trebleShortfall, 1);
      measureContent += padResult.xml.join("");
      trebleBeatPos += padResult.beatsEmitted;
    }

    // Backup to start of measure for bass staff
    const backupDuration = Math.round(trebleBeatPos * 24);
    if (backupDuration > 0) {
      measureContent += `<backup><duration>${backupDuration}</duration></backup>`;
    }

    // Apply consolidation and clamping to bass events
    const consolidatedBass = consolidateEvents([...mData.bassEvents]);
    const bassClampResult = clampEventsToMeasure(consolidatedBass);
    const clampedBass = bassClampResult.events;
    // Queue any bass overflows as pending ties for the next measure
    for (const ov of bassClampResult.overflows) {
      pendingBassTies.push(ov);
    }

    // Write ALL bass events (with rests only for significant gaps)
    let bassBeatPos = 0;
    for (let evIdx = 0; evIdx < clampedBass.length; evIdx++) {
      const ev = clampedBass[evIdx];
      const gap = ev.beatPos - bassBeatPos;

      if (gap < -0.001) {
        // Negative gap: rounding caused bassBeatPos to overshoot.
        // Snap back to avoid compounding drift.
        bassBeatPos = ev.beatPos;
      } else if (gap > MIN_REST_GAP) {
        // Add a rest for the gap — use actual emitted beats for tracking
        const restResult = generateRestXml(gap, 2);
        measureContent += restResult.xml.join("");
        bassBeatPos += restResult.beatsEmitted;
      } else if (gap > 0.001) {
        // Small gap — use <forward> to keep duration accounting correct
        const forwardDivisions = Math.round(gap * 24);
        if (forwardDivisions > 0) {
          measureContent += `<forward><duration>${forwardDivisions}</duration></forward>`;
          bassBeatPos += forwardDivisions / 24;
        }
      }

      // Add the notes (chord tags already added during merging phase)
      measureContent += ev.xml.join("");
      bassBeatPos += ev.beats;
      // Re-snap to 1/24 grid to prevent floating-point drift accumulation
      bassBeatPos = Math.round(bassBeatPos * 24) / 24;
    }

    // Pad bass to fill measure
    const bassShortfall =
      Math.round((BEATS_PER_MEASURE - bassBeatPos) * 24) / 24;
    if (bassShortfall > 0.001) {
      const padResult = generateRestXml(bassShortfall, 2);
      measureContent += padResult.xml.join("");
    }

    measures.push(
      `<measure number="${measureNum}">${measureContent}</measure>`,
    );
  }

  return measures;
}

export function generateMusicXML(
  notes: NoteResult[],
  chords: ChordResult[],
  timeSignature: "4/4" | "3/4" | "6/8" = "4/4",
  bpm: number = 120,
  fifths: number = 0,
): string {
  const measures = generateMeasureXmls(
    notes,
    chords,
    timeSignature,
    bpm,
    fifths,
  );
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n<score-partwise version="3.1">\n  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n  <part id="P1">${measures.join("")}</part></score-partwise>`;
  // console.log(xml);
  return xml;
}

function generateBlankPageMusicXML(
  timeSignature: "4/4" | "3/4" | "6/8" = "4/4",
  bpm: number = 120,
  fifths: number = 0,
  measureCount: number = 12,
): string {
  const timeConfig =
    timeSignature === "6/8"
      ? {
          beats: "6",
          beatType: "8",
          measureDuration: 72,
        }
      : timeSignature === "3/4"
        ? {
            beats: "3",
            beatType: "4",
            measureDuration: 72,
          }
        : {
            beats: "4",
            beatType: "4",
            measureDuration: 96,
          };

  const measures = Array.from({ length: measureCount }, (_, measureIndex) => {
    const measureNumber = measureIndex + 1;
    let measureContent = "";

    if (measureNumber === 1) {
      measureContent += `<attributes><divisions>24</divisions><key><fifths>${fifths}</fifths></key><time><beats>${timeConfig.beats}</beats><beat-type>${timeConfig.beatType}</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>`;
      measureContent += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`;
    }

    measureContent += `<note><rest measure="yes"/><duration>${timeConfig.measureDuration}</duration><voice>1</voice><staff>1</staff></note>`;
    measureContent += `<backup><duration>${timeConfig.measureDuration}</duration></backup>`;
    measureContent += `<note><rest measure="yes"/><duration>${timeConfig.measureDuration}</duration><voice>2</voice><staff>2</staff></note>`;

    return `<measure number="${measureNumber}">${measureContent}</measure>`;
  }).join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">${measures}</part>
</score-partwise>`;
}

// Blank grand staff shown before live chunks arrive so the engraving surface is stable.
const FALLBACK_XML = generateBlankPageMusicXML();

// Drop-in replacement component:
export default function PianoSheetMusic({
  results,
  timeSignature = "4/4",
  keySignature = 0,
  compact = false,
  viewportHeight,
  refinementVersion,
  onScoreRendered,
  onScoreScrollActiveChange,
}: PianoSheetMusicProps) {
  // accumulate incoming note blocks so live updates append to the end
  const [accumulatedNotes, setAccumulatedNotes] = useState<NoteResult[]>([]);
  const [accumulatedChords, setAccumulatedChords] = useState<ChordResult[]>([]);
  const hasReceivedDataRef = useRef<boolean>(false);
  const lastRefinementVersionRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    // If results is null and we've previously received data, this is a reset signal
    if (!results) {
      if (hasReceivedDataRef.current) {
        // Reset accumulated notes/chords when explicitly cleared
        setAccumulatedNotes([]);
        setAccumulatedChords([]);
        hasReceivedDataRef.current = false;
      }
      return;
    }

    const incomingNotes = results.notes ?? [];
    const incomingChords = results.chords ?? [];

    // Check if this is a refinement update (version changed)
    const isRefinementUpdate =
      refinementVersion !== undefined &&
      refinementVersion !== lastRefinementVersionRef.current;
    lastRefinementVersionRef.current = refinementVersion;

    // If it's a refinement update, replace all notes/chords instead of appending
    // This ensures the refined rhythm values are used
    if (
      isRefinementUpdate &&
      (incomingNotes.length > 0 || incomingChords.length > 0)
    ) {
      hasReceivedDataRef.current = true;
      setAccumulatedNotes(incomingNotes);
      setAccumulatedChords(incomingChords);
      return;
    }

    if (incomingNotes.length === 0 && incomingChords.length === 0) return;

    // Mark that we've received data
    hasReceivedDataRef.current = true;

    // Always append incoming notes/chords - no automatic reset based on timestamps
    // (Reset should only happen when explicitly triggered, not on every chunk)

    // Append incoming notes, deduplicating by (time_seconds, midi_note)
    setAccumulatedNotes((prev) => {
      const seen = new Set<string>();
      for (const n of prev)
        seen.add(`${n.time_seconds.toFixed(6)}:${n.midi_note}`);
      const toAdd: NoteResult[] = [];
      for (const n of incomingNotes) {
        const key = `${(n.time_seconds ?? 0).toFixed(6)}:${n.midi_note}`;
        if (!seen.has(key)) {
          seen.add(key);
          toAdd.push(n);
        }
      }
      return [...prev, ...toAdd];
    });

    // Append incoming chords, deduplicating by (time_seconds, midi list)
    setAccumulatedChords((prev) => {
      const seen = new Set<string>();
      for (const c of prev)
        seen.add(
          `${(c.time_seconds ?? 0).toFixed(6)}:${(c.midi_notes || []).join("-")}`,
        );
      const toAdd: ChordResult[] = [];
      for (const c of incomingChords) {
        const key = `${(c.time_seconds ?? 0).toFixed(6)}:${(c.midi_notes || []).join("-")}`;
        if (!seen.has(key)) {
          seen.add(key);
          toAdd.push(c);
        }
      }
      return [...prev, ...toAdd];
    });
  }, [results, refinementVersion]);

  // Get detected BPM from results, default to 120
  const detectedBPM = results?.analysis_summary?.detected_bpm ?? 120;

  const score = useMemo(() => {
    if (
      (!accumulatedNotes || accumulatedNotes.length === 0) &&
      (!accumulatedChords || accumulatedChords.length === 0)
    )
      return FALLBACK_XML;
    return generateMusicXML(
      accumulatedNotes,
      accumulatedChords,
      timeSignature,
      detectedBPM,
      keySignature,
    );
  }, [
    accumulatedNotes,
    accumulatedChords,
    timeSignature,
    detectedBPM,
    keySignature,
  ]);
  const hasPlayableScore = score !== FALLBACK_XML;

  const webRef = useRef<WebView>(null);
  const source = useMemo(
    () => ({
      html: OSMD_HTML,
      baseUrl: "https://osmd.local/",
    }),
    [],
  );
  const shouldFollowLatest = results?.analysis_summary?.method === "live";
  const measuresSentRef = useRef<number>(0);
  const lastXmlRef = useRef<string | null>(null);
  const pendingXmlRef = useRef<string | null>(null);
  const playAfterRenderRef = useRef(false);
  const pendingSentAtRef = useRef<number>(0);
  const pendingRenderIdRef = useRef<number | null>(null);
  const nextRenderIdRef = useRef<number>(1);
  const renderProbeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const renderRefinementRef = useRef<number | undefined>(undefined);
  const [, setDebugSnapshot] = useState<OsmdDebugSnapshot | null>(null);
  const [, setDebugEvents] = useState<string[]>([]);

  const appendDebugEvent = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    const line = `${timestamp} ${message}`;

    // console.log("[OSMD Debug]", line);

    setDebugEvents((prev) => [line, ...prev].slice(0, 8));
  }, []);

  const injectWebCommand = useCallback(
    (script: string, description: string) => {
      if (!webRef.current) {
        return;
      }

      webRef.current.injectJavaScript(`
        try {
          ${script}
        } catch (e) {
          try {
            if (window.ReactNativeWebView) {
              window.ReactNativeWebView.postMessage(JSON.stringify({
                type: 'error',
                error: 'Injected command failed (${description}): ' + String(e),
              }));
            }
          } catch (_bridgeError) {}
        }
        true;
      `);
    },
    [],
  );

  const requestDebugSnapshot = useCallback(
    (reason: string, requestId?: number | null) => {
      injectWebCommand(
        `
          if (window.__OSMD_DEBUG_SNAPSHOT) {
            window.__OSMD_DEBUG_SNAPSHOT(${JSON.stringify(reason)}, ${requestId ?? null});
          }
        `,
        `debug-snapshot-${reason}`,
      );
    },
    [injectWebCommand],
  );

  const sendRenderXml = useCallback(
    (xml: string, description: string, extraScript = "") => {
      const requestId = nextRenderIdRef.current;

      nextRenderIdRef.current += 1;
      pendingXmlRef.current = xml;
      pendingSentAtRef.current = Date.now();
      pendingRenderIdRef.current = requestId;
      appendDebugEvent(
        `render request #${requestId} (${description}) xml=${xml.length}`,
      );

      if (renderProbeTimeoutRef.current) {
        clearTimeout(renderProbeTimeoutRef.current);
      }

      renderProbeTimeoutRef.current = setTimeout(() => {
        renderProbeTimeoutRef.current = null;
        requestDebugSnapshot(`render-timeout:${description}`, requestId);
      }, 1500);

      injectWebCommand(
        `
          if (window.__OSMD_RENDER_XML) window.__OSMD_RENDER_XML(${JSON.stringify(xml)}, ${requestId});
          ${extraScript}
        `,
        description,
      );
    },
    [appendDebugEvent, injectWebCommand, requestDebugSnapshot],
  );

  const [isLandscape, setIsLandscape] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  // Initialize BPM from detected tempo, default to 120
  const [playbackBPM, setPlaybackBPM] = useState(detectedBPM);
  const [cameraMotionMode, setCameraMotionMode] =
    useState<CameraMotionMode>("smooth");
  const compactViewportHeight = Math.max(
    220,
    Math.round(viewportHeight ?? 280),
  );
  const lastDetectedBPMRef = useRef<number | undefined>(undefined);
  // webViewReady is state (not just ref) so the render effect re-fires when
  // the bridge becomes ready *after* the score is already populated.
  const [webViewReady, setWebViewReady] = useState(false);
  const webViewReadyRef = useRef<boolean>(false);
  const scoreScrollActiveRef = useRef(false);

  const updateScoreScrollActive = useCallback(
    (active: boolean) => {
      if (scoreScrollActiveRef.current === active) {
        return;
      }

      scoreScrollActiveRef.current = active;
      onScoreScrollActiveChange?.(active);
    },
    [onScoreScrollActiveChange],
  );

  useEffect(() => {
    return () => {
      updateScoreScrollActive(false);
    };
  }, [updateScoreScrollActive]);

  const requestScorePlayback = useCallback(() => {
    if (score === FALLBACK_XML) {
      appendDebugEvent("playback blocked: fallback score");
      return;
    }

    if (
      webViewReadyRef.current &&
      !pendingXmlRef.current &&
      lastXmlRef.current
    ) {
      injectWebCommand(
        `if (window.__OSMD_PLAY) window.__OSMD_PLAY(${JSON.stringify(playbackBPM)});`,
        "play-score",
      );
      return;
    }

    playAfterRenderRef.current = true;
    appendDebugEvent("queued playback until score render completes");
  }, [appendDebugEvent, injectWebCommand, playbackBPM, score]);

  const handlePlayPause = useCallback(() => {
    if (!isPlaying || isPaused) {
      requestScorePlayback();
      return;
    }

    injectWebCommand(
      `if (window.__OSMD_PAUSE) window.__OSMD_PAUSE();`,
      "pause-score",
    );
  }, [injectWebCommand, isPaused, isPlaying, requestScorePlayback]);

  const handleStopPlayback = useCallback(() => {
    playAfterRenderRef.current = false;
    injectWebCommand(
      `if (window.__OSMD_STOP) window.__OSMD_STOP();`,
      "stop-score",
    );
  }, [injectWebCommand]);

  const updatePlaybackTempo = useCallback(
    (delta: number) => {
      const nextBpm = Math.max(40, Math.min(240, playbackBPM + delta));
      if (nextBpm === playbackBPM) {
        return;
      }

      setPlaybackBPM(nextBpm);
      injectWebCommand(
        `if (window.__OSMD_SET_BPM) window.__OSMD_SET_BPM(${JSON.stringify(nextBpm)});`,
        delta > 0 ? "increase-bpm" : "decrease-bpm",
      );
    },
    [injectWebCommand, playbackBPM],
  );

  useEffect(() => {
    if (detectedBPM && detectedBPM !== lastDetectedBPMRef.current) {
      lastDetectedBPMRef.current = detectedBPM;
      setPlaybackBPM(detectedBPM);
      if (webViewReadyRef.current) {
        injectWebCommand(
          `if (window.__OSMD_SET_BPM) window.__OSMD_SET_BPM(${JSON.stringify(detectedBPM)});`,
          "set-bpm-detected",
        );
      }
    }
  }, [detectedBPM, injectWebCommand]);

  const onWebMessage = useCallback(
    async (e: WebViewMessageEvent) => {
      try {
        const msg = JSON.parse(e.nativeEvent.data);
        if (msg.type === "webview-click") {
          // Only enter landscape mode (exit is handled by exit button)
          if (!isLandscape) {
            try {
              await ScreenOrientation.lockAsync(
                ScreenOrientation.OrientationLock.LANDSCAPE,
              );
              setIsLandscape(true);
            } catch (err) {
              console.warn("Orientation lock failed", err);
            }
          }
          return;
        }

        if (msg.type === "scoreScrollActive") {
          updateScoreScrollActive(Boolean(msg.active));
          return;
        }

        if (msg.type === "ready") {
          // Mark WebView as ready
          webViewReadyRef.current = true;
          setWebViewReady(true);
          appendDebugEvent("webview ready");
          injectWebCommand(
            `
              if (window.__OSMD_SET_FOLLOW_TAIL) window.__OSMD_SET_FOLLOW_TAIL(${shouldFollowLatest ? "true" : "false"});
              if (window.__OSMD_SET_CAMERA_MODE) window.__OSMD_SET_CAMERA_MODE(${JSON.stringify(cameraMotionMode)});
              if (window.__OSMD_TOGGLE_CURSOR) window.__OSMD_TOGGLE_CURSOR(true);
              if (window.__OSMD_SET_BPM) window.__OSMD_SET_BPM(${JSON.stringify(playbackBPM ?? 120)});
            `,
            "sync-ready-state",
          );
          requestDebugSnapshot("ready");
        }
        if (msg.type === "rendered") {
          if (
            pendingRenderIdRef.current !== null &&
            msg.requestId !== pendingRenderIdRef.current
          ) {
            return;
          }

          if (renderProbeTimeoutRef.current) {
            clearTimeout(renderProbeTimeoutRef.current);
            renderProbeTimeoutRef.current = null;
          }

          // initial main render completed; mark how many measures are present
          if (typeof msg.measures === "number") {
            measuresSentRef.current = msg.measures;
            onScoreRendered?.(msg.measures);
            appendDebugEvent(
              `rendered request #${msg.requestId ?? "?"} measures=${msg.measures}`,
            );
          }
          // The webview has finished rendering the previously-sent xml; promote pending -> last
          if (pendingXmlRef.current) {
            lastXmlRef.current = pendingXmlRef.current;
            pendingXmlRef.current = null;
            pendingSentAtRef.current = 0;
            pendingRenderIdRef.current = null;
          }

          if (playAfterRenderRef.current && lastXmlRef.current) {
            playAfterRenderRef.current = false;
            appendDebugEvent("starting queued playback after render");
            injectWebCommand(
              `if (window.__OSMD_PLAY) window.__OSMD_PLAY(${JSON.stringify(playbackBPM)});`,
              "play-score-after-render",
            );
          }
        }
        if (msg.type === "appended") {
          if (typeof msg.appended === "number") {
            measuresSentRef.current += msg.appended;
          }
        }
        if (msg.type === "error") {
          appendDebugEvent(`error ${String(msg.error).slice(0, 120)}`);
          console.warn("OSMD error:", msg.error);
        }
        if (msg.type === "debugState") {
          setDebugSnapshot(msg.snapshot ?? null);
          appendDebugEvent(
            `snapshot ${msg.snapshot?.reason ?? "unknown"} measures=${msg.snapshot?.renderedMeasureCount ?? "?"} svg=${msg.snapshot?.stageSvgCount ?? "?"}`,
          );
        }

        if (msg.type === "playbackStarted") {
          setIsPlaying(true);
          setIsPaused(false);
          console.log(
            "Playback started:",
            msg.noteCount,
            "notes,",
            msg.duration.toFixed(1),
            "seconds",
          );
        }
        if (msg.type === "playbackPaused") {
          setIsPaused(true);
        }
        if (msg.type === "playbackResumed") {
          setIsPaused(false);
        }
        if (msg.type === "playbackStopped" || msg.type === "playbackEnded") {
          setIsPlaying(false);
          setIsPaused(false);
        }
        if (msg.type === "playbackError") {
          console.warn("Playback error:", msg.error);
          playAfterRenderRef.current = false;
          requestDebugSnapshot("playback-error");
          Alert.alert("Playback Error", String(msg.error ?? "Unknown error"));
          setIsPlaying(false);
          setIsPaused(false);
        }
        if (msg.type === "bpmSet") {
          setPlaybackBPM(msg.bpm);
        }
        if (msg.type === "exitFullscreen") {
          try {
            await ScreenOrientation.lockAsync(
              ScreenOrientation.OrientationLock.PORTRAIT_UP,
            );
            setIsLandscape(false);
          } catch (err) {
            console.warn("Exit fullscreen failed", err);
          }
        }
        if (msg.type === "bpmChanged") {
          setPlaybackBPM(msg.bpm);
        }
      } catch (err) {
        console.warn("webview message parse error", err);
      }
    },
    [
      appendDebugEvent,
      injectWebCommand,
      isLandscape,
      onScoreRendered,
      cameraMotionMode,
      playbackBPM,
      requestDebugSnapshot,
      shouldFollowLatest,
      updateScoreScrollActive,
    ],
  );

  const onWebLoadStart = useCallback(() => {
    appendDebugEvent("webview load start");
    updateScoreScrollActive(false);
    setWebViewReady(false);
    webViewReadyRef.current = false;
    measuresSentRef.current = 0;
    lastXmlRef.current = null;
    pendingXmlRef.current = null;
    playAfterRenderRef.current = false;
    pendingSentAtRef.current = 0;
    pendingRenderIdRef.current = null;
    renderRefinementRef.current = undefined;
    setDebugSnapshot(null);

    if (renderProbeTimeoutRef.current) {
      clearTimeout(renderProbeTimeoutRef.current);
      renderProbeTimeoutRef.current = null;
    }
  }, [appendDebugEvent, updateScoreScrollActive]);

  const onWebLoadEnd = useCallback(() => {
    appendDebugEvent("webview load end");
    requestDebugSnapshot("load-end");
  }, [appendDebugEvent, requestDebugSnapshot]);

  const onWebError = useCallback(
    (event: any) => {
      appendDebugEvent(
        `webview error ${String(event?.nativeEvent?.description ?? "unknown")}`,
      );
      console.warn("[PianoSheetMusic] WebView error", event?.nativeEvent);
    },
    [appendDebugEvent],
  );

  const onWebHttpError = useCallback(
    (event: any) => {
      appendDebugEvent(
        `webview http error ${String(event?.nativeEvent?.statusCode ?? "unknown")}`,
      );
      console.warn("[PianoSheetMusic] WebView HTTP error", event?.nativeEvent);
    },
    [appendDebugEvent],
  );

  // If the score changes (for example after live recording produces results), send the new XML
  // to the WebView so OSMD re-renders the updated score.
  useEffect(() => {
    // small guard: only post if webview ref exists
    if (!webRef.current) return;
    if (!webViewReady) return;

    // If a deferred refinement landed, the note count may not change but the
    // rhythm values did — force a clean full re-render so OSMD redraws.
    if (
      refinementVersion !== undefined &&
      refinementVersion !== renderRefinementRef.current
    ) {
      renderRefinementRef.current = refinementVersion;
      measuresSentRef.current = 0;
      lastXmlRef.current = null;
      pendingXmlRef.current = null;
      pendingSentAtRef.current = 0;
      pendingRenderIdRef.current = null;
    }

    // Recover from a missed `rendered` ack: if a pending XML has been waiting
    // more than 8 seconds, drop it and treat the next render as fresh.
    if (
      pendingXmlRef.current &&
      pendingSentAtRef.current > 0 &&
      Date.now() - pendingSentAtRef.current > 8000
    ) {
      pendingXmlRef.current = null;
      pendingSentAtRef.current = 0;
      measuresSentRef.current = 0;
      lastXmlRef.current = null;
      pendingRenderIdRef.current = null;
    }

    const measures = generateMeasureXmls(
      accumulatedNotes,
      accumulatedChords,
      timeSignature,
      undefined,
      keySignature,
    );
    const currentScoreUsesFallback = score === FALLBACK_XML;
    const pendingScoreUsesFallback = pendingXmlRef.current === FALLBACK_XML;
    const lastScoreUsesFallback = lastXmlRef.current === FALLBACK_XML;

    if (
      !currentScoreUsesFallback &&
      (pendingScoreUsesFallback || lastScoreUsesFallback)
    ) {
      pendingXmlRef.current = null;
      pendingSentAtRef.current = 0;
      measuresSentRef.current = 0;
      lastXmlRef.current = null;
      pendingRenderIdRef.current = null;
    }

    // If we have never sent anything, send the full score and save it
    if (measuresSentRef.current === 0 || !lastXmlRef.current) {
      try {
        sendRenderXml(
          score,
          "render-full-score",
          "if (window.__OSMD_TOGGLE_CURSOR) window.__OSMD_TOGGLE_CURSOR(true);",
        );
      } catch (e) {
        console.warn("renderXml post failed", e);
      }
      return;
    }

    // Otherwise, if there are new measures, compose a combined XML by inserting
    // the new measures into the previously-sent XML, then post the combined XML.
    if (measures.length > measuresSentRef.current) {
      if (!lastXmlRef.current) {
        sendRenderXml(score, "render-full-score-retry");
        return;
      }

      const newMeasures = measures.slice(measuresSentRef.current);

      // Helper: strip <attributes> blocks and renumber measure numbers relative to existing count
      const existingCount = measuresSentRef.current;
      const adjusted = newMeasures.map((m) => {
        // remove attributes
        const noAttrs = m.replace(/<attributes>[\s\S]*?<\/attributes>/i, "");
        // renumber number="n" by adding existingCount
        return noAttrs.replace(/number\s*=\s*"(\d+)"/i, function (_, p1) {
          return 'number="' + (existingCount + parseInt(p1, 10)) + '"';
        });
      });

      // insert adjusted measures into lastXmlRef before the last </part> or </score-partwise>
      let base = lastXmlRef.current || "";
      const closingPart = "</part>";
      let newXml;
      const idx = base.lastIndexOf(closingPart);
      if (idx !== -1) {
        newXml = base.slice(0, idx) + adjusted.join("") + base.slice(idx);
      } else {
        const closingScore = "</score-partwise>";
        const idx2 = base.lastIndexOf(closingScore);
        if (idx2 !== -1)
          newXml = base.slice(0, idx2) + adjusted.join("") + base.slice(idx2);
        else newXml = base + adjusted.join("");
      }

      try {
        sendRenderXml(newXml, "render-appended-score");
      } catch (e) {
        console.warn("Failed posting appended renderXml", e);
      }
    }
  }, [
    accumulatedNotes,
    accumulatedChords,
    score,
    timeSignature,
    keySignature,
    refinementVersion,
    injectWebCommand,
    sendRenderXml,
    webViewReady,
  ]);

  // unlock orientation on unmount to avoid locking the device permanently
  useEffect(() => {
    return () => {
      if (renderProbeTimeoutRef.current) {
        clearTimeout(renderProbeTimeoutRef.current);
      }
      ScreenOrientation.unlockAsync().catch(() => {});
    };
  }, []);

  // Notify WebView when fullscreen mode changes
  useEffect(() => {
    injectWebCommand(
      `if (window.__OSMD_SET_FULLSCREEN) window.__OSMD_SET_FULLSCREEN(${isLandscape ? "true" : "false"});`,
      "set-fullscreen-mode",
    );
  }, [injectWebCommand, isLandscape]);

  useEffect(() => {
    if (!webViewReady) return;

    injectWebCommand(
      `if (window.__OSMD_SET_FOLLOW_TAIL) window.__OSMD_SET_FOLLOW_TAIL(${shouldFollowLatest ? "true" : "false"});`,
      "set-follow-tail",
    );
  }, [injectWebCommand, shouldFollowLatest, webViewReady]);

  useEffect(() => {
    if (!webViewReady) return;

    injectWebCommand(
      `if (window.__OSMD_SET_CAMERA_MODE) window.__OSMD_SET_CAMERA_MODE(${JSON.stringify(cameraMotionMode)});`,
      "set-camera-mode",
    );
  }, [cameraMotionMode, injectWebCommand, webViewReady]);

  return (
    <View style={[styles.container, compact ? styles.compactContainer : null]}>
      <View
        style={[
          styles.mainContainer,
          compact ? styles.compactMainContainer : null,
        ]}
      >
        {compact ? null : (
          <View style={styles.playbackSection}>
            <View style={styles.playbackControls}>
              <View style={styles.playbackButtonRow}>
                <TouchableOpacity
                  activeOpacity={0.85}
                  style={[
                    styles.controlButton,
                    styles.playbackButton,
                    isPlaying && !isPaused
                      ? styles.warningControlButton
                      : styles.successControlButton,
                  ]}
                  onPress={handlePlayPause}
                >
                  <ThemedText style={styles.controlButtonText}>
                    {isPlaying ? (isPaused ? "Resume" : "Pause") : "Play"}
                  </ThemedText>
                </TouchableOpacity>
                <TouchableOpacity
                  activeOpacity={0.85}
                  style={[
                    styles.controlButton,
                    styles.playbackButton,
                    styles.dangerControlButton,
                  ]}
                  onPress={handleStopPlayback}
                >
                  <ThemedText style={styles.controlButtonText}>Stop</ThemedText>
                </TouchableOpacity>
              </View>

              <View style={styles.tempoControl}>
                <TouchableOpacity
                  activeOpacity={0.85}
                  style={styles.stepperButton}
                  onPress={() => updatePlaybackTempo(-10)}
                >
                  <ThemedText style={styles.stepperButtonText}>-</ThemedText>
                </TouchableOpacity>
                <View style={styles.bpmDisplay}>
                  <ThemedText style={styles.bpmValue}>{playbackBPM}</ThemedText>
                  <ThemedText style={styles.bpmLabel}>Tempo</ThemedText>
                </View>
                <TouchableOpacity
                  activeOpacity={0.85}
                  style={styles.stepperButton}
                  onPress={() => updatePlaybackTempo(10)}
                >
                  <ThemedText style={styles.stepperButtonText}>+</ThemedText>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        )}

        <View
          style={[
            styles.scoreSection,
            compact
              ? {
                  flex: 1,
                  minHeight: compactViewportHeight,
                  marginBottom: 0,
                  borderWidth: 0,
                  borderRadius: 0,
                }
              : null,
          ]}
        >
          <View
            style={[
              styles.webviewFrame,
              compact
                ? {
                    flex: 1,
                    minHeight: compactViewportHeight,
                  }
                : null,
            ]}
          >
            <WebView
              ref={webRef}
              originWhitelist={["*"]}
              source={source}
              onMessage={onWebMessage}
              onLoadStart={onWebLoadStart}
              onLoadEnd={onWebLoadEnd}
              onError={onWebError}
              onHttpError={onWebHttpError}
              javaScriptEnabled
              allowFileAccess
              allowUniversalAccessFromFileURLs
              mixedContentMode="always"
              style={[
                isLandscape ? styles.landscapeWebview : styles.webview,
                compact
                  ? {
                      flex: 1,
                      height: undefined,
                    }
                  : null,
              ]}
              nestedScrollEnabled
              scrollEnabled
            />
            {compact && !isLandscape ? (
              <View
                pointerEvents="box-none"
                style={styles.compactPlaybackOverlay}
              >
                <View style={styles.compactPlaybackBar}>
                  <ThemedText style={styles.compactPlaybackBpm}>
                    {playbackBPM} BPM
                  </ThemedText>
                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={[
                      styles.compactPlaybackButton,
                      isPlaying && !isPaused
                        ? styles.compactPlaybackButtonActive
                        : styles.compactPlaybackButtonPrimary,
                      !hasPlayableScore && !isPlaying && !isPaused
                        ? styles.compactPlaybackButtonDisabled
                        : null,
                    ]}
                    onPress={handlePlayPause}
                    disabled={!hasPlayableScore && !isPlaying && !isPaused}
                  >
                    <ThemedText style={styles.compactPlaybackButtonText}>
                      {isPlaying ? (isPaused ? "Resume" : "Pause") : "Play"}
                    </ThemedText>
                  </TouchableOpacity>
                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={[
                      styles.compactPlaybackButton,
                      styles.compactPlaybackButtonSecondary,
                      !isPlaying && !isPaused
                        ? styles.compactPlaybackButtonDisabled
                        : null,
                    ]}
                    onPress={handleStopPlayback}
                    disabled={!isPlaying && !isPaused}
                  >
                    <ThemedText style={styles.compactPlaybackButtonText}>
                      Stop
                    </ThemedText>
                  </TouchableOpacity>
                </View>
              </View>
            ) : null}
          </View>
        </View>

        {compact ? null : (
          <View style={styles.viewControlsSection}>
            <View style={styles.viewControls}>
              <View style={styles.cameraModeGroup}>
                <ThemedText style={styles.controlGroupLabel}>Camera</ThemedText>
                <View style={styles.cameraModeChips}>
                  {(["smooth", "snap"] as CameraMotionMode[]).map((mode) => (
                    <TouchableOpacity
                      key={mode}
                      activeOpacity={0.85}
                      style={[
                        styles.cameraChip,
                        cameraMotionMode === mode && styles.cameraChipActive,
                      ]}
                      onPress={() => setCameraMotionMode(mode)}
                    >
                      <ThemedText
                        style={[
                          styles.cameraChipText,
                          cameraMotionMode === mode &&
                            styles.cameraChipTextActive,
                        ]}
                      >
                        {mode === "smooth" ? "Smooth" : "Reposition"}
                      </ThemedText>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
              <TouchableOpacity
                activeOpacity={0.85}
                style={[
                  styles.controlButton,
                  styles.clearButton,
                  styles.ghostControlButton,
                ]}
                onPress={() => {
                  setAccumulatedNotes([]);
                  setAccumulatedChords([]);
                  lastXmlRef.current = null;
                  measuresSentRef.current = 0;
                  pendingXmlRef.current = null;
                  playAfterRenderRef.current = false;
                  pendingSentAtRef.current = 0;
                  pendingRenderIdRef.current = null;
                  sendRenderXml(
                    FALLBACK_XML,
                    "clear-score",
                    "if (window.__OSMD_STOP) window.__OSMD_STOP();",
                  );
                }}
              >
                <ThemedText style={styles.ghostControlButtonText}>
                  Clear
                </ThemedText>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 10,
    paddingHorizontal: 8,
    backgroundColor: "rgba(255,255,255,0.95)",
    borderRadius: 12,
    marginBottom: 20,
    width: "100%",
    overflow: "hidden",
  },
  compactContainer: {
    flex: 1,
    paddingVertical: 0,
    paddingHorizontal: 0,
    marginBottom: 0,
    backgroundColor: "transparent",
    borderRadius: 0,
  },
  mainContainer: {
    width: "100%",
    maxWidth: "100%",
  },
  compactMainContainer: {
    flex: 1,
    minHeight: 0,
  },
  title: {
    textAlign: "center",
    marginBottom: 12,
    color: "#333",
    fontSize: 18,
    fontWeight: "bold",
  },
  // Playback section
  playbackSection: {
    backgroundColor: "#f8fafc",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#d8e1ea",
  },
  sectionLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#666",
    marginBottom: 8,
    textAlign: "center",
  },
  playbackControls: {
    width: "100%",
    alignItems: "stretch",
    gap: 10,
  },
  playbackButtonRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    width: "100%",
    gap: 10,
  },
  controlButton: {
    minHeight: 46,
    borderRadius: 12,
    borderWidth: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
    maxWidth: "100%",
  },
  playbackButton: {
    flexGrow: 1,
    flexBasis: 0,
    minWidth: 124,
  },
  successControlButton: {
    backgroundColor: "#1f8f5f",
    borderColor: "#1f8f5f",
  },
  warningControlButton: {
    backgroundColor: "#c56a1f",
    borderColor: "#c56a1f",
  },
  dangerControlButton: {
    backgroundColor: "#c24135",
    borderColor: "#c24135",
  },
  ghostControlButton: {
    backgroundColor: "#ffffff",
    borderColor: "#cbd5e1",
  },
  controlButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#ffffff",
  },
  ghostControlButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#52606d",
  },
  tempoControl: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 6,
    borderWidth: 1,
    borderColor: "#d8e1ea",
    gap: 8,
  },
  stepperButton: {
    width: 44,
    minHeight: 44,
    borderRadius: 10,
    backgroundColor: "#e8eef5",
    justifyContent: "center",
    alignItems: "center",
    flexShrink: 0,
  },
  stepperButtonText: {
    fontSize: 22,
    lineHeight: 24,
    fontWeight: "700",
    color: "#334155",
  },
  bpmDisplay: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 8,
  },
  bpmValue: {
    fontSize: 19,
    fontWeight: "bold",
    color: "#1f2937",
  },
  bpmLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  scoreSection: {
    flex: 1,
    minHeight: 560,
    backgroundColor: "#fff",
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#d8e1ea",
  },
  webviewFrame: {
    width: "100%",
    minHeight: 560,
    maxWidth: "100%",
    overflow: "hidden",
  },
  compactPlaybackOverlay: {
    position: "absolute",
    top: 12,
    right: 12,
    zIndex: 3,
    elevation: 6,
  },
  compactPlaybackBar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: "rgba(15,23,42,0.78)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.16)",
  },
  compactPlaybackBpm: {
    fontSize: 10,
    fontWeight: "700",
    color: "rgba(226,232,240,0.88)",
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  compactPlaybackButton: {
    minHeight: 32,
    borderRadius: 999,
    paddingHorizontal: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  compactPlaybackButtonPrimary: {
    backgroundColor: "#0f766e",
  },
  compactPlaybackButtonActive: {
    backgroundColor: "#c56a1f",
  },
  compactPlaybackButtonSecondary: {
    backgroundColor: "rgba(255,255,255,0.16)",
  },
  compactPlaybackButtonDisabled: {
    opacity: 0.48,
  },
  compactPlaybackButtonText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#ffffff",
  },
  debugSection: {
    backgroundColor: "#f7f9fb",
    borderRadius: 10,
    padding: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#d7dde3",
  },
  debugHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  debugTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#37474f",
  },
  debugButtonRow: {
    minWidth: 72,
  },
  debugSummary: {
    fontSize: 11,
    color: "#455a64",
    marginBottom: 2,
  },
  debugLog: {
    marginTop: 6,
    borderTopWidth: 1,
    borderTopColor: "#e0e6eb",
    paddingTop: 6,
  },
  debugLogLine: {
    fontSize: 11,
    color: "#607d8b",
    marginBottom: 2,
  },
  webview: {
    height: 560,
    borderRadius: 8,
    width: "100%",
    overflow: "hidden",
    backgroundColor: "#fff",
  },
  landscapeWebview: {
    height: 400,
    borderRadius: 0,
    width: "100%",
    overflow: "hidden",
    backgroundColor: "#fff",
  },
  viewControlsSection: {
    backgroundColor: "#f8fafc",
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: "#d8e1ea",
  },
  viewControls: {
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 10,
  },
  cameraModeGroup: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 220,
    maxWidth: "100%",
    gap: 8,
  },
  controlGroupLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#52606d",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  cameraModeChips: {
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 8,
    maxWidth: "100%",
  },
  cameraChip: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    backgroundColor: "#ffffff",
  },
  cameraChipActive: {
    backgroundColor: "#1f2937",
    borderColor: "#1f2937",
  },
  cameraChipText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#475569",
  },
  cameraChipTextActive: {
    color: "#ffffff",
  },
  clearButton: {
    flexGrow: 1,
    flexBasis: 108,
    minWidth: 108,
  },
  // Legacy - keep for compatibility
  toolbar: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
    marginBottom: 8,
  },
  playbackToolbar: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
    paddingVertical: 4,
    backgroundColor: "rgba(0,0,0,0.05)",
    borderRadius: 8,
  },
  bpmText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    minWidth: 40,
    textAlign: "center",
  },
});
