import * as ScreenOrientation from "expo-screen-orientation";
import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import { Button, ScrollView, StyleSheet, View } from "react-native";
import { WebView, WebViewMessageEvent } from "react-native-webview";
import { ThemedText } from "./ThemedText";
import { OSMD_HTML } from "./osmdHTML";

interface NoteResult {
  time_seconds: number;
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
}

// Helper function to convert MIDI note to step and octave
function midiToStepOctave(midi: number): { step: string; octave: number } {
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
  const octave = Math.floor(midi / 12) - 1;
  const step = noteNames[midi % 12];
  return { step, octave };
}

// Function to generate MusicXML from notes
// timeSignature: "4/4" | "3/4" | "6/8" - controls beats per measure
// bpm: Beats per minute for playback tempo marking
function generateMeasureXmls(
  notes: NoteResult[],
  chords: ChordResult[],
  timeSignature: "4/4" | "3/4" | "6/8" = "4/4",
  bpm: number = 120,
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
    // Use floor to ensure triplets don't overflow (3 triplet notes should fit in 2 normal notes' time)
    // E.g., 3 eighth note triplets: floor(0.5 * 2/3 * 8) / 8 = floor(2.67) / 8 = 2/8 = 0.25 per note
    // Total: 0.25 * 3 = 0.75 beats (slightly less than 1 beat, but avoids overflow)
    if (triplet) beats = Math.floor(beats * (2 / 3) * 8) / 8;
    return beats;
  };

  // Helper to get MusicXML duration (divisions=8, or 24 for triplet-friendly)
  // For triplets: duration is 2/3 of normal
  // IMPORTANT: Use floor for triplets to avoid measure overflow (3 notes must fit in time of 2)
  const getNoteDuration = (
    noteType?: string,
    dotted?: boolean,
    triplet?: boolean,
  ): number => {
    let duration = 8;
    switch (noteType) {
      case "whole":
        duration = 32;
        break;
      case "half":
        duration = 16;
        break;
      case "quarter":
        duration = 8;
        break;
      case "eighth":
        duration = 4;
        break;
      case "16th":
        duration = 2;
        break;
      case "32nd":
        duration = 1;
        break;
      default:
        duration = 8;
        break;
    }
    if (dotted) duration = Math.floor(duration * 1.5);
    // Triplet: 3 notes in time of 2
    // Use floor to ensure 3 triplet notes don't exceed 2 normal notes
    // E.g., half note triplet: floor(16 * 2/3) = 10, and 10*3 = 30 < 32 (ok)
    if (triplet) duration = Math.floor((duration * 2) / 3);
    return duration;
  };

  // Helper to split beats into a list of (noteType, duration, dotted) tuples
  // Used for ties that span measure boundaries
  const splitBeatsIntoNoteTypes = (
    beats: number,
  ): { noteType: string; duration: number; beats: number }[] => {
    const result: { noteType: string; duration: number; beats: number }[] = [];
    // Note values from largest to smallest
    const noteValues = [
      { beats: 4, noteType: "whole", duration: 32 },
      { beats: 3, noteType: "half", duration: 24, dotted: true }, // dotted half
      { beats: 2, noteType: "half", duration: 16 },
      { beats: 1.5, noteType: "quarter", duration: 12, dotted: true }, // dotted quarter
      { beats: 1, noteType: "quarter", duration: 8 },
      { beats: 0.75, noteType: "eighth", duration: 6, dotted: true }, // dotted eighth
      { beats: 0.5, noteType: "eighth", duration: 4 },
      { beats: 0.25, noteType: "16th", duration: 2 },
      { beats: 0.125, noteType: "32nd", duration: 1 },
    ];

    let remaining = Math.round(beats * 8) / 8; // Round to 32nd note precision

    while (remaining >= 0.125 - 0.001) {
      let found = false;
      for (const nv of noteValues) {
        if (remaining >= nv.beats - 0.001) {
          result.push({
            noteType: nv.noteType,
            duration: nv.duration,
            beats: nv.beats,
          });
          remaining = Math.round((remaining - nv.beats) * 8) / 8;
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
  ): string => {
    const chordTag = isChord ? "<chord/>" : "";
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

    return `<note>${chordTag}${pitchXml}<duration>${duration}</duration>${tieXml}<voice>${staff}</voice><type>${noteType}</type><staff>${staff}</staff>${notationsXml}</note>`;
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
        return "<ornaments><mordent/></ornaments>";
      case "mordent_lower":
        return "<ornaments><inverted-mordent/></ornaments>";
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
  // Rounds beats to nearest 32nd note (0.125) to avoid floating-point fragmentation
  const generateRestXml = (beats: number, staff: number): string[] => {
    const rests: string[] = [];
    // Round to nearest 32nd note to avoid floating-point issues
    let remaining = Math.round(beats * 8) / 8; // Round to 1/8 beat precision
    const restValues = [
      { beats: 4, type: "whole", duration: 32 },
      { beats: 2, type: "half", duration: 16 },
      { beats: 1, type: "quarter", duration: 8 },
      { beats: 0.5, type: "eighth", duration: 4 },
      { beats: 0.25, type: "16th", duration: 2 },
      { beats: 0.125, type: "32nd", duration: 1 },
    ];
    while (remaining >= 0.125 - 0.001) {
      let found = false;
      for (const rv of restValues) {
        if (remaining >= rv.beats - 0.001) {
          rests.push(
            `<note><rest/><duration>${rv.duration}</duration><type>${rv.type}</type><staff>${staff}</staff><voice>${staff}</voice></note>`,
          );
          remaining = Math.round((remaining - rv.beats) * 8) / 8; // Round after each subtraction
          found = true;
          break;
        }
      }
      if (!found) break;
    }
    return rests;
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
    const { step, octave } = midiToStepOctave(n.midi_note);
    const staff = getStaff(n.midi_note);
    let baseStep = step;
    let alter = 0;
    if (step.includes("#")) {
      baseStep = step[0];
      alter = 1;
    } else if (step.includes("b") || step.includes("♭")) {
      baseStep = step[0];
      alter = -1;
    }
    const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
    const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
    const noteType = n.note_value || "quarter";
    const adjustedNoteType = getAdjustedNoteType(noteType);
    const dotted = n.dotted || false;
    const triplet = n.triplet || false;
    const duration = getNoteDuration(noteType, dotted, triplet);
    const dotXml = dotted ? "<dot/>" : "";
    const chordTag = isChordNote ? "<chord/>" : "";

    // Triplet-specific XML
    const timeModXml = getTimeModification(
      triplet,
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

    return `<note>${chordTag}${pitchXml}<duration>${duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${timeModXml}${dotXml}<staff>${staff}</staff>${notationsXml}</note>`;
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
    triplet?: boolean,
    tripletPosition?: "start" | "middle" | "end",
    actualNotes: number = 3,
    normalNotes: number = 2,
  ): string[] => {
    const adjustedNoteType = getAdjustedNoteType(noteType);
    const duration = getNoteDuration(noteType, dotted, triplet);
    const dotXml = dotted ? "<dot/>" : "";
    const timeModXml = getTimeModification(triplet, actualNotes, normalNotes);

    // Filter to only notes on this staff
    const staffNotes = midiList.filter((m) => getStaff(m) === staff);
    if (staffNotes.length === 0) return [];

    const xmlParts: string[] = [];
    let first = true;
    for (const midi of staffNotes) {
      const { step, octave } = midiToStepOctave(midi);
      let baseStep = step;
      let alter = 0;
      if (step.includes("#")) {
        baseStep = step[0];
        alter = 1;
      } else if (step.includes("b") || step.includes("♭")) {
        baseStep = step[0];
        alter = -1;
      }
      const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
      const pitchInner = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
      const chordTag = first ? "" : "<chord/>";
      // Only first note of chord gets triplet notations
      const tripletNotationsXml = first
        ? getTripletNotations(tripletPosition, actualNotes, normalNotes)
        : "";
      const noteXml = `<note>${chordTag}${pitchInner}<duration>${duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${timeModXml}${dotXml}<staff>${staff}</staff>${tripletNotationsXml}</note>`;
      xmlParts.push(noteXml);
      first = false;
    }
    return xmlParts;
  };

  // Build a timeline of events per staff
  type TimelineEvent = {
    time: number;
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
  // Use a tolerance that allows bass/treble alignment (independent onset detection may differ by ~20ms)
  const TIME_TOLERANCE = 0.025; // 25ms - allows bass/treble to sync despite independent detection
  const notesInChords = new Set<string>();
  for (const c of chords) {
    const time = c.time_seconds ?? 0;
    const midiList = chordToMidiList(c);
    for (const midi of midiList) {
      // Key is rounded time + midi note
      notesInChords.add(`${Math.round(time / TIME_TOLERANCE)}:${midi}`);
    }
  }

  // Process all notes (filter out notes that are already in chords at same time)
  for (const n of notes) {
    const time = n.time_seconds ?? 0;
    const noteKey = `${Math.round(time / TIME_TOLERANCE)}:${n.midi_note}`;

    // Skip notes that are part of a chord at the same time
    if (notesInChords.has(noteKey)) {
      continue;
    }

    const staff = getStaff(n.midi_note);
    // Grace notes have 0 beats - they don't take up time in the measure
    // Grace notes also cannot have triplet markings
    const isGrace = n.ornament === "grace";
    const beats = isGrace
      ? 0
      : getNoteBeats(n.note_value, n.dotted, n.triplet);
    const xml = [noteToXmlWithVoice(n, false)];
    timeline.push({
      time,
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
    const beats = getNoteBeats(noteType, dotted, triplet);

    // Split chord by staff (with triplet info)
    const trebleXml = chordMidiToXml(
      midiList,
      noteType,
      dotted,
      1,
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
      triplet,
      c.triplet_position,
      c.actual_notes,
      c.normal_notes,
    );

    if (trebleXml.length > 0) {
      const trebleMidiNotes = midiList.filter((m) => getStaff(m) === 1);
      timeline.push({
        time,
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

  // Sort timeline by time
  timeline.sort((a, b) => a.time - b.time);

  // Group events by time (events within TIME_TOLERANCE are considered simultaneous)
  type TimeGroup = {
    time: number;
    treble: TimelineEvent[];
    bass: TimelineEvent[];
  };
  const timeGroups: TimeGroup[] = [];

  for (const ev of timeline) {
    let group = timeGroups.find(
      (g) => Math.abs(g.time - ev.time) < TIME_TOLERANCE,
    );
    if (!group) {
      group = { time: ev.time, treble: [], bass: [] };
      timeGroups.push(group);
    }
    if (ev.staff === 1) {
      group.treble.push(ev);
    } else {
      group.bass.push(ev);
    }
  }

  // Sort groups by time
  timeGroups.sort((a, b) => a.time - b.time);

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
  // RHYTHM SMOOTHING: Fill small gaps to ensure natural flow
  // If there's a tiny gap between notes (< 0.25 beats), extend the previous note
  // ============================================================================
  const smoothTimelineGaps = (
    events: TimelineEvent[],
    maxGapBeats: number = 0.5,
  ): void => {
    if (events.length < 2) return;

    // Sort by time first
    events.sort((a, b) => a.time - b.time);

    const quarterDuration = 60 / bpm; // seconds per beat

    for (let i = 0; i < events.length - 1; i++) {
      const curr = events[i];
      const next = events[i + 1];

      // Calculate beat positions
      const currStartBeat = curr.time / quarterDuration;
      const currEndBeat = currStartBeat + curr.beats;
      const nextStartBeat = next.time / quarterDuration;

      const gapBeats = nextStartBeat - currEndBeat;

      // If there's a small positive gap, extend this note
      if (gapBeats > 0.001 && gapBeats <= maxGapBeats) {
        // Extend the beats to fill the gap
        curr.beats = nextStartBeat - currStartBeat;

        // Update the XML duration if possible
        // The XML already has a duration, we'll rely on the measure builder's gap filling
      }
    }
  };

  // Smooth gaps within each time group's staff
  for (const group of timeGroups) {
    smoothTimelineGaps(group.treble);
    smoothTimelineGaps(group.bass);
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
      let result = xml.replace(/<time-modification>.*?<\/time-modification>/g, "");
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

  const quarterDurationSec = 60 / bpm;

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
          const startBeat = tripletStart.time / quarterDurationSec;
          const middleBeat = tripletMiddle.time / quarterDurationSec;
          const endBeat = ev.time / quarterDurationSec;

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

      for (const seg of segments) {
        let first = xml.length === 0;
        for (const midi of tie.midiNotes) {
          const { step, octave } = midiToStepOctave(midi);
          let baseStep = step;
          let alter = 0;
          if (step.includes("#")) {
            baseStep = step[0];
            alter = 1;
          } else if (step.includes("b") || step.includes("♭")) {
            baseStep = step[0];
            alter = -1;
          }
          const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
          const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
          xml.push(
            generateNoteXmlWithTie(
              pitchXml,
              seg.duration,
              seg.noteType,
              1,
              tieType,
              !first,
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

      for (const seg of segments) {
        let first = xml.length === 0;
        for (const midi of tie.midiNotes) {
          const { step, octave } = midiToStepOctave(midi);
          let baseStep = step;
          let alter = 0;
          if (step.includes("#")) {
            baseStep = step[0];
            alter = 1;
          } else if (step.includes("b") || step.includes("♭")) {
            baseStep = step[0];
            alter = -1;
          }
          const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
          const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
          xml.push(
            generateNoteXmlWithTie(
              pitchXml,
              seg.duration,
              seg.noteType,
              2,
              tieType,
              !first,
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
      });
    } else {
      // Event overflows - split with ties
      const beatsThisMeasure = remainingInMeasure;
      const beatsRemaining = ev.beats - beatsThisMeasure;

      if (beatsThisMeasure > 0.001 && ev.midiNotes && ev.midiNotes.length > 0) {
        // Generate tied note XML for the portion that fits
        const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
        const xml: string[] = [];

        for (const seg of segments) {
          let first = xml.length === 0;
          for (const midi of ev.midiNotes) {
            const { step, octave } = midiToStepOctave(midi);
            let baseStep = step;
            let alter = 0;
            if (step.includes("#")) {
              baseStep = step[0];
              alter = 1;
            } else if (step.includes("b") || step.includes("♭")) {
              baseStep = step[0];
              alter = -1;
            }
            const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
            const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
            xml.push(
              generateNoteXmlWithTie(
                pitchXml,
                seg.duration,
                seg.noteType,
                ev.staff,
                "start",
                !first,
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

  for (const group of timeGroups) {
    // Get the max duration for this time group
    const trebleBeats =
      group.treble.length > 0
        ? Math.max(...group.treble.map((e) => e.beats))
        : 0;
    const bassBeats =
      group.bass.length > 0 ? Math.max(...group.bass.map((e) => e.beats)) : 0;
    const maxBeats = Math.max(trebleBeats, bassBeats, 0.125); // At least a 32nd note

    // Check if this event would start a new measure (no room left)
    if (currentBeatPos >= BEATS_PER_MEASURE - 0.001) {
      // Finalize current measure
      if (
        currentMeasure.trebleEvents.length > 0 ||
        currentMeasure.bassEvents.length > 0
      ) {
        measuresData.push(currentMeasure);
      }
      currentMeasure = { trebleEvents: [], bassEvents: [] };
      currentBeatPos = 0;

      // Add any pending ties to the new measure
      addPendingTiesToMeasure();
    }

    // Add treble events (with overflow handling)
    for (const ev of group.treble) {
      addEventToMeasure(ev, currentMeasure.trebleEvents, pendingTrebleTies);
    }

    // Add bass events (with overflow handling)
    for (const ev of group.bass) {
      addEventToMeasure(ev, currentMeasure.bassEvents, pendingBassTies);
    }

    // Advance beat position by the maximum event duration (capped at measure boundary)
    currentBeatPos = Math.min(currentBeatPos + maxBeats, BEATS_PER_MEASURE);

    // Check if measure is exactly full
    if (Math.abs(currentBeatPos - BEATS_PER_MEASURE) < 0.001) {
      measuresData.push(currentMeasure);
      currentMeasure = { trebleEvents: [], bassEvents: [] };
      currentBeatPos = 0;

      // Add any pending ties to the new measure
      addPendingTiesToMeasure();
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
      measureContent += `<attributes><divisions>8</divisions><key><fifths>0</fifths></key><time><beats>${beats}</beats><beat-type>${beatType}</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>`;
      // Add tempo marking for playback
      measureContent += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`;
    }

    // Sort events by beat position
    mData.trebleEvents.sort((a, b) => a.beatPos - b.beatPos);
    mData.bassEvents.sort((a, b) => a.beatPos - b.beatPos);

    // RHYTHM SMOOTHING: For small gaps, extend the previous note instead of adding rests
    // Only add rests for gaps >= one full beat (quarter note or larger)
    const MIN_REST_GAP = 0.99; // Only add rests for gaps >= one beat (quarter note)

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
        const mergedXml: string[] = [];
        const mergedMidi: number[] = [];
        const seenMidi = new Set<number>();
        let first = true;
        for (const groupEv of group) {
          for (let xi = 0; xi < groupEv.xml.length; xi++) {
            const xml = groupEv.xml[xi];
            const midi =
              xi < groupEv.midiNotes.length
                ? groupEv.midiNotes[xi]
                : midiFromXml(xml);

            // Skip duplicate MIDI notes at the same beat position
            if (midi !== null && seenMidi.has(midi)) continue;
            if (midi !== null) seenMidi.add(midi);

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
        });
      };

      for (let i = 1; i < events.length; i++) {
        const ev = events[i];
        if (Math.abs(ev.beatPos - currentGroup[0].beatPos) < 0.001) {
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
    const clampEventsToMeasure = (
      events: MeasureEventData[],
    ): MeasureEventData[] => {
      if (events.length === 0) return events;

      const result: MeasureEventData[] = [];
      let currentBeatEnd = 0; // Track where the last note ended

      for (const ev of events) {
        // Start position is either the original beatPos or where the last note ended
        // (use max to not overlap if original position is behind current end)
        const startPos = Math.max(ev.beatPos, currentBeatEnd);

        if (startPos >= BEATS_PER_MEASURE - 0.001) {
          // No room left - skip remaining events
          break;
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
            // Less than a 32nd note - skip
            break;
          }
          const segments = splitBeatsIntoNoteTypes(truncatedBeats);
          const xml: string[] = [];

          for (const seg of segments) {
            let first = xml.length === 0;
            for (const midi of ev.midiNotes) {
              const { step, octave } = midiToStepOctave(midi);
              let baseStep = step;
              let alter = 0;
              if (step.includes("#")) {
                baseStep = step[0];
                alter = 1;
              } else if (step.includes("b") || step.includes("♭")) {
                baseStep = step[0];
                alter = -1;
              }
              const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
              const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
              xml.push(
                generateNoteXmlWithTie(
                  pitchXml,
                  seg.duration,
                  seg.noteType,
                  ev.staff,
                  ev.xml.some((x) => x.includes('tie type="stop"'))
                    ? "continue"
                    : "start",
                  !first,
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
          });
          currentBeatEnd = BEATS_PER_MEASURE;
        }
      }

      return result;
    };

    // Apply consolidation and clamping to treble events
    const consolidatedTreble = consolidateEvents([...mData.trebleEvents]);
    const clampedTreble = clampEventsToMeasure(consolidatedTreble);

    // Write ALL treble events first (with rests only for significant gaps)
    let trebleBeatPos = 0;
    for (let evIdx = 0; evIdx < clampedTreble.length; evIdx++) {
      const ev = clampedTreble[evIdx];
      const gap = ev.beatPos - trebleBeatPos;

      if (gap > MIN_REST_GAP) {
        // Significant gap - add a rest
        measureContent += generateRestXml(gap, 1).join("");
        trebleBeatPos = ev.beatPos;
      } else if (gap > 0.001 && evIdx > 0) {
        // Small gap - we already extend the previous note's duration in XML
        // Just update our position tracking
        trebleBeatPos = ev.beatPos;
      }

      // Add the notes (chord tags already added during merging phase)
      measureContent += ev.xml.join("");
      trebleBeatPos += ev.beats;
    }

    // Pad treble to fill measure
    if (trebleBeatPos < BEATS_PER_MEASURE - 0.001) {
      measureContent += generateRestXml(
        BEATS_PER_MEASURE - trebleBeatPos,
        1,
      ).join("");
      trebleBeatPos = BEATS_PER_MEASURE;
    }

    // Backup to start of measure for bass staff
    const backupDuration = Math.round(trebleBeatPos * 8);
    if (backupDuration > 0) {
      measureContent += `<backup><duration>${backupDuration}</duration></backup>`;
    }

    // Apply consolidation and clamping to bass events
    const consolidatedBass = consolidateEvents([...mData.bassEvents]);
    const clampedBass = clampEventsToMeasure(consolidatedBass);

    // Write ALL bass events (with rests only for significant gaps)
    let bassBeatPos = 0;
    for (let evIdx = 0; evIdx < clampedBass.length; evIdx++) {
      const ev = clampedBass[evIdx];
      const gap = ev.beatPos - bassBeatPos;

      if (gap > MIN_REST_GAP) {
        // Significant gap - add a rest
        measureContent += generateRestXml(gap, 2).join("");
        bassBeatPos = ev.beatPos;
      } else if (gap > 0.001 && evIdx > 0) {
        // Small gap - just update position tracking
        bassBeatPos = ev.beatPos;
      }

      // Add the notes (chord tags already added during merging phase)
      measureContent += ev.xml.join("");
      bassBeatPos += ev.beats;
    }

    // Pad bass to fill measure
    if (bassBeatPos < BEATS_PER_MEASURE - 0.001) {
      measureContent += generateRestXml(
        BEATS_PER_MEASURE - bassBeatPos,
        2,
      ).join("");
    }

    measures.push(
      `<measure number="${measureNum}">${measureContent}</measure>`,
    );
  }

  return measures;
}

function generateMusicXML(
  notes: NoteResult[],
  chords: ChordResult[],
  timeSignature: "4/4" | "3/4" | "6/8" = "4/4",
  bpm: number = 120,
): string {
  const measures = generateMeasureXmls(notes, chords, timeSignature, bpm);
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n<score-partwise version="3.1">\n  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n  <part id="P1">${measures.join("")}</part></score-partwise>`;
  console.log(xml);
  return xml;
}

// A small visible fallback score (one measure with four quarter notes) used when no notes are available
const FALLBACK_XML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>8</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><type>quarter</type><staff>1</staff></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>8</duration><type>quarter</type><staff>1</staff></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><type>quarter</type><staff>1</staff></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>8</duration><type>quarter</type><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>`;

// Drop-in replacement component:
export default function PianoSheetMusic({
  results,
  timeSignature = "4/4",
}: PianoSheetMusicProps) {
  // accumulate incoming note blocks so live updates append to the end
  const [accumulatedNotes, setAccumulatedNotes] = useState<NoteResult[]>([]);
  const [accumulatedChords, setAccumulatedChords] = useState<ChordResult[]>([]);
  const hasReceivedDataRef = useRef<boolean>(false);

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
  }, [results]);

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
    );
  }, [accumulatedNotes, accumulatedChords, timeSignature, detectedBPM]);

  const webRef = useRef<WebView>(null);
  const source = useMemo(() => ({ html: OSMD_HTML }), []);
  const measuresSentRef = useRef<number>(0);
  const lastXmlRef = useRef<string | null>(null);
  const pendingXmlRef = useRef<string | null>(null);

  const post = useCallback((obj: any) => {
    webRef.current?.postMessage(JSON.stringify(obj));
  }, []);

  const [isLandscape, setIsLandscape] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  // Initialize BPM from detected tempo, default to 120
  const [playbackBPM, setPlaybackBPM] = useState(detectedBPM);
  const lastDetectedBPMRef = useRef<number | undefined>(undefined);
  const webViewReadyRef = useRef<boolean>(false);

  // Update BPM when new results come in with detected tempo (only if WebView is ready)
  useEffect(() => {
    if (detectedBPM && detectedBPM !== lastDetectedBPMRef.current) {
      lastDetectedBPMRef.current = detectedBPM;
      setPlaybackBPM(detectedBPM);
      // Only send to WebView if it's ready
      if (webViewReadyRef.current) {
        post({ type: "setBPM", bpm: detectedBPM });
      }
    }
  }, [detectedBPM, post]);

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

        if (msg.type === "ready") {
          // Mark WebView as ready
          webViewReadyRef.current = true;
          // init options are already set in the HTML; you can send more here if needed
          // post initial render and mark it as pending so we can sync cache on 'rendered'
          pendingXmlRef.current = score;
          post({ type: "renderXml", xml: score });
          post({ type: "toggleCursor", show: true }); // show follow-along cursor (static until you drive it)
          // Always set the BPM from detected tempo (playbackBPM has the current value)
          post({ type: "setBPM", bpm: playbackBPM });
          console.log(
            "[PianoSheetMusic] WebView ready, setting BPM to:",
            playbackBPM,
          );
        }
        if (msg.type === "rendered") {
          // initial main render completed; mark how many measures are present
          if (typeof msg.measures === "number")
            measuresSentRef.current = msg.measures;
          // The webview has finished rendering the previously-sent xml; promote pending -> last
          if (pendingXmlRef.current) {
            lastXmlRef.current = pendingXmlRef.current;
            pendingXmlRef.current = null;
          }
        }
        if (msg.type === "appended") {
          // appended measures ack - increment our counter if provided
          if (typeof msg.appended === "number")
            measuresSentRef.current += msg.appended;
        }
        if (msg.type === "error") {
          console.warn("OSMD error:", msg.error);
        }

        // Playback events
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
          setIsPlaying(false);
          setIsPaused(false);
        }
        if (msg.type === "bpmSet") {
          setPlaybackBPM(msg.bpm);
        }
        // Handle exit fullscreen from WebView controls
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
        // Handle BPM changes from WebView controls
        if (msg.type === "bpmChanged") {
          setPlaybackBPM(msg.bpm);
        }
      } catch (err) {
        console.warn("webview message parse error", err);
      }
    },
    [isLandscape, score, post, playbackBPM],
  );

  // If the score changes (for example after live recording produces results), send the new XML
  // to the WebView so OSMD re-renders the updated score.
  useEffect(() => {
    // small guard: only post if webview ref exists
    if (!webRef.current) return;

    const measures = generateMeasureXmls(
      accumulatedNotes,
      accumulatedChords,
      timeSignature,
    );

    // If we have never sent anything, send the full score and save it
    if (measuresSentRef.current === 0 || !lastXmlRef.current) {
      try {
        // mark this xml as pending and send it; we'll update lastXmlRef when the webview posts 'rendered'
        pendingXmlRef.current = score;
        post({ type: "renderXml", xml: score });
        post({ type: "toggleCursor", show: true });
        // don't set lastXmlRef yet; wait for 'rendered' to confirm
        // still set optimistic measures count in case the webview doesn't reply immediately
        measuresSentRef.current = measures.length;
      } catch (e) {
        console.warn("renderXml post failed", e);
      }
      return;
    }

    // Otherwise, if there are new measures, compose a combined XML by inserting
    // the new measures into the previously-sent XML, then post the combined XML.
    if (measures.length > measuresSentRef.current) {
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
        // mark pending and send combined xml; wait for 'rendered' to update lastXmlRef
        pendingXmlRef.current = newXml;
        post({ type: "renderXml", xml: newXml });
        measuresSentRef.current = measures.length; // optimistic
      } catch (e) {
        console.warn("Failed posting appended renderXml", e);
      }
    }
  }, [accumulatedNotes, accumulatedChords, post, score, timeSignature]);

  // unlock orientation on unmount to avoid locking the device permanently
  useEffect(() => {
    return () => {
      ScreenOrientation.unlockAsync().catch(() => {});
    };
  }, []);

  // Notify WebView when fullscreen mode changes
  useEffect(() => {
    post({ type: "setFullscreenMode", enabled: isLandscape });
  }, [isLandscape, post]);

  return (
    <ScrollView
      style={styles.container}
      horizontal
      scrollEnabled={!isLandscape}
    >
      <View style={styles.mainContainer}>
        {/* Playback Controls - Clean and Minimal */}
        <View style={styles.playbackSection}>
          <View style={styles.playbackControls}>
            <View style={styles.playButtonContainer}>
              <Button
                title={
                  isPlaying ? (isPaused ? "▶ Resume" : "⏸ Pause") : "▶ Play"
                }
                color={isPlaying && !isPaused ? "#e67e22" : "#27ae60"}
                onPress={() => {
                  if (!isPlaying) {
                    post({ type: "play", bpm: playbackBPM });
                  } else if (isPaused) {
                    post({ type: "play", bpm: playbackBPM });
                  } else {
                    post({ type: "pause" });
                  }
                }}
              />
            </View>
            <View style={styles.stopButtonContainer}>
              <Button
                title="⏹ Stop"
                color="#c0392b"
                onPress={() => post({ type: "stop" })}
              />
            </View>
            <View style={styles.bpmContainer}>
              <Button
                title="−"
                color="#7f8c8d"
                onPress={() => {
                  const newBPM = Math.max(40, playbackBPM - 10);
                  setPlaybackBPM(newBPM);
                  post({ type: "setBPM", bpm: newBPM });
                }}
              />
              <View style={styles.bpmDisplay}>
                <ThemedText style={styles.bpmValue}>{playbackBPM}</ThemedText>
                <ThemedText style={styles.bpmLabel}>BPM</ThemedText>
              </View>
              <Button
                title="+"
                color="#7f8c8d"
                onPress={() => {
                  const newBPM = Math.min(240, playbackBPM + 10);
                  setPlaybackBPM(newBPM);
                  post({ type: "setBPM", bpm: newBPM });
                }}
              />
            </View>
          </View>
        </View>

        {/* Score Display */}
        <View style={styles.scoreSection}>
          <ScrollView scrollEnabled={!isLandscape}>
            <WebView
              ref={webRef}
              originWhitelist={["*"]}
              source={source}
              onMessage={onWebMessage}
              javaScriptEnabled
              allowFileAccess
              allowUniversalAccessFromFileURLs
              mixedContentMode="always"
              style={isLandscape ? styles.landscapeWebview : styles.webview}
              nestedScrollEnabled={true}
              scrollEnabled={true}
            />
          </ScrollView>
        </View>

        {/* Minimal View Controls */}
        <View style={styles.viewControlsSection}>
          <View style={styles.viewControls}>
            <Button
              title="Clear"
              color="#95a5a6"
              onPress={() => {
                setAccumulatedNotes([]);
                setAccumulatedChords([]);
                lastXmlRef.current = null;
                measuresSentRef.current = 0;
                post({ type: "renderXml", xml: FALLBACK_XML });
                post({ type: "stop" });
              }}
            />
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 10,
    backgroundColor: "rgba(255,255,255,0.95)",
    borderRadius: 12,
    marginBottom: 20,
    width: "100%",
  },
  mainContainer: {
    flex: 1,
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
    backgroundColor: "#f8f9fa",
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  sectionLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#666",
    marginBottom: 8,
    textAlign: "center",
  },
  playbackControls: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 12,
  },
  playButtonContainer: {
    minWidth: 100,
  },
  stopButtonContainer: {
    minWidth: 70,
  },
  bpmContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 4,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  bpmDisplay: {
    alignItems: "center",
    paddingHorizontal: 12,
  },
  bpmValue: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#333",
  },
  bpmLabel: {
    fontSize: 10,
    color: "#888",
  },
  // Score section
  scoreSection: {
    flex: 1,
    backgroundColor: "#fff",
    borderRadius: 8,
    overflow: "hidden",
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  webview: {
    height: 350,
    borderRadius: 8,
    width: 600,
    overflow: "hidden",
    backgroundColor: "#fff",
  },
  landscapeWebview: {
    height: 400,
    borderRadius: 0,
    width: 800,
    overflow: "hidden",
    backgroundColor: "#fff",
  },
  // View controls section
  viewControlsSection: {
    backgroundColor: "#f0f0f0",
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  viewControls: {
    flexDirection: "row",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: 8,
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
