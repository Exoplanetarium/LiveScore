import * as ScreenOrientation from 'expo-screen-orientation';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, ScrollView, StyleSheet, View } from 'react-native';
import { WebView, WebViewMessageEvent } from 'react-native-webview';
import { ThemedText } from './ThemedText';
import { OSMD_HTML } from './osmdHTML';

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
  note_value?: 'whole' | 'half' | 'quarter' | 'eighth' | '16th' | '32nd';
  note_divisions?: number;
  dotted?: boolean;
  // Triplet fields
  triplet?: boolean;
  triplet_position?: 'start' | 'middle' | 'end';
  triplet_type?: 'half' | 'quarter' | 'eighth' | '16th' | '32nd';
  actual_notes?: number;  // 3 for triplet
  normal_notes?: number;  // 2 for triplet
}

interface ChordResult {
  time_seconds: number;
  frame_index: number;
  duration_seconds?: number;
  chord_quality: string;
  label: string; 
  confidence: number;
  note_score?: number;
  octave?: number;
  inversion: string;
  offset_frame?: number;
  offset_seconds?: number;
  midi_notes?: number[]; 
  root_midi?: number;
  note_value?: 'whole' | 'half' | 'quarter' | 'eighth' | '16th' | '32nd';
  note_divisions?: number;
  dotted?: boolean;
  // Triplet fields
  triplet?: boolean;
  triplet_position?: 'start' | 'middle' | 'end';
  triplet_type?: 'half' | 'quarter' | 'eighth' | '16th' | '32nd';
  actual_notes?: number;
  normal_notes?: number;
}

interface AnalysisResult {
  onsets: { duration_seconds?: number; frame_index?: number; offset_frame?: number; offset_seconds?: number; time_seconds: number }[];
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
  tempoMultiplier?: number; // 0.5 = double note values, 1 = normal, 2 = half note values
}

// Helper function to convert MIDI note to step and octave
function midiToStepOctave(midi: number): { step: string; octave: number } {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midi / 12) - 1;
  const step = noteNames[midi % 12];
  return { step, octave };
}

// Function to generate MusicXML from notes
// tempoMultiplier: 0.5 = double note values (slower), 1 = normal, 2 = half note values (faster)
// bpm: Beats per minute for playback tempo marking
function generateMeasureXmls(notes: NoteResult[], chords: ChordResult[], tempoMultiplier: number = 1, bpm: number = 120): string[] {
  const measures: string[] = [];
  const BEATS_PER_MEASURE = 4;

  // Note type order for tempo adjustments
  const noteTypeOrder = ['32nd', '16th', 'eighth', 'quarter', 'half', 'whole'];
  
  // Adjust note type based on tempo multiplier
  const adjustNoteType = (noteType?: string): string => {
    if (!noteType || tempoMultiplier === 1) return noteType || 'quarter';
    const currentIndex = noteTypeOrder.indexOf(noteType);
    if (currentIndex === -1) return noteType;
    let newIndex = currentIndex;
    if (tempoMultiplier === 2) {
      newIndex = Math.max(0, currentIndex - 1);
    } else if (tempoMultiplier === 0.5) {
      newIndex = Math.min(noteTypeOrder.length - 1, currentIndex + 1);
    }
    return noteTypeOrder[newIndex];
  };

  // Helper to get beat value for a note type (considers triplets)
  const getNoteBeats = (noteType?: string, dotted?: boolean, triplet?: boolean): number => {
    const adjustedType = adjustNoteType(noteType);
    let beats = 1;
    switch (adjustedType) {
      case 'whole': beats = 4; break;
      case 'half': beats = 2; break;
      case 'quarter': beats = 1; break;
      case 'eighth': beats = 0.5; break;
      case '16th': beats = 0.25; break;
      case '32nd': beats = 0.125; break;
      default: beats = 1; break;
    }
    if (dotted) beats *= 1.5;
    // Triplet: 3 notes in time of 2, so each note is 2/3 of normal
    // Round to avoid floating-point fragmentation (1.333... -> 1.375 or 1.25)
    if (triplet) beats = Math.round(beats * (2/3) * 8) / 8;
    return beats;
  };

  // Helper to get MusicXML duration (divisions=8, or 24 for triplet-friendly)
  // For triplets: duration is 2/3 of normal
  // IMPORTANT: Use floor for triplets to avoid measure overflow (3 notes must fit in time of 2)
  const getNoteDuration = (noteType?: string, dotted?: boolean, triplet?: boolean): number => {
    const adjustedType = adjustNoteType(noteType);
    let duration = 8;
    switch (adjustedType) {
      case 'whole': duration = 32; break;
      case 'half': duration = 16; break;
      case 'quarter': duration = 8; break;
      case 'eighth': duration = 4; break;
      case '16th': duration = 2; break;
      case '32nd': duration = 1; break;
      default: duration = 8; break;
    }
    if (dotted) duration = Math.floor(duration * 1.5);
    // Triplet: 3 notes in time of 2
    // Use floor to ensure 3 triplet notes don't exceed 2 normal notes
    // E.g., half note triplet: floor(16 * 2/3) = 10, and 10*3 = 30 < 32 (ok)
    if (triplet) duration = Math.floor(duration * 2 / 3);
    return duration;
  };
  
  const getAdjustedNoteType = (noteType?: string): string => adjustNoteType(noteType);

  // Generate triplet notation XML elements
  const getTripletNotations = (tripletPosition?: 'start' | 'middle' | 'end', actualNotes: number = 3, normalNotes: number = 2): string => {
    if (!tripletPosition) return '';
    
    if (tripletPosition === 'start') {
      return `<notations><tuplet type="start" bracket="yes" number="1"/></notations>`;
    } else if (tripletPosition === 'end') {
      return `<notations><tuplet type="stop" number="1"/></notations>`;
    }
    // Middle notes don't need tuplet notation
    return '';
  };

  // Generate time-modification XML for triplets
  const getTimeModification = (triplet?: boolean, actualNotes: number = 3, normalNotes: number = 2): string => {
    if (!triplet) return '';
    return `<time-modification><actual-notes>${actualNotes}</actual-notes><normal-notes>${normalNotes}</normal-notes></time-modification>`;
  };

  // Generate rest XML for a given number of beats
  // Rounds beats to nearest 32nd note (0.125) to avoid floating-point fragmentation
  const generateRestXml = (beats: number, staff: number): string[] => {
    const rests: string[] = [];
    // Round to nearest 32nd note to avoid floating-point issues
    let remaining = Math.round(beats * 8) / 8; // Round to 1/8 beat precision
    const restValues = [
      { beats: 4, type: 'whole', duration: 32 },
      { beats: 2, type: 'half', duration: 16 },
      { beats: 1, type: 'quarter', duration: 8 },
      { beats: 0.5, type: 'eighth', duration: 4 },
      { beats: 0.25, type: '16th', duration: 2 },
      { beats: 0.125, type: '32nd', duration: 1 },
    ];
    while (remaining >= 0.125 - 0.001) {
      let found = false;
      for (const rv of restValues) {
        if (remaining >= rv.beats - 0.001) {
          rests.push(`<note><rest/><duration>${rv.duration}</duration><type>${rv.type}</type><staff>${staff}</staff><voice>${staff}</voice></note>`);
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
  const noteToXmlWithVoice = (n: NoteResult, isChordNote: boolean = false): string => {
    const { step, octave } = midiToStepOctave(n.midi_note);
    const staff = getStaff(n.midi_note);
    let baseStep = step;
    let alter = 0;
    if (step.includes('#')) { baseStep = step[0]; alter = 1; }
    else if (step.includes('b') || step.includes('♭')) { baseStep = step[0]; alter = -1; }
    const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : '';
    const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
    const noteType = n.note_value || 'quarter';
    const adjustedNoteType = getAdjustedNoteType(noteType);
    const dotted = n.dotted || false;
    const triplet = n.triplet || false;
    const duration = getNoteDuration(noteType, dotted, triplet);
    const dotXml = dotted ? '<dot/>' : '';
    const chordTag = isChordNote ? '<chord/>' : '';
    
    // Triplet-specific XML
    const timeModXml = getTimeModification(triplet, n.actual_notes || 3, n.normal_notes || 2);
    const tripletNotationsXml = getTripletNotations(n.triplet_position, n.actual_notes || 3, n.normal_notes || 2);
    
    return `<note>${chordTag}${pitchXml}<duration>${duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${timeModXml}${dotXml}<staff>${staff}</staff>${tripletNotationsXml}</note>`;
  };

  // Helper to construct chord note MIDI list from ChordResult
  const chordToMidiList = (c: ChordResult): number[] => {
    if (c.midi_notes && c.midi_notes.length) return c.midi_notes.slice();
    const nameToSemitone: Record<string, number> = {
      C: 0, 'C#': 1, DB: 1, D: 2, 'D#': 3, EB: 3, E: 4, FB: 4,
      F: 5, 'F#': 6, GB: 6, G: 7, 'G#': 8, AB: 8, A: 9, 'A#': 10, BB: 10, B: 11, CB: 11,
    };
    const buildFromRoot = (root: number, quality?: string): number[] => {
      const q = (quality || '').toLowerCase();
      if (q.includes('maj7') || q === 'maj7') return [root, root + 4, root + 7, root + 11];
      if (q === '7' || q.includes('dom7') || q === 'dom') return [root, root + 4, root + 7, root + 10];
      if (q === 'm7' || q === 'min7') return [root, root + 3, root + 7, root + 10];
      if (q === 'm' || q === 'min') return [root, root + 3, root + 7];
      if (q === 'dim') return [root, root + 3, root + 6];
      if (q === 'aug') return [root, root + 4, root + 8];
      if (q === 'sus2') return [root, root + 2, root + 7];
      if (q === 'sus4') return [root, root + 5, root + 7];
      return [root, root + 4, root + 7];
    };
    if (typeof c.root_midi === 'number') return buildFromRoot(c.root_midi, c.chord_quality);
    if (c.label) {
      const m = String(c.label).toUpperCase().match(/^([A-G][#B]?)/);
      if (m) {
        const rootName = m[1].replace('B', 'B');
        const semitone = nameToSemitone[rootName] ?? 0;
        const octave = typeof c.octave === 'number' ? c.octave : 4;
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
    tripletPosition?: 'start' | 'middle' | 'end',
    actualNotes: number = 3,
    normalNotes: number = 2
  ): string[] => {
    const adjustedNoteType = getAdjustedNoteType(noteType);
    const duration = getNoteDuration(noteType, dotted, triplet);
    const dotXml = dotted ? '<dot/>' : '';
    const timeModXml = getTimeModification(triplet, actualNotes, normalNotes);
    
    // Filter to only notes on this staff
    const staffNotes = midiList.filter(m => getStaff(m) === staff);
    if (staffNotes.length === 0) return [];
    
    const xmlParts: string[] = [];
    let first = true;
    for (const midi of staffNotes) {
      const { step, octave } = midiToStepOctave(midi);
      let baseStep = step;
      let alter = 0;
      if (step.includes('#')) { baseStep = step[0]; alter = 1; }
      else if (step.includes('b') || step.includes('♭')) { baseStep = step[0]; alter = -1; }
      const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : '';
      const pitchInner = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
      const chordTag = first ? '' : '<chord/>';
      // Only first note of chord gets triplet notations
      const tripletNotationsXml = first ? getTripletNotations(tripletPosition, actualNotes, normalNotes) : '';
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
    // Triplet metadata for cross-measure handling
    triplet?: boolean;
    tripletPosition?: 'start' | 'middle' | 'end';
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
    const beats = getNoteBeats(n.note_value, n.dotted, n.triplet);
    const xml = [noteToXmlWithVoice(n, false)];
    timeline.push({ 
      time, 
      staff, 
      beats, 
      xml,
      triplet: n.triplet,
      tripletPosition: n.triplet_position,
      tripletType: n.triplet_type || n.note_value,
      actualNotes: n.actual_notes,
      normalNotes: n.normal_notes
    });
  }

  // Process all chords - split across staves if needed
  for (const c of chords) {
    const time = c.time_seconds ?? 0;
    let midiList = chordToMidiList(c);
    if (midiList.length === 0) continue;

    // Apply inversion
    const inversionToIndex = (inv: any, chordLen: number) => {
      if (typeof inv === 'number') return Math.max(0, Math.floor(inv));
      if (!inv) return 0;
      const s = String(inv).toLowerCase();
      if (s === 'root') return 0;
      if (s === 'first') return 1;
      if (s === 'second') return 2;
      if (s === 'third') return 3;
      if (s === 'slash') return (chordLen >= 4 ? 3 : 1);
      return 0;
    };
    const inversion = inversionToIndex(c.inversion, midiList.length);
    for (let i = 0; i < inversion; i++) {
      const n = midiList.shift();
      if (typeof n === 'number') midiList.push(n + 12);
    }

    const noteType = c.note_value || 'quarter';
    const dotted = c.dotted || false;
    const triplet = c.triplet || false;
    const beats = getNoteBeats(noteType, dotted, triplet);

    // Split chord by staff (with triplet info)
    const trebleXml = chordMidiToXml(midiList, noteType, dotted, 1, triplet, c.triplet_position, c.actual_notes, c.normal_notes);
    const bassXml = chordMidiToXml(midiList, noteType, dotted, 2, triplet, c.triplet_position, c.actual_notes, c.normal_notes);

    if (trebleXml.length > 0) {
      timeline.push({ 
        time, 
        staff: 1, 
        beats, 
        xml: trebleXml,
        triplet,
        tripletPosition: c.triplet_position,
        tripletType: c.triplet_type || noteType,
        actualNotes: c.actual_notes,
        normalNotes: c.normal_notes
      });
    }
    if (bassXml.length > 0) {
      timeline.push({ 
        time, 
        staff: 2, 
        beats, 
        xml: bassXml,
        triplet,
        tripletPosition: c.triplet_position,
        tripletType: c.triplet_type || noteType,
        actualNotes: c.actual_notes,
        normalNotes: c.normal_notes
      });
    }
  }

  // Sort timeline by time
  timeline.sort((a, b) => a.time - b.time);

  // Group events by time (events within TIME_TOLERANCE are considered simultaneous)
  type TimeGroup = { time: number; treble: TimelineEvent[]; bass: TimelineEvent[] };
  const timeGroups: TimeGroup[] = [];

  for (const ev of timeline) {
    let group = timeGroups.find(g => Math.abs(g.time - ev.time) < TIME_TOLERANCE);
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

  // MERGE simultaneous events on the same staff into chords
  // Use the same tolerance as TIME_TOLERANCE to ensure grouped notes become chords
  const CHORD_MERGE_TOLERANCE = TIME_TOLERANCE; // Match grouping tolerance
  
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
          let maxBeats = 0;
          let first = true;
          for (const ev of sg) {
            for (const xml of ev.xml) {
              if (first) {
                mergedXml.push(xml);
                first = false;
              } else {
                const chordXml = xml.replace('<note>', '<note><chord/>');
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
            triplet: sg[0].triplet,
            tripletPosition: sg[0].tripletPosition,
            tripletType: sg[0].tripletType,
            actualNotes: sg[0].actualNotes,
            normalNotes: sg[0].normalNotes
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
          let maxBeats = 0;
          let first = true;
          for (const ev of sg) {
            for (const xml of ev.xml) {
              if (first) {
                mergedXml.push(xml);
                first = false;
              } else {
                const chordXml = xml.replace('<note>', '<note><chord/>');
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
            triplet: sg[0].triplet,
            tripletPosition: sg[0].tripletPosition,
            tripletType: sg[0].tripletType,
            actualNotes: sg[0].actualNotes,
            normalNotes: sg[0].normalNotes
          });
        }
      }
      group.bass = newBass;
    }
  }

  // ============================================================================
  // TRIPLET VALIDATION: Ensure complete triplets (3 notes) stay together
  // If a triplet would span measures, strip the triplet markers
  // ============================================================================
  
  // Helper to strip triplet markers from XML strings
  const stripTripletFromXml = (xmlArr: string[]): string[] => {
    return xmlArr.map(xml => 
      xml
        .replace(/<time-modification>.*?<\/time-modification>/g, '')
        .replace(/<notations><tuplet[^>]*\/><\/notations>/g, '')
        .replace(/<notations><tuplet[^>]*><\/notations>/g, '')
    );
  };

  // Track triplet groups per staff to validate completeness
  // A valid triplet must have: start -> middle -> end (exactly 3 notes)
  for (const staff of [1, 2] as const) {
    const events = staff === 1 
      ? timeGroups.flatMap(g => g.treble)
      : timeGroups.flatMap(g => g.bass);
    
    let tripletStart: TimelineEvent | null = null;
    let tripletMiddle: TimelineEvent | null = null;
    
    for (const ev of events) {
      if (ev.tripletPosition === 'start') {
        // Start of a new triplet
        tripletStart = ev;
        tripletMiddle = null;
      } else if (ev.tripletPosition === 'middle' && tripletStart) {
        tripletMiddle = ev;
      } else if (ev.tripletPosition === 'end') {
        if (tripletStart && tripletMiddle) {
          // Valid triplet - all 3 notes present, keep the markers
          // The triplet is complete and will be rendered correctly
        } else {
          // Incomplete triplet - strip markers
          if (tripletStart) {
            tripletStart.xml = stripTripletFromXml(tripletStart.xml);
            tripletStart.triplet = false;
            tripletStart.tripletPosition = undefined;
          }
          if (tripletMiddle) {
            tripletMiddle.xml = stripTripletFromXml(tripletMiddle.xml);
            tripletMiddle.triplet = false;
            tripletMiddle.tripletPosition = undefined;
          }
          ev.xml = stripTripletFromXml(ev.xml);
          ev.triplet = false;
          ev.tripletPosition = undefined;
        }
        tripletStart = null;
        tripletMiddle = null;
      }
    }
    
    // Handle orphaned triplet starts/middles at the end
    if (tripletStart) {
      tripletStart.xml = stripTripletFromXml(tripletStart.xml);
      tripletStart.triplet = false;
      tripletStart.tripletPosition = undefined;
    }
    if (tripletMiddle) {
      tripletMiddle.xml = stripTripletFromXml(tripletMiddle.xml);
      tripletMiddle.triplet = false;
      tripletMiddle.tripletPosition = undefined;
    }
  }

  // ============================================================================
  // NEW APPROACH: Build measures by writing ALL treble first, then backup, then ALL bass
  // This ensures treble and bass play simultaneously (not sequentially)
  // ============================================================================
  
  // Group time groups by measure
  type MeasureData = {
    trebleEvents: { beatPos: number; xml: string[]; beats: number }[];
    bassEvents: { beatPos: number; xml: string[]; beats: number }[];
  };
  
  const measuresData: MeasureData[] = [];
  let currentMeasure: MeasureData = { trebleEvents: [], bassEvents: [] };
  let currentBeatPos = 0;
  
  // Track active triplets per staff to detect splits
  let activeTrebleTriplet: { events: typeof currentMeasure.trebleEvents; measureIdx: number } | null = null;
  let activeBassTriplet: { events: typeof currentMeasure.bassEvents; measureIdx: number } | null = null;
  
  for (const group of timeGroups) {
    // Get the max duration for this time group
    const trebleBeats = group.treble.length > 0 ? Math.max(...group.treble.map(e => e.beats)) : 0;
    const bassBeats = group.bass.length > 0 ? Math.max(...group.bass.map(e => e.beats)) : 0;
    const maxBeats = Math.max(trebleBeats, bassBeats, 0.125); // At least a 32nd note
    
    // Check if this event would overflow the measure
    if (currentBeatPos + maxBeats > BEATS_PER_MEASURE + 0.001) {
      // Finalize current measure
      if (currentMeasure.trebleEvents.length > 0 || currentMeasure.bassEvents.length > 0) {
        measuresData.push(currentMeasure);
      }
      currentMeasure = { trebleEvents: [], bassEvents: [] };
      currentBeatPos = 0;
      
      // If we have active triplets that weren't completed, they span measures - strip markers
      if (activeTrebleTriplet) {
        for (const ev of activeTrebleTriplet.events) {
          // Modify the xml array in place
          const stripped = stripTripletFromXml(ev.xml);
          ev.xml.length = 0;
          ev.xml.push(...stripped);
        }
        activeTrebleTriplet = null;
      }
      if (activeBassTriplet) {
        for (const ev of activeBassTriplet.events) {
          // Modify the xml array in place
          const stripped = stripTripletFromXml(ev.xml);
          ev.xml.length = 0;
          ev.xml.push(...stripped);
        }
        activeBassTriplet = null;
      }
    }
    
    // Add treble events and track triplets
    for (const ev of group.treble) {
      const evData = {
        beatPos: currentBeatPos,
        xml: ev.xml,
        beats: ev.beats
      };
      currentMeasure.trebleEvents.push(evData);
      
      // Track triplet state
      if (ev.tripletPosition === 'start') {
        activeTrebleTriplet = { events: [evData], measureIdx: measuresData.length };
      } else if (ev.tripletPosition === 'middle' && activeTrebleTriplet) {
        activeTrebleTriplet.events.push(evData);
      } else if (ev.tripletPosition === 'end') {
        activeTrebleTriplet = null; // Triplet completed successfully
      }
    }
    
    // Add bass events and track triplets
    for (const ev of group.bass) {
      const evData = {
        beatPos: currentBeatPos,
        xml: ev.xml,
        beats: ev.beats
      };
      currentMeasure.bassEvents.push(evData);
      
      // Track triplet state
      if (ev.tripletPosition === 'start') {
        activeBassTriplet = { events: [evData], measureIdx: measuresData.length };
      } else if (ev.tripletPosition === 'middle' && activeBassTriplet) {
        activeBassTriplet.events.push(evData);
      } else if (ev.tripletPosition === 'end') {
        activeBassTriplet = null; // Triplet completed successfully
      }
    }
    
    currentBeatPos += maxBeats;
    
    // Check if measure is exactly full
    if (Math.abs(currentBeatPos - BEATS_PER_MEASURE) < 0.001) {
      measuresData.push(currentMeasure);
      currentMeasure = { trebleEvents: [], bassEvents: [] };
      currentBeatPos = 0;
    }
  }
  
  // Push final measure if it has content
  if (currentMeasure.trebleEvents.length > 0 || currentMeasure.bassEvents.length > 0) {
    measuresData.push(currentMeasure);
  }
  
  // Now generate XML for each measure
  for (let mIdx = 0; mIdx < measuresData.length; mIdx++) {
    const mData = measuresData[mIdx];
    const measureNum = mIdx + 1;
    let measureContent = '';
    
    // Attributes only for first measure
    if (measureNum === 1) {
      measureContent += '<attributes><divisions>8</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>';
      // Add tempo marking for playback
      measureContent += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`;
    }
    
    // Sort events by beat position
    mData.trebleEvents.sort((a, b) => a.beatPos - b.beatPos);
    mData.bassEvents.sort((a, b) => a.beatPos - b.beatPos);
    
    // Write ALL treble events first (with rests to fill gaps)
    let trebleBeatPos = 0;
    for (const ev of mData.trebleEvents) {
      // Add rest if there's a gap
      if (ev.beatPos > trebleBeatPos + 0.001) {
        const restBeats = ev.beatPos - trebleBeatPos;
        measureContent += generateRestXml(restBeats, 1).join('');
        trebleBeatPos = ev.beatPos;
      }
      // Add the notes (chord tags already added during merging phase)
      measureContent += ev.xml.join('');
      trebleBeatPos += ev.beats;
    }
    
    // Pad treble to fill measure
    if (trebleBeatPos < BEATS_PER_MEASURE - 0.001) {
      measureContent += generateRestXml(BEATS_PER_MEASURE - trebleBeatPos, 1).join('');
      trebleBeatPos = BEATS_PER_MEASURE;
    }
    
    // Backup to start of measure for bass staff
    const backupDuration = Math.round(trebleBeatPos * 8);
    if (backupDuration > 0) {
      measureContent += `<backup><duration>${backupDuration}</duration></backup>`;
    }
    
    // Write ALL bass events (with rests to fill gaps)
    let bassBeatPos = 0;
    for (const ev of mData.bassEvents) {
      // Add rest if there's a gap
      if (ev.beatPos > bassBeatPos + 0.001) {
        const restBeats = ev.beatPos - bassBeatPos;
        measureContent += generateRestXml(restBeats, 2).join('');
        bassBeatPos = ev.beatPos;
      }
      // Add the notes (chord tags already added during merging phase)
      measureContent += ev.xml.join('');
      bassBeatPos += ev.beats;
    }
    
    // Pad bass to fill measure
    if (bassBeatPos < BEATS_PER_MEASURE - 0.001) {
      measureContent += generateRestXml(BEATS_PER_MEASURE - bassBeatPos, 2).join('');
    }
    
    measures.push(`<measure number="${measureNum}">${measureContent}</measure>`);
  }

  return measures;
}

function generateMusicXML(notes: NoteResult[], chords: ChordResult[], tempoMultiplier: number = 1, bpm: number = 120): string {
  const measures = generateMeasureXmls(notes, chords, tempoMultiplier, bpm);
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n<score-partwise version="3.1">\n  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n  <part id="P1">${measures.join('')}</part></score-partwise>`;
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
export default function PianoSheetMusic({ results, tempoMultiplier = 1 }: PianoSheetMusicProps) {
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
    if ((incomingNotes.length === 0) && (incomingChords.length === 0)) return;

    // Mark that we've received data
    hasReceivedDataRef.current = true;

    // Always append incoming notes/chords - no automatic reset based on timestamps
    // (Reset should only happen when explicitly triggered, not on every chunk)

    // Append incoming notes, deduplicating by (time_seconds, midi_note)
    setAccumulatedNotes((prev) => {
      const seen = new Set<string>();
      for (const n of prev) seen.add(`${n.time_seconds.toFixed(6)}:${n.midi_note}`);
      const toAdd: NoteResult[] = [];
      for (const n of incomingNotes) {
        const key = `${(n.time_seconds ?? 0).toFixed(6)}:${n.midi_note}`;
        if (!seen.has(key)) { seen.add(key); toAdd.push(n); }
      }
      return [...prev, ...toAdd];
    });

    // Append incoming chords, deduplicating by (time_seconds, midi list)
    setAccumulatedChords((prev) => {
      const seen = new Set<string>();
      for (const c of prev) seen.add(`${(c.time_seconds ?? 0).toFixed(6)}:${(c.midi_notes || []).join('-')}`);
      const toAdd: ChordResult[] = [];
      for (const c of incomingChords) {
        const key = `${(c.time_seconds ?? 0).toFixed(6)}:${(c.midi_notes || []).join('-')}`;
        if (!seen.has(key)) { seen.add(key); toAdd.push(c); }
      }
      return [...prev, ...toAdd];
    });
  }, [results]);

  // Get detected BPM from results, default to 120
  const detectedBPM = results?.analysis_summary?.detected_bpm ?? 120;

  const score = useMemo(() => {
    if ((!accumulatedNotes || accumulatedNotes.length === 0) && (!accumulatedChords || accumulatedChords.length === 0)) return FALLBACK_XML;
    return generateMusicXML(accumulatedNotes, accumulatedChords, tempoMultiplier, detectedBPM);
  }, [accumulatedNotes, accumulatedChords, tempoMultiplier, detectedBPM]);

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
        post({ type: 'setBPM', bpm: detectedBPM });
      }
    }
  }, [detectedBPM, post]);

  const onWebMessage = useCallback(async (e: WebViewMessageEvent) => {
    try {
      const msg = JSON.parse(e.nativeEvent.data);
      if (msg.type === 'webview-click') {
        // Only enter landscape mode (exit is handled by exit button)
        if (!isLandscape) {
          try {
            await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.LANDSCAPE);
            setIsLandscape(true);
          } catch (err) {
            console.warn('Orientation lock failed', err);
          }
        }
        return;
      }

      if (msg.type === 'ready') {
        // Mark WebView as ready
        webViewReadyRef.current = true;
        // init options are already set in the HTML; you can send more here if needed
        // post initial render and mark it as pending so we can sync cache on 'rendered'
        pendingXmlRef.current = score;
        post({ type: 'renderXml', xml: score });
        post({ type: 'toggleCursor', show: true }); // show follow-along cursor (static until you drive it)
        // Always set the BPM from detected tempo (playbackBPM has the current value)
        post({ type: 'setBPM', bpm: playbackBPM });
        console.log('[PianoSheetMusic] WebView ready, setting BPM to:', playbackBPM);
      }
      if (msg.type === 'rendered') {
        // initial main render completed; mark how many measures are present
        if (typeof msg.measures === 'number') measuresSentRef.current = msg.measures;
        // The webview has finished rendering the previously-sent xml; promote pending -> last
        if (pendingXmlRef.current) {
          lastXmlRef.current = pendingXmlRef.current;
          pendingXmlRef.current = null;
        }
      }
      if (msg.type === 'appended') {
        // appended measures ack - increment our counter if provided
        if (typeof msg.appended === 'number') measuresSentRef.current += msg.appended;
      }
      if (msg.type === 'error') {
        console.warn('OSMD error:', msg.error);
      }
      
      // Playback events
      if (msg.type === 'playbackStarted') {
        setIsPlaying(true);
        setIsPaused(false);
        console.log('Playback started:', msg.noteCount, 'notes,', msg.duration.toFixed(1), 'seconds');
      }
      if (msg.type === 'playbackPaused') {
        setIsPaused(true);
      }
      if (msg.type === 'playbackResumed') {
        setIsPaused(false);
      }
      if (msg.type === 'playbackStopped' || msg.type === 'playbackEnded') {
        setIsPlaying(false);
        setIsPaused(false);
      }
      if (msg.type === 'playbackError') {
        console.warn('Playback error:', msg.error);
        setIsPlaying(false);
        setIsPaused(false);
      }
      if (msg.type === 'bpmSet') {
        setPlaybackBPM(msg.bpm);
      }
      // Handle exit fullscreen from WebView controls
      if (msg.type === 'exitFullscreen') {
        try {
          await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT_UP);
          setIsLandscape(false);
        } catch (err) {
          console.warn('Exit fullscreen failed', err);
        }
      }
      // Handle BPM changes from WebView controls
      if (msg.type === 'bpmChanged') {
        setPlaybackBPM(msg.bpm);
      }
    } catch (err) { console.warn('webview message parse error', err); }
  }, [isLandscape, score, post, playbackBPM]);

  // If the score changes (for example after live recording produces results), send the new XML
  // to the WebView so OSMD re-renders the updated score.
  useEffect(() => {
    // small guard: only post if webview ref exists
    if (!webRef.current) return;

    const measures = generateMeasureXmls(accumulatedNotes, accumulatedChords);

    // If we have never sent anything, send the full score and save it
    if (measuresSentRef.current === 0 || !lastXmlRef.current) {
      try {
        // mark this xml as pending and send it; we'll update lastXmlRef when the webview posts 'rendered'
        pendingXmlRef.current = score;
        post({ type: 'renderXml', xml: score });
        post({ type: 'toggleCursor', show: true });
        // don't set lastXmlRef yet; wait for 'rendered' to confirm
        // still set optimistic measures count in case the webview doesn't reply immediately
        measuresSentRef.current = measures.length;
      } catch (e) { console.warn('renderXml post failed', e); }
      return;
    }

    // Otherwise, if there are new measures, compose a combined XML by inserting
    // the new measures into the previously-sent XML, then post the combined XML.
    if (measures.length > measuresSentRef.current) {
      const newMeasures = measures.slice(measuresSentRef.current);

      // Helper: strip <attributes> blocks and renumber measure numbers relative to existing count
      const existingCount = measuresSentRef.current;
      const adjusted = newMeasures.map(m => {
        // remove attributes
        const noAttrs = m.replace(/<attributes>[\s\S]*?<\/attributes>/i, '');
        // renumber number="n" by adding existingCount
        return noAttrs.replace(/number\s*=\s*"(\d+)"/i, function(_, p1) {
          return 'number="' + (existingCount + parseInt(p1, 10)) + '"';
        });
      });

      // insert adjusted measures into lastXmlRef before the last </part> or </score-partwise>
      let base = lastXmlRef.current || '';
      const closingPart = '</part>';
      let newXml;
      const idx = base.lastIndexOf(closingPart);
      if (idx !== -1) {
        newXml = base.slice(0, idx) + adjusted.join('') + base.slice(idx);
      } else {
        const closingScore = '</score-partwise>';
        const idx2 = base.lastIndexOf(closingScore);
        if (idx2 !== -1) newXml = base.slice(0, idx2) + adjusted.join('') + base.slice(idx2);
        else newXml = base + adjusted.join('');
      }

      try {
        // mark pending and send combined xml; wait for 'rendered' to update lastXmlRef
        pendingXmlRef.current = newXml;
        post({ type: 'renderXml', xml: newXml });
        measuresSentRef.current = measures.length; // optimistic
      } catch (e) { console.warn('Failed posting appended renderXml', e); }
    }

  }, [accumulatedNotes, accumulatedChords, post, score]);

  // unlock orientation on unmount to avoid locking the device permanently
  useEffect(() => {
    return () => { ScreenOrientation.unlockAsync().catch(()=>{}); };
  }, []);

  // Notify WebView when fullscreen mode changes
  useEffect(() => {
    post({ type: 'setFullscreenMode', enabled: isLandscape });
  }, [isLandscape, post]);

  return (
    <ScrollView style={styles.container} horizontal scrollEnabled={!isLandscape}>
      <View style={styles.mainContainer}>
        <ThemedText type="subtitle" style={styles.title}>Piano Sheet Music</ThemedText>

        {/* Playback Controls - Clear and Prominent */}
        <View style={styles.playbackSection}>
          <ThemedText style={styles.sectionLabel}>Playback</ThemedText>
          <View style={styles.playbackControls}>
            <View style={styles.playButtonContainer}>
              <Button 
                title={isPlaying ? (isPaused ? "▶ Resume" : "⏸ Pause") : "▶ Play"} 
                color={isPlaying && !isPaused ? "#e67e22" : "#27ae60"}
                onPress={() => {
                  if (!isPlaying) {
                    post({ type: 'play', bpm: playbackBPM });
                  } else if (isPaused) {
                    post({ type: 'play', bpm: playbackBPM });
                  } else {
                    post({ type: 'pause' });
                  }
                }} 
              />
            </View>
            <View style={styles.stopButtonContainer}>
              <Button 
                title="⏹ Stop" 
                color="#c0392b"
                onPress={() => post({ type: 'stop' })} 
              />
            </View>
            <View style={styles.bpmContainer}>
              <Button 
                title="−" 
                color="#7f8c8d"
                onPress={() => {
                  const newBPM = Math.max(40, playbackBPM - 10);
                  setPlaybackBPM(newBPM);
                  post({ type: 'setBPM', bpm: newBPM });
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
                  post({ type: 'setBPM', bpm: newBPM });
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
              originWhitelist={['*']}
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

        {/* View Controls - Secondary */}
        <View style={styles.viewControlsSection}>
          <ThemedText style={styles.sectionLabel}>View Controls</ThemedText>
          <View style={styles.viewControls}>
            <Button title="Zoom −" color="#3498db" onPress={() => post({ type: 'setZoom', zoom: 0.9 })} />
            <Button title="Zoom +" color="#3498db" onPress={() => post({ type: 'setZoom', zoom: 1.1 })} />
            <Button title="Reset Cursor" color="#9b59b6" onPress={() => post({ type: 'cursorReset' })} />
            <Button title="Clear Score" color="#e74c3c" onPress={() => {
              setAccumulatedNotes([]);
              setAccumulatedChords([]);
              lastXmlRef.current = null;
              measuresSentRef.current = 0;
              post({ type: 'renderXml', xml: FALLBACK_XML });
              post({ type: 'stop' });
            }} />
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 10,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    marginBottom: 20,
    width: '100%',
  },
  mainContainer: {
    flex: 1,
  },
  title: {
    textAlign: 'center',
    marginBottom: 12,
    color: '#333',
    fontSize: 18,
    fontWeight: 'bold',
  },
  // Playback section
  playbackSection: {
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  sectionLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
    textAlign: 'center',
  },
  playbackControls: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  playButtonContainer: {
    minWidth: 100,
  },
  stopButtonContainer: {
    minWidth: 70,
  },
  bpmContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 4,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  bpmDisplay: {
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  bpmValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  bpmLabel: {
    fontSize: 10,
    color: '#888',
  },
  // Score section
  scoreSection: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  webview: {
    height: 350,
    borderRadius: 8,
    width: 600,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  landscapeWebview: {
    height: 400,
    borderRadius: 0,
    width: 800,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  // View controls section
  viewControlsSection: {
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  viewControls: {
    flexDirection: 'row',
    justifyContent: 'center',
    flexWrap: 'wrap',
    gap: 8,
  },
  // Legacy - keep for compatibility
  toolbar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
    marginBottom: 8,
  },
  playbackToolbar: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
    paddingVertical: 4,
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: 8,
  },
  bpmText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    minWidth: 40,
    textAlign: 'center',
  }
});
