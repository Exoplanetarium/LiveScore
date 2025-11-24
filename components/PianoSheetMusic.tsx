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
function generateMeasureXmls(notes: NoteResult[], chords: ChordResult[], tempoMultiplier: number = 1): string[] {
  // Merge notes and chords into a single event sequence by time
  type Event = { time: number; kind: 'note' | 'chord'; note?: NoteResult; chord?: ChordResult };
  const events: Event[] = [];
  for (const n of notes) events.push({ time: n.time_seconds ?? 0, kind: 'note', note: n });
  for (const c of chords) events.push({ time: c.time_seconds ?? 0, kind: 'chord', chord: c });
  events.sort((a, b) => a.time - b.time);

  const measures: string[] = [];
  let currentMeasureContents: string[] = [];
  let measureNumber = 1;
  let currentBeatCount = 0; // Track beats in current measure (4/4 time = 4 beats per measure)

  // Note type order for tempo adjustments
  const noteTypeOrder = ['32nd', '16th', 'eighth', 'quarter', 'half', 'whole'];
  
  // Adjust note type based on tempo multiplier
  // tempoMultiplier > 1 = faster tempo = shorter notes (move down the order)
  // tempoMultiplier < 1 = slower tempo = longer notes (move up the order)
  const adjustNoteType = (noteType?: string): string => {
    if (!noteType || tempoMultiplier === 1) return noteType || 'quarter';
    
    const currentIndex = noteTypeOrder.indexOf(noteType);
    if (currentIndex === -1) return noteType;
    
    let newIndex = currentIndex;
    if (tempoMultiplier === 2) {
      // Double tempo = half the note value (quarter -> eighth)
      newIndex = Math.max(0, currentIndex - 1);
    } else if (tempoMultiplier === 0.5) {
      // Half tempo = double the note value (quarter -> half)
      newIndex = Math.min(noteTypeOrder.length - 1, currentIndex + 1);
    }
    
    return noteTypeOrder[newIndex];
  };

  // Helper to get beat value for a note type (with optional dotted flag)
  const getNoteBeats = (noteType?: string, dotted?: boolean): number => {
    const adjustedType = adjustNoteType(noteType);
    let beats = 1; // default quarter note
    switch (adjustedType) {
      case 'whole': beats = 4; break;
      case 'half': beats = 2; break;
      case 'quarter': beats = 1; break;
      case 'eighth': beats = 0.5; break;
      case '16th': beats = 0.25; break;
      case '32nd': beats = 0.125; break;
      default: beats = 1; break;
    }
    // Dotted notes add half their value
    if (dotted) beats *= 1.5;
    return beats;
  };

  // Helper to get MusicXML duration based on note type and dotted flag
  // divisions=8, so: whole=32, half=16, quarter=8, eighth=4, 16th=2, 32nd=1
  // dotted adds half the value
  const getNoteDuration = (noteType?: string, dotted?: boolean): number => {
    const adjustedType = adjustNoteType(noteType);
    let duration = 8; // default quarter note
    switch (adjustedType) {
      case 'whole': duration = 32; break;
      case 'half': duration = 16; break;
      case 'quarter': duration = 8; break;
      case 'eighth': duration = 4; break;
      case '16th': duration = 2; break;
      case '32nd': duration = 1; break;
      default: duration = 8; break;
    }
    // Dotted notes add half their value
    if (dotted) duration = Math.floor(duration * 1.5);
    return duration;
  };
  
  // Helper to get the adjusted note type string for MusicXML
  const getAdjustedNoteType = (noteType?: string): string => {
    return adjustNoteType(noteType);
  };

  // Track note count per measure for width calculation
  let currentMeasureNoteCount = 0;
  
  // Calculate measure width based on note count and note types
  // More notes = wider measure, shorter notes = wider measure
  const calculateMeasureWidth = (noteCount: number): number => {
    // Base width of 150 + 30 per note, minimum 150, cap at 800
    // This ensures measures with many 32nd notes get proper width
    const width = Math.min(800, Math.max(150, 100 + noteCount * 30));
    return width;
  };

  const pushMeasureIfFull = (beatsToAdd: number = 0, noteCountToAdd: number = 1) => {
    currentBeatCount += beatsToAdd;
    currentMeasureNoteCount += noteCountToAdd;
    
    if (currentBeatCount >= 4) {
      const measureWidth = calculateMeasureWidth(currentMeasureNoteCount);
      const printElement = `<print><measure-layout><measure-distance>${measureWidth}</measure-distance></measure-layout></print>`;
      const attributes = measureNumber === 1 ? '<attributes><divisions>8</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>' : '';
      const measureXml = `<measure number="${measureNumber}" width="${measureWidth}">${attributes}${printElement}${currentMeasureContents.join('')}</measure>`;
      measures.push(measureXml);
      currentMeasureContents = [];
      currentBeatCount = currentBeatCount - 4; // Carry over excess beats
      currentMeasureNoteCount = 0; // Reset note count for next measure
      measureNumber++;
    }
  };

  const noteToXml = (n: NoteResult) => {
    const { step, octave } = midiToStepOctave(n.midi_note);
    const staff = octave < 4 ? 2 : 1;
    let baseStep = step;
    let alter = 0;
    if (step.includes('#')) { baseStep = step[0]; alter = 1; }
    else if (step.includes('b') || step.includes('♭')) { baseStep = step[0]; alter = -1; }
    const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : '';
    const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
    
    // Get note value and duration from the note object, default to quarter
    const noteType = n.note_value || 'quarter';
    const adjustedNoteType = getAdjustedNoteType(noteType);
    const dotted = n.dotted || false;
    const duration = getNoteDuration(noteType, dotted);
    const dotXml = dotted ? '<dot/>' : '';
    
    return `<note>${pitchXml}<duration>${duration}</duration><type>${adjustedNoteType}</type>${dotXml}<staff>${staff}</staff></note>`;
  };

  // Helper to construct chord note MIDI list from ChordResult.
  // Behavior:
  // - If explicit `midi_notes` provided, use them.
  // - Else if `root_midi` provided, build chord from that root.
  // - Else try to parse `label` and `octave` to compute root MIDI and build chord.
  const chordToMidiList = (c: ChordResult): number[] => {
    if (c.midi_notes && c.midi_notes.length) return c.midi_notes.slice();

    // Map note names to semitone offsets
    const nameToSemitone: Record<string, number> = {
      C: 0, 'C#': 1, DB: 1,
      D: 2, 'D#': 3, EB: 3,
      E: 4, FB: 4,
      F: 5, 'F#': 6, GB: 6,
      G: 7, 'G#': 8, AB: 8,
      A: 9, 'A#': 10, BB: 10,
      B: 11, CB: 11,
    };

    const buildFromRoot = (root: number, quality?: string): number[] => {
      const q = (quality || '').toLowerCase();
      // common quality patterns
      if (q.includes('maj7') || q === 'maj7') return [root, root + 4, root + 7, root + 11];
      if (q === '7' || q.includes('dom7') || q === 'dom') return [root, root + 4, root + 7, root + 10];
      if (q === 'm7' || q === 'min7') return [root, root + 3, root + 7, root + 10];
      if (q === 'm' || q === 'min') return [root, root + 3, root + 7];
      if (q === 'dim') return [root, root + 3, root + 6];
      if (q === 'aug') return [root, root + 4, root + 8];
      if (q === 'sus2') return [root, root + 2, root + 7];
      if (q === 'sus4') return [root, root + 5, root + 7];
      // fallback: major triad
      return [root, root + 4, root + 7];
    };

    // If explicit root MIDI exists, use it
    if (typeof c.root_midi === 'number') return buildFromRoot(c.root_midi, c.chord_quality);

    // Try to parse label like 'F#:maj' or 'C:min'
    if (c.label) {
      const m = String(c.label).toUpperCase().match(/^([A-G][#B]?)/);
      if (m) {
        const rootName = m[1].replace('B', 'B');
        const semitone = nameToSemitone[rootName] ?? 0;
        const octave = typeof c.octave === 'number' ? c.octave : 4; // default octave if missing
        const rootMidi = (octave + 1) * 12 + semitone; // align with midiToStepOctave
        return buildFromRoot(rootMidi, c.chord_quality);
      }
    }

    return [];
  };

  for (const ev of events) {
    if (ev.kind === 'note' && ev.note) {
      const noteBeats = getNoteBeats(ev.note.note_value, ev.note.dotted);
      currentMeasureContents.push(noteToXml(ev.note));
      pushMeasureIfFull(noteBeats, 1);
      continue;
    }
    if (ev.kind === 'chord' && ev.chord) {
      const midiList = chordToMidiList(ev.chord);
      if (midiList.length === 0) {
        // nothing to render
        continue;
      }
      // apply inversion: accept numeric or textual inversions
      const inversionToIndex = (inv: any, chordLen: number) => {
        if (typeof inv === 'number') return Math.max(0, Math.floor(inv));
        if (!inv) return 0;
        const s = String(inv).toLowerCase();
        if (s === 'root') return 0;
        if (s === 'first') return 1;
        if (s === 'second') return 2;
        if (s === 'third') return 3;
        // 'slash' commonly indicates a different bass than root; per your rule treat as 3rd inversion for sevenths
        if (s === 'slash') return (chordLen >= 4 ? 3 : 1);
        return 0;
      };

      const inversion = inversionToIndex(ev.chord.inversion, midiList.length);
      const notesOrdered = midiList.slice();
      for (let i = 0; i < inversion; i++) {
        const n = notesOrdered.shift();
        if (typeof n === 'number') notesOrdered.push(n + 12);
      }

      // Get note value and duration from the chord object, default to quarter
      // Apply tempo multiplier to chord note value
      const baseNoteType = ev.chord.note_value || 'quarter';
      const adjustedNoteType = getAdjustedNoteType(baseNoteType);
      const dotted = ev.chord.dotted || false;
      const duration = getNoteDuration(adjustedNoteType, dotted);
      const chordBeats = getNoteBeats(adjustedNoteType, dotted);
      const dotXml = dotted ? '<dot/>' : '';

      // Create MusicXML for chord: first note without <chord/>, others with <chord/>
      let first = true;
      for (const midi of notesOrdered) {
        const { step, octave } = midiToStepOctave(midi);
        const staff = octave < 4 ? 2 : 1;
        let baseStep = step;
        let alter = 0;
        if (step.includes('#')) { baseStep = step[0]; alter = 1; }
        else if (step.includes('b') || step.includes('♭')) { baseStep = step[0]; alter = -1; }
        const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : '';
        const pitchInner = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
        const noteXml = `<note>${first ? '' : '<chord/>'}${pitchInner}<duration>${duration}</duration><type>${adjustedNoteType}</type>${dotXml}<staff>${staff}</staff></note>`;
        currentMeasureContents.push(noteXml);
        first = false;
      }
      pushMeasureIfFull(chordBeats, notesOrdered.length);
      continue;
    }
  }

  // flush remaining
  if (currentMeasureContents.length > 0) {
    const measureWidth = calculateMeasureWidth(currentMeasureNoteCount);
    const printElement = `<print><measure-layout><measure-distance>${measureWidth}</measure-distance></measure-layout></print>`;
    const measureXml = `<measure number="${measureNumber}" width="${measureWidth}">${printElement}${currentMeasureContents.join('')}</measure>`;
    measures.push(measureXml);
  }

  return measures;
}

function generateMusicXML(notes: NoteResult[], chords: ChordResult[], tempoMultiplier: number = 1): string {
  const measures = generateMeasureXmls(notes, chords, tempoMultiplier);
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n<score-partwise version="3.1">\n  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n  <part id="P1">${measures.join('')}</part></score-partwise>`;
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

  const score = useMemo(() => {
    if ((!accumulatedNotes || accumulatedNotes.length === 0) && (!accumulatedChords || accumulatedChords.length === 0)) return FALLBACK_XML;
    return generateMusicXML(accumulatedNotes, accumulatedChords, tempoMultiplier);
  }, [accumulatedNotes, accumulatedChords, tempoMultiplier]);

  const webRef = useRef<WebView>(null);
  const source = useMemo(() => ({ html: OSMD_HTML }), []);
  const measuresSentRef = useRef<number>(0);
  const lastXmlRef = useRef<string | null>(null);
  const pendingXmlRef = useRef<string | null>(null);

  const post = useCallback((obj: any) => {
    webRef.current?.postMessage(JSON.stringify(obj));
  }, []);

  const [isLandscape, setIsLandscape] = useState(false);

  const onWebMessage = useCallback(async (e: WebViewMessageEvent) => {
    try {
      const msg = JSON.parse(e.nativeEvent.data);
      if (msg.type === 'webview-click') {
        // toggle orientation between landscape and portrait
        try {
          if (!isLandscape) {
            await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.LANDSCAPE);
            setIsLandscape(true);
          } else {
            await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT);
            setIsLandscape(false);
          }
        } catch (err) {
          console.warn('Orientation lock failed', err);
        }
        return;
      }

      if (msg.type === 'ready') {
        // init options are already set in the HTML; you can send more here if needed
        // post initial render and mark it as pending so we can sync cache on 'rendered'
        pendingXmlRef.current = score;
        post({ type: 'renderXml', xml: score });
        post({ type: 'toggleCursor', show: true }); // show follow-along cursor (static until you drive it)
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
    } catch (err) { console.warn('webview message parse error', err); }
  }, [post, score, isLandscape]);

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

  return (
    <ScrollView style={styles.container} horizontal scrollEnabled={!isLandscape}>
      <View>
      <ScrollView scrollEnabled={!isLandscape}>
        <ThemedText type="subtitle" style={styles.title}>Piano Sheet Music</ThemedText>

        {/* Quick demo controls (optional) */}

        <View style={styles.toolbar}>
          <Button title="Zoom −" onPress={() => post({ type: 'setZoom', zoom: 0.9 })} />
          <Button title="Reset Cursor" onPress={() => post({ type: 'cursorReset' })} />
          <Button title="Next ▶︎" onPress={() => post({ type: 'cursorNext' })} />
          <Button title="Zoom +" onPress={() => post({ type: 'setZoom', zoom: 1.1 })} />
          <Button title="Reset Score" onPress={() => {
            setAccumulatedNotes([]);
            setAccumulatedChords([]);
            // Clear cached XML and counters, then post fallback to the webview
            lastXmlRef.current = null;
            measuresSentRef.current = 0;
            post({ type: 'renderXml', xml: FALLBACK_XML });
          }} />
        </View>

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
          // allow the web content to scroll independently (Android)
          nestedScrollEnabled={true}
          // ensure the webview itself can scroll
          scrollEnabled={true}
        />
        </ScrollView>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    marginBottom: 20,
    height: 500,
    width: '100%',
  },
  title: {
    textAlign: 'center',
    marginBottom: 12,
    color: '#333',
    fontSize: 18,
    fontWeight: 'bold',
  },
  toolbar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
    marginBottom: 8,
  },
  webview: {
    height: 500,       // adjust as needed
    borderRadius: 8,
    width: 600,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  landscapeWebview: {
    height: 500,       // adjust as needed
    borderRadius: 0,
    width: 800,
    overflow: 'hidden',
    backgroundColor: '#fff',
  }
});
