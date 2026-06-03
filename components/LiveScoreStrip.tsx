import React, { useEffect, useMemo } from "react";
import { StyleSheet, View, useWindowDimensions } from "react-native";
import Animated, {
  useAnimatedStyle,
  useFrameCallback,
  useSharedValue,
} from "react-native-reanimated";
import Svg, { Circle, Line, Text as SvgText } from "react-native-svg";

type TimeSignature = "4/4" | "3/4" | "6/8";
type NoteState = "heard" | "candidate" | "active" | "committed" | "locked";

interface StripNote {
  time_seconds: number;
  midi_note: number;
  confidence?: number;
  method?: string;
  start_beat?: number;
}

interface StripChord {
  time_seconds: number;
  midi_notes?: number[];
  confidence?: number;
  method?: string;
  start_beat?: number;
}

interface AnalysisResultLike {
  notes?: StripNote[];
  chords?: StripChord[];
  analysis_summary?: {
    detected_bpm?: number;
    duration_seconds?: number;
  };
}

interface LiveScoreStripProps {
  results?: AnalysisResultLike | null;
  bpm?: number;
  localElapsedSeconds?: number;
  localStartedAtMs?: number | null;
  isRecording?: boolean;
  timeSignature?: TimeSignature;
}

interface DrawableNote {
  id: string;
  beat: number;
  midi: number;
  confidence?: number;
  state: NoteState;
}

const HEIGHT = 150;
const PAD_X = 24;
const TOP = 18;
const LINE_GAP = 6;
const TREBLE_TOP = TOP + 6;
const BASS_TOP = TREBLE_TOP + 48;
const STAFF_BOTTOM = BASS_TOP + LINE_GAP * 4 + 11;
const STAFF_TOP = TREBLE_TOP - 11;

function getBeatsPerMeasure(timeSignature: TimeSignature) {
  if (timeSignature === "3/4") return 3;
  if (timeSignature === "6/8") return 3;
  return 4;
}

function getStateFromMethod(method?: string): NoteState {
  if (method?.includes("locked")) return "locked";
  if (method?.includes("committed")) return "committed";
  if (method?.includes("active")) return "active";
  if (method?.includes("candidate")) return "candidate";
  return "heard";
}

function getStateRank(state: NoteState) {
  switch (state) {
    case "locked":
      return 4;
    case "committed":
      return 3;
    case "active":
      return 2;
    case "candidate":
      return 1;
    default:
      return 0;
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function midiToY(midi: number) {
  if (midi >= 60) {
    const semitoneFromE4 = midi - 64;
    return TREBLE_TOP + LINE_GAP * 4 - semitoneFromE4 * (LINE_GAP / 3.5);
  }
  const semitoneFromG2 = midi - 43;
  return BASS_TOP + LINE_GAP * 4 - semitoneFromG2 * (LINE_GAP / 3.5);
}

function noteColor(state: NoteState) {
  switch (state) {
    case "locked":
    case "committed":
      return { fill: "#111827", stroke: "#020617", opacity: 0.94 };
    case "active":
      return { fill: "#0f766e", stroke: "#042f2e", opacity: 0.86 };
    case "candidate":
      return {
        fill: "rgba(20, 184, 166, 0.22)",
        stroke: "rgba(15, 118, 110, 0.56)",
        opacity: 1,
      };
    default:
      return {
        fill: "rgba(14, 165, 233, 0.16)",
        stroke: "rgba(14, 116, 144, 0.46)",
        opacity: 1,
      };
  }
}

function buildDrawableNotes(
  results: AnalysisResultLike | null | undefined,
  bpm: number,
): DrawableNote[] {
  const secondsPerBeat = 60 / Math.max(40, Math.min(240, bpm || 120));
  const rawNotes: DrawableNote[] = [];

  results?.notes?.forEach((note, index) => {
    const beat =
      typeof note.start_beat === "number" && Number.isFinite(note.start_beat)
        ? note.start_beat
        : note.time_seconds / secondsPerBeat;
    rawNotes.push({
      id: `n-${index}-${note.midi_note}-${Math.round(beat * 48)}`,
      beat,
      midi: note.midi_note,
      confidence: note.confidence,
      state: getStateFromMethod(note.method),
    });
  });

  results?.chords?.forEach((chord, chordIndex) => {
    const beat =
      typeof chord.start_beat === "number" && Number.isFinite(chord.start_beat)
        ? chord.start_beat
        : chord.time_seconds / secondsPerBeat;
    chord.midi_notes?.forEach((midi, noteIndex) => {
      rawNotes.push({
        id: `c-${chordIndex}-${noteIndex}-${midi}-${Math.round(beat * 48)}`,
        beat,
        midi,
        confidence: chord.confidence,
        state: getStateFromMethod(chord.method),
      });
    });
  });

  const deduped = new Map<string, DrawableNote>();
  for (const note of rawNotes) {
    if (!Number.isFinite(note.beat) || !Number.isFinite(note.midi)) continue;
    const key = `${note.midi}:${Math.round(note.beat * 48)}`;
    const previous = deduped.get(key);
    if (
      !previous ||
      getStateRank(note.state) > getStateRank(previous.state) ||
      ((note.confidence ?? 0) > (previous.confidence ?? 0) &&
        getStateRank(note.state) === getStateRank(previous.state))
    ) {
      deduped.set(key, note);
    }
  }

  return [...deduped.values()].sort(
    (left, right) => left.beat - right.beat || left.midi - right.midi,
  );
}

export default function LiveScoreStrip({
  results,
  bpm = 120,
  localElapsedSeconds,
  localStartedAtMs,
  isRecording = false,
  timeSignature = "4/4",
}: LiveScoreStripProps) {
  const dimensions = useWindowDimensions();
  const width = Math.max(320, dimensions.width - 32);
  const right = width - PAD_X;
  const usable = Math.max(1, right - PAD_X);
  const resolvedBpm =
    results?.analysis_summary?.detected_bpm &&
    results.analysis_summary.detected_bpm > 0
      ? results.analysis_summary.detected_bpm
      : bpm;

  const model = useMemo(() => {
    const beatsPerMeasure = getBeatsPerMeasure(timeSignature);
    const secondsPerBeat = 60 / Math.max(40, Math.min(240, resolvedBpm || 120));
    const audioBeat =
      typeof results?.analysis_summary?.duration_seconds === "number" &&
      Number.isFinite(results.analysis_summary.duration_seconds)
        ? results.analysis_summary.duration_seconds / secondsPerBeat
        : 0;
    const localBeat =
      typeof localStartedAtMs === "number" &&
      Number.isFinite(localStartedAtMs)
        ? Math.max(0, (Date.now() - localStartedAtMs) / 1000) / secondsPerBeat
        : typeof localElapsedSeconds === "number" &&
            Number.isFinite(localElapsedSeconds)
          ? localElapsedSeconds / secondsPerBeat
          : 0;
    const cursorBeat = localBeat > 0 || isRecording ? localBeat : audioBeat;
    const allNotes = buildDrawableNotes(results, resolvedBpm);
    const latestNoteBeat = allNotes.length
      ? Math.max(...allNotes.map((note) => note.beat))
      : 0;
    const latestBeat =
      cursorBeat > 0 || isRecording ? cursorBeat : latestNoteBeat;
    const currentMeasure = Math.max(
      0,
      Math.floor(latestBeat / beatsPerMeasure),
    );
    const firstMeasure = Math.max(0, currentMeasure - 1);
    const startBeat = firstMeasure * beatsPerMeasure;
    const endBeat = startBeat + beatsPerMeasure * 2;
    const visibleNotes = allNotes.filter(
      (note) => note.beat >= startBeat - 0.001 && note.beat < endBeat + 0.001,
    );
    const noteRadius =
      visibleNotes.length > 42 ? 2.9 : visibleNotes.length > 28 ? 3.4 : 4.1;

    return {
      audioBeat,
      beatsPerMeasure,
      cursorBeat,
      currentMeasure,
      endBeat,
      firstMeasure,
      noteRadius,
      startBeat,
      totalBeats: endBeat - startBeat,
      visibleNotes,
    };
  }, [
    isRecording,
    localElapsedSeconds,
    localStartedAtMs,
    resolvedBpm,
    results,
    timeSignature,
  ]);

  const cursorX = useSharedValue(PAD_X);
  const cursorOpacity = useSharedValue(0);
  const cursorBeatValue = useSharedValue(model.cursorBeat);
  const startBeatValue = useSharedValue(model.startBeat);
  const endBeatValue = useSharedValue(model.endBeat);
  const bpmValue = useSharedValue(resolvedBpm);
  const isRecordingValue = useSharedValue(isRecording);
  const localStartedAtMsValue = useSharedValue(localStartedAtMs ?? 0);
  const widthValue = useSharedValue(width);
  const updateMsValue = useSharedValue(Date.now());

  useEffect(() => {
    cursorBeatValue.value = model.cursorBeat;
    startBeatValue.value = model.startBeat;
    endBeatValue.value = model.endBeat;
    bpmValue.value = resolvedBpm;
    isRecordingValue.value = isRecording;
    localStartedAtMsValue.value = localStartedAtMs ?? 0;
    widthValue.value = width;
    updateMsValue.value = Date.now();
  }, [
    bpmValue,
    cursorBeatValue,
    endBeatValue,
    isRecording,
    isRecordingValue,
    localStartedAtMs,
    localStartedAtMsValue,
    model.cursorBeat,
    model.endBeat,
    model.startBeat,
    resolvedBpm,
    startBeatValue,
    updateMsValue,
    width,
    widthValue,
  ]);

  useFrameCallback(() => {
    "worklet";
    const elapsedMs = Math.max(0, Date.now() - updateMsValue.value);
    const nowMs = Date.now();
    const localStartedAtMs = localStartedAtMsValue.value;
    const nowBeat =
      isRecordingValue.value && localStartedAtMs > 0
        ? Math.max(0, (nowMs - localStartedAtMs) / 1000) *
          (bpmValue.value / 60)
        : cursorBeatValue.value +
          (isRecordingValue.value
            ? (elapsedMs / 1000) * (bpmValue.value / 60)
            : 0);
    const totalBeats = Math.max(0.001, endBeatValue.value - startBeatValue.value);
    const stripRight = widthValue.value - PAD_X;
    const stripUsable = Math.max(1, stripRight - PAD_X);
    cursorX.value =
      PAD_X + ((nowBeat - startBeatValue.value) / totalBeats) * stripUsable;
    cursorOpacity.value =
      nowBeat >= startBeatValue.value && nowBeat <= endBeatValue.value ? 1 : 0;
  });

  const nowLineStyle = useAnimatedStyle(() => ({
    opacity: cursorOpacity.value,
    transform: [{ translateX: cursorX.value }],
  }));

  const xForBeat = (beat: number) =>
    PAD_X + ((beat - model.startBeat) / model.totalBeats) * usable;
  const yForMidi = (midi: number) => clamp(midiToY(midi), 10, HEIGHT - 10);

  const beatBuckets = useMemo(() => {
    const buckets = new Map<number, number>();
    for (const note of model.visibleNotes) {
      const bucket = Math.round(note.beat * 12);
      buckets.set(bucket, (buckets.get(bucket) ?? 0) + 1);
    }
    return buckets;
  }, [model.visibleNotes]);
  const beatIndexes = new Map<number, number>();

  return (
    <View style={styles.container} pointerEvents="none">
      <Svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`}>
        {[TREBLE_TOP, BASS_TOP].map((staffTop) =>
          [0, 1, 2, 3, 4].map((line) => {
            const y = staffTop + line * LINE_GAP;
            return (
              <Line
                key={`staff-${staffTop}-${line}`}
                x1={PAD_X}
                y1={y}
                x2={right}
                y2={y}
                stroke="#334155"
                strokeOpacity={0.58}
                strokeWidth={1.1}
              />
            );
          }),
        )}

        {[0, 1, 2].map((measure) => {
          const x = xForBeat(
            model.startBeat + measure * model.beatsPerMeasure,
          );
          return (
            <React.Fragment key={`measure-${measure}`}>
              <Line
                x1={x}
                y1={TREBLE_TOP - 8}
                x2={x}
                y2={BASS_TOP + LINE_GAP * 4 + 8}
                stroke="#0f172a"
                strokeOpacity={0.34}
                strokeWidth={1.15}
              />
              {measure < 2 ? (
                <SvgText
                  x={x + 4}
                  y={14}
                  fill="#475569"
                  fontSize={10}
                  fontWeight="700"
                >
                  {`m.${model.firstMeasure + measure + 1}`}
                </SvgText>
              ) : null}
            </React.Fragment>
          );
        })}

        {Array.from({ length: Math.max(0, model.totalBeats - 1) }).map(
          (_, index) => {
            const x = xForBeat(model.startBeat + index + 1);
            return (
              <Line
                key={`beat-${index}`}
                x1={x}
                y1={TREBLE_TOP - 5}
                x2={x}
                y2={BASS_TOP + LINE_GAP * 4 + 5}
                stroke="#64748b"
                strokeOpacity={0.16}
                strokeWidth={0.8}
              />
            );
          },
        )}

        {model.visibleNotes.length === 0 ? (
          <SvgText
            x={width / 2}
            y={HEIGHT / 2 + 5}
            fill="#64748b"
            fillOpacity={0.72}
            fontSize={13}
            fontWeight="650"
            textAnchor="middle"
          >
            Live notes will appear here
          </SvgText>
        ) : null}

        {model.visibleNotes.map((note) => {
          const bucket = Math.round(note.beat * 12);
          const groupSize = beatBuckets.get(bucket) ?? 1;
          const groupIndex = beatIndexes.get(bucket) ?? 0;
          beatIndexes.set(bucket, groupIndex + 1);
          const jitter =
            groupSize > 1 ? (groupIndex - (groupSize - 1) / 2) * 1.4 : 0;
          const x = xForBeat(note.beat) + jitter;
          const y = yForMidi(note.midi);
          const colors = noteColor(note.state);
          const opacity =
            typeof note.confidence === "number" && note.confidence < 0.55
              ? colors.opacity * 0.55
              : colors.opacity;
          const isGhost = note.state === "heard" || note.state === "candidate";

          return (
            <React.Fragment key={note.id}>
              {isGhost ? (
                <Circle
                  cx={x}
                  cy={y}
                  r={model.noteRadius + 3.2}
                  fill="none"
                  stroke="rgba(20, 184, 166, 0.24)"
                  strokeWidth={1.4}
                />
              ) : null}
              <Circle
                cx={x}
                cy={y}
                r={model.noteRadius}
                fill={colors.fill}
                opacity={opacity}
                stroke={colors.stroke}
                strokeWidth={1.15}
              />
            </React.Fragment>
          );
        })}
      </Svg>

      <Animated.View style={[styles.nowLineWrap, nowLineStyle]}>
        <View style={styles.nowGlow} />
        <View style={styles.nowLine} />
        <View style={styles.nowCap} />
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: HEIGHT,
    minHeight: HEIGHT,
    overflow: "hidden",
    backgroundColor: "#f8fafc",
  },
  nowLineWrap: {
    position: "absolute",
    top: STAFF_TOP,
    left: 0,
    width: 1,
    height: STAFF_BOTTOM - STAFF_TOP,
  },
  nowGlow: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: -3,
    width: 7,
    borderRadius: 4,
    backgroundColor: "rgba(239, 68, 68, 0.18)",
  },
  nowLine: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    width: 1.6,
    borderRadius: 1,
    backgroundColor: "rgba(239, 68, 68, 0.82)",
  },
  nowCap: {
    position: "absolute",
    top: -3,
    left: -2.2,
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "rgba(239, 68, 68, 0.94)",
  },
});
