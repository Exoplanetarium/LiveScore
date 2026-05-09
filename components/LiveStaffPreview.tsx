import React, { useMemo } from "react";
import { StyleSheet, View, useWindowDimensions } from "react-native";
import Svg, {
    Circle,
    Ellipse,
    Line,
    Rect,
    Text as SvgText,
} from "react-native-svg";
import { ThemedText } from "./ThemedText";

type TimeSignature = "4/4" | "3/4" | "6/8";

interface LiveNoteLike {
  time_seconds: number;
  midi_note: number;
  note_name?: string;
  confidence?: number;
  duration_seconds?: number;
  start_beat?: number;
  end_beat?: number;
}

interface LiveChordLike {
  time_seconds: number;
  label: string;
  confidence?: number;
  duration_seconds?: number;
  start_beat?: number;
  end_beat?: number;
}

interface LiveStaffPreviewProps {
  notes: LiveNoteLike[];
  chords?: LiveChordLike[];
  bpm?: number;
  elapsedSeconds: number;
  timeSignature?: TimeSignature;
  isRecording?: boolean;
}

interface RenderedNote {
  id: string;
  x: number;
  xEnd: number;
  y: number;
  confidence: number;
  noteName: string;
  staff: "treble" | "bass";
}

interface RenderedChordLabel {
  id: string;
  x: number;
  label: string;
}

const PREVIEW_HEIGHT = 252;
const PREVIEW_PADDING = 16;
const STAFF_LEFT = 30;
const STAFF_RIGHT = 24;
const STAFF_LINE_SPACING = 10;
const STAFF_WIDTH_FALLBACK = 320;
const TREBLE_TOP = 58;
const BASS_TOP = 152;
const TREBLE_BOTTOM = TREBLE_TOP + STAFF_LINE_SPACING * 4;
const BASS_BOTTOM = BASS_TOP + STAFF_LINE_SPACING * 4;

function parseTimeSignature(timeSignature: TimeSignature) {
  const [beats, unit] = timeSignature.split("/").map(Number);
  return {
    beatsPerMeasure: beats,
    beatUnit: unit,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function midiToY(midi: number, staff: "treble" | "bass") {
  if (staff === "treble") {
    return TREBLE_BOTTOM - (midi - 64) * 3.1;
  }
  return BASS_BOTTOM - (midi - 43) * 3.1;
}

export default function LiveStaffPreview({
  notes,
  chords = [],
  bpm = 120,
  elapsedSeconds,
  timeSignature = "4/4",
  isRecording = false,
}: LiveStaffPreviewProps) {
  const dimensions = useWindowDimensions();
  const safeBpm = bpm > 0 ? bpm : 120;
  const { beatsPerMeasure, beatUnit } = parseTimeSignature(timeSignature);
  const effectiveBeatsPerSecond = safeBpm / 60;
  const currentBeat = elapsedSeconds * effectiveBeatsPerSecond;
  const currentMeasureIndex = Math.max(
    0,
    Math.floor(currentBeat / beatsPerMeasure),
  );
  const measureStartBeat = currentMeasureIndex * beatsPerMeasure;
  const measureEndBeat = measureStartBeat + beatsPerMeasure;
  const measureProgressBeats = clamp(
    currentBeat - measureStartBeat,
    0,
    beatsPerMeasure,
  );
  const availableWidth = Math.max(
    STAFF_WIDTH_FALLBACK,
    dimensions.width - PREVIEW_PADDING * 4,
  );
  const drawableWidth = availableWidth - STAFF_LEFT - STAFF_RIGHT;
  const beatWidth = drawableWidth / beatsPerMeasure;
  const pulse =
    0.7 + 0.3 * Math.max(0, Math.sin(measureProgressBeats * Math.PI));

  const rendered = useMemo(() => {
    const renderedNotes: RenderedNote[] = notes
      .map((note, index) => {
        const startBeat =
          note.start_beat ?? note.time_seconds * effectiveBeatsPerSecond;
        const rawEndBeat =
          note.end_beat ??
          (note.duration_seconds != null
            ? startBeat + note.duration_seconds * effectiveBeatsPerSecond
            : startBeat + 1);
        const endBeat = Math.max(rawEndBeat, startBeat + 0.25);

        if (endBeat <= measureStartBeat || startBeat >= measureEndBeat) {
          return null;
        }

        const visibleStart = clamp(startBeat, measureStartBeat, measureEndBeat);
        const visibleEnd = clamp(endBeat, measureStartBeat, measureEndBeat);
        const x = STAFF_LEFT + (visibleStart - measureStartBeat) * beatWidth;
        const xEnd = STAFF_LEFT + (visibleEnd - measureStartBeat) * beatWidth;
        const staff = note.midi_note >= 60 ? "treble" : "bass";

        return {
          id: `${startBeat}-${note.midi_note}-${index}`,
          x,
          xEnd,
          y: midiToY(note.midi_note, staff),
          confidence: note.confidence ?? 0.75,
          noteName: note.note_name ?? `MIDI ${note.midi_note}`,
          staff,
        } satisfies RenderedNote;
      })
      .filter((note): note is RenderedNote => note != null)
      .sort((left, right) => left.x - right.x);

    const renderedChordLabels: RenderedChordLabel[] = chords
      .map((chord, index) => {
        const startBeat =
          chord.start_beat ?? chord.time_seconds * effectiveBeatsPerSecond;
        if (startBeat < measureStartBeat || startBeat >= measureEndBeat) {
          return null;
        }

        return {
          id: `${startBeat}-${chord.label}-${index}`,
          x: STAFF_LEFT + (startBeat - measureStartBeat) * beatWidth,
          label: chord.label,
        } satisfies RenderedChordLabel;
      })
      .filter((chord): chord is RenderedChordLabel => chord != null)
      .slice(0, 4);

    return {
      renderedNotes,
      renderedChordLabels,
    };
  }, [
    beatWidth,
    chords,
    effectiveBeatsPerSecond,
    measureEndBeat,
    measureStartBeat,
    notes,
  ]);

  const cursorX = STAFF_LEFT + measureProgressBeats * beatWidth;
  const currentBeatLabel = `${measureProgressBeats.toFixed(2)} beats`;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.headerText}>
          <ThemedText type="subtitle" style={styles.title}>
            Live Bar Preview
          </ThemedText>
          <ThemedText style={styles.subtitle}>
            Provisional notation updates faster than the engraved OSMD score.
          </ThemedText>
        </View>
        <View style={styles.badgeWrap}>
          <View
            style={[styles.liveDot, { opacity: isRecording ? pulse : 0.4 }]}
          />
          <ThemedText style={styles.badgeText}>
            {isRecording ? "LIVE" : "READY"}
          </ThemedText>
        </View>
      </View>

      <Svg
        width="100%"
        height={PREVIEW_HEIGHT}
        viewBox={`0 0 ${availableWidth} ${PREVIEW_HEIGHT}`}
      >
        <Rect
          x={0}
          y={0}
          width={availableWidth}
          height={PREVIEW_HEIGHT}
          rx={18}
          fill="#f8fafc"
        />

        <SvgText
          x={STAFF_LEFT}
          y={24}
          fontSize={12}
          fill="#475569"
          fontWeight="700"
        >
          M.{currentMeasureIndex + 1}
        </SvgText>
        <SvgText x={STAFF_LEFT + 48} y={24} fontSize={12} fill="#64748b">
          {timeSignature} at {Math.round(safeBpm)} BPM
        </SvgText>
        <SvgText
          x={availableWidth - STAFF_RIGHT}
          y={24}
          fontSize={12}
          fill="#64748b"
          textAnchor="end"
        >
          {currentBeatLabel}
        </SvgText>

        {[TREBLE_TOP, BASS_TOP].map((top, staffIndex) => (
          <React.Fragment key={`staff-${staffIndex}`}>
            {Array.from({ length: 5 }).map((_, lineIndex) => {
              const y = top + lineIndex * STAFF_LINE_SPACING;
              return (
                <Line
                  key={`staff-line-${staffIndex}-${lineIndex}`}
                  x1={STAFF_LEFT}
                  x2={availableWidth - STAFF_RIGHT}
                  y1={y}
                  y2={y}
                  stroke="#94a3b8"
                  strokeWidth={1}
                />
              );
            })}
          </React.Fragment>
        ))}

        {Array.from({ length: beatsPerMeasure + 1 }).map((_, beatIndex) => {
          const x = STAFF_LEFT + beatIndex * beatWidth;
          return (
            <React.Fragment key={`beat-${beatIndex}`}>
              <Line
                x1={x}
                x2={x}
                y1={TREBLE_TOP - 10}
                y2={BASS_BOTTOM + 10}
                stroke={
                  beatIndex === 0 || beatIndex === beatsPerMeasure
                    ? "#334155"
                    : "#cbd5e1"
                }
                strokeWidth={
                  beatIndex === 0 || beatIndex === beatsPerMeasure ? 2 : 1
                }
                strokeDasharray={
                  beatIndex === 0 || beatIndex === beatsPerMeasure
                    ? undefined
                    : "4 6"
                }
              />
              {beatIndex < beatsPerMeasure ? (
                <SvgText
                  x={x + beatWidth / 2}
                  y={BASS_BOTTOM + 30}
                  fontSize={11}
                  fill="#64748b"
                  textAnchor="middle"
                >
                  {beatIndex + 1}
                </SvgText>
              ) : null}
            </React.Fragment>
          );
        })}

        {rendered.renderedChordLabels.map((chord) => (
          <SvgText
            key={chord.id}
            x={clamp(
              chord.x + 6,
              STAFF_LEFT + 6,
              availableWidth - STAFF_RIGHT - 6,
            )}
            y={42}
            fontSize={12}
            fill="#2563eb"
            fontWeight="700"
          >
            {chord.label}
          </SvgText>
        ))}

        {rendered.renderedNotes.map((note) => {
          const noteColor = note.confidence > 0.8 ? "#0f172a" : "#2563eb";
          const sustainWidth = Math.max(12, note.xEnd - note.x);
          const stemDirection =
            note.staff === "treble" && note.y < TREBLE_TOP + 22 ? 1 : -1;
          const stemX = note.x + (stemDirection > 0 ? -5 : 5);
          const stemY2 = note.y + stemDirection * 26;

          return (
            <React.Fragment key={note.id}>
              <Line
                x1={note.x + 10}
                x2={note.x + sustainWidth}
                y1={note.y}
                y2={note.y}
                stroke="#93c5fd"
                strokeWidth={5}
                strokeLinecap="round"
                opacity={0.65}
              />
              <Ellipse cx={note.x} cy={note.y} rx={8} ry={6} fill={noteColor} />
              <Line
                x1={stemX}
                x2={stemX}
                y1={note.y}
                y2={stemY2}
                stroke={noteColor}
                strokeWidth={1.6}
              />
            </React.Fragment>
          );
        })}

        <Line
          x1={cursorX}
          x2={cursorX}
          y1={TREBLE_TOP - 16}
          y2={BASS_BOTTOM + 16}
          stroke="#ef4444"
          strokeWidth={2}
        />
        <Circle
          cx={cursorX}
          cy={TREBLE_TOP - 22}
          r={5 + pulse * 2}
          fill="#ef4444"
          opacity={0.85}
        />

        <SvgText
          x={STAFF_LEFT - 22}
          y={TREBLE_TOP + 4}
          fontSize={10}
          fill="#64748b"
        >
          TREBLE
        </SvgText>
        <SvgText
          x={STAFF_LEFT - 14}
          y={BASS_TOP + 4}
          fontSize={10}
          fill="#64748b"
        >
          BASS
        </SvgText>

        {rendered.renderedNotes.length === 0 ? (
          <SvgText
            x={availableWidth / 2}
            y={(TREBLE_TOP + BASS_BOTTOM) / 2}
            fontSize={13}
            fill="#64748b"
            textAnchor="middle"
          >
            {isRecording
              ? "Listening for the current bar..."
              : "Start a live session to populate the bar preview."}
          </SvgText>
        ) : null}

        <Rect
          x={STAFF_LEFT}
          y={PREVIEW_HEIGHT - 18}
          width={drawableWidth}
          height={6}
          rx={3}
          fill="#e2e8f0"
        />
        <Rect
          x={STAFF_LEFT}
          y={PREVIEW_HEIGHT - 18}
          width={drawableWidth * (measureProgressBeats / beatsPerMeasure)}
          height={6}
          rx={3}
          fill="#2563eb"
        />
        <SvgText
          x={availableWidth - STAFF_RIGHT}
          y={PREVIEW_HEIGHT - 24}
          fontSize={11}
          fill="#475569"
          textAnchor="end"
        >
          Beat unit {beatUnit}
        </SvgText>
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 18,
    padding: 18,
    backgroundColor: "rgba(37, 99, 235, 0.08)",
    gap: 14,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  headerText: {
    flex: 1,
    gap: 4,
  },
  title: {
    fontSize: 18,
  },
  subtitle: {
    opacity: 0.75,
    lineHeight: 19,
  },
  badgeWrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: "rgba(255, 255, 255, 0)",
  },
  liveDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
    backgroundColor: "#ef4444",
  },
  badgeText: {
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.6,
  },
});
