import React, { useEffect, useMemo, useRef } from "react";
import { StyleSheet, View, useWindowDimensions } from "react-native";
import Animated, {
  useAnimatedStyle,
  useFrameCallback,
  useSharedValue,
} from "react-native-reanimated";
import Svg, { G, Line, Rect, Text as SvgText } from "react-native-svg";
import { ThemedText } from "./ThemedText";

interface LiveNoteLike {
  time_seconds: number;
  midi_note: number;
  note_name?: string;
  confidence?: number;
  duration_seconds?: number;
  hand?: "bass" | "treble";
}

interface LiveChordLike {
  time_seconds: number;
  midi_notes?: number[];
  label?: string;
  confidence?: number;
  duration_seconds?: number;
  hand?: "bass" | "treble";
}

interface LivePianoRollProps {
  notes: LiveNoteLike[];
  chords?: LiveChordLike[];
  bpm?: number;
  elapsedSeconds: number;
  isRecording?: boolean;
  height?: number;
}

const PIXELS_PER_SECOND = 70;
const VISIBLE_SECONDS = 8;
const PLAYHEAD_RATIO = 0.88;
const AXIS_WIDTH = 36;
const KEYBOARD_HEIGHT = 32;
const FOOTER_HEIGHT = 18;
const MIN_NOTE_RANGE = 18;
const NOTE_RANGE_PADDING = 2;
const MIDI_FLOOR = 21;
const MIDI_CEILING = 108;
const MIN_ROW_HEIGHT = 5;

const COLOR_TREBLE = "#0ea5e9";
const COLOR_BASS = "#f97316";
const COLOR_UNKNOWN = "#6366f1";
const COLOR_ACTIVE_RING = "#facc15";

function isBlackKey(midi: number) {
  return [1, 3, 6, 8, 10].includes(midi % 12);
}

function midiToShortName(midi: number) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function colorForHand(hand?: string) {
  if (hand === "treble") return COLOR_TREBLE;
  if (hand === "bass") return COLOR_BASS;
  return COLOR_UNKNOWN;
}

interface RollNote {
  id: string;
  midi: number;
  start: number;
  end: number;
  hand?: string;
  confidence: number;
}

export default function LivePianoRoll({
  notes,
  chords = [],
  bpm = 120,
  elapsedSeconds,
  isRecording = false,
  height = 280,
}: LivePianoRollProps) {
  const dimensions = useWindowDimensions();
  // Roll fills the card width; assume parent gives ~36px outer padding.
  const fullWidth = Math.max(280, dimensions.width - 36);
  const rollWidth = Math.max(160, fullWidth - AXIS_WIDTH);
  const rollHeight = Math.max(120, height - FOOTER_HEIGHT - KEYBOARD_HEIGHT);
  const playheadX = rollWidth * PLAYHEAD_RATIO;

  // ── Pitch range: expanding window of detected notes (sticky to avoid jumps) ──
  const allMidis = useMemo(() => {
    const out: number[] = [];
    for (const n of notes) {
      if (Number.isFinite(n.midi_note)) out.push(n.midi_note);
    }
    for (const c of chords) {
      if (!c.midi_notes) continue;
      for (const m of c.midi_notes) {
        if (Number.isFinite(m)) out.push(m);
      }
    }
    return out;
  }, [notes, chords]);

  const rangeRef = useRef({ min: 60 - 7, max: 60 + 7 });
  const range = useMemo(() => {
    let min = rangeRef.current.min;
    let max = rangeRef.current.max;
    for (const m of allMidis) {
      if (m - NOTE_RANGE_PADDING < min) min = m - NOTE_RANGE_PADDING;
      if (m + NOTE_RANGE_PADDING > max) max = m + NOTE_RANGE_PADDING;
    }
    if (max - min + 1 < MIN_NOTE_RANGE) {
      const center = (min + max) / 2;
      min = Math.floor(center - MIN_NOTE_RANGE / 2);
      max = Math.ceil(center + MIN_NOTE_RANGE / 2);
    }
    min = Math.max(MIDI_FLOOR, min);
    max = Math.min(MIDI_CEILING, max);
    if (max - min + 1 < MIN_NOTE_RANGE) {
      max = Math.min(MIDI_CEILING, min + MIN_NOTE_RANGE - 1);
    }

    // Cap range so each row is at least MIN_ROW_HEIGHT tall.
    const maxRows = Math.max(MIN_NOTE_RANGE, Math.floor(rollHeight / MIN_ROW_HEIGHT));
    if (max - min + 1 > maxRows) {
      const center = (min + max) / 2;
      const half = Math.floor(maxRows / 2);
      min = Math.max(MIDI_FLOOR, Math.floor(center - half));
      max = Math.min(MIDI_CEILING, min + maxRows - 1);
    }

    rangeRef.current = { min, max };
    return { min, max };
  }, [allMidis, rollHeight]);

  const numRows = range.max - range.min + 1;
  const rowHeight = rollHeight / numRows;

  // ── Animated time (smooth scroll on UI thread) ────────────────────────────
  const animatedSeconds = useSharedValue(elapsedSeconds);
  const recordingStartMs = useSharedValue(0);
  const isRecordingNative = useSharedValue(isRecording);

  useEffect(() => {
    isRecordingNative.value = isRecording;
    if (isRecording) {
      recordingStartMs.value = Date.now() - elapsedSeconds * 1000;
    } else {
      animatedSeconds.value = elapsedSeconds;
    }
    // We intentionally only depend on isRecording — elapsedSeconds drift is
    // handled in the next effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording]);

  useEffect(() => {
    if (isRecording && recordingStartMs.value > 0) {
      const expected = (Date.now() - recordingStartMs.value) / 1000;
      if (Math.abs(expected - elapsedSeconds) > 0.6) {
        recordingStartMs.value = Date.now() - elapsedSeconds * 1000;
      }
    } else if (!isRecording) {
      animatedSeconds.value = elapsedSeconds;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elapsedSeconds]);

  useFrameCallback(() => {
    "worklet";
    if (isRecordingNative.value && recordingStartMs.value > 0) {
      animatedSeconds.value = (Date.now() - recordingStartMs.value) / 1000;
    }
  });

  // ── Visible notes (clipped to a window around now) ────────────────────────
  const visibleNotes = useMemo<RollNote[]>(() => {
    const out: RollNote[] = [];
    const head = elapsedSeconds + 1.5;
    const tail = elapsedSeconds - VISIBLE_SECONDS - 1.5;
    notes.forEach((n, i) => {
      const start = n.time_seconds;
      const dur = n.duration_seconds && n.duration_seconds > 0.04 ? n.duration_seconds : 0.22;
      const end = start + dur;
      if (end < tail || start > head) return;
      if (n.midi_note < range.min || n.midi_note > range.max) return;
      out.push({
        id: `n${i}`,
        midi: n.midi_note,
        start,
        end,
        hand: n.hand,
        confidence: n.confidence ?? 0.7,
      });
    });
    chords.forEach((c, i) => {
      if (!c.midi_notes) return;
      const start = c.time_seconds;
      const dur = c.duration_seconds && c.duration_seconds > 0.04 ? c.duration_seconds : 0.22;
      const end = start + dur;
      if (end < tail || start > head) return;
      c.midi_notes.forEach((m, j) => {
        if (m < range.min || m > range.max) return;
        out.push({
          id: `c${i}-${j}`,
          midi: m,
          start,
          end,
          hand: c.hand,
          confidence: c.confidence ?? 0.7,
        });
      });
    });
    return out;
  }, [notes, chords, elapsedSeconds, range.min, range.max]);

  // SVG inner width for the scrolling layer: just enough to contain visible notes.
  const svgInnerWidth = useMemo(() => {
    let maxT = elapsedSeconds + VISIBLE_SECONDS;
    for (const n of visibleNotes) {
      if (n.end > maxT) maxT = n.end;
    }
    return Math.max(rollWidth, (maxT + 2) * PIXELS_PER_SECOND);
  }, [visibleNotes, elapsedSeconds, rollWidth]);

  // ── Pre-rendered notes (absolute coords; container translates) ────────────
  const noteElements = useMemo(
    () =>
      visibleNotes.map((n) => {
        const xStart = n.start * PIXELS_PER_SECOND;
        const xEnd = n.end * PIXELS_PER_SECOND;
        const w = Math.max(4, xEnd - xStart);
        const rowIdx = range.max - n.midi;
        const y = rowIdx * rowHeight;
        const color = colorForHand(n.hand);
        const opacity = 0.5 + 0.5 * Math.min(1, Math.max(0.1, n.confidence));
        return (
          <Rect
            key={n.id}
            x={xStart}
            y={y + 1}
            width={w}
            height={Math.max(2.5, rowHeight - 2)}
            rx={Math.min(4, rowHeight / 2)}
            ry={Math.min(4, rowHeight / 2)}
            fill={color}
            opacity={opacity}
          />
        );
      }),
    [visibleNotes, range.max, rowHeight],
  );

  // ── Beat / measure grid (in absolute time coords — moves with the layer) ──
  const beatLines = useMemo(() => {
    const safeBpm = bpm > 0 ? bpm : 120;
    const secondsPerBeat = 60 / safeBpm;
    const visStart = Math.max(0, elapsedSeconds - VISIBLE_SECONDS - 1);
    const visEnd = elapsedSeconds + 2;
    const firstBeat = Math.floor(visStart / secondsPerBeat);
    const lastBeat = Math.ceil(visEnd / secondsPerBeat);
    const lines: React.ReactElement[] = [];
    for (let b = firstBeat; b <= lastBeat; b++) {
      const x = b * secondsPerBeat * PIXELS_PER_SECOND;
      const isMeasure = b % 4 === 0;
      lines.push(
        <Line
          key={`beat-${b}`}
          x1={x}
          x2={x}
          y1={0}
          y2={rollHeight}
          stroke={isMeasure ? "rgba(100,116,139,0.55)" : "rgba(148,163,184,0.22)"}
          strokeWidth={isMeasure ? 1 : 0.5}
        />,
      );
    }
    return lines;
  }, [bpm, elapsedSeconds, rollHeight]);

  // ── Static row stripes (black-key rows) on the static layer ───────────────
  const rowStripes = useMemo(() => {
    const out: React.ReactElement[] = [];
    for (let m = range.min; m <= range.max; m++) {
      if (!isBlackKey(m)) continue;
      const rowIdx = range.max - m;
      out.push(
        <Rect
          key={`stripe-${m}`}
          x={0}
          y={rowIdx * rowHeight}
          width={rollWidth}
          height={rowHeight}
          fill="rgba(15,23,42,0.045)"
        />,
      );
    }
    return out;
  }, [range.min, range.max, rowHeight, rollWidth]);

  // ── Y-axis labels (one per C, plus min/max if not a C) ────────────────────
  const yAxisLabels = useMemo(() => {
    const labels: { midi: number; label: string; y: number }[] = [];
    const seen = new Set<number>();
    for (let m = Math.ceil(range.min / 12) * 12; m <= range.max; m += 12) {
      if (m < range.min || m > range.max) continue;
      labels.push({
        midi: m,
        label: midiToShortName(m),
        y: (range.max - m) * rowHeight,
      });
      seen.add(m);
    }
    if (!seen.has(range.max)) {
      labels.push({ midi: range.max, label: midiToShortName(range.max), y: 0 });
    }
    if (!seen.has(range.min)) {
      labels.push({
        midi: range.min,
        label: midiToShortName(range.min),
        y: (range.max - range.min) * rowHeight,
      });
    }
    return labels;
  }, [range.min, range.max, rowHeight]);

  // ── Currently-active notes (for the mini keyboard highlight) ──────────────
  const activeMidis = useMemo(() => {
    const set = new Set<number>();
    for (const n of visibleNotes) {
      if (elapsedSeconds >= n.start - 0.05 && elapsedSeconds <= n.end + 0.05) {
        set.add(n.midi);
      }
    }
    return set;
  }, [visibleNotes, elapsedSeconds]);

  // ── Animated scroll style ────────────────────────────────────────────────
  const scrollStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: playheadX - animatedSeconds.value * PIXELS_PER_SECOND },
    ],
  }));

  const playheadDotStyle = useAnimatedStyle(() => {
    const phase = (animatedSeconds.value * (bpm > 0 ? bpm : 120)) / 60;
    const pulse = 0.7 + 0.3 * Math.max(0, Math.sin(phase * Math.PI));
    return {
      opacity: isRecordingNative.value ? pulse : 0.5,
    };
  });

  const hasContent = notes.length > 0 || chords.length > 0;

  return (
    <View style={[styles.container, { height }]}>
      <View style={[styles.row, { height: rollHeight }]}>
        {/* Y axis (static) */}
        <Svg
          width={AXIS_WIDTH}
          height={rollHeight}
          style={styles.axisSvg}
        >
          <Rect x={0} y={0} width={AXIS_WIDTH} height={rollHeight} fill="#f8fafc" />
          {yAxisLabels.map((l) => (
            <G key={`yl-${l.midi}`}>
              <Line
                x1={0}
                x2={AXIS_WIDTH}
                y1={l.y + rowHeight / 2}
                y2={l.y + rowHeight / 2}
                stroke="rgba(148,163,184,0.55)"
                strokeWidth={0.5}
              />
              <SvgText
                x={4}
                y={l.y + Math.min(rowHeight / 2 + 3, 11)}
                fontSize={9}
                fontWeight="700"
                fill="#475569"
              >
                {l.label}
              </SvgText>
            </G>
          ))}
        </Svg>

        {/* Roll viewport (clipped) */}
        <View style={[styles.viewport, { width: rollWidth, height: rollHeight }]}>
          {/* Static stripes */}
          <Svg
            width={rollWidth}
            height={rollHeight}
            style={StyleSheet.absoluteFill}
            pointerEvents="none"
          >
            <Rect x={0} y={0} width={rollWidth} height={rollHeight} fill="#ffffff" />
            <G>{rowStripes}</G>
          </Svg>

          {/* Animated scrolling content */}
          <Animated.View style={[styles.scrollLayer, scrollStyle]}>
            <Svg width={svgInnerWidth} height={rollHeight}>
              <G>{beatLines}</G>
              <G>{noteElements}</G>
            </Svg>
          </Animated.View>

          {/* Playhead (static) */}
          <View
            style={[styles.playhead, { left: playheadX - 1 }]}
            pointerEvents="none"
          />
          <Animated.View
            style={[styles.playheadDot, { left: playheadX - 6 }, playheadDotStyle]}
            pointerEvents="none"
          />

          {/* Empty state */}
          {!hasContent ? (
            <View style={styles.emptyOverlay} pointerEvents="none">
              <ThemedText style={styles.emptyText}>
                {isRecording
                  ? "Listening for notes…"
                  : "Press start to roll the live notes."}
              </ThemedText>
            </View>
          ) : null}
        </View>
      </View>

      {/* Mini keyboard strip aligned with the roll's pitch range */}
      <View style={[styles.row, { height: KEYBOARD_HEIGHT }]}>
        <View style={{ width: AXIS_WIDTH }} />
        <View style={{ width: rollWidth, height: KEYBOARD_HEIGHT }}>
          <MiniKeyboardStrip
            width={rollWidth}
            height={KEYBOARD_HEIGHT}
            midiMin={range.min}
            midiMax={range.max}
            activeMidis={activeMidis}
          />
        </View>
      </View>

      {/* Footer */}
      <View style={[styles.footer, { height: FOOTER_HEIGHT }]}>
        <ThemedText style={styles.footerText}>
          ← past · live cursor →
        </ThemedText>
        <ThemedText style={styles.footerText}>
          {Math.round(bpm)} BPM · {visibleNotes.length} on screen
        </ThemedText>
      </View>
    </View>
  );
}

interface MiniKeyboardStripProps {
  width: number;
  height: number;
  midiMin: number;
  midiMax: number;
  activeMidis: Set<number>;
}

function MiniKeyboardStrip({
  width,
  height,
  midiMin,
  midiMax,
  activeMidis,
}: MiniKeyboardStripProps) {
  // Width is allotted per *all* keys in the displayed range so the strip
  // visually mirrors the roll above (one row per pitch).
  const totalKeys = midiMax - midiMin + 1;
  const keyWidth = width / totalKeys;
  const blackKeyHeight = height * 0.6;
  const elements: React.ReactElement[] = [];

  for (let m = midiMin; m <= midiMax; m++) {
    if (isBlackKey(m)) continue;
    const idx = m - midiMin;
    const x = idx * keyWidth;
    const isActive = activeMidis.has(m);
    elements.push(
      <Rect
        key={`w-${m}`}
        x={x}
        y={0}
        width={keyWidth}
        height={height}
        fill={isActive ? COLOR_ACTIVE_RING : "#fefefe"}
        stroke="rgba(15,23,42,0.25)"
        strokeWidth={0.5}
      />,
    );
  }

  for (let m = midiMin; m <= midiMax; m++) {
    if (!isBlackKey(m)) continue;
    const idx = m - midiMin;
    const x = idx * keyWidth - keyWidth * 0.3;
    const isActive = activeMidis.has(m);
    elements.push(
      <Rect
        key={`b-${m}`}
        x={x}
        y={0}
        width={keyWidth * 0.6}
        height={blackKeyHeight}
        fill={isActive ? COLOR_ACTIVE_RING : "#1e293b"}
        rx={1}
      />,
    );
  }

  return (
    <Svg width={width} height={height}>
      {elements}
    </Svg>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#f8fafc",
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
  },
  row: {
    flexDirection: "row",
    alignItems: "stretch",
  },
  axisSvg: {
    backgroundColor: "#f1f5f9",
  },
  viewport: {
    backgroundColor: "#ffffff",
    overflow: "hidden",
    position: "relative",
  },
  scrollLayer: {
    position: "absolute",
    top: 0,
    left: 0,
  },
  playhead: {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: "#ef4444",
    zIndex: 5,
  },
  playheadDot: {
    position: "absolute",
    top: 6,
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#ef4444",
    zIndex: 6,
  },
  emptyOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyText: {
    fontSize: 13,
    color: "#64748b",
  },
  footer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 10,
    backgroundColor: "#f1f5f9",
  },
  footerText: {
    fontSize: 10,
    color: "#64748b",
    letterSpacing: 0.4,
    textTransform: "uppercase",
  },
});
