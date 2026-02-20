import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import {
    Dimensions,
    Platform,
    StatusBar,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from "react-native";
import Animated, {
    runOnJS,
    useAnimatedStyle,
    useFrameCallback,
    useSharedValue,
} from "react-native-reanimated";
import Svg, { G, Line, Rect, Text as SvgText } from "react-native-svg";

// ─── Types ─────────────────────────────────────────────────────────────────────

interface NoteResult {
  time_seconds: number;
  midi_note: number;
  note_name?: string;
  duration_seconds?: number;
  hand?: "bass" | "treble";
  confidence?: number;
  note_value?: string;
}

interface ChordResult {
  time_seconds: number;
  midi_notes?: number[];
  duration_seconds?: number;
  hand?: "bass" | "treble";
  confidence?: number;
}

interface AnalysisResult {
  notes: NoteResult[];
  chords: ChordResult[];
  analysis_summary: {
    duration_seconds: number;
    detected_bpm?: number;
  };
}

interface FullScreenPianoRollProps {
  results?: AnalysisResult;
  isRecording?: boolean;
  isPlaying?: boolean;
  currentTime?: number;
  playbackMode?: "recording" | "synthesized";
  onClose?: () => void;
  onPlayPause?: () => void;
  onStop?: () => void;
  onSeek?: (time: number) => void;
  onTogglePlaybackMode?: () => void;
}

// ─── Constants ─────────────────────────────────────────────────────────────────

const MIDI_MIN = 21;
const MIDI_MAX = 108;
const TOTAL_KEYS = MIDI_MAX - MIDI_MIN + 1;

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");
const PIANO_HEIGHT = 60;
const PIXELS_PER_SECOND = 150;

const COLORS = {
  background: "#1a1a2e",
  whiteKey: "#f0f0f0",
  blackKey: "#2c3e50",
  noteActive: "#00d4ff",
  noteBass: "#ff6b6b",
  noteTreble: "#4ecdc4",
  playbackLine: "#ffd93d",
  beatLine: "rgba(255, 255, 255, 0.1)",
  measureLine: "rgba(255, 255, 255, 0.25)",
  octaveLine: "rgba(255, 255, 255, 0.15)",
};

const isBlackKey = (midi: number): boolean => {
  const note = midi % 12;
  return [1, 3, 6, 8, 10].includes(note);
};

const midiToNoteName = (midi: number): string => {
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
  return `${noteNames[midi % 12]}${octave}`;
};

interface NoteEvent {
  id: string;
  midi: number;
  startTime: number;
  duration: number;
  hand?: "bass" | "treble";
}

// ─── Piano Keyboard Component ──────────────────────────────────────────────────

const HorizontalPianoKeyboard: React.FC<{
  width: number;
  height: number;
  activeNotes?: number[];
}> = ({ width, height, activeNotes = [] }) => {
  const keyWidth = width / TOTAL_KEYS;
  const blackKeyHeight = height * 0.6;

  const keys: React.ReactElement[] = [];

  // White keys first
  for (let midi = MIDI_MIN; midi <= MIDI_MAX; midi++) {
    if (!isBlackKey(midi)) {
      const keyIndex = midi - MIDI_MIN;
      const x = keyIndex * keyWidth;
      const isActive = activeNotes.includes(midi);
      const isC = midi % 12 === 0;

      keys.push(
        <G key={`white-${midi}`}>
          <Rect
            x={x}
            y={0}
            width={keyWidth}
            height={height}
            fill={isActive ? COLORS.noteActive : COLORS.whiteKey}
            stroke="rgba(0,0,0,0.3)"
            strokeWidth={0.5}
          />
          {isC && keyWidth >= 4 && (
            <SvgText
              x={x + keyWidth / 2}
              y={height - 4}
              fontSize={Math.min(8, keyWidth - 1)}
              fill="#666"
              textAnchor="middle"
            >
              {midiToNoteName(midi)}
            </SvgText>
          )}
        </G>,
      );
    }
  }

  // Black keys on top
  for (let midi = MIDI_MIN; midi <= MIDI_MAX; midi++) {
    if (isBlackKey(midi)) {
      const keyIndex = midi - MIDI_MIN;
      const x = keyIndex * keyWidth - keyWidth * 0.3;
      const isActive = activeNotes.includes(midi);

      keys.push(
        <G key={`black-${midi}`}>
          <Rect
            x={x}
            y={0}
            width={keyWidth * 0.6}
            height={blackKeyHeight}
            fill={isActive ? COLORS.noteActive : COLORS.blackKey}
          />
        </G>,
      );
    }
  }

  return <G>{keys}</G>;
};

// ─── Main Component ────────────────────────────────────────────────────────────

export default function FullScreenPianoRoll({
  results,
  isRecording = false,
  isPlaying = false,
  currentTime = 0,
  playbackMode = "recording",
  onClose,
  onPlayPause,
  onStop,
  onSeek,
  onTogglePlaybackMode,
}: FullScreenPianoRollProps) {
  // ─── Animation state (native thread) ─────────────────────────────────────────
  const animatedOffset = useSharedValue(0);
  const isPlayingNative = useSharedValue(false);
  const playStartTimestamp = useSharedValue(0);
  const playStartOffset = useSharedValue(0);

  // Track last synced time to detect external seeks
  const lastSyncedTime = useRef(0);

  // Display time for UI (updated at lower frequency)
  const [displayTime, setDisplayTime] = useState(0);

  // ─── Dimensions ──────────────────────────────────────────────────────────────
  const viewportHeight =
    SCREEN_HEIGHT -
    PIANO_HEIGHT -
    (Platform.OS === "android" ? StatusBar.currentHeight || 0 : 44);

  const duration = results?.analysis_summary?.duration_seconds || 10;
  const bpm = results?.analysis_summary?.detected_bpm || 120;
  const PLAYBACK_LINE_Y = viewportHeight - 50;
  const keyWidth = SCREEN_WIDTH / TOTAL_KEYS;

  // ─── Convert notes to flat list ──────────────────────────────────────────────
  const noteEvents = useMemo<NoteEvent[]>(() => {
    if (!results) return [];

    const events: NoteEvent[] = [];
    let id = 0;

    results.notes.forEach((note) => {
      events.push({
        id: `note-${id++}`,
        midi: note.midi_note,
        startTime: note.time_seconds,
        duration: note.duration_seconds || 0.25,
        hand: note.hand,
      });
    });

    results.chords.forEach((chord) => {
      if (chord.midi_notes) {
        chord.midi_notes.forEach((midi) => {
          events.push({
            id: `chord-${id++}`,
            midi,
            startTime: chord.time_seconds,
            duration: chord.duration_seconds || 0.25,
            hand: chord.hand,
          });
        });
      }
    });

    return events;
  }, [results]);

  // ─── Sync with external time when not playing ────────────────────────────────
  useEffect(() => {
    if (!isPlaying) {
      // When stopped, sync to external time
      animatedOffset.value = currentTime * PIXELS_PER_SECOND;
      lastSyncedTime.current = currentTime;
      setDisplayTime(currentTime);
    } else if (Math.abs(currentTime - lastSyncedTime.current) > 0.5) {
      // User seeked while playing - resync
      animatedOffset.value = currentTime * PIXELS_PER_SECOND;
      playStartOffset.value = currentTime * PIXELS_PER_SECOND;
      playStartTimestamp.value = Date.now();
      lastSyncedTime.current = currentTime;
    }
  }, [
    currentTime,
    isPlaying,
    animatedOffset,
    playStartOffset,
    playStartTimestamp,
  ]);

  // ─── Handle play/pause state changes ─────────────────────────────────────────
  useEffect(() => {
    isPlayingNative.value = isPlaying;
    if (isPlaying) {
      // Starting playback - record start state
      playStartTimestamp.value = Date.now();
      playStartOffset.value = animatedOffset.value;
    }
  }, [
    isPlaying,
    isPlayingNative,
    playStartTimestamp,
    playStartOffset,
    animatedOffset,
  ]);

  // ─── Update display time callback ────────────────────────────────────────────
  const updateDisplayTime = useCallback((time: number) => {
    setDisplayTime(time);
    lastSyncedTime.current = time;
  }, []);

  // ─── Native animation loop (60fps on UI thread) ─────────────────────────────
  useFrameCallback(() => {
    "worklet";
    if (isPlayingNative.value) {
      const elapsedMs = Date.now() - playStartTimestamp.value;
      const elapsedSec = elapsedMs / 1000;
      const newOffset = playStartOffset.value + elapsedSec * PIXELS_PER_SECOND;
      const maxOffset = duration * PIXELS_PER_SECOND;

      if (newOffset <= maxOffset) {
        animatedOffset.value = newOffset;

        // Update JS display time at ~10fps to avoid flooding the bridge
        const frameNumber = Math.floor(elapsedSec * 10);
        const prevFrameNumber = Math.floor((elapsedSec - 0.016) * 10);
        if (frameNumber !== prevFrameNumber) {
          runOnJS(updateDisplayTime)(newOffset / PIXELS_PER_SECOND);
        }
      } else {
        animatedOffset.value = maxOffset;
        runOnJS(updateDisplayTime)(duration);
      }
    }
  });

  // ─── Animated container style ────────────────────────────────────────────────
  const animatedContainerStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: animatedOffset.value }],
  }));

  // ─── Active notes for keyboard highlighting ──────────────────────────────────
  const activeNotes = useMemo(() => {
    return noteEvents
      .filter(
        (event) =>
          displayTime >= event.startTime &&
          displayTime < event.startTime + event.duration,
      )
      .map((event) => event.midi);
  }, [noteEvents, displayTime]);

  // ─── Render notes ────────────────────────────────────────────────────────────
  // All content is positioned relative to a "content origin" at Y=0
  // Notes at time=0 should appear at the bottom of the content area
  // Notes at time=duration should appear at the top
  // contentHeight extends above to fit all notes
  const contentHeight = (duration + 2) * PIXELS_PER_SECOND;

  const renderedNotes = useMemo(() => {
    return noteEvents.map((event) => {
      const keyIndex = event.midi - MIDI_MIN;
      const x = keyIndex * keyWidth;
      const noteWidth = keyWidth - 1;
      const noteHeight = Math.max(event.duration * PIXELS_PER_SECOND, 8);

      // Note position: Y increases downward in SVG
      // Note at startTime=0 should be at the bottom of content (Y = contentHeight - noteHeight)
      // Note at startTime=T should be at Y = contentHeight - noteHeight - T * PPS
      const y =
        contentHeight - noteHeight - event.startTime * PIXELS_PER_SECOND;

      let color = COLORS.noteActive;
      if (event.hand === "bass") color = COLORS.noteBass;
      else if (event.hand === "treble") color = COLORS.noteTreble;

      return (
        <Rect
          key={event.id}
          x={x}
          y={y}
          width={noteWidth}
          height={noteHeight}
          fill={color}
          rx={3}
          ry={3}
          opacity={0.9}
        />
      );
    });
  }, [noteEvents, keyWidth, contentHeight]);

  // ─── Grid lines ──────────────────────────────────────────────────────────────
  const gridLines = useMemo(() => {
    const lines: React.ReactElement[] = [];
    const secondsPerBeat = 60 / bpm;
    const totalBeats = Math.ceil(duration / secondsPerBeat) + 8;

    for (let beat = -4; beat <= totalBeats; beat++) {
      const beatTime = beat * secondsPerBeat;
      // Same coordinate system as notes: Y = contentHeight - beatTime * PPS
      const y = contentHeight - beatTime * PIXELS_PER_SECOND;
      const isMeasure = beat % 4 === 0;

      lines.push(
        <Line
          key={`beat-${beat}`}
          x1={0}
          y1={y}
          x2={SCREEN_WIDTH}
          y2={y}
          stroke={isMeasure ? COLORS.measureLine : COLORS.beatLine}
          strokeWidth={isMeasure ? 1.5 : 0.5}
          strokeDasharray={isMeasure ? undefined : "4,4"}
        />,
      );

      if (isMeasure && beat >= 0) {
        lines.push(
          <SvgText
            key={`label-${beat}`}
            x={5}
            y={y - 5}
            fontSize={10}
            fill="rgba(255,255,255,0.5)"
          >
            {`M${beat / 4 + 1}`}
          </SvgText>,
        );
      }
    }

    return lines;
  }, [bpm, duration, contentHeight]);

  // ─── Octave lines (static) ───────────────────────────────────────────────────
  const octaveLines = useMemo(() => {
    const lines: React.ReactElement[] = [];
    for (let midi = MIDI_MIN; midi <= MIDI_MAX; midi++) {
      if (midi % 12 === 0) {
        const x = (midi - MIDI_MIN) * keyWidth;
        lines.push(
          <Line
            key={`octave-${midi}`}
            x1={x}
            y1={0}
            x2={x}
            y2={viewportHeight}
            stroke={COLORS.octaveLine}
            strokeWidth={1}
          />,
        );
      }
    }
    return lines;
  }, [keyWidth, viewportHeight]);

  // ─── SVG dimensions ──────────────────────────────────────────────────────────
  // SVG height = contentHeight + buffer for notes that have passed the playback line
  const svgHeight = contentHeight + viewportHeight;

  // Container base offset: position so content at Y=contentHeight appears at PLAYBACK_LINE_Y
  const containerBaseOffset = PLAYBACK_LINE_Y - contentHeight;

  // ─── Format time ─────────────────────────────────────────────────────────────
  const formatTime = (seconds: number): string => {
    const absSeconds = Math.abs(seconds);
    const sign = seconds < 0 ? "-" : "";
    const mins = Math.floor(absSeconds / 60);
    const secs = Math.floor(absSeconds % 60);
    return `${sign}${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <View style={styles.container}>
      <View style={styles.background} />

      {/* Roll area */}
      <View style={[styles.rollArea, { height: viewportHeight }]}>
        {/* Static octave lines */}
        <Svg
          width={SCREEN_WIDTH}
          height={viewportHeight}
          style={StyleSheet.absoluteFill}
          pointerEvents="none"
        >
          {octaveLines}
        </Svg>

        {/* Animated scrolling content */}
        <Animated.View
          style={[
            styles.scrollContent,
            { top: containerBaseOffset },
            animatedContainerStyle,
          ]}
        >
          <Svg width={SCREEN_WIDTH} height={svgHeight}>
            <G>{gridLines}</G>
            <G>{renderedNotes}</G>
          </Svg>
        </Animated.View>

        {/* Playback line - fixed */}
        <View style={[styles.playbackLine, { top: PLAYBACK_LINE_Y - 1.5 }]} />
        <View style={[styles.playbackLineGlow, { top: PLAYBACK_LINE_Y - 4 }]} />
      </View>

      {/* Piano keyboard */}
      <View style={[styles.pianoContainer, { height: PIANO_HEIGHT }]}>
        <Svg width={SCREEN_WIDTH} height={PIANO_HEIGHT}>
          <HorizontalPianoKeyboard
            width={SCREEN_WIDTH}
            height={PIANO_HEIGHT}
            activeNotes={activeNotes}
          />
        </Svg>
      </View>

      {/* Top bar */}
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeButtonText}>✕</Text>
        </TouchableOpacity>

        <View style={styles.infoSection}>
          <Text style={styles.bpmText}>{Math.round(bpm)} BPM</Text>
          <Text style={styles.timeText}>
            {formatTime(displayTime)} / {formatTime(duration)}
          </Text>
        </View>

        <View style={styles.noteCountBadge}>
          <Text style={styles.noteCountText}>{noteEvents.length} notes</Text>
        </View>
      </View>

      {/* Controls */}
      <View style={styles.controlBar}>
        <TouchableOpacity
          style={[
            styles.modeButton,
            playbackMode === "synthesized" && styles.modeButtonActive,
          ]}
          onPress={onTogglePlaybackMode}
        >
          <Text style={styles.modeButtonText}>
            {playbackMode === "recording" ? "🎙️" : "🎹"}
          </Text>
          <Text style={styles.modeLabel}>
            {playbackMode === "recording" ? "REC" : "MIDI"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.controlButton,
            styles.playButton,
            isPlaying && styles.controlButtonActive,
          ]}
          onPress={onPlayPause}
        >
          <Text style={styles.controlButtonText}>{isPlaying ? "❚❚" : "▶"}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.controlButton} onPress={onStop}>
          <Text style={styles.controlButtonText}>■</Text>
        </TouchableOpacity>
      </View>

      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>REC</Text>
        </View>
      )}

      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View
            style={[styles.legendColor, { backgroundColor: COLORS.noteTreble }]}
          />
          <Text style={styles.legendText}>R</Text>
        </View>
        <View style={styles.legendItem}>
          <View
            style={[styles.legendColor, { backgroundColor: COLORS.noteBass }]}
          />
          <Text style={styles.legendText}>L</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  background: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: COLORS.background,
  },
  rollArea: {
    flex: 1,
    overflow: "hidden",
  },
  scrollContent: {
    position: "absolute",
    left: 0,
    right: 0,
    top: 0,
  },
  playbackLine: {
    position: "absolute",
    left: 0,
    right: 0,
    height: 3,
    backgroundColor: COLORS.playbackLine,
    zIndex: 10,
  },
  playbackLineGlow: {
    position: "absolute",
    left: 0,
    right: 0,
    height: 8,
    backgroundColor: COLORS.playbackLine,
    opacity: 0.3,
    zIndex: 9,
  },
  pianoContainer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "#1a1a1a",
    borderTopWidth: 2,
    borderTopColor: "#333",
  },
  topBar: {
    position: "absolute",
    top: Platform.OS === "android" ? (StatusBar.currentHeight || 0) + 10 : 50,
    left: 0,
    right: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    zIndex: 100,
  },
  closeButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.15)",
    alignItems: "center",
    justifyContent: "center",
  },
  closeButtonText: {
    fontSize: 20,
    color: "#fff",
    fontWeight: "bold",
  },
  infoSection: {
    alignItems: "center",
  },
  bpmText: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#fff",
  },
  timeText: {
    fontSize: 12,
    color: "rgba(255,255,255,0.7)",
    marginTop: 2,
  },
  noteCountBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 12,
  },
  noteCountText: {
    fontSize: 12,
    color: "#fff",
    fontWeight: "600",
  },
  controlBar: {
    position: "absolute",
    bottom: PIANO_HEIGHT + 20,
    left: 0,
    right: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
    paddingHorizontal: 20,
    zIndex: 100,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "rgba(255,255,255,0.2)",
    alignItems: "center",
    justifyContent: "center",
  },
  controlButtonActive: {
    backgroundColor: COLORS.playbackLine,
  },
  controlButtonText: {
    fontSize: 18,
    color: "#fff",
    fontWeight: "bold",
  },
  playButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  modeButton: {
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.3)",
  },
  modeButtonActive: {
    backgroundColor: "rgba(78, 205, 196, 0.4)",
    borderColor: COLORS.noteTreble,
  },
  modeButtonText: {
    fontSize: 20,
  },
  modeLabel: {
    fontSize: 9,
    color: "#fff",
    fontWeight: "bold",
    marginTop: 2,
  },
  recordingIndicator: {
    position: "absolute",
    top: Platform.OS === "android" ? (StatusBar.currentHeight || 0) + 60 : 100,
    right: 16,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "rgba(255, 0, 0, 0.3)",
    borderRadius: 12,
    zIndex: 100,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#ff0000",
  },
  recordingText: {
    fontSize: 12,
    color: "#ff0000",
    fontWeight: "bold",
  },
  legend: {
    position: "absolute",
    bottom: PIANO_HEIGHT + 80,
    right: 16,
    flexDirection: "row",
    gap: 8,
    zIndex: 100,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: "rgba(0,0,0,0.5)",
    borderRadius: 8,
  },
  legendColor: {
    width: 12,
    height: 8,
    borderRadius: 2,
  },
  legendText: {
    fontSize: 10,
    color: "#fff",
    fontWeight: "600",
  },
});
