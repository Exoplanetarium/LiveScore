import { Ionicons } from "@expo/vector-icons";
import { Midi } from "@tonejs/midi";
import * as FileSystem from "expo-file-system";
import { LinearGradient } from "expo-linear-gradient";
import * as Sharing from "expo-sharing";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ActivityIndicator,
  Alert,
  LayoutChangeEvent,
  PermissionsAndroid,
  Platform,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  View,
} from "react-native";
import AudioRecord from "react-native-audio-record";
import LiveScoreStrip from "../components/LiveScoreStrip";
import PianoSheetMusic from "../components/PianoSheetMusic";
import { ThemedText } from "../components/ThemedText";
import { ThemedView } from "../components/ThemedView";
import { useLiveRhythm } from "../hooks/useLiveRhythm";

const BACKEND_URL =
  "https://exoplanetarium--livescore-gpu-fastapi-app.modal.run";
const CHUNK_INTERVAL_MS = 600;
const LIVE_AUDIO_SAMPLE_RATE = 44100;
const USE_LIVE_STREAM_TRANSPORT = true;
const LIVE_STREAM_CONTEXT_SEC = 1.8;
const LIVE_STREAM_INFERENCE_INTERVAL_MS = 50;
const LIVE_STREAM_TRUSTED_DELAY_MS = 100;
const LIVE_STREAM_COMMIT_DELAY_MS = 500;
const LIVE_STREAM_LOCK_DELAY_MS = 2000;
const USE_LIVE_NEURAL_PATH = true;
const USE_LIVE_ADAPTIVE_ONSET_THRESHOLD_EXPERIMENT = true;
const USE_LIVE_OSMD_ENGRAVING_EXPERIMENT = true;
const LIVE_PREVIEW_BATCH_MS = 33;
const LIVE_PREVIEW_STALE_FLUSH_MS = 180;
const LIVE_STREAM_ANALYSIS_BATCH_MS = 250;
const LIVE_OSMD_BATCH_MS = 500;
const LIVE_PREVIEW_STRIP_LOOKBACK_BEATS = 12;
const LIVE_PREVIEW_STRIP_MIN_HISTORY_SEC = 6;
const LIVE_PREVIEW_STRIP_LOOKAHEAD_SEC = 1.5;

type LiveNoiseProfile = "open" | "balanced" | "clean";

const LIVE_NOISE_PROFILE_OPTIONS: {
  value: LiveNoiseProfile;
  label: string;
  description: string;
}[] = [
  {
    value: "open",
    label: "Open",
    description:
      "Keeps faint note recovery aggressive. Best when the melody is dropping out.",
  },
  {
    value: "balanced",
    label: "Balanced",
    description:
      "Keeps the current live filter behavior, with moderate cleanup on short low-confidence notes.",
  },
  {
    value: "clean",
    label: "Clean",
    description:
      "Rejects more low-confidence, low-duration notes before they enter the live session.",
  },
];

interface OnsetResult {
  duration_seconds?: number;
  frame_index?: number;
  offset_frame?: number;
  offset_seconds?: number;
  time_seconds: number;
}

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
  hand?: "bass" | "treble";
  note_value?: "whole" | "half" | "quarter" | "eighth" | "16th" | "32nd";
  note_divisions?: number;
  dotted?: boolean;
  triplet?: boolean;
  triplet_position?: "start" | "middle" | "end";
  triplet_type?: "half" | "quarter" | "eighth" | "16th" | "32nd";
  actual_notes?: number;
  normal_notes?: number;
  start_beat?: number;
  end_beat?: number;
  duration_source?: string;
  timing_authority?: string;
  local_beat_duration?: number;
  rest_after_beats?: number;
}

interface ChordResult {
  time_seconds: number;
  frame_index?: number;
  midi_notes?: number[];
  note_names?: string[];
  root?: string;
  octave?: number;
  chord_quality?: string;
  label: string;
  inversion?: string;
  confidence: number;
  method?: string;
  offset_seconds?: number;
  duration_seconds?: number;
  hand?: "bass" | "treble";
  note_value?: "whole" | "half" | "quarter" | "eighth" | "16th" | "32nd";
  note_divisions?: number;
  dotted?: boolean;
  triplet?: boolean;
  triplet_position?: "start" | "middle" | "end";
  start_beat?: number;
  end_beat?: number;
  duration_source?: string;
  timing_authority?: string;
  local_beat_duration?: number;
  rest_after_beats?: number;
}

interface AnalysisResult {
  onsets: OnsetResult[];
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
    bass_notes?: number;
    treble_notes?: number;
    bass_chords?: number;
    treble_chords?: number;
    method?: string;
    device?: string;
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

type LivePreviewStripResult = Pick<
  AnalysisResult,
  "notes" | "chords" | "analysis_summary"
>;

interface RecordedChunkTelemetry {
  sequenceNumber: number;
  captureStartedAtMs: number;
  captureStoppedAtMs: number;
  fileReadyAtMs: number;
}

interface QueuedChunkUpload {
  path: string;
  telemetry: RecordedChunkTelemetry;
}

interface LiveStreamNotePayload {
  id?: number;
  state?: "candidate" | "active" | "committed" | "locked";
  midi_note: number;
  onset_time: number;
  offset_time?: number;
  duration?: number;
  confidence?: number;
  observations?: number;
  first_seen_time?: number;
  last_seen_time?: number;
}

interface LiveStreamDebugSample {
  midi?: number;
  onset?: number;
  offset?: number;
  confidence?: number;
  source?: string;
  attack_ratio?: number;
  attack_delta?: number;
  strong_attack?: boolean;
  reason?: string;
  id?: number;
  audio_time?: number;
  base_midi?: number;
  interval?: number;
  existing_midi?: number;
  existing_onset_time?: number;
  repeat_gap_ms?: number;
}

interface LiveStreamUpdate {
  type: string;
  session?: {
    session_id?: string;
    audio_time_sec?: number;
    current_time_sec?: number;
    sample_rate?: number;
    buffered_sec?: number;
    stream_backlog_sec?: number;
    context_sec?: number;
    inference_interval_sec?: number;
    trusted_delay_sec?: number;
    commit_delay_sec?: number;
    lock_delay_sec?: number;
    transport_mode?: string;
  };
  inference?: {
    ran?: boolean;
    reason?: string;
    inference_ms?: number;
    observation_count?: number;
    received_packet_count?: number;
    skipped_inference_count?: number;
    neural_timing?: {
      neural_total?: number;
      neural_real_time_factor?: number;
      neural_model_total?: number;
      neural_model_real_time_factor?: number;
      neural_feature_extraction?: number;
      neural_model_inference?: number;
      neural_decode_notes?: number;
    };
    analysis_summary?: {
      neural_model?: string;
      live_onset_threshold?: number;
      live_onset_threshold_profile?: string;
    };
    continuity_filter?: {
      input?: number;
      kept?: number;
      suppressed?: number;
      same_pitch_boundary?: number;
      implausible_repeat?: number;
      harmonic_sustain?: number;
      weak_birth_outside_attack?: number;
      attack_groups?: number;
      registered_attack_groups?: number;
      total_suppressed?: number;
      suppressed_samples?: LiveStreamDebugSample[];
    };
    hypothesis_update?: {
      input?: number;
      created?: number;
      matched?: number;
      stale_skipped?: number;
      promoted_active?: number;
      promoted_committed?: number;
      promoted_locked?: number;
      birth_samples?: LiveStreamDebugSample[];
    };
  };
  warmup?: {
    status?: string;
    inference_ms?: number;
    neural_timing?: {
      neural_total?: number;
      neural_real_time_factor?: number;
      neural_model_total?: number;
      neural_model_real_time_factor?: number;
      neural_feature_extraction?: number;
      neural_model_inference?: number;
      neural_decode_notes?: number;
    };
    error?: string;
  };
  heard_notes?: LiveStreamNotePayload[];
  candidate_notes?: LiveStreamNotePayload[];
  active_notes?: LiveStreamNotePayload[];
  committed_notes?: LiveStreamNotePayload[];
  locked_notes?: LiveStreamNotePayload[];
  counts?: {
    candidate?: number;
    active?: number;
    committed?: number;
    locked?: number;
  };
  refinement?: {
    needs_refresh?: boolean;
    refined_notes?: NoteResult[];
    refinement_version?: number;
    bpm?: number;
    bpm_confidence?: number;
    next_refinement_poll_ms?: number | null;
    timing_ms?: {
      display_state?: number;
    };
  };
  all_notes?: NoteResult[];
  all_chords?: ChordResult[];
  error?: string;
}

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

function getLiveStreamUrl() {
  const wsBaseUrl = BACKEND_URL.replace(/^https:/, "wss:").replace(
    /^http:/,
    "ws:",
  );
  return `${wsBaseUrl}/live/stream`;
}
interface MemoizedLivePreviewStripProps {
  analysisFallback: AnalysisResult | null;
  bpm: number;
  isRecording: boolean;
  localElapsedSeconds: number;
  localStartedAtMs: number | null;
  previewResult: LivePreviewStripResult | null;
}

const MemoizedLivePreviewStrip = React.memo(
  function MemoizedLivePreviewStrip({
    analysisFallback,
    bpm,
    isRecording,
    localElapsedSeconds,
    localStartedAtMs,
    previewResult,
  }: MemoizedLivePreviewStripProps) {
    return (
      <LiveScoreStrip
        results={previewResult ?? analysisFallback}
        bpm={bpm}
        localElapsedSeconds={localElapsedSeconds}
        localStartedAtMs={localStartedAtMs}
        isRecording={isRecording}
      />
    );
  },
  (prev, next) =>
    prev.analysisFallback === next.analysisFallback &&
    prev.bpm === next.bpm &&
    prev.isRecording === next.isRecording &&
    prev.localElapsedSeconds === next.localElapsedSeconds &&
    prev.localStartedAtMs === next.localStartedAtMs &&
    prev.previewResult === next.previewResult,
);

function midiToNoteName(midi: number) {
  const names = [
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
  const pitchClass = ((midi % 12) + 12) % 12;
  const octave = Math.floor(midi / 12) - 1;
  return `${names[pitchClass]}${octave}`;
}

function quantizeStreamDurationBeats(rawBeats: number): {
  beats: number;
  noteValue: NonNullable<NoteResult["note_value"]>;
  dotted?: boolean;
  triplet?: boolean;
} {
  const candidates: {
    beats: number;
    noteValue: NonNullable<NoteResult["note_value"]>;
    dotted?: boolean;
    triplet?: boolean;
  }[] = [
    { beats: 4, noteValue: "whole" },
    { beats: 3, noteValue: "half", dotted: true },
    { beats: 2, noteValue: "half" },
    { beats: 1.5, noteValue: "quarter", dotted: true },
    { beats: 1, noteValue: "quarter" },
    { beats: 0.75, noteValue: "eighth", dotted: true },
    { beats: 2 / 3, noteValue: "quarter", triplet: true },
    { beats: 0.5, noteValue: "eighth" },
    { beats: 0.375, noteValue: "16th", dotted: true },
    { beats: 1 / 3, noteValue: "eighth", triplet: true },
    { beats: 0.25, noteValue: "16th" },
    { beats: 1 / 6, noteValue: "16th", triplet: true },
    { beats: 0.125, noteValue: "32nd" },
  ];
  const clampedBeats = Math.max(0.125, Math.min(4, rawBeats));

  return candidates.reduce((best, candidate) =>
    Math.abs(candidate.beats - clampedBeats) <
    Math.abs(best.beats - clampedBeats)
      ? candidate
      : best,
  );
}

function getLiveStreamPayloadBounds(payload: LiveStreamNotePayload) {
  const onset = Number(payload.onset_time ?? 0);
  const duration = Math.max(
    0.04,
    Number(payload.duration ?? (payload.offset_time ?? onset) - onset),
  );
  const offset = Number(payload.offset_time ?? onset + duration);
  return { duration, offset, onset };
}

function streamPayloadToNote(
  payload: LiveStreamNotePayload,
  bpm: number = 120,
): NoteResult {
  const { duration, offset, onset } = getLiveStreamPayloadBounds(payload);
  const secondsPerBeat = 60 / Math.max(40, Math.min(240, bpm || 120));
  const startBeat = Math.round((onset / secondsPerBeat) * 24) / 24;
  const durationSpec = quantizeStreamDurationBeats(duration / secondsPerBeat);

  return {
    time_seconds: onset,
    start_beat: startBeat,
    end_beat: startBeat + durationSpec.beats,
    midi_note: payload.midi_note,
    note_name: midiToNoteName(payload.midi_note),
    method: `live_stream_${payload.state ?? "active"}`,
    confidence: payload.confidence,
    offset_seconds: offset,
    duration_seconds: Math.max(0.04, offset - onset),
    note_value: durationSpec.noteValue,
    note_divisions: durationSpec.beats,
    dotted: durationSpec.dotted,
    triplet: durationSpec.triplet,
    actual_notes: durationSpec.triplet ? 3 : undefined,
    normal_notes: durationSpec.triplet ? 2 : undefined,
    hand: payload.midi_note < 60 ? "bass" : "treble",
  };
}

function getLiveStreamNoteKey(payload: LiveStreamNotePayload) {
  if (payload.id != null) {
    return `id-${payload.id}`;
  }
  return `${payload.midi_note}-${Math.round((payload.onset_time ?? 0) * 1000)}`;
}

function buildLiveStreamPreviewResult(
  update: LiveStreamUpdate,
  bpm: number | undefined,
  previewPayloads: Map<string, LiveStreamNotePayload>,
): LivePreviewStripResult {
  const safeBpm = Math.max(40, Math.min(240, bpm || 120));
  const secondsPerBeat = 60 / safeBpm;
  const audioTimeSec = Math.max(
    0,
    update.session?.audio_time_sec ?? update.session?.current_time_sec ?? 0,
  );
  const minVisibleTimeSec = Math.max(
    0,
    audioTimeSec -
      Math.max(
        LIVE_PREVIEW_STRIP_MIN_HISTORY_SEC,
        secondsPerBeat * LIVE_PREVIEW_STRIP_LOOKBACK_BEATS,
      ),
  );
  const maxVisibleTimeSec = audioTimeSec + LIVE_PREVIEW_STRIP_LOOKAHEAD_SEC;
  const visiblePayloads = [
    ...(update.heard_notes ?? []).map((payload) => ({
      ...payload,
      state: "active" as const,
    })),
    ...(update.candidate_notes ?? []).map((payload) => ({
      ...payload,
      state: "active" as const,
    })),
    ...(update.active_notes ?? []).map((payload) => ({
      ...payload,
      state: "active" as const,
    })),
    ...(update.committed_notes ?? []),
    ...(update.locked_notes ?? []),
  ];
  const snapshotKeys = new Set<string>();

  for (const payload of visiblePayloads) {
    const key = getLiveStreamNoteKey(payload);
    snapshotKeys.add(key);
    previewPayloads.set(key, payload);
  }

  for (const [key, payload] of previewPayloads) {
    const { offset, onset } = getLiveStreamPayloadBounds(payload);
    const isTransient =
      payload.state !== "committed" && payload.state !== "locked";
    if (
      (isTransient && !snapshotKeys.has(key)) ||
      offset < minVisibleTimeSec ||
      onset > maxVisibleTimeSec
    ) {
      previewPayloads.delete(key);
    }
  }

  const notes = [...previewPayloads.values()]
    .map((payload) => streamPayloadToNote(payload, safeBpm))
    .sort(
      (left, right) =>
        left.time_seconds - right.time_seconds ||
        left.midi_note - right.midi_note,
    );

  return {
    notes,
    chords: [],
    analysis_summary: {
      total_onsets: notes.length,
      total_notes: notes.length,
      total_chords: 0,
      duration_seconds: audioTimeSec,
      sample_rate: LIVE_AUDIO_SAMPLE_RATE,
      detected_bpm: bpm,
      tempo_confidence: undefined,
      method: "live_stream_preview",
    },
  };
}

function buildLiveStreamAnalysisResult(
  update: LiveStreamUpdate,
  bpm?: number,
  bpmConfidence?: number,
  accumulatedPayloads?: Map<string, LiveStreamNotePayload>,
  includeUnstableNotes = false,
): AnalysisResult {
  const noteMap = new Map<string, NoteResult>();
  const visibleNotes = [
    ...(includeUnstableNotes ? (update.heard_notes ?? []) : []),
    ...(includeUnstableNotes ? (update.candidate_notes ?? []) : []),
    ...(includeUnstableNotes ? (update.active_notes ?? []) : []),
    ...(update.committed_notes ?? []),
    ...(update.locked_notes ?? []),
    ...(!includeUnstableNotes ? (update.active_notes ?? []) : []),
  ];

  for (const payload of visibleNotes) {
    const key = getLiveStreamNoteKey(payload);
    accumulatedPayloads?.set(key, payload);
    noteMap.set(key, streamPayloadToNote(payload, bpm));
  }

  if (accumulatedPayloads) {
    noteMap.clear();
    for (const [key, payload] of accumulatedPayloads) {
      noteMap.set(key, streamPayloadToNote(payload, bpm));
    }
  }

  const notes = [...noteMap.values()];
  const onsets = notes.map((note) => ({
    time_seconds: note.time_seconds,
    duration_seconds: note.duration_seconds,
  }));

  const result = buildLiveAnalysisResult(notes, [], onsets, bpm, bpmConfidence);
  result.analysis_summary.method = "live_stream";
  result.analysis_summary.duration_seconds = Math.max(
    result.analysis_summary.duration_seconds,
    update.session?.audio_time_sec ?? update.session?.current_time_sec ?? 0,
  );
  return result;
}

function mergeLiveStreamAnalysisPayloads(
  update: LiveStreamUpdate,
  accumulatedPayloads?: Map<string, LiveStreamNotePayload>,
  includeUnstableNotes = false,
) {
  if (!accumulatedPayloads) {
    return;
  }

  const visibleNotes = [
    ...(includeUnstableNotes ? (update.heard_notes ?? []) : []),
    ...(includeUnstableNotes ? (update.candidate_notes ?? []) : []),
    ...(includeUnstableNotes ? (update.active_notes ?? []) : []),
    ...(update.committed_notes ?? []),
    ...(update.locked_notes ?? []),
    ...(!includeUnstableNotes ? (update.active_notes ?? []) : []),
  ];

  for (const payload of visibleNotes) {
    accumulatedPayloads.set(getLiveStreamNoteKey(payload), payload);
  }
}

function buildAnalysisResultEventSignature(result: AnalysisResult) {
  const notes = result.notes
    .map((note) =>
      [
        note.midi_note,
        Math.round((note.time_seconds ?? 0) * 1000),
        Math.round((note.duration_seconds ?? 0) * 1000),
        Math.round((note.start_beat ?? -1) * 24),
        Math.round((note.note_divisions ?? -1) * 24),
        note.note_value ?? "",
        note.dotted ? 1 : 0,
        note.triplet ? 1 : 0,
      ].join(":"),
    )
    .join("|");
  const chords = result.chords
    .map((chord) =>
      [
        Math.round((chord.time_seconds ?? 0) * 1000),
        (chord.midi_notes ?? []).join(","),
        Math.round((chord.duration_seconds ?? 0) * 1000),
      ].join(":"),
    )
    .join("|");
  return `${result.notes.length}/${result.chords.length}:${notes}::${chords}`;
}

function buildLiveAnalysisResult(
  notes: NoteResult[],
  chords: ChordResult[],
  onsets: OnsetResult[],
  bpm?: number,
  bpmConfidence?: number,
): AnalysisResult {
  const sortedNotes = [...notes].sort(
    (left, right) => left.time_seconds - right.time_seconds,
  );
  const sortedChords = [...chords].sort(
    (left, right) => left.time_seconds - right.time_seconds,
  );
  const sortedOnsets = [...onsets].sort(
    (left, right) => left.time_seconds - right.time_seconds,
  );

  const lastTimes = [
    ...sortedNotes.map((note) => note.offset_seconds ?? note.time_seconds),
    ...sortedChords.map((chord) => chord.offset_seconds ?? chord.time_seconds),
    ...sortedOnsets.map((onset) => onset.time_seconds),
  ];

  const durationSeconds = lastTimes.length > 0 ? Math.max(...lastTimes) : 0;

  return {
    onsets: sortedOnsets,
    notes: sortedNotes,
    chords: sortedChords,
    analysis_summary: {
      total_onsets: sortedOnsets.length,
      total_notes: sortedNotes.length,
      total_chords: sortedChords.length,
      duration_seconds: durationSeconds,
      sample_rate: 44100,
      detected_bpm: bpm,
      tempo_confidence: bpmConfidence,
      method: "live",
    },
  };
}

interface MemoizedScoreContentProps {
  analysisResult: AnalysisResult | null;
  compact: boolean;
  hasStoppedRecording: boolean;
  isRecording: boolean;
  isWarmingUp: boolean;
  liveEngravingResult: AnalysisResult | null;
  liveEngravingVersion: number;
  onScoreScrollActiveChange: (active: boolean) => void;
  viewportHeight?: number;
}

const MemoizedScoreContent = React.memo(function MemoizedScoreContent({
  analysisResult,
  compact,
  hasStoppedRecording,
  isRecording,
  isWarmingUp,
  liveEngravingResult,
  liveEngravingVersion,
  onScoreScrollActiveChange,
  viewportHeight,
}: MemoizedScoreContentProps) {
  if (USE_LIVE_OSMD_ENGRAVING_EXPERIMENT) {
    return (
      <PianoSheetMusic
        results={liveEngravingResult ?? undefined}
        refinementVersion={liveEngravingVersion}
        compact={compact}
        showCompactPlaybackOverlay={hasStoppedRecording}
        viewportHeight={viewportHeight}
        onScoreScrollActiveChange={onScoreScrollActiveChange}
      />
    );
  }

  if (hasStoppedRecording && analysisResult) {
    return (
      <PianoSheetMusic
        results={analysisResult}
        compact={compact}
        showCompactPlaybackOverlay={hasStoppedRecording}
        viewportHeight={viewportHeight}
        onScoreScrollActiveChange={onScoreScrollActiveChange}
      />
    );
  }

  return (
    <View style={compact ? styles.liveScorePlaceholder : null}>
      <ThemedText style={styles.placeholderText}>
        {isRecording
          ? "Recording... live engraving will continue to update here while the controls stay pinned below."
          : isWarmingUp
            ? "Warming the live neural path. Recording has not started yet."
            : "Start a live session to capture notes. The piano roll above updates in real time; sheet music renders after stop."}
      </ThemedText>
    </View>
  );
});

function getConnectionStatusColor(status: ConnectionStatus) {
  switch (status) {
    case "connected":
      return "#30a46c";
    case "connecting":
      return "#f59e0b";
    case "error":
      return "#dc2626";
    default:
      return "#94a3b8";
  }
}

function getConnectionStatusText(
  status: ConnectionStatus,
  isProcessing: boolean,
  isWarmingUp: boolean,
) {
  if (isWarmingUp) {
    return "Warming neural path";
  }

  if (isProcessing) {
    return USE_LIVE_STREAM_TRANSPORT ? "Updating stream" : "Processing chunk";
  }

  switch (status) {
    case "connected":
      return USE_LIVE_STREAM_TRANSPORT
        ? "Live stream active"
        : "Live session active";
    case "connecting":
      return "Connecting";
    case "error":
      return USE_LIVE_STREAM_TRANSPORT
        ? "Live stream failed"
        : "Chunk upload failed";
    default:
      return "Idle";
  }
}

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(1);
  return `${minutes}:${remainder.padStart(4, "0")}`;
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number) {
  let timeoutHandle: ReturnType<typeof setTimeout> | null = null;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutHandle = setTimeout(() => {
      reject(new Error(`Timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
    }
  }
}

export default function LiveTranscriptionScreen() {
  const [isRecording, setIsRecording] = useState(false);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [duration, setDuration] = useState(0);
  const [recordingStartedAtMs, setRecordingStartedAtMs] = useState<
    number | null
  >(null);
  const [liveScoreViewportHeight, setLiveScoreViewportHeight] = useState(0);
  const [isScoreScrollActive, setIsScoreScrollActive] = useState(false);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null,
  );
  const analysisResultRef = useRef<AnalysisResult | null>(null);
  const [liveEngravingResult, setLiveEngravingResult] =
    useState<AnalysisResult | null>(null);
  const [livePreviewResult, setLivePreviewResult] =
    useState<LivePreviewStripResult | null>(null);
  const [liveEngravingVersion, setLiveEngravingVersion] = useState(0);
  const [sessionReady, setSessionReady] = useState(false);
  const [noiseProfile, setNoiseProfile] =
    useState<LiveNoiseProfile>("balanced");
  const scrollViewRef = useRef<ScrollView | null>(null);
  const liveScoreSectionYRef = useRef(0);

  const durationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const chunkTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const engravingFlushTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const pendingEngravingRef = useRef<{
    result: AnalysisResult | null;
    version: number;
  }>({ result: null, version: 0 });
  const livePreviewFlushTimeoutRef = useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  const pendingLivePreviewRef = useRef<LivePreviewStripResult | null>(null);
  const pendingLivePreviewQueuedAtRef = useRef<number | null>(null);
  const pendingLivePreviewAudioTimeMsRef = useRef<number | null>(null);
  const liveStreamAnalysisFlushTimeoutRef = useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  const pendingLiveStreamAnalysisUpdateRef = useRef<LiveStreamUpdate | null>(
    null,
  );
  const lastLivePreviewFlushAtRef = useRef(0);
  const lastLiveStreamAnalysisFlushAtRef = useRef(0);
  const coalescedLivePreviewUpdatesRef = useRef(0);
  const droppedLivePreviewUpdatesRef = useRef(0);
  const livePreviewFlushCountRef = useRef(0);
  const lastLivePreviewQueueWaitMsRef = useRef<number | null>(null);
  const maxLivePreviewQueueWaitMsRef = useRef(0);
  const lastLivePreviewDataLagMsRef = useRef<number | null>(null);
  const pendingChunkUploadRef = useRef<QueuedChunkUpload | null>(null);
  const isRecordingRef = useRef(false);
  const recordingTimerStartedAtRef = useRef<number | null>(null);
  const sessionReadyRef = useRef(false);
  const backendWarmupPromiseRef = useRef<Promise<void> | null>(null);
  const liveStreamSocketRef = useRef<WebSocket | null>(null);
  const liveStreamSessionIdRef = useRef<string | null>(null);
  const liveStreamPacketSequenceRef = useRef(0);
  const liveStreamNotePayloadsRef = useRef<Map<string, LiveStreamNotePayload>>(
    new Map(),
  );
  const liveStreamPreviewPayloadsRef = useRef<
    Map<string, LiveStreamNotePayload>
  >(new Map());
  const liveStreamAnalysisSignatureRef = useRef("");
  const liveStreamAudioSubscriptionRef = useRef<{
    remove?: () => void;
  } | null>(null);
  const liveStreamStartResolveRef = useRef<(() => void) | null>(null);
  const liveStreamStartRejectRef = useRef<((error: Error) => void) | null>(
    null,
  );
  const liveStreamWarmResolveRef = useRef<(() => void) | null>(null);
  const liveStreamWarmRejectRef = useRef<((error: Error) => void) | null>(null);
  const liveStreamStopResolveRef = useRef<(() => void) | null>(null);
  const currentChunkStartedAtRef = useRef<number | null>(null);
  const chunkSequenceRef = useRef(0);
  const liveStreamRecordingStartedAtRef = useRef<number | null>(null);
  const liveStreamFirstPacketSentAtRef = useRef<number | null>(null);
  const lastLiveStreamLatencyLogAtRef = useRef(0);
  const liveDebugBirthsRef = useRef(0);
  const liveDebugMatchedRef = useRef(0);
  const liveDebugStaleSkippedRef = useRef(0);
  const liveDebugSuppressedRef = useRef(0);
  const liveDebugPromotedActiveRef = useRef(0);
  const liveDebugPromotedCommittedRef = useRef(0);
  const liveDebugPromotedLockedRef = useRef(0);
  const liveDebugBirthSamplesRef = useRef<LiveStreamDebugSample[]>([]);
  const liveDebugSuppressedSamplesRef = useRef<LiveStreamDebugSample[]>([]);
  // Track concurrent uploads so the spinner only clears when the queue drains.
  const inFlightUploadsRef = useRef(0);
  const [hasStoppedRecording, setHasStoppedRecording] = useState(false);
  const selectedNoiseProfile =
    LIVE_NOISE_PROFILE_OPTIONS.find(
      (option) => option.value === noiseProfile,
    ) ?? LIVE_NOISE_PROFILE_OPTIONS[1];
  const isStartDisabled = isWarmingUp;
  const isLiveSessionLayout = true;
  const recordButtonLabel = isRecording
    ? "Stop Live Session"
    : isWarmingUp
      ? "Warming Neural Path..."
      : "Start Live Session";
  const exportableEventCount =
    (analysisResult?.notes.length ?? 0) + (analysisResult?.chords.length ?? 0);
  const hasExportableScore = exportableEventCount > 0;

  useEffect(() => {
    sessionReadyRef.current = sessionReady;
  }, [sessionReady]);

  useEffect(() => {
    analysisResultRef.current = analysisResult;
  }, [analysisResult]);

  const clearLivePreviewQueue = useCallback(() => {
    if (livePreviewFlushTimeoutRef.current) {
      clearTimeout(livePreviewFlushTimeoutRef.current);
      livePreviewFlushTimeoutRef.current = null;
    }
    pendingLivePreviewRef.current = null;
    pendingLivePreviewQueuedAtRef.current = null;
    pendingLivePreviewAudioTimeMsRef.current = null;
    lastLivePreviewFlushAtRef.current = 0;
    coalescedLivePreviewUpdatesRef.current = 0;
    droppedLivePreviewUpdatesRef.current = 0;
    livePreviewFlushCountRef.current = 0;
    lastLivePreviewQueueWaitMsRef.current = null;
    maxLivePreviewQueueWaitMsRef.current = 0;
    lastLivePreviewDataLagMsRef.current = null;
  }, []);

  const clearLiveStreamAnalysisQueue = useCallback(() => {
    if (liveStreamAnalysisFlushTimeoutRef.current) {
      clearTimeout(liveStreamAnalysisFlushTimeoutRef.current);
      liveStreamAnalysisFlushTimeoutRef.current = null;
    }
    pendingLiveStreamAnalysisUpdateRef.current = null;
    lastLiveStreamAnalysisFlushAtRef.current = 0;
  }, []);

  const handleLiveScoreSectionLayout = useCallback(
    (event: LayoutChangeEvent) => {
      liveScoreSectionYRef.current = event.nativeEvent.layout.y;
    },
    [],
  );

  const clearPendingChunkUpload = useCallback(() => {
    const pendingUpload = pendingChunkUploadRef.current;
    pendingChunkUploadRef.current = null;
    if (!pendingUpload) {
      return;
    }

    FileSystem.deleteAsync(pendingUpload.path, { idempotent: true }).catch(
      () => {},
    );
  }, []);

  const logChunkPipelineTiming = useCallback(
    (
      telemetry: RecordedChunkTelemetry,
      uploadStartedAtMs: number,
      uploadFinishedAtMs: number,
      mergeQueuedAtMs: number,
      firstFrameAtMs: number,
      timing?: {
        analysisPath?: string;
        chunkTotalMs?: number;
        chunkInferenceMs?: number;
        neuralTotalMs?: number;
        modelInferenceMs?: number;
        realTimeFactor?: number;
        neuralError?: string;
        onsetThreshold?: number;
        onsetThresholdProfile?: string;
        onsetThresholdExperiment?: string;
        chunkRms?: number;
        chunkPeak?: number;
        chunkCrestFactor?: number;
      },
    ) => {
      const captureWindowMs =
        telemetry.captureStoppedAtMs - telemetry.captureStartedAtMs;
      const filePreparationMs =
        telemetry.fileReadyAtMs - telemetry.captureStoppedAtMs;
      const queueDelayMs = uploadStartedAtMs - telemetry.fileReadyAtMs;
      const requestRoundTripMs = uploadFinishedAtMs - uploadStartedAtMs;
      const frontendOverheadMs =
        timing?.chunkTotalMs != null
          ? Math.max(0, requestRoundTripMs - timing.chunkTotalMs)
          : undefined;
      const mergeToFrameMs = firstFrameAtMs - mergeQueuedAtMs;
      const timeToVisibleMs = firstFrameAtMs - telemetry.captureStartedAtMs;

      console.log("[Live] Chunk pipeline", {
        chunk: telemetry.sequenceNumber,
        captureWindowMs,
        filePreparationMs,
        queueDelayMs,
        requestRoundTripMs,
        backendChunkMs: timing?.chunkTotalMs,
        backendInferenceMs: timing?.chunkInferenceMs,
        neuralTotalMs: timing?.neuralTotalMs,
        modelInferenceMs: timing?.modelInferenceMs,
        frontendOverheadMs,
        mergeToFrameMs,
        timeToVisibleMs,
        analysisPath: timing?.analysisPath,
        onsetThreshold: timing?.onsetThreshold,
        onsetThresholdProfile: timing?.onsetThresholdProfile,
        onsetThresholdExperiment: timing?.onsetThresholdExperiment,
        chunkRms: timing?.chunkRms,
        chunkPeak: timing?.chunkPeak,
        chunkCrestFactor: timing?.chunkCrestFactor,
        neuralError: timing?.neuralError,
        realTimeFactor: timing?.realTimeFactor,
      });
    },
    [],
  );

  const mergeChunkIntoResult = useCallback(
    (
      previous: AnalysisResult | null,
      notes: NoteResult[],
      chords: ChordResult[],
      onsets: OnsetResult[],
      bpm?: number,
      bpmConfidence?: number,
    ) => {
      return buildLiveAnalysisResult(
        [...(previous?.notes ?? []), ...notes],
        [...(previous?.chords ?? []), ...chords],
        [...(previous?.onsets ?? []), ...onsets],
        bpm,
        bpmConfidence ?? previous?.analysis_summary.tempo_confidence,
      );
    },
    [],
  );

  const {
    createSession,
    processAudioChunk,
    resetSession,
    stopPolling,
    currentBpm,
    version: liveRefinementVersion,
  } = useLiveRhythm({
    onRefinementReady: (result) => {
      setAnalysisResult((previous) => {
        const nextResult = buildLiveAnalysisResult(
          result.notes as NoteResult[],
          result.chords as ChordResult[],
          previous?.onsets ?? [],
          result.bpm,
          result.bpmConfidence,
        );
        analysisResultRef.current = nextResult;
        return nextResult;
      });
      setSessionReady(true);
      sessionReadyRef.current = true;
      setConnectionStatus("connected");
    },
  });

  const currentBpmRef = useRef(120);
  useEffect(() => {
    currentBpmRef.current = currentBpm;
  }, [currentBpm]);

  const warmBackend = useCallback(async () => {
    const response = await fetch(`${BACKEND_URL}/warmup`);
    if (!response.ok) {
      throw new Error(`Warmup failed: ${response.status}`);
    }

    await response.json();
  }, []);

  const ensureBackendWarm = useCallback(async () => {
    if (!backendWarmupPromiseRef.current) {
      backendWarmupPromiseRef.current = warmBackend().catch((error) => {
        backendWarmupPromiseRef.current = null;
        throw error;
      });
    }

    return backendWarmupPromiseRef.current;
  }, [warmBackend]);

  const removeLiveStreamAudioSubscription = useCallback(() => {
    liveStreamAudioSubscriptionRef.current?.remove?.();
    liveStreamAudioSubscriptionRef.current = null;
  }, []);

  const handleLiveScoreViewportLayout = useCallback(
    (event: LayoutChangeEvent) => {
      const nextHeight = Math.max(
        220,
        Math.floor(event.nativeEvent.layout.height),
      );
      setLiveScoreViewportHeight((previous) =>
        Math.abs(previous - nextHeight) < 8 ? previous : nextHeight,
      );
    },
    [],
  );

  const flushLivePreviewResult = useCallback(() => {
    if (livePreviewFlushTimeoutRef.current) {
      clearTimeout(livePreviewFlushTimeoutRef.current);
      livePreviewFlushTimeoutRef.current = null;
    }

    const nextResult = pendingLivePreviewRef.current;
    const queuedAtMs = pendingLivePreviewQueuedAtRef.current;
    const previewAudioTimeMs = pendingLivePreviewAudioTimeMsRef.current;
    pendingLivePreviewRef.current = null;
    pendingLivePreviewQueuedAtRef.current = null;
    pendingLivePreviewAudioTimeMsRef.current = null;
    if (!nextResult) {
      return;
    }

    const nowMs = Date.now();
    lastLivePreviewFlushAtRef.current = nowMs;
    livePreviewFlushCountRef.current += 1;
    if (queuedAtMs != null) {
      const queueWaitMs = nowMs - queuedAtMs;
      lastLivePreviewQueueWaitMsRef.current = queueWaitMs;
      maxLivePreviewQueueWaitMsRef.current = Math.max(
        maxLivePreviewQueueWaitMsRef.current,
        queueWaitMs,
      );
    }
    if (
      previewAudioTimeMs != null &&
      liveStreamFirstPacketSentAtRef.current != null
    ) {
      lastLivePreviewDataLagMsRef.current = Math.max(
        0,
        nowMs - liveStreamFirstPacketSentAtRef.current - previewAudioTimeMs,
      );
    }
    setLivePreviewResult(nextResult);
  }, []);

  const queueLivePreviewResult = useCallback(
    (nextResult: LivePreviewStripResult) => {
      if (pendingLivePreviewRef.current) {
        droppedLivePreviewUpdatesRef.current += 1;
      }
      pendingLivePreviewRef.current = nextResult;
      const nowMs = Date.now();
      pendingLivePreviewQueuedAtRef.current = nowMs;
      const previewAudioTimeMs =
        (nextResult.analysis_summary.duration_seconds ?? 0) * 1000;
      pendingLivePreviewAudioTimeMsRef.current = previewAudioTimeMs;
      const elapsedMs = nowMs - lastLivePreviewFlushAtRef.current;
      const firstPacketSentAtMs = liveStreamFirstPacketSentAtRef.current;
      const previewDataLagMs =
        firstPacketSentAtMs != null
          ? Math.max(0, nowMs - firstPacketSentAtMs - previewAudioTimeMs)
          : 0;

      if (
        elapsedMs >= LIVE_PREVIEW_BATCH_MS ||
        previewDataLagMs >= LIVE_PREVIEW_STALE_FLUSH_MS
      ) {
        flushLivePreviewResult();
        return;
      }

      coalescedLivePreviewUpdatesRef.current += 1;
      if (!livePreviewFlushTimeoutRef.current) {
        livePreviewFlushTimeoutRef.current = setTimeout(
          flushLivePreviewResult,
          Math.max(1, LIVE_PREVIEW_BATCH_MS - elapsedMs),
        );
      }
    },
    [flushLivePreviewResult],
  );

  const flushLiveStreamAnalysisResult = useCallback(() => {
    if (liveStreamAnalysisFlushTimeoutRef.current) {
      clearTimeout(liveStreamAnalysisFlushTimeoutRef.current);
      liveStreamAnalysisFlushTimeoutRef.current = null;
    }

    const pendingUpdate = pendingLiveStreamAnalysisUpdateRef.current;
    pendingLiveStreamAnalysisUpdateRef.current = null;
    if (!pendingUpdate) {
      return;
    }

    lastLiveStreamAnalysisFlushAtRef.current = Date.now();
    const nextResult = buildLiveStreamAnalysisResult(
      pendingUpdate,
      currentBpmRef.current || 120,
      undefined,
      liveStreamNotePayloadsRef.current,
    );
    const nextSignature = buildAnalysisResultEventSignature(nextResult);
    if (nextSignature !== liveStreamAnalysisSignatureRef.current) {
      liveStreamAnalysisSignatureRef.current = nextSignature;
      analysisResultRef.current = nextResult;
      setAnalysisResult(nextResult);
    }
  }, []);

  const queueLiveStreamAnalysisUpdate = useCallback(
    (update: LiveStreamUpdate) => {
      pendingLiveStreamAnalysisUpdateRef.current = update;
      const elapsedMs = Date.now() - lastLiveStreamAnalysisFlushAtRef.current;
      if (elapsedMs >= LIVE_STREAM_ANALYSIS_BATCH_MS) {
        flushLiveStreamAnalysisResult();
        return;
      }

      if (!liveStreamAnalysisFlushTimeoutRef.current) {
        liveStreamAnalysisFlushTimeoutRef.current = setTimeout(
          flushLiveStreamAnalysisResult,
          Math.max(1, LIVE_STREAM_ANALYSIS_BATCH_MS - elapsedMs),
        );
      }
    },
    [flushLiveStreamAnalysisResult],
  );

  const flushLiveEngraving = useCallback(() => {
    engravingFlushTimeoutRef.current = null;
    const pending = pendingEngravingRef.current;
    setLiveEngravingResult(pending.result);
    setLiveEngravingVersion(pending.version);
  }, []);

  const queueLiveEngraving = useCallback(
    (result: AnalysisResult | null, version: number) => {
      pendingEngravingRef.current = { result, version };
      if (engravingFlushTimeoutRef.current) {
        return;
      }

      engravingFlushTimeoutRef.current = setTimeout(() => {
        flushLiveEngraving();
      }, LIVE_OSMD_BATCH_MS);
    },
    [flushLiveEngraving],
  );

  const handleLiveStreamPayload = useCallback(
    (data: LiveStreamUpdate) => {
      if (data.type === "live_stream_started") {
        setSessionReady(true);
        sessionReadyRef.current = true;
        setConnectionStatus("connected");
        liveStreamStartResolveRef.current?.();
        liveStreamStartResolveRef.current = null;
        liveStreamStartRejectRef.current = null;
        return;
      }

      if (data.type === "live_stream_warmed") {
        console.log("[LiveStream] warmed", {
          warmupStatus: data.warmup?.status,
          warmupInferenceMs: data.warmup?.inference_ms,
          warmupNeuralTotalMs: data.warmup?.neural_timing?.neural_total,
          warmupModelTotalMs: data.warmup?.neural_timing?.neural_model_total,
          backendInferenceIntervalMs:
            data.session?.inference_interval_sec != null
              ? Math.round(data.session.inference_interval_sec * 1000)
              : undefined,
          backendContextMs:
            data.session?.context_sec != null
              ? Math.round(data.session.context_sec * 1000)
              : undefined,
          warmupError: data.warmup?.error,
        });
        if (data.warmup?.error) {
          liveStreamWarmRejectRef.current?.(
            new Error(String(data.warmup.error)),
          );
        } else {
          liveStreamWarmResolveRef.current?.();
        }
        liveStreamWarmResolveRef.current = null;
        liveStreamWarmRejectRef.current = null;
        return;
      }

      if (data.type === "live_stream_update") {
        if (data.inference?.ran) {
          const hypothesisUpdate = data.inference.hypothesis_update;
          const continuityFilter = data.inference.continuity_filter;
          liveDebugBirthsRef.current += hypothesisUpdate?.created ?? 0;
          liveDebugMatchedRef.current += hypothesisUpdate?.matched ?? 0;
          liveDebugStaleSkippedRef.current +=
            hypothesisUpdate?.stale_skipped ?? 0;
          liveDebugSuppressedRef.current += continuityFilter?.suppressed ?? 0;
          liveDebugPromotedActiveRef.current +=
            hypothesisUpdate?.promoted_active ?? 0;
          liveDebugPromotedCommittedRef.current +=
            hypothesisUpdate?.promoted_committed ?? 0;
          liveDebugPromotedLockedRef.current +=
            hypothesisUpdate?.promoted_locked ?? 0;
          if (hypothesisUpdate?.birth_samples?.length) {
            liveDebugBirthSamplesRef.current = [
              ...liveDebugBirthSamplesRef.current,
              ...hypothesisUpdate.birth_samples,
            ].slice(-12);
          }
          if (continuityFilter?.suppressed_samples?.length) {
            liveDebugSuppressedSamplesRef.current = [
              ...liveDebugSuppressedSamplesRef.current,
              ...continuityFilter.suppressed_samples,
            ].slice(-12);
          }

          const nowMs = Date.now();
          const backendAudioTimeMs =
            ((data.session?.audio_time_sec ??
              data.session?.current_time_sec ??
              0) ||
              0) * 1000;
          const recordingStartedAtMs = liveStreamRecordingStartedAtRef.current;
          if (
            recordingStartedAtMs != null &&
            nowMs - lastLiveStreamLatencyLogAtRef.current >= 1000
          ) {
            lastLiveStreamLatencyLogAtRef.current = nowMs;
            const wallElapsedMs = nowMs - recordingStartedAtMs;
            const firstPacketSentAtMs = liveStreamFirstPacketSentAtRef.current;
            const packetElapsedMs =
              firstPacketSentAtMs != null
                ? nowMs - firstPacketSentAtMs
                : undefined;
            const audioBacklogMs = Math.max(
              0,
              wallElapsedMs - backendAudioTimeMs,
            );
            const packetAudioBacklogMs =
              packetElapsedMs != null
                ? Math.max(0, packetElapsedMs - backendAudioTimeMs)
                : undefined;
            const neuralTiming = data.inference.neural_timing;
            const coalescedPreviewUpdates =
              coalescedLivePreviewUpdatesRef.current;
            const droppedPreviewUpdates = droppedLivePreviewUpdatesRef.current;
            const previewFlushes = livePreviewFlushCountRef.current;
            const previewLastQueueWaitMs =
              lastLivePreviewQueueWaitMsRef.current;
            const previewMaxQueueWaitMs = maxLivePreviewQueueWaitMsRef.current;
            const previewDataLagMs = lastLivePreviewDataLagMsRef.current;
            const noteBirths = liveDebugBirthsRef.current;
            const noteMatches = liveDebugMatchedRef.current;
            const noteStaleSkipped = liveDebugStaleSkippedRef.current;
            const noteSuppressed = liveDebugSuppressedRef.current;
            const notePromotedActive = liveDebugPromotedActiveRef.current;
            const notePromotedCommitted = liveDebugPromotedCommittedRef.current;
            const notePromotedLocked = liveDebugPromotedLockedRef.current;
            const noteBirthSamples = liveDebugBirthSamplesRef.current;
            const noteSuppressedSamples = liveDebugSuppressedSamplesRef.current;
            coalescedLivePreviewUpdatesRef.current = 0;
            droppedLivePreviewUpdatesRef.current = 0;
            livePreviewFlushCountRef.current = 0;
            maxLivePreviewQueueWaitMsRef.current = 0;
            liveDebugBirthsRef.current = 0;
            liveDebugMatchedRef.current = 0;
            liveDebugStaleSkippedRef.current = 0;
            liveDebugSuppressedRef.current = 0;
            liveDebugPromotedActiveRef.current = 0;
            liveDebugPromotedCommittedRef.current = 0;
            liveDebugPromotedLockedRef.current = 0;
            liveDebugBirthSamplesRef.current = [];
            liveDebugSuppressedSamplesRef.current = [];
            console.log("[LiveStream] latency", {
              wallElapsedMs,
              packetElapsedMs:
                packetElapsedMs != null
                  ? Math.round(packetElapsedMs)
                  : undefined,
              backendAudioTimeMs: Math.round(backendAudioTimeMs),
              audioBacklogMs: Math.round(audioBacklogMs),
              packetAudioBacklogMs:
                packetAudioBacklogMs != null
                  ? Math.round(packetAudioBacklogMs)
                  : undefined,
              serverBacklogMs:
                data.session?.stream_backlog_sec != null
                  ? Math.round(data.session.stream_backlog_sec * 1000)
                  : undefined,
              inferenceMs: data.inference.inference_ms,
              neuralTotalMs: neuralTiming?.neural_total,
              neuralRtf: neuralTiming?.neural_real_time_factor,
              modelTotalMs: neuralTiming?.neural_model_total,
              modelRtf: neuralTiming?.neural_model_real_time_factor,
              observations: data.inference.observation_count,
              receivedPackets: data.inference.received_packet_count,
              skippedInferences: data.inference.skipped_inference_count,
              coalescedPreviewUpdates,
              droppedPreviewUpdates,
              previewFlushes,
              previewLastQueueWaitMs:
                previewLastQueueWaitMs != null
                  ? Math.round(previewLastQueueWaitMs)
                  : undefined,
              previewMaxQueueWaitMs: Math.round(previewMaxQueueWaitMs),
              previewDataLagMs:
                previewDataLagMs != null
                  ? Math.round(previewDataLagMs)
                  : undefined,
              requestedInferenceIntervalMs: LIVE_STREAM_INFERENCE_INTERVAL_MS,
              backendInferenceIntervalMs:
                data.session?.inference_interval_sec != null
                  ? Math.round(data.session.inference_interval_sec * 1000)
                  : undefined,
              backendContextMs:
                data.session?.context_sec != null
                  ? Math.round(data.session.context_sec * 1000)
                  : undefined,
              transportMode: data.session?.transport_mode,
              bufferedSec: data.session?.buffered_sec,
              onsetThreshold:
                data.inference.analysis_summary?.live_onset_threshold,
              onsetProfile:
                data.inference.analysis_summary?.live_onset_threshold_profile,
              continuitySuppressed: continuityFilter?.suppressed,
              continuitySamePitchBoundary:
                continuityFilter?.same_pitch_boundary,
              continuityImplausibleRepeat: continuityFilter?.implausible_repeat,
              continuityHarmonicSustain: continuityFilter?.harmonic_sustain,
              continuityWeakBirthOutsideAttack:
                continuityFilter?.weak_birth_outside_attack,
              continuityAttackGroups: continuityFilter?.attack_groups,
              continuityRegisteredAttackGroups:
                continuityFilter?.registered_attack_groups,
              continuityTotalSuppressed: continuityFilter?.total_suppressed,
              noteBirths,
              noteMatches,
              noteStaleSkipped,
              noteSuppressed,
              notePromotedActive,
              notePromotedCommitted,
              notePromotedLocked,
              noteBirthSamples,
              noteSuppressedSamples,
              liveCounts: data.counts,
            });
          }

          mergeLiveStreamAnalysisPayloads(
            data,
            liveStreamNotePayloadsRef.current,
          );
          const nextPreviewResult = buildLiveStreamPreviewResult(
            data,
            currentBpmRef.current || 120,
            liveStreamPreviewPayloadsRef.current,
          );
          queueLivePreviewResult(nextPreviewResult);
          queueLiveStreamAnalysisUpdate(data);
        }
        setConnectionStatus("connected");
        setIsProcessing(false);
        return;
      }

      if (data.type === "live_stream_stopped") {
        flushLiveStreamAnalysisResult();
        liveStreamStopResolveRef.current?.();
        liveStreamStopResolveRef.current = null;
        setIsProcessing(false);
        return;
      }

      if (data.type === "live_stream_error") {
        const error = new Error(data.error || "Live stream failed");
        if (
          liveStreamWarmResolveRef.current &&
          String(data.error || "").includes("Unsupported message type: warmup")
        ) {
          console.warn(
            "[LiveStream] Live-path warmup is not supported by this backend deployment; continuing without it.",
          );
          liveStreamWarmResolveRef.current();
          liveStreamWarmResolveRef.current = null;
          liveStreamWarmRejectRef.current = null;
          setConnectionStatus("connected");
          setIsProcessing(false);
          return;
        }
        liveStreamStartRejectRef.current?.(error);
        liveStreamStartResolveRef.current = null;
        liveStreamStartRejectRef.current = null;
        liveStreamWarmRejectRef.current?.(error);
        liveStreamWarmResolveRef.current = null;
        liveStreamWarmRejectRef.current = null;
        liveStreamStopResolveRef.current?.();
        liveStreamStopResolveRef.current = null;
        setConnectionStatus("error");
        setIsProcessing(false);
      }
    },
    [
      flushLiveStreamAnalysisResult,
      queueLivePreviewResult,
      queueLiveStreamAnalysisUpdate,
    ],
  );

  const openLiveStreamSocket = useCallback(
    async (streamSessionId: string) => {
      const existingSocket = liveStreamSocketRef.current;
      if (
        existingSocket &&
        existingSocket.readyState !== WebSocket.CLOSED &&
        existingSocket.readyState !== WebSocket.CLOSING
      ) {
        existingSocket.close();
      }

      const socket = new WebSocket(getLiveStreamUrl());
      liveStreamSocketRef.current = socket;
      liveStreamSessionIdRef.current = streamSessionId;
      liveStreamFirstPacketSentAtRef.current = null;
      liveStreamNotePayloadsRef.current = new Map();
      liveStreamPreviewPayloadsRef.current = new Map();
      liveStreamAnalysisSignatureRef.current = "";
      liveDebugBirthsRef.current = 0;
      liveDebugMatchedRef.current = 0;
      liveDebugStaleSkippedRef.current = 0;
      liveDebugSuppressedRef.current = 0;
      liveDebugPromotedActiveRef.current = 0;
      liveDebugPromotedCommittedRef.current = 0;
      liveDebugPromotedLockedRef.current = 0;
      liveDebugBirthSamplesRef.current = [];
      liveDebugSuppressedSamplesRef.current = [];
      clearLiveStreamAnalysisQueue();
      clearLivePreviewQueue();
      setLivePreviewResult(null);

      await withTimeout(
        new Promise<void>((resolve, reject) => {
          liveStreamStartResolveRef.current = resolve;
          liveStreamStartRejectRef.current = reject;

          socket.onopen = () => {
            socket.send(
              JSON.stringify({
                type: "start",
                session_id: streamSessionId,
                sample_rate: LIVE_AUDIO_SAMPLE_RATE,
                context_sec: LIVE_STREAM_CONTEXT_SEC,
                inference_interval_ms: LIVE_STREAM_INFERENCE_INTERVAL_MS,
                trusted_delay_ms: LIVE_STREAM_TRUSTED_DELAY_MS,
                commit_delay_ms: LIVE_STREAM_COMMIT_DELAY_MS,
                lock_delay_ms: LIVE_STREAM_LOCK_DELAY_MS,
              }),
            );
          };

          socket.onmessage = (event) => {
            try {
              handleLiveStreamPayload(JSON.parse(String(event.data)));
            } catch (error) {
              console.warn("[LiveStream] Could not parse message", error);
            }
          };

          socket.onerror = () => {
            reject(new Error("Live stream socket error"));
          };

          socket.onclose = () => {
            liveStreamSocketRef.current = null;
            liveStreamStartResolveRef.current = null;
            liveStreamStartRejectRef.current = null;
            liveStreamWarmResolveRef.current = null;
            liveStreamWarmRejectRef.current = null;
            liveStreamStopResolveRef.current?.();
            liveStreamStopResolveRef.current = null;
          };
        }),
        8000,
      );

      if (socket.readyState === WebSocket.OPEN) {
        await withTimeout(
          new Promise<void>((resolve, reject) => {
            liveStreamWarmResolveRef.current = resolve;
            liveStreamWarmRejectRef.current = reject;
            socket.send(
              JSON.stringify({
                type: "warmup",
                session_id: streamSessionId,
              }),
            );
          }),
          10000,
        ).catch((error) => {
          console.warn("[LiveStream] Live-path warmup failed", error);
          liveStreamWarmResolveRef.current = null;
          liveStreamWarmRejectRef.current = null;
        });
      }
    },
    [
      clearLivePreviewQueue,
      clearLiveStreamAnalysisQueue,
      handleLiveStreamPayload,
    ],
  );

  const installLiveStreamAudioSubscription = useCallback(() => {
    removeLiveStreamAudioSubscription();
    liveStreamPacketSequenceRef.current = 0;
    const subscription = AudioRecord.on("data", (pcm16Base64: string) => {
      if (!isRecordingRef.current) {
        return;
      }

      const socket = liveStreamSocketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }

      liveStreamPacketSequenceRef.current += 1;
      const packetSentAtMs = Date.now();
      if (liveStreamFirstPacketSentAtRef.current == null) {
        liveStreamFirstPacketSentAtRef.current = packetSentAtMs;
      }
      socket.send(
        JSON.stringify({
          type: "audio_packet",
          session_id: liveStreamSessionIdRef.current,
          sample_rate: LIVE_AUDIO_SAMPLE_RATE,
          encoding: "pcm16",
          pcm16_base64: pcm16Base64,
          sequence_number: liveStreamPacketSequenceRef.current,
          client_sent_at_ms: packetSentAtMs,
        }),
      );
    }) as unknown as { remove?: () => void };
    liveStreamAudioSubscriptionRef.current = subscription;
  }, [removeLiveStreamAudioSubscription]);

  const stopLiveStreamSocket = useCallback(async () => {
    const socket = liveStreamSocketRef.current;
    if (!socket) {
      return;
    }

    if (socket.readyState === WebSocket.OPEN) {
      await withTimeout(
        new Promise<void>((resolve) => {
          liveStreamStopResolveRef.current = resolve;
          socket.send(
            JSON.stringify({
              type: "stop",
              session_id: liveStreamSessionIdRef.current,
            }),
          );
        }),
        2500,
      ).catch((error) => {
        console.warn("[LiveStream] Timed out waiting for stream stop", error);
      });
    }

    socket.close();
    liveStreamSocketRef.current = null;
    liveStreamSessionIdRef.current = null;
    liveStreamWarmResolveRef.current = null;
    liveStreamWarmRejectRef.current = null;
    liveStreamStopResolveRef.current = null;
  }, []);

  useEffect(() => {
    const audioOptions = {
      sampleRate: LIVE_AUDIO_SAMPLE_RATE,
      channels: 1,
      bitsPerSample: 16,
      audioSource: 6,
      wavFile: "temp_audio.wav",
    };

    AudioRecord.init(audioOptions);
    backendWarmupPromiseRef.current = warmBackend().catch((error) => {
      console.warn("Backend warmup failed", error);
      backendWarmupPromiseRef.current = null;
    });

    return () => {
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
      if (chunkTimeoutRef.current) {
        clearTimeout(chunkTimeoutRef.current);
      }
      if (engravingFlushTimeoutRef.current) {
        clearTimeout(engravingFlushTimeoutRef.current);
      }
      clearLiveStreamAnalysisQueue();
      clearLivePreviewQueue();
      clearPendingChunkUpload();
      removeLiveStreamAudioSubscription();
      liveStreamSocketRef.current?.close();
      liveStreamSocketRef.current = null;
      liveStreamSessionIdRef.current = null;
      liveStreamStartResolveRef.current = null;
      liveStreamStartRejectRef.current = null;
      liveStreamWarmResolveRef.current = null;
      liveStreamWarmRejectRef.current = null;
      liveStreamStopResolveRef.current = null;
      liveStreamRecordingStartedAtRef.current = null;
      liveStreamFirstPacketSentAtRef.current = null;
      analysisResultRef.current = null;
      lastLiveStreamLatencyLogAtRef.current = 0;
      liveDebugBirthsRef.current = 0;
      liveDebugMatchedRef.current = 0;
      liveDebugStaleSkippedRef.current = 0;
      liveDebugSuppressedRef.current = 0;
      liveDebugPromotedActiveRef.current = 0;
      liveDebugPromotedCommittedRef.current = 0;
      liveDebugPromotedLockedRef.current = 0;
      liveDebugBirthSamplesRef.current = [];
      liveDebugSuppressedSamplesRef.current = [];
      recordingTimerStartedAtRef.current = null;
      currentChunkStartedAtRef.current = null;
      chunkSequenceRef.current = 0;
      try {
        AudioRecord.stop();
      } catch {
        // Ignore if already stopped.
      }
      sessionReadyRef.current = false;
      void resetSession();
    };
  }, [
    clearPendingChunkUpload,
    clearLiveStreamAnalysisQueue,
    clearLivePreviewQueue,
    removeLiveStreamAudioSubscription,
    resetSession,
    warmBackend,
  ]);

  useEffect(() => {
    if (!USE_LIVE_OSMD_ENGRAVING_EXPERIMENT) {
      return;
    }

    queueLiveEngraving(analysisResult, liveRefinementVersion);
  }, [analysisResult, liveRefinementVersion, queueLiveEngraving]);

  const requestPermissions = useCallback(async () => {
    if (Platform.OS === "android") {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: "Microphone Permission",
            message:
              "This app needs access to your microphone to transcribe live audio.",
            buttonNeutral: "Ask Me Later",
            buttonNegative: "Cancel",
            buttonPositive: "OK",
          },
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch {
        return false;
      }
    }

    return true;
  }, []);

  const resolveRecordedFilePath = useCallback(async (rawPath: string) => {
    const candidates = new Set<string>();
    const fileName = rawPath.split("/").pop() ?? rawPath;

    candidates.add(rawPath);
    if (!rawPath.startsWith("file://")) {
      candidates.add(`file://${rawPath}`);
    }

    for (const baseDir of [
      FileSystem.documentDirectory,
      FileSystem.cacheDirectory,
    ]) {
      if (!baseDir) {
        continue;
      }
      candidates.add(`${baseDir}${fileName}`);
      candidates.add(`${baseDir}temp_audio.wav`);
    }

    for (const candidate of candidates) {
      try {
        const info = await FileSystem.getInfoAsync(candidate);
        if (info.exists) {
          return candidate;
        }
      } catch {
        // Try the next candidate.
      }
    }

    return null;
  }, []);

  const processRecordedChunk = useCallback(
    async (resolvedPath: string, telemetry: RecordedChunkTelemetry) => {
      inFlightUploadsRef.current += 1;
      setIsProcessing(true);
      const uploadStartedAtMs = Date.now();
      try {
        if (!sessionReadyRef.current) {
          throw new Error("Live session is not ready yet");
        }

        const chunk = await processAudioChunk(resolvedPath, {
          noiseProfile,
          useNeuralLive: USE_LIVE_NEURAL_PATH,
          useAdaptiveOnsetThreshold:
            USE_LIVE_ADAPTIVE_ONSET_THRESHOLD_EXPERIMENT,
        });
        const uploadFinishedAtMs = Date.now();
        const mergeQueuedAtMs = Date.now();
        setAnalysisResult((previous) => {
          const nextResult = mergeChunkIntoResult(
            previous,
            chunk.coarseNotes as NoteResult[],
            chunk.coarseChords as ChordResult[],
            chunk.onsets as OnsetResult[],
            chunk.bpm,
          );
          analysisResultRef.current = nextResult;
          return nextResult;
        });
        requestAnimationFrame(() => {
          logChunkPipelineTiming(
            telemetry,
            uploadStartedAtMs,
            uploadFinishedAtMs,
            mergeQueuedAtMs,
            Date.now(),
            chunk.timing,
          );
        });
        setConnectionStatus("connected");
      } finally {
        inFlightUploadsRef.current = Math.max(
          0,
          inFlightUploadsRef.current - 1,
        );
        if (inFlightUploadsRef.current === 0) {
          setIsProcessing(false);
        }
      }
    },
    [
      logChunkPipelineTiming,
      mergeChunkIntoResult,
      noiseProfile,
      processAudioChunk,
    ],
  );

  const drainPendingChunkUploads = useCallback(() => {
    if (inFlightUploadsRef.current > 0) {
      return;
    }

    const nextUpload = pendingChunkUploadRef.current;
    if (!nextUpload) {
      return;
    }

    pendingChunkUploadRef.current = null;

    void (async () => {
      try {
        await processRecordedChunk(nextUpload.path, nextUpload.telemetry);
      } catch (error) {
        console.error("Live chunk analysis failed", error);
        setConnectionStatus("error");
      } finally {
        FileSystem.deleteAsync(nextUpload.path, { idempotent: true }).catch(
          () => {},
        );
        drainPendingChunkUploads();
      }
    })();
  }, [processRecordedChunk]);

  const waitForChunkQueueToDrain = useCallback(async (timeoutMs = 4000) => {
    if (
      inFlightUploadsRef.current === 0 &&
      pendingChunkUploadRef.current == null
    ) {
      return;
    }

    await withTimeout(
      new Promise<void>((resolve) => {
        const pollQueue = () => {
          if (
            inFlightUploadsRef.current === 0 &&
            pendingChunkUploadRef.current == null
          ) {
            resolve();
            return;
          }

          setTimeout(pollQueue, 50);
        };

        pollQueue();
      }),
      timeoutMs,
    ).catch((error) => {
      console.warn("[Live] Timed out waiting for chunk queue to drain", {
        error,
        inFlightUploads: inFlightUploadsRef.current,
        hasPendingChunk: pendingChunkUploadRef.current != null,
      });
    });
  }, []);

  const enqueueRecordedChunkUpload = useCallback(
    (path: string, telemetry: RecordedChunkTelemetry) => {
      const existingPending = pendingChunkUploadRef.current;
      if (existingPending) {
        pendingChunkUploadRef.current = { path, telemetry };
        console.warn(
          "[Live] Chunk backlog detected; replacing older pending chunk",
          {
            droppedChunk: existingPending.telemetry.sequenceNumber,
            nextChunk: telemetry.sequenceNumber,
            inFlightUploads: inFlightUploadsRef.current,
          },
        );
        FileSystem.deleteAsync(existingPending.path, {
          idempotent: true,
        }).catch(() => {});
        return;
      }

      pendingChunkUploadRef.current = { path, telemetry };

      if (inFlightUploadsRef.current > 0) {
        console.warn(
          "[Live] Chunk backlog detected; queued one pending chunk",
          {
            queuedChunk: telemetry.sequenceNumber,
            inFlightUploads: inFlightUploadsRef.current,
          },
        );
        return;
      }

      drainPendingChunkUploads();
    },
    [drainPendingChunkUploads],
  );

  // Stops the current short recording, swaps the WAV file out of the way,
  // restarts capture immediately so the mic stays open, then uploads the
  // captured chunk in the background.
  const analyzeRecordingChunk = useCallback(async () => {
    if (!isRecordingRef.current) {
      return;
    }

    let chunkPath: string | null = null;
    const captureStoppedAtMs = Date.now();
    const captureStartedAtMs =
      currentChunkStartedAtRef.current ??
      captureStoppedAtMs - CHUNK_INTERVAL_MS;
    currentChunkStartedAtRef.current = null;
    try {
      const audioFile = await AudioRecord.stop();
      const resolved = await resolveRecordedFilePath(audioFile);
      if (!resolved) {
        throw new Error("Recorded chunk could not be located on disk");
      }
      // The recorder reuses the same wav filename, so the next start() would
      // clobber this file. Move it to a unique cache path before restarting.
      const target = `${FileSystem.cacheDirectory ?? ""}live_chunk_${Date.now()}.wav`;
      try {
        await FileSystem.moveAsync({ from: resolved, to: target });
        chunkPath = target;
      } catch {
        try {
          await FileSystem.copyAsync({ from: resolved, to: target });
          chunkPath = target;
        } catch (copyError) {
          console.warn(
            "Could not relocate live chunk; uploading from original path",
            copyError,
          );
          chunkPath = resolved;
        }
      }
    } catch (error) {
      console.error("Failed to capture chunk", error);
      setConnectionStatus("error");
    }

    // Restart the mic ASAP — do NOT wait for the upload below.
    if (isRecordingRef.current) {
      try {
        currentChunkStartedAtRef.current = Date.now();
        AudioRecord.start();
        chunkTimeoutRef.current = setTimeout(
          analyzeRecordingChunk,
          CHUNK_INTERVAL_MS,
        );
      } catch (error) {
        console.error("Failed to restart chunk recording", error);
        setConnectionStatus("error");
      }
    }

    if (chunkPath) {
      const telemetry: RecordedChunkTelemetry = {
        sequenceNumber: chunkSequenceRef.current + 1,
        captureStartedAtMs,
        captureStoppedAtMs,
        fileReadyAtMs: Date.now(),
      };
      chunkSequenceRef.current = telemetry.sequenceNumber;
      enqueueRecordedChunkUpload(chunkPath, telemetry);
    }
  }, [enqueueRecordedChunkUpload, resolveRecordedFilePath]);

  const startLiveTranscription = useCallback(async () => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        Alert.alert(
          "Permission Required",
          "Please grant microphone permissions to use live transcription.",
        );
        return;
      }

      setIsWarmingUp(true);
      setConnectionStatus("connecting");
      setDuration(0);
      setRecordingStartedAtMs(null);
      setIsRecording(false);
      isRecordingRef.current = false;
      recordingTimerStartedAtRef.current = null;
      setSessionReady(false);
      sessionReadyRef.current = false;
      setHasStoppedRecording(false);
      inFlightUploadsRef.current = 0;
      clearPendingChunkUpload();
      currentChunkStartedAtRef.current = null;
      chunkSequenceRef.current = 0;

      await resetSession();

      analysisResultRef.current = null;
      setAnalysisResult(null);
      setLiveEngravingResult(null);
      clearLiveStreamAnalysisQueue();
      clearLivePreviewQueue();
      liveStreamPreviewPayloadsRef.current = new Map();
      setLivePreviewResult(null);
      setLiveEngravingVersion(0);
      pendingEngravingRef.current = { result: null, version: 0 };
      if (engravingFlushTimeoutRef.current) {
        clearTimeout(engravingFlushTimeoutRef.current);
        engravingFlushTimeoutRef.current = null;
      }

      let sessionStartError: unknown = null;
      const sessionPromise = withTimeout(
        createSession(currentBpm || 120),
        8000,
      ).catch((error) => {
        sessionStartError = error;
      });
      try {
        await withTimeout(ensureBackendWarm(), 30000);
      } catch (error) {
        console.warn("Backend warmup timed out or failed", error);
      }

      await sessionPromise;
      if (sessionStartError) {
        throw sessionStartError;
      }

      if (USE_LIVE_STREAM_TRANSPORT) {
        const streamSessionId = `stream_${Date.now()}_${Math.random()
          .toString(36)
          .slice(2, 10)}`;
        await openLiveStreamSocket(streamSessionId);
        installLiveStreamAudioSubscription();
        const startedAtMs = Date.now();
        isRecordingRef.current = true;
        recordingTimerStartedAtRef.current = startedAtMs;
        setRecordingStartedAtMs(startedAtMs);
        liveStreamRecordingStartedAtRef.current = startedAtMs;
        liveStreamFirstPacketSentAtRef.current = null;
        lastLiveStreamLatencyLogAtRef.current = 0;
        AudioRecord.start();
      } else {
        const startedAtMs = Date.now();
        recordingTimerStartedAtRef.current = startedAtMs;
        setRecordingStartedAtMs(startedAtMs);
        currentChunkStartedAtRef.current = startedAtMs;
        AudioRecord.start();
        isRecordingRef.current = true;
        chunkTimeoutRef.current = setTimeout(
          analyzeRecordingChunk,
          CHUNK_INTERVAL_MS,
        );
      }
      setIsWarmingUp(false);
      setIsRecording(true);
      setSessionReady(true);
      sessionReadyRef.current = true;
      setConnectionStatus("connected");
      durationIntervalRef.current = setInterval(() => {
        const startedAtMs = recordingTimerStartedAtRef.current;
        if (startedAtMs != null) {
          setDuration(Math.max(0, (Date.now() - startedAtMs) / 1000));
        }
      }, 500);
    } catch (error) {
      console.error("Failed to start live transcription", error);
      removeLiveStreamAudioSubscription();
      await stopLiveStreamSocket();
      try {
        AudioRecord.stop();
      } catch {
        // Ignore if audio recording never started.
      }
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
        durationIntervalRef.current = null;
      }
      if (chunkTimeoutRef.current) {
        clearTimeout(chunkTimeoutRef.current);
        chunkTimeoutRef.current = null;
      }
      setIsWarmingUp(false);
      setConnectionStatus("error");
      setSessionReady(false);
      sessionReadyRef.current = false;
      setDuration(0);
      setRecordingStartedAtMs(null);
      setIsRecording(false);
      isRecordingRef.current = false;
      recordingTimerStartedAtRef.current = null;
      liveStreamRecordingStartedAtRef.current = null;
      liveStreamFirstPacketSentAtRef.current = null;
      lastLiveStreamLatencyLogAtRef.current = 0;
      clearLivePreviewQueue();
      Alert.alert(
        "Live Session Error",
        "The app could not create the live backend session. Recording was not started.",
      );
    }
  }, [
    analyzeRecordingChunk,
    clearPendingChunkUpload,
    clearLiveStreamAnalysisQueue,
    clearLivePreviewQueue,
    createSession,
    currentBpm,
    ensureBackendWarm,
    installLiveStreamAudioSubscription,
    openLiveStreamSocket,
    requestPermissions,
    resetSession,
    removeLiveStreamAudioSubscription,
    stopLiveStreamSocket,
  ]);

  const stopLiveTranscription = useCallback(async () => {
    const wasRecording = isRecordingRef.current;
    isRecordingRef.current = false;
    setIsRecording(false);
    setRecordingStartedAtMs(null);
    recordingTimerStartedAtRef.current = null;
    liveStreamRecordingStartedAtRef.current = null;
    liveStreamFirstPacketSentAtRef.current = null;
    lastLiveStreamLatencyLogAtRef.current = 0;
    clearLivePreviewQueue();

    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }
    if (chunkTimeoutRef.current) {
      clearTimeout(chunkTimeoutRef.current);
      chunkTimeoutRef.current = null;
    }

    try {
      if (wasRecording) {
        if (USE_LIVE_STREAM_TRANSPORT) {
          removeLiveStreamAudioSubscription();
          try {
            const finalAudioFile = await AudioRecord.stop();
            const resolved = await resolveRecordedFilePath(finalAudioFile);
            if (resolved) {
              FileSystem.deleteAsync(resolved, { idempotent: true }).catch(
                () => {},
              );
            }
          } catch (error) {
            console.warn("Failed to stop live audio recorder", error);
          }
          await stopLiveStreamSocket();
          flushLiveStreamAnalysisResult();
        } else {
          let finalChunkPath: string | null = null;
          const captureStoppedAtMs = Date.now();
          const captureStartedAtMs =
            currentChunkStartedAtRef.current ?? captureStoppedAtMs;
          currentChunkStartedAtRef.current = null;
          try {
            const finalAudioFile = await AudioRecord.stop();
            const resolved = await resolveRecordedFilePath(finalAudioFile);
            if (resolved) {
              const target = `${FileSystem.cacheDirectory ?? ""}live_chunk_${Date.now()}_final.wav`;
              try {
                await FileSystem.moveAsync({ from: resolved, to: target });
                finalChunkPath = target;
              } catch {
                try {
                  await FileSystem.copyAsync({ from: resolved, to: target });
                  finalChunkPath = target;
                } catch {
                  finalChunkPath = resolved;
                }
              }
            }
          } catch (error) {
            console.warn("Failed to capture final chunk", error);
          }

          if (
            inFlightUploadsRef.current > 0 ||
            pendingChunkUploadRef.current != null
          ) {
            drainPendingChunkUploads();
            await waitForChunkQueueToDrain();
          }

          if (finalChunkPath) {
            const finalTelemetry: RecordedChunkTelemetry = {
              sequenceNumber: chunkSequenceRef.current + 1,
              captureStartedAtMs,
              captureStoppedAtMs,
              fileReadyAtMs: Date.now(),
            };
            chunkSequenceRef.current = finalTelemetry.sequenceNumber;
            try {
              await processRecordedChunk(finalChunkPath, finalTelemetry);
            } catch (error) {
              console.warn("Final chunk analysis failed", error);
            } finally {
              FileSystem.deleteAsync(finalChunkPath, {
                idempotent: true,
              }).catch(() => {});
            }
          }
        }
      }

      stopPolling();

      setSessionReady(false);
      sessionReadyRef.current = false;
      setHasStoppedRecording(true);
    } catch (error) {
      console.error("Failed to stop live transcription", error);
      setSessionReady(false);
      sessionReadyRef.current = false;
      setConnectionStatus("error");
      Alert.alert("Error", "Failed to stop the live transcription session.");
    } finally {
      setConnectionStatus("disconnected");
      inFlightUploadsRef.current = 0;
      setIsProcessing(false);
    }
  }, [
    clearLivePreviewQueue,
    drainPendingChunkUploads,
    flushLiveStreamAnalysisResult,
    processRecordedChunk,
    removeLiveStreamAudioSubscription,
    resolveRecordedFilePath,
    stopPolling,
    stopLiveStreamSocket,
    waitForChunkQueueToDrain,
  ]);

  const exportMIDI = useCallback(async () => {
    if (!analysisResult || !hasExportableScore) {
      Alert.alert(
        "No Score Yet",
        "Finish generating a score before exporting MIDI.",
      );
      return;
    }

    try {
      const bpm =
        analysisResult.analysis_summary?.detected_bpm || currentBpm || 120;
      const midi = new Midi();
      midi.header.setTempo(bpm);

      const track = midi.addTrack();
      track.name = "Piano";
      track.channel = 0;

      for (const note of analysisResult.notes || []) {
        const duration =
          note.duration_seconds ??
          (note.offset_seconds != null
            ? note.offset_seconds - note.time_seconds
            : 0.25);
        track.addNote({
          midi: note.midi_note,
          time: note.time_seconds,
          duration: Math.max(duration, 0.01),
          velocity: note.confidence ?? 0.8,
        });
      }

      for (const chord of analysisResult.chords || []) {
        if (!chord.midi_notes) {
          continue;
        }

        const duration =
          chord.duration_seconds ??
          (chord.offset_seconds != null
            ? chord.offset_seconds - chord.time_seconds
            : 0.25);
        for (const pitch of chord.midi_notes) {
          track.addNote({
            midi: pitch,
            time: chord.time_seconds,
            duration: Math.max(duration, 0.01),
            velocity: chord.confidence ?? 0.8,
          });
        }
      }

      if (!FileSystem.cacheDirectory) {
        throw new Error("Cache directory is unavailable on this device.");
      }

      const fileUri = `${FileSystem.cacheDirectory}live_score.mid`;
      const midiBytes = midi.toArray();
      const binary = String.fromCharCode(...midiBytes);
      const base64 = btoa(binary);

      await FileSystem.writeAsStringAsync(fileUri, base64, {
        encoding: FileSystem.EncodingType.Base64,
      });
      await Sharing.shareAsync(fileUri, {
        mimeType: "audio/midi",
        dialogTitle: "Export MIDI",
        UTI: "public.midi-audio",
      });
    } catch (error: any) {
      Alert.alert("Export Failed", error.message || "Could not export MIDI.");
    }
  }, [analysisResult, currentBpm, hasExportableScore]);

  const renderMidiExportCard = () => (
    <LinearGradient
      colors={["rgba(15,23,42,0.96)", "rgba(30,41,59,0.92)"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={styles.exportCard}
    >
      <View style={styles.exportCardHeader}>
        <ThemedText
          style={styles.exportCardTitle}
          lightColor="#f8fafc"
          darkColor="#f8fafc"
        >
          MIDI Export
        </ThemedText>
        <ThemedText
          style={styles.exportCardHint}
          lightColor="rgba(226,232,240,0.8)"
          darkColor="rgba(226,232,240,0.8)"
        >
          {hasExportableScore
            ? `${exportableEventCount} events ready to share as a .mid file.`
            : "Generate a score first, then export it as a .mid file."}
        </ThemedText>
      </View>

      <TouchableOpacity
        style={[
          styles.exportActionButton,
          !hasExportableScore ? styles.exportActionButtonDisabled : null,
        ]}
        onPress={exportMIDI}
        disabled={!hasExportableScore}
      >
        <Ionicons name="download-outline" size={18} color="#ffffff" />
        <ThemedText style={styles.exportActionButtonText}>
          Download MIDI
        </ThemedText>
      </TouchableOpacity>
    </LinearGradient>
  );

  const recentEvents = useMemo(
    () =>
      [
        ...(analysisResult?.notes ?? []).map((note) => ({
          keyBase: `note-${note.time_seconds}-${note.note_name ?? note.midi_note ?? "unknown"}`,
          time: note.time_seconds,
          icon: "musical-note" as const,
          label: note.note_name ?? `MIDI ${note.midi_note ?? "?"}`,
          detail:
            note.confidence != null
              ? `${Math.round(note.confidence * 100)}% confidence`
              : "Detected note",
          color: "#30a46c",
        })),
        ...(analysisResult?.chords ?? []).map((chord) => ({
          keyBase: `chord-${chord.time_seconds}-${chord.label}`,
          time: chord.time_seconds,
          icon: "library" as const,
          label: chord.label,
          detail:
            chord.confidence != null
              ? `${Math.round(chord.confidence * 100)}% confidence`
              : "Detected chord",
          color: "#2563eb",
        })),
      ]
        .sort((left, right) => right.time - left.time)
        .map((event, index) => ({
          ...event,
          key: `${event.keyBase}-${index}`,
        }))
        .slice(0, 12),
    [analysisResult],
  );

  const compactScoreViewportHeight = Math.max(
    220,
    liveScoreViewportHeight || 280,
  );

  if (isLiveSessionLayout) {
    const liveStatusColor = getConnectionStatusColor(connectionStatus);

    return (
      <LinearGradient
        colors={["#04070f", "#0b1220", "#111c30"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.liveSessionScreen}
      >
        <ScrollView
          ref={scrollViewRef}
          style={styles.liveSessionScrollView}
          contentContainerStyle={styles.liveSessionScrollContent}
          showsVerticalScrollIndicator={false}
          nestedScrollEnabled
          scrollEnabled={!isScoreScrollActive}
        >
          <View style={styles.liveSessionWorkspace}>
            <View style={styles.liveTopBar}>
              <ThemedText
                style={styles.liveTopBarTitle}
                lightColor="#f8fafc"
                darkColor="#f8fafc"
                numberOfLines={1}
              >
                {USE_LIVE_OSMD_ENGRAVING_EXPERIMENT ? "Live" : "Score"}
              </ThemedText>

              <View style={styles.liveTopBarMetrics}>
                <View style={styles.liveInlineMetric}>
                  <ThemedText
                    style={styles.liveInlineMetricValue}
                    lightColor="#f8fafc"
                    darkColor="#f8fafc"
                  >
                    {formatDuration(duration)}
                  </ThemedText>
                  <ThemedText style={styles.liveInlineMetricLabel}>
                    time
                  </ThemedText>
                </View>
                <View style={styles.liveInlineMetric}>
                  <ThemedText
                    style={styles.liveInlineMetricValue}
                    lightColor="#f8fafc"
                    darkColor="#f8fafc"
                  >
                    {Math.round(currentBpm || 120)}
                  </ThemedText>
                  <ThemedText style={styles.liveInlineMetricLabel}>
                    bpm
                  </ThemedText>
                </View>
                <View style={styles.liveInlineMetric}>
                  <ThemedText
                    style={styles.liveInlineMetricValue}
                    lightColor="#f8fafc"
                    darkColor="#f8fafc"
                  >
                    {(analysisResult?.analysis_summary.total_notes ?? 0) +
                      (analysisResult?.analysis_summary.total_chords ?? 0)}
                  </ThemedText>
                  <ThemedText style={styles.liveInlineMetricLabel}>
                    events
                  </ThemedText>
                </View>
              </View>

              <View
                style={[
                  styles.liveStatusChip,
                  {
                    backgroundColor: `${liveStatusColor}1f`,
                    borderColor: `${liveStatusColor}55`,
                  },
                ]}
              >
                <View
                  style={[
                    styles.statusDot,
                    { backgroundColor: liveStatusColor },
                  ]}
                />
              </View>
            </View>

            <View
              style={styles.liveScorePane}
              onLayout={handleLiveScoreSectionLayout}
            >
              <View style={styles.liveScoreStrip}>
                <MemoizedLivePreviewStrip
                  previewResult={livePreviewResult}
                  analysisFallback={analysisResult}
                  bpm={currentBpm || 120}
                  localElapsedSeconds={duration}
                  localStartedAtMs={recordingStartedAtMs}
                  isRecording={isRecording}
                />
              </View>
              <View
                style={styles.liveScoreViewport}
                onLayout={handleLiveScoreViewportLayout}
              >
                <MemoizedScoreContent
                  analysisResult={analysisResult}
                  compact
                  hasStoppedRecording={hasStoppedRecording}
                  isRecording={isRecording}
                  isWarmingUp={isWarmingUp}
                  liveEngravingResult={liveEngravingResult}
                  liveEngravingVersion={liveEngravingVersion}
                  onScoreScrollActiveChange={setIsScoreScrollActive}
                  viewportHeight={compactScoreViewportHeight}
                />
              </View>
            </View>

            <LinearGradient
              colors={["rgba(255,255,255,0.1)", "rgba(148,163,184,0.14)"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.liveControlDock}
            >
              <View style={styles.optionRow}>
                {LIVE_NOISE_PROFILE_OPTIONS.map((option) => {
                  const isActive = option.value === noiseProfile;

                  return (
                    <TouchableOpacity
                      key={option.value}
                      style={[
                        styles.optionChip,
                        isActive ? styles.optionChipActive : null,
                      ]}
                      onPress={() => setNoiseProfile(option.value)}
                    >
                      <ThemedText
                        style={[
                          styles.optionChipText,
                          isActive ? styles.optionChipTextActive : null,
                        ]}
                      >
                        {option.label}
                      </ThemedText>
                    </TouchableOpacity>
                  );
                })}
              </View>

              <TouchableOpacity
                style={[
                  styles.recordButton,
                  styles.liveRecordButton,
                  isRecording
                    ? styles.stopButton
                    : isWarmingUp
                      ? styles.warmingButton
                      : styles.startButton,
                  isStartDisabled ? styles.recordButtonDisabled : null,
                ]}
                onPress={
                  isRecording ? stopLiveTranscription : startLiveTranscription
                }
                disabled={isStartDisabled}
              >
                {isWarmingUp ? (
                  <ActivityIndicator size="small" color="#ffffff" />
                ) : (
                  <Ionicons
                    name={isRecording ? "stop" : "radio"}
                    size={20}
                    color="white"
                  />
                )}
                <ThemedText style={styles.recordButtonText}>
                  {recordButtonLabel}
                </ThemedText>
              </TouchableOpacity>
            </LinearGradient>

            {renderMidiExportCard()}
          </View>
        </ScrollView>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient
      colors={["#f5f7fb", "#e9eef5", "#f8fafc"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={styles.screenBackground}
    >
      <ScrollView
        ref={scrollViewRef}
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        nestedScrollEnabled
        scrollEnabled={!isScoreScrollActive}
      >
        <ThemedView
          style={styles.container}
          lightColor="transparent"
          darkColor="transparent"
        >
          <View style={styles.header}>
            <ThemedText type="title" style={styles.title}>
              Live Piano Transcription
            </ThemedText>
            <ThemedText style={styles.subtitle}>
              This tab uses the live chunk pipeline. The previous
              record-then-analyze screen is available in the Classic tab.
            </ThemedText>
          </View>

          <LinearGradient
            colors={["rgba(255,255,255,0.96)", "rgba(226,232,240,0.82)"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.statusCard}
          >
            <View style={styles.statusRow}>
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor: getConnectionStatusColor(connectionStatus),
                  },
                ]}
              />
              <ThemedText style={styles.statusLabel}>
                {getConnectionStatusText(
                  connectionStatus,
                  isProcessing,
                  isWarmingUp,
                )}
              </ThemedText>
            </View>

            <View style={styles.statsRow}>
              <View style={styles.statBox}>
                <ThemedText style={styles.statValue}>
                  {formatDuration(duration)}
                </ThemedText>
                <ThemedText style={styles.statLabel}>Elapsed</ThemedText>
              </View>
              <View style={styles.statBox}>
                <ThemedText style={styles.statValue}>
                  {Math.round(currentBpm || 120)}
                </ThemedText>
                <ThemedText style={styles.statLabel}>BPM</ThemedText>
              </View>
              <View style={styles.statBox}>
                <ThemedText style={styles.statValue}>
                  {analysisResult?.analysis_summary.total_notes ?? 0}
                </ThemedText>
                <ThemedText style={styles.statLabel}>Notes</ThemedText>
              </View>
              <View style={styles.statBox}>
                <ThemedText style={styles.statValue}>
                  {analysisResult?.analysis_summary.total_chords ?? 0}
                </ThemedText>
                <ThemedText style={styles.statLabel}>Chords</ThemedText>
              </View>
            </View>

            <View style={styles.controlSection}>
              <View style={styles.controlHeaderRow}>
                <ThemedText style={styles.controlLabel}>
                  Low-noise filter
                </ThemedText>
                <ThemedText style={styles.controlValue}>
                  {selectedNoiseProfile.label}
                </ThemedText>
              </View>

              <View style={styles.optionRow}>
                {LIVE_NOISE_PROFILE_OPTIONS.map((option) => {
                  const isActive = option.value === noiseProfile;

                  return (
                    <TouchableOpacity
                      key={option.value}
                      style={[
                        styles.optionChip,
                        isActive ? styles.optionChipActive : null,
                      ]}
                      onPress={() => setNoiseProfile(option.value)}
                    >
                      <ThemedText
                        style={[
                          styles.optionChipText,
                          isActive ? styles.optionChipTextActive : null,
                        ]}
                      >
                        {option.label}
                      </ThemedText>
                    </TouchableOpacity>
                  );
                })}
              </View>

              <ThemedText style={styles.controlHint}>
                {selectedNoiseProfile.description}
              </ThemedText>
            </View>

            <TouchableOpacity
              style={[
                styles.recordButton,
                isRecording
                  ? styles.stopButton
                  : isWarmingUp
                    ? styles.warmingButton
                    : styles.startButton,
                isStartDisabled ? styles.recordButtonDisabled : null,
              ]}
              onPress={
                isRecording ? stopLiveTranscription : startLiveTranscription
              }
              disabled={isStartDisabled}
            >
              {isWarmingUp ? (
                <ActivityIndicator size="small" color="#ffffff" />
              ) : (
                <Ionicons
                  name={isRecording ? "stop" : "radio"}
                  size={24}
                  color="white"
                />
              )}
              <ThemedText style={styles.recordButtonText}>
                {recordButtonLabel}
              </ThemedText>
            </TouchableOpacity>
          </LinearGradient>

          {renderMidiExportCard()}

          <LinearGradient
            colors={["rgba(255,255,255,0.92)", "rgba(241,245,249,0.76)"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.card}
            onLayout={handleLiveScoreSectionLayout}
          >
            <ThemedText type="subtitle" style={styles.cardTitle}>
              {USE_LIVE_OSMD_ENGRAVING_EXPERIMENT
                ? "Live OSMD Engraving"
                : "Committed Score"}
            </ThemedText>
            <MemoizedScoreContent
              analysisResult={analysisResult}
              compact={false}
              hasStoppedRecording={hasStoppedRecording}
              isRecording={isRecording}
              isWarmingUp={isWarmingUp}
              liveEngravingResult={liveEngravingResult}
              liveEngravingVersion={liveEngravingVersion}
              onScoreScrollActiveChange={setIsScoreScrollActive}
            />
          </LinearGradient>

          <LinearGradient
            colors={["rgba(255,255,255,0.92)", "rgba(241,245,249,0.76)"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.card}
          >
            <ThemedText type="subtitle" style={styles.cardTitle}>
              Recent Detections
            </ThemedText>
            {recentEvents.length === 0 ? (
              <ThemedText style={styles.placeholderText}>
                No notes detected yet.
              </ThemedText>
            ) : (
              <View style={styles.eventsList}>
                {recentEvents.map((event) => (
                  <View key={event.key} style={styles.eventRow}>
                    <View
                      style={[
                        styles.eventIcon,
                        { backgroundColor: `${event.color}1A` },
                      ]}
                    >
                      <Ionicons
                        name={event.icon}
                        size={16}
                        color={event.color}
                      />
                    </View>
                    <View style={styles.eventTextWrap}>
                      <ThemedText style={styles.eventLabel}>
                        {event.label}
                      </ThemedText>
                      <ThemedText style={styles.eventDetail}>
                        {event.time.toFixed(2)}s · {event.detail}
                      </ThemedText>
                    </View>
                  </View>
                ))}
              </View>
            )}
          </LinearGradient>

          <LinearGradient
            colors={["#0f172a", "#111827", "#0f172a"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.infoCard}
          >
            <ThemedText
              style={styles.infoTitle}
              lightColor="#f8fafc"
              darkColor="#f8fafc"
            >
              Live pipeline notes
            </ThemedText>
            <ThemedText
              style={styles.infoText}
              lightColor="rgba(226,232,240,0.84)"
              darkColor="rgba(226,232,240,0.84)"
            >
              Audio is captured in short WAV chunks, sent to the overlap-aware
              live endpoint, displayed immediately with coarse rhythm values,
              and then refreshed when deferred refinement lands.
            </ThemedText>
          </LinearGradient>
        </ThemedView>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  screenBackground: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
    backgroundColor: "transparent",
  },
  scrollContent: {
    flexGrow: 1,
  },
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 48,
    paddingBottom: 40,
    gap: 18,
    backgroundColor: "transparent",
  },
  liveSessionScreen: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 48,
    paddingBottom: 16,
  },
  liveSessionScrollView: {
    flex: 1,
  },
  liveSessionScrollContent: {
    flexGrow: 1,
    paddingBottom: 4,
  },
  liveSessionWorkspace: {
    flex: 1,
    minHeight: 0,
    gap: 12,
  },
  liveTopBar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 18,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.12)",
  },
  liveTopBarTitle: {
    fontSize: 18,
    fontWeight: "800",
    letterSpacing: -0.3,
    color: "#f8fafc",
  },
  liveTopBarMetrics: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 14,
  },
  liveInlineMetric: {
    flexDirection: "row",
    alignItems: "baseline",
    gap: 4,
  },
  liveInlineMetricValue: {
    fontSize: 14,
    fontWeight: "700",
    lineHeight: 18,
    color: "#f8fafc",
  },
  liveInlineMetricLabel: {
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    color: "rgba(191,219,254,0.72)",
    fontWeight: "700",
  },
  liveScorePane: {
    flex: 1,
    minHeight: 0,
    borderRadius: 24,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.2)",
    backgroundColor: "rgba(248,250,252,0.96)",
    shadowColor: "#020617",
    shadowOffset: { width: 0, height: 20 },
    shadowOpacity: 0.22,
    shadowRadius: 32,
    elevation: 12,
  },
  liveStatusChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    flexShrink: 1,
  },
  liveStatusChipText: {
    fontSize: 11,
    fontWeight: "700",
  },
  liveScoreViewport: {
    flex: 1,
    minHeight: 220,
    minWidth: 0,
  },
  liveScoreStrip: {
    height: 150,
    minHeight: 150,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(148,163,184,0.24)",
    backgroundColor: "#f8fafc",
  },
  liveScorePlaceholder: {
    flex: 1,
    minHeight: 220,
    borderRadius: 18,
    paddingHorizontal: 18,
    paddingVertical: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.92)",
  },
  liveControlDock: {
    gap: 12,
    padding: 14,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.18)",
    shadowColor: "#020617",
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.18,
    shadowRadius: 24,
    elevation: 8,
  },
  liveRecordButton: {
    minHeight: 44,
  },
  exportCard: {
    borderRadius: 24,
    padding: 16,
    gap: 14,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.24)",
    shadowColor: "#020617",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.16,
    shadowRadius: 24,
    elevation: 8,
  },
  exportCardHeader: {
    gap: 6,
  },
  exportCardTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#f8fafc",
    letterSpacing: -0.2,
  },
  exportCardHint: {
    fontSize: 12,
    lineHeight: 19,
    color: "rgba(226,232,240,0.8)",
  },
  exportActionButton: {
    minHeight: 48,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.18)",
    backgroundColor: "rgba(255,255,255,0.12)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  exportActionButtonDisabled: {
    opacity: 0.45,
  },
  exportActionButtonText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  header: {
    gap: 10,
  },
  title: {
    textAlign: "center",
    color: "#0b1220",
    letterSpacing: -1,
  },
  subtitle: {
    textAlign: "center",
    lineHeight: 22,
    opacity: 1,
    color: "#475569",
    alignSelf: "center",
    maxWidth: 680,
  },
  statusCard: {
    borderRadius: 24,
    padding: 20,
    gap: 18,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.18)",
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.08,
    shadowRadius: 32,
    elevation: 10,
    overflow: "hidden",
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
  },
  statusSpinner: {
    width: 10,
    height: 10,
  },
  statusLabel: {
    flex: 1,
    fontSize: 15,
    fontWeight: "700",
    color: "#0b1220",
  },
  statsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
  },
  statBox: {
    minWidth: "22%",
    flex: 1,
    borderRadius: 18,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderColor: "rgba(148,163,184,0.16)",
    borderWidth: 1,
    alignItems: "center",
    gap: 4,
    backgroundColor: "rgba(255,255,255,0.58)",
  },
  statValue: {
    fontSize: 13,
    fontWeight: "700",
    color: "#0f172a",
  },
  statLabel: {
    fontSize: 9,
    opacity: 1,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    color: "#64748b",
  },
  controlSection: {
    gap: 10,
  },
  controlHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  controlLabel: {
    fontSize: 13,
    fontWeight: "700",
    color: "#0f172a",
  },
  controlValue: {
    fontSize: 12,
    fontWeight: "700",
    color: "#0f766e",
  },
  optionRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  optionChip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.64)",
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  optionChipActive: {
    backgroundColor: "#0f172a",
    borderColor: "#0f172a",
  },
  optionChipText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#334155",
    letterSpacing: 0.2,
  },
  optionChipTextActive: {
    color: "#f8fafc",
  },
  controlHint: {
    fontSize: 12,
    lineHeight: 19,
    opacity: 1,
    color: "#64748b",
  },
  recordButton: {
    borderRadius: 18,
    minHeight: 58,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.12,
    shadowRadius: 18,
    elevation: 8,
  },
  startButton: {
    backgroundColor: "#0f172a",
  },
  warmingButton: {
    backgroundColor: "#d97706",
  },
  stopButton: {
    backgroundColor: "#dc2626",
  },
  recordButtonDisabled: {
    opacity: 0.7,
  },
  recordButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  card: {
    borderRadius: 24,
    paddingVertical: 18,
    gap: 12,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.16)",
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.08,
    shadowRadius: 30,
    elevation: 8,
    overflow: "hidden",
  },
  cardTitle: {
    fontSize: 18,
    color: "#0f172a",
    paddingHorizontal: 18,
    letterSpacing: -0.3,
  },
  cardDescription: {
    lineHeight: 21,
    opacity: 0.82,
  },
  placeholderText: {
    lineHeight: 21,
    color: "#64748b",
  },
  eventsList: {
    gap: 10,
    paddingHorizontal: 18,
    paddingBottom: 4,
  },
  eventRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 6,
  },
  eventIcon: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: "center",
    justifyContent: "center",
  },
  eventTextWrap: {
    flex: 1,
    gap: 2,
  },
  eventLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: "#0f172a",
  },
  eventDetail: {
    fontSize: 12,
    opacity: 1,
    color: "#64748b",
  },
  infoCard: {
    borderRadius: 24,
    padding: 20,
    gap: 10,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.16)",
    shadowColor: "#020617",
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.2,
    shadowRadius: 28,
    elevation: 8,
    overflow: "hidden",
  },
  infoTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: "#f8fafc",
  },
  infoText: {
    lineHeight: 21,
    opacity: 1,
    color: "rgba(226,232,240,0.84)",
  },
});
