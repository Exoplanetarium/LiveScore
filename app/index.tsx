import { Ionicons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
import React, { useCallback, useEffect, useRef, useState } from "react";
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
import PianoSheetMusic from "../components/PianoSheetMusic";
import { ThemedText } from "../components/ThemedText";
import { ThemedView } from "../components/ThemedView";
import { useLiveRhythm } from "../hooks/useLiveRhythm";

const BACKEND_URL =
  "https://exoplanetarium--livescore-gpu-fastapi-app.modal.run";
const CHUNK_INTERVAL_MS = 600;
const USE_LIVE_NEURAL_PATH = true;
const USE_LIVE_OSMD_ENGRAVING_EXPERIMENT = true;
const LIVE_OSMD_BATCH_MS = 40;

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

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

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
    return "Processing chunk";
  }

  switch (status) {
    case "connected":
      return "Live session active";
    case "connecting":
      return "Connecting";
    case "error":
      return "Chunk upload failed";
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
  const [liveScoreViewportHeight, setLiveScoreViewportHeight] = useState(0);
  const [isScoreScrollActive, setIsScoreScrollActive] = useState(false);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null,
  );
  const [liveEngravingResult, setLiveEngravingResult] =
    useState<AnalysisResult | null>(null);
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
  const pendingChunkUploadRef = useRef<QueuedChunkUpload | null>(null);
  const isRecordingRef = useRef(false);
  const sessionReadyRef = useRef(false);
  const backendWarmupPromiseRef = useRef<Promise<void> | null>(null);
  const currentChunkStartedAtRef = useRef<number | null>(null);
  const chunkSequenceRef = useRef(0);
  // Track concurrent uploads so the spinner only clears when the queue drains.
  const inFlightUploadsRef = useRef(0);
  const [hasFinalizedRecording, setHasFinalizedRecording] = useState(false);
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

  useEffect(() => {
    sessionReadyRef.current = sessionReady;
  }, [sessionReady]);

  const handleLiveScoreSectionLayout = useCallback(
    (event: LayoutChangeEvent) => {
      liveScoreSectionYRef.current = event.nativeEvent.layout.y;
    },
    [],
  );

  const scrollToLiveScoreSection = useCallback(() => {
    const targetY = Math.max(0, liveScoreSectionYRef.current - 12);
    requestAnimationFrame(() => {
      scrollViewRef.current?.scrollTo({ y: targetY, animated: false });
    });
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
    finalizeSession,
    resetSession,
    currentBpm,
    version: liveRefinementVersion,
  } = useLiveRhythm({
    onRefinementReady: (result) => {
      setAnalysisResult((previous) =>
        buildLiveAnalysisResult(
          result.notes as NoteResult[],
          result.chords as ChordResult[],
          previous?.onsets ?? [],
          result.bpm,
          result.bpmConfidence,
        ),
      );
      setSessionReady(true);
      sessionReadyRef.current = true;
      setConnectionStatus("connected");
    },
  });

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

  useEffect(() => {
    const audioOptions = {
      sampleRate: 44100,
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
      clearPendingChunkUpload();
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
  }, [clearPendingChunkUpload, resetSession, warmBackend]);

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
        });
        const uploadFinishedAtMs = Date.now();
        const mergeQueuedAtMs = Date.now();
        setAnalysisResult((previous) =>
          mergeChunkIntoResult(
            previous,
            chunk.coarseNotes as NoteResult[],
            chunk.coarseChords as ChordResult[],
            chunk.onsets as OnsetResult[],
            chunk.bpm,
          ),
        );
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

      scrollToLiveScoreSection();
      setIsWarmingUp(true);
      setConnectionStatus("connecting");
      setDuration(0);
      setIsRecording(false);
      isRecordingRef.current = false;
      setSessionReady(false);
      sessionReadyRef.current = false;
      setHasFinalizedRecording(false);
      inFlightUploadsRef.current = 0;
      clearPendingChunkUpload();
      currentChunkStartedAtRef.current = null;
      chunkSequenceRef.current = 0;

      await resetSession();

      setAnalysisResult(null);
      setLiveEngravingResult(null);
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

      currentChunkStartedAtRef.current = Date.now();
      AudioRecord.start();
      setIsWarmingUp(false);
      setIsRecording(true);
      scrollToLiveScoreSection();
      isRecordingRef.current = true;
      setSessionReady(true);
      sessionReadyRef.current = true;
      setConnectionStatus("connected");
      durationIntervalRef.current = setInterval(() => {
        setDuration((previous) => previous + 0.1);
      }, 100);
      chunkTimeoutRef.current = setTimeout(
        analyzeRecordingChunk,
        CHUNK_INTERVAL_MS,
      );
    } catch (error) {
      console.error("Failed to start live transcription", error);
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
      setIsRecording(false);
      isRecordingRef.current = false;
      Alert.alert(
        "Live Session Error",
        "The app could not create the live backend session. Recording was not started.",
      );
    }
  }, [
    analyzeRecordingChunk,
    clearPendingChunkUpload,
    createSession,
    currentBpm,
    ensureBackendWarm,
    requestPermissions,
    resetSession,
    scrollToLiveScoreSection,
  ]);

  const stopLiveTranscription = useCallback(async () => {
    const wasRecording = isRecordingRef.current;
    isRecordingRef.current = false;
    setIsRecording(false);

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

      const finalResult = await finalizeSession();
      if (finalResult.notes.length > 0 || finalResult.chords.length > 0) {
        setAnalysisResult((previous) =>
          buildLiveAnalysisResult(
            finalResult.notes as NoteResult[],
            finalResult.chords as ChordResult[],
            (finalResult.onsets as OnsetResult[] | undefined) ??
              previous?.onsets ??
              [],
            finalResult.bpm,
            finalResult.bpmConfidence,
          ),
        );
      }

      setSessionReady(false);
      sessionReadyRef.current = false;
      setHasFinalizedRecording(true);
    } catch (error) {
      console.error("Failed to stop live transcription", error);
      setSessionReady(false);
      sessionReadyRef.current = false;
      setConnectionStatus("error");
      Alert.alert(
        "Error",
        "Failed to finalize the live transcription session.",
      );
    } finally {
      setConnectionStatus("disconnected");
      inFlightUploadsRef.current = 0;
      setIsProcessing(false);
    }
  }, [finalizeSession, processRecordedChunk, resolveRecordedFilePath]);

  const recentEvents = [
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
    .slice(0, 12);

  const compactScoreViewportHeight = Math.max(
    220,
    liveScoreViewportHeight || 280,
  );

  const renderScoreContent = (compact: boolean) => {
    if (USE_LIVE_OSMD_ENGRAVING_EXPERIMENT) {
      return (
        <PianoSheetMusic
          results={liveEngravingResult ?? undefined}
          refinementVersion={liveEngravingVersion}
          compact={compact}
          viewportHeight={compact ? compactScoreViewportHeight : undefined}
          onScoreScrollActiveChange={setIsScoreScrollActive}
        />
      );
    }

    if (hasFinalizedRecording && analysisResult) {
      return (
        <PianoSheetMusic
          results={analysisResult}
          compact={compact}
          viewportHeight={compact ? compactScoreViewportHeight : undefined}
          onScoreScrollActiveChange={setIsScoreScrollActive}
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
  };

  if (isLiveSessionLayout) {
    const liveStatusColor = getConnectionStatusColor(connectionStatus);

    return (
      <ThemedView style={styles.liveSessionScreen}>
        <View style={styles.liveSessionWorkspace}>
          <View style={styles.liveTopBar}>
            <ThemedText
              style={styles.liveTopBarTitle}
              lightColor="#0f172a"
              darkColor="#f8fafc"
              numberOfLines={1}
            >
              {USE_LIVE_OSMD_ENGRAVING_EXPERIMENT ? "Live" : "Score"}
            </ThemedText>

            <View style={styles.liveTopBarMetrics}>
              <View style={styles.liveInlineMetric}>
                <ThemedText
                  style={styles.liveInlineMetricValue}
                  lightColor="#0f172a"
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
                  lightColor="#0f172a"
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
                  lightColor="#0f172a"
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
                style={[styles.statusDot, { backgroundColor: liveStatusColor }]}
              />
              {isProcessing || isWarmingUp ? (
                <ActivityIndicator size="small" color={liveStatusColor} />
              ) : null}
            </View>
          </View>

          <View style={styles.liveScorePane}>
            <View
              style={styles.liveScoreViewport}
              onLayout={handleLiveScoreViewportLayout}
            >
              {renderScoreContent(true)}
            </View>
          </View>

          <View style={styles.liveControlDock}>
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
          </View>
        </View>
      </ThemedView>
    );
  }

  return (
    <ScrollView
      ref={scrollViewRef}
      style={styles.scrollView}
      contentContainerStyle={styles.scrollContent}
      showsVerticalScrollIndicator={false}
      nestedScrollEnabled
      scrollEnabled={!isScoreScrollActive}
    >
      <ThemedView style={styles.container}>
        <View style={styles.header}>
          <ThemedText type="title" style={styles.title}>
            Live Piano Transcription
          </ThemedText>
          <ThemedText style={styles.subtitle}>
            This tab uses the live chunk pipeline. The previous
            record-then-analyze screen is available in the Classic tab.
          </ThemedText>
        </View>

        <View style={styles.statusCard}>
          <View style={styles.statusRow}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: getConnectionStatusColor(connectionStatus) },
              ]}
            />
            <ThemedText style={styles.statusLabel}>
              {getConnectionStatusText(
                connectionStatus,
                isProcessing,
                isWarmingUp,
              )}
            </ThemedText>
            {isProcessing || isWarmingUp ? (
              <ActivityIndicator size="small" color="#2f95dc" />
            ) : null}
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
        </View>

        <View style={styles.card} onLayout={handleLiveScoreSectionLayout}>
          <ThemedText type="subtitle" style={styles.cardTitle}>
            {USE_LIVE_OSMD_ENGRAVING_EXPERIMENT
              ? "Live OSMD Engraving"
              : "Committed Score"}
          </ThemedText>
          {renderScoreContent(false)}
        </View>

        <View style={styles.card}>
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
                    <Ionicons name={event.icon} size={16} color={event.color} />
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
        </View>

        <View style={styles.infoCard}>
          <ThemedText style={styles.infoTitle}>Live pipeline notes</ThemedText>
          <ThemedText style={styles.infoText}>
            Audio is captured in short WAV chunks, sent to the overlap-aware
            live endpoint, displayed immediately with coarse rhythm values, and
            then refreshed when deferred refinement lands.
          </ThemedText>
        </View>
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
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
  },
  liveSessionScreen: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 48,
    paddingBottom: 16,
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
    paddingHorizontal: 4,
  },
  liveTopBarTitle: {
    fontSize: 18,
    fontWeight: "800",
    letterSpacing: -0.3,
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
  },
  liveInlineMetricLabel: {
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    color: "#64748b",
    fontWeight: "600",
  },
  liveScorePane: {
    flex: 1,
    minHeight: 0,
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  liveStatusChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
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
  liveScorePlaceholder: {
    flex: 1,
    minHeight: 220,
    borderRadius: 10,
    paddingHorizontal: 18,
    paddingVertical: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#ffffff",
  },
  liveControlDock: {
    gap: 10,
    padding: 12,
    borderRadius: 16,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  liveRecordButton: {
    minHeight: 44,
  },
  header: {
    gap: 8,
  },
  title: {
    textAlign: "center",
  },
  subtitle: {
    textAlign: "center",
    lineHeight: 21,
    opacity: 0.78,
  },
  statusCard: {
    borderRadius: 18,
    padding: 18,
    backgroundColor: "rgba(47, 149, 220, 0.08)",
    gap: 16,
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
  statusLabel: {
    flex: 1,
    fontSize: 15,
    fontWeight: "600",
  },
  statsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  statBox: {
    minWidth: "22%",
    flex: 1,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderColor: "#2563eb1A",
    borderWidth: 1,
    alignItems: "center",
    gap: 4,
  },
  statValue: {
    fontSize: 12,
    fontWeight: "700",
  },
  statLabel: {
    fontSize: 9,
    opacity: 0.72,
    textTransform: "uppercase",
    letterSpacing: 0.4,
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
  },
  controlValue: {
    fontSize: 12,
    fontWeight: "700",
    color: "#2563eb",
  },
  optionRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  optionChip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#2563eb33",
    backgroundColor: "rgba(255, 255, 255, 0.55)",
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  optionChipActive: {
    backgroundColor: "#2563eb",
    borderColor: "#2563eb",
  },
  optionChipText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#1d4ed8",
  },
  optionChipTextActive: {
    color: "#ffffff",
  },
  controlHint: {
    fontSize: 12,
    lineHeight: 18,
    opacity: 0.72,
  },
  recordButton: {
    borderRadius: 16,
    minHeight: 56,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  startButton: {
    backgroundColor: "#2563eb",
  },
  warmingButton: {
    backgroundColor: "#f59e0b",
  },
  stopButton: {
    backgroundColor: "#dc2626",
  },
  recordButtonDisabled: {
    opacity: 0.92,
  },
  recordButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "700",
  },
  card: {
    borderRadius: 18,
    paddingVertical: 18,
    backgroundColor: "rgba(15, 23, 42, 0.04)",
    gap: 12,
  },
  cardTitle: {
    fontSize: 18,
    color: "#0f172a",
  },
  cardDescription: {
    lineHeight: 21,
    opacity: 0.82,
  },
  placeholderText: {
    lineHeight: 21,
    color: "#475569",
  },
  eventsList: {
    gap: 10,
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
  },
  eventDetail: {
    fontSize: 12,
    opacity: 0.72,
  },
  infoCard: {
    borderRadius: 18,
    padding: 18,
    backgroundColor: "rgba(245, 158, 11, 0.12)",
    gap: 8,
  },
  infoTitle: {
    fontSize: 15,
    fontWeight: "700",
  },
  infoText: {
    lineHeight: 20,
    opacity: 0.82,
  },
});
