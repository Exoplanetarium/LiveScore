/**
 * Live Rhythm Refinement Hook
 *
 * Provides a React hook for managing live transcription with deferred refinement.
 *
 * Usage:
 * 1. Call `createSession()` when starting recording
 * 2. After each analysis result, call `processNotes(notes, chords)`
 * 3. The hook polls for refinements and triggers `onRefinementReady` callback
 * 4. Call `finalizeSession()` when recording stops
 *
 * The `onRefinementReady` callback receives all notes with best quantization,
 * plus a `version` number for cache invalidation.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// Backend URL - same as main app
const BACKEND_URL =
  "https://exoplanetarium--livescore-gpu-fastapi-app.modal.run";

interface LiveRefinementResult {
  onsets?: any[];
  notes: any[];
  chords: any[];
  bpm: number;
  bpmConfidence: number;
  version: number;
}

interface UseLiveRhythmOptions {
  /** How often to poll for refinements (ms). Default: 800 */
  pollIntervalMs?: number;
  /** Callback when refined notes are ready */
  onRefinementReady?: (result: LiveRefinementResult) => void;
  /** Callback for BPM updates */
  onBpmChange?: (bpm: number, confidence: number) => void;
}

interface LiveAudioChunkOptions {
  noiseProfile?: string;
  useNeuralLive?: boolean;
  useAdaptiveOnsetThreshold?: boolean;
}

interface LiveAudioChunkTiming {
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
}

interface ProcessAudioChunkResult {
  coarseNotes: any[];
  coarseChords: any[];
  bpm: number;
  needsRefresh: boolean;
  onsets: any[];
  timing?: LiveAudioChunkTiming;
}

interface UseLiveRhythmReturn {
  /** Create a new live session */
  createSession: (initialBpm?: number) => Promise<void>;
  /** Process one recorded audio chunk file through analysis + live quantization */
  processAudioChunk: (
    fileUri: string,
    options?: LiveAudioChunkOptions,
  ) => Promise<ProcessAudioChunkResult>;
  /** Process newly detected notes */
  processNotes: (
    notes: any[],
    chords?: any[],
  ) => Promise<{
    coarseNotes: any[];
    bpm: number;
    needsRefresh: boolean;
  }>;
  /** Finalize session and get all refined notes */
  finalizeSession: () => Promise<LiveRefinementResult>;
  /** Reset the current session */
  resetSession: () => Promise<void>;
  /** Current session ID */
  sessionId: string | null;
  /** Whether a session is active */
  isActive: boolean;
  /** Current BPM estimate */
  currentBpm: number;
  /** Current refinement version */
  version: number;
  /** Whether refinement polling is active */
  isPolling: boolean;
  /** Manually start polling (usually auto-started) */
  startPolling: () => void;
  /** Stop polling (usually auto-stopped on finalize) */
  stopPolling: () => void;
}

export function useLiveRhythm(
  options: UseLiveRhythmOptions = {},
): UseLiveRhythmReturn {
  const { pollIntervalMs = 800, onRefinementReady, onBpmChange } = options;

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [currentBpm, setCurrentBpm] = useState(120);
  const [version, setVersion] = useState(0);
  const [isPolling, setIsPolling] = useState(false);

  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastVersionRef = useRef(0);
  const sessionIdRef = useRef<string | null>(null);
  const isActiveRef = useRef(false);
  const currentBpmRef = useRef(120);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    isActiveRef.current = isActive;
  }, [isActive]);

  useEffect(() => {
    currentBpmRef.current = currentBpm;
  }, [currentBpm]);

  // Generate unique session ID
  const generateSessionId = useCallback(() => {
    return `live_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Create a new session
  const createSession = useCallback(
    async (initialBpm = 120) => {
      const newSessionId = generateSessionId();

      try {
        const response = await fetch(`${BACKEND_URL}/live/session/create`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: newSessionId,
            initial_bpm: initialBpm,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to create session: ${response.status}`);
        }

        const data = await response.json();
        setSessionId(newSessionId);
        sessionIdRef.current = newSessionId;
        setIsActive(true);
        isActiveRef.current = true;
        setCurrentBpm(data.bpm || initialBpm);
        currentBpmRef.current = data.bpm || initialBpm;
        setVersion(0);
        lastVersionRef.current = 0;

        console.log("[LiveRhythm] Session created:", newSessionId);
      } catch (error) {
        console.error("[LiveRhythm] Failed to create session:", error);
        throw error;
      }
    },
    [generateSessionId],
  );

  // Stop polling
  const stopPolling = useCallback(() => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
    setIsPolling(false);
    console.log("[LiveRhythm] Polling stopped");
  }, []);

  // Poll for refinements
  const pollForRefinements = useCallback(async () => {
    const activeSessionId = sessionIdRef.current;
    if (!activeSessionId || !isActiveRef.current) return;

    try {
      const response = await fetch(`${BACKEND_URL}/live/check-refinement`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: activeSessionId }),
      });

      if (!response.ok) {
        stopPolling();
        return;
      }

      const data = await response.json();
      const nextPollDelayMs =
        typeof data.next_refinement_poll_ms === "number"
          ? data.next_refinement_poll_ms
          : null;

      // Update BPM
      if (data.bpm && data.bpm !== currentBpmRef.current) {
        setCurrentBpm(data.bpm);
        currentBpmRef.current = data.bpm;
        onBpmChange?.(data.bpm, data.bpm_confidence || 0);
      }

      // Check if we have new refinements
      if (
        data.needs_refresh &&
        data.refinement_version > lastVersionRef.current
      ) {
        lastVersionRef.current = data.refinement_version;
        setVersion(data.refinement_version);

        onRefinementReady?.({
          notes: data.all_notes || [],
          chords: data.all_chords || [],
          bpm: data.bpm || currentBpmRef.current,
          bpmConfidence: data.bpm_confidence || 0,
          version: data.refinement_version,
        });
      }

      if (nextPollDelayMs != null) {
        if (pollTimeoutRef.current) {
          clearTimeout(pollTimeoutRef.current);
        }
        setIsPolling(true);
        pollTimeoutRef.current = setTimeout(
          () => {
            pollTimeoutRef.current = null;
            void pollForRefinements();
          },
          Math.max(250, nextPollDelayMs),
        );
      } else {
        stopPolling();
      }
    } catch (error) {
      console.error("[LiveRhythm] Poll error:", error);
      stopPolling();
    }
  }, [onBpmChange, onRefinementReady, stopPolling]);

  const armRefinementPollingWindow = useCallback(
    (nextPollDelayMs: number | null | undefined) => {
      if (nextPollDelayMs == null) {
        stopPolling();
        return;
      }

      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }

      setIsPolling(true);
      pollTimeoutRef.current = setTimeout(
        () => {
          pollTimeoutRef.current = null;
          void pollForRefinements();
        },
        Math.max(250, nextPollDelayMs),
      );
      console.log("[LiveRhythm] Polling window armed", nextPollDelayMs);
    },
    [pollForRefinements, stopPolling],
  );

  const processNotes = useCallback(
    async (notes: any[], chords: any[] = []) => {
      const activeSessionId = sessionIdRef.current;
      const activeBpm = currentBpmRef.current;

      if (!activeSessionId) {
        console.warn("[LiveRhythm] No active session");
        return { coarseNotes: notes, bpm: activeBpm, needsRefresh: false };
      }

      try {
        const response = await fetch(`${BACKEND_URL}/live/process`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: activeSessionId,
            notes,
            chords,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to process notes: ${response.status}`);
        }

        const data = await response.json();

        if (data.bpm && data.bpm !== currentBpmRef.current) {
          setCurrentBpm(data.bpm);
          currentBpmRef.current = data.bpm;
          onBpmChange?.(data.bpm, data.bpm_confidence || 0);
        }

        const hasNewRefinement =
          data.needs_refresh &&
          data.refinement_version > lastVersionRef.current;

        if (data.refinement_version > lastVersionRef.current) {
          lastVersionRef.current = data.refinement_version;
          setVersion(data.refinement_version);
        }

        if (hasNewRefinement && data.all_notes) {
          onRefinementReady?.({
            notes: data.all_notes || [],
            chords: data.all_chords || [],
            bpm: data.bpm || currentBpmRef.current,
            bpmConfidence: data.bpm_confidence || 0,
            version: data.refinement_version,
          });
        }

        armRefinementPollingWindow(data.next_refinement_poll_ms);

        return {
          coarseNotes: data.coarse_notes || notes,
          bpm: data.bpm || currentBpmRef.current,
          needsRefresh: data.needs_refresh || false,
        };
      } catch (error) {
        console.error("[LiveRhythm] Failed to process notes:", error);
        return {
          coarseNotes: notes,
          bpm: currentBpmRef.current,
          needsRefresh: false,
        };
      }
    },
    [armRefinementPollingWindow, onBpmChange, onRefinementReady],
  );

  const processAudioChunk = useCallback(
    async (
      fileUri: string,
      options: LiveAudioChunkOptions = {},
    ): Promise<ProcessAudioChunkResult> => {
      const activeSessionId = sessionIdRef.current;
      const activeBpm = currentBpmRef.current;

      if (!activeSessionId) {
        console.warn("[LiveRhythm] No active session");
        return {
          coarseNotes: [],
          coarseChords: [],
          bpm: activeBpm,
          needsRefresh: false,
          onsets: [],
        };
      }

      try {
        const formData = new FormData();
        const useNeuralLive = options.useNeuralLive ?? true;
        const useAdaptiveOnsetThreshold =
          options.useAdaptiveOnsetThreshold ?? true;
        // @ts-ignore React Native FormData supports file objects
        formData.append("file", {
          uri: fileUri,
          type: "audio/wav",
          name: "chunk.wav",
        });
        formData.append("session_id", activeSessionId);
        formData.append("use_neural_live", useNeuralLive ? "true" : "false");
        formData.append(
          "adaptive_onset_threshold",
          useAdaptiveOnsetThreshold ? "true" : "false",
        );
        if (options.noiseProfile) {
          formData.append("noise_profile", options.noiseProfile);
        }

        const response = await fetch(`${BACKEND_URL}/live/audio-chunk`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to process audio chunk: ${response.status}`);
        }

        const data = await response.json();
        const rawTiming =
          data && typeof data._timing_ms === "object" && data._timing_ms
            ? (data._timing_ms as Record<string, unknown>)
            : undefined;
        const analysisSummary =
          data &&
          typeof data.analysis_summary === "object" &&
          data.analysis_summary
            ? (data.analysis_summary as Record<string, unknown>)
            : undefined;
        const timing: LiveAudioChunkTiming | undefined = rawTiming
          ? {
              analysisPath:
                typeof rawTiming.analysis_path === "string"
                  ? rawTiming.analysis_path
                  : typeof data.analysis_path === "string"
                    ? data.analysis_path
                    : undefined,
              chunkTotalMs:
                typeof rawTiming.chunk_total === "number"
                  ? rawTiming.chunk_total
                  : undefined,
              chunkInferenceMs:
                typeof rawTiming.chunk_inference === "number"
                  ? rawTiming.chunk_inference
                  : undefined,
              neuralTotalMs:
                typeof rawTiming.neural_total === "number"
                  ? rawTiming.neural_total
                  : undefined,
              modelInferenceMs:
                typeof rawTiming.neural_model_inference === "number"
                  ? rawTiming.neural_model_inference
                  : undefined,
              realTimeFactor:
                typeof rawTiming.real_time_factor === "number"
                  ? rawTiming.real_time_factor
                  : undefined,
              neuralError:
                typeof rawTiming.neural_error === "string"
                  ? rawTiming.neural_error
                  : typeof data.neural_error === "string"
                    ? data.neural_error
                    : typeof data.fallback_reason === "string"
                      ? data.fallback_reason
                      : typeof data.stream_info?.neural_error === "string"
                        ? data.stream_info.neural_error
                        : undefined,
              onsetThreshold:
                typeof rawTiming.neural_onset_threshold_selected === "number"
                  ? rawTiming.neural_onset_threshold_selected
                  : undefined,
              onsetThresholdProfile:
                typeof analysisSummary?.live_onset_threshold_profile ===
                "string"
                  ? analysisSummary.live_onset_threshold_profile
                  : undefined,
              onsetThresholdExperiment:
                typeof analysisSummary?.live_onset_threshold_experiment ===
                "string"
                  ? analysisSummary.live_onset_threshold_experiment
                  : undefined,
              chunkRms:
                typeof rawTiming.neural_chunk_rms === "number"
                  ? rawTiming.neural_chunk_rms
                  : undefined,
              chunkPeak:
                typeof rawTiming.neural_chunk_peak === "number"
                  ? rawTiming.neural_chunk_peak
                  : undefined,
              chunkCrestFactor:
                typeof rawTiming.neural_chunk_crest_factor === "number"
                  ? rawTiming.neural_chunk_crest_factor
                  : undefined,
            }
          : undefined;

        if (timing) {
          const analysisPath = timing.analysisPath ?? "unknown";
          const neuralError =
            timing.neuralError ??
            data.neural_error ??
            data.fallback_reason ??
            data.stream_info?.neural_error;
          const fallbackDiagnostic =
            typeof analysisPath === "string" &&
            analysisPath.endsWith("_fallback") &&
            !neuralError
              ? "backend response is missing neural_error; likely stale backend deploy"
              : undefined;

          console.log("[LiveRhythm] Chunk latency", {
            analysisPath,
            totalMs: timing.chunkTotalMs,
            inferenceMs: timing.chunkInferenceMs,
            neuralTotalMs: timing.neuralTotalMs,
            modelInferenceMs: timing.modelInferenceMs,
            onsetThreshold: timing.onsetThreshold,
            onsetThresholdProfile: timing.onsetThresholdProfile,
            onsetThresholdExperiment: timing.onsetThresholdExperiment,
            chunkRms: timing.chunkRms,
            chunkPeak: timing.chunkPeak,
            chunkCrestFactor: timing.chunkCrestFactor,
            neuralError,
            fallbackDiagnostic,
            realTimeFactor: timing.realTimeFactor,
          });
        }

        if (data.bpm && data.bpm !== currentBpmRef.current) {
          setCurrentBpm(data.bpm);
          currentBpmRef.current = data.bpm;
          onBpmChange?.(data.bpm, data.bpm_confidence || 0);
        }

        const hasNewRefinement =
          data.needs_refresh &&
          data.refinement_version > lastVersionRef.current;

        if (data.refinement_version > lastVersionRef.current) {
          lastVersionRef.current = data.refinement_version;
          setVersion(data.refinement_version);
        }

        if (hasNewRefinement && data.all_notes) {
          onRefinementReady?.({
            notes: data.all_notes || [],
            chords: data.all_chords || [],
            bpm: data.bpm || currentBpmRef.current,
            bpmConfidence: data.bpm_confidence || 0,
            version: data.refinement_version,
          });
        }

        armRefinementPollingWindow(data.next_refinement_poll_ms);

        return {
          coarseNotes: data.notes || [],
          coarseChords: data.chords || [],
          bpm: data.bpm || currentBpmRef.current,
          needsRefresh: data.needs_refresh || false,
          onsets: data.onsets || [],
          timing,
        };
      } catch (error) {
        console.error("[LiveRhythm] Failed to process audio chunk:", error);
        return {
          coarseNotes: [],
          coarseChords: [],
          bpm: currentBpmRef.current,
          needsRefresh: false,
          onsets: [],
          timing: undefined,
        };
      }
    },
    [armRefinementPollingWindow, onBpmChange, onRefinementReady],
  );

  // Start polling
  const startPolling = useCallback(() => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
    }

    setIsPolling(true);
    pollTimeoutRef.current = setTimeout(
      () => {
        pollTimeoutRef.current = null;
        void pollForRefinements();
      },
      Math.max(250, pollIntervalMs),
    );
    console.log("[LiveRhythm] Polling started");
  }, [pollForRefinements, pollIntervalMs]);

  // Finalize session
  const finalizeSession =
    useCallback(async (): Promise<LiveRefinementResult> => {
      stopPolling();

      if (!sessionId) {
        return {
          onsets: [],
          notes: [],
          chords: [],
          bpm: currentBpm,
          bpmConfidence: 0,
          version: 0,
        };
      }

      try {
        const response = await fetch(`${BACKEND_URL}/live/finalize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionIdRef.current }),
        });

        if (!response.ok) {
          throw new Error(`Failed to finalize session: ${response.status}`);
        }

        const data = await response.json();

        const result: LiveRefinementResult = {
          onsets: data.onsets || [],
          notes: data.notes || [],
          chords: data.chords || [],
          bpm: data.bpm || currentBpmRef.current,
          bpmConfidence: data.bpm_confidence || 0,
          version: data.refinement_version || version,
        };

        // Notify callback with final refinement
        onRefinementReady?.(result);

        console.log(
          "[LiveRhythm] Session finalized:",
          data.total_notes,
          "notes",
        );

        return result;
      } catch (error) {
        console.error("[LiveRhythm] Failed to finalize session:", error);
        return {
          onsets: [],
          notes: [],
          chords: [],
          bpm: currentBpmRef.current,
          bpmConfidence: 0,
          version,
        };
      }
    }, [stopPolling, sessionId, currentBpm, version, onRefinementReady]);

  // Reset session
  const resetSession = useCallback(async () => {
    stopPolling();

    if (sessionIdRef.current) {
      try {
        await fetch(`${BACKEND_URL}/live/session/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionIdRef.current }),
        });
      } catch (error) {
        console.error("[LiveRhythm] Failed to reset session:", error);
      }
    }

    setSessionId(null);
    sessionIdRef.current = null;
    setIsActive(false);
    isActiveRef.current = false;
    setVersion(0);
    lastVersionRef.current = 0;
    console.log("[LiveRhythm] Session reset");
  }, [stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, []);

  return {
    createSession,
    processAudioChunk,
    processNotes,
    finalizeSession,
    resetSession,
    sessionId,
    isActive,
    currentBpm,
    version,
    isPolling,
    startPolling,
    stopPolling,
  };
}

export default useLiveRhythm;
