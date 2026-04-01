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

interface UseLiveRhythmReturn {
  /** Create a new live session */
  createSession: (initialBpm?: number) => Promise<void>;
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

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastVersionRef = useRef(0);

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
        setIsActive(true);
        setCurrentBpm(data.bpm || initialBpm);
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

  // Process notes
  const processNotes = useCallback(
    async (notes: any[], chords: any[] = []) => {
      if (!sessionId) {
        console.warn("[LiveRhythm] No active session");
        return { coarseNotes: notes, bpm: currentBpm, needsRefresh: false };
      }

      try {
        const response = await fetch(`${BACKEND_URL}/live/process`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            notes,
            chords,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to process notes: ${response.status}`);
        }

        const data = await response.json();

        // Update BPM if changed
        if (data.bpm && data.bpm !== currentBpm) {
          setCurrentBpm(data.bpm);
          onBpmChange?.(data.bpm, data.bpm_confidence || 0);
        }

        // Check for version update
        if (data.refinement_version > lastVersionRef.current) {
          lastVersionRef.current = data.refinement_version;
          setVersion(data.refinement_version);
        }

        // Start polling if not already (done in effect below)
        // Note: polling auto-starts when first notes are processed

        return {
          coarseNotes: data.coarse_notes || notes,
          bpm: data.bpm || currentBpm,
          needsRefresh: data.needs_refresh || false,
        };
      } catch (error) {
        console.error("[LiveRhythm] Failed to process notes:", error);
        return { coarseNotes: notes, bpm: currentBpm, needsRefresh: false };
      }
    },
    [sessionId, currentBpm, onBpmChange],
  );

  // Poll for refinements
  const pollForRefinements = useCallback(async () => {
    if (!sessionId || !isActive) return;

    try {
      const response = await fetch(`${BACKEND_URL}/live/check-refinement`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) return;

      const data = await response.json();

      // Update BPM
      if (data.bpm && data.bpm !== currentBpm) {
        setCurrentBpm(data.bpm);
        onBpmChange?.(data.bpm, data.bpm_confidence || 0);
      }

      // Check if we have new refinements
      if (
        data.needs_refresh &&
        data.refinement_version > lastVersionRef.current
      ) {
        lastVersionRef.current = data.refinement_version;
        setVersion(data.refinement_version);

        // Fetch all notes with best quantization
        const allNotesResponse = await fetch(
          `${BACKEND_URL}/live/get-all-notes`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
          },
        );

        if (allNotesResponse.ok) {
          const allData = await allNotesResponse.json();
          onRefinementReady?.({
            notes: allData.notes || [],
            chords: allData.chords || [],
            bpm: allData.bpm || currentBpm,
            bpmConfidence: allData.bpm_confidence || 0,
            version: allData.refinement_version || data.refinement_version,
          });
        }
      }
    } catch (error) {
      console.error("[LiveRhythm] Poll error:", error);
    }
  }, [sessionId, isActive, currentBpm, onBpmChange, onRefinementReady]);

  // Start polling
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;

    setIsPolling(true);
    pollIntervalRef.current = setInterval(pollForRefinements, pollIntervalMs);
    console.log("[LiveRhythm] Polling started");
  }, [pollForRefinements, pollIntervalMs]);

  // Stop polling
  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    setIsPolling(false);
    console.log("[LiveRhythm] Polling stopped");
  }, []);

  // Finalize session
  const finalizeSession =
    useCallback(async (): Promise<LiveRefinementResult> => {
      stopPolling();

      if (!sessionId) {
        return {
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
          body: JSON.stringify({ session_id: sessionId }),
        });

        if (!response.ok) {
          throw new Error(`Failed to finalize session: ${response.status}`);
        }

        const data = await response.json();

        const result: LiveRefinementResult = {
          notes: data.notes || [],
          chords: data.chords || [],
          bpm: data.bpm || currentBpm,
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
          notes: [],
          chords: [],
          bpm: currentBpm,
          bpmConfidence: 0,
          version,
        };
      }
    }, [sessionId, currentBpm, version, stopPolling, onRefinementReady]);

  // Reset session
  const resetSession = useCallback(async () => {
    stopPolling();

    if (sessionId) {
      try {
        await fetch(`${BACKEND_URL}/live/session/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
      } catch (error) {
        console.error("[LiveRhythm] Failed to reset session:", error);
      }
    }

    setIsActive(false);
    setVersion(0);
    lastVersionRef.current = 0;
    console.log("[LiveRhythm] Session reset");
  }, [sessionId, stopPolling]);

  // Auto-start polling when session becomes active
  useEffect(() => {
    if (isActive && !isPolling && sessionId) {
      startPolling();
    }
  }, [isActive, isPolling, sessionId, startPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  return {
    createSession,
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
