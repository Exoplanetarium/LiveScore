import React, { useCallback, useEffect, useMemo, useRef } from "react";
import { StyleSheet, View } from "react-native";
import { WebView, WebViewMessageEvent } from "react-native-webview";

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

interface LivePianoRollWebViewProps {
  notes: LiveNoteLike[];
  chords?: LiveChordLike[];
  bpm?: number;
  elapsedSeconds: number;
  isRecording?: boolean;
  height?: number;
}

type LiveRenderEvent = {
  id: string;
  midi: number;
  start: number;
  end: number;
  hand?: string;
  confidence: number;
  kind: "note" | "chord";
};

type LiveRenderDelta =
  | { op: "upsert"; event: LiveRenderEvent }
  | { op: "remove"; id: string };

const DELTA_BATCH_MS = 40;
const DEFAULT_DURATION_SECONDS = 0.22;

const LIVE_PIANO_ROLL_HTML = `
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <style>
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #f6f8fb;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #root {
      width: 100%;
      height: 100%;
      position: relative;
      background:
        radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 36%),
        linear-gradient(180deg, #fbfdff 0%, #eef4fa 100%);
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
  </style>
</head>
<body>
  <div id="root">
    <canvas id="roll"></canvas>
  </div>
  <script>
    (function() {
      var state = {
        events: new Map(),
        bpm: 120,
        elapsedSeconds: 0,
        isRecording: false,
        transportAnchorElapsed: 0,
        transportAnchorNow: 0,
        stickyMinMidi: 53,
        stickyMaxMidi: 67,
      };

      var config = {
        pixelsPerSecond: 70,
        visibleSeconds: 8,
        playheadRatio: 0.88,
        axisWidth: 36,
        keyboardHeight: 32,
        footerHeight: 18,
        minRange: 18,
        rowPadding: 2,
        notePadding: 2,
        midiFloor: 21,
        midiCeiling: 108,
        minRowHeight: 5,
      };

      var canvas = document.getElementById("roll");
      var ctx = canvas.getContext("2d");
      var dpr = Math.max(1, window.devicePixelRatio || 1);
      var width = 0;
      var height = 0;
      var rollWidth = 0;
      var rollHeight = 0;

      function post(message) {
        try {
          if (window.ReactNativeWebView) {
            window.ReactNativeWebView.postMessage(JSON.stringify(message));
          }
        } catch (error) {}
      }

      function resizeCanvas() {
        var root = document.getElementById("root");
        if (!root) {
          return;
        }
        var nextWidth = Math.max(1, root.clientWidth);
        var nextHeight = Math.max(1, root.clientHeight);
        if (nextWidth === width && nextHeight === height) {
          return;
        }
        width = nextWidth;
        height = nextHeight;
        rollWidth = Math.max(160, width - config.axisWidth);
        rollHeight = Math.max(120, height - config.keyboardHeight - config.footerHeight);
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function currentElapsedSeconds() {
        if (!state.isRecording) {
          return state.transportAnchorElapsed;
        }
        return state.transportAnchorElapsed + (performance.now() - state.transportAnchorNow) / 1000;
      }

      function isBlackKey(midi) {
        var note = ((midi % 12) + 12) % 12;
        return note === 1 || note === 3 || note === 6 || note === 8 || note === 10;
      }

      function midiToShortName(midi) {
        var names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        return names[midi % 12] + String(Math.floor(midi / 12) - 1);
      }

      function colorForHand(hand) {
        if (hand === "treble") return "#0ea5e9";
        if (hand === "bass") return "#f97316";
        return "#6366f1";
      }

      function computeRange() {
        var minMidi = state.stickyMinMidi;
        var maxMidi = state.stickyMaxMidi;

        state.events.forEach(function(event) {
          minMidi = Math.min(minMidi, event.midi - 2);
          maxMidi = Math.max(maxMidi, event.midi + 2);
        });

        if (maxMidi - minMidi + 1 < config.minRange) {
          var center = (minMidi + maxMidi) / 2;
          minMidi = Math.floor(center - config.minRange / 2);
          maxMidi = Math.ceil(center + config.minRange / 2);
        }

        minMidi = Math.max(config.midiFloor, minMidi);
        maxMidi = Math.min(config.midiCeiling, maxMidi);

        var maxRows = Math.max(config.minRange, Math.floor(rollHeight / config.minRowHeight));
        if (maxMidi - minMidi + 1 > maxRows) {
          var nextCenter = (minMidi + maxMidi) / 2;
          var halfRange = Math.floor(maxRows / 2);
          minMidi = Math.max(config.midiFloor, Math.floor(nextCenter - halfRange));
          maxMidi = Math.min(config.midiCeiling, minMidi + maxRows - 1);
        }

        state.stickyMinMidi = minMidi;
        state.stickyMaxMidi = maxMidi;

        return {
          min: minMidi,
          max: maxMidi,
          rows: maxMidi - minMidi + 1,
          rowHeight: rollHeight / (maxMidi - minMidi + 1),
        };
      }

      function drawBackground(range) {
        ctx.clearRect(0, 0, width, height);

        ctx.fillStyle = "rgba(255,255,255,0.92)";
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = "rgba(248,250,252,0.96)";
        ctx.fillRect(config.axisWidth, 0, rollWidth, rollHeight);

        for (var midi = range.min; midi <= range.max; midi += 1) {
          if (!isBlackKey(midi)) {
            continue;
          }
          var rowIndex = range.max - midi;
          var rowY = rowIndex * range.rowHeight;
          ctx.fillStyle = "rgba(15,23,42,0.045)";
          ctx.fillRect(config.axisWidth, rowY, rollWidth, range.rowHeight);
        }
      }

      function drawGrid(elapsedSeconds) {
        var safeBpm = state.bpm > 0 ? state.bpm : 120;
        var secondsPerBeat = 60 / safeBpm;
        var firstBeat = Math.floor(Math.max(0, elapsedSeconds - config.visibleSeconds - 1) / secondsPerBeat);
        var lastBeat = Math.ceil((elapsedSeconds + 2) / secondsPerBeat);
        var translationX = config.axisWidth + rollWidth * config.playheadRatio - (elapsedSeconds * config.pixelsPerSecond);

        for (var beat = firstBeat; beat <= lastBeat; beat += 1) {
          var x = beat * secondsPerBeat * config.pixelsPerSecond + translationX;
          var isMeasure = beat % 4 === 0;
          ctx.strokeStyle = isMeasure ? "rgba(100,116,139,0.55)" : "rgba(148,163,184,0.22)";
          ctx.lineWidth = isMeasure ? 1 : 0.5;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, rollHeight);
          ctx.stroke();
        }
      }

      function drawYAxis(range) {
        ctx.fillStyle = "rgba(241,245,249,0.98)";
        ctx.fillRect(0, 0, config.axisWidth, rollHeight);

        ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.fillStyle = "#475569";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        var labels = [];
        for (var midi = range.min; midi <= range.max; midi += 1) {
          if (midi % 12 === 0) {
            labels.push(midi);
          }
        }
        if (labels.indexOf(range.max) === -1) {
          labels.push(range.max);
        }
        if (labels.indexOf(range.min) === -1) {
          labels.push(range.min);
        }

        labels.sort(function(left, right) { return right - left; });

        labels.forEach(function(midi) {
          var rowIndex = range.max - midi;
          var y = rowIndex * range.rowHeight + range.rowHeight / 2;
          ctx.fillText(midiToShortName(midi), config.axisWidth - 6, y);
        });
      }

      function drawNotes(range, elapsedSeconds) {
        var translationX = config.axisWidth + rollWidth * config.playheadRatio - (elapsedSeconds * config.pixelsPerSecond);
        var tail = elapsedSeconds - config.visibleSeconds - 1.5;
        var head = elapsedSeconds + 1.5;

        state.events.forEach(function(event) {
          if (event.end < tail || event.start > head) {
            return;
          }
          if (event.midi < range.min || event.midi > range.max) {
            return;
          }

          var rowIndex = range.max - event.midi;
          var y = rowIndex * range.rowHeight + 1;
          var x = translationX + event.start * config.pixelsPerSecond;
          var widthSeconds = Math.max(0.05, event.end - event.start);
          var noteWidth = Math.max(4, widthSeconds * config.pixelsPerSecond);
          var noteHeight = Math.max(2.5, range.rowHeight - config.notePadding);

          ctx.fillStyle = colorForHand(event.hand);
          ctx.globalAlpha = 0.45 + 0.55 * Math.max(0.15, Math.min(1, event.confidence));
          ctx.beginPath();
          var radius = Math.min(4, noteHeight / 2);
          var noteRight = x + noteWidth;
          var noteBottom = y + noteHeight;
          ctx.moveTo(x + radius, y);
          ctx.lineTo(noteRight - radius, y);
          ctx.quadraticCurveTo(noteRight, y, noteRight, y + radius);
          ctx.lineTo(noteRight, noteBottom - radius);
          ctx.quadraticCurveTo(noteRight, noteBottom, noteRight - radius, noteBottom);
          ctx.lineTo(x + radius, noteBottom);
          ctx.quadraticCurveTo(x, noteBottom, x, noteBottom - radius);
          ctx.lineTo(x, y + radius);
          ctx.quadraticCurveTo(x, y, x + radius, y);
          ctx.closePath();
          ctx.fill();
        });
        ctx.globalAlpha = 1;
      }

      function drawPlayhead() {
        var playheadX = config.axisWidth + rollWidth * config.playheadRatio;
        ctx.strokeStyle = "rgba(250,204,21,0.95)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, rollHeight);
        ctx.stroke();
      }

      function drawKeyboard(range) {
        var keyboardTop = rollHeight;
        ctx.fillStyle = "rgba(255,255,255,0.98)";
        ctx.fillRect(0, keyboardTop, width, config.keyboardHeight + config.footerHeight);

        var naturalCount = 0;
        for (var midi = range.min; midi <= range.max; midi += 1) {
          if (!isBlackKey(midi)) {
            naturalCount += 1;
          }
        }
        var whiteKeyWidth = naturalCount > 0 ? rollWidth / naturalCount : rollWidth;
        var cursorX = config.axisWidth;

        for (var currentMidi = range.min; currentMidi <= range.max; currentMidi += 1) {
          if (isBlackKey(currentMidi)) {
            continue;
          }
          ctx.fillStyle = "rgba(255,255,255,0.98)";
          ctx.strokeStyle = "rgba(148,163,184,0.35)";
          ctx.lineWidth = 1;
          ctx.fillRect(cursorX, keyboardTop, whiteKeyWidth, config.keyboardHeight);
          ctx.strokeRect(cursorX, keyboardTop, whiteKeyWidth, config.keyboardHeight);
          cursorX += whiteKeyWidth;
        }

        cursorX = config.axisWidth;
        for (var blackMidi = range.min; blackMidi <= range.max; blackMidi += 1) {
          if (isBlackKey(blackMidi)) {
            ctx.fillStyle = "rgba(15,23,42,0.88)";
            ctx.fillRect(cursorX - whiteKeyWidth * 0.28, keyboardTop, whiteKeyWidth * 0.56, config.keyboardHeight * 0.62);
          } else {
            cursorX += whiteKeyWidth;
          }
        }

        ctx.fillStyle = "#64748b";
        ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("Live WebView Roll", 12, keyboardTop + config.keyboardHeight + config.footerHeight / 2);

        ctx.textAlign = "right";
        ctx.fillText(Math.round(state.bpm || 120) + " BPM", width - 12, keyboardTop + config.keyboardHeight + config.footerHeight / 2);
      }

      function renderFrame() {
        resizeCanvas();
        var elapsedSeconds = currentElapsedSeconds();
        var range = computeRange();
        drawBackground(range);
        drawGrid(elapsedSeconds);
        drawYAxis(range);
        drawNotes(range, elapsedSeconds);
        drawPlayhead();
        drawKeyboard(range);
        requestAnimationFrame(renderFrame);
      }

      function applyNoteDeltas(events) {
        if (!Array.isArray(events)) {
          return;
        }

        events.forEach(function(delta) {
          if (!delta || typeof delta !== "object") {
            return;
          }
          if (delta.op === "remove") {
            state.events.delete(delta.id);
            return;
          }
          if (delta.op === "upsert" && delta.event && delta.event.id) {
            state.events.set(delta.event.id, delta.event);
          }
        });
      }

      function replaceAll(events) {
        state.events.clear();
        if (!Array.isArray(events)) {
          return;
        }
        events.forEach(function(event) {
          if (event && event.id) {
            state.events.set(event.id, event);
          }
        });
      }

      function updateTransport(message) {
        if (typeof message.bpm === "number" && isFinite(message.bpm)) {
          state.bpm = message.bpm;
        }
        state.isRecording = !!message.isRecording;
        state.transportAnchorElapsed = Number(message.elapsedSeconds || 0);
        state.transportAnchorNow = performance.now();
      }

      function onMessage(event) {
        try {
          var message = JSON.parse(event.data);
          if (message.type === "replace_all") {
            replaceAll(message.events);
            return;
          }
          if (message.type === "notes_delta") {
            applyNoteDeltas(message.events);
            return;
          }
          if (message.type === "transport") {
            updateTransport(message);
          }
        } catch (error) {}
      }

      window.addEventListener("message", onMessage);
      document.addEventListener("message", onMessage);
      window.addEventListener("resize", resizeCanvas);

      resizeCanvas();
      requestAnimationFrame(renderFrame);
      post({ type: "ready" });
    })();
  <\/script>
</body>
</html>
`;

function roundTo(value: number, digits: number) {
  return Number(value.toFixed(digits));
}

function eventDuration(durationSeconds?: number) {
  if (durationSeconds && durationSeconds > 0.04) {
    return durationSeconds;
  }
  return DEFAULT_DURATION_SECONDS;
}

function buildRenderEvents(
  notes: LiveNoteLike[],
  chords: LiveChordLike[],
): LiveRenderEvent[] {
  const staged: (Omit<LiveRenderEvent, "id"> & { baseKey: string })[] = [];

  notes.forEach((note) => {
    staged.push({
      baseKey: `n:${note.midi_note}:${roundTo(note.time_seconds, 3)}:${note.hand ?? ""}`,
      midi: note.midi_note,
      start: roundTo(note.time_seconds, 3),
      end: roundTo(note.time_seconds + eventDuration(note.duration_seconds), 3),
      hand: note.hand,
      confidence: roundTo(note.confidence ?? 0.7, 2),
      kind: "note",
    });
  });

  chords.forEach((chord) => {
    const midiNotes = chord.midi_notes ?? [];
    midiNotes.forEach((midi) => {
      staged.push({
        baseKey: `c:${midi}:${roundTo(chord.time_seconds, 3)}:${chord.hand ?? ""}:${chord.label ?? ""}`,
        midi,
        start: roundTo(chord.time_seconds, 3),
        end: roundTo(
          chord.time_seconds + eventDuration(chord.duration_seconds),
          3,
        ),
        hand: chord.hand,
        confidence: roundTo(chord.confidence ?? 0.7, 2),
        kind: "chord",
      });
    });
  });

  staged.sort((left, right) => {
    if (left.start !== right.start) {
      return left.start - right.start;
    }
    if (left.midi !== right.midi) {
      return left.midi - right.midi;
    }
    return left.kind.localeCompare(right.kind);
  });

  const occurrences = new Map<string, number>();
  return staged.map((event) => {
    const occurrence = occurrences.get(event.baseKey) ?? 0;
    occurrences.set(event.baseKey, occurrence + 1);
    return {
      id: `${event.baseKey}:${occurrence}`,
      midi: event.midi,
      start: event.start,
      end: event.end,
      hand: event.hand,
      confidence: event.confidence,
      kind: event.kind,
    };
  });
}

function eventsEqual(left: LiveRenderEvent, right: LiveRenderEvent) {
  return (
    left.midi === right.midi &&
    left.start === right.start &&
    left.end === right.end &&
    left.hand === right.hand &&
    left.confidence === right.confidence &&
    left.kind === right.kind
  );
}

function compactDeltas(deltas: LiveRenderDelta[]) {
  const latest = new Map<string, LiveRenderDelta>();
  deltas.forEach((delta) => {
    const key = delta.op === "remove" ? delta.id : delta.event.id;
    latest.set(key, delta);
  });
  return Array.from(latest.values());
}

export default function LivePianoRollWebView({
  notes,
  chords = [],
  bpm = 120,
  elapsedSeconds,
  isRecording = false,
  height = 280,
}: LivePianoRollWebViewProps) {
  const webRef = useRef<WebView>(null);
  const readyRef = useRef(false);
  const latestEventsRef = useRef<Map<string, LiveRenderEvent>>(new Map());
  const previousEventsRef = useRef<Map<string, LiveRenderEvent>>(new Map());
  const pendingDeltasRef = useRef<LiveRenderDelta[]>([]);
  const flushTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestTransportRef = useRef({ bpm, elapsedSeconds, isRecording });
  const source = useMemo(
    () => ({
      html: LIVE_PIANO_ROLL_HTML,
      baseUrl: "https://live-roll.local/",
    }),
    [],
  );

  const postMessage = useCallback((message: Record<string, unknown>) => {
    if (!readyRef.current || !webRef.current) {
      return false;
    }

    webRef.current.postMessage(JSON.stringify(message));
    return true;
  }, []);

  const sendTransportSnapshot = useCallback(() => {
    const transport = latestTransportRef.current;
    postMessage({
      type: "transport",
      bpm: transport.bpm,
      elapsedSeconds: transport.elapsedSeconds,
      isRecording: transport.isRecording,
      sentAtMs: Date.now(),
    });
  }, [postMessage]);

  const sendFullSnapshot = useCallback(() => {
    if (
      !postMessage({
        type: "replace_all",
        events: Array.from(latestEventsRef.current.values()),
      })
    ) {
      return;
    }

    sendTransportSnapshot();
  }, [postMessage, sendTransportSnapshot]);

  const flushDeltaQueue = useCallback(() => {
    flushTimeoutRef.current = null;
    if (!readyRef.current || !webRef.current) {
      return;
    }

    const batched = compactDeltas(pendingDeltasRef.current);
    pendingDeltasRef.current = [];
    if (batched.length === 0) {
      return;
    }

    postMessage({ type: "notes_delta", events: batched });
  }, [postMessage]);

  const scheduleDeltaFlush = useCallback(() => {
    if (flushTimeoutRef.current) {
      return;
    }

    flushTimeoutRef.current = setTimeout(() => {
      flushDeltaQueue();
    }, DELTA_BATCH_MS);
  }, [flushDeltaQueue]);

  const enqueueDeltas = useCallback(
    (deltas: LiveRenderDelta[]) => {
      if (deltas.length === 0) {
        return;
      }

      pendingDeltasRef.current.push(...deltas);
      scheduleDeltaFlush();
    },
    [scheduleDeltaFlush],
  );

  useEffect(() => {
    latestTransportRef.current = { bpm, elapsedSeconds, isRecording };
    sendTransportSnapshot();
  }, [bpm, elapsedSeconds, isRecording, sendTransportSnapshot]);

  useEffect(() => {
    const nextEvents = buildRenderEvents(notes, chords);
    const nextMap = new Map(nextEvents.map((event) => [event.id, event]));
    const previousMap = previousEventsRef.current;

    latestEventsRef.current = nextMap;
    previousEventsRef.current = nextMap;

    if (!readyRef.current) {
      return;
    }

    if (previousMap.size === 0 && nextMap.size > 0) {
      sendFullSnapshot();
      return;
    }

    const deltas: LiveRenderDelta[] = [];

    previousMap.forEach((previousEvent, id) => {
      if (!nextMap.has(id)) {
        deltas.push({ op: "remove", id });
      }
    });

    nextMap.forEach((nextEvent, id) => {
      const previousEvent = previousMap.get(id);
      if (!previousEvent || !eventsEqual(previousEvent, nextEvent)) {
        deltas.push({ op: "upsert", event: nextEvent });
      }
    });

    enqueueDeltas(deltas);
  }, [notes, chords, enqueueDeltas, sendFullSnapshot]);

  useEffect(() => {
    return () => {
      if (flushTimeoutRef.current) {
        clearTimeout(flushTimeoutRef.current);
      }
    };
  }, []);

  const onMessage = useCallback(
    (event: WebViewMessageEvent) => {
      try {
        const message = JSON.parse(event.nativeEvent.data);
        if (message.type === "ready") {
          readyRef.current = true;
          pendingDeltasRef.current = [];
          sendFullSnapshot();
        }
      } catch {
        // Ignore malformed messages from the experimental renderer.
      }
    },
    [sendFullSnapshot],
  );

  return (
    <View style={[styles.container, { height }]}>
      <WebView
        ref={webRef}
        originWhitelist={["*"]}
        source={source}
        onMessage={onMessage}
        javaScriptEnabled
        domStorageEnabled
        scrollEnabled={false}
        bounces={false}
        style={styles.webview}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    borderRadius: 18,
    overflow: "hidden",
    backgroundColor: "rgba(255,255,255,0.95)",
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.18)",
  },
  webview: {
    flex: 1,
    backgroundColor: "transparent",
  },
});
