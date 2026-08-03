// Shared score export helpers. Used by both the live screen and the Library so
// the MIDI / MusicXML serialization lives in exactly one place.

import { Midi } from "@tonejs/midi";
import * as FileSystem from "expo-file-system";
import * as Sharing from "expo-sharing";
import { generateMusicXML } from "../components/PianoSheetMusic";
import type { AnalysisResult } from "../types/score";

/** Number of events that can actually be written out. */
export function exportableEventCount(analysis: AnalysisResult | null): number {
  if (!analysis) {
    return 0;
  }
  return (analysis.notes?.length ?? 0) + (analysis.chords?.length ?? 0);
}

// Encode a byte array as base64 without spreading the whole array into
// String.fromCharCode (which blows the call stack for large scores).
function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(
      null,
      Array.from(bytes.subarray(i, i + CHUNK)),
    );
  }
  return btoa(binary);
}

// Strip characters that are unsafe in a filename so a user-entered title can be
// used as the export base name.
function sanitizeBaseName(name: string, fallback: string): string {
  const cleaned = name
    .trim()
    .replace(/[^a-z0-9\-_ ]/gi, "")
    .replace(/\s+/g, "_")
    .slice(0, 60);
  return cleaned || fallback;
}

function resolveBpm(analysis: AnalysisResult, bpm?: number): number {
  return analysis.analysis_summary?.detected_bpm || bpm || 120;
}

/** Build a MIDI file from the score and open the system share sheet. */
export async function exportScoreAsMidi(
  analysis: AnalysisResult,
  bpm?: number,
  baseName = "live_score",
): Promise<void> {
  if (!FileSystem.cacheDirectory) {
    throw new Error("Cache directory is unavailable on this device.");
  }

  const tempo = resolveBpm(analysis, bpm);
  const midi = new Midi();
  midi.header.setTempo(tempo);

  const track = midi.addTrack();
  track.name = "Piano";
  track.channel = 0;

  for (const note of analysis.notes || []) {
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

  for (const chord of analysis.chords || []) {
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

  const fileUri = `${FileSystem.cacheDirectory}${sanitizeBaseName(baseName, "live_score")}.mid`;
  const base64 = bytesToBase64(midi.toArray());
  await FileSystem.writeAsStringAsync(fileUri, base64, {
    encoding: FileSystem.EncodingType.Base64,
  });
  await Sharing.shareAsync(fileUri, {
    mimeType: "audio/midi",
    dialogTitle: "Export MIDI",
    UTI: "public.midi-audio",
  });
}

/** Build a MusicXML file from the engraved score and open the share sheet. */
export async function exportScoreAsMusicXml(
  analysis: AnalysisResult,
  bpm?: number,
  baseName = "live_score",
): Promise<void> {
  if (!FileSystem.cacheDirectory) {
    throw new Error("Cache directory is unavailable on this device.");
  }

  const tempo = resolveBpm(analysis, bpm);
  const xml = generateMusicXML(
    analysis.notes || [],
    analysis.chords || [],
    "4/4",
    tempo,
    0,
  );

  const fileUri = `${FileSystem.cacheDirectory}${sanitizeBaseName(baseName, "live_score")}.musicxml`;
  await FileSystem.writeAsStringAsync(fileUri, xml, {
    encoding: FileSystem.EncodingType.UTF8,
  });
  await Sharing.shareAsync(fileUri, {
    mimeType: "application/vnd.recordare.musicxml+xml",
    dialogTitle: "Export MusicXML",
    UTI: "com.recordare.musicxml",
  });
}
