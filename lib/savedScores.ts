// Persistent storage for saved recordings / scores.
//
// Each saved recording is one JSON file under documentDirectory/saved_scores/.
// A lightweight index.json holds just the metadata for every saved recording so
// the Library list can render without reading (and parsing) every full score —
// the full AnalysisResult is only loaded when a recording is opened. This keeps
// the list view cheap regardless of how many notes each recording contains.

import * as FileSystem from "expo-file-system";
import type { AnalysisResult } from "../types/score";

export interface SavedScoreMeta {
  id: string;
  title: string;
  createdAt: number; // ms epoch
  bpm: number;
  durationSeconds: number;
  noteCount: number;
  chordCount: number;
}

export interface SavedScore extends SavedScoreMeta {
  analysis: AnalysisResult;
}

const SAVED_DIR = `${FileSystem.documentDirectory ?? ""}saved_scores/`;
const INDEX_PATH = `${SAVED_DIR}index.json`;

function scorePath(id: string) {
  return `${SAVED_DIR}${id}.json`;
}

async function ensureDir() {
  if (!FileSystem.documentDirectory) {
    throw new Error("Persistent storage is unavailable on this platform.");
  }
  const info = await FileSystem.getInfoAsync(SAVED_DIR);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(SAVED_DIR, { intermediates: true });
  }
}

async function readIndex(): Promise<SavedScoreMeta[]> {
  try {
    const info = await FileSystem.getInfoAsync(INDEX_PATH);
    if (!info.exists) {
      return [];
    }
    const raw = await FileSystem.readAsStringAsync(INDEX_PATH);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as SavedScoreMeta[]) : [];
  } catch {
    // A corrupt index should not brick the Library; treat it as empty.
    return [];
  }
}

async function writeIndex(entries: SavedScoreMeta[]) {
  await ensureDir();
  await FileSystem.writeAsStringAsync(INDEX_PATH, JSON.stringify(entries));
}

/** Metadata for every saved recording, newest first. */
export async function listSavedScores(): Promise<SavedScoreMeta[]> {
  const entries = await readIndex();
  return [...entries].sort((a, b) => b.createdAt - a.createdAt);
}

export interface SaveScoreInput {
  title: string;
  bpm: number;
  analysis: AnalysisResult;
}

/** Persist a recording and return its metadata. */
export async function saveScore(input: SaveScoreInput): Promise<SavedScoreMeta> {
  await ensureDir();

  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const meta: SavedScoreMeta = {
    id,
    title: input.title.trim() || "Untitled recording",
    createdAt: Date.now(),
    bpm: Math.round(input.bpm || input.analysis.analysis_summary?.detected_bpm || 120),
    durationSeconds: input.analysis.analysis_summary?.duration_seconds ?? 0,
    noteCount: input.analysis.notes?.length ?? 0,
    chordCount: input.analysis.chords?.length ?? 0,
  };

  const record: SavedScore = { ...meta, analysis: input.analysis };
  await FileSystem.writeAsStringAsync(scorePath(id), JSON.stringify(record));

  const index = await readIndex();
  index.push(meta);
  await writeIndex(index);

  return meta;
}

/** Load the full recording (including the AnalysisResult) by id. */
export async function loadSavedScore(id: string): Promise<SavedScore | null> {
  try {
    const info = await FileSystem.getInfoAsync(scorePath(id));
    if (!info.exists) {
      return null;
    }
    const raw = await FileSystem.readAsStringAsync(scorePath(id));
    return JSON.parse(raw) as SavedScore;
  } catch {
    return null;
  }
}

/** Delete a recording's file and remove it from the index. */
export async function deleteSavedScore(id: string): Promise<void> {
  await FileSystem.deleteAsync(scorePath(id), { idempotent: true });
  const index = await readIndex();
  await writeIndex(index.filter((entry) => entry.id !== id));
}

/** Rename a recording (updates both the index and the stored file). */
export async function renameSavedScore(id: string, title: string): Promise<void> {
  const nextTitle = title.trim() || "Untitled recording";
  const index = await readIndex();
  await writeIndex(
    index.map((entry) =>
      entry.id === id ? { ...entry, title: nextTitle } : entry,
    ),
  );
  const record = await loadSavedScore(id);
  if (record) {
    await FileSystem.writeAsStringAsync(
      scorePath(id),
      JSON.stringify({ ...record, title: nextTitle }),
    );
  }
}
