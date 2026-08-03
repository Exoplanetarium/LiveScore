// Shared score data shapes. These mirror the interfaces defined inline in
// app/index.tsx and components/PianoSheetMusic.tsx; they are kept structurally
// identical so an AnalysisResult produced by the live screen can be persisted,
// reloaded, and handed back to PianoSheetMusic / the export helpers without
// casting. Defined here so the saved-score storage, the export helpers, and the
// Library screen can share one source of truth.

export interface OnsetResult {
  duration_seconds?: number;
  frame_index?: number;
  offset_frame?: number;
  offset_seconds?: number;
  time_seconds: number;
}

export interface NoteResult {
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

export interface ChordResult {
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

export interface AnalysisSummary {
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
}

export interface AnalysisResult {
  onsets: OnsetResult[];
  notes: NoteResult[];
  chords: ChordResult[];
  analysis_summary: AnalysisSummary;
}
