- backend/main.py now exposes POST /live/audio-chunk to combine overlap-aware chunk analysis with LiveTranscriptionSession.process_notes.
- Live session reset/delete also clears the matching \_stream_sessions overlap state so audio timing state does not leak across sessions.
- hooks/useLiveRhythm.ts now has processAudioChunk(fileUri) for direct chunk upload to the unified live endpoint.
- Current live backend shape is split into Stage 1 chunk audio detection + coarse quantization now, with deferred refinement still polled through /live/check-refinement and /live/get-all-notes.
- app/\_layout.tsx now exposes a Live tab at index and a Classic tab backed by app/classic.tsx, while hiding index_old/index_backup from the tab bar.
- app/index.tsx is now the live-first UI that records chunked WAV audio, sends it through useLiveRhythm.processAudioChunk, and finalizes via the live session endpoints.

- components/PianoSheetMusic.tsx had a fixed 560px score viewport; live-session layout now needs compact mode plus caller-driven viewportHeight so the score and record controls can coexist on shorter phones.
