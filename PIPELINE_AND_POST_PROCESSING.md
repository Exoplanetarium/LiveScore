# LiveScore Pipeline and Post-Processing

Last organized: 2026-06-17

This document describes the pipeline currently wired in this repo, with the live
path separated from the fuller/classic analysis path. It also lists the
post-processing features that are active, fallback-only, or currently disabled.

## Active Live Pipeline

The live UI is the main working path.

Frontend entry point:
- `app/index.tsx`
- `hooks/useLiveRhythm.ts`
- `components/PianoSheetMusic.tsx`
- `components/osmdHTML.ts`

Backend entry point:
- `POST /live/audio-chunk` in `backend/main.py`
- live rhythm session state in `backend/live_rhythm.py`
- live neural transcription in `backend/detect_note.py`

Default live frontend switches:
- `USE_LIVE_NEURAL_PATH = true`
- `USE_LIVE_ADAPTIVE_ONSET_THRESHOLD_EXPERIMENT = true`
- `USE_LIVE_OSMD_ENGRAVING_EXPERIMENT = true`
- `LIVE_STREAM_CONTEXT_SEC = 1.8`
- `LIVE_STREAM_INFERENCE_INTERVAL_MS = 70`
- `LIVE_STREAM_TRUSTED_DELAY_MS = 180`
- `LIVE_STREAM_COMMIT_DELAY_MS = 500`
- `LIVE_STREAM_LOCK_DELAY_MS = 2000`
- `LIVE_STREAM_ANALYSIS_BATCH_MS = 250`
- `LIVE_OSMD_BATCH_MS = 500`

Backend chunk defaults:
- uploaded audio is decoded to mono float32 at 44.1 kHz
- `LIVE_CONTEXT_SEC = 2.4` unless overridden by env
- `OVERLAP_SAMPLES = 4096`, about 93 ms at 44.1 kHz
- `CONTEXT_SAMPLES = max(OVERLAP_SAMPLES, LIVE_CONTEXT_SEC * 44100)`
- `MIN_STREAM_ANALYSIS_SAMPLES = 16385`
- `CHUNK_END_GUARD_SEC = 0.025`
- `CHUNK_END_MICRO_EVENT_MAX_DURATION_SEC = 0.045`

Live data flow:
1. The app records chunked WAV audio and calls `processAudioChunk(...)`.
2. `useLiveRhythm.ts` posts the chunk to `/live/audio-chunk` with:
   - `session_id`
   - `file`
   - `use_neural_live`
   - `adaptive_onset_threshold`
   - optional `noise_profile`
3. `backend/main.py` decodes the uploaded bytes, prepends the session tail, and
   runs analysis on the full context window.
4. If `use_neural_live` is true, the backend calls
   `analyze_audio_live_neural(...)`.
5. Live neural transcription tries models in this order:
   - `enhanced_mel`, if available
   - `mel_baseline`, if available
   - `custom_velocity_weighted`, if available
6. Neural note events are converted into LiveScore notes/chords by onset group.
7. Chunk events are deduped, filtered at chunk boundaries, shifted to absolute
   session time, and recovered/deduped across overlap.
8. The shifted notes/chords are routed into `LiveTranscriptionSession.process_notes(...)`.
9. Stage 1 quantizes immediately for display.
10. Stage 2 runs deferred refinement when enough lookahead is available.
11. When refinement changes the score, the backend includes `all_notes` and
    `all_chords`, and the frontend replaces the displayed analysis result.
12. `PianoSheetMusic.tsx` converts backend-authored timing fields into
    MusicXML, and `osmdHTML.ts` renders with OSMD.

## Live Neural Detection

`analyze_audio_live_neural(...)` is the active low-latency detector.

## Developed Audio-to-MIDI Model

The primary model you developed is the enhanced mel audio-to-MIDI transcriber:

- training code: `backend/rhythm_training/train_enhanced_mel_transcriber.py`
- runtime wrapper: `GpuEnhancedMelTranscriber` in `backend/gpu_ops.py`
- default checkpoint path:
  `backend/rhythm_training/enhanced_mel_transcription.pt`
- fallback ancestor: `MelBaselineTranscriber` in
  `backend/rhythm_training/train_mel_baseline.py`

The model converts piano audio into MIDI-like note events:

- `onset_time`
- `offset_time`
- `midi_note`
- `velocity`
- onset/offset probabilities
- optional `note_value_class`, `note_value_name`, and `note_value_confidence`
- optional decode source tags such as `primary_onset`,
  `soft_polyphony_rescue`, or `lattice_calibrated`

Model input and feature frontend:

- audio is resampled to 16 kHz for model inference
- one STFT frontend: `n_fft = 2048`, `hop_length = 256`
- frame spacing is about 16 ms at 16 kHz
- feature tensor is a 229-bin log-mel spectrogram
- the enhanced model keeps the fast mel-only frontend from the mel baseline

Core architecture (forward pass, in order):

1. `FrequencyConvStack` compresses the mel bins while preserving the frequency
   axis, turning the log-mel input into per-frame spectral features.
2. Two readouts run on those features:
   - `PitchLocalReadout` uses learned per-key queries plus a pitch-frequency
     prior to extract local spectral evidence for each of the 88 keys.
   - a global projection feeds a Conformer stack for full-context modeling.
3. Global Conformer context and local pitch readout are combined into one hidden
   state per key per frame.
4. Dilated per-key temporal convolutions refine each key's sequence over time.
5. Optional cross-key attention then models interactions between keys.

Default enhanced size: `d_model = 384`, `n_layers = 10`, `n_heads = 8`,
`conv_channels = 192`.

Prediction heads:

- raw onset head
- raw offset head
- velocity head
- event-refinement GRU over onset, offset, and velocity-weighted event evidence
- frame/sustain head conditioned on refined onset and offset logits
- pedal head
- sounding-frame head
- 12-class note-value head

Runtime decoding:

- model outputs are accumulated over overlapping 10-second feature chunks
- overlapping frame probabilities are averaged
- `decode_enhanced_note_events(...)` finds onset peaks per key, locates offsets
  from the explicit offset head and frame drop, filters low-velocity events,
  pools note-value probabilities near onset, and dedupes repeated same-pitch
  events within the duplicate window
- optional soft-polyphony and lattice rescue paths exist, but they are off in
  the live default

## Why The Audio-to-MIDI Model Is Fast

The speed is mostly architectural and pipeline-level, not magic.

- The frontend is simple: one 229-bin log-mel spectrogram. It avoids the older
  heavier stack of CQT, chroma, HPSS, onset functions, and multi-resolution
  handcrafted features.
- The model runs at 16 kHz, not 44.1 kHz, so it processes far fewer audio
  samples while keeping enough piano bandwidth for transcription.
- Feature extraction runs on the GPU through the model wrapper, so the pipeline
  avoids a CPU feature bottleneck.
- Inference is a single vectorized forward pass over all frames and all 88 keys,
  instead of per-onset or per-pitch iterative analysis.
- The runtime wrapper uses `torch.no_grad()` and `model.eval()`, so training
  graph bookkeeping and dropout are not active.
- The live path sends only the rolling context window, not the whole recording,
  so normal live chunks stay short.
- For live audio, expensive optional recovery passes are off by default:
  soft-polyphony rescue, lattice rescue, and enhanced harmonic filtering.
- The decoder is mostly array peak picking, thresholding, offset lookup, velocity
  filtering, and duplicate suppression. That is much cheaper than another neural
  pass or a large search.
- The checkpoint is loaded once as a singleton, then reused for chunks.
- The live pipeline skips the full classic pass's neural beat tracking,
  regularized local tempo curve, full beat-grid rhythm quantization, acoustic
  validation, and notation proximity scoring. Those are used in classic/fallback
  contexts, not in the normal live neural chunk path.

In short: the model is fast because it made the right bargain. It keeps the
learned part powerful enough to predict onset, offset, sustain, velocity, pedal,
and note value directly, while keeping the input representation and live decode
path lean.

For `enhanced_mel`, runtime defaults are env-tunable:
- `LIVE_ENHANCED_ONSET_BASE`, default `0.60`
- `LIVE_ENHANCED_OFFSET_BASE`, default `0.35`
- `LIVE_ENHANCED_MIN_VELOCITY`, default `8`
- `LIVE_ENHANCED_DUPLICATE_WINDOW_SEC`, default `0.04`
- `LIVE_ENHANCED_MERGE_GAP_SEC`, default `0.0`
- `LIVE_ENHANCED_FILTER_HARMONICS`, default off
- `LIVE_ENHANCED_SOFT_POLYPHONY_RESCUE`, default off
- `LIVE_ENHANCED_LATTICE_RESCUE`, default off

For `mel_baseline`, runtime defaults are:
- `LIVE_ONSET_BASE`, default `0.46`
- frame threshold `0.5`

Adaptive onset thresholding is active by default. It adjusts the base threshold
from chunk loudness features:
- soft/sparse chunks lower the threshold slightly for recall
- loud/dense chunks raise it slightly for precision
- telemetry includes RMS, peak, crest factor, selected threshold, and profile

## Live Chunk Post-Processing

These features run around the live neural output before it enters the rhythm
session.

Active:
- note dedupe by MIDI and time tolerance, choosing the stronger event
- chord dedupe by pitch-set/signature and time tolerance
- chunk-end micro-event filter for very short events at the weak-context tail
- absolute time shift from context-window time to session time
- overlap recovery in the most recent 4096-sample recovery band
- recent-event duplicate suppression across chunk boundaries
- recent event retention for overlap matching
- per-chunk timing telemetry

Fallback-only:
- if live neural is unavailable, `/live/audio-chunk` falls back to
  `analyze_audio_optimized` or `analyze_audio`
- in that fallback path only, second-pass soft gap fill and live noise
  gate/contention pass are applied to the chunk

Live noise profiles are defined but matter mainly to the fallback/contention
path:
- `open`
- `balanced`, the current frontend default
- `clean`

The profile controls confidence and short-duration thresholds for tentative
notes/chords, plus the soft-onset thresholds used by second-pass recovery.

## Live Rhythm and Display Post-Processing

`backend/live_rhythm.py` is the active score-shaping layer after chunk detection.

Active tempo/grid features:
- onset clustering tolerance: `0.03` sec
- incremental tempo tracking from recent onset IOIs
- natural tempo octave guard enabled by default:
  `LIVE_TEMPO_OCTAVE_GUARD = 1`
- natural tempo range: 60-160 BPM
- shared `BeatGrid` with phase, period, subdivision, and anchor state

Active Stage 1 features:
- immediate coarse quantization in `quantize_batch_coarse`
- candidates: `0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0` beats
- grid-based duration when the beat grid is anchored
- duration fallback through fraction snapping
- triplets are suppressed in the coarse pass

Active Stage 2 features:
- deferred refinement after roughly 2 beats, bounded from 0.25 to 4 seconds
- minimum refinement window: 2 beats and 2 notes
- lookahead: 2 beats, capped at 8 notes
- Viterbi-style sequence decoding over a trailing onset window
- binary grid decode at 8 subdivisions per beat
- ternary grid decode at 12 subdivisions per beat
- binary/ternary selection by lower cost, with a ternary bias
- output fields include `note_value`, `note_divisions`, `dotted`, `triplet`,
  `start_grid_idx`, `start_beat`, `grid_subdivision`, and
  `quantization_confidence`

Active score-duration policy:
- `LIVE_SCORE_DURATION_POLICY = ioi_same_voice`
- `LIVE_VOICE_ASSIGNMENT = per_hand`
- display durations are based on IOI to the next event in the same voice lane
- per-hand voice assignment means one notation lane per staff by default

Active display-surface cleanup:
- final display start beats are recomputed from raw `time_seconds` at the final
  reported BPM, snapped by `LIVE_DISPLAY_BEAT_SNAP_DIV = 12`
- display note dedupe tolerance: `0.05` sec
- display note grouping tolerance: `0.03` sec
- display chord reconcile tolerance: `0.01` sec
- chord-group selection uses the learned pairwise display model if available
- otherwise chord-group selection falls back to a confidence/density/conflict
  heuristic
- reconciled chords are expanded into note events for note-level dedupe

## Frontend Notation and Rendering Post-Processing

`components/PianoSheetMusic.tsx` treats backend timing fields as authoritative
when available.

Active MusicXML behavior:
- `divisions = 24`
- `start_beat` is preferred over deriving position from wall-clock time
- `note_divisions` is preferred over recomputing duration from `note_value`
- canonical duration specs include whole through 32nd, dotted values, and
  triplet values down to 32nd triplet
- ties are generated when durations cross measure boundaries
- voices/staves are written into MusicXML
- backend triplet flags are rendered with time modification/tuplet notation
- playback is parsed back from generated MusicXML so playback follows notation

Active OSMD behavior:
- live score updates are batched by `LIVE_OSMD_BATCH_MS = 500`
- OSMD renders the generated MusicXML in a WebView
- OSMD camera/follow-tail behavior is managed in `osmdHTML.ts`
- Tone.js playback follows the parsed MusicXML, including tie continuation and
  ornament expansion where present in the XML

## Classic / Full-Pass Pipeline

The fuller file-based path still exists and is used by classic analysis or
fallbacks, not by the normal live chunk route when live neural succeeds.

Entry point:
- `analyze_audio(..., use_neural=True)` routes to `analyze_audio_neural(...)`
- `analyze_audio(..., use_neural=False)` routes to independent-hand or
  optimized DSP paths

Full neural path:
1. Load deterministic 44.1 kHz audio.
2. Resample to the selected model sample rate.
3. Try enhanced/mel/custom neural models, then ByteDance fallback.
4. Group note events by onset into notes/chords.
5. Detect tempo with neural beat tracking.
6. Fall back to IOI tempo if beat confidence is low.
7. Refine tempo through onset-grid alignment.
8. Build a regularized local beat grid.
9. Detect dominant subdivisions.
10. Quantize bass/treble notes and chords separately.
11. Apply acoustic-duration validation.
12. Apply unified rhythm post-processing.
13. Call coherence smoothing, currently disabled.
14. Detect triplets per hand.
15. Apply backend timing authority fields.
16. Compute notation proximity metrics.

Full-pass active post-processing:
- beat-grid quantization when enough beats exist
- IOI/ML fallback quantization when a beat grid is not available
- local tempo curve regularization
- tempo octave/grid refinement
- hand-separated quantization
- run pre-tagging before quantization
- acoustic duration cross-validation
- unified rhythm cleanup:
  - run normalization
  - tempo-relative gap/rest handling
  - similar-duration outlier correction
- per-hand triplet detection for notes and chords
- triplet stripping from grace notes
- backend timing authority: `start_beat`, `end_beat`, `local_beat_duration`,
  and `timing_authority`

Full-pass disabled or not active:
- ornament detection is explicitly disabled in `analyze_audio_neural`
- coherence smoothing returns notes unchanged
- the old broad memory-file pipeline mentions some older post-processing steps
  that are no longer active in the live path

## Short List: Active Post-Processing Features

Currently active in the main live path:
- adaptive onset threshold selection
- neural event grouping into notes/chords
- note/chord dedupe
- chunk-end micro-event suppression
- overlap recovery and duplicate suppression
- absolute session-time shifting
- incremental tempo tracking
- tempo octave guard
- immediate coarse beat-grid quantization
- deferred binary/ternary grid refinement
- same-voice IOI score-duration policy
- per-hand voice assignment
- final display beat resnap at reported BPM
- display note/chord reconciliation
- optional learned pairwise chord canonicalization when the model is available
- backend-authored MusicXML timing through `start_beat` and `note_divisions`
- tie generation across measures
- OSMD live engraving and Tone.js playback from generated MusicXML

Currently active only in fallback/classic paths:
- second-pass soft gap fill
- live noise gate/contention pass
- beat-grid rhythm quantization from neural beat tracking
- acoustic-duration validation
- unified rhythm cleanup
- per-hand triplet detection
- notation proximity scoring

Currently disabled:
- live enhanced soft-polyphony rescue by default
- live enhanced lattice rescue by default
- live enhanced harmonic filtering by default
- ornament detection in full neural analysis
- coherence smoothing
