# Live Neural Latency Budget

## Purpose

This document defines the latency targets and success criteria for adding a neural pathway to the live score while preserving a genuinely live user experience.

The core constraint is simple:

- The live score must feel immediate while recording.
- A neural model should contribute during live capture, not only at finalize time.
- The heavy music-analysis steps that improve notation quality must not block the fast path.

## Product Goals

### Primary Goals

- Add a neural-network pathway to live transcription during recording.
- Keep the fast path under 500 ms per run on a warmed GPU for short live windows.
- Accept in-memory audio arrays directly, with no temp-file write in the live path.
- Return provisional note events fast enough that the score updates feel live.
- Preserve the higher-quality full pass on stop/finalize.

### Secondary Goals

- Allow background retro-correction of recent live notes without interrupting the fast path.
- Reuse existing neural wrappers where possible instead of inventing a separate model stack.
- Keep the live neural path measurable with stage-by-stage timing logs.

### Non-Goals For The Fast Path

- Full beat detection on every chunk.
- Full rhythm quantization on every chunk.
- Full classic-grade note/chord cleanup on every chunk.
- Reconstructing final engraved notation quality before the user stops recording.

## Design Principle

The live neural system should be split into three paths:

1. Fast path
   Neural note-event inference on a short in-memory window. This path is latency-critical.

2. Background correction path
   Longer rolling-window retro-correction that can revise recent notes asynchronously.

3. Finalize path
   The full classic-quality pass that runs when recording stops.

The fast path should stop at provisional note/chord events plus minimal grouping. It should not do full musical cleanup inline.

## Fast Path Target

### User-Visible Goal

For a live chunk of about 1.0 to 1.5 seconds, the system should return a neural-assisted result quickly enough that the UI can update without feeling delayed.

### Hard Target

- P95 end-to-end fast-path latency: less than 500 ms

### Preferred Target

- P50 end-to-end fast-path latency: less than 350 ms
- P95 end-to-end fast-path latency: less than 500 ms

### Real-Time Factor Target

For a window of length $W$ seconds, the fast path should satisfy:

$$
RTF = \frac{\text{latency}}{W}
$$

For a 1.2 second chunk, a 500 ms budget implies:

$$
RTF < \frac{0.5}{1.2} \approx 0.42
$$

That should be treated as the upper bound, not the desired steady-state operating point.

## Fast Path Latency Budget

This is the target budget for one live neural run on a warmed GPU.

| Stage                              |     Target | Hard Ceiling | Notes                                                     |
| ---------------------------------- | ---------: | -----------: | --------------------------------------------------------- |
| Audio handoff / buffer assembly    |    5-15 ms |        25 ms | Includes array slicing and overlap handling               |
| Resample to model sample rate      |   10-35 ms |        50 ms | Only if source is not already at model SR                 |
| Feature extraction                 |  40-100 ms |       130 ms | Log-mel or model-specific front end                       |
| Model inference                    |  80-180 ms |       250 ms | Main GPU forward pass                                     |
| Note decode / grouping             |   20-60 ms |        90 ms | Frame-to-event conversion and basic simultaneity grouping |
| Live-session merge / serialization |   10-30 ms |        40 ms | Merge into live session and return JSON                   |
| Total                              | 165-420 ms |       500 ms | Fast path must stay below this                            |

The point of this budget is not precision. The point is to force discipline: every stage has to justify its presence in the live critical path.

## Background Correction Budget

The background correction path is allowed to be slower because it is not user-blocking.

Suggested target:

- Window length: 2.5 to 5.0 seconds
- Cadence: every 2 to 4 live updates, or every 2 to 3 seconds
- Budget per run: 700 to 1500 ms
- Output: corrected trailing notes only, not a full re-analysis of the entire recording

This path can include more cleanup than the fast path, but it still should avoid the full finalize pipeline.

## Finalize Budget

The finalize path is allowed to be materially slower because the user has already stopped recording.

Suggested goal:

- Full classic-style pass on the accumulated recording
- Target: fastest possible while preserving quality
- No strict 500 ms requirement

Finalize remains the quality bridge between live provisional output and final notation.

## What Determines Neural Speed In This Repo

The neural latency in this codebase is primarily driven by:

1. Audio window length
   Longer windows create more frames, more feature work, more model work, and more decode work.

2. Feature extraction cost
   This includes mel computation or any other front-end representation.

3. Sequence length through the model
   Transformer-based components get more expensive as frame count grows.

4. GPU warm state
   Cold starts, model loads, and one-time setup costs must be excluded from the live budget.

5. Transfer overhead
   Moving arrays to GPU and synchronizing work matters, especially on short windows.

6. Decode complexity
   Turning logits into note events is not free and scales with frame count and prediction density.

7. Post-processing scope
   Beat detection, tempo refinement, subdivision analysis, and rhythm quantization can easily consume the entire latency budget if kept in the fast path.

## Scope Of The Fast Neural Path

The fast path should include only:

1. In-memory audio input
2. Optional resampling
3. Neural transcription
4. Lightweight decode to note events
5. Minimal chord grouping
6. Merge into live session

The fast path should exclude:

1. Full beat tracking
2. Tempo-grid search
3. Full rhythm quantization
4. Final notation cleanup
5. Any file write or reload

## Recommended Architecture

### Step 1: Narrow The Live Neural Contract

Create a live neural inference function that accepts:

- audio array
- sample rate
- onset/frame thresholds
- optional overlap context

And returns:

- provisional note events
- provisional chord groups
- stage timings

### Step 2: Keep The Current Live Rhythm Layer

Feed those provisional note events into the existing live session machinery so the UI still benefits from live merging and deferred refinement.

### Step 3: Add Background Retro-Correction

Run a longer-window neural pass asynchronously and use it only to revise recent notes, not to block the immediate display.

### Step 4: Keep Finalize As The Quality Path

Do not try to force the fast path to become the finalize path.

## Success Criteria

The live neural effort should be considered successful if all of the following are true:

1. Live neural fast path accepts array input directly.
2. No temp-file write is required for live neural inference.
3. P95 latency stays under 500 ms for short live windows on warmed GPU.
4. The UI receives provisional neural-assisted notes during recording.
5. Finalize still improves quality over the in-progress display.
6. Stage timings are logged so regressions are visible immediately.

## Instrumentation Requirements

Every live neural run should log at least:

- audio duration ms
- resample ms
- feature extraction ms
- model inference ms
- decode ms
- merge ms
- total ms
- real-time factor

Without this, there is no real latency budget, only a hope.

## Initial Implementation Sequence

1. Refactor the neural path to accept array input plus sample rate.
2. Build a minimal fast-path neural function that stops after note-event decode.
3. Add stage timing logs for every run.
4. Wire the fast neural result into the existing live session pipeline.
5. Benchmark 1.0 s, 1.2 s, and 1.5 s windows.
6. Only after the fast path is stable, add a background retro-correction worker.

## Decision Rule

If the fast neural path cannot consistently stay under the latency budget for short windows, then it should not replace the current live detector outright. In that case, it should run as a background correction path while the existing live detector continues to provide the immediate display.
