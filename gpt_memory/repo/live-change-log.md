# LiveScore Change Log

This is the running log for Copilot-assisted changes in the live transcription,
benchmark, and live-notation evaluation workflow.

Update rule:

- Append a dated entry after each material code change.
- Record benchmark validations and rollbacks, not just successful edits.
- Keep this file as the current source of truth for the May 2026 live-accuracy work.

## 2026-05-29

### Earlier session changes already landed

- Restored live-tab MIDI export so generated scores can be downloaded again.
- Updated the live screen layout so the start-live-session area and the generated score can scroll vertically.
- Fixed live notation tempo selection in `backend/detect_note.py` so tempo refinement uses better onset clustering and avoids doubled-tempo bias on notation output.
- Removed the old finalize path from the live flow so stop-time behavior is simpler and more consistent.
- Added prior-art and benchmark-comparison docs:
  - `gpt_memory/live_transcription_prior_art_and_bar.md`
  - `gpt_memory/live_paper_benchmark_matrix.md`

### Benchmark harness integration

- `backend/test_ensemble_accuracy.py`
  - Added strict onset-tolerance sweeps at 10/20/30 ms.
  - Added CLI support for overriding the strict onset tolerance list.
  - Summary output now reports strict onset F1 alongside the existing 50 ms note metrics.

- `backend/test_experiment.py`
  - Added the same strict onset-tolerance sweeps at 10/20/30 ms.
  - Threaded `audio_time_sec` through live and retro display snapshots.
  - Split algorithmic latency from wall-clock display latency.
  - Added clip-level failure buckets.
  - Added per-boundary miss diagnostics and tag summaries.
  - Added benchmark JSON outputs that now expose the new diagnostic fields.

### Diagnostic findings from the fixed manifest

- Full-manifest benchmark run on `backend/live_benchmark_replay_auto_v2.json` showed:
  - `boundary_miss_failure` on 47 / 48 clips.
  - `control_boundary_miss=1034`.
  - `no_control_coarse_candidate=1032`.
- Interpretation: most remaining live boundary failures were happening before retro correction, because the live path often produced no coarse candidate at all near the missed boundary note.

### Live overlap recovery fix

- `backend/main.py`
  - Added short-lived stream-session memory for recently emitted notes and chords.
  - Changed overlap handling so note/chord events found in the next chunk's overlap prefix are only suppressed when they truly duplicate something already emitted.
  - Added overlap recovery counters to `stream_info` for debugging.

- Validation:
  - Hard clip `clip_047` (“Etude Op. 42, Nos. 4 & 5”) improved from `28/51` matched notes to `31/51`.
  - The same clip improved from `4/13` to `7/13` matched boundary notes.
  - Duplicate notes on that clip were brought back down to `0` after tightening the overlap duplicate matcher.

- Full-manifest aggregate improvement versus the pre-fix run:
  - Control average F1: `0.4072 -> 0.4227`
  - Control average boundary miss rate: `0.7099 -> 0.6603`
  - Control boundary matched-note sum: `346 -> 398`
  - Control duplicates per 100 notes: `0.1186 -> 0.1082`
  - Treatment moved similarly:
    - F1: `0.4067 -> 0.4213`
    - Boundary miss rate: `0.7099 -> 0.6603`
    - Boundary matched-note sum: `346 -> 398`
    - Duplicates per 100 notes: `0.1186 -> 0.1082`

- Result files:
  - Baseline benchmark output: `backend/live_benchmark_replay_auto_v2_results_20260529.json`
  - Post-overlap-fix benchmark output: `backend/live_benchmark_replay_auto_v2_results_20260529_overlapfix_v2.json`

### Attempted and rolled back

- `backend/main.py`
  - Tried turning the chunk-tail micro-event suppression into a one-chunk deferral/release mechanism so short notes near the chunk end could be reconsidered on the next chunk.
  - Validation on `clip_047` showed no change.
  - Full-manifest comparison against the validated overlap-fix build also showed no aggregate change.
  - The experiment was rolled back to avoid carrying neutral complexity.

### Current state after rollback

- The overlap recovery fix is kept.
- The tail-deferral experiment is not kept.
- The dominant remaining live-boundary tag is still `no_control_coarse_candidate`, which means the next likely improvement surface is upstream raw live-neural recall near chunk boundaries rather than seam bookkeeping or tail-event deferral.

### Chord-aware note-level benchmark fix

- `backend/test_experiment.py`
  - Added a helper that expands emitted chord payloads into note-shaped events before note-level scoring.
  - Updated both streamed live replay and direct live-neural replay to score combined `notes + chord member pitches` instead of `notes` alone.
  - Kept the live session and API payload shape unchanged; this change fixes benchmark accounting rather than changing the underlying model output.

- Why this mattered:
  - The live neural converter can emit simultaneous pitches as `chords` instead of individual `notes`.
  - The benchmark had been evaluating only `result["notes"]`, so polyphonic clips were severely undercounted at note level and many misses were mislabeled as `no_control_coarse_candidate`.

- Focused validation on `clip_010` (`Etude "Pour les accords"`, start `0s`):
  - Control F1: `0.0182 -> 0.8755`
  - Control matched notes: `1 / 108 -> 102 / 108`
  - Control boundary matched notes: `1 / 34 -> 33 / 34`
  - Boundary tag count on the clip collapsed from a near-total miss pattern to a single remaining `no_control_coarse_candidate` miss.

- Full-manifest aggregate improvement versus `backend/live_benchmark_replay_auto_v2_results_20260529_overlapfix_v2.json`:
  - Control average F1: `0.4227 -> 0.8320`
  - Treatment average F1: `0.4213 -> 0.8282`
  - Retro-correction average F1: `0.4260 -> 0.8265`
  - Control boundary matched-note sum: `398 -> 1251`
  - Treatment boundary matched-note sum: `398 -> 1251`
  - Retro-correction boundary matched-note sum: `416 -> 1257`
  - `control_boundary_miss`: `982 -> 129`
  - `no_control_coarse_candidate`: `980 -> 126`

- Result files:
  - Focused validation output: `backend/_tmp_clip_010_chord_eval.json`
  - Updated full-manifest output: `backend/live_benchmark_replay_auto_v2_results_20260529_chordnotes.json`

- Interpretation:
  - This is a benchmark-correctness win that materially improves reported overall F1 by counting polyphonic chord emissions at note level.
  - The remaining `no_control_coarse_candidate` cases are now much closer to the true upstream recall problem, so future boundary debugging should be less confounded by representation loss.

### Ranked shortlist generated from the latest validated benchmark

- Added `gpt_memory/repo/live-benchmark-shortlist-20260529.md`.
- Ranked three target sets from `backend/live_benchmark_replay_auto_v2_results_20260529_overlapfix_v2.json`:
  - boundary-recall first attack set,
  - general low-F1 set,
  - stabilization / revision-churn set.
- Immediate debugging recommendation stays the same:
  - start with the boundary recall set,
  - instrument raw live-neural misses within the first `100 ms` after chunk boundaries,
  - determine whether failures are absent in the raw neural output or dropped later in the live path.

### Shortlist reranked against the chord-aware benchmark

- Updated `gpt_memory/repo/live-benchmark-shortlist-20260529.md` to use `backend/live_benchmark_replay_auto_v2_results_20260529_chordnotes.json` instead of the older `overlapfix_v2` result file.
- The old shortlist was superseded because several former low-F1 / boundary-catastrophe clips were benchmark-accounting artifacts from ignoring chord member pitches at note level.
- New boundary-first attack set after reranking:
  - `clip_026`,
  - `clip_017`,
  - `clip_006`,
  - `clip_027`,
  - `clip_016`,
  - `clip_019`.
- Updated headline metrics reflected in the shortlist:
  - Control average F1: `0.8320`
  - Control average boundary miss rate: `0.0782`
  - Control boundary matched-note sum: `1251`
  - `control_boundary_miss`: `129`
  - `no_control_coarse_candidate`: `126`

### Final display accuracy metrics added

- `backend/test_experiment.py`
  - Added a chord-aware `final_display_note_events` surface for live benchmark runs by combining displayed notes with stored coarse chords expanded back into note-level events.
  - Added final-display note metrics against GT MIDI:
    - `display_note_precision`, `display_note_recall`, `display_note_f1`
    - `display_offset_f1`
    - `display_note_value_accuracy`
    - `display_strict_onset_metrics`
  - Added onset-cluster structure metrics on the final displayed score state:
    - `display_cluster_precision`, `display_cluster_recall`, `display_cluster_f1`
    - `display_cluster_avg_jaccard`
    - `display_cluster_overclustered_matches`, `display_cluster_underclustered_matches`, `display_cluster_pitch_conflict_matches`
    - `display_cluster_unmatched_ground_truth`
  - Kept the older notation latency / stabilization metrics intact so historical latency comparisons remain available.

- Why this mattered:
  - The existing `final_display_notes` benchmark surface was note-only and could miss chord members entirely because live chords were stored separately from refined notes.
  - The new final-display metrics measure the post-score symbolic state more directly, while the onset-cluster metric specifically penalizes bad chord snapping instead of giving full credit for pitch-set-only matches.

- Focused validation on `clip_010` (`Etude "Pour les accords"`, start `0s`):
  - Raw note F1 stayed at `0.8755`.
  - Final display note F1 came out higher at `0.9273` because the displayed symbolic state retains more of the chord-member pitches than the raw note-only stream.
  - Final display cluster F1 came out much lower at `0.5854`, with `avg_jaccard=0.9140`, `overclustered_matches=3`, `underclustered_matches=2`, `pitch_conflict_matches=2`.
  - Interpretation: this clip confirms the user-facing issue directly. The final displayed score contains many correct pitches, but the onset-cluster structure is still materially wrong because notes are being snapped together or otherwise grouped incorrectly.

- Result file:
  - Focused validation output: `backend/_tmp_clip_010_display_accuracy.json`

## 2026-05-30

### Full-manifest display-structure rerank

- Ran the full manifest with the new final-display metrics and saved the result to `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster.json`.
- Added a new structure-focused shortlist file: `gpt_memory/repo/live-benchmark-shortlist-20260530-displaycluster.md`.
- This rerank is intentionally separate from the older note-level shortlist so raw boundary-recall work and displayed-score-structure work can be tracked independently.

- Headline display metrics from the manifest:
  - Control average final display note F1: `0.8566`
  - Control average final display cluster F1: `0.5967`
  - Treatment average final display note F1: `0.8525`
  - Treatment average final display cluster F1: `0.5942`
  - Retro average final display note F1: `0.5932`
  - Retro average final display cluster F1: `0.4723`

- Interpretation:
  - The live path is materially better at recovering pitches than preserving the final displayed chord / onset structure.
  - Adaptive thresholding is not moving final display structure much on this manifest.
  - Retro correction is substantially worse than the baseline live path on final display structure.
  - The next likely main-pipeline improvement surface is note-to-chord grouping, onset-cluster formation, or later quantization / display retention rather than threshold selection alone.

- Worst control clips by final display cluster F1:
  - `clip_031` (`0.1194`)
  - `clip_017` (`0.1728`)
  - `clip_024` (`0.1905`)
  - `clip_006` (`0.2439`)
  - `clip_035` (`0.3099`)
  - `clip_012` (`0.3448`)

- Largest control note-vs-structure gaps:
  - `clip_017` (`0.6444`)
  - `clip_006` (`0.6329`)
  - `clip_031` (`0.6204`)
  - `clip_024` (`0.5984`)
  - `clip_035` (`0.4492`)
  - `clip_044` (`0.4216`)

### Runtime grouping fix for structure-gap clips

- `backend/detect_note.py`
  - Replaced the fixed-width live neural simultaneity grouping with `_group_neural_note_events_by_onset(...)`, which shrinks the allowed onset span as a candidate chord grows.
  - Wired both the offline neural conversion path and `_convert_neural_note_events_to_results(...)` to the same adaptive grouping helper so benchmark and runtime stay aligned.

- `backend/live_rhythm.py`
  - Updated `LiveTranscriptionSession.process_notes(...)` so both `notes` and `chords` contribute onset times to the live tempo tracker before coarse quantization.

- `backend/test_experiment.py`
  - Mirrored the same combined note+chord onset feed in `_process_live_session_chunk(...)` so focused benchmark runs reflect the runtime timing path.

- Focused validation on the two highest-priority structure-gap clips:
  - Result file: `backend/_tmp_clip_017_031_adaptive_grouping_final.json`
  - `clip_017`
    - control display cluster F1: `0.1728 -> 0.2927`
    - exact cluster matches: `7 -> 12`
    - overclustered matches: `7 -> 5`
    - pitch-conflict matches: `9 -> 7`
    - display note F1 stayed flat at `0.8173`
  - `clip_031`
    - control display cluster F1 stayed flat at `0.1194`
    - overclustered matches stayed at `15`
    - interpretation: this clip is still controlled by a nearby structure surface beyond the first adaptive grouping change.

- Attempted and rolled back immediately:
  - Tightened the adaptive grouping thresholds further to force earlier splits in smaller clusters.
  - Focused validation showed no gain on `clip_031` and worse note-path behavior on `clip_017` (higher duplicates / lower note metrics), so that tighter variant was reverted.

### Full-manifest results for the kept runtime grouping fix

- Ran the full manifest with the kept runtime change and saved the result to `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_runtimefix.json`.

- Aggregate deltas versus `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster.json`:
  - Control average display cluster F1: `0.5967 -> 0.6056` (`+0.0090`)
  - Control average display note F1: `0.8566 -> 0.8570` (`+0.0004`)
  - Control average cluster Jaccard: `0.8517 -> 0.8541` (`+0.0025`)
  - Control average onset-alignment recall: `0.9600 -> 0.9652` (`+0.0052`)
  - Control boundary miss rate: unchanged at `0.0782`
  - Control duplicates per 100 notes: `5.9175 -> 6.0252` (`+0.1077`)

- Same pattern on other arms:
  - Treatment average display cluster F1: `0.5942 -> 0.6027` (`+0.0085`)
  - Retro average display cluster F1: `0.4723 -> 0.4883` (`+0.0159`)

- Clip-level control winners:
  - `clip_017`: `0.1728 -> 0.2927` (`+0.1198`)
  - `clip_012`: `0.3448 -> 0.4483` (`+0.1034`)
  - `clip_015`: `0.5298 -> 0.6234` (`+0.0936`)
  - `clip_001`: `0.7442 -> 0.8182` (`+0.0740`)
  - `clip_019`: `0.4474 -> 0.4805` (`+0.0332`)

- Clip-level control regressions:
  - `clip_006`: `0.2439 -> 0.1905` (`-0.0534`), with pitch-conflict matches increasing by `6`
  - `clip_022`: `0.4310 -> 0.4103` (`-0.0208`)

- Coverage summary:
  - Control clips improved: `11`
  - Control clips regressed: `3`
  - Control clips unchanged: `34`

- Interpretation:
  - The kept adaptive grouping change is a net positive for displayed score structure at manifest scale.
  - The gain is concentrated in denser polyphonic clips, which matches the original hypothesis.
  - Boundary recall is unchanged, so this is a grouping / notation-structure win rather than a recall win.
  - The next follow-up should target the remaining regressions, especially `clip_006`, before promoting this further.

### Follow-up display-surface experiments after the runtime grouping fix

- Attempted but not kept: rebuilt the displayed note/chord surface from `get_all_notes()` plus regrouped note events.
  - Focused validation on `clip_031`, `clip_024`, `clip_006`, and `clip_017` showed no gain on the pure overclustering failures (`clip_031`, `clip_024`) and a material regression on `clip_017`.
  - The regrouping layer was removed from the kept path.

- Attempted but not kept: changed `DeferredRefinementState.get_all_notes()` to replace pending notes by `(time_seconds, midi_note)` identity rather than the older time-only replacement.
  - Focused validation improved `clip_017` further, but the full manifest regressed aggregate control display cluster F1 from `0.6056` to `0.6041`.
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_getallnotesfix.json`
  - This variant was reverted.

### Kept 10 ms display chord reconciliation fix

- `backend/live_rhythm.py`
  - Added a narrow display-only chord reconciliation pass inside `_build_display_surface(...)`.
  - Chords whose displayed onset times land within `10 ms` of each other are treated as overlapping alternative hypotheses.
  - The display surface now keeps a single primary chord candidate per such window, preferring the earliest onset and breaking exact-time ties by confidence and chord size.
  - The note surface is unchanged; only the displayed chord set and derived `note_events` are reconciled.

- Why this mattered:
  - On the worst structure clips, the live session was retaining multiple near-duplicate chord hypotheses from overlapping chunks.
  - Those hypotheses were all being expanded into the final display note-event surface, so the displayed onset cluster inherited the union of several incompatible chord pitch sets.
  - A narrow `10 ms` reconciliation window was the smallest tested range that improved the pure overclustering clips without collapsing too many legitimate nearby events.

- Focused validation:
  - Result file: `backend/_tmp_display_chord_reconcile_focus.json`
  - `clip_031`: control display cluster F1 `0.1194 -> 0.1493`
  - `clip_024`: control display cluster F1 `0.1905 -> 0.2381`
  - `clip_017`: control display cluster F1 `0.2927 -> 0.3133`
  - `clip_006`: control display cluster F1 `0.1905 -> 0.1667`

- Full-manifest results:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_chordreconcile10ms.json`
  - Aggregate deltas versus the kept runtime grouping fix (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_runtimefix.json`):
    - Control average display cluster F1: `0.6056 -> 0.6064` (`+0.0007`)
    - Control average display note F1: `0.8570 -> 0.8558` (`-0.0012`)
    - Treatment average display cluster F1: `0.6027 -> 0.6030` (`+0.0004`)
    - Retro average display cluster F1: unchanged at `0.4883`

- Biggest control improvements versus the kept runtime grouping fix:
  - `clip_024`: `0.1905 -> 0.2381` (`+0.0476`)
  - `clip_032`: `0.7009 -> 0.7350` (`+0.0342`)
  - `clip_031`: `0.1194 -> 0.1493` (`+0.0299`)
  - `clip_017`: `0.2927 -> 0.3133` (`+0.0206`)
  - `clip_038`: `0.8649 -> 0.8829` (`+0.0180`)

- Biggest control regressions versus the kept runtime grouping fix:
  - `clip_027`: `0.6042 -> 0.5474` (`-0.0568`)
  - `clip_035`: `0.3099 -> 0.2817` (`-0.0282`)
  - `clip_015`: `0.6234 -> 0.5974` (`-0.0260`)
  - `clip_006`: `0.1905 -> 0.1667` (`-0.0238`)
  - `clip_018`: `0.6744 -> 0.6512` (`-0.0233`)

- Coverage summary:
  - Control clips improved: `9`
  - Control clips regressed: `7`
  - Control clips unchanged: `32`

- Interpretation:
  - This is a small but real manifest-scale structure win on top of the kept runtime grouping fix.
  - The gain comes specifically from reducing union-style overclustering in the final displayed score state.
  - The cost is a slight note-F1 drop and a handful of mid-pack regressions, so any next follow-up should target the new losers (`clip_027`, `clip_035`, `clip_015`, `clip_006`, `clip_018`) rather than widen the reconciliation window further.

### Kept local subset-search display arbitration

- `backend/live_rhythm.py`
  - Replaced the display-side single-winner chord reconciliation with a small subset-search over each `10 ms` overlapping chord group.
  - The new arbitration keeps the subset of candidates with the best confidence-plus-density score while penalizing subset / high-overlap pitch conflicts between candidates.
  - For unusually large overlap groups (`> 7` candidates), the code falls back to the old single-primary selection to avoid combinatorial blowups.

- Why this mattered:
  - The `10 ms` single-winner fix helped the worst overclustering clips, but it was still too aggressive on some mixed clips that needed more than one compatible hypothesis to survive the display pass.
  - A tiny local search is the first version in this repo that can represent: "collapse incompatible alternatives, but keep mutually compatible chord hypotheses from the same overlap window."

- Attempted and not kept during this pass:
  - A simpler overlap-based winner-picking variant that only collapsed subset / high-Jaccard chord pairs.
  - Probe results were worse than the current `10 ms` build, so that variant was not implemented.

- Focused validation:
  - Result file: `backend/_tmp_subsetsearch_focus.json`
  - Versus `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_chordreconcile10ms.json`:
    - `clip_031`: unchanged at `0.1493`
    - `clip_024`: unchanged at `0.2381`
    - `clip_017`: unchanged cluster F1 at `0.3133`, note F1 `0.8072 -> 0.8112`
    - `clip_006`: unchanged at `0.1667`
    - `clip_027`: `0.5474 -> 0.5684` (`+0.0211`)
    - `clip_035`: `0.2817 -> 0.3099` (`+0.0282`)

- Full-manifest results:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_subsetsearch.json`
  - Aggregate deltas versus `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_chordreconcile10ms.json`:
    - Control average display cluster F1: `0.6064 -> 0.6067` (`+0.0004`)
    - Control average display note F1: `0.8558 -> 0.8574` (`+0.0016`)
    - Treatment average display cluster F1: `0.6030 -> 0.6040` (`+0.0009`)
    - Treatment average display note F1: `0.8515 -> 0.8534` (`+0.0019`)
    - Retro average display cluster F1: unchanged at `0.4883`

- Biggest control improvements versus the prior `10 ms` build:
  - `clip_035`: `0.2817 -> 0.3099` (`+0.0282`)
  - `clip_027`: `0.5474 -> 0.5684` (`+0.0211`)
  - `clip_020`: `0.4841 -> 0.4968` (`+0.0127`)

- Biggest control regressions versus the prior `10 ms` build:
  - `clip_022`: `0.4274 -> 0.4103` (`-0.0171`)
  - `clip_028`: `0.5333 -> 0.5185` (`-0.0148`)
  - `clip_019`: `0.4935 -> 0.4805` (`-0.0130`)

- Coverage summary:
  - Control clips improved: `3`
  - Control clips regressed: `3`
  - Control clips unchanged: `42`

- Interpretation:
  - This is another modest but real manifest-scale gain on top of the already-kept display reconciliation build.
  - Unlike the prior `10 ms` winner-take-all patch, this version also improves average display note F1 instead of sacrificing it.
  - The remaining follow-up should target the new regressions (`clip_022`, `clip_028`, `clip_019`) and inspect whether the subset-search scoring is being too permissive on specific overlap shapes.

### Attempted learned local subset scorer

- Kept infrastructure, not kept as the active runtime policy:
  - `backend/main.py`
    - Added `_timing_ms.display_state` timing on live refresh / get-all-notes responses so display-state assembly cost can be measured independently from chunk inference.
  - `backend/display_chord_subset_model.py`
    - Added a plain-Python linear subset scorer loader plus reusable subset feature extraction.
  - `backend/train_display_chord_subset_model.py`
    - Added a replay-driven trainer that enumerates all local overlapping-chord subsets, scores them against ground-truth local pitch-set F1, and exports a small Ridge model JSON.
  - `backend/live_rhythm.py`
    - Added an optional learned path for `_select_display_chord_subset(...)` that only activates when a model JSON is explicitly supplied via `LIVE_DISPLAY_CHORD_MODEL` (or the default file exists). Without a model file, runtime behavior stays on the kept heuristic subset-search path.

- Why this was tried:
  - The goal was to replace the hand-tuned confidence / overlap weights with a data-driven local scorer while preserving the same bounded subset search and avoiding any second neural model in the live path.
  - This keeps runtime inference CPU-cheap and targets the actual cluster-F1 discrepancy: selecting the best union of near-simultaneous chord hypotheses for one displayed onset cluster.

- Holdout training sanity check:
  - Trained on `42` manifest clips (excluding `clip_031`, `clip_024`, `clip_017`, `clip_006`, `clip_027`, `clip_035`).
  - Exported model summary:
    - dataset groups: `791`
    - dataset subsets: `1618`
    - train `R^2`: `0.8827`
    - per-group exact oracle choice rate: `0.9646`
  - Interpretation: the linear scorer fit the local replay targets well enough to benchmark, so the next question was generalization, not basic trainability.

- Focused holdout benchmark:
  - Result file: `backend/_tmp_learned_subset_focus.json` (temporary; not kept)
  - Versus the kept subset-search build (`backend/_tmp_subsetsearch_focus.json`):
    - `clip_031`: `0.1493 -> 0.1194` (`-0.0299`)
    - `clip_024`: `0.2381 -> 0.1905` (`-0.0476`)
    - `clip_017`: unchanged at `0.3133`
    - `clip_006`: `0.1667 -> 0.1905` (`+0.0238`)
    - `clip_027`: unchanged at `0.5684`
    - `clip_035`: unchanged at `0.3099`
  - Interpretation: the first learned feature set improved one previous loser but gave back the pure overclustering wins, so it was not robust enough to replace the kept subset-search rule.

- Full-train / full-manifest benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_learnedsubset.json`
  - Aggregate deltas versus the kept subset-search build (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_subsetsearch.json`):
    - Control average display cluster F1: `0.6067 -> 0.6044` (`-0.0024`)
    - Control average display note F1: `0.8574 -> 0.8559` (`-0.0015`)
    - Treatment average display cluster F1: `0.6040 -> 0.6014` (`-0.0026`)
    - Retro average display cluster F1: unchanged at `0.4883`
  - Coverage summary versus the kept subset-search build:
    - Control clips improved: `2`
    - Control clips regressed: `6`
    - Control clips unchanged: `40`
  - Biggest control improvements:
    - `clip_006`: `0.1667 -> 0.1905` (`+0.0238`)
    - `clip_022`: `0.4103 -> 0.4274` (`+0.0171`)
  - Biggest control regressions:
    - `clip_024`: `0.2381 -> 0.1905` (`-0.0476`)
    - `clip_032`: `0.7350 -> 0.7009` (`-0.0342`)
    - `clip_031`: `0.1493 -> 0.1194` (`-0.0299`)
    - `clip_038`: `0.8829 -> 0.8649` (`-0.0180`)

- Latency measurement:
  - Since the benchmark chunk timings do not include `session.get_display_state()`, the display-path cost was measured directly with a microbenchmark over representative clips.
  - Mean `_build_display_surface(...)` time on six clips:
    - kept subset-search: `0.7461 ms`
    - learned subset scorer: `1.1760 ms`
    - delta: `+0.4299 ms`
  - Interpretation: latency was not the blocking issue. The learned scorer stayed cheap enough for the live path, but its first feature set was not accurate enough.

- Decision:
  - Do not ship a default learned model JSON yet.
  - Keep the trainer/runtime hook infrastructure so future iterations can improve the feature set or target without reopening the plumbing work.

### Kept pairwise same-event merger with canonical pitch voting

- Replaced the prior direct subset scorer implementation:
  - Deleted `backend/display_chord_subset_model.py`
  - Deleted `backend/train_display_chord_subset_model.py`
  - Added `backend/display_chord_pairwise_model.py`
  - Added `backend/train_display_chord_pairwise_model.py`
  - Promoted the exported full-manifest model to `backend/display_chord_pairwise_model.json`

- Runtime behavior:
  - `backend/live_rhythm.py`
    - The learned path now applies only to overlap groups with `2..7` chord candidates.
    - It uses an exact local partition search over pairwise same-event probabilities, then canonical pitch voting inside each predicted component.
    - Singleton groups and unusually large groups still stay on the kept heuristic path.
  - `backend/main.py`
    - The previously-added `_timing_ms.display_state` timing remains in place.

- Why this replaced the direct subset scorer:
  - The direct subset scorer could only choose or drop raw candidate subsets.
  - The pairwise formulation is better aligned to the actual error mode: deciding which chord hypotheses are duplicate observations of the same latent onset event, then voting the canonical pitch set for that event.
  - This also allows learned note-versus-chord correction for merged hypotheses instead of only keeping raw chord alternatives.

- Holdout training sanity check:
  - Trained on `42` manifest clips (excluding `clip_031`, `clip_024`, `clip_017`, `clip_006`, `clip_027`, `clip_035`).
  - Exported model summary:
    - dataset groups: `791`
    - pair examples: `18`
    - pitch examples: `2249`
    - pair train AUC: `1.0000`
    - pitch train AUC: `0.6221`
    - pitch threshold: `0.200`
  - Interpretation:
    - Pairwise partitioning itself is not the bottleneck in this repo because most local overlap groups are singletons.
    - Canonical pitch voting is the harder learned problem and dominates generalization quality.

- Focused holdout benchmark:
  - Result file: `backend/_tmp_pairwise_focus_v2.json` (temporary; not kept)
  - Versus the kept subset-search build (`backend/_tmp_subsetsearch_focus.json`):
    - `clip_031`: unchanged at `0.1493`
    - `clip_024`: unchanged at `0.2381`
    - `clip_017`: unchanged cluster F1 at `0.3133`, note F1 `0.8112 -> 0.8217`
    - `clip_006`: unchanged at `0.1667`
    - `clip_027`: `0.5684 -> 0.5474` (`-0.0211`)
    - `clip_035`: unchanged at `0.3099`
  - Interpretation:
    - The pairwise model preserved the pure overclustering wins but still regressed one mixed clip on holdout (`clip_027`), so a full-manifest check was required before promoting it.

- Full-train / full-manifest benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`
  - Aggregate deltas versus the kept subset-search build (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_subsetsearch.json`):
    - Control average display cluster F1: `0.6067 -> 0.6072` (`+0.0004`)
    - Control average display note F1: `0.8574 -> 0.8575` (`+0.0001`)
    - Control average display offset F1: `0.6865 -> 0.6869` (`+0.0004`)
    - Treatment average display cluster F1: `0.6040 -> 0.6044` (`+0.0004`)
    - Retro average display cluster F1: unchanged at `0.4883`

- Coverage summary versus the kept subset-search build:
  - Control clips improved: `4`
  - Control clips regressed: `4`
  - Control clips unchanged: `40`

- Biggest control improvements:
  - `clip_006`: `0.1667 -> 0.2143` (`+0.0476`)
  - `clip_002`: `0.8269 -> 0.8462` (`+0.0192`)
  - `clip_028`: `0.5185 -> 0.5333` (`+0.0148`)
  - `clip_019`: `0.4805 -> 0.4935` (`+0.0130`)

- Biggest control regressions:
  - `clip_018`: `0.6512 -> 0.6279` (`-0.0233`)
  - `clip_027`: `0.5684 -> 0.5474` (`-0.0211`)
  - `clip_021`: `0.5000 -> 0.4833` (`-0.0167`)
  - `clip_015`: `0.5974 -> 0.5844` (`-0.0130`)

- Display-path latency measurement:
  - The benchmark chunk timings still do not include `session.get_display_state()`, so display cost was measured directly with a microbenchmark over representative clips.
  - Mean `_build_display_surface(...)` time on six clips:
    - kept subset-search: `0.8015 ms`
    - pairwise model: `0.8857 ms`
    - delta: `+0.0842 ms`
  - Interpretation:
    - The pairwise model is materially cheaper than the earlier direct subset scorer and is small enough to ship on the live display path.

- Decision:
  - Ship the pairwise same-event model as the default learned display merger via `backend/display_chord_pairwise_model.json`.
  - Keep the heuristic subset-search path as the fallback for singleton or large overlap groups.

### Rejected cluster-aligned pitch-set editor over mixed note/chord event groups

- Goal:
  - Test whether a learned pitch editor operating inside already-formed display event groups could produce a meaningful cluster-F1 gain without changing onset grouping.
  - Candidate groups were formed from `live_session.get_all_notes()` plus expanded members of `live_session.coarse_chords`.

- Temporary implementation that was tested and later deleted:
  - Added `backend/display_event_pitch_editor_model.py`
  - Added `backend/train_display_event_pitch_editor_model.py`
  - Temporarily hooked `backend/live_rhythm.py` so the editor could replace the normal display surface when `LIVE_DISPLAY_EVENT_PITCH_EDITOR_MODEL` was set.

- Training signal:
  - Holdout model, excluding `clip_006`, `clip_017`, `clip_018`, `clip_021`, `clip_027`, `clip_028`:
    - clip count: `42`
    - dataset groups: `780`
    - pitch examples: `2339`
    - GT pitch coverage inside candidate groups: `1982 / 2295` (`0.8636`)
    - pitch train AUC: `0.6885`
  - Full model:
    - clip count: `48`
    - dataset groups: `971`
    - pitch examples: `3072`
    - GT pitch coverage inside candidate groups: `2593 / 3091` (`0.8389`)
    - pitch train AUC: `0.6926`
  - Interpretation:
    - This option had much more direct supervision than the pairwise edge model.
    - The extra supervision still did not translate into better display behavior.

- Focused holdout benchmark on `clip_006`, `clip_017`, `clip_018`, `clip_021`, `clip_027`, `clip_028`:
  - Result files:
    - baseline: `backend/_tmp_pitch_editor_focus_baseline.json` (temporary; not kept)
    - candidate: `backend/_tmp_pitch_editor_focus_candidate.json` (temporary; not kept)
  - Control deltas versus current pairwise default:
    - display cluster F1: `0.4532 -> 0.4089` (`-0.0443`)
    - display note F1: `0.8669 -> 0.8297` (`-0.0371`)
    - display offset F1: `0.7982 -> 0.7624` (`-0.0358`)
  - Biggest focused regression:
    - `clip_028`: cluster F1 `-0.1211`

- Full-train / full-manifest benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pitcheditor.json`
  - Aggregate deltas versus the kept pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.6072 -> 0.5874` (`-0.0198`)
    - Control display note F1: `0.8575 -> 0.8423` (`-0.0152`)
    - Control display offset F1: `0.6869 -> 0.6761` (`-0.0109`)
    - Treatment display cluster F1: `0.6044 -> 0.5857` (`-0.0187`)
    - Retro display cluster F1: `0.4883 -> 0.4749` (`-0.0134`)

- Paired significance against the pairwise default:
  - Control display cluster F1:
    - 95% CI: `[-0.0292, -0.0111]`
    - randomization `p=0.0000`
    - improved/regressed/unchanged clips: `5 / 24 / 19`
  - Interpretation:
    - This is not variance. The pitch editor is materially and significantly worse than the current pairwise path.

- Biggest full-manifest control improvements:
  - `clip_027`: `+0.0333`
  - `clip_002`: `+0.0225`
  - `clip_024`: `+0.0133`
  - `clip_006`: `+0.0120`
  - `clip_010`: `+0.0114`

- Biggest full-manifest control regressions:
  - `clip_009`: `-0.1183`
  - `clip_028`: `-0.0764`
  - `clip_036`: `-0.0690`
  - `clip_017`: `-0.0596`
  - `clip_025`: `-0.0445`

- Decision:
  - Reject the cluster-aligned pitch-set editor path.
  - Delete the temporary runtime/trainer scaffolding and keep the live path on the current pairwise merger.
  - The failure mode is not just lack of data; direct pitch-set editing over mixed note/chord evidence suppresses too many true pitches in this pipeline.

### Rejected pairwise ambiguity-guard fallback for worst new regressions

- Goal:
  - Diagnose the largest new pairwise regressions against the kept subset-search display path and patch their shared local failure mode without widening latency.

- Diagnosis:
  - The worst regressions were concentrated in near-duplicate two-chord groups.
  - Two local failure modes showed up:
    - merged components with stable shared core pitches and ambiguous singleton fringe pitch probabilities near the pitch threshold
    - split components where slot disagreement kept both overlapping observed chords alive even though the group should visually collapse to one candidate

- Temporary implementation that was tested and later rolled back:
  - Added an observed-candidate fallback in `backend/display_chord_pairwise_model.py` for ambiguous merged two-chord components.
  - Extended the same fallback to high-overlap two-chord groups that the pairwise partitioner incorrectly left split.

- Focused validation on `clip_018`, `clip_027`, `clip_021`, `clip_015`, `clip_006`, `clip_002`, `clip_019`, `clip_028`:
  - Result file: `backend/_tmp_pairwise_ambiguity_guard_focus.json`
  - Versus the current pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.5413 -> 0.5411` (`-0.0002`)
    - Control display note F1: `0.8855 -> 0.8895` (`+0.0041`)
    - Control display offset F1: `0.8216 -> 0.8259` (`+0.0043`)
  - Versus the kept subset-search build (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_subsetsearch.json`):
    - Control display cluster F1: `0.5387 -> 0.5411` (`+0.0024`)
  - Interpretation:
    - The targeted guard cleaned up the diagnosed regressions on the focused clips, but it also gave back the large `clip_006` win that made the existing pairwise default look better on that slice.

- Full-manifest benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise_ambiguityguard.json`
  - Aggregate deltas versus the kept pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.6072 -> 0.6068` (`-0.0003`)
    - Control display note F1: `0.8575 -> 0.8578` (`+0.0002`)
    - Control display offset F1: `0.6869 -> 0.6871` (`+0.0001`)
  - Paired significance versus the kept pairwise default:
    - Control display cluster F1 95% CI: `[-0.0036, +0.0026]`
    - randomization `p=0.8477`
  - Interpretation:
    - The ambiguity guard addressed the diagnosed tail behavior, but it did not produce a meaningful manifest-scale win and was slightly negative on the primary control cluster metric.

- Decision:
  - Reject and roll back the ambiguity-guard fallback.
  - Keep the current pairwise default unchanged.
  - Preserve the diagnosis: the worst new regressions are dominated by near-duplicate two-chord groups, but patching them with a simple observed-candidate fallback is not strong enough to justify shipping.

### Structured duplicate-pair resolver for high-overlap two-chord groups

- Goal:
  - Replace the brittle fallback logic with a structured resolver for exactly-two high-overlap chord groups, so merged and split near-duplicate cases are handled by choosing among a small set of musically plausible outcomes rather than by raw pitch thresholding alone.

- Implementation:
  - Added an optional `duplicate_pair_model` path to `backend/display_chord_pairwise_model.py`.
  - Added candidate generation for `left`, `right`, `union`, and `vote` pitch sets on duplicate two-chord groups.
  - Added a low pair-probability fallback that collapses split duplicates to the stronger observed chord.
  - Added a small superset-shrink correction so a larger candidate loses to a subset when the only added pitches are below the pitch threshold and come from a not-better chord.
  - Extended `backend/train_display_chord_pairwise_model.py` to export a temporary duplicate-pair scorer trained on candidate outcomes.

- Training artifact:
  - Temporary model file: `backend/_tmp_display_chord_pairwise_model_duplicatepair.json`
  - Duplicate-pair training coverage:
    - groups: `19`
    - candidate rows: `43`
    - train AUC: `0.8202`

- Focused validation on `clip_018`, `clip_027`, `clip_021`, `clip_015`, `clip_006`, `clip_002`, `clip_019`, `clip_028`:
  - Result file: `backend/_tmp_duplicatepair_focus.json`
  - Versus the kept pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.5413 -> 0.5463` (`+0.0050`)
    - Control display note F1: `0.8855 -> 0.8875` (`+0.0020`)
    - Control display offset F1: `0.8216 -> 0.8237` (`+0.0021`)
  - Clip-level control cluster changes versus the kept pairwise default:
    - `clip_018`: `+0.0233`
    - `clip_021`: `+0.0167`
    - all other focused clips unchanged versus the kept pairwise default
  - Interpretation:
    - This is the first repair that improved the targeted regression clips without giving back the existing pairwise wins on `clip_006` and `clip_028`.

- Full-manifest benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_duplicatepair.json`
  - Aggregate deltas versus the kept pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.6072 -> 0.6074` (`+0.0002`)
    - Control display note F1: `0.8575 -> 0.8582` (`+0.0007`)
    - Control display offset F1: `0.6869 -> 0.6874` (`+0.0005`)
  - Paired significance versus the kept pairwise default:
    - Control display cluster F1 95% CI: `[-0.0026, +0.0027]`
    - randomization `p=0.9365`

- Decision:
  - Keep the structured duplicate-pair resolver code and trainer support as an opt-in experiment, but do not promote the temporary model into `backend/display_chord_pairwise_model.json` yet.
  - The local solution is real and fixes the diagnosed high-overlap two-chord failure mode, but the current manifest contains too few such groups for the improvement to clear the user's bar at repo scale.

### More-data retrain for the duplicate-pair candidate scorer

- Goal:
  - Retrain only the duplicate-pair candidate scorer on substantially more mined duplicate-pair groups while keeping runtime cost effectively unchanged.

- Trainer change:
  - Extended `backend/train_display_chord_pairwise_model.py` with:
    - `--augment-chunk-seconds`
    - `--augment-noise-profiles`
  - The trainer now aggregates replay-mined rows across multiple chunk-size and noise-profile passes in one run, while preserving the previous single-pass behavior by default.

- More-data retrain:
  - Temporary model file: `backend/_tmp_display_chord_pairwise_model_duplicatepair_moredata.json`
  - Mining sweep:
    - chunk sizes: `0.6`, `0.9`, `1.2`
    - noise profiles: `balanced`, `open`, `clean`
    - total mining passes: `9`
  - Training coverage increase:
    - duplicate-pair groups: `19 -> 282`
    - duplicate-pair candidate rows: `43 -> 663`
  - Raw retrain result:
    - Retraining pair, pitch, and duplicate-pair models together regressed the focused benchmark, so the candidate-scorer retrain had to be isolated from the base pair/pitch models.

- Isolated candidate-scorer evaluation:
  - Built `backend/_tmp_display_chord_pairwise_model_duplicatepair_moredata_isolated.json` by keeping the shipped `pair_model` and `pitch_model` from `backend/display_chord_pairwise_model.json` and swapping in only the more-data `duplicate_pair_model`.
  - Focused benchmark result file: `backend/_tmp_duplicatepair_moredata_isolated_focus.json`
  - Focused deltas versus the kept pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.5413 -> 0.5463` (`+0.0050`)
    - Control display note F1: `0.8855 -> 0.8884` (`+0.0029`)
    - Control display offset F1: `0.8216 -> 0.8247` (`+0.0031`)

- Full-manifest benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_duplicatepair_moredata_isolated.json`
  - Aggregate deltas versus the kept pairwise default (`backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_pairwise.json`):
    - Control display cluster F1: `0.6072 -> 0.6074` (`+0.0002`)
    - Control display note F1: `0.8575 -> 0.8585` (`+0.0009`)
    - Control display offset F1: `0.6869 -> 0.6877` (`+0.0007`)
  - Paired significance versus the kept pairwise default:
    - Control display cluster F1 95% CI: `[-0.0026, +0.0027]`
    - randomization `p=0.9365`

- Latency:
  - Runtime feature count and duplicate-pair code path are unchanged; only the duplicate-pair weights were retrained.
  - Full-manifest stabilization timing means stayed in the same range as the shipped pairwise baseline, so this retrain did not introduce a meaningful latency penalty.

- Decision:
  - Keep the multi-pass mining support in the trainer.
  - Do not promote the more-data duplicate-pair scorer into the shipped model.
  - Conclusion: more data improved training coverage substantially and nudged secondary metrics up, but it did not materially change manifest-scale display cluster F1.

### Manifest duplicate-pair decomposition and shared-core candidate expansion

- Goal:
  - Determine whether the next duplicate-pair improvement should target scorer choice, candidate generation, or upstream pitch recovery.

- Manifest decomposition audit:
  - Artifact: `backend/_tmp_duplicate_pair_manifest_decomposition.json`
  - Scope: actual full-manifest benchmark config (`chunk_seconds=1.2`, `noise_profile=balanced`), not the widened mining sweep.
  - Duplicate-pair groups on that config: `19`
  - Bucket counts before changing candidate generation:
    - exact GT already in candidates: `10/19`
    - candidate-space limit with GT still present in the observed union: `4/19`
    - upstream-missing GT pitches: `5/19`
    - exact-in-candidate groups still chosen incorrectly by the isolated more-data scorer: `4/10`
  - Interpretation:
    - The next bottleneck was not just upstream pitch loss; the current `left/right/union/vote` candidate set was also leaving real recoverable GT shapes off the table.

- Runtime candidate-space change:
  - Updated `backend/display_chord_pairwise_model.py` so duplicate-pair candidate generation now also proposes shared-core variants:
    - `shared`
    - `shared + one left unique`
    - `shared + one right unique`
    - `shared + strongest left unique + strongest right unique`
  - Rationale:
    - The candidate-space-limit examples were usually GT pitch sets that kept the shared core and only one fringe unique pitch, which the original `left/right/union/vote` set could not express.

- Post-change manifest audit:
  - Artifact: `backend/_tmp_duplicate_pair_manifest_decomposition_postcandidate.json`
  - Result:
    - exact GT in candidates: `10/19 -> 14/19`
    - candidate-space-limit bucket: `4/19 -> 0/19`
    - upstream-missing GT bucket unchanged: `5/19`
  - Interpretation:
    - The shared-core candidate expansion is real and fixes the representation ceiling.

- Retraining on the expanded candidate space:
  - Temporary model file: `backend/_tmp_display_chord_pairwise_model_duplicatepair_moredata_sharedcandidates.json`
  - Isolated evaluation artifact: `backend/_tmp_display_chord_pairwise_model_duplicatepair_moredata_sharedcandidates_isolated.json`
  - Training coverage:
    - duplicate-pair candidate rows: `663 -> 945`
    - duplicate-pair train ROC AUC: `0.7354 -> 0.7496`
  - Manifest selection audit with the retrained isolated scorer:
    - Artifact: `backend/_tmp_duplicate_pair_manifest_decomposition_sharedcandidates_retrained.json`
    - exact GT in candidates stayed at `14/19`
    - exact-in-candidate groups chosen correctly dropped to `4/14`
    - exact-in-candidate groups chosen incorrectly rose to `10/14`
  - Interpretation:
    - More candidate coverage alone was not enough; the current linear duplicate-pair scorer did not learn to rank the new shared-core candidates well and in fact got worse when retrained on them.

- Decision:
  - Keep the shared-core candidate expansion code as experimental groundwork; it is inert for the shipped default because the default model still has no `duplicate_pair_model` block.
  - Do not promote the retrained shared-core duplicate-pair model.
  - Updated next direction:
    - Move from broad candidate-scorer retraining toward richer/context-aware duplicate-pair features or a local pitch-repair signal, since simple pitch probabilities and broader mining still do not disambiguate the newly exposed candidate choices.

### Rolled back: grid-slot-gated chord reconciliation

- Goal:
  - Break the flat 10 ms reconciliation window with a hard gate: if two chords within the window have different known `start_grid_idx` values, treat them as distinct musical events and place them in separate reconciliation groups.

- Implementation:
  - Modified the chord grouping loop in `_build_display_surface(...)` in `backend/live_rhythm.py` to check `_display_event_slot()` on the incoming chord and the current group anchor. If both slots are non-None and differ, start a new group regardless of time proximity.

- Focused validation (`clip_017`, `clip_024`, `clip_031`):
  - Result file: `backend/_tmp_gridslot_focus.json` (temporary; not kept)
  - `clip_017`: display cluster F1 `0.3133 -> 0.3133` (unchanged)
  - `clip_024`: display cluster F1 `0.2381 -> 0.1905` (`-0.0476`, regression)
  - `clip_031`: display cluster F1 `0.1493 -> 0.1493` (unchanged)

- Root cause of regression:
  - Coarse chord `start_grid_idx` values are assigned during live quantization when the tempo tracker is still refining. Two chord emissions that represent the same physical onset event — emitted from adjacent overlapping chunks — can receive different `start_grid_idx` ticks if the BPM estimate shifted slightly between chunks. The slot gate treats these as distinct events and prevents the pairwise merger from collapsing them, which caused the clip_024 overclustering to worsen.
  - Unlike `_group_display_note_events(...)` (which groups refined notes over a 30 ms window after tempo has largely stabilized), the 10 ms chord reconciliation window operates on raw coarse chord emissions where slot assignments are not yet stable enough to be load-bearing.

- Decision:
  - Rolled back immediately. The current pairwise merger is restored as the default.
  - Next approach should focus on the upstream assignment instability rather than adding slot-based gates at the display surface level.

### Investigated and rolled back: sustained-note cross-chunk deduplication

- Goal:
  - Apply the physical constraint that the same note cannot be re-articulated faster than ~80ms. If a chord from chunk N is still sounding (offset extends past the next chord's onset) and the two chords share >= 70% Jaccard pitch similarity, suppress the later chord as a re-emission.

- Implementation (then reverted):
  - Added `_drop_sustained_chord_reemissions()` pre-filter to `backend/live_rhythm.py`.
  - Called it in `_build_display_surface(...)` on sanitized chords before the 10ms reconcile grouping.
  - Fixed a bug: `offset_seconds` on most coarse chords is a chunk-relative value, not an absolute timestamp. The correct absolute offset is `time_seconds + duration_seconds`.
  - Added `gap_after_left_offset_ms` and `left_has_duration` to `PAIR_FEATURE_NAMES` and `extract_pair_features` in `backend/display_chord_pairwise_model.py` for future retraining.

- Diagnosis of why the pre-filter fires rarely:
  - For clips 031 and 006 (worst overclustering): 0 suppressions. These clips have chord inter-onset intervals > 100ms on average; no high-Jaccard pairs exist within the 150ms dedup window.
  - For clips 024, 017, 027: 1–2 suppressions each.
  - Total across 6 focused clips: 4 suppressions.

- Root cause confirmed:
  - Overclustering in clips 031, 024, 006 comes from INDIVIDUAL singleton chord emissions (not from merging multiple hypotheses). The neural model emits a single chord with more pitches than the GT, and the pitch vote model cannot filter within a singleton: all pitches in a singleton component have identical support features (`pitch_support_fraction = 1.0`, `pitch_confidence_fraction = 1.0`). The pitch vote is a no-op for singletons.
  - The physical-constraint pre-filter addresses a real correctness concern but is not the bottleneck for the current worst clips.

- GPU non-determinism note:
  - Individual clip cluster F1 varies by ±0.05 between runs due to floating-point differences in GPU inference. Focused tests on 1–6 clips are unreliable. Only full-manifest (48-clip) averages are stable.

- Full-manifest result for the pre-filter (correct offset implementation):
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster_sustaindedup.json`
  - Control average display cluster F1: `0.6073` (`+0.0001` vs pairwise default of `0.6072`)
  - Not meaningful.

- Decision:
  - Reverted the pre-filter (removed `_drop_sustained_chord_reemissions` from `live_rhythm.py`).
  - Kept the two new pair model features (`gap_after_left_offset_ms`, `left_has_duration`) in `display_chord_pairwise_model.py` for use in future retrains.

### Pair model retrain with gap features (no change to manifest performance)

- Retrained `display_chord_pairwise_model.json` with the two new pair features:
  - `gap_after_left_offset_ms`: right chord onset minus left chord offset, in ms (negative = left note still sounding; positive = silence gap between them).
  - `left_has_duration`: indicator for whether `duration_seconds` was available on the left chord.
- Training outcomes unchanged from the previous model:
  - Pair model: AUC `1.0000`, accuracy `1.0000` (31 pair examples, overfitting expected).
  - Pitch model: AUC `0.6252`, threshold `0.200` (identical to prior retrain).
- Conclusion: the pair model's training set (31 examples) is too small for new features to be meaningful. The pitch model's discriminative weakness is intrinsic to the current feature set, not feature count.

- Next direction identified:
  - The pitch vote model is fundamentally limited for singleton chord components: all pitches in a singleton get identical vote features, so the model cannot choose between them. The only ways to fix overclustering in singleton-dominated clips (031, 024, 006) are:
    1. Per-note probability from the neural model (expose individual piano-roll probabilities from `detect_note.py` as a pitch vote feature).
    2. Wider reconcile window so more chords land in the same group, giving the pitch vote model cross-chord evidence.
  - Option 2 is the lower-risk immediate experiment.

### Tested and rolled back: wider chord reconcile window (25 ms)

- Goal:
  - With a wider reconcile window, more chord pairs land in the same group, giving the pitch vote model cross-chord evidence instead of running on singletons. This might allow it to filter spurious pitches even for events that come from different overlapping chunks but are > 10ms apart.

- Change:
  - `DISPLAY_CHORD_RECONCILE_TOLERANCE_SEC`: `0.01 -> 0.025` in `backend/live_rhythm.py`.

- Focused validation (`clip_031`, `clip_024`, `clip_017`, `clip_006`):
  - `clip_017`: cluster F1 `0.3133 -> 0.2892` (`-0.0241`)
  - `clip_024`: unchanged at `0.2381`
  - `clip_031`: unchanged at `0.1493`
  - `clip_006`: `0.2143 -> 0.2169` (`+0.0026`)

- Root cause of the regression:
  - At 25ms, genuinely adjacent events in fast melodic passages (clip_017 especially) land in the same group. The pairwise model classifies them as different events (correct), creating two components — but the pitch vote then runs separately on each smaller component with reduced support evidence for each pitch. The result is more missed pitches (underclustered matches increased from 13 to 15 for clip_017).

- Decision:
  - Reverted. `DISPLAY_CHORD_RECONCILE_TOLERANCE_SEC` stays at `0.01`.

### Display surface improvement ceiling identified

- After trying 5 approaches (grid-slot gate, physical pre-filter, gap-feature retrain, wider reconcile window, attempted pair-feature augmentation), the conclusion is:
  - **The display surface alone cannot fix the dominant remaining error mode.**
  - Clips 031, 024, 006 overcluster because individual singleton chord emissions from the neural model contain more pitches than the GT. The pitch vote model cannot discriminate between pitches within a singleton (all pitches have `pitch_support_fraction = 1.0` for singletons, giving no differential signal).
  - The current pairwise merger+pitch-vote combination is the best achievable at this layer without per-note probability signals from the neural model.

- Current baseline (kept, unchanged):
  - `backend/display_chord_pairwise_model.json`: retrained with two new features (`gap_after_left_offset_ms`, `left_has_duration`) but producing the same manifest performance as before.
  - Control average display cluster F1: `0.6072` (may read as `0.6073` due to non-determinism in GPU inference).

- Next highest-impact direction:
  - Expose per-note piano roll probabilities from the mel_baseline model inside chord events (in `backend/detect_note.py`). These raw probabilities would give the pitch vote model a strong new feature (`pitch_neural_probability`) that directly discriminates real vs. spurious pitches even in singleton components.

## 2026-05-30

### Implemented and reverted: `pitch_neural_probability` for pitch vote model

- Goal:
  - Add per-note neural onset probability as a discriminative feature in the pitch vote model. For singleton chord components (where all existing features are equal), `onset_prob` breaks the symmetry and enables pitch-level filtering.

- Implementation details:
  - `backend/rhythm_training/train_ensemble.py`:
    - Removed the `event.pop('onset_prob', None)` at the end of `decode_note_events()` so each note event carries its raw onset probability forward. This field was previously stripped before return; it is now preserved as useful metadata.
  - `backend/detect_note.py` (`_convert_neural_note_events_to_results()`):
    - Added `chord_dict['note_probabilities'] = [float(event.get('onset_prob', 0.5)) for event in sorted_group]` — a list of neural onset probabilities in the same sorted order as `midi_notes`. Kept on chord dicts going forward (not reverted).
  - `backend/display_chord_pairwise_model.py`:
    - Added `pitch_neural_probability` to `PITCH_FEATURE_NAMES`.
    - Added `_chord_note_probability()` helper to look up a pitch's onset_prob from `note_probabilities`.
    - Added `pitch_neural_probability` = mean onset_prob across supporters in `extract_pitch_vote_features()`.

- Training results (chunk_seconds=0.6 to match benchmark):
  - Pitch AUC: `0.6252 -> 0.9173` — a large genuine jump.
  - Pitch threshold (auto-selected): `0.200 -> 0.350`.
  - Root: the `onset_prob` values are strongly discriminative on training data. The feature coefficient was `+1.645` (largest in the model).

- Full-manifest benchmark results (48 clips):
  - `display_cluster_f1`: ref=`0.6115`, cand=`0.6088`, diff=`-0.0028` (p=0.18, not significant).
  - `display_note_f1`: ref=`0.8601`, cand=`0.8564`, diff=`-0.0037` (p=0.0081, **statistically significant regression**).

- Root cause of regression:
  - The pitch model at threshold=0.35 removes notes with lower `onset_prob`. However, false-positive pitches detected by the neural model have `onset_prob > 0.38` (they cleared the neural detection floor). This overlaps with some GT pitches that are played softly or are in chord voices with lower detection confidence. The model cannot cleanly separate them in the 0.38–0.60 `onset_prob` range, causing false negatives.
  - Additionally, focused inspection of the worst clips (031, 006, 024) showed that the treatment delta was exactly `+0.000` for cluster F1 on every one of them. The dominant error mode for those clips is **missed detections** (`missed_gt` = 12–18 GT clusters per clip), not pitch conflicts. No display-surface model can create detections the neural model didn't emit.

- Decision:
  - Reverted `pitch_neural_probability` from `PITCH_FEATURE_NAMES` and `extract_pitch_vote_features()`.
  - **Kept** `onset_prob` preservation in `train_ensemble.py` (non-destructive metadata, useful for future use).
  - **Kept** `note_probabilities` on chord dicts in `detect_note.py` (low-cost metadata, available to future models or heuristics).
  - Retrained `display_chord_pairwise_model.json` back to baseline (chunk_seconds=1.2): pitch AUC=0.6252, threshold=0.200, same as before.

### Root cause of cluster F1 ceiling identified: neural detection recall

- After 6 total display-surface experiments (grid-slot gate, sustained-note pre-filter, gap features, wider reconcile window, pitch_neural_probability), the conclusion is now definitive:
  - The cluster F1 bottleneck for clips 031, 006, 024 is not display-surface logic — it is **neural model recall**. The mel_baseline transcriber simply does not fire for 12–18 of the 20–44 GT note clusters in those clips. No pairwise merging, pitch filtering, or reconciliation strategy can recover notes the model never emitted.

- Next direction:
  - Investigate neural detection recall. Options:
    1. Lower the live onset threshold (currently `0.38`) to detect softer notes.
    2. Investigate whether the live noise gate (`chord_min_confidence: 0.50`, `balanced` profile) is filtering valid low-velocity chords.
    3. Examine whether the mel_baseline model needs retraining with harder recall-focused objectives for the specific failure clips.

## 2026-05-31

### Recall diagnosis corrected, and frame rising-edge onset recovery (rejected)

- Goal:
  - Find a new method that drastically improves model recall, test it on the full manifest with `test_experiment.py`, and record the outcome.

- Recall diagnosis re-measured (corrects the prior change-log narrative):
  - The earlier claim that "the mel_baseline does not fire for 12–18 of 20–44 GT clusters" did not match a fresh measurement. On the full manifest (control arm, 48 clips):
    - mean note recall `0.9237`
    - mean note precision `0.7656`
  - Recall is already high; **precision (over-prediction) is the weaker axis**, not recall.
  - Decoded the raw onset/frame/velocity probability maps directly for the worst-recall clips (`clip_026`, `clip_035`, `clip_017`, `clip_024`, `clip_006`, `clip_031`) and categorized every missed GT note (50 ms / same pitch):
    - `~70%` of misses are **never emitted by the onset head** — the onset probability peak within ±3 frames of the GT onset never reaches `0.30`. Example bucket counts: `clip_026` 22/23, `clip_035` 14/14, `clip_017` 29/31, `clip_024` 11/11, `clip_006` 24/24, `clip_031` 6/7.
    - Of the missed notes, only `~30%` have **frame-head evidence** (frame prob `>= 0.5`) near the GT onset (e.g. `clip_026` 8/23, `clip_017` 7/31, `clip_006` 5/24).
  - Decode-side ablations on the raw maps confirmed the ceiling:
    - Disabling `filter_harmonics` recovered `~0–1` notes (the octave/harmonic filter barely fires on these clips; the `project_runs_include_octaves` concern is real but not the recall bottleneck here).
    - Disabling peak-picking recovered `~0–1` notes while inflating predictions by `+30–40` (catastrophic precision cost for no recall gain).
    - Lowering the onset threshold to `0.30` recovered `~0–2` notes.
  - Conclusion: no decode-side trick on the onset head can recover notes the onset head does not represent. Decode-side recall is capped at roughly `+2–3%`.

- New method tested: **frame rising-edge onset recovery** (standard Onsets-and-Frames idea, never used in this repo).
  - Implemented (then reverted) an optional `recover_frame_onsets` post-pass in `decode_note_events()` (`backend/rhythm_training/train_ensemble.py`): after onset-peak decoding and harmonic filtering, scan each pitch's frame probability for a sharp rising edge across `frame_threshold` that is not already covered by an onset event, gated by a minimum onset-prob hint and a minimum frame-rise so it only fires where frame evidence is strong (avoiding pedal/sustain smear).
  - Wired (then reverted) an env-gated hook in both `MelBaselineTranscriber.transcribe(...)` paths in `backend/gpu_ops.py` (`LIVE_FRAME_ONSET_RECOVERY`, with `LIVE_FRAME_ONSET_FLOOR` / `LIVE_FRAME_ONSET_FRAME_THR` / `LIVE_FRAME_ONSET_RISE` overrides).

- Behavior found:
  - Offline (full-audio) the gate recovers real notes at a cost of `~1.5–3` false positives per true note.
  - In the **live chunked path** the precision cost is worse: chunk boundaries and dense polyphony re-trigger frame edges, so dense clips (`clip_026`, `clip_031`) gained `+9–14` predictions with `0` true recoveries, while melodic low-recall clips (`clip_017`, `clip_035`) gained real notes.
  - Tightening the gate (onset floor `0.25 -> 0.28`, frame thr `0.60 -> 0.62`, rise `0.25 -> 0.28`) reduced the dense-clip false positives without losing the melodic-clip recoveries; this is the variant taken to full manifest.

- Full-manifest result (control arm, 48 clips; `chunk_seconds=0.6`, `noise_profile=balanced`, retro off):
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260531_frameonset_recovery.json`
  - Aggregate deltas versus a matched fresh baseline (recovery off; control display cluster F1 reproduced the shipped `0.6072`):
    - note recall: `0.9237 -> 0.9268` (`+0.0031`), improved on `13` clips, regressed on `0`
    - note precision: `0.7656 -> 0.7409` (`-0.0247`)
    - note F1: `0.8318 -> 0.8171` (`-0.0148`)
    - display note recall: `0.9162 -> 0.9188` (`+0.0026`)
    - display note F1: `0.8575 -> 0.8460` (`-0.0116`)
    - display cluster F1: `0.6072 -> 0.5967` (`-0.0104`)
  - Paired significance versus the matched baseline:
    - display cluster F1: 95% CI `[-0.0156, -0.0056]`, randomization `p=0.0003`, signs `4/24/20`, best `clip_035 +0.0282`, worst `clip_007 -0.0714`
    - display note F1: 95% CI `[-0.0160, -0.0074]`, `p=0.0000`
    - display offset F1: 95% CI `[-0.0100, -0.0046]`, `p=0.0000`

- Decision:
  - Reject frame rising-edge onset recovery and revert both code edits (`decode_note_events` post-pass and the `gpu_ops` env hook). The live path is unchanged.
  - The method is a _clean but tiny_ recall win (`+0.31%` note recall, never regresses recall on any clip), but it costs `~3x` more precision, significantly regressing every display metric including the primary `display_cluster_f1`. This is the same structure-limited pattern seen across the prior display-surface experiments.

- Corrected next direction:
  - Decode-side recall is exhausted. The only lever that can drastically raise recall is the onset head itself: `~70%` of misses have neither onset nor frame evidence and are invisible to the current model. This requires **retraining the mel_baseline onset head** (recall-focused loss / onset-positive weighting / harder negative mining), not further live-path or decode heuristics.
  - Separately, the fresh measurement shows the manifest is precision-limited (mean note precision `0.77` vs recall `0.92`), so precision-side work (reducing over-prediction) is the higher-leverage target for `display_cluster_f1` than recall.

### KEPT: longer neural inference context per live chunk (largest cluster-F1 win to date)

- Goal:
  - Improve precision substantially. The fresh measurement showed the manifest is precision-limited (mean note precision `0.77` vs recall `0.92`).

- False-positive diagnosis:
  - On the worst-precision clips, categorized every false positive that survives into the display surface. The dominant category is **phantom** (`~55%`): predicted notes with no GT note nearby at any pitch. Note confidence barely separates matched notes from phantoms on most clips (e.g. `clip_009` matched `0.412` vs FP `0.395`), so confidence thresholding cannot remove them cleanly. Harmonic (`+12/+19/+24`) and octave-down (`-12`) satellites are the next largest groups; octave-down is risky to filter because it overlaps genuine bass octave doublings.
  - Root cause found: **the neural model is far more precise with more audio context.** The live path feeds the transcriber only the `0.6s` chunk plus `OVERLAP_SAMPLES = 4096` (`~93ms`) of history. Running the same `mel_baseline` model on longer windows (offline, 7 worst-precision clips, onset `0.38`):
    - window `0.6s`: recall `0.872`, precision `0.683`, FP `164`
    - window `1.2s`: recall `0.894`, precision `0.748`, FP `122`
    - window `2.4s`: recall `0.899`, precision `0.790`, FP `97`
    - full clip: recall `0.906`, precision `0.848`, FP `66`
  - Both precision and recall improve monotonically with context; there is no tradeoff. Example: `clip_009` is perfect on full audio (`10/10`, `0` FP) but emits `8` phantom FPs on `0.6s` chunks. This is the same short-chunk degradation that caps recall, viewed from the precision side.
  - A real-time rolling simulation (run on the last `2.4s` but commit only the newest `0.6s` region) reproduced most of the gain: precision `0.683 -> 0.767`, recall slightly up.

- Implementation (kept, shipped default):
  - `backend/main.py`
    - Decoupled **inference context** from the **emission/overlap region**. Added `LIVE_CONTEXT_SEC` (default `2.4`, env-tunable; `0` restores the legacy `93ms` behavior) and `CONTEXT_SAMPLES`.
    - The per-session history tail kept for the neural pass is now up to `CONTEXT_SAMPLES` instead of `OVERLAP_SAMPLES`, so the transcriber sees `~2.4s` of left-context for calibration.
    - `OVERLAP_SAMPLES` (`~93ms`) is now used only as the boundary **recovery band**: `_shift_overlap_events_with_recent_dedupe(...)` takes a `recovery_band_sec` and drops history notes older than `overlap_sec - recovery_band_sec` (those were already emitted by earlier chunks; the audio is present only as context). This preserves the tuned 2026-05-29 boundary-recovery behavior and keeps the `RECENT_EVENT_RETENTION_SEC = 0.25` dedup window valid.

- Full-manifest result (control arm, 48 clips; `chunk_seconds=0.6`, `noise_profile=balanced`, retro off):
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260531_longcontext.json`
  - Aggregate deltas, legacy `93ms` (`LIVE_CONTEXT_SEC=0`) -> `2.4s` context:
    - note precision: `0.7656 -> 0.8182` (`+0.0527`)
    - note recall: `0.9237 -> 0.9280` (`+0.0043`)
    - note F1: `0.8318 -> 0.8663` (`+0.0345`)
    - display note precision: `0.8180 -> 0.8643` (`+0.0463`)
    - display note F1: `0.8575 -> 0.8878` (`+0.0303`)
    - display cluster F1: `0.6069 -> 0.6561` (`+0.0492`)
    - display offset F1: `0.6869 -> 0.6984` (`+0.0115`)
    - duplicates per 100 notes: `6.0252 -> 4.7303` (`-1.2949`)
    - latency: avg chunk `16.2 -> 22.2 ms` (`+6.0`), p95 chunk `33.6 -> 35.8 ms` (`+2.2`) — still `~16x` faster than real time.
  - Paired significance versus the legacy baseline:
    - display cluster F1: diff `+0.0492`, 95% CI `[+0.0284, +0.0698]`, randomization `p=0.0000`, signs `36/9/3`, best `clip_027 +0.2186`, worst `clip_010 -0.1751`
    - display note F1: diff `+0.0303`, 95% CI `[+0.0195, +0.0417]`, `p=0.0000`, signs `41/6/1`
    - display offset F1: diff `+0.0115`, 95% CI `[-0.0086, +0.0283]`, `p=0.2539`

- Interpretation:
  - This is the largest `display_cluster_f1` improvement recorded in this log by an order of magnitude. The entire prior display-surface program moved cluster F1 from `0.5967` to `0.6072` across many experiments; this single pipeline change takes it to `0.6561`, while also raising precision (the requested goal), recall, and reducing duplicates, at a negligible latency cost.
  - It also reframes both ceilings: short-chunk inference was the shared root cause of the over-prediction (precision) problem and a meaningful slice of the recall problem. Feeding the model more left-context is a pipeline fix that does not require retraining.

- Decision:
  - Keep `LIVE_CONTEXT_SEC = 2.4` as the shipped default in the live path.

- Remaining follow-ups:
  - The biggest regression is `clip_010` (`-0.1751`); its note precision/recall are essentially unchanged (`prec 0.857 -> 0.836`, recall flat), so the dip is a display **clustering/structure** interaction (under-clustered matches `4 -> 6`), not a detection regression. The other regressors (`clip_019`, `clip_016`, `clip_003`) are similar small structure shifts. Investigate the display grouping interaction on these before tuning context length further.
  - Context length was set to `2.4s` as a latency/accuracy balance; the offline sweep suggests full-clip context is marginally better still. A context-length sweep on the full manifest (e.g. `1.8 / 2.4 / 3.6s`) is the obvious next tuning step.
  - Retraining the onset head remains the only lever for the `~70%` model-invisible missed notes (recall ceiling).

### KEPT: raised base onset threshold 0.38 -> 0.46 (stacks on long context)

- Rationale:
  - With the longer `LIVE_CONTEXT_SEC` window the `mel_baseline` onset head is well-calibrated, so the old short-chunk base threshold of `0.38` was leaving easily-prunable false positives. Re-swept the base threshold at `2.4s` context.
- `backend/detect_note.py`
  - Base onset threshold for the mel path is now `0.46` (env-tunable via `LIVE_ONSET_BASE`), fed into `_select_live_neural_onset_threshold(...)` which still applies its +/-0.04 loudness adjustment.
- Full-manifest sweep (control arm, 48 clips, `2.4s` context, retro off), `display_cluster_f1` / note precision / note recall:
  - `0.38`: `0.6561` / `0.8182` / `0.9280`
  - `0.42`: `0.6652` / `0.8341` / `0.9247`
  - `0.46`: `0.6775` / `0.8516` / `0.9217` <- shipped
  - `0.50`: `0.6858` / `0.8670` / `0.9189`
  - `0.54`: `0.6912` / `0.8816` / `0.9151`
  - `0.58`: `0.6966` / `0.8897` / `0.9112`
  - `0.62`: `0.7019` / `0.9041` / `0.9068`
- Decision:
  - `display_cluster_f1` keeps rising past `0.46` only because the metric is precision-biased; recall erodes monotonically. Shipped `0.46` as the balanced point: `+0.0214` cluster F1 and `+0.0334` note precision for only `-0.0063` note recall. Did not chase the cluster-F1 max (`0.62+`) because it sacrifices real-note recall the user cares about.
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260531_onsetbase046.json`
- Cumulative this session (legacy `93ms`/`0.38` -> `2.4s`/`0.46`): display cluster F1 `0.6069 -> 0.6775` (`+0.0706`), note precision `0.766 -> 0.852` (`+0.086`), note recall `0.924 -> 0.922`.

### REJECTED: display-surface upper-partial suppressor

- Idea: the remaining FPs at `2.4s`/`0.46` are `~45%` phantom (no GT nearby; spurious, unfixable) and `~46%` harmonic (`harm_up +12/+19/+24` and `oct_down -12`). Tried suppressing `+12/+19/+24` notes that sit on top of a stronger concurrent note in the display `note_events` (octave-down deliberately protected).
- Result (full manifest vs `0.46` baseline): display cluster F1 `0.6775 -> 0.5701` (`-0.1074`), display note recall `0.9142 -> 0.8075` (`-0.1067`), display note precision only `+0.0063`. Regressed `29/48` clips.
- Why: in real polyphonic piano, notes a 12th/octave+third above another are overwhelmingly **genuine chord voicings / melody**, not spectral partials, and confidence does not separate the two. Same lesson as `project_runs_include_octaves`: do not filter by harmonic interval.
- Decision: reverted entirely (`live_rhythm.py` restored).
- Precision ceiling note: cheap pipeline precision levers are now exhausted — the threshold raise banked the safe gain; the remaining FPs are either phantom (spurious, no signal) or harmonic-interval notes that cannot be removed without destroying real notes. Further precision needs the model (better onset calibration), not post-processing.

## 2026-06-02

### FIXED: live beat-grid phase bug behind compressed score durations

- Symptom: a live recording around 9 seconds long could generate a score around 4 seconds long, even when recording/playback started together.
- Root cause: live note annotation was using local chunk/phase beat indices instead of absolute beat positions on the session beat grid.
- Implementation:
  - `backend/live_rhythm.py`: added `BeatGrid.absolute_beat_at_idx(...)`.
  - Switched coarse and refined live annotation to use absolute beat positions.
  - Added `backend/test_live_rhythm_grid.py` to guard the absolute-grid behavior.
- Decision: keep. This fixes duration/grid consistency separately from neural transcription quality.

### ADDED: enhanced mel transcriber model and inference path

- New training/inference assets:
  - `backend/rhythm_training/train_enhanced_mel_transcriber.py`
  - `backend/rhythm_training/enhanced_mel_transcription.pt`
- Architecture highlights:
  - `EnhancedMelTranscriber` with larger mel-only Conv/Conformer stack.
  - Pitch-local frequency readout.
  - Explicit `offset_logits` head.
  - `decode_enhanced_note_events(...)`.
  - `_build_model_from_config(...)` for checkpoint-compatible inference loading.
- Trained checkpoint reported by user:
  - `event_f1=0.9414496391590838`
  - `event_precision=0.9569432927218218`
  - `event_recall=0.9264497004878651`
  - `onset_f1=0.8728227183542482`
  - `offset_f1=0.5291791264414719`
  - Config includes `conv_channels=192`, `d_model=384`, `n_layers=10`, `n_heads=8`, `event_hidden=192`, `n_note_value_classes=12`, `sample_rate=16000`, `hop_length=256`, `n_mels=229`.
- `backend/gpu_ops.py`:
  - Added `GpuEnhancedMelTranscriber`.
  - Added `get_gpu_enhanced_mel_transcriber()` and `get_gpu_enhanced_mel_transcriber_status()`.
  - Search paths include env overrides `LIVE_ENHANCED_MEL_MODEL_PATH` / `ENHANCED_MEL_MODEL_PATH`, local backend paths, and `/root/rhythm_training/enhanced_mel_transcription.pt`.
  - Returns the same public contract as the old mel baseline path: `est_note_events` plus `_inference_timing_ms`.
- `backend/detect_note.py`:
  - Live neural priority is now `enhanced_mel -> mel_baseline -> custom_velocity_weighted`.
  - Full neural file-path inference also tries enhanced mel before mel baseline.
  - Loader-status errors now include `enhanced_mel`.
  - Added enhanced env knobs: `LIVE_ENHANCED_ONSET_BASE`, `LIVE_ENHANCED_OFFSET_BASE`, `LIVE_ENHANCED_MIN_VELOCITY`, `LIVE_ENHANCED_FILTER_HARMONICS`, plus full-inference equivalents.
- `backend/main.py`: `/warmup` preloads enhanced mel and reports `enhanced_mel_model`.
- `backend/modal_deploy.py`: Modal image now includes the enhanced training file/checkpoint and warms enhanced mel on startup.
- Smoke validation:
  - Checkpoint loaded locally on CUDA with `43,225,009` params.
  - One-second silence returned zero events and timing metadata.
  - `analyze_audio_live_neural(...)` smoke reported `neural_model=enhanced_mel`.

### REJECTED: uncalibrated enhanced mel default at onset 0.50

- First frozen-manifest run:
  - Command: `backend\env\Scripts\python.exe backend\test_experiment.py --benchmark-manifest backend\live_benchmark_replay_auto_v2.json --no-run-retro-correction --output-json backend\live_benchmark_replay_auto_v2_results_20260602_enhancedmel.json`
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedmel.json`
  - Harness note: `test_experiment.py` runs the local app analyzer, not a remote Modal endpoint. Since the local analyzer now prioritizes `enhanced_mel`, it still validates the inference path Modal will use after deploy.
- Against `backend/live_benchmark_replay_auto_v2_results_20260531_onsetbase046.json`:
  - control note F1: `0.8823 -> 0.8737` (`-0.0086`)
  - control display note F1: `0.8985 -> 0.8880` (`-0.0106`)
  - control display cluster F1: `0.6775 -> 0.6570` (`-0.0205`)
  - control offset F1: `0.6970 -> 0.7153` (`+0.0183`)
- Interpretation: the enhanced model was not simply worse; the old live threshold heuristics were miscalibrated for the new model. The explicit offset head helped immediately, while note/control/cluster F1 needed a much higher onset threshold.

### KEPT: enhanced mel onset calibration 0.50 -> 0.75

- Sweep:
  - Temporary runner: `backend/_tmp_enhanced_sweep.py`.
  - Frozen 48-clip manifest, retro disabled, `LIVE_ENHANCED_OFFSET_BASE=0.35`.
  - Swept `LIVE_ENHANCED_ONSET_BASE` and spot-checked `LIVE_ENHANCED_MIN_VELOCITY` / `LIVE_ENHANCED_FILTER_HARMONICS`.
  - `min_velocity` and `filter_harmonics` were neutral. `filter_harmonics` currently has no effect because `decode_enhanced_note_events(...)` does not implement the old mel-baseline harmonic-filter branch.
- Control-arm sweep results:
  - `0.50`: note F1 `0.8737`, display note F1 `0.8880`, cluster F1 `0.6570`, offset F1 `0.7153`
  - `0.55`: note F1 `0.8831`, display note F1 `0.8970`, cluster F1 `0.6629`, offset F1 `0.7219`
  - `0.60`: note F1 `0.8933`, display note F1 `0.9064`, cluster F1 `0.6759`, offset F1 `0.7277`
  - `0.65`: note F1 `0.8975`, display note F1 `0.9091`, cluster F1 `0.6831`, offset F1 `0.7311`
  - `0.70`: note F1 `0.9054`, display note F1 `0.9168`, cluster F1 `0.6989`, offset F1 `0.7346`
  - `0.75`: note F1 `0.9134`, display note F1 `0.9239`, cluster F1 `0.7038`, offset F1 `0.7395`
  - `0.80`: note F1 `0.9134`, display note F1 `0.9252`, cluster F1 `0.7029`, offset F1 `0.7353`
  - `0.85`: note F1 `0.8734`, display note F1 `0.8795`, cluster F1 `0.6290`, offset F1 `0.7026`
  - `0.90`: note F1 `0.5372`, display note F1 `0.5368`, cluster F1 `0.3285`, offset F1 `0.4237`
- Best balanced threshold:
  - `LIVE_ENHANCED_ONSET_BASE=0.75`.
  - `0.80` slightly improves display note F1 but slightly lowers cluster F1 and offset F1, so `0.75` is the better display-structure default.
- Best result file:
  - `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedmel_on075_v08_fh0.json`
- Paired comparison versus `backend/live_benchmark_replay_auto_v2_results_20260531_onsetbase046.json`:
  - control display cluster F1: `0.6775 -> 0.7038` (`+0.0262`), 95% CI `[-0.0067, +0.0604]`, `p=0.1301`
  - control display note F1: `0.8985 -> 0.9239` (`+0.0253`), 95% CI `[+0.0154, +0.0364]`, `p=0.0000`
  - control display offset F1: `0.7048 -> 0.7457` (`+0.0409`), 95% CI `[+0.0249, +0.0581]`, `p=0.0000`
  - treatment display cluster F1: `0.6729 -> 0.6980` (`+0.0251`), 95% CI `[-0.0051, +0.0590]`, `p=0.1313`
  - treatment display note F1: `0.8956 -> 0.9177` (`+0.0221`), 95% CI `[+0.0109, +0.0342]`, `p=0.0006`
  - treatment display offset F1: `0.7033 -> 0.7414` (`+0.0382`), 95% CI `[+0.0222, +0.0551]`, `p=0.0001`
- Paired comparison versus `backend/live_benchmark_replay_auto_v2_results_20260531_distillctx_onset050.json`:
  - control display cluster F1: `0.6628 -> 0.7038` (`+0.0410`), 95% CI `[+0.0073, +0.0752]`, `p=0.0208`
  - control display note F1: `0.8901 -> 0.9239` (`+0.0337`), 95% CI `[+0.0219, +0.0483]`, `p=0.0000`
  - control display offset F1: `0.7169 -> 0.7457` (`+0.0288`), 95% CI `[+0.0156, +0.0429]`, `p=0.0001`
  - treatment display cluster F1: `0.6571 -> 0.6980` (`+0.0409`), 95% CI `[+0.0092, +0.0748]`, `p=0.0163`
  - treatment display note F1: `0.8858 -> 0.9177` (`+0.0319`), 95% CI `[+0.0186, +0.0476]`, `p=0.0000`
  - treatment display offset F1: `0.7145 -> 0.7414` (`+0.0269`), 95% CI `[+0.0130, +0.0414]`, `p=0.0003`
- Implementation decision:
  - `backend/detect_note.py`: changed enhanced live default from `LIVE_ENHANCED_ONSET_BASE=0.50` to `0.75`.
  - `backend/detect_note.py`: changed enhanced full-inference default from `ENHANCED_MEL_ONSET_THRESHOLD=0.50` to `0.75`.
  - Keep `LIVE_ENHANCED_OFFSET_BASE=0.35`, `LIVE_ENHANCED_MIN_VELOCITY=8`, and `LIVE_ENHANCED_FILTER_HARMONICS=0`.
- Conclusion:
  - The enhanced model is a net win after calibration: better note F1, better display cluster F1, and clearly better offset/display-offset F1.
  - The initially worse result was a threshold-calibration mismatch, not evidence that the new neural model was inferior.

### KEPT: tighter enhanced onset grouping step ratio 0.65 -> 0.50

- Goal:
  - Raise `display_cluster_f1` after the enhanced model/onset calibration work.
  - The neural event F1 was already high; the remaining target was display-structure grouping: over/under-clustered note groups rather than raw note detection.
- Implementation:
  - `backend/detect_note.py`
    - Exposed onset-grouping parameters as env knobs:
      - `LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC`
      - `LIVE_NEURAL_GROUP_MIN_TOLERANCE_SEC`
      - `LIVE_NEURAL_GROUP_SHRINK_SEC`
      - `LIVE_NEURAL_GROUP_STEP_RATIO`
    - Changed the default step ratio from `0.65` to `0.50`.
  - Rationale:
    - The base span tolerance still allows simultaneous notes to group.
    - The tighter step ratio makes it harder for a chain of nearby-but-not-truly-simultaneous attacks to merge into one displayed cluster.
- Sweep:
  - Temporary runner: `backend/_tmp_enhanced_cluster_sweep.py`
  - Logs: `backend/benchmark_artifacts/enhanced_cluster_20260602/`
  - Summary: `backend/enhanced_mel_cluster_sweep_20260602_summary.json`
  - Baseline: enhanced default with adaptive-cap fix (`LIVE_ENHANCED_ONSET_BASE=0.75`, `LIVE_CONTEXT_SEC=2.4`, `LIVE_NEURAL_GROUP_STEP_RATIO=0.65`).
- Tested grouping variants:
  - base tolerance `0.020`: control cluster `0.7097`, treatment cluster `0.7061`
  - base tolerance `0.025`: control cluster `0.7015`, treatment cluster `0.7032`
  - base tolerance `0.035`: control cluster `0.7019`, treatment cluster `0.7050`
  - base tolerance `0.040`: control cluster `0.7035`, treatment cluster `0.7048`
  - shrink `0.002`: control cluster `0.7040`, treatment cluster `0.7068`
  - shrink `0.006`: control cluster `0.7015`, treatment cluster `0.7035`
  - step ratio `0.50`: control cluster `0.7097`, treatment cluster `0.7079`
  - step ratio `0.80`: control cluster `0.7024`, treatment cluster `0.7057`
- Winner:
  - `LIVE_NEURAL_GROUP_STEP_RATIO=0.50`
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050.json`
- Paired comparison versus current enhanced default (`backend/live_benchmark_replay_auto_v2_results_20260602_enhancedheur_default_adaptfix.json`):
  - control display cluster F1: `0.7038 -> 0.7097` (`+0.0059`), 95% CI `[-0.0115, +0.0270]`, `p=0.5880`
  - treatment display cluster F1: `0.7058 -> 0.7079` (`+0.0021`), 95% CI `[-0.0112, +0.0168]`, `p=0.7798`
  - control display offset F1: `0.7457 -> 0.7493` (`+0.0035`)
  - treatment display offset F1: `0.7445 -> 0.7473` (`+0.0027`)
  - display note F1 is effectively unchanged (`control -0.0001`, treatment `+0.0014`).
- Paired comparison versus old mel baseline (`backend/live_benchmark_replay_auto_v2_results_20260531_onsetbase046.json`):
  - control display cluster F1: `0.6775 -> 0.7097` (`+0.0321`), 95% CI `[+0.0015, +0.0656]`, `p=0.0519`
  - treatment display cluster F1: `0.6729 -> 0.7079` (`+0.0350`), 95% CI `[+0.0043, +0.0687]`, `p=0.0345`
  - control display note F1: `0.8985 -> 0.9237` (`+0.0252`)
  - treatment display note F1: `0.8956 -> 0.9227` (`+0.0271`)
  - control display offset F1: `0.7048 -> 0.7493` (`+0.0445`)
  - treatment display offset F1: `0.7033 -> 0.7473` (`+0.0440`)
- Decision:
  - Keep `LIVE_NEURAL_GROUP_STEP_RATIO=0.50` as the shipped default.
  - The incremental improvement over the enhanced adaptive-cap default is modest and not statistically significant on its own, but it raises both control and treatment cluster F1 and offset F1 with no meaningful note-F1 cost.

### TESTED / REJECTED: chord-member confidence outlier pruning

- Idea:
  - Within simultaneous note groups, drop very weak chord members only if they are both below an absolute onset-probability floor and below a ratio of the group median onset probability.
  - This avoids interval-based harmonic filtering, which previously destroyed real piano voicings.
- Implementation:
  - `backend/detect_note.py`
    - Added disabled-by-default env-gated group pruning:
      - `LIVE_NEURAL_GROUP_PRUNE_ENABLED`
      - `LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE`
      - `LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET`
      - `LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO`
- Sweep results:
  - `abs=0.55`, `ratio=0.55`, `min_size=3`: control cluster `0.7038`, treatment cluster `0.7058`
  - `abs=0.60`, `ratio=0.60`, `min_size=3`: control cluster `0.7038`, treatment cluster `0.7058`
  - `abs=0.65`, `ratio=0.70`, `min_size=3`: control cluster `0.7038`, treatment cluster `0.7057`
  - `abs=0.60`, `ratio=0.60`, `min_size=4`: control cluster `0.7038`, treatment cluster `0.7057`
- Decision:
  - Reject as a default. The tested settings were effectively neutral/no-op and did not raise cluster F1.
  - Keep the env-gated implementation available for future diagnostics, but leave `LIVE_NEURAL_GROUP_PRUNE_ENABLED=0`.

### ADDED: first continuous WebSocket live-stream backend

- Motivation:
  - The existing live path still treats each frontend recording chunk as a mini transcription job, then shifts/dedupes chunk-relative output back onto an absolute timeline.
  - That is a workable benchmark harness, but it is not a true live architecture: packet/chunk boundaries leak into semantic note decisions, and the score layer receives unstable intermediate guesses as if they were final.
  - The new direction separates transport chunking from semantic state:
    - audio packets are just transport,
    - backend session owns the continuous timeline,
    - neural windows are rolling observations,
    - note hypotheses become stable over time.
- `backend/main.py`
  - Added `ContinuousLiveStreamSession`.
  - Added packet decoder `_decode_stream_packet_audio(...)`.
  - Added global `_continuous_live_stream_sessions`.
  - Added WebSocket endpoint:
    - `/live/stream`
- WebSocket protocol:
  - Start:
    - `{"type":"start","session_id":"...","sample_rate":44100,"inference_interval_ms":100,"trusted_delay_ms":180,"commit_delay_ms":500,"lock_delay_ms":2000}`
  - Audio packet as PCM16 base64:
    - `{"type":"audio_packet","sample_rate":44100,"encoding":"pcm16","pcm16_base64":"..."}`
  - Audio packet as float samples:
    - `{"type":"audio_packet","sample_rate":44100,"samples":[...]}`
  - Flush / stop:
    - `{"type":"flush"}`
    - `{"type":"stop"}`
- Streaming session behavior:
  - Maintains an audio ring buffer on absolute session time.
  - Appends small frontend packets without treating them as semantic analysis chunks.
  - Runs enhanced live neural inference on rolling windows at a configurable tick (default `100 ms`).
  - Uses context window defaulting to `LIVE_CONTEXT_SEC` (`2.4s`).
  - Converts model output from window-relative time to absolute session time.
  - Updates note hypotheses by pitch/onset matching instead of regenerating truth from scratch.
  - Emits note lifecycle layers:
    - `heard_notes`
    - `candidate_notes`
    - `active_notes`
    - `committed_notes`
    - `locked_notes`
  - Default timing:
    - trusted delay `180 ms`
    - commit delay `500 ms`
    - lock delay `2000 ms`
- Smoke validation:
  - Instantiated `ContinuousLiveStreamSession`.
  - Appended one second of silence.
  - Forced inference.
  - Enhanced model loaded successfully and returned a `live_stream_update` with zero notes:
    - `counts={'heard':0,'candidate':0,'active':0,'committed':0,'locked':0}`
- Decision:
  - Keep as the first backend-only prototype of the true streaming architecture.
  - This does not replace `/live/audio-chunk` yet and does not wire into the frontend score renderer yet.
  - Next step is frontend packet streaming over WebSocket and then feeding `committed_notes` into notation state with measure locking.

### ADDED: frontend WebSocket live-stream transport wiring

- Motivation:
  - Move the live tab's normal transport path away from repeatedly stopping the recorder, saving a WAV chunk, uploading it, and restarting capture.
  - Keep transport packet boundaries separate from semantic note decisions by sending native PCM buffers to the backend stream session.
- `app/index.tsx`
  - Added `USE_LIVE_STREAM_TRANSPORT=true`.
  - Added WebSocket URL derivation from `BACKEND_URL`:
    - `https -> wss`
    - `http -> ws`
  - Added stream payload types for `active_notes`, `committed_notes`, and `locked_notes`.
  - Added conversion from backend stream note hypotheses to the existing `AnalysisResult` / `NoteResult` shape used by the live score and recent event list.
  - Uses visible notes from:
    - `committed_notes`
    - `active_notes`
  - Keeps `candidate_notes` out of the score surface so very early guesses do not jitter the notation.
- Recorder behavior:
  - Uses `react-native-audio-record`'s native `AudioRecord.on("data", ...)` event.
  - Sends base64 PCM16 buffers directly over `/live/stream` as:
    - `{"type":"audio_packet","sample_rate":44100,"encoding":"pcm16","pcm16_base64":"..."}`
  - Starts the microphone once at session start and stops it once at session stop.
  - The old `/live/audio-chunk` uploader remains in the file behind the feature flag for fallback/debug.
- Session behavior:
  - Opens `/live/stream`.
  - Sends `start` with:
    - inference interval `100 ms`
    - trusted delay `180 ms`
    - commit delay `500 ms`
    - lock delay `2000 ms`
  - On stop, sends `stop` and waits briefly for the final forced backend update.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.
  - Direct `npx tsc --noEmit` from PowerShell was blocked by local execution policy (`npx.ps1`), so the check was rerun through `cmd /c`.
- Current caveat:
  - This wires live PCM packet transport and immediate note rendering.
  - It does not yet add chord grouping or frontend measure-lock rendering from `locked_notes`.
  - The existing score renderer receives stream notes through the old `AnalysisResult` shape; a dedicated stable/live notation layer is still the next architecture step.

### FIXED: live-stream notation duration and score follow behavior

- Problem:
  - The stream adapter emitted raw note seconds but not the notation fields the MusicXML renderer expects:
    - `start_beat`
    - `end_beat`
    - `note_value`
    - `note_divisions`
    - dotted/triplet metadata
  - Without those fields, the renderer often fell back to quarter-note defaults.
  - `PianoSheetMusic` also treated stream snapshots like chunk deltas, so early note durations were appended once and not updated as the backend hypothesis matured.
  - The score follow-tail mode only matched `method="live"`, so `method="live_stream"` did not auto-follow generated notation.
- `app/index.tsx`
  - Added stream duration quantization from seconds to beat units using the current BPM.
  - Added note-value metadata for stream notes before passing them to the score renderer.
  - Added app-side triplet metadata fields to `NoteResult`.
- `components/PianoSheetMusic.tsx`
  - Treats `analysis_summary.method === "live_stream"` as a live snapshot and replaces accumulated score notes on each update instead of appending stale note versions.
  - Enables follow-tail for both `live` and `live_stream` score updates.
- `components/osmdHTML.ts`
  - Added vertical follow-tail scrolling for wrapped portrait score layout.
  - Added a short programmatic-scroll guard so auto-scroll does not mark itself as a user drag.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### FIXED: live-stream history truncation and render-depth churn

- Problem:
  - The live-stream backend emits bounded lifecycle arrays:
    - `active_notes[-64:]`
    - `committed_notes[-256:]`
    - `locked_notes[-256:]`
  - The frontend was treating each `live_stream_update` as the complete score snapshot.
  - `PianoSheetMusic` then replaced its stream score with that bounded snapshot.
  - Result: notes from the beginning of a longer live recording could disappear from the score as newer notes arrived.
- `app/index.tsx`
  - Added a per-session `liveStreamNotePayloadsRef` map keyed by backend note id, falling back to pitch+onset.
  - Merges each stream update into the full-session note map before building `AnalysisResult`.
  - Clears that map only when a new WebSocket stream session starts.
  - Added `liveStreamAnalysisSignatureRef` so `setAnalysisResult` only runs when note/chord content materially changes.
- `components/PianoSheetMusic.tsx`
  - Kept the stable note/chord signature guard.
  - Kept no-op append protection so identical incoming notes/chords return previous state instead of creating fresh arrays.
- Reverted:
  - Removed the compact WebView content-height resizing experiment.
  - Removed the WebView-side rendered `contentHeight` message.
  - Restored compact portrait WebView scrolling/layout behavior to the earlier fixed-viewport path.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### ADDED: score-quality diagnostic report

- Motivation:
  - `display_cluster_f1` is a strict exact onset-cluster metric, not a full generated-score quality metric.
  - It can stay low even when note F1 is high if chords are mostly right but underclustered, overclustered, or missing one pitch.
  - The live UX needs a broader report that separates:
    - pitch coverage,
    - duration / offset quality,
    - note-value quality,
    - chord/set exactness versus partial correctness,
    - onset/grid alignment,
    - stability/revision behavior.
- Added:
  - `backend/score_quality_report.py`
- Usage:
  - `backend\env\Scripts\python.exe -B backend\score_quality_report.py <test_experiment_results.json>`
- Outputs:
  - `<results_stem>_score_quality_report.json`
  - `<results_stem>_score_quality_report.md`
- First run:
  - Source:
    - `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050.json`
  - Outputs:
    - `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050_score_quality_report.json`
    - `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050_score_quality_report.md`
- Key findings from the first report:
  - Control:
    - score-quality diagnostic index: `0.7737`
    - display note F1: `0.9248`
    - display offset F1: `0.8384`
    - exact cluster F1: `0.7007`
    - cluster Jaccard: `0.8867`
  - Treatment:
    - score-quality diagnostic index: `0.7740`
    - display note F1: `0.9249`
    - display offset F1: `0.8379`
    - exact cluster F1: `0.7012`
    - cluster Jaccard: `0.8872`
  - Interpretation:
    - The score is much closer on pitch-set overlap than exact cluster F1 alone suggests.
    - To reach exact cluster F1 `0.80` on this result, the system needs about `201` more exact cluster matches, assuming predicted/ground-truth cluster counts stay fixed.
    - The largest strict-cluster bucket is underclustering:
      - control underclustered matches: `321`
      - treatment underclustered matches: `320`
    - This suggests the next score-quality work should focus on chord completion / simultaneity grouping and notation stability, not just neural onset detection.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\score_quality_report.py` passed.
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: removed arbitrary score-quality index from report

- Problem:
  - The initial report included a weighted `score_quality_index`.
  - That number was arbitrary and could hide the specific failure modes the report is supposed to expose.
- `backend/score_quality_report.py`
  - Removed `score_quality_index` from JSON output.
  - Removed `score_quality` from the Markdown summary table.
  - Removed it from CLI console output.
  - Kept the underlying submetrics and failure breakdowns:
    - note F1,
    - offset F1,
    - note-value accuracy,
    - exact cluster F1,
    - cluster Jaccard,
    - under/over/pitch-conflict cluster counts,
    - stability/revision metrics,
    - boundary recall.
- Regenerated:
  - `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050_score_quality_report.json`
  - `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050_score_quality_report.md`
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\score_quality_report.py` passed.

### KEPT: enhanced adaptive threshold cap fix

- Problem:
  - `_select_live_neural_onset_threshold(...)` still had an old mel-baseline hard cap for the loud/dense profile:
    - `selected = min(0.46, base_onset_threshold + 0.02)`
  - This was correct for the old `mel_baseline` default around `0.46`, but wrong for calibrated enhanced mel at `0.75`.
  - In enhanced treatment/adaptive runs, loud/dense chunks could be pushed down to `0.46`, effectively undoing the enhanced threshold calibration.
- Implementation:
  - `backend/detect_note.py`
    - Keep the old `0.46` cap only when `base_onset_threshold <= 0.50`.
    - For enhanced/high-threshold models, allow `base + 0.02` up to `0.95`.
  - Old mel behavior is preserved for `LIVE_ONSET_BASE=0.46`.
  - Enhanced loud/dense adaptive chunks can now select `0.77` when the base is `0.75`.
- Benchmark:
  - Result file: `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedheur_default_adaptfix.json`
  - Compared against the previous best enhanced run `backend/live_benchmark_replay_auto_v2_results_20260602_enhancedmel_on075_v08_fh0.json`.
- Paired treatment/adaptive improvement:
  - display cluster F1: `0.6980 -> 0.7058` (`+0.0078`), 95% CI `[+0.0027, +0.0142]`, `p=0.0013`
  - display note F1: `0.9177 -> 0.9213` (`+0.0036`), 95% CI `[-0.0010, +0.0091]`, `p=0.1792`
  - display offset F1: `0.7414 -> 0.7445` (`+0.0031`), 95% CI `[-0.0016, +0.0086]`, `p=0.2625`
  - Control/fixed arm is unchanged, as expected.
- Final comparison versus old mel `backend/live_benchmark_replay_auto_v2_results_20260531_onsetbase046.json`:
  - control display cluster F1: `0.6775 -> 0.7038` (`+0.0262`)
  - treatment display cluster F1: `0.6729 -> 0.7058` (`+0.0329`), 95% CI `[+0.0028, +0.0657]`, `p=0.0427`
  - control display note F1: `0.8985 -> 0.9239` (`+0.0253`)
  - treatment display note F1: `0.8956 -> 0.9213` (`+0.0257`)
  - control display offset F1: `0.7048 -> 0.7457` (`+0.0409`)
  - treatment display offset F1: `0.7033 -> 0.7445` (`+0.0413`)
- Decision:
  - Keep. This is a true inherited-heuristic bug fix and makes adaptive mode compatible with the enhanced model.

### TESTED / REJECTED: enhanced offset, context, duplicate, and merge heuristics

- Harness:
  - Temporary runner: `backend/_tmp_enhanced_heuristic_sweep.py`
  - Logs: `backend/benchmark_artifacts/enhanced_heuristics_20260602/`
  - Aggregate summary: `backend/enhanced_mel_heuristic_sweep_20260602_all_summary.json`
  - Baseline for this sweep:
    - `LIVE_ENHANCED_ONSET_BASE=0.75`
    - `LIVE_ENHANCED_OFFSET_BASE=0.35`
    - `LIVE_CONTEXT_SEC=2.4`
    - `LIVE_ENHANCED_DUPLICATE_WINDOW_SEC=0.04`
    - `LIVE_ENHANCED_MERGE_GAP_SEC=0.0`
    - adaptive-cap fix enabled
- Baseline after adaptive-cap fix:
  - control: note F1 `0.9134`, display note F1 `0.9239`, display cluster F1 `0.7038`, display offset F1 `0.7457`, duplicates/100 `3.54`
  - treatment: note F1 `0.9115`, display note F1 `0.9213`, display cluster F1 `0.7058`, display offset F1 `0.7445`, duplicates/100 `3.54`

- Offset threshold sweep:
  - `LIVE_ENHANCED_OFFSET_BASE=0.25`: control cluster `0.7029`, display offset `0.7434`
  - `0.45`: control cluster `0.7038`, display offset `0.7446`
  - `0.55`: control cluster `0.7038`, display offset `0.7441`
  - Decision: reject. Default `0.35` keeps the same/better cluster F1 and best display offset F1.

- Context length sweep:
  - `LIVE_CONTEXT_SEC=1.8`: control cluster `0.6873`
  - `3.0`: control cluster `0.6841`
  - `3.6`: control cluster `0.6999`, display note F1 `0.9243`, display offset F1 `0.7470`, duplicates/100 `3.14`
  - Retuned `3.6s` context:
    - onset `0.70`: control cluster `0.6913`
    - onset `0.80`: control cluster `0.6961`
    - onset `0.85`: control cluster `0.6314`
  - Decision: reject for shipped default. `3.6s` improves duplicates and offset slightly, but loses primary display cluster F1 versus `2.4s`/`0.75`.

- Duplicate-window sweep:
  - Added optional enhanced decoder parameter/env:
    - `duplicate_window_sec`
    - `LIVE_ENHANCED_DUPLICATE_WINDOW_SEC`
  - `0.06`: control cluster `0.7038`, display note `0.9239`, duplicates/100 `3.54`
  - `0.08`: control cluster `0.7038`, display note `0.9237`, duplicates/100 `3.52`
  - Decision: neutral/reject as default. Wider window barely changes duplicates and slightly hurts display note F1 at `0.08`.

- Same-pitch merge-gap sweep:
  - Added optional enhanced decoder parameter/env:
    - `merge_gap_sec`
    - `LIVE_ENHANCED_MERGE_GAP_SEC`
  - `0.04`: control cluster `0.7034`, treatment cluster `0.7067`
  - `0.08`: control cluster `0.6949`, treatment cluster `0.6987`
  - Decision: reject as default. `0.04` gives a tiny treatment cluster gain but regresses control cluster; `0.08` clearly regresses.

- Final kept heuristic from this batch:
  - Adaptive threshold cap fix only.
  - Keep enhanced defaults:
    - `LIVE_ENHANCED_ONSET_BASE=0.75`
    - `LIVE_ENHANCED_OFFSET_BASE=0.35`
    - `LIVE_CONTEXT_SEC=2.4`
    - `LIVE_ENHANCED_DUPLICATE_WINDOW_SEC=0.04`
    - `LIVE_ENHANCED_MERGE_GAP_SEC=0.0`

### BENCHMARKED: current score-quality breakdown, 2026-06-03

- Ran the fixed 48-clip replay benchmark with current code and enhanced model:
  - Command: `backend\env\Scripts\python.exe -B backend\test_experiment.py --benchmark-manifest backend\live_benchmark_replay_auto_v2.json --no-run-retro-correction --output-json backend\live_benchmark_replay_auto_v2_results_20260603_current.json`
  - Report: `backend/live_benchmark_replay_auto_v2_results_20260603_current_score_quality_report.md`
- Current raw-cluster score-quality aggregate:
  - control: note F1 `0.9248`, offset F1 `0.8384`, note-value accuracy `26.7%`, exact cluster F1 `0.7007`, cluster Jaccard `0.8867`, weighted avg revisions `1.4592`
  - treatment: note F1 `0.9247`, offset F1 `0.8376`, note-value accuracy `26.8%`, exact cluster F1 `0.7002`, cluster Jaccard `0.8870`, weighted avg revisions `1.4476`
- Compared with previous `20260602_enhancedcluster_group_step050`:
  - control is exactly unchanged on paired display metrics.
  - treatment slipped slightly from the previous run, mostly one regression on `clip_033`; display cluster mean `0.7079 -> 0.7061`, display note mean `0.9227 -> 0.9225`, display offset mean `0.7473 -> 0.7470`.
- Failure breakdown:
  - 492 missing notes vs 153 extra notes in control. Recall is the larger note-level problem.
  - 370 duration/offset errors among 3964 matched notes.
  - 2906 note-value errors among 3964 evaluable matched notes.
  - Cluster exactness needs about 201 more exact cluster matches to hit `0.80` with the same predicted/GT cluster counts.
  - Cluster errors are dominated by underclustering: under `321`, over `133`, pitch conflicts `60`, unmatched GT clusters `120`.
  - Boundary diagnostics: 122 control boundary misses, 120 tagged `no_control_coarse_candidate`, so nearly all boundary misses are real missing detections rather than later quantization drift.
  - Stability is still expensive: weighted stabilization median about `1250 ms`, avg revisions about `1.46`, max revisions `5`.
- Extra diagnostic:
  - Ran slot-consensus cluster metric:
    - Command: `backend\env\Scripts\python.exe -B backend\test_experiment.py --benchmark-manifest backend\live_benchmark_replay_auto_v2.json --no-run-retro-correction --cluster-metric-slot-consensus --output-json backend\live_benchmark_replay_auto_v2_results_20260603_slotconsensus.json`
    - Report: `backend/live_benchmark_replay_auto_v2_results_20260603_slotconsensus_score_quality_report.md`
  - Slot consensus lowered cluster F1: control `0.7007 -> 0.6845`, treatment `0.7002 -> 0.6830`.
  - Decision: do not keep slot-consensus metric/behavior as an improvement. The cluster gap is not just score grouping jitter; it is mostly missing chord tones plus some duration/offset instability.
- Next likely improvement targets:
  - Model/postprocessing recall for dense chords and boundary-adjacent notes, especially `Pour le piano` and `Gnomenreigen` clips.
  - Duration/note-value decoding or live rhythm quantization; note-value accuracy is the weakest score-facing submetric.
  - Stabilization policy; the user experience likely benefits more from fewer late revisions than from tiny F1 changes.

### CHANGED: onset-to-onset score durations and stable measure layout, 2026-06-03

- User observation:
  - Acoustic duration/note-value inference is intrinsically unreliable for piano notation because of pedal, harmonics, decay, and expressive releases.
  - For live score display, the next onset/IOI is a better notation authority than trying to detect when the previous note stopped sounding.
- Implemented in `components/PianoSheetMusic.tsx`:
  - Added an IOI duration pass after onset/time grouping.
  - Displayed note durations now come from the distance to the next onset cluster, quantized to simple score values from whole through 32nd.
  - The final event uses the previous IOI as a provisional duration, or a quarter note if there is no context yet.
  - The pass intentionally strips triplet tuplets from IOI-derived durations for now, keeping this first version simple and stable.
  - Added a render guard so XML changes inside the current measure are still sent to OSMD even when the total measure count has not grown.
- Implemented in `components/osmdHTML.ts`:
  - Forced fixed measure widths in OSMD.
  - Kept portrait layout at 3 measures per system.
  - Enabled XML new-system handling and added stable new-system breaks every 3 measures from the generated MusicXML.
  - Set last-system scaling to `1.0` so the final line does not stretch while new measures are arriving.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: first live-stream latency reduction pass, 2026-06-03

- User report:
  - Live transcription falls behind real-time audio.
  - After about a 15 second recording, the app can take another 15-20 seconds to catch up.
- Likely bottleneck:
  - The live WebSocket path was requesting neural inference every `100 ms` while each inference used a rolling `2.4 s` context window.
  - If deployed GPU inference ever exceeds the inference interval, the WebSocket receive loop can fall behind packet arrival and process stale packets after recording stops.
- Implemented in `app/index.tsx`:
  - Added low-latency stream constants:
    - `LIVE_STREAM_CONTEXT_SEC = 1.8`
    - `LIVE_STREAM_INFERENCE_INTERVAL_MS = 250`
  - Start messages now send `context_sec` explicitly.
  - Audio packets now include `client_sent_at_ms`.
  - Added once-per-second `[LiveStream] latency` logs with:
    - recording wall elapsed
    - backend audio time
    - estimated audio backlog
    - server backlog
    - inference ms
    - neural/model real-time factors
    - packet count and skipped inference count
    - current onset threshold/profile
  - Aligned live stream score duration metadata to backend `audio_time_sec`.
- Implemented in `backend/main.py`:
  - Continuous live stream sessions now track packet count, packet sequence, first audio wall time, and stream backlog.
  - WebSocket packet handler passes packet timing metadata into the session.
  - Live stream updates include backlog and inference telemetry.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
- Next readout:
  - If `audioBacklogMs` grows steadily, backend inference/receive-loop throughput is the bottleneck.
  - If `audioBacklogMs` stays low but notes appear late, the bottleneck is likely frontend rendering/OSMD batching or note state delays (`trusted_delay_ms`, `commit_delay_ms`).

### CHANGED: lightweight live score strip above OSMD, 2026-06-03

- User goal:
  - Reduce perceived notation latency from OSMD/WebView rendering.
  - Show the previous and current measure immediately in a small live layer while the full OSMD score updates less often.
- Implemented in `components/LiveScoreStrip.tsx`:
  - Added a lightweight WebView/SVG renderer for the previous and current measure.
  - Draws a compact grand-staff preview directly from live note/chord observations without invoking OSMD.
  - Keeps this layer intentionally approximate so it can behave as immediate visual feedback rather than final engraving.
- Implemented in `app/index.tsx`:
  - Mounted the live strip directly above the existing OSMD WebView.
  - Increased `LIVE_OSMD_BATCH_MS` to `500` so OSMD acts as the steadier committed-score layer while the strip handles fast feedback.
- Follow-up correction:
  - The first strip version still used the same stabilized `analysisResult` as OSMD, so it inherited the same apparent measure lag.
  - Added a separate `livePreviewResult` fed by backend `heard_notes`, `candidate_notes`, and `active_notes` so the strip can show the earliest available neural hypotheses while OSMD remains stabilized.
  - Changed the strip's current-measure window to follow backend audio time instead of the latest detected note, avoiding a trailing window when detection is late.
  - Reduced dense-passage clutter by deduping repeated pitch/onset observations and drawing compact pitch dots without stems.
- Follow-up liveness UI:
  - Added ghost-note rendering in the live strip: heard/candidate notes appear as pale provisional marks, while active/committed/locked notes become darker.
  - Added a moving red audio-time now-line so the user can see the app tracking the performance even between note confirmations.
  - Kept the strip window anchored to backend audio time, with note time as a fallback only before audio-time metadata is available.
- Performance correction:
  - Replaced the live strip WebView/HTML/SVG implementation with native `react-native-svg` rendering.
  - Moved the now-line to a Reanimated UI-thread animation so it can move smoothly without React re-rendering the strip every frame.
  - Kept ghost-note semantics and the audio-time anchored two-measure window, but removed WebView document reloads and frame-by-frame SVG string reconstruction.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: faster live inference cadence with UI preview backpressure, 2026-06-03

- User goal:
  - Move visible live-preview latency closer to the warmed model runtime, without reducing model context, thresholds, or score accuracy.
  - Ensure the frontend can keep up when backend updates arrive faster.
- Implemented in `app/index.tsx`:
  - Lowered `LIVE_STREAM_INFERENCE_INTERVAL_MS` from `250` to `90`.
  - Kept `LIVE_STREAM_CONTEXT_SEC = 1.8`, trusted delay, commit delay, and OSMD batching unchanged so accuracy/stabilization behavior is not weakened.
  - Added `LIVE_PREVIEW_BATCH_MS = 66` and a coalescing preview queue:
    - backend updates can arrive as fast as the stream supports;
    - React preview state is flushed at most about 15fps;
    - if multiple preview updates arrive before the next flush, only the newest one is rendered.
  - Added `coalescedPreviewUpdates` to the existing `[LiveStream] latency` log so UI backpressure can be observed.
  - Added cleanup for pending preview flushes on start, stop, socket reset, failure, and unmount.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.

### CHANGED: log-driven live latency reduction, 2026-06-03

- User-provided log readout:
  - Warm inference was fast after startup: usually `13-17 ms`.
  - First inference still spiked to about `1539 ms`.
  - Audio packets arrived every about `40 ms`.
  - Effective inference cadence looked like about `280 ms`, suggesting the deployed stream was still behaving near the older `250 ms` interval rather than the requested `90 ms`.
- Implemented in `backend/main.py`:
  - Added a WebSocket `warmup` message for `/live/stream`.
  - The warmup runs `analyze_audio_live_neural` on a synthetic live-context buffer using the same live neural path as streaming inference.
  - The warmup does not append audio to the session, advance `sample_cursor`, or create hypotheses, so recording timestamps are not shifted.
- Implemented in `app/index.tsx`:
  - After `live_stream_started`, the app now sends `warmup` and waits before `AudioRecord.start()`.
  - Added backward tolerance: if an older backend deployment does not support WebSocket warmup, the app warns and continues instead of breaking recording.
  - Added `backendInferenceIntervalMs`, `backendContextMs`, and requested interval to `[LiveStream] latency` logs so future logs can prove whether the deployed backend accepted the low-latency interval.
  - Added `[LiveStream] warmed` logs with warmup timing.
- Implemented in `components/LiveScoreStrip.tsx`:
  - The moving now-line and visible two-measure window can now use the local recording clock as the primary timing source, with backend audio time as fallback.
  - This improves perceived real-time response without changing model context, thresholds, inference results, or score stabilization.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.

### EXPERIMENT: hand-independent onset-to-onset note extension, 2026-06-03

- User request:
  - Try extending every played note to the next onset in its own hand, to see how the generated score feels.
- Implemented in `components/PianoSheetMusic.tsx`:
  - Added per-hand IOI duration estimation for the score rhythm pass.
  - Treble events now use the distance to the next treble onset.
  - Bass events now use the distance to the next bass onset.
  - If a hand has no later onset, the pass falls back to the previous same-hand IOI, then to the global onset IOI.
  - This intentionally preserves model onsets and pitch decisions; it only changes score-facing displayed durations.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: 70ms stream cadence and packet-anchored backlog metric, 2026-06-03

- User-provided log readout:
  - Backend warmup worked and the deployed stream accepted `backendInferenceIntervalMs: 90`.
  - Effective inference cadence was still about `120 ms` because audio packets arrive every about `40 ms`, so a `90 ms` gate can only fire on every third packet.
  - Session-wall `audioBacklogMs` still included startup/recorder delay before the first audio packet, so it overstated true stream lag.
- Implemented in `app/index.tsx`:
  - Lowered `LIVE_STREAM_INFERENCE_INTERVAL_MS` from `90` to `70`, targeting every-second-packet inference at roughly `80 ms` cadence while keeping the same `1.8 s` model context.
  - Added `liveStreamFirstPacketSentAtRef` to anchor stream-lag measurement to the first actual audio packet.
  - Added `packetElapsedMs` and `packetAudioBacklogMs` to `[LiveStream] latency` logs while keeping the existing button/session-anchored `audioBacklogMs`.
  - Reset the first-packet anchor on stream open, recording start, stop, failure, and unmount.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: live preview UI-lag telemetry, 2026-06-03

- User concern:
  - The backend is now inferring at an effective `80 ms` cadence, but the live dot visualization may still feel a couple hundred milliseconds behind.
- Log readout before this change:
  - `coalescedPreviewUpdates` was regularly nonzero even with an `80 ms` inference cadence, suggesting preview updates may be arriving to JS in bursts or waiting behind UI work.
  - Existing logs did not distinguish backend data staleness from preview queue/render delay.
- Implemented in `app/index.tsx`:
  - Added preview queue telemetry:
    - `previewFlushes`
    - `droppedPreviewUpdates`
    - `previewLastQueueWaitMs`
    - `previewMaxQueueWaitMs`
    - `previewDataLagMs`
  - `previewDataLagMs` compares local first-packet time to the backend audio time represented by the preview result when it is flushed to React state.
  - Per-window preview counters reset after each `[LiveStream] latency` log.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: faster live preview flush and stale-preview bypass, 2026-06-03

- User-provided telemetry readout:
  - `previewDataLagMs` was usually around `200-250 ms`, with a max around `425 ms`.
  - `previewMaxQueueWaitMs` reached `100-200 ms`, showing that the preview queue itself could add visible lag.
  - Backend inference was already healthy at `70 ms` requested / `80 ms` effective cadence.
- Implemented in `app/index.tsx`:
  - Lowered `LIVE_PREVIEW_BATCH_MS` from `66` to `33`, targeting about `30 fps` preview updates.
  - Added `LIVE_PREVIEW_STALE_FLUSH_MS = 180`.
  - The live preview now flushes immediately when pending preview data is already at least `180 ms` behind local first-packet time.
  - Backend/model cadence, context, thresholds, and OSMD batching are unchanged.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: absolute-clock live preview now-line, 2026-06-03

- User observation:
  - The live dot preview is fast enough after the preview flush tuning, but the red time bar still appears to lag.
- Cause:
  - `LiveScoreStrip` was using `localElapsedSeconds={duration}`.
  - `duration` was advanced by a JS `setInterval` that added `0.1` seconds per tick, so it could drift when the JS thread was busy.
- Implemented in `app/index.tsx`:
  - Added an absolute `recordingStartedAtMs` state and `recordingTimerStartedAtRef`.
  - The duration timer now computes elapsed time from `Date.now() - recordingStartedAtMs` instead of incrementing by a fixed amount.
  - Passed `localStartedAtMs` into `LiveScoreStrip`.
- Implemented in `components/LiveScoreStrip.tsx`:
  - The red now-line now computes its beat position directly from `Date.now() - localStartedAtMs` inside the Reanimated frame callback.
  - `localElapsedSeconds` remains as a fallback, but the live path no longer depends on JS timer increments for the now-line.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: decoupled live WebSocket receive and inference worker, 2026-06-03

- User goal:
  - Reduce preview latency without changing model context, thresholds, or accuracy behavior.
  - Keep receiving audio packets while inference runs instead of blocking the WebSocket receive loop during each model call.
- Implemented in `backend/main.py`:
  - Added an async inference worker for `/live/stream`.
  - The WebSocket receive loop now decodes/appends audio packets and signals the worker, then immediately returns to reading packets.
  - The inference worker runs `maybe_run_inference` on the latest rolling buffer and sends updates through a locked send path.
  - Added session thread-safety:
    - short lock around audio buffer appends/snapshots/status/hypothesis updates;
    - separate inference lock so flush/stop/worker cannot run overlapping model calls;
    - model inference runs outside the audio-buffer lock so packet receive can continue.
  - Added `transport_mode: "decoupled"` to stream session status.
- Implemented in `app/index.tsx`:
  - Added `transportMode` to `[LiveStream] latency` logs so deployed behavior can be verified.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: streaming continuity filter for pedaled boundary rebirths, 2026-06-04

- User hypothesis:
  - Pedal may be causing havoc because sustained notes whose true onsets happened before the current rolling window can be re-detected as new notes near the next window boundary.
  - Same-pitch repeats should require clear attack evidence; otherwise they are probably sustained/pedaled continuity, not new score events.
- Implemented in `backend/main.py`:
  - Added per-observation attack metrics from the rolling audio window:
    - `attack_ratio`
    - `attack_delta`
    - `has_strong_attack`
  - Added `_filter_stream_continuity` before live hypotheses are updated.
  - The filter suppresses newly-created observations when:
    - the same pitch was already active/recent and the new observation appears near the analysis-window boundary with weak attack;
    - the same pitch is repeated implausibly fast with weak attack;
    - a weak-attack upper pitch is explainable as a harmonic of a recently sounding lower pitch.
  - Existing matched hypotheses still update normally, so stable notes are not erased by the filter.
  - Added cumulative/per-inference continuity counters to the live stream payload.
- Implemented in `app/index.tsx`:
  - Added continuity-filter fields to the live stream type.
  - Added `continuitySuppressed`, `continuitySamePitchBoundary`, `continuityImplausibleRepeat`, `continuityHarmonicSustain`, and `continuityTotalSuppressed` to `[LiveStream] latency` logs.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
  - `cmd /c npx tsc --noEmit` passed.
  - Smoke-tested the attack metric and verified a shifted same-pitch boundary rebirth with weak attack is suppressed.

### CHANGED: note-birth and suppression diagnostics for live pedal failures, 2026-06-04

- User observation:
  - The one-chord pedaled ring-out test showed many single notes on screen even though sampled latency logs showed only sparse observations.
  - The existing logs sampled one inference per second and did not expose note births, so they could miss hypotheses created between sampled lines.
- Implemented in `backend/main.py`:
  - `_update_hypotheses` now returns per-inference hypothesis diagnostics:
    - created note count;
    - matched observation count;
    - stale skipped count;
    - active/committed/locked promotion counts;
    - compact `birth_samples` with MIDI, onset, confidence, source, attack ratio/delta, and hypothesis id.
  - `_filter_stream_continuity` now always reports cumulative `total_suppressed`, even on zero-observation updates.
  - Suppressed candidates now include compact `suppressed_samples` with MIDI, onset, confidence, attack evidence, source, reason, and harmonic/repeat context.
- Implemented in `app/index.tsx`:
  - Added live debug sample types.
  - Accumulates note birth/suppression/promotion diagnostics across all live stream updates between the once-per-second `[LiveStream] latency` logs.
  - Logs:
    - `noteBirths`, `noteMatches`, `noteStaleSkipped`, `noteSuppressed`;
    - `notePromotedActive`, `notePromotedCommitted`, `notePromotedLocked`;
    - recent `noteBirthSamples` and `noteSuppressedSamples`;
    - `liveCounts`.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: attack-group gated note birth for live pedal ring-out, 2026-06-04

- User goal:
  - Reconcile onset-only precision with frame/sustain recall without relying on unconstrained frame-based note creation.
  - Stop the one-chord pedaled ring-out failure where weak/no-attack candidates became many committed single notes.
- Implemented in `backend/main.py`:
  - Added short-lived live attack groups created by observations with strong broadband attack evidence.
  - Unmatched weak-attack observations can now create notes only if they are inside a short attack-group rescue window or have extremely high confidence.
  - Matched observations still update existing hypotheses.
  - Strong-attack observations still create new hypotheses normally.
  - Added a new continuity suppression reason and counter: `weak_birth_outside_attack`.
  - Added attack-group telemetry:
    - `attack_groups`
    - `registered_attack_groups`
- Implemented in `app/index.tsx`:
  - Added `continuityWeakBirthOutsideAttack`, `continuityAttackGroups`, and `continuityRegisteredAttackGroups` to `[LiveStream] latency` logs.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
  - `cmd /c npx tsc --noEmit` passed.
  - Targeted smoke test kept initial strong chord attacks and suppressed weak/no-attack follow-up notes as `weak_birth_outside_attack`.

### CHANGED: perceptually grounded same-staff chord grouping tolerance, 2026-06-04

- User goal:
  - Reduce visually broken-up same-staff chords after attack-group note birth, where notes a few milliseconds apart were rendered as separate 32nds.
- Rationale:
  - The previous `5 ms` same-staff tolerance was stricter than a musically useful simultaneity threshold.
  - Updated to `20 ms`, grounded in Ira Hirsh's auditory temporal-order range: below roughly this range, temporal order is not reliably perceived as separate musical events. Reference: Hirsh's temporal-order work as discussed in "The Times of Ira Hirsh: Multiple Ranges of Auditory Temporal Perception" (https://pmc.ncbi.nlm.nih.gov/articles/PMC1363770/).
- Implemented in `components/PianoSheetMusic.tsx`:
  - Changed `SAME_STAFF_TOLERANCE` from `0.005` to `0.02`.
  - Changed the local `CHORD_MERGE_TOLERANCE` from exact-time matching (`0`) to `SAME_STAFF_TOLERANCE`, so notes grouped as same-staff simultaneous are not split apart by the later chord-merge pass.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: piano-specific 30ms notation simultaneity experiment, 2026-06-04

- User goal:
  - Try the piano-specific onset-asynchrony range from Goebl & Parncutt after dense pedaled/chord-heavy material still showed broken-up attacks.
- Rationale:
  - Goebl & Parncutt's "Perception of onset asynchronies: Acoustic piano versus synthesized complex versus pure tones" tested onset asynchronies in 10ms steps from -50ms to +50ms and found reliable ordering for acoustic piano closer to about `30 ms`, slightly looser than the classic Hirsh `20 ms` temporal-order range.
  - Reference: https://www.ofai.at/~werner.goebl/papers/goebl_parncutt_smpc2001.pdf
- Implemented in `components/PianoSheetMusic.tsx`:
  - Changed score-side `SAME_STAFF_TOLERANCE` from `0.02` to `0.03`.
  - Changed `CROSS_STAFF_TIME_TOLERANCE` from `0.025` to `0.03` so cross-staff grouping does not silently cap the 30ms same-staff experiment.
  - Made `BEAT_GROUP_TOLERANCE` and `SAME_STAFF_BEAT_TOLERANCE` at least as wide as the corresponding 30ms time tolerance converted into beats at the current BPM, while still preserving the existing minimum grid tolerances.
  - Kept `CHORD_MERGE_TOLERANCE = SAME_STAFF_TOLERANCE`.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: disable harmonic explain-away for treble melody range, 2026-06-04

- User observation:
  - Dense, pedaled, chord-heavy material still lost much of the melody after attack-group gating.
  - Logs showed many upper/melody-range candidates being suppressed as `harmonic_sustain`, including MIDI values above middle C.
- Experiment:
  - Keep attack-group birth gating and ring-out protection.
  - Stop using the broad harmonic explain-away rule for melody-range candidates.
- Implemented in `backend/main.py`:
  - Added `STREAM_HARMONIC_SUPPRESS_MAX_MIDI = 59`.
  - `harmonic_sustain` suppression now applies only when the candidate pitch is `<= 59`.
  - MIDI `60+` candidates can still be suppressed by other guards such as `weak_birth_outside_attack`, but they are no longer discarded merely because they are interval-related to a lower sounding note.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
  - Focused smoke check confirmed MIDI `73` is no longer classified as `harmonic_sustain`; weak/no-attack candidates outside attack groups still route through `weak_birth_outside_attack`.

### CHANGED: attack-group-gated harmonic suppression, 2026-06-04

- User observation:
  - The MIDI `60+` harmonic carve-out did not recover melody in dense pedaled material because much of the melody lived in the middle register.
- Experiment:
  - Replace the range-based carve-out with an attack-context rule.
  - Harmonic explain-away should only fire during ring-out/no-attack contexts, not near a real attack group.
- Implemented in `backend/main.py`:
  - Removed `STREAM_HARMONIC_SUPPRESS_MAX_MIDI`.
  - Moved attack-group lookup before harmonic suppression.
  - `harmonic_sustain` suppression now requires `attack_group is None` in addition to weak attack and confidence below `0.78`.
  - Candidates that look harmonically related but occur near a registered attack group are allowed to continue into normal attack-group rescue / note-birth logic.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py` passed.
  - Focused smoke check confirmed a weak harmonic-looking middle-register candidate is kept near an attack group, while the same candidate is suppressed as `harmonic_sustain` during ring-out.

### CHANGED: strict 10ms notation grouping diagnostic, 2026-06-04

- User observation:
  - Dense pedaled melody still dropped out after backend harmonic suppression was made attack-group-aware.
  - User suspected chord grouping may be absorbing melody notes rather than the backend failing to emit them.
- Experiment:
  - Temporarily narrow score-side chord/time grouping to `10 ms` to test whether melody returns when only extremely close onsets are merged.
  - This intentionally steps below the earlier Hirsh/Goebl perceptual ranges; it is a diagnostic setting, not a final perceptual claim.
- Implemented in `components/PianoSheetMusic.tsx`:
  - Changed `SAME_STAFF_TOLERANCE` from `0.03` to `0.01`.
  - Changed `CROSS_STAFF_TIME_TOLERANCE` from `0.03` to `0.01`.
  - Existing beat tolerances still derive from the current time tolerances and BPM.
  - `CHORD_MERGE_TOLERANCE` continues to follow `SAME_STAFF_TOLERANCE`.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### REVERTED: strict 10ms notation grouping diagnostic, 2026-06-04

- User result:
  - Dropping score-side grouping to `10 ms` did not restore the missing melody in dense pedaled material.
  - This suggests chord grouping was not the primary cause of the dropout.
- Reverted in `components/PianoSheetMusic.tsx`:
  - Restored `SAME_STAFF_TOLERANCE` from `0.01` to `0.03`.
  - Restored `CROSS_STAFF_TIME_TOLERANCE` from `0.01` to `0.03`.
  - Kept beat tolerances derived from the active time tolerances and BPM.
- Next suspected suppressors:
  - backend `weak_birth_outside_attack`;
  - same-pitch / implausible-repeat suppression;
  - candidate aging before commit;
  - score-side duplicate-MIDI removal inside the same staff/time group.
- Validation:
  - `cmd /c npx tsc --noEmit` passed.

### CHANGED: wider attack-group rescue window, 2026-06-04

- User result:
  - Reverting the strict `10 ms` score grouping diagnostic did not restore the missing dense pedaled melody.
  - This points back to backend note-birth gating rather than notation chord grouping.
- Experiment:
  - Increase the live attack-group rescue window from `0.16 s` to `0.25 s`.
  - Goal: let weak/middle-range melody candidates survive when they are plausibly part of a nearby real attack, while keeping ring-out-only candidates outside the rescue window suppressed.
- Implemented in `backend/main.py`:
  - Changed `STREAM_ATTACK_GROUP_RESCUE_SEC` from `0.16` to `0.25`.

### ADDED: decoder-only tuning sweep harness, 2026-06-04

- User goal:
  - Improve transcription/score accuracy without increasing latency.
  - Avoid a larger/slower model; tune decoder decisions against the existing benchmark metrics instead.
- Added `backend/tune_decoder_settings.py`:
  - Runs `backend/test_experiment.py` repeatedly with candidate environment-variable decoder settings.
  - Sweeps zero-latency knobs such as:
    - `LIVE_ENHANCED_ONSET_BASE`;
    - `LIVE_ENHANCED_OFFSET_BASE`;
    - `LIVE_ENHANCED_DUPLICATE_WINDOW_SEC`;
    - `LIVE_ENHANCED_FILTER_HARMONICS`;
    - live neural grouping/pruning tolerances in the full preset.
  - Writes per-candidate benchmark JSON/log files under `backend/benchmark_artifacts/decoder_tuning_*`.
  - Writes `decoder_tuning_summary.json` and `decoder_tuning_summary.md`.
  - Ranks candidates by score submetrics directly:
    - display note precision/recall/F1;
    - display offset F1;
    - display cluster F1;
    - cluster average Jaccard;
    - strict onset F1;
    - duplicate rate;
    - measured chunk timing.
  - Includes a latency guard so a candidate is marked only if its mean clip-level p95 chunk time stays within the configured tolerance of baseline.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\tune_decoder_settings.py` passed.
  - Smoke-ran one real candidate on `clip_001`:
    - command: `backend\env\Scripts\python.exe backend\tune_decoder_settings.py --preset quick --clip-ids clip_001 --max-candidates 1 --no-warmup`
    - output: `backend/benchmark_artifacts/decoder_tuning_20260604_112716/decoder_tuning_summary.md`
    - baseline smoke metrics: display note F1 `0.9388`, recall `0.8846`, precision `1.0000`, display cluster F1 `0.8947`, mean clip p95 chunk `34.54 ms`.

### ADDED: continuous-stream decoder replay benchmark, 2026-06-04

- User result:
  - The decoder-only `test_experiment.py` sweep was disappointing and likely measured the older chunk-upload replay path rather than the newer continuous stream hypothesis decoder.
  - The real live improvement came from `ContinuousLiveStreamSession` attack-group note-birth gating, so benchmark coverage needed to move there.
- Added `backend/tune_continuous_stream_decoder.py`:
  - Replays benchmark audio as small PCM packets through `ContinuousLiveStreamSession`.
  - Uses live app defaults by default:
    - `40 ms` packets;
    - `1.8 s` context;
    - `70 ms` inference interval;
    - `180 ms` trusted delay;
    - `500 ms` commit delay;
    - `2000 ms` lock delay.
  - Measures both surfaces:
    - `score`: mirrors the app's accumulated score notes (`active`, `committed`, `locked`);
    - `preview`: includes unstable `heard`/`candidate` notes as well.
  - Sweeps continuous decoder constants directly:
    - `STREAM_ATTACK_GROUP_RESCUE_SEC`;
    - `STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE`;
    - `STREAM_MIN_REPEAT_SEC`;
    - `STREAM_WEAK_BIRTH_HIGH_CONFIDENCE`;
    - `STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE`;
    - `STREAM_CONTINUITY_BOUNDARY_SEC`.
  - Writes per-candidate JSON/log files and a ranked Markdown/JSON summary under `backend/benchmark_artifacts/continuous_decoder_*`.
  - Ranks using score-quality submetrics directly:
    - note precision/recall/F1;
    - offset F1;
    - cluster F1 and cluster Jaccard;
    - strict onset F1;
    - boundary recall;
    - duplicate rate;
    - p95 inference timing.
  - Redirects verbose model timing logs into per-candidate `.log` files by default; use `--show-model-logs` to print them.
- Refactored defaults in `backend/main.py` without behavior change:
  - Added `STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE = 0.50`.
  - Added `STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE = 0.78`.
  - Replaced the previous inline `0.50` and `0.78` checks with those named constants so the replay benchmark can sweep them.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\main.py backend\tune_continuous_stream_decoder.py` passed.
  - Smoke-ran one candidate on `clip_001`:
    - command: `backend\env\Scripts\python.exe backend\tune_continuous_stream_decoder.py --preset quick --clip-ids clip_001 --max-candidates 1 --tail-padding-sec 0.2`
    - output: `backend/benchmark_artifacts/continuous_decoder_20260604_121331/continuous_decoder_summary.md`
    - baseline smoke metrics on score surface: note F1 `0.6500`, recall `0.5000`, precision `0.9286`, cluster F1 `0.5333`, p95 inference `38.52 ms`.

### CHANGED: ship 200ms same-pitch repeat suppression, 2026-06-04

- User decision:
  - Ship the cleanest continuous-stream benchmark win.
  - Hold off on any cluster-cleaner/post-decoder grouping work for now.
- Benchmark evidence:
  - Source: `backend/benchmark_artifacts/continuous_decoder_20260604_121946/continuous_decoder_summary.md`.
  - Candidate: `repeat_200ms`.
  - Change: `STREAM_MIN_REPEAT_SEC = 0.20`.
  - Score-surface aggregate on clips `clip_001`, `clip_002`, `clip_010`, `clip_017`, `clip_031`:
    - note F1 `0.8768 -> 0.8802`;
    - precision `0.8793 -> 0.8861`;
    - recall unchanged at `0.8743`;
    - cluster F1 `0.4508 -> 0.4586`;
    - offset F1 `0.8462 -> 0.8495`;
    - predicted notes `522 -> 518`;
    - matched notes unchanged at `459`.
  - Interpretation:
    - The wider repeat suppression removed four likely false same-pitch rebirths without losing matched notes.
- Implemented in `backend/main.py`:
  - Changed `STREAM_MIN_REPEAT_SEC` from `0.16` to `0.20`.

### CHANGED: make enhanced-model fine-tuning replay-safe, 2026-06-04

- User problem:
  - Hard-case-only fine-tuning produced a worse checkpoint than the prior enhanced model.
  - Diagnosis: this was closer to narrow retraining than careful fine-tuning, with risk of catastrophic forgetting and a validation distribution that was no longer apples-to-apples.
- Implemented in `backend/rhythm_training/train_enhanced_mel_transcriber.py`:
  - Added mixed fine-tune sampling:
    - general MAESTRO replay samples plus hard-case manifest samples;
    - default hard-case sampling ratio `0.25`;
    - configurable via `--finetune-hard-ratio` and `--finetune-samples-per-epoch`.
  - Added source tagging for batches:
    - `sample_source=0` for general replay;
    - `sample_source=1` for hard-case manifest samples.
  - Added teacher preservation during fine-tuning:
    - default teacher is the init checkpoint;
    - default preservation weight `0.20`;
    - applies to general replay samples only, so hard cases can still adapt;
    - preserves onset, offset, frame, velocity, and note-value outputs with configurable term weights.
  - Added dual validation for fine-tuning:
    - hard-case validation remains the primary target when a validation manifest is provided;
    - optional general validation runs in parallel by default;
    - checkpoint selection uses a weighted hard/general metric, defaulting to equal weight via `--finetune-general-val-weight 0.50`.
  - Added fine-tune-safe defaults:
    - `--lr` now defaults to `0.08` for `--finetune` and `1.0` otherwise;
    - default fine-tune output remains `backend/rhythm_training/enhanced_mel_transcription_finetuned.pt`, so reruns overwrite the old fine-tune checkpoint unless `--model-path` is supplied.
  - Fine-tune parameter groups remain available:
    - `--finetune-scope heads|decoder|full`;
    - head/decoder/backbone LR scales.
- Validation:
  - `backend\env\Scripts\python.exe -B -m py_compile backend\rhythm_training\train_enhanced_mel_transcriber.py` passed.
  - `backend\env\Scripts\python.exe backend\rhythm_training\train_enhanced_mel_transcriber.py --benchmark --device cpu --benchmark-runs 1` passed.

### CHANGED: diagnose enhanced score-duration accuracy and add pedal/sounding-score instrumentation, 2026-06-09

- User goal:
  - Test whether MAESTRO sustain-pedal data can improve displayed score durations without spending more 20-hour training runs on low-yield fine-tuning.
- Implemented in `backend/rhythm_training/train_enhanced_mel_transcriber.py`:
  - Added MAESTRO `CC64` sustain-pedal extraction:
    - continuous pedal target `0..1`;
    - binary sustain intervals using threshold `CC64 >= 64`.
  - Added pedal-informed `sounding_frame` target:
    - note sound extends through sustain-pedal release;
    - extension is capped by the next same-pitch onset;
    - unrelated pitches do not terminate held/sustained notes, preserving voicing-style holds.
  - Added new model outputs:
    - `pedal_head`;
    - `sounding_frame_head`.
  - Added new loss terms and CLI flags:
    - `--pedal-weight`;
    - `--sounding-frame-weight`.
  - Added validation reporting for:
    - `sound_f1`;
    - `pedal_f1`;
    - score-style F1.
  - Added validation workload caps:
    - `--max-val-samples`;
    - `--val-sampling`;
    - `--max-score-val-samples`;
    - `--score-val-sampling`.
  - Added score-duration policy support:
    - `--score-duration-policy`;
    - default policy set to `ioi_same_hand` after offline sweep.
  - Added MIDI target caching and persistent DataLoader workers:
    - per-piece MIDI parsing is cached;
    - DataLoader workers persist across epochs so caches survive epoch boundaries.
  - Made resume more robust when changing fine-tune scope:
    - if optimizer/scheduler state does not fit the current parameter groups, resume now warns and uses a fresh optimizer/scheduler while still loading model weights.
  - Fixed note-value loss/decoder alignment:
    - note-value loss now trains only on frames where both onset target and physical frame target are active;
    - note-value decode pooling now looks forward from onset instead of pooling pre-onset/sustain regions.
- Added diagnostic scripts:
  - `backend/rhythm_training/diagnose_score_duration_errors.py`
    - dumps score-duration mismatches where pitch and score-grid onset match but duration class fails;
    - writes JSON and CSV under `backend/rhythm_training/score_duration_diagnostics/`.
  - `backend/rhythm_training/sweep_score_duration_policies.py`
    - compares duration-assignment policies without retraining.
  - `backend/rhythm_training/learn_score_duration_lookup.py`
    - experimental lookup-table chooser for duration policy selection.
- Offline findings:
  - Fine-tuning did not meaningfully improve score duration:
    - hard event F1 stayed near `0.949-0.950`;
    - general event F1 stayed near `0.960-0.961`;
    - score onset F1 stayed near `0.903-0.904`;
    - score duration accuracy stayed near `0.136-0.157` when using the note-value head.
  - Diagnostic run on 512 hard-validation samples:
    - score F1 `0.1416`;
    - score onset F1 `0.9028`;
    - duration accuracy `0.1569`.
  - Duration target analysis:
    - GT score class matched physical duration about `19.8%`;
    - GT score class matched sounding/pedal duration about `54.4%`;
    - model decoded duration matched GT class about `21.2%`.
  - Policy sweep on 512 hard-validation samples:
    - `head`: F1 `0.1416`, duration accuracy `0.1569`;
    - `physical_duration`: F1 `0.1812`, duration accuracy `0.2007`;
    - `sounding_duration`: F1 `0.1913`, duration accuracy `0.2120`;
    - `sounding_same_pitch_cap`: F1 `0.1987`, duration accuracy `0.2201`;
    - `hybrid_cleanup`: F1 `0.3469`, duration accuracy `0.3843`;
    - `ioi_same_hand`: F1 `0.4506`, duration accuracy `0.4991`;
    - oracle over available candidate policies: F1 `0.5334`, duration accuracy `0.5909`.
  - Extra simple hybrids did not beat `ioi_same_hand`.
  - Experimental lookup-table chooser on a 512-sample even/odd validation split improved held-out duration accuracy:
    - `ioi_same_hand` baseline `0.4893`;
    - best lookup (`ioi_head_sound:min4`) `0.5579`.
    - Not promoted as default yet because train-to-validation lookup run timed out under current machine contention.
- Current interpretation:
  - The enhanced transcriber already finds the notes/onsets well.
  - Displayed score duration is primarily a post-processing/notation problem, not a model fine-tuning problem.
  - Best safe deployable policy so far:
    - assign visible duration from next onset in the same hand;
    - cap by next same-pitch onset.
  - Remaining upside likely comes from a lightweight per-note chooser between IOI/sounding/head-derived candidates, not from further GPU fine-tuning.
- Validation:
  - `python -m py_compile backend/rhythm_training/train_enhanced_mel_transcriber.py backend/rhythm_training/diagnose_score_duration_errors.py backend/rhythm_training/sweep_score_duration_policies.py backend/rhythm_training/learn_score_duration_lookup.py` passed.
  - `python diagnose_score_duration_errors.py --samples 512 --batch-size 8 --num-workers 4 --device cuda` completed and wrote `score_duration_diagnostics/score_duration_summary.json` plus `score_duration_diagnostics/score_duration_mismatches.csv`.
  - `python sweep_score_duration_policies.py --samples 512 --batch-size 8 --num-workers 4 --device cuda` completed and wrote `score_duration_diagnostics/policy_sweep_summary.json`.
  - `python learn_score_duration_lookup.py --samples 512 --batch-size 8 --num-workers 0 --device cuda --output-path score_duration_diagnostics/lookup_policy_summary_512.json` completed.

## CHANGED: promote same-hand IOI score durations into live pipeline, 2026-06-09

- Updated `backend/live_rhythm.py` so the main live score quantization path now uses the deployable score-duration policy found by diagnostics:
  - default policy is `LIVE_SCORE_DURATION_POLICY=ioi_same_hand`;
  - visible note duration is assigned from the next onset in the same hand;
  - duration is capped by the next same-pitch onset when that happens first;
  - hand falls back from `midi_note < 60` when no explicit `hand` field is present.
- Applied the policy in both places that write score-facing durations:
  - immediate/coarse live quantization now passes policy-aware `next_onset_seconds`;
  - deferred Viterbi refinement keeps its refined onset/grid placement, then replaces only the visible duration fields with same-hand IOI duration where lookahead exists.
- Added score-duration metadata to policy-adjusted notes:
  - `duration_source='ioi_same_hand'`;
  - `score_duration_seconds=<policy duration>`.
- The policy can be disabled/restored to old adjacent-onset behavior with:
  - `LIVE_SCORE_DURATION_POLICY=adjacent` or any value other than `ioi_same_hand`.
- Validation:
  - `python -m py_compile backend/live_rhythm.py backend/main.py` passed.
  - Coarse smoke test confirmed an interleaved bass note at `0.0s` with treble notes at `0.25s/0.50s` and next bass onset at `1.0s` becomes a 2-beat half note instead of ending at the first treble onset.
  - Deferred-refinement smoke test confirmed the same case keeps `duration_source=ioi_same_hand` after refinement instead of being overwritten by regular Viterbi duration.

## CHANGED: add soft-polyphony rescue for quiet inner voices, 2026-06-09

- Added a conservative rescue path to `backend/rhythm_training/train_enhanced_mel_transcriber.py`:
  - primary enhanced decoding still uses the normal onset threshold;
  - a secondary per-key pass can recover below-threshold local onset peaks when the same key has frame evidence, velocity evidence, and a local onset bump;
  - rescued events are marked with `decode_source='soft_polyphony_rescue'` and `rescue_onset_delta`.
- Threaded the rescue through `backend/gpu_ops.py` for the GPU enhanced mel transcriber:
  - `soft_polyphony_rescue`;
  - `soft_polyphony_onset_threshold`;
  - `soft_polyphony_frame_threshold`;
  - `soft_polyphony_min_velocity`;
  - `soft_polyphony_min_delta`;
  - `soft_polyphony_lookback_sec`;
  - timing metadata now includes `soft_polyphony_rescued_events`.
- Enabled the rescue in live enhanced mode in `backend/detect_note.py`:
  - default `LIVE_ENHANCED_SOFT_POLYPHONY_RESCUE=1`;
  - default `LIVE_ENHANCED_SOFT_POLYPHONY_ONSET=0.45`;
  - default `LIVE_ENHANCED_SOFT_POLYPHONY_FRAME=0.35`;
  - default `LIVE_ENHANCED_SOFT_POLYPHONY_MIN_VELOCITY=4`;
  - default `LIVE_ENHANCED_SOFT_POLYPHONY_MIN_DELTA=0.05`;
  - default `LIVE_ENHANCED_SOFT_POLYPHONY_LOOKBACK_SEC=0.08`.
- Carried rescue markers into live converted payloads:
  - single notes can include `decode_source`;
  - chords can include `decode_sources` and `soft_polyphony_rescued_notes`.
- Updated `backend/main.py` streaming continuity handling:
  - live stream observations now preserve `decode_source`;
  - `soft_polyphony_rescue` observations can pass the weak-birth-outside-attack gate, while harmonic-sustain suppression remains active.
- Validation:
  - `python -m py_compile backend/rhythm_training/train_enhanced_mel_transcriber.py backend/gpu_ops.py backend/detect_note.py backend/main.py` passed.
  - Synthetic decoder smoke test confirmed a 0.50 onset peak under a 0.75 primary threshold is not emitted without rescue, but is emitted with rescue when frame/velocity evidence exists.
  - Negative/control smoke tests confirmed low-frame candidates are not rescued and normal high-threshold primary onsets remain `decode_source='primary_onset'`.
  - Direct `ContinuousLiveStreamSession` smoke import was blocked in this local shell by missing `uvicorn`; the module still compiles.

## CHANGED: disable soft-polyphony rescue by default after live test, 2026-06-09

- User live test showed the rescue increased note count but produced musically wrong events:
  - incorrect onset timing;
  - incorrect displayed note values;
  - incorrect displayed durations.
- Updated `backend/detect_note.py` so live enhanced mode now defaults to:
  - `LIVE_ENHANCED_SOFT_POLYPHONY_RESCUE=0`.
- Kept the rescue code behind the env flag for future controlled diagnostics, but removed it from the default/main behavior.
- Current interpretation:
  - below-threshold per-key onset rescue is too weakly anchored when used as an independent event creator;
  - the better next experiment is likely an analysis/diagnostic pass that records inner-voice frame energy around expected onsets, not automatic event insertion.

## ADDED: inner-voice enhanced-model evidence diagnostic, 2026-06-09

- Added `backend/rhythm_training/diagnose_inner_voice_evidence.py`.
- Purpose:
  - inspect enhanced-mel onset/frame/velocity probabilities around missing quiet inner voices;
  - distinguish whether an inner note was decoded, present as frame energy with weak onset, present as weak onset only, mistimed, or absent/buried;
  - keep this as diagnostics only, with no live transcription behavior change.
- CLI examples:
  - `python backend/rhythm_training/diagnose_inner_voice_evidence.py --audio path/to/passage.wav`
  - `python backend/rhythm_training/diagnose_inner_voice_evidence.py --audio path/to/passage.wav --start-sec 12.5 --duration-sec 8 --expected 64@0.82,67@1.31`
- Outputs under `backend/rhythm_training/inner_voice_diagnostics/` by default:
  - `inner_voice_summary.json`;
  - `inner_voice_expected.csv`;
  - `inner_voice_near_misses.csv`;
  - `inner_voice_decoded_events.csv`.
- Validation:
  - `python -m py_compile backend/rhythm_training/diagnose_inner_voice_evidence.py` passed.
  - `python backend/rhythm_training/diagnose_inner_voice_evidence.py --help` passed.

## CHANGED: run inner-voice diagnostic on bad MAESTRO validation clips, 2026-06-09

- Extended `backend/rhythm_training/diagnose_inner_voice_evidence.py` with precomputed segment mode:
  - `--segment-id <id>`;
  - `--split validation|train|test`.
- Reason:
  - raw-audio mode was blocked in the local shell because `librosa` is not installed;
  - precomputed segment mode uses the same MAESTRO mel features already used for validation, so it can still inspect model probabilities for the exact clip.
- Scanned 256 hard validation segments at live-ish thresholds:
  - onset threshold `0.75`;
  - offset threshold `0.35`;
  - frame threshold `0.5`;
  - min velocity `8`.
- Worst useful segment found:
  - segment id `1270`;
  - MAESTRO audio: `backend/rhythm_training/maestro_midi/2008/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--3.wav`;
  - piece: Franz Liszt, `Transcendental Etude No. 12 "Chasse-neige"`;
  - segment start: `100.0s`;
  - event precision `0.978`;
  - event recall `0.604`;
  - event F1 `0.747`;
  - `116` false negatives;
  - `42` overlapping middle-register false negatives.
- Ran:
  - `python backend/rhythm_training/diagnose_inner_voice_evidence.py --segment-id 1270 --split validation --expected "74@0.6536458333333286,74@0.90234375,74@1.3033854166666572,62@1.3893229166666572,73@1.6953125,52@1.8125,55@2.0859375,52@2.088541666666657,73@2.1731770833333286,76@2.24609375,73@2.3125,55@2.3645833333333286" --output-dir inner_voice_diagnostics_sid1270 --device cuda --onset-threshold 0.75 --frame-threshold 0.5 --near-miss-onset 0.25 --max-candidates 250`
- Outputs:
  - `backend/rhythm_training/inner_voice_diagnostics_sid1270/inner_voice_summary.json`;
  - `backend/rhythm_training/inner_voice_diagnostics_sid1270/inner_voice_expected.csv`;
  - `backend/rhythm_training/inner_voice_diagnostics_sid1270/inner_voice_near_misses.csv`;
  - `backend/rhythm_training/inner_voice_diagnostics_sid1270/inner_voice_decoded_events.csv`.
- Findings on the 12 inspected overlapping middle-register false negatives:
  - `7` were `frame_present_weak_onset`;
  - `3` were `frame_present_no_onset`;
  - `2` were `absent_or_buried`.
- Interpretation:
  - many missed inner notes are visible to the frame/velocity heads and have onset peaks just below the live threshold;
  - the earlier automatic rescue failed because it created standalone events from weak candidates without sequence-level timing/duration context;
  - a better next experiment is not simple threshold lowering, but a sequence-aware candidate lattice/decoder that can include weak per-key candidates when they align with nearby decoded attacks or score-grid positions.

## ADDED: calibrated lattice-candidate experiment for inner voices, 2026-06-09

- Added `backend/rhythm_training/train_lattice_candidate_calibrator.py`.
- Purpose:
  - replace arbitrary weak-note scoring with a learned probability `P(real note | weak candidate context)`;
  - choose the acceptance threshold from a held-out precision/recall curve;
  - evaluate event-F1 impact before any live-pipeline wiring.
- Candidate features:
  - onset peak;
  - frame peak;
  - velocity peak/int;
  - local onset delta;
  - previous onset level;
  - distance to nearest primary onset cluster;
  - primary cluster size;
  - pitch distance to anchor cluster;
  - same-pitch recent primary flag;
  - harmonic-to-anchor flag;
  - active-frame-before value;
  - middle-register flag;
  - normalized pitch.
- Candidate acceptance:
  - train a `StandardScaler + LogisticRegression(class_weight='balanced')`;
  - choose threshold from validation precision/recall;
  - accepted weak events are snapped to the nearest primary onset cluster;
  - max additions per anchor and max anchor distance still bound the search space.
- Important implementation fixes:
  - relative manifest paths now resolve from `backend/rhythm_training`;
  - hard-case manifest `selection` order is preserved;
  - `EnhancedPrecomputedMelDataset` sorting is overridden in the experiment so train/eval split follows hard-case ranking order.
- Main held-out hard-ranked experiment:
  - command:
    - `python backend/rhythm_training/train_lattice_candidate_calibrator.py --segment-manifest mel_hard_case_manifest_validation_pedal_onset_v2.json --max-segments 256 --train-fraction 0.5 --output-dir lattice_candidate_calibrator_hard256_ordered_p75 --device cuda --target-precision 0.75 --progress-every 64`
  - trained on first 128 hard-ranked validation segments;
  - evaluated on next 128 hard-ranked validation segments;
  - candidate AP `0.676`;
  - candidate ROC-AUC `0.747`;
  - chosen threshold `0.7287`;
  - primary event F1 `0.9136`, precision `0.9878`, recall `0.8497`;
  - lattice event F1 `0.9225`, precision `0.9792`, recall `0.8720`;
  - delta F1 `+0.0089`;
  - delta recall `+0.0222`;
  - delta precision `-0.0086`;
  - additions: `496` total, `3.88` per segment average.
- Safer held-out operating point:
  - command:
    - `python backend/rhythm_training/train_lattice_candidate_calibrator.py --segment-manifest mel_hard_case_manifest_validation_pedal_onset_v2.json --max-segments 256 --train-fraction 0.5 --output-dir lattice_candidate_calibrator_hard256_ordered_p85 --device cuda --target-precision 0.85 --progress-every 128`
  - chosen threshold `0.7752`;
  - lattice event F1 `0.9196`;
  - delta F1 `+0.0061`;
  - delta recall `+0.0126`;
  - delta precision `-0.0026`;
  - additions: `247` total, `1.93` per segment average.
- Specific held-out bad clip `sid=1270` under the `p75` experiment:
  - primary F1 `0.7442`;
  - lattice F1 `0.7753`;
  - primary recall `0.6007`;
  - lattice recall `0.6416`;
  - primary precision `0.9778`;
  - lattice precision `0.9792`;
  - added `12` events.
- Interpretation:
  - the calibrated lattice approach is meaningfully better than arbitrary rescue;
  - it improves recall on hard inner-voice/dense clips while preserving high event precision;
  - still not ready for live default because score/duration quality of added events needs visual/audio inspection and downstream notation evaluation.

## CHANGED: lower live enhanced onset default after hard-case sweep, 2026-06-09

- Updated `backend/detect_note.py` live enhanced mode:
  - default `LIVE_ENHANCED_ONSET_BASE` changed from `0.75` to `0.60`.
- Reason:
  - hard-case sweep on the top 256 MAESTRO validation segments showed the old `0.75` threshold was over-prioritizing precision and missing many inner/dense notes.
- Sweep results:
  - threshold `0.75`: precision `0.9859`, recall `0.8519`, F1 `0.9140`;
  - threshold `0.65`: precision `0.9711`, recall `0.8842`, F1 `0.9257`;
  - threshold `0.60`: precision `0.9636`, recall `0.8928`, F1 `0.9268`.
- Expected tradeoff versus old default:
  - about `+1.28` event-F1 points;
  - about `+4.09` recall points;
  - about `-2.23` precision points.
- The env override remains available:
  - `LIVE_ENHANCED_ONSET_BASE=0.65` for a more conservative recall boost;
  - `LIVE_ENHANCED_ONSET_BASE=0.75` to restore the old behavior.
- Validation:
  - `python -m py_compile backend/detect_note.py` passed.

## CHANGED: voice-lane score durations in main live pipeline, 2026-06-09

- Added neutral pitch-lane voice assignment in `backend/live_rhythm.py`:
  - treble notes split into high/mid/low lanes;
  - bass notes split into low/mid/high lanes;
  - events now carry `voice_id`, `voice_index`, and `voice_assignment` when enabled.
- Changed the live score duration default:
  - `LIVE_SCORE_DURATION_POLICY=ioi_same_voice`;
  - previous same-hand behavior can be restored with `LIVE_SCORE_DURATION_POLICY=ioi_same_hand`;
  - voice assignment can be disabled with `LIVE_VOICE_ASSIGNMENT=off`.
- Applied same-voice IOI duration policy in both coarse live quantization and deferred refinement:
  - duration is capped by the next onset in the same voice lane;
  - repeated same-pitch onsets still cap the duration to avoid overlap.
- Updated `components/PianoSheetMusic.tsx` so score-facing retiming uses backend `voice_id` when present, falling back to hand IOI when not present.
- Updated chord handling so split-staff chord events prefer the staff-specific `voice_ids` entry before chord-level `voice_id`.
- Validation:
  - `python -m py_compile backend/live_rhythm.py` passed;
  - `npx.cmd tsc --noEmit` passed;
  - synthetic live-rhythm smoke test passed: a held high treble note at `0.00s` extended to the next high-lane onset at `1.00s`, while middle treble notes used middle-lane IOI.

## CHANGED: MusicXML voice lanes for overlapping same-staff notes, 2026-06-09

- Updated `components/PianoSheetMusic.tsx` so backend `voice_id` now maps to real MusicXML voice numbers:
  - treble lanes map to voices `1`, `3`, and `5`;
  - bass lanes map to voices `2`, `4`, and `6`;
  - notes, chords, rests, forwards, and generated ties preserve the mapped voice number.
- Changed same-staff chord merging to require the same `voiceNumber`:
  - simultaneous events in different lanes are no longer forced into one chord;
  - same-lane simultaneous notes can still merge as a chord.
- Changed measure clamping to track current end time per voice:
  - a held melody no longer pushes a later inner-voice event to the end of the melody;
  - different voices can overlap on the same staff.
- Added MusicXML `<backup>` emission when a later event starts before the current XML cursor:
  - this lets the score return to the correct onset for inner-voice notes after writing a held note.
- Validation:
  - `npx.cmd tsc --noEmit` passed;
  - `git diff --check -- components/PianoSheetMusic.tsx` passed apart from the repo's normal CRLF warning.

## CHANGED: wire calibrated lattice inner-voice rescue into the live path, 2026-06-09

- User problem:
  - Quiet inner voices played under held outer notes were "not recognized at all" in the live score.
- Root-cause diagnosis (live continuity gating, not the model):
  - Inner voices reach the enhanced-mel decoder fine (prior diagnostics: `frame_present_weak_onset`).
  - They were then dropped by two gates in `ContinuousLiveStreamSession._filter_stream_continuity` (`backend/main.py`):
    - `harmonic_sustain`: a weak note 7-36 semitones above a held/recent note with interval mod 12 in {0,4,7} and `confidence < 0.78` is suppressed as ring-out. Inner voices are frequently an octave/third/fifth above a held outer note.
    - `weak_birth_outside_attack`: a weak-attack note not within `STREAM_ATTACK_GROUP_RESCUE_SEC` of a fresh strong attack is suppressed unless `confidence >= STREAM_WEAK_BIRTH_HIGH_CONFIDENCE` (0.86). An inner voice under an already-attacked held note is by definition outside a fresh attack cluster.
  - This also explains why lowering `LIVE_ENHANCED_ONSET_BASE` 0.75->0.60 alone did not help: the decode-stage recall win was undone by these downstream gates.
- Decision (from the user): wire the offline-validated lattice calibrator into the live path rather than loosen the gates ad hoc.
- Implementation:
  - Added `backend/rhythm_training/export_lattice_calibrator_json.py`:
    - exports a trained `lattice_candidate_calibrator.pkl` to a dependency-free JSON (scaler mean/scale, logistic coef/intercept, calibrated threshold, candidate-generation knobs).
    - promoted `hard256_ordered_p75` (threshold `0.7287`) to `backend/rhythm_training/lattice_candidate_calibrator.json`.
  - Added `backend/lattice_candidate_decoder.py`:
    - pure-numpy runtime port of the trainer's candidate generation, 14-feature extraction, scoring, and anchor-snapping acceptance.
    - reproduces `Pipeline(StandardScaler, LogisticRegression).predict_proba` from the exported weights (verified max abs diff vs sklearn `3.5e-18`).
    - accepted events are marked `decode_source='lattice_calibrated'` with `lattice_probability`.
  - `backend/rhythm_training/train_enhanced_mel_transcriber.py`:
    - `decode_enhanced_note_events(...)` gained `lattice_rescue` / `lattice_model_path` params; after primary decode it appends calibrated lattice events and assigns them note-value class/name/confidence like primary events.
  - `backend/gpu_ops.py`:
    - `GpuEnhancedMelTranscriber.transcribe(...)` threads `lattice_rescue` / `lattice_model_path` and reports `lattice_rescued_events`.
  - `backend/detect_note.py` (live enhanced path in `analyze_audio_live_neural`):
    - default `LIVE_ENHANCED_LATTICE_RESCUE=1`; optional `LIVE_LATTICE_MODEL` override;
    - timing surfaces `neural_lattice_rescue` and `neural_lattice_rescued_events`.
  - `backend/main.py`:
    - `lattice_calibrated` observations now bypass BOTH the `harmonic_sustain` and `weak_birth_outside_attack` gates (calibrated rescues are deliberate, anchor-snapped below-threshold events).
- Known caveat / follow-up:
  - The shipped calibrator was trained with `primary_onset_threshold=0.75`, but the live default is now `0.60`, so at runtime the weak-candidate pool (0.25-0.60) is narrower and anchor features are computed against a denser primary set. Calibration may drift; consider retraining at `primary_onset_threshold=0.60` if live recall is short.
- Validation:
  - `python -m py_compile main.py gpu_ops.py detect_note.py lattice_candidate_decoder.py rhythm_training/train_enhanced_mel_transcriber.py rhythm_training/export_lattice_calibrator_json.py` passed.
  - Runtime-vs-trainer parity test (synthetic probs, forced acceptance): identical accepted events on midi/onset/offset/velocity.
  - End-to-end `decode_enhanced_note_events(..., lattice_rescue=True)`: emits `lattice_calibrated` events carrying note-value metadata alongside primary events.
  - Not yet run: continuous-stream benchmark (`backend/tune_continuous_stream_decoder.py`) and a real live recording check.

## FINDING: lattice rescue is inert at primary 0.60; gates are not the live culprit either, 2026-06-09

- Ran the continuous-stream benchmark (score surface) on the dense/inner-voice clips `clip_017`, `clip_031`.
- Lattice on vs off was identical: note F1 `0.8186`, recall `0.8919`, cluster F1 `0.0699` (only p95 inference jitter differed).
- Direct instrumentation on `clip_017` audio: the live enhanced path generated `23` weak candidates; max calibrated score `0.648`, median `0.371`; `0` candidates cleared the shipped threshold `0.7287`. So nothing was rescued.
- Root cause: the shipped calibrator was trained with `primary_onset_threshold=0.75`, but live runs at `0.60`. The `0.60` primary already captures the recoverable inner voices as primary events, leaving only a weak, low-precision sub-0.60 candidate pool.
- Retrained the calibrator at `primary_onset_threshold=0.60` (`lattice_candidate_calibrator_p60_p75`, target precision 0.75):
  - candidate AP `0.676 -> 0.406`; ROC-AUC `0.747 -> 0.651`.
  - chosen threshold `0.7343` with eval recall `0.0049` (4 additions on 128 eval segments).
  - Confirms: almost no high-precision lattice headroom remains at the live operating point.
- Gate-relaxation diagnostic on the same clips (temporary tuner candidates, since reverted):
  - baseline: recall `0.8919`, precision `0.7564`, F1 `0.8186`; suppressed `137` (weak_birth `71`, harmonic `14`).
  - gates fully off: recall `0.9054`, precision `0.7263`, F1 `0.8060`; suppressed `50`.
  - Removing `85` suppressions added only `4` true matched notes (`264 -> 268`) for `+20` predictions. The gates were suppressing almost entirely false onsets; they are not the inner-voice culprit on these clips.
- Interpretation:
  - On the benchmark clips, inner-voice note recall is already `~0.89`; neither the lattice rescue nor gate relaxation materially improves it.
  - The benchmark does not reproduce the user's "not recognized at all" report, so tuning against MAESTRO clips is optimizing against the wrong evidence.
  - Note also `cluster_f1 ~ 0.07` on these clips: pitches are found but onset-cluster/display structure is badly wrong, which can read as "inner voice missing" in the displayed score even when the note is matched.
- Decisions:
  - Flipped `LIVE_ENHANCED_LATTICE_RESCUE` default to `0` (inert + costs candidate generation at primary 0.60). Plumbing, model JSON, exporter, and runtime decoder are kept for retraining/diagnostics.
  - Next step is to capture a real recording where the user's inner voices fail and run `backend/rhythm_training/diagnose_inner_voice_evidence.py --audio <wav>` on it, rather than continue tuning against the benchmark.
- Artifacts:
  - `backend/benchmark_artifacts/lattice_eval_off/`, `.../lattice_eval_on/`, `.../lattice_eval_gatediag/`
  - `backend/rhythm_training/lattice_candidate_calibrator_p60_p75/`

## ROOT CAUSE: live RMS attack birth gate kills soft inner voices, 2026-06-09

- Reproduced the user's failure on real audio (`backend/rhythm_training/test_inner_voice.wav`, 34.6s) instead of MAESTRO.
- Added tooling:
  - `backend/transcribe_wav_to_midi.py`: high-accuracy offline full-file transcription with the enhanced model (whole-file inference, RTF 0.02x). Produced `backend/rhythm_training/test_inner_voice.mid` (237 notes, 196 in inner band 52-76, 156 onsets entering under >=2 held notes). The model detects inner voices well offline.
  - `backend/diff_offline_vs_live.py`: replays a WAV through `ContinuousLiveStreamSession` (real live path) and diffs emitted notes against a reference MIDI, with per-gate suppression totals and gate-override flags (`--relax-gates`, `--weak-birth-conf`, `--harmonic-max-conf`).
- Findings on `test_inner_voice.wav` (reference = offline model MIDI, 237 notes):
  - baseline live (weak_birth 0.86 / harmonic 0.78): recall vs model `0.46`, live score notes `189`, 128 missing (121 inner-band, 95 inner-voice-like). Suppression over replay: `weak_birth_outside_attack=1809`, `harmonic_sustain=1277`.
  - harmonic gate off only: unchanged at `0.46` (weak_birth absorbs it).
  - weak_birth gate off only: recall `0.80`, notes `325` (harmonic still removes 1332).
  - both gates off: recall `0.996` (236/237) BUT notes `189 -> 476` (precision ~`0.50`).
  - weak_birth threshold `0.78/0.70/0.60`: no change vs `0.86`. The missing inner voices have model confidence `<0.60`, the same region as decode noise, so a confidence threshold cannot separate them.
- Mechanism:
  - `weak_attack` is computed from audio RMS around the onset (`ContinuousLiveStreamSession._attack_metrics`: `attack_ratio = post_rms / pre_rms`). A quiet note added on top of loud held notes makes almost no RMS bump, so a soft inner voice is ALWAYS `weak_attack`.
  - With `confidence < STREAM_WEAK_BIRTH_HIGH_CONFIDENCE` and no fresh attack group, `weak_birth_outside_attack` suppresses it. The RMS attack heuristic physically cannot see a soft note under sustained louder notes, even though the model's onset head does.
  - The gates exist to clean noisier per-window live decode (vs full-context offline), so simply disabling them recovers recall but doubles the note count (precision collapse).
- Interpretation / next step:
  - The fix must replace the single-window RMS-attack + confidence birth gate with something that trusts the model's own onset/frame evidence and uses temporal persistence (multi-window agreement) to reject noise. This is an architectural change to the live birth/promotion path with a real precision tradeoff; approach to be decided with the user.
  - MAESTRO benchmark must not be used as the sole validation surface for this (it does not reproduce the failure); use `diff_offline_vs_live.py` against the offline reference, watching both recall AND live note count (precision).

## CHANGED: frame-evidence birth + persistence display gate for soft inner voices, 2026-06-09

- Fix for the root cause above (live RMS-attack birth gate deleting soft inner voices). Approach chosen by user: combine model frame evidence at birth with temporal persistence before display.
- Discriminator validation (gates-off replay of `test_inner_voice.wav`, real vs noise live notes):
  - observation count (persistence): real median `22`, noise median `1`.
  - decoded duration (frame sustain): real median `0.160s`, noise median `0.050s` (single-frame blips).
  - model confidence does NOT separate them (real `0.426` vs noise `0.415`), confirming confidence/RMS-attack is the wrong signal.
- Implemented in `backend/main.py` (`ContinuousLiveStreamSession`):
  - Added constants `STREAM_FRAME_EVIDENCE_SEC=0.08`, `STREAM_MIN_DISPLAY_OBSERVATIONS=3`, `STREAM_DISPLAY_FRAME_EVIDENCE_SEC=0.15`.
  - Birth (`_filter_stream_continuity`): a weak-attack observation whose decoded sustain `>= STREAM_FRAME_EVIDENCE_SEC` now bypasses BOTH the `harmonic_sustain` and `weak_birth_outside_attack` gates, so real soft notes are born instead of deleted.
  - Display (`_update_hypotheses`): candidate->active and candidate/active->committed promotion now also require `observations >= STREAM_MIN_DISPLAY_OBSERVATIONS` OR sustain `>= STREAM_DISPLAY_FRAME_EVIDENCE_SEC`. Single-window noise stays a hidden candidate and never reaches the score.
  - Aging (`_age_hypotheses`): stale `active` notes are now aged out like `candidate` (>1.0s unseen), so non-persistent notes promoted to active cannot leak onto the score surface.
- Validation on `test_inner_voice.wav` (`backend/diff_offline_vs_live.py`, reference = offline model MIDI, 237 notes):
  - baseline (broken): recall `0.46`, live score notes `189`.
  - both gates off: recall `0.996`, live score notes `476` (precision ~`0.50`).
  - combined fix: recall `0.992` (235/237, 0 inner-voice-like missing), live score notes `242` (precision ~`0.97`).
- No regression on MAESTRO continuous-stream benchmark (`clip_017`,`clip_031`, score surface):
  - note F1 `0.8186 -> 0.8993`; precision `0.7564 -> 0.9250`; recall `0.8919 -> 0.8750`; cluster F1 `0.0699 -> 0.2812`; duplicates/100 `0.0`.
  - 5-clip aggregate (`clip_001,002,010,017,031`): note F1 `0.9138`, recall `0.8781`, cluster F1 `0.5914`, p95 inference `26 ms`.
  - The persistence/frame-evidence display gate removes live-decode noise generally, so it improves precision and display-cluster structure as a side effect rather than regressing the normal case.
- New tooling kept: `backend/transcribe_wav_to_midi.py`, `backend/diff_offline_vs_live.py`.
- Validation: `python -m py_compile backend/main.py` passed.
- Full-manifest A/B (all 48 clips, continuous-stream score surface, fix-off via temporary `innervoice_fix_off` override since reverted):
  - note F1: `0.7840 -> 0.8921` (`+0.108`)
  - precision: `0.7705 -> 0.9667` (`+0.196`)
  - recall: `0.7980 -> 0.8281` (`+0.030`)
  - cluster F1: `0.4312 -> 0.6510` (`+0.220`)
  - duplicates/100: `0.2817 -> 0.0262` (`-0.255`)
  - Result files: `backend/benchmark_artifacts/innervoice_fix_fullmanifest/`, `backend/benchmark_artifacts/innervoice_fix_off_fullmanifest/`.
  - Both recall AND precision improved suite-wide; this is a general live-path quality win, not just an inner-voice repair.
- Note: `test_experiment.py` does NOT exercise this fix (it simulates the older `live_rhythm.LiveTranscriptionSession` chunk path, not `ContinuousLiveStreamSession`); use `tune_continuous_stream_decoder.py` to benchmark this change.

## 2026-06-10

### CHANGED: removed RMS-attack birth gating from the continuous live path (largest note+cluster F1 win on the streaming score surface)

- Goal:
  - Raise the F1 of the generated score (continuous-stream score surface) by `>= 0.1`.
- Baseline re-measured (48-clip manifest, `tune_continuous_stream_decoder.py`, score surface):
  - note F1 `0.8929`, precision `0.9660`, recall `0.8301`, cluster F1 `0.6519`, duplicates/100 `0.026`.
  - Run time per candidate is ~82s, so full-manifest sweeps are cheap now.
- Oracle decomposition of the cluster-F1 headroom (new `_tmp_f1plus_oracle.py` method: replace matched-note onsets with GT onsets, and/or append missed GT notes):
  - perfect timing alone: `0.652 -> 0.718` (`+0.066`)
  - perfect recall alone: `0.652 -> 0.855` (`+0.203`)
  - both: `0.941`
  - Conclusion: recall was the dominant lever; 757 of 4456 GT notes never reached the score surface vs only 130 false positives.
- Stage trace of missing notes (instrumented replay, `_tmp_f1plus_recall_trace.py`):
  - `clip_041` (worst recall, 0.439): of 66 GT notes, `35` were decoded by the model but suppressed by the continuity birth gates; only `2` were never decoded. The gated notes were a soft repeated-note alternation (MIDI 65/67) — weak RMS attack, duration `< 0.08s` so the frame-evidence bypass never fired, confidence `< 0.86`. Every window re-suppressed them as `weak_birth_outside_attack` / repeat gating.
  - Onset-threshold sweep confirmed decode is not the bottleneck: lowering `LIVE_ENHANCED_ONSET_BASE` 0.60 -> 0.55/0.50/0.45 was neutral-to-negative (recall +0.8pt max, matched precision loss).
- Change (`backend/main.py`):
  - Added `STREAM_RMS_BIRTH_GATES = False` master switch. With it off, `_filter_stream_continuity` keeps every decoded observation (attack-group registration still runs); the four RMS-attack birth gates (`same_pitch_boundary`, `implausible_repeat`, `harmonic_sustain`, `weak_birth_outside_attack`) no longer fire. Birth gating is delegated entirely to the 2026-06-09 persistence/frame-evidence display gate (`STREAM_MIN_DISPLAY_OBSERVATIONS=3` OR sustain `>= 0.15s`), which already separates real notes (median ~22 observations) from single-window decode noise (median 1).
  - Gate code is kept behind the switch for diagnostics.
- Full-manifest A/B (48 clips, score surface; `ship_legacy_gates` control reproduced the old baseline exactly, so the delta is fully attributable):
  - note F1: `0.8929 -> 0.9475` (`+0.0546`)
  - note recall: `0.8301 -> 0.9374` (`+0.1073`)
  - note precision: `0.9660 -> 0.9578` (`-0.0082`)
  - cluster F1: `0.6519 -> 0.7401` (`+0.0882`)
  - duplicates/100: `0.026 -> 0.183` (still negligible)
  - p95 inference unchanged (~25ms).
  - Identical aggregates were obtained in two independent full-manifest runs (`birth_gates_off` in `f1plus_sweep1`, `ship_ema` in `f1plus_ship`).
  - Clip-level: `clip_028` recall `0.550 -> ~0.96` (126/131 GT notes displayed, 0 gated, 4 never decoded).
- Real pedaled-audio regression check (`test_inner_voice.wav`, reference = offline model MIDI, the failure case the gates were originally built for):
  - gates on: 242 score notes, recall `0.992`
  - gates off: 250 score notes, recall `0.996`
  - No ring-out collapse (the pre-persistence collapse was 476 notes). The persistence display gate holds precision on real pedal audio.
- Tested and rejected in the same pass:
  - Median onset smoothing over an observation-onset history (replacing the 0.7/0.3 EMA): note F1 wash, cluster F1 regressed `0.7401 -> 0.7293` at manifest scale. Reverted; EMA stays.
  - `STREAM_MIN_DISPLAY_OBSERVATIONS=2`: cluster F1 `0.7178` (worse than 3).
  - `STREAM_DISPLAY_FRAME_EVIDENCE_SEC=0.10`: cluster F1 `0.7261` (worse).
  - `LIVE_ENHANCED_ONSET_BASE=0.55` on top of gates-off: cluster F1 `0.7196` (worse).
  - `STREAM_FRAME_EVIDENCE_SEC=0.05` birth relaxation (gates on): cluster F1 `0.6879` — strictly dominated by gates-off.
- Structure finding (remaining cluster-F1 gap, for future work):
  - On fast-run clips (`clip_017` cluster F1 ~0.10 with note F1 ~0.89) the failure is event slicing: model onset errors (~±25ms locally) against ~65ms inter-onset spacing make the 50ms single-linkage exact-cluster metric chain adjacent events; matched-note onset error is otherwise tiny (median ±5ms per clip). This is decode-timing-fundamental, not fixable at the hypothesis/display layer; the timing oracle caps it at `+0.066`.
- Harness change (`backend/tune_continuous_stream_decoder.py`):
  - Candidates can now carry env-var overrides (`env` field), used for `LIVE_ENHANCED_ONSET_BASE` sweeps; added `ship_default` / `ship_legacy_gates` and threshold/gate candidates.
- Diagnostic tooling added (kept):
  - `backend/_tmp_f1plus_oracle.py` (timing/recall oracle decomposition)
  - `backend/_tmp_f1plus_recall_trace.py` (per-stage missing-note attribution: displayed / hypothesis / gated / never-decoded)
  - `backend/_tmp_f1plus_cluster_diag.py`, `_tmp_f1plus_onset_err.py`, `_tmp_f1plus_agg.py`, `_tmp_f1plus_compare.py`, `_tmp_f1plus_report.py`, `_tmp_f1plus_pedal_check.py`
- Result files:
  - `backend/benchmark_artifacts/f1plus_baseline/` (old defaults)
  - `backend/benchmark_artifacts/f1plus_sweep1/` (threshold/gate/display sweep)
  - `backend/benchmark_artifacts/f1plus_ship/` (final A/B incl. legacy control)
- Validation:
  - `python -m py_compile backend/main.py backend/tune_continuous_stream_decoder.py` passed.
  - Final-state smoke run on `clip_017`+`clip_028` passed (note F1 `0.9226`).

## 2026-06-13

### CHANGED: display start_beat re-snapped to the final reported tempo (clean score-F1 win, note/cluster F1 unchanged)

- Goal:
  - Raise gold12 score F1 (priority) without regressing note F1 or cluster F1.
- Baseline reproduced exactly on the gold12 oracle harness (`test_experiment.py`, control arm):
  - note F1 `0.8566`, cluster F1 `0.6820`, score edit accuracy `0.2862`, score exact-token F1 `0.0297`, offset F1 `0.5230`.
- Diagnosis (instant iteration on the captured `score_payloads/*.json` via `tools/scorediff/run.js`, no neural re-run):
  - Cheap levers confirmed dead ends: confidence filtering (FP conf median 0.509 vs matched 0.575 — not separable), `LIVE_SCORE_DURATION_POLICY` switch (zero change).
  - The renderer derives printed note values **only** from each event's `start_beat` (clearing `note_value`/`note_divisions`/`duration_seconds` had zero effect; `start_beat` was the single lever).
  - The live `start_beat` is `time_at_idx(idx)/period` with `period = 60/current_bpm` **frozen at quantization time**. As the tempo tracker keeps refining BPM, early notes carry stale-tempo beats, so `start_beat` drifts from the raw onset beat at the final reported tempo the score renders at. Measured drift on `clip_012`: `start_beat` runs from `+0.13` to `+0.75` beat ahead of raw over the clip (grid tempo ~120-124 vs reported 112.5). On `clip_005` (sustained) grid≈reported, no drift — which is why removing the grid _helped_ staccato/dense clips but _hurt_ sustained ones. This is the `score_vs_midi_timing_divergence` cause, refined: it is stale-tempo `start_beat`, not duration estimation.
  - Sweep on payloads: re-expressing `start_beat = round(time_seconds/beat_dur * N)/N` at the **final** bpm — raw (N=∞) `0.3632` (8 up/3 down, sustained clips regress), 1/16 `0.3375`, 1/24 `0.3632`, **1/12 `0.3727` (10 up/1 down, sustained clips _improve_)**. 1/12 keeps jitter removal while removing drift; cleanest.
- Change (`backend/live_rhythm.py`):
  - Added `LIVE_DISPLAY_BEAT_SNAP_DIV` (default `12`, `0` restores legacy frozen-grid behaviour) and `_normalize_display_beats(events, bpm)`.
  - `_build_display_surface(notes, chords, bpm=0.0)` now re-snaps every emitted note/chord `start_beat` from its `time_seconds` at the passed BPM; `get_display_state()` passes `tempo_tracker.current_bpm`. Flows to both the app display payload and the benchmark score payload.
- Full-manifest gold12 A/B (`_fix_beatsnap.json` vs `gold12_oracle_results.json`):
  - Control score edit accuracy `0.2862 -> 0.3727` (`+0.0865`, +30%); score token F1 `0.0297 -> 0.0555` (+87%).
  - Control note F1 `0.8566 -> 0.8566` and cluster F1 `0.6820 -> 0.6820` — exactly unchanged (both key off `time_seconds`, not `start_beat`).
  - Treatment arm moved the same way: score edit `0.2707 -> 0.3549`, note/cluster flat.
  - Per-clip control score: 9 up, 2 down (`clip_010` 0.000->0.304, `clip_012` 0.000->0.223, `clip_001` +0.176; small regressions `clip_009` -0.075, `clip_003` -0.049).
- Validation:
  - `python -m py_compile backend/live_rhythm.py` passed.
  - Result file: `backend/benchmark_artifacts/gold12_reference_prep_20260612/_fix_beatsnap.json`.
- Remaining headroom (future work): note F1 (`0.857`) is FP-limited (~11% FP, not confidence-separable) and cluster F1 (`0.682`, avg_jaccard 0.92 = off by ~1 pitch) is downstream of note recall/precision — both need model-level work, no cheap post-processing lever found.

### RETROACTIVE LOG: "Inner voice fix" commit `a054794` (2026-06-09) — previously unlogged parts

Backfilling the shippable runtime parts of commit `a054794` that the dated entries above did not
cover. (The score-duration _instrumentation_ is logged under 2026-06-09 "diagnose enhanced
score-duration accuracy"; the continuous-stream decoder + RMS-birth-gate work under 2026-06-04/06-10;
the lattice calibrator under the lattice entries. The items below were not documented anywhere.)

- Inner-voice rescue, env-gated and shipped OFF by default (`backend/detect_note.py`,
  `backend/main.py`):
  - `LIVE_ENHANCED_SOFT_POLYPHONY_RESCUE` (default `0`): a second-pass decode that recovers soft
    near-simultaneous inner voices below the 0.60 primary onset, with its own onset/frame/velocity/
    delta/lookback thresholds (`LIVE_ENHANCED_SOFT_POLYPHONY_*`); tags recovered notes
    `soft_polyphony_rescue` and counts `soft_polyphony_rescued_notes` per chord.
  - `LIVE_ENHANCED_LATTICE_RESCUE` (default `0`) + `LIVE_LATTICE_MODEL`: lattice candidate decoder
    rescue path.
  - Both default OFF: per the inner-voice investigation neither beat the 0.60 primary at manifest
    scale (the primary already absorbs recoverable inner voices). Kept as infrastructure +
    diagnostics (`diagnose_inner_voice_evidence.py`, `lattice_candidate_decoder.py`,
    `train_lattice_candidate_calibrator.py`, `export_lattice_calibrator_json.py`).
- Voice assignment + multi-voice score rendering, shipped ON (`backend/live_rhythm.py`,
  `components/PianoSheetMusic.tsx`):
  - `LIVE_VOICE_ASSIGNMENT` (default `pitch_lanes`): `assign_voice_ids` splits each staff's events
    into pitch-lane voices (`_voice_id_from_pitch`, `_event_hand`, `_event_voice_id`) so inner voices
    print as separate voices instead of being packed into one per staff. Renderer extended (+202) to
    emit `<voice>`/`<backup>` per assigned voice.
- Score duration policy, shipped ON (`backend/live_rhythm.py`):
  - `LIVE_SCORE_DURATION_POLICY` (default `ioi_same_voice`): printed note length runs to the next
    onset in the same voice lane (`_next_policy_onset`,
    `_apply_score_duration_policy_to_quantized_window`), writing `score_duration_seconds`.
    NOTE: superseded for score positioning by the 2026-06-13 `start_beat` re-snap — the renderer
    derives printed note values from `start_beat` spacing, not from this policy's `note_value`.
- New offline tooling added (kept): `transcribe_wav_to_midi.py`, `diff_offline_vs_live.py`,
  `tune_decoder_settings.py`, `sweep_score_duration_policies.py`, `learn_score_duration_lookup.py`,
  `diagnose_enhanced_mel_transcriber.py`. App live-feedback UI work in `app/index.tsx` (+139).

## 2026-06-14

### ADDED: full-context teacher to live-window student distillation path

- Goal:
  - Implement the highest-priority model-level idea for the remaining note/cluster F1 headroom:
    train the enhanced transcriber under the same rolling-window conditions used by live inference,
    while distilling soft targets from a full-context teacher.
- Change (`backend/rhythm_training/train_enhanced_mel_transcriber.py`):
  - `EnhancedPrecomputedMelDataset` now supports live-style student crops via
    `--train-window-sec` and `--emit-window-sec`.
  - When `--live-window-distill-weight > 0`, each training sample also carries uncropped
    `teacher_features`, a `crop_start_frame`, and a `loss_mask`.
  - The normal supervised loss now respects `loss_mask`, so only the final emit region contributes
    gradients while earlier frames act as left context.
  - Added full-context teacher slicing: the teacher runs on the uncropped 10s feature segment, then
    its logits/probabilities are sliced onto the student's cropped window.
  - Added temperature-scaled live-window distillation terms for onset, offset, frame, sounding frame,
    pedal, velocity, and note-value heads. Defaults emphasize onset/frame/offset and leave pedal and
    note-value distillation off unless explicitly requested.
  - Teacher loading now uses the checkpoint config when available, so a teacher checkpoint can be
    loaded even if the student CLI args differ.
  - The existing teacher-preservation loss was made compatible with cropped student windows and remains
    enabled only for fine-tuning.
- Suggested first GPU run:
  - `python backend/rhythm_training/train_enhanced_mel_transcriber.py --train --finetune --init-from backend/rhythm_training/enhanced_mel_transcription.pt --teacher-from backend/rhythm_training/enhanced_mel_transcription.pt --model-path backend/rhythm_training/enhanced_mel_transcription_livewindow_distill.pt --train-window-sec 2.4 --emit-window-sec 0.6 --live-window-distill-weight 0.5 --train-segment-manifest backend/rhythm_training/mel_hard_case_manifest_train_pedal_onset_v2.json --validation-segment-manifest backend/rhythm_training/mel_hard_case_manifest_validation_pedal_onset_v2.json --save-best-on event_f1`
- Validation:
  - `python -m py_compile backend/rhythm_training/train_enhanced_mel_transcriber.py` passed.
  - In-memory smoke test passed for masked supervised loss plus full-context-teacher slicing/distillation
    on random tensors.

### CHANGED: score-duration lookup policy now reaches the actual validation path

- Goal:
  - Finish the easier score-duration follow-up by making the learned duration lookup policy usable in the
    enhanced-transcriber evaluator, instead of only in offline diagnostics.
- Change (`backend/rhythm_training/train_enhanced_mel_transcriber.py`):
  - `evaluate(...)` now accepts `score_duration_lookup_path` and loads the lookup table once per eval run.
  - Score matching now passes the loaded lookup into `match_score_events(...)`, so
    `--score-duration-policy lookup_ioi_head_sound` no longer silently falls back to plain
    `ioi_same_hand` during validation.
  - Checkpoint config / validation plumbing now preserve and forward `score_duration_lookup_path`.
  - CLI now exposes:
    - `--score-duration-lookup-path`.
- Diagnostics / artifact note:
  - Older lookup summaries already in the repo were stale for this path: some predated the
    `best_lookup.table` payload expected by the loader.
  - Regenerated a fresh smoke artifact with the current learner:
    - `backend/rhythm_training/score_duration_diagnostics/lookup_policy_summary_regen_smoke.json`
- Smoke validation:
  - `python -m py_compile backend/rhythm_training/train_enhanced_mel_transcriber.py` passed.
  - Regenerated smoke lookup with:
    - `python backend/rhythm_training/learn_score_duration_lookup.py --samples 8 --sampling leading --num-workers 0 ...`
  - Tiny held-out evaluator check on 8 validation samples, same checkpoint / decode path:
    - baseline `ioi_same_hand`: exact score-event F1 `0.376929`, duration accuracy `0.422972`
    - lookup `lookup_ioi_head_sound`: exact score-event F1 `0.402112`, duration accuracy `0.451231`
- Decision:
  - Keep the evaluator/CLI wiring.
  - Do not promote the lookup policy as the default live duration policy yet; first regenerate a real
    full-size lookup table and validate it on the saved checkpoint / broader held-out slice.

### VALIDATED: score-duration lookup beats `ioi_same_hand` on a broader held-out slice

- Follow-through on the decision above (regenerate full-size table + validate on the checkpoint /
  broader slice).
- Generated a real full-size, evaluator-compatible 3-part `ioi|head|sound` table:
  - Fit on the TRAIN manifest (`mel_hard_case_manifest_train_pedal_onset_v2.json`, 768 samples ->
    `99,479` matched rows) using `enhanced_mel_transcription_pedal_score_repair_latest.pt`, so it is
    disjoint from the validation slice.
  - `min_count` chosen by an internal even/odd train split (`min4` best, heldDurAcc `0.4811`); refit on
    all rows -> `396` entries.
  - Note: the auto-best policy in the existing full regen (`lookup_policy_summary_regen.json`) is the
    4-part `ioi_head_sound_conf`, which the evaluator loader rejects (it requires a 3-part key); hence
    the dedicated 3-part fit here.
  - Artifact: `backend/rhythm_training/score_duration_diagnostics/lookup_ioi_head_sound_heldout_full.json`.
- Held-out validation through the real `evaluate(...)` path (256 validation samples, spread; vs the
  prior 8-sample smoke), same checkpoint / decode path:
  - baseline `ioi_same_hand`: score F1 `0.4499`, precision `0.4559`, recall `0.4440`, durAcc `0.4976`.
  - `lookup_ioi_head_sound`: score F1 `0.5127`, precision `0.5196`, recall `0.5059`, durAcc `0.5671`.
  - Deltas: F1 `+0.0628`, precision `+0.0636`, recall `+0.0620`, duration accuracy `+0.0695`.
- Interpretation:
  - The gain reproduces the smoke-scale result (`0.377 -> 0.402` F1, `0.423 -> 0.451` durAcc) at ~32x the
    held-out sample budget, fit on a disjoint manifest, so it is not a small-sample / leakage artifact.
- Tooling (temporary diagnostic, kept): `backend/rhythm_training/_tmp_validate_score_duration_lookup.py`.
- Decision:
  - The deferred validation prerequisite is satisfied; the learned lookup is a genuine held-out win over
    the current default `ioi_same_hand`. Promoting `lookup_ioi_head_sound` (with this table) to the
    default live score-duration policy is now a go/no-go call for the user, not blocked on further
    evidence.

### PROMOTED: `lookup_ioi_head_sound` is now the default score-duration policy in the enhanced-transcriber evaluator

- Promoted the validated held-out table to a stable shipped path:
  - `backend/rhythm_training/score_duration_lookup.json` (copy of the held-out
    `score_duration_diagnostics/lookup_ioi_head_sound_heldout_full.json`; 396 entries, 3-part
    `ioi|head|sound`).
- `backend/rhythm_training/train_enhanced_mel_transcriber.py`:
  - `--score-duration-policy` now defaults to `lookup_ioi_head_sound` (was `ioi_same_hand`) and
    `--score-duration-lookup-path` defaults to the shipped table, but ONLY when the table file exists
    next to the script. If the file is absent both fall back to the legacy `ioi_same_hand` / `None`, so
    the change cannot crash an environment that lacks the artifact.
  - Verified resolved defaults: policy `lookup_ioi_head_sound`, lookup path -> shipped table.
  - `python -m py_compile` passed.
- SCOPE / IMPORTANT — this promotion is evaluator/benchmark-side only:
  - The live runtime score path (`backend/live_rhythm.py`, `LIVE_SCORE_DURATION_POLICY`) only implements
    `ioi_same_hand` / `ioi_same_voice` and has NO lookup-table support, so it is unchanged.
  - The 2026-06-13 `start_beat` re-snap means the live renderer derives printed note values from
    `start_beat` spacing, not from the duration policy's `note_value`; a live port of this lookup would
    therefore need its own port + gold12/continuous-stream validation and could be inert at render time.
    Left as a deliberate follow-up, not shipped blind.

### REVERTED the default flip — renderer proves note_value (hence the lookup) cannot reach the user

- Traced the actual render path before shipping the lookup to users. The score the user sees AND the
  gold12/scorediff reference are BOTH produced by `generateMusicXML` ->
  `generateMeasureXmls` in `components/PianoSheetMusic.tsx`.
- `generateMeasureXmls` (the loop at ~`1414`) UNCONDITIONALLY overwrites every event's printed duration:
  - `getIoiDurationSpec(estimateVoiceIoiBeats(group, staff, voiceId))` then
    `retimeXmlToDurationSpec(ev.xml, durationSpec)`.
  - `estimateVoiceIoiBeats = next.beatStart - current.beatStart` — the gap to the next onset in the SAME
    voice lane. `note_value` / `note_divisions` set earlier by `getDurationSpec` are discarded.
- Consequence: the printed note duration is purely the per-voice `start_beat` IOI. The score-duration
  policy's `note_value` output never reaches the rendered score. The lookup's win is confined to the
  offline `match_score_events` `duration_accuracy` metric, which is decoupled from the product (and from
  gold12, whose reference is rendered through the same IOI machine). This is the mechanistic proof behind
  the 2026-06-13 observation that `start_beat` was the single score lever.
- Action: reverted the `train_enhanced_mel_transcriber.py` default flip back to `ioi_same_hand` / `None`
  to keep the offline headline score-F1 comparable to history and to avoid biasing `--save-best-on
score_f1` checkpoint selection toward a product-decoupled metric. Kept as opt-in: the
  `lookup_ioi_head_sound` policy, the loader, `score_duration_lookup.json`, and a code comment pointing
  here.
- Real user-facing duration levers (for future work), in order: (1) `start_beat` accuracy — already the
  2026-06-13 win; (2) voice assignment (`voiceId`) quality, since it decides which IOI gap is measured;
  (3) note recall/precision, since missing/extra onsets change the gaps. The model's `note_value` head is
  NOT a user-facing lever as long as the renderer retimes by `start_beat` IOI.

## 2026-06-15

### CHANGED: default live voice assignment now collapses to one lane per hand

- `backend/live_rhythm.py`
  - `LIVE_VOICE_ASSIGNMENT` now defaults to `per_hand` instead of `pitch_lanes`.
  - `_voice_id_from_pitch(...)` now short-circuits to one notation lane per hand when
    `LIVE_VOICE_ASSIGNMENT=per_hand`, so per-voice IOI is intentionally identical to per-hand IOI.
  - The old multi-lane behavior is still available via `LIVE_VOICE_ASSIGNMENT=pitch_lanes`.

- Why this change was kept:
  - The score/reference oracle used in the gold12 evaluation is voiceless, so pitch-lane voice splitting
    adds structure the metric cannot reward.
  - On mostly-monophonic-per-hand material, pitch-bucket lanes also fragment melodic continuity whenever a
    line crosses a bucket boundary, which over-extends printed durations by measuring IOI to the next note
    in the wrong lane.
  - One lane per hand makes the live per-voice duration policy behave like the previously better
    per-hand policy while preserving env-revertibility for genuine polyphonic-within-hand material.

- Gold12 benchmark result:
  - old default `pitch_lanes`: score edit accuracy `28.62%`
  - new default `per_hand`: score edit accuracy `41.10%` (`+12.48 pts`)
  - streaming separation variants (`2`/`3` voices) were strictly worse here: `24.2%` / `22.4%`
  - `8 / 12` clips improved;
    biggest wins were `clip_003 +35`, `clip_005 +30`, `clip_008 +28`, `clip_006 +23`
  - `2` small/noisy 12-token clips regressed (`clip_001`, `clip_010`)
  - `2` fabricated-note clips stayed at `0` and were unaffected

- Validation:
  - Confirmed the real `assign_voice_ids` path reproduces the `41.10%` result exactly, not just the
    experimental harness.

- Tradeoff:
  - Genuine within-hand polyphony now serializes into one lane, so a held inner voice under a moving line
    can be visually truncated earlier than a true multi-voice rendering would allow.
  - This does not drop notes, so it does not conflict with the inner-voice recall work.
  - Given the current voiceless GT/reference, `per_hand` is the metric-optimal default until voice-aware
    ground truth exists.

- Tooling kept:
  - Added GPU-free A/B harness `backend/_voice_experiment/rederive_voices.py`.
  - The harness re-derives `voice_id` on dumped payloads and rescoring can be done through `scorediff`
    without rerunning neural inference.

### REJECTED (tested, negative): downbeat-phase metrical grid (idea #10)

- Hypothesis: `_normalize_display_beats` (`backend/live_rhythm.py:604`) anchors `start_beat` at absolute
  `t=0` (the arbitrary clip cut), so measure barlines do not start on a real downbeat; detecting the
  downbeat and re-anchoring the grid should improve the engraved score.
- Experiment: GPU-free harness `backend/_tmp_downbeat/rederive_downbeat.py` estimates a bar-phase `phi`
  from a metrical-accent template and re-derives `start_beat` for pred (and GT) payloads; scored with
  `tools/scorediff/run.js` against `oracle_gt_midi_payloads.json`.
- Result (gold12 mean `score_edit_accuracy`):
  - phase-0 control: `29.4%`
  - BOTH pred+GT re-anchored by a PERFECT ORACLE downbeat: `23.6%` (`-5.8`)
  - realistic estimate -> oracle reference: `24.9%`
- Even a perfect oracle downbeat REGRESSES the metric. Phase-0 is "wrong" about which beat is bar 1 but
  keeps onsets on clean integer beats AND is consistent pred-vs-reference; a fractional `phi` shift moves
  notes to messier within-bar positions and splits more notes across barlines. Whole-bar pickup-offset
  fix (to avoid negative-beat clamping) changed nothing. The detector itself only matched the oracle
  bar-phase `6/12`.
- Conclusion: not pursued. Only possible upside is cosmetic barline placement, but a 50%-accurate
  detector would mis-bar half the clips = net visual loss.

### REJECTED (tested, negative): acoustic onset snapping (idea #8); corrects an onset-error premise

- Note: parabolic sub-frame onset interpolation on the probability curve is ALREADY shipped
  (`backend/rhythm_training/train_ensemble.py` `decode_note_events`, ~L1439), used by the live
  `GpuMelBaselineTranscriber`. The untried variant tested here was snapping onsets to acoustic
  spectral-flux peaks.
- Experiment: `backend/_tmp_downbeat/acoustic_onset_snap.py` on the LIVE-path dumped payloads
  (0.6s chunks), matching pred notes to GT MIDI by pitch and comparing `|onset_pred - onset_gt|`
  before/after snapping to the nearest flux peak within +/-45ms.
- Result: live model onsets are ALREADY `~5.0ms` median error vs GT (mean signed `-0.3ms`, ~zero bias),
  including the fast-run subset (`4.9ms`). Snapping to flux peaks made it `5x` WORSE
  (median `5.0 -> 26.7ms`); of 323 snapped, `307` moved FARTHER from GT (flux envelope lags and blurs in
  runs). MAESTRO clips are Disklavier-aligned, so GT MIDI onsets are the true acoustic onsets — which is
  why flux is worse.
- Conclusion: not pursued. IMPORTANT premise correction: the "+/-25ms onset error vs 65ms IOI" framing in
  earlier notes is wrong for MATCHED notes — their onset timing is `~5ms`, so it is NOT the cluster-F1
  bottleneck. Fast-run cluster slicing must come from quantization/grid grouping or missing/extra cluster
  members, not raw onset accuracy.

### Next lever (queued, not yet built): SPRT evidence-accumulation display gate (idea #5)

- Target: the fabricated-onset precision killers visible in scorediff (e.g. clip_009=23, clip_006=18,
  clip_008=10 fabricated onsets). Replace the count-based persistence gate (`>=3 obs OR >=0.15s sustain`)
  with a sequential probability ratio test over per-window onset evidence so consistent-but-soft notes
  survive while flickering noise is rejected. To be A/B'd on the same gold12 payloads + the `_tmp_f1plus`
  oracle decomposition.

### BUILT + REJECTED (tested, negative on the continuous path): SPRT display gate (idea #5)

- Implemented an SPRT evidence-accumulation display gate in `ContinuousLiveStreamSession`
  (`backend/main.py`): per-hypothesis `llr` accumulates `STREAM_SPRT_OBS_LLR` (persistence credit) +
  `STREAM_SPRT_SLOPE*(conf - STREAM_SPRT_NEUTRAL_CONF)` per observation, plus a sustain term; promotion
  when `llr + sustain >= STREAM_SPRT_ACCEPT_LLR`. Env/attr-toggled `STREAM_SPRT_GATE` (default off);
  legacy gate exactly preserved when off. Shared `_stream_display_ready(...)` helper for both gates.
- A/B with `tune_continuous_stream_decoder.py` on the gold12 manifest (12 clips, the REAL
  `ContinuousLiveStreamSession` committed-notes path):
  | candidate | note F1 | precision | recall | cluster F1 |
  | --- | --- | --- | --- | --- |
  | baseline_current (legacy count gate) | **0.9386** | 0.9534 | 0.9243 | 0.7246 |
  | sprt_obs_heavy | 0.9192 | 0.9091 | 0.9296 | 0.6686 |
  | sprt_recall | 0.8635 | 0.8704 | 0.8566 | 0.5590 |
  | sprt_precision | 0.7498 | 0.9606 | 0.6149 | 0.5170 |
- No SPRT parameterization beats the legacy count gate. Persistence-dominant SPRT matches recall but
  loses precision (adds noise); confidence-leaning SPRT collapses recall (rejects real soft notes).
  Root cause = the documented wall: real soft inner voices and persistent decode noise BOTH sit at
  confidence < 0.60 and BOTH recur across windows, so neither count nor confidence separates them. The
  plain count gate is already near-optimal here.
- TWO IMPORTANT ARCHITECTURE FINDINGS (the premise of #5 was mis-aimed):
  1. The continuous `/live/stream` path (committed_notes; the gate's home) is ALREADY precision-strong
     (P=0.95). Its weak axes are recall (0.92) and cluster F1 (0.72), NOT precision.
  2. The "fabricated notes" in gold12 scorediff come from the OTHER live path: the legacy
     `/live/audio-chunk` + `/live/check-refinement` path (`LiveTranscriptionSession` in `live_rhythm.py`),
     which has NO persistence/observation gate at all. gold12 (`dump_app_payloads` ->
     `_analyze_uploaded_stream_chunk` + LiveTranscriptionSession) measures THIS path. The app
     (`app/index.tsx`) feeds the engraved score from `analysisResult`, written by BOTH the websocket
     continuous path AND `processRecordedChunk` (legacy path).

- RESOLVED 2026-06-15 (frontend trace): `USE_LIVE_STREAM_TRANSPORT = true` (app/index.tsx:35) =>
  production live sessions use ONLY the continuous websocket path (start L2127 / stop L2237); the legacy
  `processRecordedChunk` chunk-upload path is the `else` branch and is NOT exercised. The engraved score
  is `buildLiveStreamAnalysisResult` over `committed_notes + locked_notes + active_notes` (L564) — the
  GATED continuous surface. CONCLUSION: **production score = continuous path (already P=0.95); the gold12
  `dump_app_payloads` scorediff measures the LEGACY path the app does not use.** The reachable
  production levers are RECALL (0.92) and CLUSTER F1 (0.72), not precision. Score-level (MusicXML)
  benchmarking should be re-pointed at the continuous committed_notes payload, not the legacy dump.
- Tooling: SPRT candidates were A/B'd in `tune_continuous_stream_decoder.py` via the existing
  `_candidate(... env=...)` / `override_live_attrs` mechanism and the gold12 manifest.
- REVERTED 2026-06-15: the SPRT implementation was removed from `backend/main.py` and the SPRT candidates
  removed from `tune_continuous_stream_decoder.py` (dead end; not worth carrying env-toggled dead code in
  the production module). This change-log entry preserves the result so it is not re-attempted. The
  continuous-path harness + its `env`-override candidate sweep remain intact.

### REJECTED #1 onset-coincidence pre-clustering + #2 harmonic-prior display gate (both DEAD ENDS)

- Targets: cluster F1 (#1) and recall (#2) on the production continuous path, without latency cost.
- #1 implementation (continuous path, env-toggled `STREAM_ONSET_SNAP_SEC`, default 0=off): in
  `_build_update`, snap displayed notes (committed+locked+active) whose onsets fall within the window to
  a shared MEAN onset (single-linkage from each cluster's first onset), so a struck chord engraves as
  one event. Rationale: live decode spreads chord members; thought to split a chord across grid cells.
- #2 implementation (env-toggled `STREAM_HARMONIC_PRIOR_ENABLE`, default off): relax the persistence
  display gate (`>=3 obs OR >=0.15s sustain`) to `>=2 obs OR >=0.10s` for a candidate that forms a
  consonant interval (pc-distance in {0,3,4,5,7,8,9}) with a currently-SOUNDING already-gated
  (active/committed/locked) note within 1.2s. Support anchored only to real notes to protect precision.
  Principled successor to the dead SPRT idea (#5): harmonic context as the discriminator SPRT lacked.
- A/B on the gold12 manifest (12 clips, real `ContinuousLiveStreamSession` committed-notes surface;
  default harness stream config -> baseline note F1 0.9085 / R 0.8566 / P 0.9670 / cluster F1 0.6881):
  | candidate | note F1 | recall | precision | cluster F1 | dup/100 |
  | --- | --- | --- | --- | --- | --- |
  | baseline_current | **0.9085** | 0.8566 | 0.9670 | **0.6881** | 0.45 |
  | onset_snap_15ms | 0.9085 | 0.8566 | 0.9670 | 0.6830 (−0.0051) | 0.30 |
  | onset_snap_25ms | 0.9085 | 0.8566 | 0.9670 | 0.6697 (−0.0183) | 0.30 |
  | onset_snap_35ms | 0.9085 | 0.8566 | 0.9670 | 0.6697 (−0.0183) | 0.30 |
  | harmonic_prior_obs2 | 0.9086 | 0.8645 (+0.0080) | 0.9574 (−0.0097) | 0.6819 (−0.0062) | 0.44 |
  | harmonic_prior_obs2_sustain80 | 0.9073 | 0.8645 | 0.9545 | 0.6788 | 0.44 |
  | harmonic_prior_triads_only ({0,4,7}) | 0.9086 | 0.8645 | 0.9574 | 0.6819 | 0.44 |
  | snap15_harmonic_prior | 0.9086 | 0.8645 | 0.9574 | 0.6779 | 0.44 |
- #1 VERDICT — dead end. Zero effect on note F1/recall (matched-note onsets are already ~5ms accurate,
  so near-coincident notes already share a 50ms cluster); snapping to the MEAN only PERTURBS onsets and
  occasionally over-merges two correctly-separated pred clusters -> cluster F1 strictly WORSE at every
  window. Lone positive: fewer duplicates (0.45->0.30). NOTE: the *intended* benefit (chord co-assignment
  to one MusicXML beat cell) is a SCORE-RENDER effect this onset-cluster metric (50ms tol) does not
  measure; to test that hypothesis would require the scorediff renderer, which measures the LEGACY path,
  not production. So #1 is not merely negative — it is unmeasurable on the production harness.
- #2 VERDICT — dead end. Consistent +0.8% recall but −1.0% precision => note F1 dead flat (+0.0001),
  cluster F1 −0.6%. Restricting consonance to triads {0,4,7} was IDENTICAL to the full set {0,3,4,5,7,8,9}
  — so consonance does NOT separate real soft inner voices from noise on this corpus; recall and the
  admitted noise move 1:1. Same wall as SPRT (#5): real soft notes and persistent noise are both
  harmonically plausible AND both recur. The plain count gate remains near-optimal.
- REVERTED 2026-06-15: both features removed from `backend/main.py` (production module back to HEAD, zero
  diff) and both candidate sets removed from `tune_continuous_stream_decoder.py`. Kept: a `default=str`
  robustness fix on the candidate-log header (a `set` attr value had crashed `json.dumps`). The
  continuous-path harness + `env`-override sweep remain intact. Raw results:
  `backend/benchmark_artifacts/ideas12_snap_harmonic{,_b}/`.

### REJECTED #3 global probability calibration (== onset-threshold sweep; near-optimal already)

- A global monotonic (isotonic/temperature) calibration of the onset head, applied before a fixed
  threshold, is mathematically identical to moving that threshold. So #3-global was tested as the
  existing `LIVE_ENHANCED_ONSET_BASE` sweep on gold12 (12 clips, continuous committed-notes surface).
  | onset base | recall | precision | note F1 | cluster F1 |
  | --- | --- | --- | --- | --- |
  | 0.60 (baseline) | 0.8566 | 0.9670 | **0.9085** | 0.6881 |
  | 0.55 | +0.0027 | −0.0042 | −0.0004 | +0.0031 |
  | 0.50 | +0.0066 | −0.0195 | −0.0050 | −0.0173 |
  | 0.45 | +0.0080 | −0.0208 | −0.0049 | −0.0184 |
- The threshold already sits at its optimum; every step down trades recall for MORE precision than it
  gains. 0.55 is a near-wash (tiny recall + cluster gain at tiny precision cost) but within noise.
  Global calibration is therefore a dead end. The only #3 variant with a chance is PER-REGISTER
  calibration (separate curves for soft/high vs loud/low so relative ordering shifts) — a model-level
  change, not a knob. Raw: `backend/benchmark_artifacts/idea3_onset_sweep/`.

### FINDING: real GT chords are NOT simultaneous — within-chord onset spread is 24–62ms

- Motivated by "how far apart are the first and last note of a real chord?" Measured GT MIDI within-cluster
  onset spread (max−min onset) on the densest gold12 clips:
  - clip_009 (Debussy "Pour les accords", block chords, mean 5.65 notes/onset): spread median **24ms**,
    mean 25, p90 37, max 40. **91% of chords span >15ms**, 45% span >25ms. Tightest block chord = 13ms.
  - clip_006 / clip_008 (Gnomenreigen, fast figuration): at 50ms clustering median ~27–32ms, but at 80ms
    clustering the median jumps to **57–62ms** with **69–83% of groups spanning >50ms** — these are
    broken/rolled arpeggios the 50ms tolerance was artificially capping, not true chords.
- IMPLICATIONS:
  1. Confirms WHY #1 onset-snap failed: 91% of real chords are WIDER than 15ms; collapsing onsets to a
     point falsifies genuine human/pedal timing. No fixed snap window is correct (block chords ~25ms,
     figuration 50–79ms).
  2. Part of the "cluster F1 loss" is a MEASUREMENT ARTIFACT: the 50ms `ONSET_CLUSTER_TOLERANCE_SEC`
     inconsistently splits 25–79ms-wide GT groups between pred and reference, so even a perfect
     transcriber loses cluster points. Cluster F1 is not a pure model-quality signal.
  3. Next step for cluster F1 should target the METRIC/render grid (beat-quantized cluster eval), not the
     decoder. Tool: `compute_onset_cluster_metrics(... onset_tolerance_sec=...)` is parameterized;
     analysis script lived inline (load_midi_notes + slice_gt_notes + cluster_note_onsets).

### FINDING: cluster F1 is ~95% a metric artifact — production chord-grouping is near-perfect (pairwise co-onset F1 0.97–0.98)

- Followed up #2's "beat-quantized cluster eval" idea. First blocker: gold12 GT has NO usable beat grid —
  both source and excerpt MIDI report a single 120bpm default tempo (performance MIDI, timing in seconds),
  so a tempo/grid-quantized metric is **not buildable** from this benchmark. Pivoted to a tempo-free,
  jitter-robust metric.
- Quantified the artifact first (perfect-pitch ceiling). Took gold12 GT, added realistic Gaussian onset
  jitter (live onsets are ~5ms accurate, see [[downbeat_and_onset_snap_dead_ends]] #8), 40 trials/clip:
  | metric | self (GT-vs-GT) | +5ms jitter | +10ms | +15ms |
  | --- | --- | --- | --- | --- |
  | current single-linkage 50ms cluster F1 | 1.00 | 0.928 | 0.859 | 0.766 |
  | pairwise co-onset (W=50ms) F1 | 1.00 | **0.979** | **0.943** | **0.904** |
  A PERFECT-pitch transcriber with only 5ms jitter already loses ~7 cluster-F1 pts under the current
  metric (14 at 10ms), worst on dense clips (clip_006 0.74@5ms, clip_009 0.96→0.59@15ms). Pairwise
  recovers nearly all of it on the dense polyphony where it matters.
- THEN computed both metrics on the REAL production decoder output (idea3 `baseline_current.json`
  `score_notes`, the committed continuous surface) vs gold12 GT — no GPU rerun, same predictions:
  | | current cluster F1 | pairwise co-onset F1 | pair P | pair R |
  | --- | --- | --- | --- | --- |
  | unweighted mean | 0.752 | **0.983** | 0.978 | 0.990 |
  | GT-note-weighted | 0.639 | **0.971** | 0.960 | 0.984 |
  Per-clip, the worst "cluster" clips were never broken: clip_009 0.489→**1.000**, clip_006 0.538→0.974,
  clip_011 0.548→0.960, clip_004 0.622→0.989.
- CONCLUSION: the production decoder groups the notes it gets right into chords nearly perfectly. The
  headline cluster F1 (0.688) was conflating (a) missed notes = recall (real, 0.857) with (b) where
  single-linkage drew boundaries on 25–79ms-wide chords = artifact. ~95% of the apparent cluster-F1 gap
  is the metric. This closes the entire decoder-tuning line for cluster F1 (see also #1/#2/#3/#5 dead
  ends). The one remaining real accuracy axis is RECALL (model-level), not clustering.
- Why pairwise is robust: scores the "are notes A,B struck together" relation only over COMMONLY-MATCHED
  notes — no anchor, no transitive chaining, so it doesn't flip on where a wide chord's boundary lands;
  it also deliberately factors out recall, giving a clean grouping-quality signal. Weakness: noisy on
  very sparse clips (clip_001 9 notes) where few pairs carry the score — fine since clustering is
  irrelevant there.
- Tooling (temp, throwaway): `backend/_tmp_cluster_artifact_ceiling.py` (jitter ceiling + pairwise),
  `backend/_tmp_pairwise_on_real_preds.py` (pairwise on real preds). NOT yet wired into test_experiment.py.

## 2026-06-16

### FIXED: the 2026-06-12 RMS-birth-gate removal was NEVER ACTUALLY IN THE CODE — re-landed it (largest recall lever, recovered)

- DISCOVERY: the 2026-06-12 entry ("removed RMS-attack birth gating from the continuous live path",
  the single biggest documented recall win) described a change that was never committed. `git log -S
  STREAM_RMS_BIRTH_GATES -- main.py` returns NOTHING — the master switch never existed in any commit,
  and the four gates (`same_pitch_boundary` / `implausible_repeat` / `harmonic_sustain` /
  `weak_birth_outside_attack`) were still firing in `_filter_stream_continuity`. The shipped production
  recall was therefore the OLD gated number, not the 0.9374 the log claimed was live.
- CONFIRMED by reproduction before fixing: full-manifest `baseline_current` (HEAD `a054794`) =
  note F1 `0.8929`, recall `0.8301`, cluster F1 `0.6519` — exactly the pre-removal 2026-06-10 baseline.
- Re-implemented exactly as the 2026-06-12 entry specified (`backend/main.py`):
  - Added `STREAM_RMS_BIRTH_GATES = False` (default off).
  - In `_filter_stream_continuity`, after the `_match_hypothesis` short-circuit, when the switch is off
    every decoded observation is kept; attack-group registration still runs. Birth/noise rejection is
    delegated entirely to the persistence + frame-evidence display gate
    (`STREAM_MIN_DISPLAY_OBSERVATIONS=3` OR sustain `>= 0.15s`). Gate code kept behind the switch.
- Full-manifest A/B (48 clips, `tune_continuous_stream_decoder.py`, score surface; new `gates_on`
  candidate flips the switch back for a clean control). Artifacts:
  `backend/benchmark_artifacts/recall_distill_ab/gates_full/`.
  | metric | gates_on (legacy) | gates_off (NEW DEFAULT) | delta |
  | --- | --- | --- | --- |
  | note F1 | 0.8929 | **0.9475** | +0.0545 |
  | note recall | 0.8301 | **0.9374** | **+0.1073** |
  | note precision | 0.9660 | 0.9578 | -0.0082 |
  | cluster F1 | 0.6519 | 0.7401 | +0.0882 |
  | dup/100 | 0.026 | 0.183 | +0.157 (still negligible) |
  | p95 inference ms | 26.4 | 24.1 | -2.3 |
  These reproduce the 2026-06-12 gates-off numbers to the digit, so the validated win is now genuinely
  shipped. gold12 (12 clips) moved the same way: recall `0.8566 -> ~`, F1 `0.9085 -> ~` (see below).

### VALIDATED + ready to promote: the unlogged `livewindow_distill` checkpoint adds a small further recall gain

- DISCOVERY: `backend/rhythm_training/enhanced_mel_transcription_livewindow_distill_latest.pt` (Jun 13)
  already exists from the 2026-06-14 full-context-teacher distillation build, but was NEVER benchmarked
  or logged. Production loads `enhanced_mel_transcription.pt` by default
  (`ENHANCED_MEL_MODEL_PATH`/`LIVE_ENHANCED_MEL_MODEL_PATH` override; `gpu_ops.py`).
- A/B by pointing `ENHANCED_MEL_MODEL_PATH` at the distill checkpoint (separate process, fresh load).
  Stacked ON TOP of gates-off, full manifest. Artifacts:
  `backend/benchmark_artifacts/recall_distill_ab/distill_full_gatesoff/`.
  | metric | gates_off, default model | gates_off + distill model |
  | --- | --- | --- |
  | note F1 | 0.9475 | **0.9488** |
  | note recall | 0.9374 | **0.9434** (+0.0060) |
  | note precision | 0.9578 | 0.9542 |
  | cluster F1 | 0.7401 | 0.7418 |
  - The distill gain concentrates on the dense-polyphony clips with real recall headroom
    (gold12: clip_006 +2.7pt, clip_002 +1.1pt, clip_009 +0.8pt; clip_011 -1.6pt), net positive,
    p95 inference unchanged (~25ms). This is the FIRST validated model-level recall gain and confirms
    the distillation direction is correct.
- PROMOTED 2026-06-16 (user go/no-go = yes): backed up the old default to
  `backend/rhythm_training/enhanced_mel_transcription.pre_distill_backup.pt` (525MB) and copied the
  distill checkpoint over `backend/rhythm_training/enhanced_mel_transcription.pt` (now 214MB). Verified
  the plain default (no env override) loads the distill weights: gold12 recall `0.8566 -> 0.9309`,
  note F1 `0.9085 -> 0.9435` (gates-off + distill combined vs the originally-shipped gated state).
  Revert by restoring the backup. Bigger headroom is a longer/heavier distillation run (the Jun 13
  checkpoint was a first pass at `--live-window-distill-weight 0.5`).
- Harness: added a permanent `gates_on` control candidate to `tune_continuous_stream_decoder.py` so
  this regression cannot silently recur.
- Validation: `python -m py_compile backend/main.py` passed; baselines reproduced exactly twice.

### RE-TESTED + STILL REJECTED: soft-polyphony rescue (`LIVE_ENHANCED_SOFT_POLYPHONY_RESCUE=1`) post-gates-off

- The 2026-06-09 rejection of the soft inner-voice second-pass decode was made with the RMS birth gates
  ON. Re-A/B'd on the NEW default (gates-off + distill model) since recall is the target and the
  conditions changed. Continuous path confirmed to read the flag via `analyze_audio_live_neural`.
- Full-manifest A/B vs current default (artifacts `backend/benchmark_artifacts/recall_distill_ab/softpoly_full/`):
  | metric | default | +soft_poly | delta |
  | --- | --- | --- | --- |
  | recall | 0.9434 | 0.9511 | **+0.0076** |
  | precision | 0.9542 | 0.9251 | **-0.0290** |
  | note F1 | 0.9488 | 0.9379 | -0.0108 |
  | cluster F1 | 0.7418 | 0.7137 | -0.0281 |
  | dup/100 | 0.227 | 0.480 | +0.253 |
  | p95 inference ms | 24.6 | 32.7 | +8.1 |
- VERDICT: dead end even post-gates-off. +0.8pt recall costs −2.9pt precision (bad trade), F1/cluster
  both regress, dups double, +8ms latency. The rescued sub-0.60 candidates recur across windows like
  real notes, so the persistence display gate cannot separate them — same wall as SPRT/#2 harmonic
  prior. Default stays `LIVE_ENHANCED_SOFT_POLYPHONY_RESCUE=0`. Recall is now genuinely model-bound;
  the only lever left is a heavier distillation run.

## 2026-06-17

### VALIDATED: no-note-value model on the production continuous surface; onset 0.70 is the best current default

- Corrected the benchmark surface after a false start: production live app behavior should be judged with
  `backend/tune_continuous_stream_decoder.py` / `ContinuousLiveStreamSession`, not the legacy Gold-12
  app-payload path (`dump_app_payloads.py` / `test_experiment.py`). The old `0.9488` note F1 came from
  the full-manifest continuous surface in
  `backend/benchmark_artifacts/recall_distill_ab/distill_full_gatesoff/`.
- Evaluated the no-note-value checkpoint explicitly:
  `backend/rhythm_training/enhanced_mel_transcription_no_nv.pt` (43,303,193 params), gates off,
  soft-polyphony off, lattice off. Same-surface comparison at onset `0.60`:
  | model | note F1 | precision | recall | cluster F1 | p95 ms |
  | --- | --- | --- | --- | --- | --- |
  | livewindow-distill, with note-value head | 0.9488 | 0.9542 | 0.9434 | 0.7418 | 24.59 |
  | no-note-value checkpoint | 0.9490 | 0.9586 | 0.9396 | 0.7395 | 22.61 |
  Result: no-note-value is effectively tied/slightly higher on note F1, faster, and cleaner on precision;
  it trades away a small amount of recall/cluster.
- Post-processing sweep at fixed onset `0.60`:
  `backend/benchmark_artifacts/no_nv_continuous_postprocess_sweep_on060/`. Most stream-state knobs
  (rescue window/confidence, repeat suppression, weak-birth confidence, harmonic confidence, boundary
  window) were inert under gates-off and exactly matched baseline. Display/frame-evidence relaxations
  regressed slightly; `gates_on` regressed hard. Separate decode-side checks rejected both alternatives:
  soft-polyphony F1 `0.9395`, cluster `0.7161`, p95 `28.34ms`; lattice F1 `0.9480`, cluster `0.7377`,
  p95 `25.61ms`.
- Swept onset threshold on the same 48-clip continuous surface:
  `backend/benchmark_artifacts/no_nv_continuous_onset_sweep_full/`.
  | onset | note F1 | precision | recall | cluster F1 | dup/100 | p95 ms |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0.55 | 0.9468 | 0.9503 | 0.9434 | 0.7345 | 0.27 | 22.61 |
  | 0.60 | 0.9490 | 0.9586 | 0.9396 | 0.7395 | 0.16 | 22.63 |
  | 0.65 | 0.9522 | 0.9689 | 0.9360 | 0.7496 | 0.09 | 24.13 |
  | 0.70 | **0.9533** | 0.9778 | 0.9300 | **0.7534** | 0.05 | 24.17 |
  | 0.75 | 0.9525 | 0.9837 | 0.9232 | 0.7512 | 0.02 | 22.81 |
  | 0.80 | 0.9469 | 0.9902 | 0.9073 | 0.7459 | 0.02 | 31.02 |
  Accuracy winner is `LIVE_ENHANCED_ONSET_BASE=0.70`. `0.75` is a conservative near-tie with slightly
  lower latency, but worse F1/cluster/recall; `0.70` is still faster than the old distill run's p95.
- PROMOTED: set the default `LIVE_ENHANCED_ONSET_BASE` in `backend/detect_note.py` to `0.70`. Added
  explicit onset-sweep candidates in `backend/tune_continuous_stream_decoder.py`. `backend/gpu_ops.py`
  and `backend/modal_deploy.py` already prefer/package the no-note-value checkpoint.
- Validation: `env\Scripts\python.exe -m py_compile backend/detect_note.py backend/tune_continuous_stream_decoder.py`.

## 2026-06-18

### Display-latency sweep on the continuous /live/stream path

Goal: lower live display latency without losing accuracy. Latency on the shipped
WebSocket path (`ContinuousLiveStreamSession`) is policy, not compute — p95 model
inference is ~25 ms (RTF well under 1), so the wait is set by the hypothesis
state machine, not the GPU.

Harness: `backend/_latency_sweep.py` replays the gold12 suite
(`benchmark_artifacts/gold12_reference_prep_20260612/benchmark_manifest_gold12.json`)
through `ContinuousLiveStreamSession`, micro-averaged note metrics @50 ms (+strict
30 ms) on the `score` (stable) surface. Latency-to-visible ~= `max(trusted_delay,
STREAM_MIN_DISPLAY_OBSERVATIONS * inference_interval_ms)`.

Sweep 1 — trusted_delay x min_observations (interval 70 ms):

  | config | trusted | obs | ~latency | P | R | F1@50 |
  | --- | --- | --- | --- | --- | --- | --- |
  | baseline | 180 | 3 | 210 ms | 0.984 | 0.918 | 0.950 |
  | td120_obs3 | 120 | 3 | 210 ms | 0.984 | 0.918 | 0.950 |
  | obs2 | 180/120/80 | 2 | 140 ms | 0.982 | 0.922 | 0.951 |
  | obs1 | 80 | 1 | 80 ms | 0.575 | 0.944 | 0.715 |

Sweep 2 — inference_interval at obs2 (trusted held low so obs gate binds):

  | interval | ~latency | P | R | F1@50 | infer p95 | hop RTF |
  | --- | --- | --- | --- | --- | --- | --- |
  | 100 ms | 200 ms | 0.986 | 0.923 | 0.953 | 25 ms | 0.25 |
  | 70 ms | 140 ms | 0.982 | 0.922 | 0.951 | 25 ms | 0.36 |
  | 50 ms | 100 ms | 0.982 | 0.922 | 0.951 | 25 ms | 0.50 |
  | 40 ms | 80 ms | 0.863 | 0.932 | 0.897 | 25 ms | 0.63 |

Findings:
- `trusted_delay` alone is INERT — the observation gate binds; tuning it in
  isolation does nothing. Diagnostic `td120_obs3 == baseline` proves it.
- `obs 3 -> 2` is free (F1 0.950 -> 0.951, +3 matched), cuts 210 -> 140 ms.
- `obs 1` is a precision cliff (P 0.98 -> 0.575; predicted 707 -> 1237) — the
  corroboration gate is what holds precision; do not run obs1 without a
  confidence-adaptive promotion path.
- `interval 50 ms` at obs2 is free (identical F1 0.951), cuts 140 -> 100 ms.
- `interval 40 ms` TANKS (P 0.98 -> 0.86, F1 -5.4 pt). The knee is the acoustic
  floor: 40 ms hop ~= A0 fundamental period (~36 ms), so two observations 40 ms
  apart are too correlated to corroborate. Latency floor is acoustic, not compute.

Net: 210 ms -> 100 ms (52%) at flat F1 0.951, below the 128 ms literature low end.

### PROMOTED: ship obs2 + 50 ms hop

- `backend/main.py`: `STREAM_MIN_DISPLAY_OBSERVATIONS 3 -> 2`; session class
  defaults `inference_interval_sec 0.10 -> 0.05`, `trusted_delay_sec 0.18 -> 0.10`;
  factory fallbacks `100 -> 50 ms` / `180 -> 100 ms`.
- `app/index.tsx`: `LIVE_STREAM_INFERENCE_INTERVAL_MS 70 -> 50`,
  `LIVE_STREAM_TRUSTED_DELAY_MS 180 -> 100`.
- Deploy split: `obs=2` is backend (ships via Modal). The interval/trusted values
  are sent by the frontend at session open, so they ship via app rebuild, not
  Modal; backend defaults were aligned to 50/100 so a Modal-only deploy is still
  correct if the client omits them.
- Caveat: numbers are gold12 DISPLAY-note F1 (12 clips, small N), not raw-onset F1
  comparable to the literature 95%. Robust finding is the frontier shape, not the
  absolute 0.95. The next sub-100 ms lever is confidence-adaptive obs1.

## 2026-06-21

### Lowered live onset threshold 0.75 -> 0.70 for recall

- `backend/detect_note.py`: `LIVE_ENHANCED_ONSET_BASE` default `0.75 -> 0.70` (env
  default string ~L8789 + fallback ~L8791), in `analyze_audio_live_neural` — the
  live continuous path. The offline `analyze_audio_neural` threshold
  (`ENHANCED_MEL_ONSET_THRESHOLD`, ~L7923) is unchanged at 0.75.
- Rationale: 0.75 sat ABOVE the F1 peak. Per the current-model threshold sweep,
  0.70 is strictly better than 0.75 on BOTH recall (0.9232 -> 0.9300, +0.68pt) and
  note F1 (0.9525 -> 0.9533, +0.08pt), spending only ~0.6pt of surplus precision
  (0.9837 -> 0.9778). Dup/100 rises 0.02 -> 0.05 (still negligible).
- This is an operating-point move along the known curve, NOT a model change; the
  promoted distill checkpoint is untouched. Fully reversible via the env var.
- Context: chosen as the cheap recall lever while the paper is being written;
  large/model-level recall work (heavier distill, octave completion) deliberately
  deferred until after submission to avoid invalidating reported numbers. Genuine
  misses remain model-bound (never decoded into the lattice) per the 2026-06-19
  operating-point analysis; threshold cannot recover those.

## 2026-06-24

### Removed the adaptive (loudness-based) live onset selector — inert on ablation

- Motivation: audit of the live-decode heuristic parameters. The per-chunk
  loudness-based onset selector (`_select_live_neural_onset_threshold`) carried
  ~8 hand-set magic numbers (`rms 0.024/0.060/0.110`, `peak 0.45`, `crest 2.30`,
  `+/-0.04/0.02`, floor `0.30`, caps `0.46/0.95`) and was tagged an experiment
  (`adaptive_onset_loudness_v1`) but had never been A/B'd against a fixed base.

- Ablation (48-clip live replay, `live_benchmark_replay_auto_v2`, via
  `test_experiment.py` control[adaptive OFF/fixed] vs treatment[adaptive ON]):
  - @ base 0.70: dF1 **-0.0002**, dRecall +0.0011, dPrec -0.0016
  - @ base 0.75: dF1 **-0.0004**, dRecall -0.0004, dPrec -0.0002
  - Verdict: statistically inert (moves nothing past the 4th decimal). The
    selector is pure complexity with no measurable effect on the frontier.

- Change (`backend/detect_note.py`):
  - Deleted `_select_live_neural_onset_threshold` and all 3 call sites in
    `analyze_audio_live_neural` (enhanced_mel / mel_baseline / custom branches).
    The live path now uses the fixed base threshold directly:
    `LIVE_ENHANCED_ONSET_BASE` (0.70), `LIVE_ONSET_BASE` (0.46), custom 0.33.
  - Dropped the now-dead `neural_chunk_rms / _peak / _crest_factor` timings and
    the loudness `profile`; `live_onset_threshold_profile` is now the constant
    `fixed_onset`, `live_onset_threshold_experiment` = `fixed_onset_baseline`.
  - `adaptive_onset_threshold` kept as a DEPRECATED no-op param at the function
    boundary (docstring-flagged) to avoid a ~20-site refactor across the FastAPI
    endpoints (`main.py`) and the benchmark harness. Full param removal is a
    follow-up; the boolean now has no effect.

- Harness fixups:
  - `backend/test_experiment.py`: `force_live_onset_threshold` no longer
    monkeypatches the deleted selector; it now overrides the base-threshold env
    vars (`LIVE_ENHANCED_ONSET_BASE`, `LIVE_ONSET_BASE`) within the context.
    Removed the module-level `_ORIGINAL_LIVE_THRESHOLD_SELECTOR` reference that
    crashed import.
  - `backend/tune_decoder_settings.py`: added a reproducible `enhanced_onset_075`
    candidate (pins live base to 0.75) used for this audit.

- Verification: post-removal smoke (3 clips) now shows control == treatment
  exactly (0.9193 @0.70, 0.9362 @0.75), confirming the flag is inert and that the
  fixed-base numbers are unchanged vs the pre-removal fixed arm (behavior-
  preserving). Both files byte-compile. Latency unaffected (one fewer per-chunk
  numpy pass). Artifacts: `benchmark_artifacts/adaptive_audit_full/`.

- NOT changed: live base stays 0.70 (the deliberate recall lever from
  2026-06-21). The same 48-clip sweep shows fixed 0.75 would trade -1.0pt recall
  for +1.75pt precision (+0.33pt net F1) — a precision-leaning operating-point
  move, left as an explicit decision, not applied.
