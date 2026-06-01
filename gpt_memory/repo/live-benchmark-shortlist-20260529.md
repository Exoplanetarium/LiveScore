# Live Benchmark Priority Shortlist

Source benchmark:

- `backend/live_benchmark_replay_auto_v2_results_20260529_chordnotes.json`

Purpose:

- Rank the current live benchmark clips into the most useful next debugging sets.
- Prioritize clips that are bad for distinct reasons instead of averaging everything together.
- Replace the now-stale pre-chord-aware shortlist, which over-penalized polyphonic clips by ignoring pitches emitted inside `chords`.

## First Attack Set: Boundary Recall

These are the best clips to debug first if the goal is to reduce `no_control_coarse_candidate` and improve boundary recall.

1. `clip_026` - Concert Etude "Waldesrauschen", start `12s`
   - control F1 `0.7838`
   - boundary miss rate `0.2195`
   - missed boundary notes `9`
   - `no_control_coarse_candidate` tags `9`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`, `note_value_offset_failure`

2. `clip_017` - Pour le piano (Complete), start `51s`
   - control F1 `0.7980`
   - boundary miss rate `0.1343`
   - missed boundary notes `9`
   - `no_control_coarse_candidate` tags `9`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

3. `clip_006` - Pour le piano (Complete), start `57s`
   - control F1 `0.8476`
   - boundary miss rate `0.1270`
   - missed boundary notes `8`
   - `no_control_coarse_candidate` tags `8`
   - buckets: `runtime_only_win`, `boundary_miss_failure`, `high_revision_slow_stabilization`

4. `clip_027` - Sonata in D Major, K. 96 L. 465, start `9s`
   - control F1 `0.8276`
   - boundary miss rate `0.1628`
   - missed boundary notes `7`
   - `no_control_coarse_candidate` tags `7`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

5. `clip_016` - Concert Etude "Gnomenreigen", start `24s`
   - control F1 `0.8373`
   - boundary miss rate `0.1522`
   - missed boundary notes `7`
   - `no_control_coarse_candidate` tags `7`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

6. `clip_019` - Concert Etude No. 2, "Gnomenreigen", S. 145/2, start `63s`
   - control F1 `0.8105`
   - boundary miss rate `0.1273`
   - missed boundary notes `7`
   - `no_control_coarse_candidate` tags `6`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

Why this set:

- These are now the clips with the highest remaining true `no_control_coarse_candidate` counts after chord-member pitches are counted correctly at note level.
- They still combine meaningful boundary miss counts with enough surviving recall errors to make seam instrumentation worthwhile.
- This set is now the cleanest place to inspect raw live-neural misses right after chunk boundaries without the old note-vs-chord accounting distortion.

## Second Attack Set: General Low-F1 Failures

These clips are the lowest-F1 cases overall and are good for checking whether fixes generalize beyond the strict boundary subset.

1. `clip_014` - Sonata No. 9, start `6s` - control F1 `0.6102`
2. `clip_005` - Hungarian Rhapsody No. 9, start `48s` - control F1 `0.6957`
3. `clip_031` - Hungarian Rhapsody No. 9, start `18s` - control F1 `0.7054`
4. `clip_009` - Sonata in A Major, K. 208, start `3s` - control F1 `0.7143`
5. `clip_035` - Estampes (Complete), start `81s` - control F1 `0.7241`
6. `clip_012` - Entragete, Op. 63, start `75s` - control F1 `0.7395`
7. `clip_024` - Hungarian Rhapsody No. 9 in E-flat Major, S. 244, start `21s` - control F1 `0.7594`
8. `clip_041` - Sonata No. 9, Op. 68, start `72s` - control F1 `0.7600`
9. `clip_044` - Pagodas from Estampes No. 1, start `18s` - control F1 `0.7603`
10. `clip_001` - Sonata in D Minor, K. 9, start `42s` - control F1 `0.7619`

Why this set:

- These are the true lowest-F1 clips after the benchmark stopped dropping chord-contained pitches.
- Several are no longer boundary-first failures, which makes them useful for checking whether future fixes are actually improving broader rhythm, timing, or notation quality instead of only seam recall.

## Third Attack Set: Stabilization / Revision Churn

These clips are best for improving live notation stability after first display.

1. `clip_018` - Sonata in D Major, K96, start `9s`
   - stabilize p95 `4970.1 ms`
   - visible p95 `564.4 ms`
   - avg revisions `0.93`

2. `clip_005` - Hungarian Rhapsody No. 9, start `48s`
   - stabilize p95 `4718.4 ms`
   - visible p95 `586.0 ms`
   - avg revisions `2.88`

3. `clip_014` - Sonata No. 9, start `6s`
   - stabilize p95 `4710.4 ms`
   - visible p95 `597.8 ms`
   - avg revisions `1.50`

4. `clip_042` - Sonata in D Major, K. 96 L. 465, start `63s`
   - stabilize p95 `4119.0 ms`
   - visible p95 `748.0 ms`
   - avg revisions `2.80`

5. `clip_043` - Sonata in D Minor, K. 9 L. 413, start `12s`
   - stabilize p95 `4071.0 ms`
   - visible p95 `636.2 ms`
   - avg revisions `1.57`

Why this set:

- These clips are not mainly about first visibility.
- They are where the display is fast enough to appear but slow to settle.
- Use these after recall work if the next target is notation churn or revision count.

## Recommended Order

1. Start with the updated boundary recall set.
2. Instrument raw live-neural emissions on these clips within the first `100 ms` after each chunk boundary.
3. For each miss, distinguish whether the pitch is absent in raw neural output, present in raw note events but lost during conversion, or present in converted note/chord payloads but not retained by the live path.
4. After boundary recall improves, use the low-F1 set to check that fixes generalize beyond seam misses.
5. Only after recall quality is acceptable, move to the stabilization set.

## Current Headline Metrics

- Control average F1: `0.8320`
- Control average boundary miss rate: `0.0782`
- Control boundary matched-note sum: `1251`
- Control duplicate rate: `5.9175` per 100 notes
- Treatment average F1: `0.8282`
- Treatment average boundary miss rate: `0.0782`
- Treatment boundary matched-note sum: `1251`
- Treatment duplicate rate: `5.8497` per 100 notes

Notes:

- The duplicate-rate figures are now based on chord-expanded note-level predictions, so they are not directly comparable to the older note-only shortlist.
- The headline shift from the earlier shortlist is mostly a benchmark-accounting correction, not a sudden raw-model leap.
