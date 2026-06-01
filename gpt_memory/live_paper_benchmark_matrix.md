# Live Paper Benchmark Matrix

Updated: 2026-05-29

## Purpose

This note maps LiveScore's current benchmark plan to the live / streaming piano
papers already listed in `gpt_memory/live_transcription_prior_art_and_bar.md`.

The goal is to answer one practical question:

- which published live-paper claims can the current LiveScore metrics compare to
  directly
- which ones are only partially comparable
- which ones need new evaluation slices before the comparison is fair

## Status Labels

- `Direct`: same target and metric family are already present in the current
  benchmark plan.
- `Partial`: same general axis, but the paper and LiveScore use different
  tolerances, latency definitions, or task framing.
- `No`: the paper evaluates something that the current benchmark plan does not
  currently measure.

## Current LiveScore Metric Inventory

These are the main metrics already defined in `gpt_memory/live_accuracy_benchmark_plan.txt`.

| Metric family                                  | Current status | Notes                                                                                          |
| ---------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------- |
| Note precision / recall / F1                   | Yes            | Standard note-level accuracy with 50 ms onset tolerance and exact pitch match                  |
| Offset-aware note F1                           | Yes            | Important for sustained-note and chopped-note problems                                         |
| Tempo MAE / tempo octave error                 | Yes            | Useful for notation quality, but not standard in the live AMT papers reviewed here             |
| Rhythm note-value accuracy                     | Yes            | Useful for a notation paper, but mostly unmatched by the live event-transcription papers       |
| Duplicate-rate                                 | Yes            | Strong streaming systems metric; not commonly reported in the live papers reviewed here        |
| Boundary miss rate                             | Yes            | Directly relevant to chunked streaming; not commonly reported in the live papers reviewed here |
| Stabilization latency                          | Yes            | Strong live-UI metric; not commonly reported in the live papers reviewed here                  |
| Time-to-visible                                | Yes            | Strong product-facing live metric; not commonly reported in the live papers reviewed here      |
| P50 / P95 chunk latency                        | Yes            | End-to-end runtime metric                                                                      |
| Real-time factor                               | Yes            | Useful for live systems evaluation                                                             |
| Velocity metrics                               | No             | Missing today                                                                                  |
| Sustain pedal metrics                          | No             | Missing today                                                                                  |
| Strict onset-tolerance F1 at 10 / 20 / 30 ms   | No             | Missing today; needed for minimum-latency comparisons                                          |
| Algorithmic lookahead / future-context latency | No             | Missing today; current latency focus is end-to-end runtime                                     |
| Parameter count / GFLOPs                       | No             | Missing today; some real-time papers report efficiency this way                                |

## Paper-By-Paper Matrix

### Fernandez 2023

Paper:

- Onsets and Velocities: Affordable Real-Time Piano Transcription Using
  Convolutional Neural Networks

What the paper emphasizes:

- onset detection
- onset + velocity prediction
- modest model size
- real-time capability claim

What LiveScore can compare now:

- `Direct`: note-level onset-focused accuracy, if reported in the same MAESTRO /
  mir_eval-style setup
- `Partial`: runtime / real-time claim, because the paper framing available here
  is not aligned to LiveScore's P95 end-to-end chunk latency reporting
- `No`: velocity metrics
- `No`: note-with-offset metrics
- `No`: pedal metrics
- `No`: duplicate-rate, boundary miss, stabilization latency
- `No`: notation / rhythm metrics

Bottom line:

- This is only a partial direct comparison with the current stack.
- Today, LiveScore can compare against its onset-oriented note accuracy story.
- It cannot yet compare against the velocity part of the claim.

What to add for a stronger comparison:

1. Velocity-aware note metric.
2. A cleaner published latency protocol, ideally separating algorithmic delay
   from measured wall-clock runtime.

### Kwon et al. 2024

Paper:

- Towards Efficient and Real-Time Piano Transcription Using Neural
  Autoregressive Models

What the paper emphasizes:

- note-level F1
- note-with-offset F1
- note-with-offset-and-velocity F1
- note duration accuracy
- parameter count
- model latency sweeps from 128 ms to 320 ms

What LiveScore can compare now:

- `Direct`: note precision / recall / F1
- `Direct`: offset-aware note F1
- `Partial`: latency, because Kwon reports model / architectural latency while
  LiveScore currently emphasizes end-to-end P95 chunk latency
- `No`: note-with-velocity metrics
- `No`: note duration accuracy as defined in that paper
- `No`: parameter count / GFLOPs reporting in the benchmark stack
- `No`: duplicate-rate, boundary miss, stabilization latency
- `No`: notation / rhythm metrics as a direct apples-to-apples comparison

Bottom line:

- This is the strongest current live-paper comparison that LiveScore can already
  make.
- You can compare directly on standard note quality and note-with-offset
  quality.
- The latency comparison is only partial unless you add algorithmic-latency
  reporting.

What to add for a stronger comparison:

1. Velocity-aware evaluation.
2. Parameter count and possibly GFLOPs.
3. An explicit algorithmic latency figure alongside runtime P95.

### Wei et al. 2025

Paper:

- Streaming Piano Transcription Based on Consistent Onset and Offset Decoding
  with Sustain Pedal Detection

What the paper emphasizes:

- streaming audio-to-MIDI transcription
- note-level onset F1
- note-level onset + duration F1
- frame-level metrics
- sustain pedal handling
- reported streaming latency of 380 ms based on future-frame dependency

What LiveScore can compare now:

- `Direct`: note precision / recall / F1
- `Direct`: offset-aware note F1
- `Partial`: latency, because Wei's paper reports streaming latency in terms of
  model future context, while LiveScore currently reports runtime latency
- `No`: frame-level metrics
- `No`: sustain pedal metrics
- `No`: duplicate-rate, boundary miss, stabilization latency
- `No`: notation / rhythm metrics as a direct apples-to-apples comparison

Bottom line:

- This is also a strong current comparison target.
- LiveScore can compare directly on standard note-level and note-with-offset
  accuracy.
- Latency is only directionally comparable until algorithmic delay is reported
  separately from runtime.

What to add for a stronger comparison:

1. Sustain pedal evaluation.
2. Separate reporting of algorithmic lookahead latency.
3. Optional frame-level metrics if you want maximum similarity to their table.

### Hu et al. 2025

Paper:

- Exploring System Adaptations for Minimum Latency Real-Time Piano
  Transcription

What the paper emphasizes:

- strict causality
- minimum-latency real-time transcription
- onset-focused evaluation at 10, 20, and 30 ms timing tolerances
- algorithmic delay from preprocessing and lookahead
- tradeoff between stricter latency and note accuracy

What LiveScore can compare now:

- `Partial`: end-to-end latency, but only loosely, because Hu is mainly about
  minimum algorithmic latency and stricter timing tolerances rather than wall-clock
  P95 chunk runtime
- `No`: note onset F1 at 10 / 20 / 30 ms
- `No`: onset + offset F1 at strict tolerances
- `No`: strict causal / lookahead accounting
- `No`: duplicate-rate, boundary miss, stabilization latency
- `No`: notation / rhythm metrics as a direct apples-to-apples comparison

Bottom line:

- With the current benchmark plan, this is not yet a fair direct comparison.
- Your default 50 ms note F1 is too forgiving for a paper that explicitly shifts
  evaluation to 10-30 ms to reflect minimum-latency usability.

What to add for a fair comparison:

1. Onset-only note F1 at 10 ms, 20 ms, and 30 ms.
2. Offset-aware note F1 at those same stricter tolerances if you want to compare
   beyond onset detection.
3. An explicit algorithmic-latency measure:
   future frames, lookahead, or effective prediction delay in ms.

### Peter, Hu, Widmer 2025

Paper:

- Pairing Real-Time Piano Transcription with Symbol-level Tracking for Precise
  and Robust Score Following

What the paper emphasizes:

- downstream score following
- absolute tracking error
- robustness / tracking success
- audio-symbolic real-time pipeline

What LiveScore can compare now:

- `No`: standard AMT note metrics are not the main task in that paper
- `No`: current LiveScore benchmark plan does not include score-following
  success, tracking precision, or alignment robustness metrics

Bottom line:

- This is relevant as adjacent evidence for symbolic real-time pipelines.
- It is not a direct benchmark target for the current LiveScore evaluation stack.

What to add for a fair comparison:

1. A downstream score-following benchmark.
2. Tracking error and robustness metrics.

## Condensed Comparison Table

| Paper             | Standard note F1 | Offset-aware note F1 | Strict-tolerance onset F1 | Latency | Streaming stability metrics | Notation / rhythm metrics | Overall current comparability                            |
| ----------------- | ---------------- | -------------------- | ------------------------- | ------- | --------------------------- | ------------------------- | -------------------------------------------------------- |
| Fernandez 2023    | Direct           | No                   | No                        | Partial | No                          | No                        | Partial                                                  |
| Kwon et al. 2024  | Direct           | Direct               | No                        | Partial | No                          | No                        | Strongest current apples-to-apples live-paper comparison |
| Wei et al. 2025   | Direct           | Direct               | No                        | Partial | No                          | No                        | Strong current apples-to-apples live-paper comparison    |
| Hu et al. 2025    | No               | No                   | No                        | Partial | No                          | No                        | Not fair yet without stricter timing metrics             |
| Peter et al. 2025 | No               | No                   | No                        | No      | No                          | No                        | Different task                                           |

## What The Current Stack Is Best At

The current LiveScore benchmark plan is strongest on these two fronts:

1. Standard AMT comparability
   - note F1
   - offset-aware note F1
   - basic latency reporting

2. Streaming-systems differentiation
   - duplicate-rate
   - boundary miss rate
   - stabilization latency
   - time-to-visible

This means:

- LiveScore can already compare reasonably well to Kwon 2024 and Wei 2025 on
  mainstream note-level metrics.
- LiveScore cannot yet compare fairly to the minimum-latency line of work led by
  Hu 2025.
- LiveScore already measures several live-quality metrics that those papers do
  not report at all.

## Important Interpretation

Several of LiveScore's most interesting metrics are currently unmatched by the
reviewed live papers:

1. Duplicate-rate
2. Boundary miss rate
3. Stabilization latency
4. Time-to-visible
5. Rhythm note-value accuracy

That is good and bad.

Good:

- these metrics support a stronger systems paper about live usability and
  retro-correction.

Bad:

- they do not replace standard note-level comparisons to prior AMT papers.

So the paper should likely report both:

1. standard note-level metrics that connect to existing literature
2. new streaming-systems metrics that justify LiveScore's specific contribution

## Highest-Value Additions

If the goal is to make the paper comparison matrix much stronger with minimal
extra work, the highest-value additions are:

1. Add onset-only note F1 at 10 / 20 / 30 ms.
   - This unlocks a fairer comparison to Hu 2025.

2. Separate algorithmic latency from runtime latency.
   - Report both future-context delay and measured wall-clock P95.
   - This unlocks cleaner comparison to Kwon 2024, Wei 2025, and Hu 2025.

3. Add a minimal velocity metric.
   - This unlocks a cleaner comparison to Fernandez 2023 and Kwon 2024.

4. Add optional sustain pedal evaluation.
   - This unlocks a cleaner comparison to Wei 2025.

## Recommended Benchmark Story

If you want the most defensible paper story with the least extra benchmark work,
the comparison structure should be:

1. Compare directly to Kwon 2024 and Wei 2025 on standard note-level metrics.
2. Add stricter timing metrics so Hu 2025 becomes comparable.
3. Keep duplicate-rate, boundary miss, stabilization latency, and time-to-visible
   as your differentiating live-systems contribution.
4. Do not spend effort trying to force a direct benchmark comparison to Peter
   2025 unless you decide to build a downstream score-following experiment.

## Sources Used

Primary internal sources:

1. `gpt_memory/live_accuracy_benchmark_plan.txt`
2. `LIVE_NEURAL_LATENCY_BUDGET.md`
3. `gpt_memory/live_transcription_prior_art_and_bar.md`

Primary external sources:

1. Fernandez 2023 abstract page
   - https://arxiv.org/abs/2303.04485
2. Kwon et al. 2024 HTML paper
   - https://arxiv.org/html/2404.06818v1
3. Wei et al. 2025 HTML paper
   - https://arxiv.org/html/2503.01362v1
4. Hu et al. 2025 HTML paper
   - https://arxiv.org/html/2509.07586v1
5. Peter et al. 2025 abstract page
   - https://arxiv.org/abs/2505.05078
