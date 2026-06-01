# Live Transcription Prior Art And Bar To Beat

Updated: 2026-05-29

## Purpose

This note summarizes the most relevant existing methods and products for a paper
about LiveScore's proposed angle:

- latency-bounded live piano transcription
- overlap-aware chunked inference
- asynchronous retro-correction / deferred refinement
- score-oriented output rather than MIDI-only output

The goal is not just to list prior work. The goal is to define what LiveScore
would actually need to demonstrate in order to beat those methods in a paper.

## Correction To Earlier Framing

There is now a meaningful recent literature on real-time / streaming piano
transcription.

So the paper should not claim that live transcription is absent. The narrower
and more defensible claim is:

- much of the benchmark-dominant AMT literature is still offline
- recent streaming papers are often audio-to-MIDI or event-centric
- notation-first live transcription with explicit retro-correction,
  stabilization, and seam-aware evaluation is still comparatively less covered

## Core Framing

The strongest paper claim is not:

- "our piano transcription model has the highest offline F1"

The strongest paper claim is:

- "under a real live-latency budget, our split fast-path plus retro-correction
  architecture improves the accuracy-latency tradeoff relative to standard
  streaming baselines and is more directly useful for in-progress score display"

Why this framing is strongest for this repo:

1. The repo already has an explicit live latency budget of P95 under 500 ms.
2. The repo already has overlap-aware chunk infrastructure and deferred
   refinement hooks.
3. The internal diagnostics already point to chunk-boundary errors as a primary
   live failure mode.
4. Many strong academic AMT systems are offline or full-context systems, while
   the recent real-time papers are still mostly event-centric rather than live
   notation systems.
5. Most commercial audio-to-score tools are black boxes and are described as
   seconds-level conversion systems rather than in-progress live score systems.

## Relevant Constraints From This Repo

LiveScore's own method has to respect these constraints to stay on-message:

1. Fast path P95 latency should stay below about 500 ms on warmed GPU.
2. Heavy steps such as full beat tracking, tempo-grid search, and full notation
   cleanup should not live in the fast path.
3. The system is trying to display notation during recording, not only after a
   full offline pass.
4. The most important live failure modes are boundary misses, duplicate onsets,
   stabilization delay, and later score cleanup drift.

## Comparison Table

| Comparator                                                                                                    | What It Is                                                                             | Qualifications / Why It Matters                                                                                              | Main Constraints                                                                                                                                                                                | What LiveScore Must Show To Beat It                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Onsets and Frames (Hawthorne et al., 2018)                                                                    | Canonical solo-piano AMT model using onset-conditioned frame prediction                | Landmark piano transcription method; strong public benchmark reference; directly relevant architectural ancestor             | Framed as offline piano-to-MIDI, not a live notation system; no hard live latency budget; does not target chunk-boundary repair or stabilization latency                                        | LiveScore does not need to beat its raw offline F1 outright. It needs to show a better accuracy-latency tradeoff: responsive in-progress score updates plus better streaming boundary behavior under a fixed P95 latency budget                     |
| High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times (Kong et al., 2020/2021) | High-resolution piano AMT with precise onset/offset regression and pedal transcription | Very strong piano-specific baseline; reported onset F1 of 96.72 percent on MAESTRO; important for onset quality expectations | Still an offline-style piano transcription system; no public live chunking story; no explicit streaming stability or user-visible latency target                                                | LiveScore should aim to preserve as much of this onset quality as possible in streaming conditions. The win would be: lower chunk-boundary miss rate and lower stabilization delay at live latency, even if absolute offline onset F1 remains lower |
| MT3 (Gardner et al., 2022)                                                                                    | General-purpose multitask multitrack transformer transcription                         | Strong modern sequence model; relevant because it represents the broad "general AMT" direction rather than piano-only AMT    | Not a live piano score system; not optimized for low-latency in-progress notation; broader scope makes it less likely to dominate the piano live setting                                        | LiveScore should beat MT3 on piano-specific live criteria: lower latency, faster time-to-visible, better stabilization, and more useful score-oriented output during capture                                                                        |
| Basic Pitch (Spotify)                                                                                         | Open-source local audio-to-MIDI tool for single-instrument audio                       | Important practical baseline because it is accessible, local, and user-facing; useful comparison for lightweight deployment  | MIDI-first rather than notation-first; public site does not present a live chunked score workflow; not piano-specialized; no published live stabilization metrics                               | LiveScore should show better piano-specific note/rhythm quality plus true live score updates, not just audio-to-MIDI conversion                                                                                                                     |
| Klang.io / Piano2Notes                                                                                        | Commercial audio-to-sheet-music transcription product                                  | Important commercial comparison because it is directly score-oriented and piano-focused                                      | Public positioning emphasizes creating sheet music within seconds from recorded audio; no public benchmark methodology; no public live latency claim; black-box comparison is hard to reproduce | LiveScore can beat it in a paper by offering something it does not publicly claim: measurable in-progress live notation under a latency budget, plus transparent reproducible evaluation                                                            |
| Naive chunked streaming baseline                                                                              | Same model family, but run per chunk without retro-correction or strong seam logic     | This is the most important scientific baseline because it isolates whether the architecture matters                          | Suffers from chunk-boundary duplicates, seam misses, chopped sustains, and unstable recent notes                                                                                                | This is the baseline LiveScore absolutely must beat. If retro-correction and overlap-aware repair do not clearly outperform naive chunking, the paper angle weakens substantially                                                                   |

## Live / Streaming Papers That Should Be In Scope

| Comparator                                                                                                                  | What It Is                                                                                                       | Qualifications / Why It Matters                                                                                                                                      | Main Constraints                                                                                                                                                               | What LiveScore Must Show To Beat It                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Onsets and Velocities: Affordable Real-Time Piano Transcription Using Convolutional Neural Networks (Fernandez, 2023)       | Small CNN-based real-time piano transcription model focused on onset and velocity prediction                     | Important because it explicitly targets affordable real-time deployment and reports competitive MAESTRO results with modest model size                               | Narrower task than score generation: onset and velocity focus, lower temporal resolution, and no explicit notation-cleanup or retro-correction story                           | LiveScore should show that it can stay live while producing more useful symbolic output: better rhythm, duration, and score readability, not just event onsets                                  |
| Towards Efficient and Real-Time Piano Transcription Using Neural Autoregressive Models (Kwon et al., 2024)                  | Real-time piano transcription with lightweight autoregressive architectures and latency/size analysis            | Strong direct comparator because it explicitly trades model size and latency against piano transcription accuracy                                                    | Still model-centric and audio-to-note oriented; no explicit chunk-seam repair, stabilization-latency, or score-rendering evaluation                                            | LiveScore should match the spirit of its real-time accuracy/efficiency tradeoff while adding better streaming stability and notation-level usefulness                                           |
| Streaming Piano Transcription Based on Consistent Onset and Offset Decoding with Sustain Pedal Detection (Wei et al., 2025) | Explicit streaming piano audio-to-MIDI model with separate onset and offset decoding plus sustain pedal handling | This is one of the most relevant direct streaming piano baselines because it targets sequential onset/offset decoding rather than offline full-context transcription | Still MIDI-first; the abstract does not frame the problem as live notation display, stabilization, or chunk-boundary UI quality                                                | LiveScore should show comparable note-event quality while outperforming on notation-oriented metrics and explicit live quality metrics such as time-to-visible and stabilization latency        |
| Exploring System Adaptations For Minimum Latency Real-Time Piano Transcription (Hu et al., 2025)                            | Strictly causal, minimum-latency real-time piano transcription baseline                                          | Very relevant because it explicitly targets the sub-30 ms regime and studies how accuracy drops under stricter causality                                             | Its target regime is harsher than LiveScore's current latency budget, and the framing is still event transcription rather than score generation or retro-corrective refinement | LiveScore does not need to match the same latency target. It needs to show that a looser but still live budget buys materially better symbolic stability, boundary repair, and notation quality |

Adjacent live work worth citing even if it is not a direct baseline:

- Pairing Real-Time Piano Transcription with Symbol-level Tracking for Precise and Robust Score Following (Peter, Hu, Widmer, 2025) is score-following rather than score generation, but it is still useful evidence that converting live piano audio into a symbolic stream can improve a downstream real-time task.

## What "Beat" Should Mean By Baseline Family

### 1. Offline Piano AMT Papers

For Onsets and Frames and Kong et al., "beat" should not mean:

- strictly higher offline MAESTRO F1

That is the wrong contest for this repo.

For these systems, "beat" should mean:

1. LiveScore delivers useful in-progress notation during recording.
2. P95 live chunk latency stays under the declared budget.
3. Streaming-specific metrics are materially better than straightforward
   streaming baselines.
4. Overall piano transcription quality remains respectable enough that the live
   tradeoff is clearly worth it.

In other words, the paper should argue:

- those systems are stronger offline references
- our contribution is the live systems tradeoff, not raw full-context SOTA

### 2. Live / Streaming Piano Papers

For Fernandez, Kwon, Wei, and Hu type systems, "beat" should mean:

1. Comparable or better note-event quality at a declared live latency.
2. Better stabilization and boundary behavior over time, not just better final
   aggregate note F1.
3. Better notation usefulness: rhythm grouping, note values, and readable score
   output instead of MIDI-only event streams.
4. A clear advantage from overlap-aware repair and retro-correction over purely
   causal decoding.
5. Reproducible latency-quality curves rather than only one speed / accuracy
   operating point.

For the strict minimum-latency line of work, the comparison needs one nuance:

- LiveScore does not need to win on raw latency if the comparator targets the
  sub-30 ms regime.
- LiveScore instead needs to show that its slightly looser budget produces much
  better symbolic stability and notation utility.

### 3. General AMT Systems

For MT3-type systems, "beat" should mean:

1. Better piano-specific performance in the live regime.
2. Lower latency and less decoding overhead.
3. Better stability for live score display.
4. Better notation-oriented output rather than just symbolic transcription.

### 4. Practical Open-Source Tools

For Basic Pitch, "beat" should mean:

1. Better piano note accuracy.
2. Better rhythm / note-value handling.
3. A real score-generation story rather than MIDI-only output.
4. Actual live in-progress updates rather than simple local conversion.

### 5. Commercial Products

For Klang.io / Piano2Notes, "beat" should mean one of two things:

1. A stronger scientific claim:
   transparent public metrics, reproducible benchmark suites, and true live
   operation under a fixed latency budget.

2. A stronger product claim:
   the score appears while the musician is still playing, not only after upload
   or post-hoc conversion.

It is risky to build the paper around "we beat a commercial black box" unless
you are prepared to run a careful manual black-box benchmark.

## The Main Baseline The Paper Must Beat

The paper's main "must beat" baseline should be:

- naive chunked streaming without overlap-aware seam repair and without
  asynchronous retro-correction

Why this is the strongest primary baseline:

1. It is directly relevant to the claimed contribution.
2. It is reproducible.
3. It isolates the value of the architecture rather than conflating model and
   systems changes.
4. It matches the repo's existing diagnostics, which already point to chunk
   boundaries as a primary live failure mode.

This means the paper's headline result should probably be phrased as:

- compared with straightforward chunked streaming, our overlap-aware and
  retro-corrective architecture reduces boundary miss rate, duplicate-rate, and
  stabilization latency while preserving live responsiveness

## Concrete Bar LiveScore Should Hit

Using the repo's current benchmark philosophy, the method would have a strong
case if it can show most of the following:

1. P95 live chunk latency below 500 ms.
2. No material P95 latency regression versus the current fast baseline.
3. Boundary miss rate improved by at least 20 percent versus naive chunking.
4. Duplicate-rate improved by at least 15 percent versus naive chunking.
5. Note F1 improved by at least 1 absolute point, or at minimum does not
   regress while the streaming-specific metrics improve.
6. Stabilization latency decreases clearly.
7. Time-to-visible remains low enough that the UI still feels live.

If those hold on both replayed audio and device-capture audio, the paper angle
is credible.

## Strong Claim, Weak Claim, And Stretch Claim

### Strong Claim

Under a fixed live latency budget, LiveScore's chunked fast path plus
retro-correction improves streaming transcription quality relative to standard
chunked baselines.

Why this is strong:

- directly supported by the repo's architecture
- directly benchmarkable
- aligned with the known failure modes
- does not require claiming offline state of the art

### Weak Claim

LiveScore is simply a better piano transcription model.

Why this is weak:

- the repo's strongest unique contribution is systems behavior under live
  constraints, not just the backbone model

### Stretch Claim

LiveScore beats commercial audio-to-score products in real use.

Why this is risky:

- commercial systems are black boxes
- public evaluation protocols are usually absent
- comparisons can become anecdotal unless run very carefully

## Suggested Positioning In The Paper

Recommended positioning:

- Prior offline AMT work shows what is possible with full context.
- Recent streaming piano papers show that the live setting is real and active,
  but they mostly evaluate event transcription and causal decoding rather than
  live score readability.
- Prior practical tools show demand for automatic transcription and editable
  symbolic output.
- What remains underexplored is live score generation that stays responsive
  while repairing chunk-boundary and short-context errors.
- LiveScore's contribution is a latency-aware architecture for that regime.

## Suggested Main Figure Or Table

The most useful main comparison table would likely be:

| System                 | Live during performance                                      | Public latency target                                                                           | Score-oriented output | Boundary repair                                          | Public reproducible benchmark |
| ---------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------- | ----------------------------- |
| Onsets and Frames      | No                                                           | No                                                                                              | Indirect / downstream | No                                                       | Yes                           |
| Kong et al.            | No                                                           | No                                                                                              | Indirect / downstream | No                                                       | Yes                           |
| Fernandez 2023         | Yes                                                          | Real-time claim, but no explicit P95 target on the arXiv abstract page                          | No, event-centric     | No explicit retro-correction                             | Yes                           |
| Kwon et al. 2024       | Yes                                                          | Real-time inference studied, but no simple public latency budget statement on the abstract page | No, event-centric     | No explicit seam metrics                                 | Yes                           |
| Wei et al. 2025        | Yes                                                          | Streaming framing, but no explicit user-visible latency budget on the abstract page             | No, MIDI-first        | Onset / offset consistency, but not notation seam repair | Yes                           |
| Hu et al. 2025         | Yes                                                          | Yes, explicitly discusses applications needing below 30 ms                                      | No, event-centric     | No retro-corrective refinement                           | Yes                           |
| MT3                    | No                                                           | No                                                                                              | Indirect / downstream | No                                                       | Yes                           |
| Basic Pitch            | Not really                                                   | No                                                                                              | No, MIDI-first        | No public story                                          | Yes                           |
| Klang.io / Piano2Notes | Publicly described as fast, but not clearly in-progress live | No public number                                                                                | Yes                   | Unknown                                                  | No                            |
| LiveScore target       | Yes                                                          | Yes, P95 under 500 ms                                                                           | Yes                   | Yes                                                      | Yes                           |

## Sources Used For This Note

External sources checked:

1. Magenta Onsets and Frames page
   - https://magenta.tensorflow.org/onsets-frames

2. Kong et al., High-resolution Piano Transcription with Pedals by Regressing
   Onset and Offset Times
   - https://arxiv.org/abs/2010.01815

3. Google Research MT3 page
   - https://research.google/pubs/mt3-multi-task-multitrack-music-transcription/

4. Spotify Basic Pitch site
   - https://basicpitch.spotify.com/

5. Klang.io public product site
   - https://klang.io/

6. Fernandez, Onsets and Velocities: Affordable Real-Time Piano Transcription
   Using Convolutional Neural Networks
   - https://arxiv.org/abs/2303.04485

7. Kwon et al., Towards Efficient and Real-Time Piano Transcription Using
   Neural Autoregressive Models
   - https://arxiv.org/abs/2404.06818

8. Wei et al., Streaming Piano Transcription Based on Consistent Onset and
   Offset Decoding with Sustain Pedal Detection
   - https://arxiv.org/abs/2503.01362

9. Hu et al., Exploring System Adaptations For Minimum Latency Real-Time Piano
   Transcription
   - https://arxiv.org/abs/2509.07586

10. Peter et al., Pairing Real-Time Piano Transcription with Symbol-level
    Tracking for Precise and Robust Score Following
    - https://arxiv.org/abs/2505.05078

Internal repo references used:

1. LIVE_NEURAL_LATENCY_BUDGET.md
2. gpt_memory/live_accuracy_research.txt
3. gpt_memory/live_accuracy_benchmark_plan.txt
4. /memories/repo/live-architecture.md
5. /memories/repo/retro-extend-diagnostic.md

## Bottom Line

There is enough recent streaming piano literature that the paper should address
it directly.

The strongest angle is still not "our chunking method exists".

It is:

- within the live-transcription literature, under a hard live latency budget,
  overlap-aware chunking plus retro-correction yields a better streaming
  accuracy-latency tradeoff than straightforward chunked or purely causal event
  transcription, while producing genuinely live score output

That is the comparison space where LiveScore has the clearest chance to make a
defensible and interesting contribution.
