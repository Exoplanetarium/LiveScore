# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:12:49
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | birth_gates_off | True | 0.9475 | 0.9374 | 0.9578 | 0.7401 | 0.9016 | 0.9411 | 0.9290 | 0.18 | 26.18 |
| 2 | display_obs_2 | True | 0.8934 | 0.8360 | 0.9593 | 0.6464 | 0.8692 | 0.8872 | 0.8290 | 0.08 | 26.02 |
| 3 | onset_055 | True | 0.8933 | 0.8348 | 0.9605 | 0.6493 | 0.8721 | 0.8875 | 0.8268 | 0.08 | 24.86 |
| 4 | onset_050 | True | 0.8922 | 0.8380 | 0.9540 | 0.6431 | 0.8687 | 0.8863 | 0.8290 | 0.13 | 26.17 |
| 5 | onset_045 | True | 0.8896 | 0.8391 | 0.9466 | 0.6366 | 0.8663 | 0.8829 | 0.8290 | 0.13 | 25.86 |
| 6 | frame_evidence_50ms | False | 0.9141 | 0.8689 | 0.9641 | 0.6879 | 0.8830 | 0.9089 | 0.8645 | 0.02 | 26.98 |
