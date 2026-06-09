# Continuous Stream Decoder Sweep

Generated: 2026-06-09T16:01:42
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.8186 | 0.8919 | 0.7564 | 0.0699 | 0.6334 | 0.8124 | 0.9293 | 0.00 | 29.46 |
| 2 | gate_relaxed_diag | True | 0.8148 | 0.8919 | 0.7500 | 0.0680 | 0.6064 | 0.8086 | 0.9293 | 0.00 | 26.47 |
| 3 | gate_off_diag | True | 0.8060 | 0.9054 | 0.7263 | 0.0774 | 0.6060 | 0.7970 | 0.9495 | 0.27 | 23.99 |
