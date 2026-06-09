# Continuous Stream Decoder Sweep

Generated: 2026-06-09T15:56:47
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.8186 | 0.8919 | 0.7564 | 0.0699 | 0.6334 | 0.8124 | 0.9293 | 0.00 | 38.41 |
