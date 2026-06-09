# Continuous Stream Decoder Sweep

Generated: 2026-06-09T16:48:37
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.8921 | 0.8281 | 0.9667 | 0.6510 | 0.8712 | 0.8870 | 0.8217 | 0.03 | 26.50 |
