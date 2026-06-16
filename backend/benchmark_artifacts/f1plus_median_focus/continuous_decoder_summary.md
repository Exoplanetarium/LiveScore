# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:11:58
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9110 | 0.8720 | 0.9537 | 0.5529 | 0.8457 | 0.9027 | 0.8845 | 0.10 | 34.08 |
