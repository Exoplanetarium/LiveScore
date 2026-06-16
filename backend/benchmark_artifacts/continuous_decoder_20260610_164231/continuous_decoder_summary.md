# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:42:36
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9519 | 0.9543 | 0.9495 | 0.7984 | 0.9188 | 0.9468 | 0.9474 | 0.00 | 26.68 |
