# Continuous Stream Decoder Sweep

Generated: 2026-06-16T08:35:18
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9435 | 0.9309 | 0.9563 | 0.7158 | 0.8963 | 0.9260 | 0.9107 | 0.55 | 24.56 |
