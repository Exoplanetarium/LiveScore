# Continuous Stream Decoder Sweep

Generated: 2026-06-09T16:29:16
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9138 | 0.8781 | 0.9525 | 0.5914 | 0.8602 | 0.9098 | 0.9064 | 0.21 | 26.15 |
