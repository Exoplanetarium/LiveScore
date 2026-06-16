# Continuous Stream Decoder Sweep

Generated: 2026-06-16T08:29:57
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9488 | 0.9434 | 0.9542 | 0.7418 | 0.9028 | 0.9422 | 0.9304 | 0.23 | 24.59 |
