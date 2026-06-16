# Continuous Stream Decoder Sweep

Generated: 2026-06-16T08:19:57
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9118 | 0.8645 | 0.9644 | 0.6788 | 0.8873 | 0.8950 | 0.8036 | 0.44 | 25.30 |
