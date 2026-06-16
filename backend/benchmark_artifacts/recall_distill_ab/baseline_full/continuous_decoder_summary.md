# Continuous Stream Decoder Sweep

Generated: 2026-06-16T08:22:59
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.8929 | 0.8301 | 0.9660 | 0.6519 | 0.8717 | 0.8874 | 0.8232 | 0.03 | 23.73 |
