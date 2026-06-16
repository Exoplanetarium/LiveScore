# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:44:57
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9397 | 0.9492 | 0.9303 | 0.7603 | 0.9040 | 0.8894 | 0.9474 | 1.49 | 27.30 |
