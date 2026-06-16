# Continuous Stream Decoder Sweep

Generated: 2026-06-13T09:55:48
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9475 | 0.9374 | 0.9578 | 0.7401 | 0.9016 | 0.9411 | 0.9290 | 0.18 | 24.96 |
