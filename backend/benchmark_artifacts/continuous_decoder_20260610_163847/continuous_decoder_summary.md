# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:38:52
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9284 | 0.9543 | 0.9038 | 0.7347 | 0.8964 | 0.9235 | 0.9474 | 0.48 | 27.28 |
