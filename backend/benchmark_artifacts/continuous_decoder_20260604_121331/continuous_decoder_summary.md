# Continuous Stream Decoder Sweep

Generated: 2026-06-04T12:13:35
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.6500 | 0.5000 | 0.9286 | 0.5333 | 0.9250 | 0.6500 | 0.5000 | 0.00 | 38.52 |
