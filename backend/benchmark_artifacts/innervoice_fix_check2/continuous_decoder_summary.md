# Continuous Stream Decoder Sweep

Generated: 2026-06-09T16:29:41
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.8993 | 0.8750 | 0.9250 | 0.2812 | 0.7450 | 0.8958 | 0.9192 | 0.00 | 27.42 |
