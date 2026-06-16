# Continuous Stream Decoder Sweep

Generated: 2026-06-16T08:19:23
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9085 | 0.8566 | 0.9670 | 0.6881 | 0.8870 | 0.8930 | 0.7902 | 0.45 | 24.37 |
