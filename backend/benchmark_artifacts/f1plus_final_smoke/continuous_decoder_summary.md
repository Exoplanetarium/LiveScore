# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:26:29
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ship_default | True | 0.9226 | 0.9085 | 0.9371 | 0.5794 | 0.8460 | 0.9164 | 0.9231 | 0.00 | 26.40 |
