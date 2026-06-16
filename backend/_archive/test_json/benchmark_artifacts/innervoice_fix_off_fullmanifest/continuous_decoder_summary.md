# Continuous Stream Decoder Sweep

Generated: 2026-06-09T16:50:51
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | innervoice_fix_off | True | 0.7840 | 0.7980 | 0.7705 | 0.4312 | 0.7609 | 0.7776 | 0.7906 | 0.28 | 25.15 |
