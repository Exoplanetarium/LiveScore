# Continuous Stream Decoder Sweep

Generated: 2026-06-15T18:14:14
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9085 | 0.8566 | 0.9670 | 0.6881 | 0.8870 | 0.8930 | 0.7902 | 0.45 | 25.14 |
| 2 | onset_055 | True | 0.9081 | 0.8592 | 0.9628 | 0.6911 | 0.8889 | 0.8940 | 0.7991 | 0.45 | 25.14 |
| 3 | onset_045 | True | 0.9035 | 0.8645 | 0.9462 | 0.6697 | 0.8816 | 0.8869 | 0.7991 | 0.44 | 25.22 |
| 4 | onset_050 | True | 0.9034 | 0.8632 | 0.9475 | 0.6707 | 0.8825 | 0.8867 | 0.7946 | 0.44 | 25.13 |
