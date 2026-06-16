# Continuous Stream Decoder Sweep

Generated: 2026-06-15T18:07:34
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | harmonic_prior_triads_only | True | 0.9086 | 0.8645 | 0.9574 | 0.6819 | 0.8842 | 0.8932 | 0.7991 | 0.44 | 26.59 |
| 2 | snap15_harmonic_prior | True | 0.9086 | 0.8645 | 0.9574 | 0.6779 | 0.8821 | 0.8904 | 0.7991 | 0.44 | 26.00 |
| 3 | baseline_current | True | 0.9085 | 0.8566 | 0.9670 | 0.6881 | 0.8870 | 0.8930 | 0.7902 | 0.45 | 26.65 |
