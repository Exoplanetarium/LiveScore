# Decoder Tuning Sweep

Generated: 2026-06-04T11:55:42

This ranks decoder-only settings. It uses score submetrics directly; there is no combined score-quality number.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Dup/100 | p95 chunk ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline_current | True | 0.9247 | 0.8896 | 0.9626 | 0.6830 | 0.8768 | 0.9186 | 4.35 | 25.79 |
| 2 | enhanced_onset_070 | True | 0.9209 | 0.8988 | 0.9441 | 0.6749 | 0.8726 | 0.9135 | 4.62 | 26.31 |
| 3 | enhanced_onset_080 | True | 0.9152 | 0.8622 | 0.9751 | 0.6738 | 0.8721 | 0.9107 | 3.93 | 27.45 |
| 4 | group_base_040 | False | 0.9262 | 0.8918 | 0.9634 | 0.6986 | 0.8858 | 0.9076 | 4.02 | 38.89 |
| 5 | duplicate_window_060ms | False | 0.9250 | 0.8903 | 0.9626 | 0.6833 | 0.8766 | 0.9190 | 4.34 | 38.62 |
| 6 | enhanced_offset_025 | False | 0.9249 | 0.8900 | 0.9626 | 0.6840 | 0.8771 | 0.9188 | 4.34 | 31.24 |
| 7 | enhanced_offset_045 | False | 0.9249 | 0.8900 | 0.9626 | 0.6840 | 0.8771 | 0.9188 | 4.34 | 38.53 |
| 8 | group_base_020 | False | 0.9249 | 0.8900 | 0.9626 | 0.6840 | 0.8771 | 0.9188 | 4.34 | 36.77 |
| 9 | harmonic_filter_on | False | 0.9247 | 0.8896 | 0.9626 | 0.6830 | 0.8768 | 0.9186 | 4.35 | 39.10 |
| 10 | group_prune_on | False | 0.9247 | 0.8896 | 0.9626 | 0.6830 | 0.8768 | 0.9186 | 4.35 | 37.83 |
| 11 | recall_plus_dup | False | 0.9211 | 0.8992 | 0.9442 | 0.6759 | 0.8729 | 0.9138 | 4.62 | 37.73 |
| 12 | recall_plus_harmonic_filter | False | 0.9211 | 0.8992 | 0.9442 | 0.6759 | 0.8729 | 0.9138 | 4.62 | 38.53 |
| 13 | enhanced_onset_065 | False | 0.9189 | 0.9075 | 0.9305 | 0.6709 | 0.8719 | 0.9109 | 4.60 | 38.28 |
| 14 | enhanced_onset_085 | False | 0.8523 | 0.7498 | 0.9873 | 0.5982 | 0.8328 | 0.8490 | 2.87 | 37.16 |
