# Score Quality Report

Source: `backend\live_benchmark_replay_auto_v2_results_20260603_slotconsensus.json`

display_cluster_f1 is strict exact onset-cluster F1. It measures whether each onset-aligned pitch set exactly matches ground truth; it is useful for chord exactness, but it is not by itself a full generated-score accuracy metric.

## Summary

| arm | note_f1 | offset_f1 | note_value_acc | exact_cluster_f1 | cluster_jaccard | avg_revisions |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0.9250 | 0.8387 | 26.7% | 0.6845 | 0.8772 | 1.4553 |
| treatment | 0.9247 | 0.8376 | 26.8% | 0.6830 | 0.8768 | 1.4519 |

## Control

### Counts

| area | matched/exact | predicted/evaluable | ground truth | errors |
| --- | --- | --- | --- | --- |
| notes | 3966 | 4119 | 4456 | missing 490, extra 153 |
| durations | 3596 | 4119 | 4456 | 370 |
| note values | 1058.0 | 3966 | - | 2908.0 |
| clusters | 1381 | 1982 | 2053 | under 327, over 145, pitch 64 |

Exact cluster F1 needs about `233` additional exact cluster matches to reach `0.80`, assuming predicted/ground-truth cluster counts stay fixed.

### Timing And Stability

| metric | value |
| --- | --- |
| boundary recall | 91.2% |
| weighted time-to-visible median | 345 ms |
| weighted stabilization median | 1249 ms |
| weighted avg revisions | 1.4553 |
| max revisions | 5.0000 |
| duplicates / 100 notes | 4.3214 |

### Worst Cluster Clips

| clip | cluster_f1 | jaccard | under | over | pitch |
| --- | --- | --- | --- | --- | --- |
| clip_017 | 0.2338 | 0.7364 | 18 | 4 | 4 |
| clip_006 | 0.4096 | 0.7964 | 12 | 3 | 6 |
| clip_044 | 0.4500 | 0.7802 | 7 | 0 | 2 |
| clip_035 | 0.4857 | 0.7578 | 14 | 1 | 2 |
| clip_015 | 0.4966 | 0.8154 | 15 | 14 | 2 |

### Most Unstable Clips

| clip | avg revisions | max revisions | stabilization | note_f1 |
| --- | --- | --- | --- | --- |
| clip_010 | 2.6154 | 4.0000 | 2178 ms | 0.9573 |
| clip_044 | 2.5000 | 4.0000 | 1965 ms | 0.8081 |
| clip_018 | 2.3333 | 5.0000 | 1990 ms | 0.9425 |
| clip_030 | 2.3182 | 4.0000 | 2109 ms | 0.9206 |
| clip_047 | 2.1111 | 3.0000 | 1706 ms | 0.9245 |

## Treatment

### Counts

| area | matched/exact | predicted/evaluable | ground truth | errors |
| --- | --- | --- | --- | --- |
| notes | 3964 | 4118 | 4456 | missing 492, extra 154 |
| durations | 3591 | 4118 | 4456 | 373 |
| note values | 1062.0 | 3964 | - | 2902.0 |
| clusters | 1379 | 1985 | 2053 | under 330, over 146, pitch 63 |

Exact cluster F1 needs about `237` additional exact cluster matches to reach `0.80`, assuming predicted/ground-truth cluster counts stay fixed.

### Timing And Stability

| metric | value |
| --- | --- |
| boundary recall | 91.2% |
| weighted time-to-visible median | 345 ms |
| weighted stabilization median | 1228 ms |
| weighted avg revisions | 1.4519 |
| max revisions | 5.0000 |
| duplicates / 100 notes | 4.3468 |

### Worst Cluster Clips

| clip | cluster_f1 | jaccard | under | over | pitch |
| --- | --- | --- | --- | --- | --- |
| clip_017 | 0.2338 | 0.7364 | 18 | 4 | 4 |
| clip_014 | 0.4000 | 0.7037 | 0 | 2 | 3 |
| clip_006 | 0.4096 | 0.7964 | 12 | 3 | 6 |
| clip_044 | 0.4500 | 0.7820 | 6 | 1 | 2 |
| clip_015 | 0.4690 | 0.7928 | 16 | 15 | 2 |

### Most Unstable Clips

| clip | avg revisions | max revisions | stabilization | note_f1 |
| --- | --- | --- | --- | --- |
| clip_010 | 2.6154 | 4.0000 | 2184 ms | 0.9573 |
| clip_044 | 2.4091 | 4.0000 | 1969 ms | 0.8235 |
| clip_030 | 2.3182 | 4.0000 | 2107 ms | 0.9255 |
| clip_018 | 2.2727 | 5.0000 | 1985 ms | 0.9425 |
| clip_003 | 2.2222 | 4.0000 | 1874 ms | 0.9000 |
