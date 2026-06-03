# Score Quality Report

Source: `backend\live_benchmark_replay_auto_v2_results_20260602_enhancedcluster_group_step050.json`

display_cluster_f1 is strict exact onset-cluster F1. It measures whether each onset-aligned pitch set exactly matches ground truth; it is useful for chord exactness, but it is not by itself a full generated-score accuracy metric.

## Summary

| arm | note_f1 | offset_f1 | note_value_acc | exact_cluster_f1 | cluster_jaccard | avg_revisions |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0.9248 | 0.8384 | 26.7% | 0.7007 | 0.8867 | 1.4506 |
| treatment | 0.9249 | 0.8379 | 26.8% | 0.7012 | 0.8872 | 1.4463 |

## Control

### Counts

| area | matched/exact | predicted/evaluable | ground truth | errors |
| --- | --- | --- | --- | --- |
| notes | 3964 | 4117 | 4456 | missing 492, extra 153 |
| durations | 3594 | 4117 | 4456 | 370 |
| note values | 1058.0 | 3964 | - | 2906.0 |
| clusters | 1419 | 1997 | 2053 | under 321, over 133, pitch 60 |

Exact cluster F1 needs about `201` additional exact cluster matches to reach `0.80`, assuming predicted/ground-truth cluster counts stay fixed.

### Timing And Stability

| metric | value |
| --- | --- |
| boundary recall | 91.2% |
| weighted time-to-visible median | 347 ms |
| weighted stabilization median | 1250 ms |
| weighted avg revisions | 1.4506 |
| max revisions | 5.0000 |
| duplicates / 100 notes | 4.3235 |

### Worst Cluster Clips

| clip | cluster_f1 | jaccard | under | over | pitch |
| --- | --- | --- | --- | --- | --- |
| clip_017 | 0.1818 | 0.7162 | 17 | 4 | 7 |
| clip_006 | 0.3902 | 0.8215 | 12 | 4 | 5 |
| clip_044 | 0.4500 | 0.7802 | 7 | 0 | 2 |
| clip_035 | 0.4857 | 0.7578 | 14 | 1 | 2 |
| clip_015 | 0.4861 | 0.8202 | 15 | 15 | 1 |

### Most Unstable Clips

| clip | avg revisions | max revisions | stabilization | note_f1 |
| --- | --- | --- | --- | --- |
| clip_044 | 2.5000 | 4.0000 | 1964 ms | 0.8081 |
| clip_018 | 2.3333 | 5.0000 | 1990 ms | 0.9425 |
| clip_030 | 2.3182 | 4.0000 | 2105 ms | 0.9206 |
| clip_047 | 2.1111 | 3.0000 | 1714 ms | 0.9245 |
| clip_033 | 2.0952 | 3.0000 | 1920 ms | 0.9111 |

## Treatment

### Counts

| area | matched/exact | predicted/evaluable | ground truth | errors |
| --- | --- | --- | --- | --- |
| notes | 3966 | 4120 | 4456 | missing 490, extra 154 |
| durations | 3593 | 4120 | 4456 | 373 |
| note values | 1062.0 | 3966 | - | 2904.0 |
| clusters | 1421 | 2000 | 2053 | under 320, over 133, pitch 59 |

Exact cluster F1 needs about `201` additional exact cluster matches to reach `0.80`, assuming predicted/ground-truth cluster counts stay fixed.

### Timing And Stability

| metric | value |
| --- | --- |
| boundary recall | 91.2% |
| weighted time-to-visible median | 345 ms |
| weighted stabilization median | 1225 ms |
| weighted avg revisions | 1.4463 |
| max revisions | 5.0000 |
| duplicates / 100 notes | 4.3447 |

### Worst Cluster Clips

| clip | cluster_f1 | jaccard | under | over | pitch |
| --- | --- | --- | --- | --- | --- |
| clip_017 | 0.1818 | 0.7162 | 17 | 4 | 7 |
| clip_006 | 0.3902 | 0.8215 | 12 | 4 | 5 |
| clip_014 | 0.4000 | 0.7037 | 0 | 2 | 3 |
| clip_044 | 0.4500 | 0.7820 | 6 | 1 | 2 |
| clip_035 | 0.4789 | 0.7652 | 14 | 1 | 2 |

### Most Unstable Clips

| clip | avg revisions | max revisions | stabilization | note_f1 |
| --- | --- | --- | --- | --- |
| clip_044 | 2.4091 | 4.0000 | 1968 ms | 0.8235 |
| clip_018 | 2.3333 | 5.0000 | 1988 ms | 0.9425 |
| clip_030 | 2.3182 | 4.0000 | 2105 ms | 0.9255 |
| clip_003 | 2.2222 | 4.0000 | 1868 ms | 0.9000 |
| clip_011 | 2.2000 | 3.0000 | 2343 ms | 0.9412 |
