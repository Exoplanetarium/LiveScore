# Continuous Stream Decoder Sweep

Generated: 2026-06-10T16:24:26
Surface: `score`

This replays audio through `ContinuousLiveStreamSession`, not the older chunk-upload benchmark path.

| Rank | Candidate | Latency OK | Note F1 | Recall | Precision | Cluster F1 | Jaccard | 20ms F1 | Boundary Recall | Dup/100 | p95 inference ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ship_default | True | 0.9478 | 0.9378 | 0.9580 | 0.7293 | 0.8978 | 0.9403 | 0.9290 | 0.18 | 25.23 |
| 2 | ship_ema | True | 0.9475 | 0.9374 | 0.9578 | 0.7401 | 0.9016 | 0.9411 | 0.9290 | 0.18 | 25.68 |
| 3 | ship_disp_frame_100ms | True | 0.9468 | 0.9381 | 0.9556 | 0.7261 | 0.8964 | 0.9393 | 0.9290 | 0.18 | 24.75 |
| 4 | ship_obs2 | True | 0.9451 | 0.9419 | 0.9483 | 0.7178 | 0.8942 | 0.9370 | 0.9312 | 0.27 | 25.24 |
| 5 | ship_onset055 | True | 0.9447 | 0.9414 | 0.9480 | 0.7196 | 0.8946 | 0.9366 | 0.9312 | 0.29 | 25.37 |
| 6 | ship_legacy_gates | True | 0.8929 | 0.8301 | 0.9660 | 0.6519 | 0.8717 | 0.8874 | 0.8232 | 0.03 | 24.68 |
