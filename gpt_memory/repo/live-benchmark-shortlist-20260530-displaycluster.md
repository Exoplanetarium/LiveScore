# Live Benchmark Display-Structure Shortlist

Source benchmark:

- `backend/live_benchmark_replay_auto_v2_results_20260530_displaycluster.json`

Purpose:

- Rank clips by how wrong the final displayed score structure is, not just whether the right pitches appear somewhere in the emitted note stream.
- Use control `display_cluster_f1` as the primary ranking signal.
- Use the gap `display_note_f1 - display_cluster_f1` to find clips where pitch recall looks acceptable but chord / onset grouping is still wrong.
- Keep `gpt_memory/repo/live-benchmark-shortlist-20260529.md` for raw note-level and seam-recall debugging; use this file when the target is the displayed score itself.

## Headline Metrics

- Control average final display note F1: `0.8566`
- Control average final display cluster F1: `0.5967`
- Treatment average final display note F1: `0.8525`
- Treatment average final display cluster F1: `0.5942`
- Retro average final display note F1: `0.5932`
- Retro average final display cluster F1: `0.4723`
- Average control note-to-structure gap: `0.2599`

Interpretation:

- The live system is doing materially better at recovering pitches than it is at preserving the final displayed chord / onset structure.
- Adaptive thresholding does not materially improve displayed structure on this manifest.
- Retro correction is substantially worse on final display structure than the baseline live path.
- The next likely improvement surface is note-to-chord grouping, onset-cluster formation, or later quantization / display retention, not threshold selection alone.

## First Attack Set: Lowest Final Display Cluster F1

These are the worst clips if the goal is to improve the displayed score structure directly.

1. `clip_031` - Hungarian Rhapsody No. 9, start `18s`
   - control display cluster F1 `0.1194`
   - control display note F1 `0.7398`
   - gap `0.6204`
   - exact cluster matches `4 / 26`
   - predicted clusters `41`
   - overclustered `15`, underclustered `1`, pitch conflicts `5`
   - boundary miss rate `0.0313`
   - buckets: `boundary_miss_failure`

2. `clip_017` - Pour le piano (Complete), start `51s`
   - control display cluster F1 `0.1728`
   - control display note F1 `0.8173`
   - gap `0.6444`
   - exact cluster matches `7 / 41`
   - predicted clusters `40`
   - overclustered `7`, underclustered `11`, pitch conflicts `9`
   - boundary miss rate `0.1343`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

3. `clip_024` - Hungarian Rhapsody No. 9 in E-flat Major, S. 244, start `21s`
   - control display cluster F1 `0.1905`
   - control display note F1 `0.7889`
   - gap `0.5984`
   - exact cluster matches `4 / 20`
   - predicted clusters `22`
   - overclustered `12`, underclustered `3`, pitch conflicts `1`
   - boundary miss rate `0.2308`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

4. `clip_006` - Pour le piano (Complete), start `57s`
   - control display cluster F1 `0.2439`
   - control display note F1 `0.8768`
   - gap `0.6329`
   - exact cluster matches `10 / 44`
   - predicted clusters `38`
   - overclustered `13`, underclustered `7`, pitch conflicts `5`
   - boundary miss rate `0.1270`
   - buckets: `runtime_only_win`, `boundary_miss_failure`, `high_revision_slow_stabilization`

5. `clip_035` - Estampes (Complete), start `81s`
   - control display cluster F1 `0.3099`
   - control display note F1 `0.7590`
   - gap `0.4492`
   - exact cluster matches `11 / 34`
   - predicted clusters `37`
   - overclustered `9`, underclustered `10`, pitch conflicts `3`
   - boundary miss rate `0.1333`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`, `note_value_offset_failure`

6. `clip_012` - Entragete, Op. 63, start `75s`
   - control display cluster F1 `0.3448`
   - control display note F1 `0.7652`
   - gap `0.4204`
   - exact cluster matches `10 / 24`
   - predicted clusters `34`
   - overclustered `6`, underclustered `4`, pitch conflicts `2`
   - boundary miss rate `0.1818`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

7. `clip_014` - Sonata No. 9, start `6s`
   - control display cluster F1 `0.3478`
   - control display note F1 `0.6207`
   - gap `0.2729`
   - exact cluster matches `4 / 9`
   - predicted clusters `14`
   - overclustered `3`, underclustered `1`, pitch conflicts `1`
   - boundary miss rate `0.0000`
   - buckets: `high_revision_slow_stabilization`, `note_value_offset_failure`

8. `clip_044` - Pagodas from Estampes No. 1, start `18s`
   - control display cluster F1 `0.3784`
   - control display note F1 `0.8000`
   - gap `0.4216`
   - exact cluster matches `7 / 19`
   - predicted clusters `18`
   - overclustered `7`, underclustered `2`, pitch conflicts `1`
   - boundary miss rate `0.2105`
   - buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`, `note_value_offset_failure`

9. `clip_022` - Au bord d'une source, start `30s`
   - control display cluster F1 `0.4310`
   - control display note F1 `0.8259`
   - gap `0.3949`
   - exact cluster matches `25 / 57`
   - predicted clusters `59`
   - overclustered `18`, underclustered `10`, pitch conflicts `4`
   - boundary miss rate `0.0652`
   - buckets: `runtime_only_win`, `boundary_miss_failure`, `high_revision_slow_stabilization`

10. `clip_019` - Concert Etude No. 2, "Gnomenreigen", S. 145/2, start `63s`

- control display cluster F1 `0.4474`
- control display note F1 `0.8392`
- gap `0.3919`
- exact cluster matches `34 / 78`
- predicted clusters `74`
- overclustered `25`, underclustered `11`, pitch conflicts `2`
- boundary miss rate `0.1273`
- buckets: `boundary_miss_failure`, `high_revision_slow_stabilization`

Why this set:

- `clip_031` is the clearest pure structure failure: very low cluster F1 with only a small boundary miss rate, so it should isolate onset-cluster / chord-grouping logic rather than raw seam recall.
- `clip_017`, `clip_024`, and `clip_006` combine severe structure collapse with meaningful boundary misses, so they are the best early probes if the goal is to improve both displayed structure and overall live correctness.
- The dominant pattern is overclustering, often accompanied by non-trivial underclustering and pitch-conflict counts.

## Second Attack Set: Largest Note-vs-Structure Gaps

These are the clips where note-level success looks much better than the displayed structure, so they are the best probes for the exact user-visible problem: the right notes are present somewhere, but they are snapped together wrong.

1. `clip_017` - Pour le piano (Complete), start `51s` - gap `0.6444`, note F1 `0.8173`, cluster F1 `0.1728`
2. `clip_006` - Pour le piano (Complete), start `57s` - gap `0.6329`, note F1 `0.8768`, cluster F1 `0.2439`
3. `clip_031` - Hungarian Rhapsody No. 9, start `18s` - gap `0.6204`, note F1 `0.7398`, cluster F1 `0.1194`
4. `clip_024` - Hungarian Rhapsody No. 9 in E-flat Major, S. 244, start `21s` - gap `0.5984`, note F1 `0.7889`, cluster F1 `0.1905`
5. `clip_035` - Estampes (Complete), start `81s` - gap `0.4492`, note F1 `0.7590`, cluster F1 `0.3099`
6. `clip_044` - Pagodas from Estampes No. 1, start `18s` - gap `0.4216`, note F1 `0.8000`, cluster F1 `0.3784`
7. `clip_012` - Entragete, Op. 63, start `75s` - gap `0.4204`, note F1 `0.7652`, cluster F1 `0.3448`
8. `clip_020` - Concert Etude No. 2 "Gnomenreigen", start `30s` - gap `0.4137`, note F1 `0.9008`, cluster F1 `0.4872`
9. `clip_015` - Concert Etude No. 2 "Gnomenreigen", start `21s` - gap `0.4012`, note F1 `0.9310`, cluster F1 `0.5298`
10. `clip_022` - Au bord d'une source, start `30s` - gap `0.3949`, note F1 `0.8259`, cluster F1 `0.4310`

Why this set:

- These clips best expose where note-level metrics are too optimistic about user-visible score quality.
- `clip_020` and `clip_015` are especially useful because their note F1 is already high, which means improvements there are likely to come from better grouping and structure rather than better pitch recall.

## Recommended Order

1. Start with `clip_017` and `clip_006`.
2. Use `clip_031` next to isolate pure overclustering behavior with less boundary noise.
3. Use `clip_024`, `clip_035`, and `clip_044` after that to check whether fixes generalize across denser polyphonic textures.
4. Use `clip_020` and `clip_015` to test whether grouping-specific fixes improve the displayed score even when note-level recall is already strong.
5. Keep the 20260529 shortlist for raw boundary-recall work; use this file for displayed-score correctness.
