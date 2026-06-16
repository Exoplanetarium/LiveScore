# Gold-12 Reference Prep

Create corrected reference MusicXML files for these 12 benchmark excerpts.

Folders:
- `audio_wav/`: exact audio excerpts used in the 48-clip run.
- `midi_excerpt/`: matching MIDI excerpts, sliced to start at 0s.
- `reference_musicxml/`: save final corrected refs here as `clip_001.musicxml` through `clip_012.musicxml`.

After creating refs, run:
```powershell
.\backend\env\Scripts\python.exe .\backend\test_experiment.py `
  --benchmark-manifest .\backend\benchmark_artifacts\gold12_reference_prep_20260612\benchmark_manifest_gold12.json `
  --reference-musicxml-dir .\backend\benchmark_artifacts\gold12_reference_prep_20260612\reference_musicxml `
  --require-reference-musicxml `
  --run-retro-correction false `
  --output-json .\backend\benchmark_artifacts\gold12_reference_prep_20260612\gold12_results.json
```

| clip | title | source time | audio | midi | save ref as | midi notes |
| --- | --- | --- | --- | --- | --- | --- |
| clip_001 | Sonata in A Major, K. 208 | 0.00-6.00s | `clip_001__Sonata in A Major_ K. 208.wav` | `clip_001__Sonata in A Major_ K. 208.mid` | `clip_001.musicxml` | 9 |
| clip_002 | Two Sonatas | 21.00-27.00s | `clip_002__Two Sonatas.wav` | `clip_002__Two Sonatas.mid` | `clip_002.musicxml` | 96 |
| clip_003 | Sonata in D Minor, K. 213 | 9.00-15.00s | `clip_003__Sonata in D Minor_ K. 213.wav` | `clip_003__Sonata in D Minor_ K. 213.mid` | `clip_003.musicxml` | 20 |
| clip_004 | Rem. of Don Juan | 15.00-21.00s | `clip_004__Rem. of Don Juan.wav` | `clip_004__Rem. of Don Juan.mid` | `clip_004.musicxml` | 42 |
| clip_005 | Transcendental Etude No. 11 "Harmonies du Soir" | 36.00-42.00s | `clip_005__Transcendental Etude No. 11 _Harmonies du Soir_.wav` | `clip_005__Transcendental Etude No. 11 _Harmonies du Soir_.mid` | `clip_005.musicxml` | 12 |
| clip_006 | Concert Etude No. 2 "Gnomenreigen" | 29.00-35.00s | `clip_006__Concert Etude No. 2 _Gnomenreigen_.wav` | `clip_006__Concert Etude No. 2 _Gnomenreigen_.mid` | `clip_006.musicxml` | 188 |
| clip_007 | Transcendental Etude No. 11 "Harmonies du Soir" | 23.00-29.00s | `clip_007__Transcendental Etude No. 11 _Harmonies du Soir_.wav` | `clip_007__Transcendental Etude No. 11 _Harmonies du Soir_.mid` | `clip_007.musicxml` | 12 |
| clip_008 | Concert Etude No. 2, "Gnomenreigen", S. 145/2 | 23.00-29.00s | `clip_008__Concert Etude No. 2_ _Gnomenreigen__ S. 145_2.wav` | `clip_008__Concert Etude No. 2_ _Gnomenreigen__ S. 145_2.mid` | `clip_008.musicxml` | 161 |
| clip_009 | Etude "Pour les accords" | 2.00-8.00s | `clip_009__Etude _Pour les accords_.wav` | `clip_009__Etude _Pour les accords_.mid` | `clip_009.musicxml` | 130 |
| clip_010 | Sonata No. 9, Op. 68, "Black Mass" | 10.00-16.00s | `clip_010__Sonata No. 9_ Op. 68_ _Black Mass_.wav` | `clip_010__Sonata No. 9_ Op. 68_ _Black Mass_.mid` | `clip_010.musicxml` | 16 |
| clip_011 | Réminiscences de Don Juan, S.418 | 37.00-43.00s | `clip_011__Réminiscences de Don Juan_ S.418.wav` | `clip_011__Réminiscences de Don Juan_ S.418.mid` | `clip_011.musicxml` | 63 |
| clip_012 | Sonata No. 9 | 0.00-6.00s | `clip_012__Sonata No. 9.wav` | `clip_012__Sonata No. 9.mid` | `clip_012.musicxml` | 18 |
