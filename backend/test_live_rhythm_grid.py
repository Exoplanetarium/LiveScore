from live_rhythm import BeatGrid, apply_window_decode, quantize_coarse


def test_coarse_start_beat_is_absolute_from_session_start():
    grid = BeatGrid(phase=5.0, period=0.5, subdivision=8, anchored=True)
    note = {
        "time_seconds": 9.0,
        "duration_seconds": 0.5,
        "midi_note": 60,
    }

    quantize_coarse(note, bpm=120.0, grid=grid)

    assert note["start_grid_idx"] == 64
    assert note["start_beat"] == 18.0
    assert note["start_beat"] * grid.period == note["cluster_metric_time_seconds"]


def test_refined_start_beat_is_absolute_from_session_start():
    grid = BeatGrid(phase=5.0, period=0.5, subdivision=8, anchored=True)
    notes = [
        {"time_seconds": 5.0, "duration_seconds": 0.5, "midi_note": 60},
        {"time_seconds": 5.5, "duration_seconds": 0.5, "midi_note": 62},
    ]
    decode = {
        "grid": grid,
        "subdivision": grid.subdivision,
        "indices": [0, 8],
        "durations_units": [8, 8],
    }

    apply_window_decode(notes, decode)

    assert notes[0]["start_beat"] == 10.0
    assert notes[1]["start_beat"] == 11.0
    assert notes[0]["start_beat"] * grid.period == notes[0]["cluster_metric_time_seconds"]
    assert notes[1]["start_beat"] * grid.period == notes[1]["cluster_metric_time_seconds"]
