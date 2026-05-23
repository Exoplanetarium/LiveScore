import logging
import math
import os
import time
from collections import defaultdict
from io import BytesIO
from typing import Dict, List


# ─── Timing Instrumentation ───────────────────────────────────────────────────
class TimingTracker:
    """Track timing of pipeline stages for real-time optimization."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_request_timings: Dict[str, float] = {}
        
    def start(self, stage: str):
        """Start timing a stage."""
        self.current_request_timings[f"{stage}_start"] = time.perf_counter()
        
    def stop(self, stage: str):
        """Stop timing a stage and record."""
        start_key = f"{stage}_start"
        if start_key in self.current_request_timings:
            elapsed_ms = (time.perf_counter() - self.current_request_timings[start_key]) * 1000
            self.timings[stage].append(elapsed_ms)
            del self.current_request_timings[start_key]
            return elapsed_ms
        return 0.0
    
    def get_request_summary(self) -> Dict[str, float]:
        """Get timings from the most recent request."""
        summary = {}
        for stage, times in self.timings.items():
            if times:
                summary[f"{stage}_ms"] = times[-1]
        return summary
    
    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        stats = {}
        for stage, times in self.timings.items():
            if times:
                import numpy as np
                arr = np.array(times)
                stats[stage] = {
                    "count": len(times),
                    "mean_ms": float(np.mean(arr)),
                    "std_ms": float(np.std(arr)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                }
        return stats
        
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.current_request_timings.clear()

# Global timing tracker
TIMER = TimingTracker()

# consistency between local and server
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
# Force the same OpenBLAS micro-kernel on both boxes (avoid AVX-512 vs AVX2 drift)
os.environ["OPENBLAS_CORETYPE"] = "HASWELL"   # works on AVX2/AVX-512 machines
os.environ["PYTHONHASHSEED"] = "0"

import tempfile

import numpy as np
import soundfile as sf
import uvicorn
from detect_note import (analyze_audio, analyze_audio_live_neural,
                         analyze_audio_optimized, read_wav,
                         second_pass_gap_fill)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.signal import resample_poly

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for streaming requests
class AudioStreamRequest(BaseModel):
    audio_data: List[float]
    sample_rate: int = 44100

# Streaming session models
class StreamReset(BaseModel):
    session_id: str

OVERLAP_SAMPLES = 4096  # ~93ms @ 44.1kHz, > n_fft for continuity
MIN_STREAM_ANALYSIS_SAMPLES = 16385  # torch.stft(center=True, reflect) needs input > 16384 for CQT
_stream_sessions: Dict[str, Dict] = {}

# A/B test flag: set to True to use optimized pipeline (4x faster)
# Once verified accurate, you can remove this and always use analyze_audio_optimized
USE_OPTIMIZED_PIPELINE = True

# Create FastAPI instance
app = FastAPI(
    title="LiveScore Audio Analysis API",
    description="Piano note detection and transcription API",
    version="1.0.0"
)

# Add CORS middleware to allow React Native frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper: make results JSON serializable (convert numpy types/arrays)
def make_json_serializable(obj, path="root"):
    try:
        if isinstance(obj, dict):
            return {k: make_json_serializable(v, f"{path}.{k}") for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item, f"{path}[{i}]") for i, item in enumerate(obj)]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            # Fallback: try to convert to string for unknown types
            print(f"[WARN] Unknown type at {path}: {type(obj)} = {repr(obj)[:100]}")
            try:
                return str(obj)
            except:
                return None
    except Exception as e:
        print(f"[ERROR] Serialization failed at {path}: {type(obj)} - {e}")
        raise

def load_audio_deterministic(path, target_sr=44100):
    # Read raw PCM deterministically
    y, sr = sf.read(path, dtype="float32", always_2d=True)  # shape (N, ch)
    y = y.mean(axis=1).astype(np.float32, copy=False)       # force mono by ourselves

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        y = resample_poly(y, up, down).astype(np.float32, copy=False)  # deterministic polyphase
    return y, target_sr

def _load_bytes_to_pcm(data: bytes, target_sr: int = 44100) -> np.ndarray:
    """Decode uploaded audio bytes to mono float32 PCM at target_sr."""
    with sf.SoundFile(BytesIO(data)) as f:
        y = f.read(always_2d=True, dtype='float32')  # (N, ch)
        sr = f.samplerate
    y = y.mean(axis=1).astype(np.float32, copy=False)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        y = resample_poly(y, up, down).astype(np.float32, copy=False)
    return y


def _append_stream_audio_chunk(sess: Dict, x_chunk: np.ndarray) -> None:
    history = sess.setdefault("full_audio_chunks", [])
    history.append(x_chunk.astype(np.float32, copy=False))


def _get_stream_session_audio(sess: Dict) -> np.ndarray | None:
    chunks = sess.get("full_audio_chunks") or []
    if not chunks:
        return None

    try:
        return np.concatenate(chunks).astype(np.float32, copy=False)
    except Exception:
        return None


def _run_classic_finalize_analysis(audio: np.ndarray, debug: bool = False) -> Dict:
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, 44100)
            temp_audio_path = tmp_file.name

        return analyze_audio(
            temp_audio_path,
            debug,
            True,
            True,
            True,
            "cuda",
        )
    except Exception:
        logger.warning(
            "Classic finalize pass failed; falling back to optimized analysis",
            exc_info=True,
        )
        return analyze_audio_optimized(audio, debug)
    finally:
        if temp_audio_path:
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass

def _get_session(session_id: str) -> Dict:
    s = _stream_sessions.get(session_id)
    if s is None:
        s = {
            "tail": np.zeros(0, dtype=np.float32),
            "sample_cursor": 0,  # samples processed (without overlap)
            "full_audio_chunks": [],
        }
        _stream_sessions[session_id] = s
    return s


def _clear_stream_session(session_id: str) -> None:
    _stream_sessions.pop(session_id, None)


def _event_time(event: Dict) -> float:
    try:
        return float(event.get("time_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_duration(event: Dict) -> float:
    duration = event.get("duration_seconds")
    if duration is not None:
        try:
            return max(0.0, float(duration))
        except (TypeError, ValueError):
            pass

    offset = event.get("offset_seconds")
    if offset is not None:
        try:
            return max(0.0, float(offset) - _event_time(event))
        except (TypeError, ValueError):
            pass

    return 0.0


def _event_confidence(event: Dict) -> float:
    try:
        return float(event.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_rank(event: Dict) -> float:
    rank = (_event_confidence(event) * 2.0) + min(_event_duration(event), 0.25)
    if str(event.get("method") or "") != "second_pass_soft":
        rank += 0.1
    return rank


def _note_pitch_sort_value(event: Dict) -> int:
    try:
        return int(event.get("midi_note", -999))
    except (TypeError, ValueError):
        return -999


def _note_has_chord_support(
    note: Dict,
    chords: List[Dict],
    time_tolerance_sec: float = 0.05,
) -> bool:
    midi = note.get("midi_note")
    if midi is None:
        return False

    try:
        midi_int = int(midi)
    except (TypeError, ValueError):
        return False

    note_time = _event_time(note)
    for chord in chords:
        if abs(_event_time(chord) - note_time) > time_tolerance_sec:
            continue

        try:
            chord_midis = [int(value) for value in (chord.get("midi_notes") or [])]
        except (TypeError, ValueError):
            continue

        if midi_int in chord_midis:
            return True

    return False


def _dedupe_note_events(
    notes: List[Dict],
    time_tolerance_sec: float = 0.05,
) -> List[Dict]:
    deduped: List[Dict] = []

    for note in sorted(
        notes or [],
        key=lambda event: (_event_time(event), _note_pitch_sort_value(event)),
    ):
        note_copy = dict(note)
        midi = note_copy.get("midi_note")

        try:
            midi_int = int(midi)
        except (TypeError, ValueError):
            deduped.append(note_copy)
            continue

        note_copy["midi_note"] = midi_int

        duplicate_idx = None
        for idx in range(len(deduped) - 1, -1, -1):
            existing = deduped[idx]
            if existing.get("midi_note") != midi_int:
                continue
            if abs(_event_time(existing) - _event_time(note_copy)) <= time_tolerance_sec:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            deduped.append(note_copy)
            continue

        if _event_rank(note_copy) > _event_rank(deduped[duplicate_idx]):
            deduped[duplicate_idx] = note_copy

    return sorted(deduped, key=_event_time)


def _chord_signature(chord: Dict):
    midi_notes = chord.get("midi_notes") or []
    try:
        normalized = tuple(sorted(int(value) for value in midi_notes))
    except (TypeError, ValueError):
        normalized = ()

    if normalized:
        return normalized

    return str(chord.get("label") or "")


def _dedupe_chord_events(
    chords: List[Dict],
    time_tolerance_sec: float = 0.06,
) -> List[Dict]:
    deduped: List[Dict] = []

    for chord in sorted(chords or [], key=_event_time):
        chord_copy = dict(chord)
        signature = _chord_signature(chord_copy)

        duplicate_idx = None
        for idx in range(len(deduped) - 1, -1, -1):
            existing = deduped[idx]
            if _chord_signature(existing) != signature:
                continue
            if abs(_event_time(existing) - _event_time(chord_copy)) <= time_tolerance_sec:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            deduped.append(chord_copy)
            continue

        if _event_rank(chord_copy) > _event_rank(deduped[duplicate_idx]):
            deduped[duplicate_idx] = chord_copy

    return sorted(deduped, key=_event_time)


LIVE_NOISE_FILTER_PROFILES = {
    "open": {
        "id": "open",
        "note_min_confidence": 0.44,
        "short_note_max_duration": 0.08,
        "short_note_min_confidence": 0.72,
        "unsupported_note_max_duration": 0.12,
        "unsupported_note_min_confidence": 0.58,
        "chord_min_confidence": 0.44,
        "short_chord_max_duration": 0.12,
        "short_chord_min_confidence": 0.62,
        "chunk_second_pass_soft_k": 1.28,
        "contention_soft_k": 1.38,
    },
    "balanced": {
        "id": "balanced",
        "note_min_confidence": 0.50,
        "short_note_max_duration": 0.08,
        "short_note_min_confidence": 0.78,
        "unsupported_note_max_duration": 0.12,
        "unsupported_note_min_confidence": 0.64,
        "chord_min_confidence": 0.50,
        "short_chord_max_duration": 0.12,
        "short_chord_min_confidence": 0.68,
        "chunk_second_pass_soft_k": 1.35,
        "contention_soft_k": 1.45,
    },
    "clean": {
        "id": "clean",
        "note_min_confidence": 0.58,
        "short_note_max_duration": 0.10,
        "short_note_min_confidence": 0.84,
        "unsupported_note_max_duration": 0.14,
        "unsupported_note_min_confidence": 0.72,
        "chord_min_confidence": 0.58,
        "short_chord_max_duration": 0.14,
        "short_chord_min_confidence": 0.76,
        "chunk_second_pass_soft_k": 1.52,
        "contention_soft_k": 1.62,
    },
}


def _resolve_live_noise_filter_profile(profile_name: str) -> Dict:
    profile_key = str(profile_name or "balanced").strip().lower()
    return LIVE_NOISE_FILTER_PROFILES.get(
        profile_key,
        LIVE_NOISE_FILTER_PROFILES["balanced"],
    )


def _is_tentative_live_note(note: Dict, chords: List[Dict], profile: Dict) -> bool:
    confidence = _event_confidence(note)
    duration = _event_duration(note)
    method = str(note.get("method") or "")

    if method == "second_pass_soft":
        return True

    if confidence < float(profile["note_min_confidence"]):
        return True

    if (
        duration <= float(profile["short_note_max_duration"])
        and confidence < float(profile["short_note_min_confidence"])
    ):
        return True

    if (
        duration <= float(profile["unsupported_note_max_duration"])
        and confidence < float(profile["unsupported_note_min_confidence"])
        and not _note_has_chord_support(note, chords)
    ):
        return True

    return False


def _is_tentative_live_chord(chord: Dict, profile: Dict) -> bool:
    confidence = _event_confidence(chord)
    duration = _event_duration(chord)
    method = str(chord.get("method") or "")

    if method == "second_pass_soft":
        return True

    if confidence < float(profile["chord_min_confidence"]):
        return True

    if (
        duration <= float(profile["short_chord_max_duration"])
        and confidence < float(profile["short_chord_min_confidence"])
    ):
        return True

    return False


def _apply_live_noise_gate_and_contention_pass(
    audio: np.ndarray,
    notes: List[Dict],
    chords: List[Dict],
    noise_profile: str = "balanced",
    debug: bool = False,
) -> Dict:
    profile = _resolve_live_noise_filter_profile(noise_profile)
    deduped_notes = _dedupe_note_events(notes)
    deduped_chords = _dedupe_chord_events(chords)

    stable_chords: List[Dict] = []
    tentative_chords: List[Dict] = []
    for chord in deduped_chords:
        if _is_tentative_live_chord(chord, profile):
            tentative_chords.append(chord)
        else:
            stable_chords.append(chord)

    stable_notes: List[Dict] = []
    tentative_notes: List[Dict] = []
    for note in deduped_notes:
        if _is_tentative_live_note(note, stable_chords, profile):
            tentative_notes.append(note)
        else:
            stable_notes.append(note)

    stats = {
        "deduped_notes": len(deduped_notes),
        "deduped_chords": len(deduped_chords),
        "tentative_notes": len(tentative_notes),
        "tentative_chords": len(tentative_chords),
        "contention_recovered_notes": 0,
        "contention_recovered_chords": 0,
        "profile": profile["id"],
    }

    if not tentative_notes and not tentative_chords:
        return {
            "notes": stable_notes,
            "chords": stable_chords,
            "stats": stats,
        }

    contention = second_pass_gap_fill(
        audio,
        stable_notes,
        stable_chords,
        min_gap_seconds=0.08,
        soft_K=float(profile["contention_soft_k"]),
        debug=debug,
    )

    recovered_notes = _dedupe_note_events(contention.get("notes") or [])
    recovered_chords = _dedupe_chord_events(contention.get("chords") or [])

    stats["contention_recovered_notes"] = len(recovered_notes)
    stats["contention_recovered_chords"] = len(recovered_chords)

    return {
        "notes": _dedupe_note_events([*stable_notes, *recovered_notes]),
        "chords": _dedupe_chord_events([*stable_chords, *recovered_chords]),
        "stats": stats,
    }


async def _analyze_uploaded_stream_chunk(
    session_id: str,
    data: bytes,
    debug: bool = False,
    noise_profile: str = "balanced",
    use_neural_live: bool = True,
    adaptive_onset_threshold: bool = True,
) -> Dict:
    """Decode one uploaded chunk, preserve overlap continuity, and return absolute-time events."""
    TIMER.start("chunk_total")
    TIMER.start("chunk_decode")

    x_chunk = _load_bytes_to_pcm(data, target_sr=44100)
    chunk_decode_ms = TIMER.stop("chunk_decode")
    chunk_duration_ms = len(x_chunk) / 44100 * 1000

    sess = _get_session(session_id)
    _append_stream_audio_chunk(sess, x_chunk)
    tail = sess["tail"]
    if tail.size > 0:
        x_full = np.concatenate([tail, x_chunk])
    else:
        x_full = x_chunk

    if x_full.size < MIN_STREAM_ANALYSIS_SAMPLES:
        sess["tail"] = x_full.astype(np.float32, copy=False)
        return {
            "onsets": [],
            "notes": [],
            "chords": [],
            "analysis_summary": {
                "total_onsets": 0,
                "total_notes": 0,
                "total_chords": 0,
                "duration_seconds": float(x_full.size / 44100.0),
                "sample_rate": 44100,
            },
            "stream_info": {
                "session_id": session_id,
                "chunk_samples": int(x_chunk.size),
                "overlap_samples": int(tail.size),
                "sample_cursor": int(sess["sample_cursor"]),
                "processed_sample_rate": 44100,
                "buffered_until_ready": True,
                "required_samples": MIN_STREAM_ANALYSIS_SAMPLES,
                "buffered_samples": int(x_full.size),
            },
            "_timing_ms": {
                "analysis_path": "buffering",
                "neural_requested": bool(use_neural_live),
                "adaptive_onset_threshold_requested": bool(adaptive_onset_threshold),
                "chunk_decode": round(chunk_decode_ms, 2),
                "chunk_inference": 0.0,
                "chunk_total": round(TIMER.stop("chunk_total"), 2),
                "chunk_audio_duration": round(chunk_duration_ms, 2),
                "real_time_factor": 0.0,
            },
        }

    overlap_sec = float(tail.size) / 44100.0

    TIMER.start("chunk_inference")
    analysis_path = "optimized" if USE_OPTIMIZED_PIPELINE else "split_pipeline"
    neural_error = None
    if use_neural_live:
        results = await run_in_threadpool(
            analyze_audio_live_neural,
            x_full,
            44100,
            debug,
            60,
            "cuda",
            adaptive_onset_threshold,
        )
        if results.get("error"):
            neural_error = str(results.get("error"))
            logger.warning(
                "Live neural path unavailable for session %s, falling back to %s: %s",
                session_id,
                analysis_path,
                neural_error,
            )
            analyzer = analyze_audio_optimized if USE_OPTIMIZED_PIPELINE else analyze_audio
            results = await run_in_threadpool(analyzer, x_full, debug)
            analysis_path = f"{analysis_path}_fallback"
        else:
            analysis_path = str(
                results.get("analysis_summary", {}).get("analysis_path")
                or "live_neural"
            )
    else:
        analyzer = analyze_audio_optimized if USE_OPTIMIZED_PIPELINE else analyze_audio
        results = await run_in_threadpool(analyzer, x_full, debug)
    chunk_inference_ms = TIMER.stop("chunk_inference")
    chunk_second_pass_ms = 0.0
    chunk_contention_ms = 0.0
    live_filter_stats = None
    live_noise_profile = _resolve_live_noise_filter_profile(noise_profile)
    results.setdefault("analysis_summary", {})["analysis_path"] = analysis_path

    # Live mode previously refined rhythm later but accepted first-pass pitch
    # decisions as-is. Reuse the existing soft-note gap fill here so missed
    # quiet notes inside the current chunk can enter the live session too.
    if analysis_path == "live_neural":
        results["notes"] = _dedupe_note_events(results.get("notes") or [])
        results["chords"] = _dedupe_chord_events(results.get("chords") or [])
    elif results.get("notes") or results.get("chords"):
        TIMER.start("chunk_second_pass")
        second_pass = await run_in_threadpool(
            second_pass_gap_fill,
            x_full,
            results.get("notes", []),
            results.get("chords", []),
            0.25,
            float(live_noise_profile["chunk_second_pass_soft_k"]),
            debug,
        )
        chunk_second_pass_ms = TIMER.stop("chunk_second_pass")

        extra_notes = second_pass.get("notes") or []
        extra_chords = second_pass.get("chords") or []
        if extra_notes:
            results["notes"] = sorted(
                [*(results.get("notes") or []), *extra_notes],
                key=lambda event: event.get("time_seconds", 0.0),
            )
        if extra_chords:
            results["chords"] = sorted(
                [*(results.get("chords") or []), *extra_chords],
                key=lambda event: event.get("time_seconds", 0.0),
            )

        results["notes"] = _dedupe_note_events(results.get("notes") or [])
        results["chords"] = _dedupe_chord_events(results.get("chords") or [])

        TIMER.start("chunk_contention")
        contention_result = await run_in_threadpool(
            _apply_live_noise_gate_and_contention_pass,
            x_full,
            results.get("notes") or [],
            results.get("chords") or [],
            noise_profile,
            debug,
        )
        chunk_contention_ms = TIMER.stop("chunk_contention")

        results["notes"] = contention_result.get("notes") or []
        results["chords"] = contention_result.get("chords") or []
        live_filter_stats = contention_result.get("stats")

    def _shift_and_filter_events(evts):
        out = []
        for event in evts or []:
            time_seconds = float(event.get("time_seconds", 0.0))
            if time_seconds < overlap_sec:
                continue

            absolute_time = (sess["sample_cursor"] / 44100.0) + (time_seconds - overlap_sec)
            shifted = dict(event)
            shifted["time_seconds"] = round(absolute_time, 6)
            out.append(shifted)
        return out

    results_filtered = {
        "onsets": _shift_and_filter_events(results.get("onsets")),
        "notes": _shift_and_filter_events(results.get("notes")),
        "chords": _shift_and_filter_events(results.get("chords")),
        "analysis_summary": results.get("analysis_summary", {}),
    }

    results_filtered["analysis_summary"] = {
        **results_filtered["analysis_summary"],
        "total_onsets": len(results_filtered["onsets"]),
        "total_notes": len(results_filtered["notes"]),
        "total_chords": len(results_filtered["chords"]),
    }

    sess["sample_cursor"] += int(x_chunk.size)

    take = min(OVERLAP_SAMPLES, x_full.size)
    sess["tail"] = x_full[-take:].astype(np.float32, copy=False)

    chunk_total_ms = TIMER.stop("chunk_total")
    real_time_factor = chunk_total_ms / chunk_duration_ms if chunk_duration_ms > 0 else 0

    results_filtered["stream_info"] = {
        "session_id": session_id,
        "chunk_samples": int(x_chunk.size),
        "overlap_samples": int(tail.size),
        "sample_cursor": int(sess["sample_cursor"]),
        "processed_sample_rate": 44100,
    }
    if neural_error is not None:
        results_filtered["stream_info"]["neural_error"] = neural_error
    if live_filter_stats is not None:
        results_filtered["stream_info"]["live_filter"] = live_filter_stats

    analyzer_timings = results.get("_timing_ms") or {}
    results_filtered["_timing_ms"] = {
        "analysis_path": analysis_path,
        "neural_requested": bool(use_neural_live),
        "neural_error": neural_error,
        **analyzer_timings,
        "chunk_decode": round(chunk_decode_ms, 2),
        "chunk_inference": round(chunk_inference_ms, 2),
        "chunk_second_pass": round(chunk_second_pass_ms, 2),
        "chunk_contention": round(chunk_contention_ms, 2),
        "chunk_total": round(chunk_total_ms, 2),
        "chunk_audio_duration": round(chunk_duration_ms, 2),
        "real_time_factor": round(real_time_factor, 3),
    }

    neural_total_ms = analyzer_timings.get("neural_total")
    neural_model_inference_ms = analyzer_timings.get("neural_model_inference")
    neural_model_name = results.get("analysis_summary", {}).get("neural_model")
    neural_suffix = ""
    if analysis_path == "live_neural":
        neural_parts = []
        if neural_model_name:
            neural_parts.append(f"model={neural_model_name}")
        if isinstance(neural_total_ms, (int, float)):
            neural_parts.append(f"neural_total={neural_total_ms:.1f}ms")
        if isinstance(neural_model_inference_ms, (int, float)):
            neural_parts.append(f"model_inference={neural_model_inference_ms:.1f}ms")
        if neural_parts:
            neural_suffix = " | " + ", ".join(neural_parts)
    elif neural_error:
        neural_suffix = f" | neural_error={neural_error}"

    print(
        f"[TIMING] stream_chunk[{session_id}]({analysis_path}): decode={chunk_decode_ms:.1f}ms, "
        f"inference={chunk_inference_ms:.1f}ms, second_pass={chunk_second_pass_ms:.1f}ms, contention={chunk_contention_ms:.1f}ms, TOTAL={chunk_total_ms:.1f}ms | "
        f"audio={chunk_duration_ms:.0f}ms, RTF={real_time_factor:.2f}x{neural_suffix}"
    )

    return results_filtered

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "LiveScore Audio Analysis API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "LiveScore Audio Analysis API",
        "version": "1.0.0"
    }

@app.get("/timing-stats")
async def timing_stats():
    """Get timing statistics for performance analysis."""
    return {
        "stats": TIMER.get_stats(),
        "last_request": TIMER.get_request_summary(),
    }

@app.post("/timing-reset")
async def timing_reset():
    """Reset timing statistics."""
    TIMER.reset()
    return {"status": "reset"}

@app.get("/warmup")
async def warmup():
    """
    Warmup endpoint - call this when app opens to pre-load models.
    This ensures the first recording request is fast.
    """
    try:
        from gpu_ops import (get_gpu_mel_baseline_transcriber,
                             get_gpu_rhythm_model, get_gpu_transcriber,
                             get_gpu_transformer_model)

        ensemble = get_gpu_mel_baseline_transcriber()
        rhythm = get_gpu_rhythm_model()
        transformer = get_gpu_transformer_model()
        transcriber = get_gpu_transcriber()
        warmup_audio = np.zeros(MIN_STREAM_ANALYSIS_SAMPLES, dtype=np.float32)
        analyzer = analyze_audio_optimized if USE_OPTIMIZED_PIPELINE else analyze_audio
        await run_in_threadpool(analyzer, warmup_audio, False)
        
        return {
            "status": "warm",
            "ensemble_model": ensemble is not None and ensemble.initialized,
            "rhythm_model": rhythm is not None,
            "transformer_model": transformer is not None and transformer.initialized,
            "transcriber_model": transcriber is not None and transcriber.initialized,
            "analysis_pipeline": "primed",
        }
    except Exception as e:
        return {
            "status": "warmup_failed",
            "error": str(e)
        }

@app.post("/analyze")
async def analyze_audio_file(
    file: UploadFile = File(...),
    debug: bool = False,
    use_neural: bool = True,  # Default to neural for higher accuracy
    device: str = "cuda"  # 'cuda' for GPU, 'cpu' for CPU
):
    """
    Analyze an uploaded audio file and return detected notes and onsets.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        debug: Whether to include debug information in response
        use_neural: If True, use neural network transcription (higher accuracy, default: True)
        device: Device for neural inference ('cuda' for GPU, 'cpu' for CPU)
        
    Returns:
        JSON with detected notes, chords, onsets, and analysis metadata
        
    Neural output format:
        - notes: [{time_seconds, midi_note, note_name, frequency_hz, method, confidence, 
                   offset_seconds, duration_seconds, hand (bass/treble)}]
        - chords: [{time_seconds, midi_notes, note_names, root, octave, label, inversion,
                    method, confidence, offset_seconds, duration_seconds, hand (bass/treble)}]
    """
    
    # Validate file type
    allowed_types = ['audio/wav', 'audio/wave', 'audio/x-wav', 'audio/aac', 'audio/mpeg', 'audio/mp3']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Supported types: {allowed_types}"
        )
    
    try:
        TIMER.start("total_request")
        TIMER.start("file_read")
        
        # Read the uploaded file
        data = await file.read()
        if len(data) > 100*1024*1024:
            raise HTTPException(413, "File too large")
            
        file_read_ms = TIMER.stop("file_read")
        
        #! DEBUG
        import binascii
        import tempfile

        # 1) Log filename, content-type, and size
        # print(f"[DEBUG] upload filename={file.filename!r}, content_type={file.content_type!r}, size={len(data)} bytes")
        # 2) Log the first few bytes in hex (should start with 'RIFF')
        # hex_header = binascii.hexlify(data[:12]).decode('ascii', errors='ignore')
        # print(f"[DEBUG] first 12 bytes (hex): {hex_header}")
        # 3) Write a copy to a temp file so you can download/inspect it
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav")
        tmp.write(data)
        tmp.flush()
        # print(f"[DEBUG] data upload written to {tmp.name}")
        #! END DEBUG

        tmp.close()
        
        TIMER.start("audio_decode")
        audio = read_wav(tmp.name)
        decode_ms = TIMER.stop("audio_decode")
        
        # Keep temp file path for neural transcription
        temp_audio_path = tmp.name

        try:
            # Analyze the audio in a threadpool (blocking CPU work)
            # For neural transcription, pass the file path; for traditional, pass the array
            TIMER.start("inference")
            if use_neural:
                results = await run_in_threadpool(
                    analyze_audio, temp_audio_path, debug, 
                    True, True, True, device  # use_split, independent_hands, use_neural, device
                )
            else:
                results = await run_in_threadpool(analyze_audio, audio, debug)
            inference_ms = TIMER.stop("inference")
            
            # Clean up temp file after analysis
            try:
                os.unlink(temp_audio_path)
            except:
                pass

            # Add metadata about the uploaded file
            results["file_info"] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "processed_sample_rate": 44100,
                "channels": 1 if (isinstance(audio, np.ndarray) and audio.ndim == 1) else (audio.shape[1] if isinstance(audio, np.ndarray) and audio.ndim > 1 else 1)
            }

            # Ensure JSON serializable
            print("[DEBUG] Building JSON response...")
            TIMER.start("serialization")
            try:
                clean_results = make_json_serializable(results)
                print(f"[DEBUG] JSON serialization successful, returning response")
            except Exception as ser_err:
                import traceback
                print(f"[ERROR] JSON serialization failed: {ser_err}")
                traceback.print_exc()
                raise
            serialization_ms = TIMER.stop("serialization")
            total_ms = TIMER.stop("total_request")
            
            # Add timing info to response
            clean_results["_timing_ms"] = {
                "file_read": round(file_read_ms, 2),
                "audio_decode": round(decode_ms, 2),
                "inference": round(inference_ms, 2),
                "serialization": round(serialization_ms, 2),
                "total": round(total_ms, 2),
            }
            
            print(f"[TIMING] /analyze: file_read={file_read_ms:.1f}ms, decode={decode_ms:.1f}ms, inference={inference_ms:.1f}ms, serial={serialization_ms:.1f}ms, TOTAL={total_ms:.1f}ms")

            return JSONResponse(content=clean_results)

        except Exception as e:
            import traceback
            print(f"[ERROR] Audio analysis failed: {str(e)}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Audio analysis failed: {str(e)}"
            )
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )

class SecondPassRequest(BaseModel):
    """Request model for second pass gap-fill detection."""
    notes: List[dict]
    chords: List[dict]
    min_gap_seconds: float = 0.25
    soft_k: float = 1.2


@app.post("/analyze-second-pass")
async def analyze_second_pass(
    file: UploadFile = File(...),
    notes: str = Form(...),  # JSON string of existing notes
    chords: str = Form(...),  # JSON string of existing chords
    min_gap_seconds: float = Form(0.25),
    soft_k: float = Form(1.2),
    debug: bool = Form(False),
):
    """
    Second pass detection to find soft notes missed in gaps.
    
    Args:
        file: Audio file (same as original analysis)
        notes: JSON string of existing notes from first pass
        chords: JSON string of existing chords from first pass
        min_gap_seconds: Minimum gap duration to search (default: 0.25s)
        soft_k: Sensitivity threshold (lower = more sensitive, default: 1.2)
        debug: Whether to include debug information
        
    Returns:
        JSON with NEW notes and chords found in gaps
    """
    import json
    
    try:
        # Parse existing notes/chords
        existing_notes = json.loads(notes)
        existing_chords = json.loads(chords)
        
        # Read the uploaded file
        data = await file.read()
        if len(data) > 100*1024*1024:
            raise HTTPException(413, "File too large")
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data)
            tmp.flush()
            temp_path = tmp.name
        
        try:
            # Run second pass detection
            results = await run_in_threadpool(
                second_pass_gap_fill,
                temp_path,
                existing_notes,
                existing_chords,
                min_gap_seconds,
                soft_k,
                debug
            )
            
            # Clean up
            os.unlink(temp_path)
            
            clean_results = make_json_serializable(results)
            return JSONResponse(content=clean_results)
            
        except Exception as e:
            os.unlink(temp_path)
            raise
            
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in notes or chords: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Second pass analysis failed: {str(e)}")


@app.post("/analyze-raw")
async def analyze_raw_audio(
    sample_rate: int = 44100,
    debug: bool = False,
    audio_data: UploadFile = File(...)
):
    """
    Analyze raw audio data (for recorded audio from React Native).
    
    Args:
        audio_data: Raw audio file
        sample_rate: Sample rate of the audio data
        debug: Whether to include debug information
        
    Returns:
        JSON with analysis results
    """
    try:
        # Read the uploaded raw audio data
        content = await audio_data.read()
        
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio = np.frombuffer(content, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0  # Normalize to [-1, 1]
        
        # Resample if needed
        if sample_rate != 44100:
            from scipy.signal import resample
            target_length = int(len(audio) * 44100 / sample_rate)
            audio = resample(audio, target_length)
        
        # Analyze the audio in a threadpool
        results = await run_in_threadpool(analyze_audio, audio, debug)

        # Add metadata
        results["file_info"] = {
            "filename": "recorded_audio",
            "content_type": "audio/raw",
            "original_sample_rate": int(sample_rate) if hasattr(sample_rate, '__int__') else sample_rate,
            "processed_sample_rate": 44100,
            "channels": 1
        }

        clean_results = make_json_serializable(results)

        return JSONResponse(content=clean_results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Raw audio analysis failed: {str(e)}"
        )

@app.post("/analyze-stream")
async def analyze_audio_stream(request: AudioStreamRequest):
    """
    Analyze streaming audio data for real-time piano detection
    """
    try:
        logger.info(f"Received audio stream: {len(request.audio_data)} samples at {request.sample_rate}Hz")
        
        # Validate input
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        if len(request.audio_data) < 1024:
            raise HTTPException(status_code=400, detail="Audio data too short for analysis")
        
        # Convert to numpy array
        audio_array = np.array(request.audio_data, dtype=np.float32)
        
        # Ensure audio is in correct range [-1, 1]
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        logger.info(f"Processing audio array: shape={audio_array.shape}, range=[{np.min(audio_array):.3f}, {np.max(audio_array):.3f}]")
        
        # Additional audio analysis for debugging
        duration_sec = len(audio_array) / request.sample_rate
        logger.info(f"Audio duration: {duration_sec:.3f} seconds")
        
        # Check for silence
        rms_energy = np.sqrt(np.mean(audio_array**2))
        logger.info(f"RMS energy: {rms_energy:.6f}")
        
        if rms_energy < 0.001:
            logger.warning("Audio appears to be very quiet or silent")
        
        # Check for clipping
        clipped_samples = np.sum(np.abs(audio_array) >= 0.99)
        if clipped_samples > 0:
            logger.warning(f"Audio may be clipped: {clipped_samples} samples at max level")
        
        logger.info(f"Original audio stats: samples={len(audio_array)}, sample_rate={request.sample_rate}, duration={duration_sec:.3f}s")

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write audio data as WAV file
            sf.write(tmp_file.name, audio_array, request.sample_rate)
            tmp_file_path = tmp_file.name
        
        try:
            # Use read_wav() function for consistent processing
            logger.info(f"Processing audio through read_wav() function")
            processed_audio = read_wav(tmp_file_path)
            logger.info(f"Audio processed: shape={processed_audio.shape}, range=[{np.min(processed_audio):.3f}, {np.max(processed_audio):.3f}]")
            
            # Log comparison between original and processed audio
            logger.info(f"Audio processing comparison:")
            logger.info(f"  Original: {len(audio_array)} samples, range=[{np.min(audio_array):.3f}, {np.max(audio_array):.3f}]")
            logger.info(f"  Processed: {len(processed_audio)} samples, range=[{np.min(processed_audio):.3f}, {np.max(processed_audio):.3f}]")
            logger.info(f"  Sample rate conversion: {request.sample_rate}Hz -> 44100Hz")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Analyze the processed audio using our detection system
            results = await run_in_threadpool(analyze_audio, processed_audio, False)
        except Exception as e:
            # Clean up temporary file in case of error
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            raise e
        
        # Detailed logging of what was detected
        detected_notes = results.get('notes', [])
        detected_chords = results.get('chords', [])
        detected_onsets = results.get('onsets', [])
        
        logger.info(f"Analysis complete: {len(detected_onsets)} onsets, {len(detected_notes)} notes, {len(detected_chords)} chords")
        
        # Log individual detections for debugging
        if detected_notes:
            logger.info("Detected notes:")
            for note in detected_notes:
                logger.info(f"  {note['time_seconds']:.3f}s: {note['note_name']} ({note['method']}, conf={note['confidence']:.2f})")
        
        if detected_chords:
            logger.info("Detected chords:")
            for chord in detected_chords:
                logger.info(f"  {chord['time_seconds']:.3f}s: {chord['label']} {chord['inversion']} (conf={chord['confidence']:.2f})")
        
        if not detected_notes and not detected_chords:
            logger.warning("No notes or chords detected in this audio chunk")

        # Clean results for JSON serialization
        clean_results = make_json_serializable(results)

        # Add streaming metadata
        clean_results["stream_info"] = {
            "samples_received": len(request.audio_data),
            "original_sample_rate": request.sample_rate,
            "processed_sample_rate": 44100,  # read_wav() always outputs 44.1kHz
            "duration_seconds": len(request.audio_data) / request.sample_rate,
            "analysis_type": "real_time_stream",
            "processing_method": "read_wav_function"
        }

        return JSONResponse(content=clean_results)

    except Exception as e:
        logger.error(f"Streaming analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming analysis failed: {str(e)}")

@app.post("/stream/reset")
async def stream_reset(payload: StreamReset):
    """Reset a streaming session so state (tail, cursors) clears."""
    sid = payload.session_id
    _clear_stream_session(sid)
    return {"status": "reset", "session_id": sid}

@app.post("/stream/chunk")
async def stream_chunk(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    debug: bool = False,
):
    """Analyze one recorded chunk while preserving overlap/state per session.
    Client should send each chunk file with the same session_id for a recording session.
    """
    try:
        data = await file.read()
        results_filtered = await _analyze_uploaded_stream_chunk(session_id, data, debug)
        return JSONResponse(content=make_json_serializable(results_filtered))

    except Exception as e:
        logger.error(f"stream_chunk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"stream_chunk failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Live Transcription Endpoints (Low-latency with deferred refinement)
# ─────────────────────────────────────────────────────────────────────────────

from live_rhythm import LiveTranscriptionSession, cleanup_stale_sessions
from live_rhythm import delete_session as delete_live_session
from live_rhythm import get_or_create_session as get_live_session


class LiveSessionCreate(BaseModel):
    session_id: str
    initial_bpm: float = 120.0


class LiveNotesInput(BaseModel):
    session_id: str
    notes: List[dict]
    chords: List[dict] = []


class LiveSessionQuery(BaseModel):
    session_id: str


@app.post("/live/session/create")
async def live_session_create(req: LiveSessionCreate):
    """
    Create a new live transcription session.
    
    Returns session info with initial tempo settings.
    """
    session = get_live_session(req.session_id)
    if req.initial_bpm:
        session.tempo_tracker.current_bpm = req.initial_bpm
        session.tempo_tracker.initial_bpm = req.initial_bpm
    
    return {
        "status": "created",
        "session_id": req.session_id,
        "bpm": session.tempo_tracker.current_bpm,
    }


@app.post("/live/session/reset")
async def live_session_reset(req: LiveSessionQuery):
    """Reset an existing live session (clears all notes, resets tempo)."""
    session = get_live_session(req.session_id)
    session.reset()
    _clear_stream_session(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


@app.post("/live/session/delete")
async def live_session_delete(req: LiveSessionQuery):
    """Delete a live session."""
    deleted = delete_live_session(req.session_id)
    _clear_stream_session(req.session_id)
    return {"status": "deleted" if deleted else "not_found", "session_id": req.session_id}


@app.post("/live/audio-chunk")
async def live_process_audio_chunk(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    noise_profile: str = Form("balanced"),
    use_neural_live: bool = Form(True),
    adaptive_onset_threshold: bool = Form(True),
    debug: bool = False,
):
    """
    Analyze an uploaded chunk and immediately route detections into the live rhythm session.

    This is the backend bridge between overlap-aware chunk audio analysis and the
    two-stage live quantizer. Frontends can call one endpoint per chunk and get
    both immediate notation and deferred-refinement state updates.
    """
    try:
        data = await file.read()
        chunk_result = await _analyze_uploaded_stream_chunk(
            session_id,
            data,
            debug,
            noise_profile,
            use_neural_live,
            adaptive_onset_threshold,
        )

        session = get_live_session(session_id)
        live_result = await run_in_threadpool(
            session.process_notes,
            chunk_result.get("notes", []),
            chunk_result.get("chords", []),
        )

        response = {
            "onsets": chunk_result.get("onsets", []),
            "notes": live_result.get("coarse_notes", []),
            "chords": live_result.get("coarse_chords", []),
            "detected_notes": chunk_result.get("notes", []),
            "detected_chords": chunk_result.get("chords", []),
            "analysis_summary": chunk_result.get("analysis_summary", {}),
            "analysis_path": chunk_result.get("analysis_summary", {}).get("analysis_path"),
            "bpm": live_result.get("bpm"),
            "bpm_confidence": live_result.get("bpm_confidence"),
            "beat_grid": live_result.get("beat_grid", session.grid_payload()),
            "needs_refresh": live_result.get("needs_refresh", False),
            "refined_notes": live_result.get("refined_notes") or [],
            "refinement_version": live_result.get("refinement_version", 0),
            "stream_info": chunk_result.get("stream_info", {}),
            "_timing_ms": chunk_result.get("_timing_ms", {}),
            "next_refinement_poll_ms": session.get_next_refinement_delay_ms(),
        }

        response["neural_error"] = (
            response.get("_timing_ms", {}).get("neural_error")
            or response.get("stream_info", {}).get("neural_error")
        )
        response["fallback_reason"] = response["neural_error"]

        if response["needs_refresh"]:
            response["all_notes"] = session.get_all_notes()
            response["all_chords"] = session.coarse_chords

        return JSONResponse(content=make_json_serializable(response))
    except Exception as e:
        logger.error(f"live_process_audio_chunk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"live audio chunk failed: {str(e)}")


@app.post("/live/process")
async def live_process_notes(req: LiveNotesInput):
    """
    Process notes in live mode with two-stage quantization.
    
    Stage 1 (immediate): Coarse quantization for instant display
    Stage 2 (deferred): Refined quantization ~1s later
    
    Returns:
        - coarse_notes: Immediately quantized notes (display these now)
        - bpm: Current tempo estimate
        - bpm_confidence: How confident the tempo estimate is (0-1)
        - needs_refresh: If true, frontend should reload the score
        - refined_notes: Notes that were refined (if any)
        - refinement_version: Version number for cache invalidation
    """
    session = get_live_session(req.session_id)
    
    result = await run_in_threadpool(
        session.process_notes, req.notes, req.chords
    )
    result["next_refinement_poll_ms"] = session.get_next_refinement_delay_ms()
    if result.get("needs_refresh"):
        result["all_notes"] = session.get_all_notes()
        result["all_chords"] = session.coarse_chords
    
    return JSONResponse(content=make_json_serializable(result))


@app.post("/live/check-refinement")
async def live_check_refinement(req: LiveSessionQuery):
    """
    Check if any notes are ready for refinement.
    
    Call this periodically (every 500ms-1s) to get refined notes.
    
    Returns:
        - needs_refresh: If true, refined notes are available
        - refined_notes: List of refined notes (if any)
        - refinement_version: Current version number
        - all_notes: Complete list of notes with best available quantization
    """
    import time
    session = get_live_session(req.session_id)
    
    bpm, confidence = session.get_current_bpm()
    current_time = time.time()
    grid = session.beat_grid
    refined = session.refinement_state.check_refinement(current_time, bpm, grid=grid)

    payload = {
        "needs_refresh": refined is not None and len(refined) > 0,
        "refined_notes": refined or [],
        "refinement_version": session.refinement_state.get_refinement_version(),
        "bpm": bpm,
        "bpm_confidence": confidence,
        "beat_grid": session.grid_payload(),
        "next_refinement_poll_ms": session.get_next_refinement_delay_ms(current_time),
    }
    if payload["needs_refresh"]:
        payload["all_notes"] = session.get_all_notes()
        payload["all_chords"] = session.coarse_chords

    return JSONResponse(content=make_json_serializable(payload))


@app.post("/live/get-all-notes")
async def live_get_all_notes(req: LiveSessionQuery):
    """
    Get all notes with best available quantization.
    
    Use this when the frontend needs to refresh the full score.
    """
    session = get_live_session(req.session_id)
    all_notes = session.get_all_notes()
    bpm, confidence = session.get_current_bpm()
    
    return JSONResponse(content=make_json_serializable({
        "notes": all_notes,
        "chords": session.coarse_chords,
        "bpm": bpm,
        "bpm_confidence": confidence,
        "beat_grid": session.grid_payload(),
        "refinement_version": session.refinement_state.get_refinement_version(),
    }))


@app.post("/live/finalize")
async def live_finalize_session(req: LiveSessionQuery):
    """
    Finalize a live session (e.g., when recording stops).
    
    Forces refinement of all pending notes and returns final results.
    Use this endpoint when the user stops recording.
    """
    session = get_live_session(req.session_id)

    stream_session = _get_session(req.session_id)
    full_audio = _get_stream_session_audio(stream_session)

    if full_audio is not None and full_audio.size >= MIN_STREAM_ANALYSIS_SAMPLES:
        final_results = await run_in_threadpool(
            _run_classic_finalize_analysis,
            full_audio,
        )
        analysis_summary = final_results.get("analysis_summary", {}) or {}
        final_notes = final_results.get("notes") or []
        final_chords = final_results.get("chords") or []
        final_onsets = final_results.get("onsets") or []

        return JSONResponse(content=make_json_serializable({
            "status": "finalized",
            "finalization_mode": "classic_full_pass",
            "notes": final_notes,
            "chords": final_chords,
            "onsets": final_onsets,
            "bpm": analysis_summary.get("detected_bpm"),
            "bpm_confidence": analysis_summary.get("tempo_confidence"),
            "analysis_summary": analysis_summary,
            "beat_grid": session.grid_payload(),
            "total_notes": len(final_notes),
            "total_chords": len(final_chords),
            "refinement_version": session.refinement_state.get_refinement_version(),
        }))

    # Force refinement of all pending notes when a classic full pass is unavailable.
    await run_in_threadpool(session.force_refinement)

    all_notes = session.get_all_notes()
    bpm, confidence = session.get_current_bpm()

    return JSONResponse(content=make_json_serializable({
        "status": "finalized",
        "finalization_mode": "live_refinement_only",
        "notes": all_notes,
        "chords": session.coarse_chords,
        "onsets": [],
        "bpm": bpm,
        "bpm_confidence": confidence,
        "beat_grid": session.grid_payload(),
        "total_notes": len(all_notes),
        "total_chords": len(session.coarse_chords),
        "refinement_version": session.refinement_state.get_refinement_version(),
    }))


@app.get("/live/cleanup")
async def live_cleanup_sessions(max_age_seconds: float = 3600.0):
    """Remove stale live sessions."""
    removed = cleanup_stale_sessions(max_age_seconds)
    return {"status": "cleaned", "removed_sessions": removed}


if __name__ == "__main__":
    # Get port from environment variable (Railway sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    
