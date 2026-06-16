import asyncio
import base64
import logging
import math
import os
import threading
import time
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple


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


def _get_timed_display_state(session) -> Tuple[Dict, float]:
    TIMER.start("display_state")
    display_state = session.get_display_state()
    display_state_ms = TIMER.stop("display_state")
    return display_state or {}, display_state_ms

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
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
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

OVERLAP_SAMPLES = 4096  # ~93ms @ 44.1kHz, > n_fft for continuity. Used as the
                        # boundary-recovery band (see _shift_overlap_events_with_recent_dedupe).
# Inference left-context: the neural transcriber is markedly more accurate (both
# precision and recall) when it sees more than the bare 0.6s chunk. We prepend a
# longer history tail purely for model calibration, but still only emit/recover
# notes in the newest chunk plus the small OVERLAP_SAMPLES recovery band; notes
# deeper in the context were already emitted by earlier chunks and are dropped.
# Tunable via LIVE_CONTEXT_SEC for benchmarking; 0 disables (legacy 93ms behavior).
try:
    LIVE_CONTEXT_SEC = float(os.environ.get("LIVE_CONTEXT_SEC", "2.4"))
except (TypeError, ValueError):
    LIVE_CONTEXT_SEC = 2.4
CONTEXT_SAMPLES = max(OVERLAP_SAMPLES, int(LIVE_CONTEXT_SEC * 44100))
MIN_STREAM_ANALYSIS_SAMPLES = 16385  # torch.stft(center=True, reflect) needs input > 16384 for CQT
CHUNK_END_GUARD_SEC = 0.025
CHUNK_END_MICRO_EVENT_MAX_DURATION_SEC = 0.045
NOTE_EVENT_DEDUPE_TOLERANCE_SEC = 0.05
OVERLAP_RECOVERY_NOTE_MATCH_SEC = 0.08
CHORD_EVENT_DEDUPE_TOLERANCE_SEC = 0.06
RECENT_EVENT_RETENTION_SEC = 0.25
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


def _get_session(session_id: str) -> Dict:
    s = _stream_sessions.get(session_id)
    if s is None:
        s = {
            "tail": np.zeros(0, dtype=np.float32),
            "sample_cursor": 0,  # samples processed (without overlap)
            "recent_notes": [],
            "recent_chords": [],
        }
        _stream_sessions[session_id] = s
    return s


def _clear_stream_session(session_id: str) -> None:
    _stream_sessions.pop(session_id, None)


def _decode_stream_packet_audio(message: Dict, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Decode a websocket audio packet to mono float32 PCM at target_sr."""
    source_sr = int(message.get("sample_rate") or target_sr)

    if "samples" in message:
        audio = np.asarray(message.get("samples") or [], dtype=np.float32)
    else:
        payload_b64 = (
            message.get("pcm16_base64")
            or message.get("audio_base64")
            or message.get("audio_b64")
            or ""
        )
        if not payload_b64:
            return np.zeros(0, dtype=np.float32), target_sr
        raw = base64.b64decode(str(payload_b64))
        encoding = str(message.get("encoding") or "pcm16").lower()
        if encoding in {"float32", "f32", "pcm_float32"}:
            audio = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        else:
            audio = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio, target_sr

    if source_sr != target_sr:
        gcd = math.gcd(source_sr, target_sr)
        up, down = target_sr // gcd, source_sr // gcd
        audio = resample_poly(audio, up, down).astype(np.float32, copy=False)
    return audio, target_sr


def _note_payload_from_hypothesis(hypothesis: Dict) -> Dict:
    onset = float(hypothesis.get("onset_time", 0.0) or 0.0)
    offset = float(hypothesis.get("offset_time", onset) or onset)
    return {
        "id": int(hypothesis.get("id", 0) or 0),
        "state": str(hypothesis.get("state") or "candidate"),
        "midi_note": int(hypothesis.get("midi_note", 0) or 0),
        "onset_time": round(onset, 6),
        "offset_time": round(offset, 6),
        "duration": round(max(0.0, offset - onset), 6),
        "confidence": round(float(hypothesis.get("confidence", 0.0) or 0.0), 4),
        "observations": int(hypothesis.get("observations", 0) or 0),
        "first_seen_time": round(float(hypothesis.get("first_seen_time", 0.0) or 0.0), 6),
        "last_seen_time": round(float(hypothesis.get("last_seen_time", 0.0) or 0.0), 6),
    }


STREAM_ATTACK_PRE_SEC = 0.08
STREAM_ATTACK_POST_SEC = 0.08
STREAM_ATTACK_GAP_SEC = 0.012
STREAM_ATTACK_RATIO_STRONG = 1.45
STREAM_ATTACK_DELTA_STRONG = 0.010
STREAM_CONTINUITY_BOUNDARY_SEC = 0.30
STREAM_SAME_PITCH_RECENT_SEC = 0.45
STREAM_MIN_REPEAT_SEC = 0.20
STREAM_HARMONIC_RECENT_SEC = 1.20
STREAM_HARMONIC_INTERVALS = {0, 4, 7}
STREAM_DEBUG_SAMPLE_LIMIT = 12
STREAM_ATTACK_GROUP_MERGE_SEC = 0.12
STREAM_ATTACK_GROUP_RESCUE_SEC = 0.25
STREAM_ATTACK_GROUP_KEEP_SEC = 3.0
STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE = 0.50
STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE = 0.78
STREAM_WEAK_BIRTH_HIGH_CONFIDENCE = 0.86

# Soft inner voices sounded under held louder notes produce almost no audio-RMS
# attack, so the RMS-based weak-attack heuristic cannot see them and the
# weak-birth / harmonic-sustain gates delete real notes. The model's frame head
# does see them: a real note sustains for several frames (decoded duration well
# above the single-frame floor), while live-decode noise is a 1-frame blip.
# STREAM_FRAME_EVIDENCE_SEC lets a weak-attack observation be *born* when the
# model sustains it; persistence then decides whether it reaches the score.
STREAM_FRAME_EVIDENCE_SEC = 0.08
# A born hypothesis only reaches the displayed (committed) surface once it is
# either re-observed across enough overlapping windows (real notes recur many
# times; noise is seen once or twice) OR carries strong frame evidence. This is
# what rejects the noise that floods in when the birth gates are merely relaxed.
STREAM_MIN_DISPLAY_OBSERVATIONS = 3
STREAM_DISPLAY_FRAME_EVIDENCE_SEC = 0.15
# Master switch for the RMS-attack birth gates in _filter_stream_continuity
# (same_pitch_boundary / implausible_repeat / harmonic_sustain /
# weak_birth_outside_attack). When False, every decoded observation is born and
# birth/noise rejection is delegated entirely to the persistence + frame-evidence
# display gate above. The RMS-attack heuristic physically cannot see a soft note
# struck under sustained louder notes, so the gates cost real recall (soft inner
# voices, soft repeated notes) while the display gate already separates real notes
# (median ~22 observations) from single-window decode noise (median 1).
STREAM_RMS_BIRTH_GATES = False


class ContinuousLiveStreamSession:
    """Continuous packet-driven live transcription state.

    Audio packets are transport units only. Neural inference runs on rolling
    windows, then updates note hypotheses on absolute session time.
    """

    def __init__(
        self,
        session_id: str,
        sample_rate: int = 44100,
        context_sec: float = 2.4,
        inference_interval_sec: float = 0.10,
        trusted_delay_sec: float = 0.18,
        commit_delay_sec: float = 0.50,
        lock_delay_sec: float = 2.0,
        max_buffer_sec: float = 12.0,
    ):
        self.session_id = session_id
        self.sample_rate = int(sample_rate)
        self.context_sec = float(context_sec)
        self.inference_interval_sec = float(inference_interval_sec)
        self.trusted_delay_sec = float(trusted_delay_sec)
        self.commit_delay_sec = float(commit_delay_sec)
        self.lock_delay_sec = float(lock_delay_sec)
        self.max_buffer_sec = float(max_buffer_sec)

        self.audio = np.zeros(0, dtype=np.float32)
        self.absolute_start_sample = 0
        self.sample_cursor = 0
        self.last_inference_sample = 0
        self.created_at = time.time()
        self.last_update = self.created_at
        self.first_audio_wall_time: Optional[float] = None
        self.next_note_id = 1
        self.hypotheses: List[Dict] = []
        self.attack_groups: List[Dict] = []
        self.received_packet_count = 0
        self.skipped_inference_count = 0
        self.continuity_filter_total: Dict[str, int] = defaultdict(int)
        self.last_packet_sequence: Optional[int] = None
        self.last_client_sent_at_ms: Optional[float] = None
        self.last_server_received_at_ms: Optional[float] = None
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @property
    def current_time_sec(self) -> float:
        return float(self.sample_cursor) / float(self.sample_rate)

    def append_audio(self, audio: np.ndarray, packet_metadata: Optional[Dict] = None) -> Dict:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.size == 0:
            return self.status()
        with self._lock:
            if self.first_audio_wall_time is None:
                self.first_audio_wall_time = time.time()
            self.received_packet_count += 1
            if packet_metadata:
                sequence = packet_metadata.get("sequence_number")
                if sequence is not None:
                    try:
                        self.last_packet_sequence = int(sequence)
                    except (TypeError, ValueError):
                        pass
                client_sent_at_ms = packet_metadata.get("client_sent_at_ms")
                if client_sent_at_ms is not None:
                    try:
                        self.last_client_sent_at_ms = float(client_sent_at_ms)
                    except (TypeError, ValueError):
                        pass
                server_received_at_ms = packet_metadata.get("server_received_at_ms")
                if server_received_at_ms is not None:
                    try:
                        self.last_server_received_at_ms = float(server_received_at_ms)
                    except (TypeError, ValueError):
                        pass
            self.audio = np.concatenate([self.audio, audio.astype(np.float32, copy=False)])
            self.sample_cursor += int(audio.size)
            self.last_update = time.time()
            self._trim_audio_buffer()
        return self.status()

    def status(self) -> Dict:
        with self._lock:
            buffered_samples = int(self.audio.size)
            current_time_sec = self.current_time_sec
            wall_anchor = self.first_audio_wall_time or self.created_at
            wall_elapsed_sec = max(0.0, time.time() - wall_anchor)
            stream_backlog_sec = max(0.0, wall_elapsed_sec - current_time_sec)
            return {
                "session_id": self.session_id,
                "sample_rate": self.sample_rate,
                "sample_cursor": int(self.sample_cursor),
                "audio_time_sec": round(current_time_sec, 6),
                "wall_elapsed_sec": round(wall_elapsed_sec, 6),
                "stream_backlog_sec": round(stream_backlog_sec, 6),
                "buffered_sec": round(buffered_samples / float(self.sample_rate), 6),
                "context_sec": self.context_sec,
                "inference_interval_sec": self.inference_interval_sec,
                "trusted_delay_sec": self.trusted_delay_sec,
                "commit_delay_sec": self.commit_delay_sec,
                "lock_delay_sec": self.lock_delay_sec,
                "received_packet_count": int(self.received_packet_count),
                "last_packet_sequence": self.last_packet_sequence,
                "last_client_sent_at_ms": self.last_client_sent_at_ms,
                "last_server_received_at_ms": self.last_server_received_at_ms,
                "transport_mode": "decoupled",
            }

    def maybe_run_inference(self, force: bool = False) -> Optional[Dict]:
        with self._inference_lock:
            with self._lock:
                interval_samples = max(1, int(round(self.inference_interval_sec * self.sample_rate)))
                if not force and (self.sample_cursor - self.last_inference_sample) < interval_samples:
                    self.skipped_inference_count += 1
                    return None
                if self.audio.size < MIN_STREAM_ANALYSIS_SAMPLES:
                    return self._build_update(inference_ran=False, reason="insufficient_audio")

                context_samples = max(MIN_STREAM_ANALYSIS_SAMPLES, int(round(self.context_sec * self.sample_rate)))
                window_end_sample = self.sample_cursor
                window_start_sample = max(self.absolute_start_sample, window_end_sample - context_samples)
                rel_start = max(0, int(window_start_sample - self.absolute_start_sample))
                window_audio = self.audio[rel_start:].astype(np.float32, copy=True)
                if window_audio.size < MIN_STREAM_ANALYSIS_SAMPLES:
                    return self._build_update(inference_ran=False, reason="insufficient_window")
                self.last_inference_sample = window_end_sample

            inference_started = time.perf_counter()
            result = analyze_audio_live_neural(
                window_audio,
                sr=self.sample_rate,
                debug=False,
                split_midi=60,
                device="cuda",
                adaptive_onset_threshold=True,
            )
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            stream_backlog_sec = self.status().get("stream_backlog_sec", 0.0)

            if result.get("error"):
                return self._build_update(
                    inference_ran=True,
                    reason="inference_error",
                    inference_ms=inference_ms,
                    received_packet_count=self.received_packet_count,
                    skipped_inference_count=self.skipped_inference_count,
                    error=str(result.get("error")),
                )

            window_start_sec = float(window_start_sample) / float(self.sample_rate)
            observations = self._observations_from_result(result, window_start_sec, window_audio)
            with self._lock:
                observations, continuity_filter = self._filter_stream_continuity(
                    observations,
                    window_start_sec,
                )
                trusted_cutoff_sec = max(0.0, self.current_time_sec - self.trusted_delay_sec)
                hypothesis_update = self._update_hypotheses(observations, trusted_cutoff_sec)
                self._age_hypotheses()

            return self._build_update(
                inference_ran=True,
                reason="ok",
                inference_ms=inference_ms,
                observation_count=len(observations),
                received_packet_count=self.received_packet_count,
                skipped_inference_count=self.skipped_inference_count,
                stream_backlog_sec=stream_backlog_sec,
                neural_timing=result.get("_timing_ms") or {},
                analysis_summary=result.get("analysis_summary") or {},
                continuity_filter=continuity_filter,
                hypothesis_update=hypothesis_update,
            )

    def warmup_live_path(self) -> Dict:
        context_samples = max(
            MIN_STREAM_ANALYSIS_SAMPLES,
            int(round(self.context_sec * self.sample_rate)),
        )
        warmup_audio = np.zeros(context_samples, dtype=np.float32)
        inference_started = time.perf_counter()
        result = analyze_audio_live_neural(
            warmup_audio,
            sr=self.sample_rate,
            debug=False,
            split_midi=60,
            device="cuda",
            adaptive_onset_threshold=True,
        )
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        return make_json_serializable({
            "status": "warm",
            "inference_ms": inference_ms,
            "neural_timing": result.get("_timing_ms") or {},
            "analysis_summary": result.get("analysis_summary") or {},
            "error": result.get("error"),
        })

    def _observations_from_result(self, result: Dict, window_start_sec: float, window_audio: np.ndarray) -> List[Dict]:
        observations: List[Dict] = []

        for note in result.get("notes") or []:
            onset = window_start_sec + float(note.get("time_seconds", 0.0) or 0.0)
            duration = float(note.get("duration_seconds", 0.0) or 0.0)
            offset = window_start_sec + float(note.get("offset_seconds", note.get("time_seconds", 0.0) + duration) or 0.0)
            attack = self._attack_metrics(window_audio, onset - window_start_sec)
            observations.append({
                "midi_note": int(note.get("midi_note", 0) or 0),
                "onset_time": onset,
                "offset_time": max(offset, onset + max(duration, 0.04)),
                "confidence": float(note.get("confidence", 0.0) or 0.0),
                "source": "note",
                "decode_source": note.get("decode_source"),
                **attack,
            })

        for chord in result.get("chords") or []:
            onset = window_start_sec + float(chord.get("time_seconds", 0.0) or 0.0)
            duration = float(chord.get("duration_seconds", 0.0) or 0.0)
            offset = window_start_sec + float(chord.get("offset_seconds", chord.get("time_seconds", 0.0) + duration) or 0.0)
            confidence = float(chord.get("confidence", 0.0) or 0.0)
            attack = self._attack_metrics(window_audio, onset - window_start_sec)
            decode_sources = chord.get("decode_sources") or []
            for idx, midi_note in enumerate(chord.get("midi_notes") or []):
                observations.append({
                    "midi_note": int(midi_note),
                    "onset_time": onset,
                    "offset_time": max(offset, onset + max(duration, 0.04)),
                    "confidence": confidence,
                    "source": "chord_member",
                    "decode_source": decode_sources[idx] if idx < len(decode_sources) else None,
                    **attack,
                })

        observations.sort(key=lambda item: (item["onset_time"], item["midi_note"]))
        return observations

    def _attack_metrics(self, window_audio: np.ndarray, local_onset_sec: float) -> Dict:
        onset_sample = int(round(float(local_onset_sec) * self.sample_rate))
        if window_audio.size == 0 or onset_sample < 0 or onset_sample >= window_audio.size:
            return {
                "attack_ratio": 1.0,
                "attack_delta": 0.0,
                "has_strong_attack": False,
            }

        pre_start = max(0, onset_sample - int(round(STREAM_ATTACK_PRE_SEC * self.sample_rate)))
        pre_end = max(pre_start, onset_sample - int(round(STREAM_ATTACK_GAP_SEC * self.sample_rate)))
        post_start = onset_sample
        post_end = min(window_audio.size, onset_sample + int(round(STREAM_ATTACK_POST_SEC * self.sample_rate)))
        pre = window_audio[pre_start:pre_end]
        post = window_audio[post_start:post_end]
        if pre.size == 0 or post.size == 0:
            return {
                "attack_ratio": 1.0,
                "attack_delta": 0.0,
                "has_strong_attack": False,
            }

        pre_rms = float(np.sqrt(np.mean(np.square(pre, dtype=np.float32))) + 1e-6)
        post_rms = float(np.sqrt(np.mean(np.square(post, dtype=np.float32))) + 1e-6)
        attack_ratio = post_rms / pre_rms
        attack_delta = post_rms - pre_rms
        return {
            "attack_ratio": float(attack_ratio),
            "attack_delta": float(attack_delta),
            "has_strong_attack": bool(
                attack_ratio >= STREAM_ATTACK_RATIO_STRONG
                or attack_delta >= STREAM_ATTACK_DELTA_STRONG
            ),
        }

    def _append_observation_debug_sample(
        self,
        target: List[Dict],
        observation: Dict,
        reason: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        if len(target) >= STREAM_DEBUG_SAMPLE_LIMIT:
            return
        onset = float(observation.get("onset_time", 0.0) or 0.0)
        sample = {
            "midi": int(observation.get("midi_note", 0) or 0),
            "onset": round(onset, 4),
            "offset": round(float(observation.get("offset_time", onset) or onset), 4),
            "confidence": round(float(observation.get("confidence", 0.0) or 0.0), 3),
            "source": str(observation.get("source") or "unknown"),
            "attack_ratio": round(float(observation.get("attack_ratio", 1.0) or 1.0), 3),
            "attack_delta": round(float(observation.get("attack_delta", 0.0) or 0.0), 4),
            "strong_attack": bool(observation.get("has_strong_attack")),
        }
        if reason:
            sample["reason"] = reason
        if extra:
            sample.update(extra)
        if observation.get("decode_source"):
            sample["decode_source"] = str(observation.get("decode_source"))
        target.append(sample)

    def _same_pitch_recent_hypothesis(self, pitch: int, onset: float) -> Optional[Dict]:
        best: Optional[Dict] = None
        best_gap: Optional[float] = None
        for hypothesis in self.hypotheses:
            if int(hypothesis.get("midi_note", -1)) != pitch:
                continue
            hyp_onset = float(hypothesis.get("onset_time", 0.0) or 0.0)
            hyp_offset = float(hypothesis.get("offset_time", hyp_onset) or hyp_onset)
            hyp_last_seen = float(hypothesis.get("last_seen_time", hyp_onset) or hyp_onset)
            if hyp_onset > onset + 0.06:
                continue
            overlap_gap = onset - hyp_offset
            seen_gap = self.current_time_sec - hyp_last_seen
            repeat_gap = onset - hyp_onset
            is_recent = (
                overlap_gap <= STREAM_SAME_PITCH_RECENT_SEC
                or seen_gap <= STREAM_SAME_PITCH_RECENT_SEC
                or repeat_gap <= STREAM_MIN_REPEAT_SEC
            )
            if not is_recent:
                continue
            score_gap = max(0.0, min(overlap_gap, seen_gap, repeat_gap))
            if best_gap is None or score_gap < best_gap:
                best = hypothesis
                best_gap = score_gap
        return best

    def _lower_harmonic_explainer(self, pitch: int, onset: float) -> Optional[Dict]:
        best: Optional[Dict] = None
        best_interval: Optional[int] = None
        now_sec = self.current_time_sec
        for hypothesis in self.hypotheses:
            base_pitch = int(hypothesis.get("midi_note", -1))
            interval = pitch - base_pitch
            if interval < 7 or interval > 36:
                continue
            if interval % 12 not in STREAM_HARMONIC_INTERVALS:
                continue

            hyp_onset = float(hypothesis.get("onset_time", 0.0) or 0.0)
            hyp_offset = float(hypothesis.get("offset_time", hyp_onset) or hyp_onset)
            hyp_last_seen = float(hypothesis.get("last_seen_time", hyp_onset) or hyp_onset)
            is_sounding = hyp_offset >= onset - STREAM_HARMONIC_RECENT_SEC
            was_recently_seen = now_sec - hyp_last_seen <= STREAM_HARMONIC_RECENT_SEC
            if not (is_sounding or was_recently_seen):
                continue
            if best_interval is None or interval < best_interval:
                best = hypothesis
                best_interval = interval
        return best

    def _prune_attack_groups(self) -> None:
        now_sec = self.current_time_sec
        self.attack_groups = [
            group
            for group in self.attack_groups
            if now_sec - float(group.get("last_seen_time", group.get("time", 0.0)) or 0.0)
            <= STREAM_ATTACK_GROUP_KEEP_SEC
        ]

    def _register_attack_groups(self, observations: List[Dict]) -> int:
        self._prune_attack_groups()
        registered = 0
        for observation in observations:
            if not bool(observation.get("has_strong_attack")):
                continue
            onset = float(observation.get("onset_time", 0.0) or 0.0)
            pitch = int(observation.get("midi_note", 0) or 0)
            strength = max(
                float(observation.get("attack_ratio", 1.0) or 1.0),
                1.0 + (float(observation.get("attack_delta", 0.0) or 0.0) * 100.0),
            )
            best = None
            best_error = None
            for group in self.attack_groups:
                error = abs(float(group.get("time", 0.0) or 0.0) - onset)
                if error <= STREAM_ATTACK_GROUP_MERGE_SEC and (
                    best_error is None or error < best_error
                ):
                    best = group
                    best_error = error
            if best is None:
                self.attack_groups.append({
                    "time": onset,
                    "strength": strength,
                    "pitches": {pitch},
                    "last_seen_time": self.current_time_sec,
                })
                registered += 1
            else:
                count = int(best.get("count", len(best.get("pitches", [])) or 1) or 1)
                best["time"] = ((float(best.get("time", onset) or onset) * count) + onset) / float(count + 1)
                best["strength"] = max(float(best.get("strength", 0.0) or 0.0), strength)
                pitches = best.get("pitches")
                if not isinstance(pitches, set):
                    pitches = set(pitches or [])
                pitches.add(pitch)
                best["pitches"] = pitches
                best["count"] = count + 1
                best["last_seen_time"] = self.current_time_sec
        return registered

    def _nearest_attack_group(self, onset: float) -> Optional[Dict]:
        self._prune_attack_groups()
        best = None
        best_error = None
        for group in self.attack_groups:
            error = abs(float(group.get("time", 0.0) or 0.0) - onset)
            if error <= STREAM_ATTACK_GROUP_RESCUE_SEC and (
                best_error is None or error < best_error
            ):
                best = group
                best_error = error
        return best

    def _filter_stream_continuity(
        self,
        observations: List[Dict],
        window_start_sec: float,
    ) -> Tuple[List[Dict], Dict]:
        stats = {
            "input": len(observations),
            "kept": 0,
            "suppressed": 0,
            "same_pitch_boundary": 0,
            "implausible_repeat": 0,
            "harmonic_sustain": 0,
            "weak_birth_outside_attack": 0,
            "attack_groups": len(self.attack_groups),
            "registered_attack_groups": 0,
            "suppressed_samples": [],
            "total_suppressed": int(self.continuity_filter_total.get("suppressed", 0)),
        }
        if not observations:
            self._prune_attack_groups()
            stats["attack_groups"] = len(self.attack_groups)
            return observations, stats

        stats["registered_attack_groups"] = self._register_attack_groups(observations)
        stats["attack_groups"] = len(self.attack_groups)

        kept: List[Dict] = []
        for observation in observations:
            pitch = int(observation.get("midi_note", 0) or 0)
            onset = float(observation.get("onset_time", 0.0) or 0.0)
            local_onset = onset - window_start_sec
            strong_attack = bool(observation.get("has_strong_attack"))
            attack_ratio = float(observation.get("attack_ratio", 1.0) or 1.0)
            attack_delta = float(observation.get("attack_delta", 0.0) or 0.0)
            weak_attack = (
                not strong_attack
                and attack_ratio < STREAM_ATTACK_RATIO_STRONG
                and attack_delta < STREAM_ATTACK_DELTA_STRONG
            )

            if self._match_hypothesis(observation) is not None:
                kept.append(observation)
                continue

            # When the RMS-attack birth gates are disabled, keep every decoded
            # observation here; the persistence + frame-evidence display gate is
            # the only birth/noise filter. attack-group registration above still
            # runs so rescue bookkeeping stays consistent if gates are re-enabled.
            if not STREAM_RMS_BIRTH_GATES:
                kept.append(observation)
                continue

            same_pitch = self._same_pitch_recent_hypothesis(pitch, onset)
            if same_pitch is not None and weak_attack:
                hyp_onset = float(same_pitch.get("onset_time", onset) or onset)
                repeat_gap = onset - hyp_onset
                if local_onset <= STREAM_CONTINUITY_BOUNDARY_SEC:
                    stats["same_pitch_boundary"] += 1
                    stats["suppressed"] += 1
                    self._append_observation_debug_sample(
                        stats["suppressed_samples"],
                        observation,
                        reason="same_pitch_boundary",
                        extra={
                            "existing_midi": int(same_pitch.get("midi_note", pitch) or pitch),
                            "existing_onset_time": round(hyp_onset, 4),
                        },
                    )
                    continue
                if repeat_gap <= STREAM_MIN_REPEAT_SEC:
                    stats["implausible_repeat"] += 1
                    stats["suppressed"] += 1
                    self._append_observation_debug_sample(
                        stats["suppressed_samples"],
                        observation,
                        reason="implausible_repeat",
                        extra={
                            "existing_midi": int(same_pitch.get("midi_note", pitch) or pitch),
                            "repeat_gap_ms": int(round(max(0.0, repeat_gap) * 1000.0)),
                        },
                    )
                    continue

            confidence = float(observation.get("confidence", 0.0) or 0.0)
            attack_group = self._nearest_attack_group(onset)

            decode_source = str(observation.get("decode_source") or "")
            # Calibrated inner-voice rescues are deliberate below-threshold events
            # snapped onto a real attack cluster, so they must bypass both the
            # harmonic-sustain and weak-birth gates that exist to kill incidental
            # ring-out. Without this they are dropped exactly like the misses we
            # are trying to recover (quiet voices above a held outer note).
            lattice_rescued = decode_source == "lattice_calibrated"

            # Model frame evidence: a quiet inner voice the model sustains for
            # several frames is real even though its audio-RMS attack is weak.
            # This is the only signal that lets such notes be born; persistence
            # (below, at promotion time) then keeps noise out of the score.
            note_duration = float(observation.get("offset_time", onset) or onset) - onset
            has_frame_evidence = note_duration >= STREAM_FRAME_EVIDENCE_SEC

            harmonic_base = self._lower_harmonic_explainer(pitch, onset)
            if (
                harmonic_base is not None
                and attack_group is None
                and weak_attack
                and not lattice_rescued
                and not has_frame_evidence
                and confidence < STREAM_HARMONIC_SUPPRESS_MAX_CONFIDENCE
            ):
                stats["harmonic_sustain"] += 1
                stats["suppressed"] += 1
                self._append_observation_debug_sample(
                    stats["suppressed_samples"],
                    observation,
                    reason="harmonic_sustain",
                    extra={
                        "base_midi": int(harmonic_base.get("midi_note", 0) or 0),
                        "interval": int(pitch - int(harmonic_base.get("midi_note", pitch) or pitch)),
                    },
                )
                continue

            source = str(observation.get("source") or "")
            can_rescue_from_decode = decode_source in {
                "soft_polyphony_rescue",
                "lattice_calibrated",
            }
            can_rescue_from_attack_group = (
                attack_group is not None
                and (
                    source == "chord_member"
                    or confidence >= STREAM_ATTACK_GROUP_RESCUE_MIN_CONFIDENCE
                )
            )
            if (
                weak_attack
                and not can_rescue_from_decode
                and not can_rescue_from_attack_group
                and not has_frame_evidence
                and confidence < STREAM_WEAK_BIRTH_HIGH_CONFIDENCE
            ):
                stats["weak_birth_outside_attack"] += 1
                stats["suppressed"] += 1
                extra = {}
                if attack_group is not None:
                    extra["attack_group_time"] = round(float(attack_group.get("time", 0.0) or 0.0), 4)
                    extra["attack_group_dt_ms"] = int(round(abs(onset - float(attack_group.get("time", 0.0) or 0.0)) * 1000.0))
                self._append_observation_debug_sample(
                    stats["suppressed_samples"],
                    observation,
                    reason="weak_birth_outside_attack",
                    extra=extra,
                )
                continue

            kept.append(observation)

        stats["kept"] = len(kept)
        for key, value in stats.items():
            if key == "suppressed_samples":
                continue
            self.continuity_filter_total[key] += int(value)
        stats["total_suppressed"] = int(self.continuity_filter_total.get("suppressed", 0))
        return kept, stats

    def _match_hypothesis(self, observation: Dict, tolerance_sec: float = 0.06) -> Optional[Dict]:
        pitch = int(observation["midi_note"])
        onset = float(observation["onset_time"])
        best = None
        best_error = None
        for hypothesis in self.hypotheses:
            if int(hypothesis.get("midi_note", -1)) != pitch:
                continue
            error = abs(float(hypothesis.get("onset_time", 0.0) or 0.0) - onset)
            if error <= tolerance_sec and (best_error is None or error < best_error):
                best = hypothesis
                best_error = error
        return best

    def _update_hypotheses(self, observations: List[Dict], trusted_cutoff_sec: float) -> Dict:
        now_sec = self.current_time_sec
        stats = {
            "input": len(observations),
            "created": 0,
            "matched": 0,
            "stale_skipped": 0,
            "promoted_active": 0,
            "promoted_committed": 0,
            "promoted_locked": 0,
            "birth_samples": [],
        }
        for observation in observations:
            onset = float(observation["onset_time"])
            if onset < max(0.0, now_sec - self.context_sec - 0.25):
                stats["stale_skipped"] += 1
                continue

            hypothesis = self._match_hypothesis(observation)
            if hypothesis is None:
                hypothesis = {
                    "id": self.next_note_id,
                    "state": "candidate",
                    "midi_note": int(observation["midi_note"]),
                    "onset_time": onset,
                    "offset_time": float(observation["offset_time"]),
                    "confidence": float(observation["confidence"]),
                    "observations": 0,
                    "first_seen_time": now_sec,
                    "last_seen_time": now_sec,
                }
                self.next_note_id += 1
                self.hypotheses.append(hypothesis)
                stats["created"] += 1
                self._append_observation_debug_sample(
                    stats["birth_samples"],
                    observation,
                    reason="created",
                    extra={
                        "id": int(hypothesis["id"]),
                        "audio_time": round(now_sec, 4),
                    },
                )
            else:
                stats["matched"] += 1

            observations_count = int(hypothesis.get("observations", 0) or 0) + 1
            old_conf = float(hypothesis.get("confidence", 0.0) or 0.0)
            new_conf = float(observation.get("confidence", 0.0) or 0.0)
            if str(hypothesis.get("state")) not in {"committed", "locked"}:
                old_onset = float(hypothesis.get("onset_time", onset) or onset)
                hypothesis["onset_time"] = (old_onset * 0.7) + (onset * 0.3)
            hypothesis["offset_time"] = max(float(hypothesis.get("offset_time", onset) or onset), float(observation["offset_time"]))
            hypothesis["confidence"] = max(old_conf, new_conf)
            hypothesis["observations"] = observations_count
            hypothesis["last_seen_time"] = now_sec

            # Persistence/frame-evidence gate for reaching any displayed surface.
            # Real notes are re-observed across many overlapping windows or carry
            # strong frame evidence; single-window decode noise has neither, so it
            # stays a (hidden) candidate and ages out instead of entering the score.
            display_duration = (
                float(hypothesis.get("offset_time", onset) or onset)
                - float(hypothesis.get("onset_time", onset) or onset)
            )
            display_ready = (
                observations_count >= STREAM_MIN_DISPLAY_OBSERVATIONS
                or display_duration >= STREAM_DISPLAY_FRAME_EVIDENCE_SEC
            )
            if (
                onset <= trusted_cutoff_sec
                and str(hypothesis.get("state")) == "candidate"
                and display_ready
            ):
                hypothesis["state"] = "active"
                stats["promoted_active"] += 1

        for hypothesis in self.hypotheses:
            state = str(hypothesis.get("state") or "candidate")
            onset = float(hypothesis.get("onset_time", 0.0) or 0.0)
            obs = int(hypothesis.get("observations", 0) or 0)
            dur = float(hypothesis.get("offset_time", onset) or onset) - onset
            display_ready = (
                obs >= STREAM_MIN_DISPLAY_OBSERVATIONS
                or dur >= STREAM_DISPLAY_FRAME_EVIDENCE_SEC
            )
            if (
                state in {"candidate", "active"}
                and onset <= now_sec - self.commit_delay_sec
                and display_ready
            ):
                hypothesis["state"] = "committed"
                hypothesis["committed_time"] = now_sec
                stats["promoted_committed"] += 1
            if state == "committed" and onset <= now_sec - self.lock_delay_sec:
                hypothesis["state"] = "locked"
                hypothesis["locked_time"] = now_sec
                stats["promoted_locked"] += 1
        return stats

    def _age_hypotheses(self) -> None:
        now_sec = self.current_time_sec
        keep = []
        for hypothesis in self.hypotheses:
            state = str(hypothesis.get("state") or "candidate")
            last_seen = float(hypothesis.get("last_seen_time", 0.0) or 0.0)
            onset = float(hypothesis.get("onset_time", 0.0) or 0.0)
            if state in {"candidate", "active"} and (now_sec - last_seen) > 1.0:
                continue
            if state in {"committed", "locked"} and onset < now_sec - 60.0:
                continue
            keep.append(hypothesis)
        self.hypotheses = keep

    def _trim_audio_buffer(self) -> None:
        max_samples = max(MIN_STREAM_ANALYSIS_SAMPLES, int(round(self.max_buffer_sec * self.sample_rate)))
        if self.audio.size <= max_samples:
            return
        drop = int(self.audio.size - max_samples)
        self.audio = self.audio[drop:].astype(np.float32, copy=False)
        self.absolute_start_sample += drop

    def _build_update(self, inference_ran: bool, reason: str = "ok", **extra) -> Dict:
        with self._lock:
            candidates = []
            active = []
            committed = []
            locked = []
            heard = []
            now_sec = self.current_time_sec

            for hypothesis in sorted(self.hypotheses, key=lambda item: (item.get("onset_time", 0.0), item.get("midi_note", 0))):
                payload = _note_payload_from_hypothesis(hypothesis)
                state = payload["state"]
                if now_sec - float(payload["last_seen_time"]) <= 0.75:
                    heard.append(payload)
                if state == "candidate":
                    candidates.append(payload)
                elif state == "active":
                    active.append(payload)
                elif state == "locked":
                    locked.append(payload)
                    committed.append(payload)
                else:
                    committed.append(payload)

            update = {
                "type": "live_stream_update",
                "session": self.status(),
                "inference": {
                    "ran": bool(inference_ran),
                    "reason": reason,
                    **extra,
                },
                "heard_notes": heard[-64:],
                "candidate_notes": candidates[-64:],
                "active_notes": active[-64:],
                "committed_notes": committed[-256:],
                "locked_notes": locked[-256:],
                "counts": {
                    "heard": len(heard),
                    "candidate": len(candidates),
                    "active": len(active),
                    "committed": len(committed),
                    "locked": len(locked),
                },
            }
        return make_json_serializable(update)


_continuous_live_stream_sessions: Dict[str, ContinuousLiveStreamSession] = {}


def _get_continuous_live_stream_session(
    session_id: str,
    sample_rate: int = 44100,
    context_sec: Optional[float] = None,
    inference_interval_ms: Optional[float] = None,
    trusted_delay_ms: Optional[float] = None,
    commit_delay_ms: Optional[float] = None,
    lock_delay_ms: Optional[float] = None,
) -> ContinuousLiveStreamSession:
    session = _continuous_live_stream_sessions.get(session_id)
    if session is not None:
        return session

    session = ContinuousLiveStreamSession(
        session_id=session_id,
        sample_rate=sample_rate,
        context_sec=float(context_sec if context_sec is not None else LIVE_CONTEXT_SEC),
        inference_interval_sec=float(inference_interval_ms if inference_interval_ms is not None else 100.0) / 1000.0,
        trusted_delay_sec=float(trusted_delay_ms if trusted_delay_ms is not None else 180.0) / 1000.0,
        commit_delay_sec=float(commit_delay_ms if commit_delay_ms is not None else 500.0) / 1000.0,
        lock_delay_sec=float(lock_delay_ms if lock_delay_ms is not None else 2000.0) / 1000.0,
    )
    _continuous_live_stream_sessions[session_id] = session
    return session


def _clear_continuous_live_stream_session(session_id: str) -> None:
    _continuous_live_stream_sessions.pop(session_id, None)


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


def _drop_chunk_end_micro_events(
    events: List[Dict],
    analysis_window_sec: float,
    guard_sec: float = CHUNK_END_GUARD_SEC,
    max_duration_sec: float = CHUNK_END_MICRO_EVENT_MAX_DURATION_SEC,
) -> Tuple[List[Dict], int]:
    """Suppress short events emitted at the chunk tail where context is weakest.

    The live chunk-gap diagnostic showed most chunk-only false positives were
    ~16ms notes fired within ~5ms of the chunk end. Defer these edge events so
    the next chunk or finalization pass can decide them with more future context.
    """
    if not events or analysis_window_sec <= 0:
        return list(events or []), 0

    keep: List[Dict] = []
    dropped = 0
    threshold = max(0.0, analysis_window_sec - guard_sec)
    for event in events:
        onset = _event_time(event)
        duration = _event_duration(event)
        if onset >= threshold and duration <= max_duration_sec:
            dropped += 1
            continue
        keep.append(event)
    return keep, dropped


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


def _note_events_match(
    event_a: Dict,
    event_b: Dict,
    time_tolerance_sec: float = OVERLAP_RECOVERY_NOTE_MATCH_SEC,
) -> bool:
    try:
        midi_a = int(event_a.get("midi_note"))
        midi_b = int(event_b.get("midi_note"))
    except (TypeError, ValueError):
        return False

    if midi_a != midi_b:
        return False

    onset_a = _event_time(event_a)
    onset_b = _event_time(event_b)
    if abs(onset_a - onset_b) <= time_tolerance_sec:
        return True

    offset_a = onset_a + _event_duration(event_a)
    offset_b = onset_b + _event_duration(event_b)
    return min(offset_a, offset_b) >= (max(onset_a, onset_b) - 0.02)


def _chord_events_match(
    event_a: Dict,
    event_b: Dict,
    time_tolerance_sec: float = CHORD_EVENT_DEDUPE_TOLERANCE_SEC,
) -> bool:
    return (
        _chord_signature(event_a) == _chord_signature(event_b)
        and abs(_event_time(event_a) - _event_time(event_b)) <= time_tolerance_sec
    )


def _trim_recent_events(
    events: List[Dict],
    current_time_sec: float,
    retention_sec: float = RECENT_EVENT_RETENTION_SEC,
) -> List[Dict]:
    cutoff = current_time_sec - retention_sec
    if cutoff <= 0.0:
        return list(events)
    return [dict(event) for event in events if _event_time(event) >= cutoff]


def _shift_overlap_events_with_recent_dedupe(
    events: List[Dict],
    absolute_chunk_start_sec: float,
    overlap_sec: float,
    recent_events: List[Dict],
    duplicate_matcher,
    recovery_band_sec: float | None = None,
) -> Tuple[List[Dict], int, int]:
    emitted: List[Dict] = []
    overlap_recovered = 0
    overlap_duplicates_skipped = 0

    # Notes in the prepended history older than the recovery band were already
    # emitted by earlier chunks; that audio is present only as model context, so
    # drop them here rather than re-recovering them as duplicates.
    recovery_floor_sec = (
        overlap_sec - recovery_band_sec
        if recovery_band_sec is not None
        else 0.0
    )

    for event in events or []:
        time_seconds = _event_time(event)
        absolute_time = absolute_chunk_start_sec + (time_seconds - overlap_sec)
        shifted = dict(event)
        shifted["time_seconds"] = round(absolute_time, 6)

        if time_seconds < overlap_sec:
            if time_seconds < recovery_floor_sec:
                continue
            if any(duplicate_matcher(shifted, existing) for existing in recent_events):
                overlap_duplicates_skipped += 1
                continue
            overlap_recovered += 1

        emitted.append(shifted)

    return emitted, overlap_recovered, overlap_duplicates_skipped


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
    absolute_chunk_start_sec = float(sess["sample_cursor"]) / 44100.0

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

    analysis_window_sec = float(x_full.size) / 44100.0
    boundary_micro_notes_filtered = 0
    boundary_micro_chords_filtered = 0
    results["notes"], boundary_micro_notes_filtered = _drop_chunk_end_micro_events(
        results.get("notes") or [],
        analysis_window_sec,
    )
    results["chords"], boundary_micro_chords_filtered = _drop_chunk_end_micro_events(
        results.get("chords") or [],
        analysis_window_sec,
    )

    # Only the most recent OVERLAP_SAMPLES of the prepended context acts as the
    # boundary recovery band; older context audio is calibration-only.
    recovery_band_sec = min(overlap_sec, float(OVERLAP_SAMPLES) / 44100.0)

    def _shift_and_filter_events(evts):
        out = []
        for event in evts or []:
            time_seconds = float(event.get("time_seconds", 0.0))
            if time_seconds < overlap_sec:
                continue

            absolute_time = absolute_chunk_start_sec + (time_seconds - overlap_sec)
            shifted = dict(event)
            shifted["time_seconds"] = round(absolute_time, 6)
            out.append(shifted)
        return out

    shifted_notes, overlap_notes_recovered, overlap_note_duplicates_skipped = _shift_overlap_events_with_recent_dedupe(
        results.get("notes") or [],
        absolute_chunk_start_sec,
        overlap_sec,
        sess.get("recent_notes") or [],
        _note_events_match,
        recovery_band_sec=recovery_band_sec,
    )
    shifted_chords, overlap_chords_recovered, overlap_chord_duplicates_skipped = _shift_overlap_events_with_recent_dedupe(
        results.get("chords") or [],
        absolute_chunk_start_sec,
        overlap_sec,
        sess.get("recent_chords") or [],
        _chord_events_match,
        recovery_band_sec=recovery_band_sec,
    )

    results_filtered = {
        "onsets": _shift_and_filter_events(results.get("onsets")),
        "notes": shifted_notes,
        "chords": shifted_chords,
        "analysis_summary": results.get("analysis_summary", {}),
    }

    results_filtered["analysis_summary"] = {
        **results_filtered["analysis_summary"],
        "total_onsets": len(results_filtered["onsets"]),
        "total_notes": len(results_filtered["notes"]),
        "total_chords": len(results_filtered["chords"]),
    }

    next_sample_cursor = sess["sample_cursor"] + int(x_chunk.size)
    next_cursor_sec = float(next_sample_cursor) / 44100.0
    sess["recent_notes"] = _trim_recent_events(
        _dedupe_note_events([*(sess.get("recent_notes") or []), *results_filtered["notes"]], NOTE_EVENT_DEDUPE_TOLERANCE_SEC),
        next_cursor_sec,
    )
    sess["recent_chords"] = _trim_recent_events(
        _dedupe_chord_events([*(sess.get("recent_chords") or []), *results_filtered["chords"]], CHORD_EVENT_DEDUPE_TOLERANCE_SEC),
        next_cursor_sec,
    )
    sess["sample_cursor"] = next_sample_cursor

    take = min(CONTEXT_SAMPLES, x_full.size)
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
    if (
        overlap_notes_recovered
        or overlap_note_duplicates_skipped
        or overlap_chords_recovered
        or overlap_chord_duplicates_skipped
    ):
        results_filtered["stream_info"]["overlap_recovery"] = {
            "notes_recovered": overlap_notes_recovered,
            "note_duplicates_skipped": overlap_note_duplicates_skipped,
            "chords_recovered": overlap_chords_recovered,
            "chord_duplicates_skipped": overlap_chord_duplicates_skipped,
        }
    if boundary_micro_notes_filtered or boundary_micro_chords_filtered:
        results_filtered["stream_info"]["boundary_micro_filter"] = {
            "notes_filtered": boundary_micro_notes_filtered,
            "chords_filtered": boundary_micro_chords_filtered,
            "guard_sec": CHUNK_END_GUARD_SEC,
            "max_duration_sec": CHUNK_END_MICRO_EVENT_MAX_DURATION_SEC,
        }

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


@app.websocket("/live/stream")
async def live_stream_websocket(websocket: WebSocket):
    """Continuous live audio stream endpoint.

    Message protocol:
      {"type": "start", "session_id": "...", "sample_rate": 44100,
       "inference_interval_ms": 100, "trusted_delay_ms": 180}

      {"type": "audio_packet", "pcm16_base64": "...", "sample_rate": 44100}
      or
      {"type": "audio_packet", "samples": [float, ...], "sample_rate": 44100}

      {"type": "warmup"}
      {"type": "flush"}
      {"type": "stop"}
    """
    await websocket.accept()
    session: Optional[ContinuousLiveStreamSession] = None
    session_id: Optional[str] = None
    send_lock = asyncio.Lock()
    audio_event = asyncio.Event()
    stop_event = asyncio.Event()
    inference_task: Optional[asyncio.Task] = None

    async def send_json(payload: Dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def inference_worker() -> None:
        try:
            while not stop_event.is_set():
                await audio_event.wait()
                audio_event.clear()
                if stop_event.is_set():
                    break

                while not stop_event.is_set():
                    current_session = session
                    if current_session is None:
                        break

                    update = await run_in_threadpool(current_session.maybe_run_inference, False)
                    if update is not None:
                        await send_json(update)

                    # If packets arrived during inference, process the newest
                    # buffer immediately; otherwise wait for the next packet.
                    if not audio_event.is_set():
                        break
                    audio_event.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("live_stream inference_worker error: %s", exc, exc_info=True)

    def ensure_inference_worker() -> None:
        nonlocal inference_task
        if inference_task is None or inference_task.done():
            stop_event.clear()
            inference_task = asyncio.create_task(inference_worker())

    try:
        while True:
            message = await websocket.receive_json()
            message_type = str(message.get("type") or "audio_packet")

            if message_type == "start":
                session_id = str(message.get("session_id") or f"continuous-{int(time.time() * 1000)}")
                sample_rate = int(message.get("sample_rate") or 44100)
                session = _get_continuous_live_stream_session(
                    session_id,
                    sample_rate=sample_rate,
                    context_sec=message.get("context_sec"),
                    inference_interval_ms=message.get("inference_interval_ms"),
                    trusted_delay_ms=message.get("trusted_delay_ms"),
                    commit_delay_ms=message.get("commit_delay_ms"),
                    lock_delay_ms=message.get("lock_delay_ms"),
                )
                ensure_inference_worker()
                await send_json({
                    "type": "live_stream_started",
                    "session": session.status(),
                })
                continue

            if session is None:
                session_id = str(message.get("session_id") or f"continuous-{int(time.time() * 1000)}")
                session = _get_continuous_live_stream_session(
                    session_id,
                    sample_rate=int(message.get("sample_rate") or 44100),
                )
                ensure_inference_worker()

            if message_type in {"warmup", "warm"}:
                warmup_result = await run_in_threadpool(session.warmup_live_path)
                await send_json({
                    "type": "live_stream_warmed",
                    "session": session.status(),
                    "warmup": warmup_result,
                })
                continue

            if message_type in {"stop", "close"}:
                stop_event.set()
                audio_event.set()
                if inference_task is not None and not inference_task.done():
                    try:
                        await asyncio.wait_for(inference_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        inference_task.cancel()
                update = await run_in_threadpool(session.maybe_run_inference, True)
                if update is not None:
                    await send_json(update)
                await send_json({
                    "type": "live_stream_stopped",
                    "session": session.status(),
                })
                break

            if message_type == "flush":
                update = await run_in_threadpool(session.maybe_run_inference, True)
                if update is not None:
                    await send_json(update)
                else:
                    await send_json({
                        "type": "live_stream_update",
                        "session": session.status(),
                        "inference": {"ran": False, "reason": "flush_noop"},
                    })
                continue

            if message_type not in {"audio_packet", "audio"}:
                await send_json({
                    "type": "live_stream_error",
                    "error": f"Unsupported message type: {message_type}",
                })
                continue

            server_received_at_ms = time.time() * 1000.0
            audio, _ = _decode_stream_packet_audio(message, target_sr=session.sample_rate)
            session.append_audio(
                audio,
                {
                    "sequence_number": message.get("sequence_number"),
                    "client_sent_at_ms": message.get("client_sent_at_ms"),
                    "server_received_at_ms": server_received_at_ms,
                },
            )
            ensure_inference_worker()
            audio_event.set()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("live_stream_websocket error: %s", exc, exc_info=True)
        try:
            await send_json({
                "type": "live_stream_error",
                "error": str(exc),
            })
        except Exception:
            pass
    finally:
        stop_event.set()
        audio_event.set()
        if inference_task is not None and not inference_task.done():
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if session_id:
            # Keep the session available for a short reconnect/debug window unless
            # the client explicitly requested deletion through existing live APIs.
            pass


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
        from gpu_ops import (get_gpu_enhanced_mel_transcriber,
                             get_gpu_mel_baseline_transcriber,
                             get_gpu_rhythm_model, get_gpu_transcriber,
                             get_gpu_transformer_model)

        enhanced = get_gpu_enhanced_mel_transcriber()
        ensemble = get_gpu_mel_baseline_transcriber()
        rhythm = get_gpu_rhythm_model()
        transformer = get_gpu_transformer_model()
        transcriber = get_gpu_transcriber()
        warmup_audio = np.zeros(MIN_STREAM_ANALYSIS_SAMPLES, dtype=np.float32)
        analyzer = analyze_audio_optimized if USE_OPTIMIZED_PIPELINE else analyze_audio
        await run_in_threadpool(analyzer, warmup_audio, False)
        
        return {
            "status": "warm",
            "enhanced_mel_model": enhanced is not None and enhanced.initialized,
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
            display_state, display_state_ms = _get_timed_display_state(session)
            response["all_notes"] = display_state.get("notes", [])
            response["all_chords"] = display_state.get("chords", [])
            response.setdefault("_timing_ms", {})["display_state"] = round(display_state_ms, 2)

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
        display_state, display_state_ms = _get_timed_display_state(session)
        result["all_notes"] = display_state.get("notes", [])
        result["all_chords"] = display_state.get("chords", [])
        result.setdefault("_timing_ms", {})["display_state"] = round(display_state_ms, 2)
    
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
        display_state, display_state_ms = _get_timed_display_state(session)
        payload["all_notes"] = display_state.get("notes", [])
        payload["all_chords"] = display_state.get("chords", [])
        payload.setdefault("_timing_ms", {})["display_state"] = round(display_state_ms, 2)

    return JSONResponse(content=make_json_serializable(payload))


@app.post("/live/get-all-notes")
async def live_get_all_notes(req: LiveSessionQuery):
    """
    Get all notes with best available quantization.
    
    Use this when the frontend needs to refresh the full score.
    """
    session = get_live_session(req.session_id)
    display_state, display_state_ms = _get_timed_display_state(session)
    bpm, confidence = session.get_current_bpm()
    
    return JSONResponse(content=make_json_serializable({
        "notes": display_state.get("notes", []),
        "chords": display_state.get("chords", []),
        "bpm": bpm,
        "bpm_confidence": confidence,
        "beat_grid": session.grid_payload(),
        "refinement_version": session.refinement_state.get_refinement_version(),
        "_timing_ms": {"display_state": round(display_state_ms, 2)},
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
    
