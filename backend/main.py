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
from detect_note import (analyze_audio, analyze_audio_optimized, read_wav,
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
        }
        _stream_sessions[session_id] = s
    return s

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
                             get_gpu_rhythm_model)

        ensemble = get_gpu_mel_baseline_transcriber()
        rhythm = get_gpu_rhythm_model()
        
        return {
            "status": "warm",
            "ensemble_model": ensemble is not None and ensemble.initialized,
            "rhythm_model": rhythm is not None,
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
        print(f"[DEBUG] upload filename={file.filename!r}, content_type={file.content_type!r}, size={len(data)} bytes")

        # 2) Log the first few bytes in hex (should start with 'RIFF')
        hex_header = binascii.hexlify(data[:12]).decode('ascii', errors='ignore')
        print(f"[DEBUG] first 12 bytes (hex): {hex_header}")

        # 3) Write a copy to a temp file so you can download/inspect it
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav")
        tmp.write(data)
        tmp.flush()
        print(f"[DEBUG] data upload written to {tmp.name}")
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
    if sid in _stream_sessions:
        del _stream_sessions[sid]
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
        TIMER.start("chunk_total")
        TIMER.start("chunk_decode")
        
        # Read and decode to PCM float32 mono 44.1k
        data = await file.read()
        x_chunk = _load_bytes_to_pcm(data, target_sr=44100)
        chunk_decode_ms = TIMER.stop("chunk_decode")
        chunk_duration_ms = len(x_chunk) / 44100 * 1000

        # Get session and build analysis buffer with previous tail for overlap continuity
        sess = _get_session(session_id)
        tail = sess["tail"]
        if tail.size > 0:
            x_full = np.concatenate([tail, x_chunk])
        else:
            x_full = x_chunk

        overlap_sec = float(tail.size) / 44100.0

        # Run the main analyzer on the combined buffer (use optimized if flag is set)
        TIMER.start("chunk_inference")
        analyzer = analyze_audio_optimized if USE_OPTIMIZED_PIPELINE else analyze_audio
        results = await run_in_threadpool(analyzer, x_full, debug)
        chunk_inference_ms = TIMER.stop("chunk_inference")

        # Filter out detections that lie within the leading overlap region
        def _shift_and_filter_events(evts):
            out = []
            for e in evts or []:
                t = float(e.get("time_seconds", 0.0))
                if t >= overlap_sec:  # keep only beyond overlap
                    # Convert to absolute time based on cursor (exclude overlap)
                    abs_t = (sess["sample_cursor"] / 44100.0) + (t - overlap_sec)
                    e2 = dict(e)
                    e2["time_seconds"] = round(abs_t, 6)
                    out.append(e2)
            return out

        results_filtered = {
            "onsets": _shift_and_filter_events(results.get("onsets")),
            "notes": _shift_and_filter_events(results.get("notes")),
            "chords": _shift_and_filter_events(results.get("chords")),
            "analysis_summary": results.get("analysis_summary", {}),
        }

        # Update session state: advance cursor by the NON-overlap chunk length
        sess["sample_cursor"] += int(x_chunk.size)

        # Keep new tail from the end of the combined buffer
        take = min(OVERLAP_SAMPLES, x_full.size)
        sess["tail"] = x_full[-take:].astype(np.float32, copy=False)

        chunk_total_ms = TIMER.stop("chunk_total")
        
        # Calculate real-time factor (< 1.0 means faster than real-time)
        rtf = chunk_total_ms / chunk_duration_ms if chunk_duration_ms > 0 else 0
        
        # Pack stream metadata with timing
        results_filtered["stream_info"] = {
            "session_id": session_id,
            "chunk_samples": int(x_chunk.size),
            "overlap_samples": int(tail.size),
            "sample_cursor": int(sess["sample_cursor"]),
            "processed_sample_rate": 44100,
        }
        
        results_filtered["_timing_ms"] = {
            "chunk_decode": round(chunk_decode_ms, 2),
            "chunk_inference": round(chunk_inference_ms, 2),
            "chunk_total": round(chunk_total_ms, 2),
            "chunk_audio_duration": round(chunk_duration_ms, 2),
            "real_time_factor": round(rtf, 3),
        }
        
        print(f"[TIMING] /stream/chunk: decode={chunk_decode_ms:.1f}ms, inference={chunk_inference_ms:.1f}ms, TOTAL={chunk_total_ms:.1f}ms | audio={chunk_duration_ms:.0f}ms, RTF={rtf:.2f}x")

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
    return {"status": "reset", "session_id": req.session_id}


@app.post("/live/session/delete")
async def live_session_delete(req: LiveSessionQuery):
    """Delete a live session."""
    deleted = delete_live_session(req.session_id)
    return {"status": "deleted" if deleted else "not_found", "session_id": req.session_id}


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
    refined = session.refinement_state.check_refinement(time.time(), bpm)
    
    return JSONResponse(content=make_json_serializable({
        "needs_refresh": refined is not None and len(refined) > 0,
        "refined_notes": refined or [],
        "refinement_version": session.refinement_state.get_refinement_version(),
        "bpm": bpm,
        "bpm_confidence": confidence,
    }))


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
    
    # Force refinement of all pending notes
    await run_in_threadpool(session.force_refinement)
    
    all_notes = session.get_all_notes()
    bpm, confidence = session.get_current_bpm()
    
    return JSONResponse(content=make_json_serializable({
        "status": "finalized",
        "notes": all_notes,
        "chords": session.coarse_chords,
        "bpm": bpm,
        "bpm_confidence": confidence,
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
    
