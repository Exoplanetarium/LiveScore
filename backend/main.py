import logging
import math
import os
from io import BytesIO
from typing import Dict, List

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
from detect_note import analyze_audio, analyze_audio_optimized, read_wav
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
        # Read the uploaded file
        data = await file.read()
        if len(data) > 100*1024*1024:
            raise HTTPException(413, "File too large")
        
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

        audio = read_wav(tmp.name)
        
        # Keep temp file path for neural transcription
        temp_audio_path = tmp.name

        try:
            # Analyze the audio in a threadpool (blocking CPU work)
            # For neural transcription, pass the file path; for traditional, pass the array
            if use_neural:
                results = await run_in_threadpool(
                    analyze_audio, temp_audio_path, debug, 
                    True, True, True, device  # use_split, independent_hands, use_neural, device
                )
            else:
                results = await run_in_threadpool(analyze_audio, audio, debug)
            
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
            try:
                clean_results = make_json_serializable(results)
                print(f"[DEBUG] JSON serialization successful, returning response")
            except Exception as ser_err:
                import traceback
                print(f"[ERROR] JSON serialization failed: {ser_err}")
                traceback.print_exc()
                raise

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
        # Read and decode to PCM float32 mono 44.1k
        data = await file.read()
        x_chunk = _load_bytes_to_pcm(data, target_sr=44100)

        # Get session and build analysis buffer with previous tail for overlap continuity
        sess = _get_session(session_id)
        tail = sess["tail"]
        if tail.size > 0:
            x_full = np.concatenate([tail, x_chunk])
        else:
            x_full = x_chunk

        overlap_sec = float(tail.size) / 44100.0

        # Run the main analyzer on the combined buffer (use optimized if flag is set)
        analyzer = analyze_audio_optimized if USE_OPTIMIZED_PIPELINE else analyze_audio
        results = await run_in_threadpool(analyzer, x_full, debug)

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

        # Pack stream metadata
        results_filtered["stream_info"] = {
            "session_id": session_id,
            "chunk_samples": int(x_chunk.size),
            "overlap_samples": int(tail.size),
            "sample_cursor": int(sess["sample_cursor"]),
            "processed_sample_rate": 44100,
        }

        return JSONResponse(content=make_json_serializable(results_filtered))

    except Exception as e:
        logger.error(f"stream_chunk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"stream_chunk failed: {str(e)}")

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
