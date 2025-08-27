import logging
import os
import hashlib
import json
import math
from io import BytesIO
from typing import List, Optional

# consistency between local and server
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
# Force the same OpenBLAS micro-kernel on both boxes (avoid AVX-512 vs AVX2 drift)
os.environ["OPENBLAS_CORETYPE"] = "HASWELL"   # works on AVX2/AVX-512 machines
os.environ["PYTHONHASHSEED"] = "0"

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from detect_note import analyze_audio
from fastapi import FastAPI, File, HTTPException, UploadFile
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
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_audio_deterministic(path, target_sr=44100):
    # Read raw PCM deterministically
    y, sr = sf.read(path, dtype="float32", always_2d=True)  # shape (N, ch)
    y = y.mean(axis=1).astype(np.float32, copy=False)       # force mono by ourselves

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        y = resample_poly(y, up, down).astype(np.float32, copy=False)  # deterministic polyphase
    return y, target_sr

def read_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 44100:
        raise ValueError(f"Expected {44100} Hz, got {sr}")
    # simple one‐pole HPF: y[n] = x[n] - x[n-1] + alpha y[n-1]
    alpha = 0.95
    y = np.empty_like(audio)
    prev_x, prev_y = audio[0], audio[0]
    y[0] = prev_y
    for i in range(1, len(audio)):
        y[i] = audio[i] - prev_x + alpha * prev_y
        prev_x, prev_y = audio[i], y[i]
    return y

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
    debug: bool = False
):
    """
    Analyze an uploaded audio file and return detected notes and onsets.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        debug: Whether to include debug information in response
        
    Returns:
        JSON with detected notes, onsets, and analysis metadata
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
        os.unlink(tmp.name)

        try:
            # Analyze the audio in a threadpool (blocking CPU work)
            results = await run_in_threadpool(analyze_audio, audio, debug)

            # Add metadata about the uploaded file
            results["file_info"] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "processed_sample_rate": 44100,
                "channels": 1 if (isinstance(audio, np.ndarray) and audio.ndim == 1) else (audio.shape[1] if isinstance(audio, np.ndarray) and audio.ndim > 1 else 1)
            }

            # Ensure JSON serializable
            clean_results = make_json_serializable(results)

            return JSONResponse(content=clean_results)

        except Exception as e:
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
        
        # Analyze the audio using our detection system
        results = await run_in_threadpool(analyze_audio, audio_array, False)
        
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
            "sample_rate": request.sample_rate,
            "duration_seconds": len(request.audio_data) / request.sample_rate,
            "analysis_type": "real_time_stream"
        }

        return JSONResponse(content=clean_results)

    except Exception as e:
        logger.error(f"Streaming analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming analysis failed: {str(e)}")

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
