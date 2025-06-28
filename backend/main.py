from io import BytesIO
import os

from fastapi.concurrency import run_in_threadpool
import numpy as np
import soundfile as sf
import uvicorn
import librosa
from detect_note import analyze_audio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
        import binascii, tempfile

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

        bio = BytesIO(data)
        audio, sr = librosa.load(bio, sr=44100, mono=True)

        try:
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 44.1kHz if needed
            if sr != 44100:
                from scipy.signal import resample
                target_length = int(len(audio) * 44100 / sr)
                audio = resample(audio, target_length)
            
            # Analyze the audio
            results = await run_in_threadpool(analyze_audio, audio, debug)
            
            # Add metadata about the uploaded file
            results["file_info"] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "original_sample_rate": sr,
                "processed_sample_rate": 44100,
                "channels": 1 if audio.ndim == 1 else audio.shape[1]
            }
            
            return JSONResponse(content=results)
            
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
        
        # Analyze the audio
        results = analyze_audio(audio, debug=debug)
        
        # Add metadata
        results["file_info"] = {
            "filename": "recorded_audio",
            "content_type": "audio/raw",
            "original_sample_rate": sample_rate,
            "processed_sample_rate": 44100,
            "channels": 1
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Raw audio analysis failed: {str(e)}"
        )

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
