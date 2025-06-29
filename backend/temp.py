"""
def analyze_audio(wav_path_or_array, debug=False):
    try:
        # Read audio
        if isinstance(wav_path_or_array, str):
            audio = read_wav(wav_path_or_array)
        else:
            audio = wav_path_or_array
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
    except Exception as e:
        return {"error": f"Failed to read audio: {str(e)}"}
    
    def midi_to_name(m):
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{names[m%12]}{(m//12)-1}"
    
    # Traditional onset detection
    frames = frame_audio(audio)
    mags = np.array([compute_magnitude(f) for f in frames])
    flux = normalize(compute_flux(mags))
    traditional_onsets = find_onsets(flux)
    onsets = traditional_onsets
    
    if debug:
        print(f"Using traditional onset detection: {len(onsets)} onsets at frames: {onsets}")
    
    # Chord analysis for the entire audio
    if debug:
        print("Performing chord analysis...")
    

    # Results structure
    results = {
        "onsets": [],
        "notes": [],
        "analysis_summary": {
            "total_onsets": len(onsets),
            "duration_seconds": len(audio) / SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE
        }
    }
    
    # Process each onset
    for i, onset in enumerate(onsets):
        idx = min(onset + 3, len(frames)-1)
        frame = frames[idx]
        
        # Convert frame index to time
        time_seconds = onset * HOP_SIZE / SAMPLE_RATE
        
        if debug:
            print(f"\n=== ONSET {i+1} at frame {onset} ({time_seconds:.2f}s) ===")
        
        # Method 1: FFT-based detection
        fft_note, freqs, fft_mag = detect_fundamental_from_fft(frame)
        
        # Method 2: CQT-based detection
        cqt_mag = compute_cqt(frame)
        
        # Method 3: Simple CQT method
        simple_note = detect_fundamental_simple(cqt_mag, debug=debug)
        
        # Method 4: HPS method
        hps_notes = pick_pitches_HPS(cqt_mag, max_voices=1)
        
        # Method 5: Robust detection with octave correction
        robust_note, robust_method = detect_pitch_robust(frame, cqt_mag, fft_mag, freqs, debug=debug)
        
        # Choose the best method (prefer robust method if available)
        if robust_note:
            final_note = robust_note
            method_used = f"Robust ({robust_method})"
            confidence = 0.9
        elif simple_note and hps_notes and simple_note == hps_notes[0]:
            # Both CQT methods agree
            final_note = simple_note
            method_used = "CQT (consensus)"
            confidence = 0.8
        elif simple_note:
            # Use simple CQT method as primary
            final_note = simple_note
            method_used = "CQT (simple)"
            confidence = 0.7
        elif hps_notes:
            # Fall back to HPS
            final_note = hps_notes[0]
            method_used = "CQT (HPS)"
            confidence = 0.6
        elif fft_note:
            # FFT as last resort
            final_note = fft_note
            method_used = "FFT"
            confidence = 0.5
        else:
            final_note = None
            method_used = "None"
            confidence = 0.0
        
        # Add onset information
        onset_info = {
            "time_seconds": round(time_seconds, 3),
            "frame_index": int(onset)
        }
        results["onsets"].append(onset_info)
        
        # Add note information if detected
        if final_note:
            note_info = {
                "time_seconds": round(time_seconds, 3),
                "frame_index": int(onset),
                "midi_note": int(final_note),
                "note_name": midi_to_name(final_note),
                "frequency_hz": round(440.0 * 2**((final_note - 69)/12), 2),
                "method": method_used,
                "confidence": confidence
            }
            results["notes"].append(note_info)
            
            if debug:
                print(f"FINAL: {midi_to_name(final_note)} - Single note (method: {method_used})")
        else:
            if debug:
                print("FINAL: No note detected")
    
    return results
"""