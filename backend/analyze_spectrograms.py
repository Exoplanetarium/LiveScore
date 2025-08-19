#!/usr/bin/env python3
"""
Spectrogram Analysis Tool
Displays spectrograms of test audio files to visualize differences between
rapid single notes and chords for improving chord/note detection logic.
"""

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

# Audio processing parameters
SAMPLE_RATE = 22050
HOP_SIZE = 512
FRAME_SIZE = 2048
N_FFT = 2048

def load_audio(file_path):
    """Load audio file with error handling"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        print(f"✓ Loaded {file_path}")
        print(f"  Duration: {len(audio) / sr:.2f}s, Sample rate: {sr}Hz")
        return audio
    except Exception as e:
        print(f"✗ Failed to load {file_path}: {e}")
        return None

def compute_chroma_features(audio):
    """Compute chroma features for analysis"""
    chroma = librosa.feature.chroma_cqt(
        y=audio, 
        sr=SAMPLE_RATE, 
        hop_length=HOP_SIZE,
        bins_per_octave=12
    )
    return chroma

def compute_spectral_features(audio):
    """Compute various spectral features for analysis"""
    # Standard STFT spectrogram
    D = librosa.stft(audio, hop_length=HOP_SIZE, n_fft=N_FFT)
    magnitude = np.abs(D)
    
    # CQT spectrogram
    C = np.abs(librosa.cqt(
        y=audio, 
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        n_bins=84,  # 7 octaves * 12 bins
        bins_per_octave=12,
        fmin=librosa.note_to_hz('C1')
    ))
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        n_fft=N_FFT,
        n_mels=128
    )
    
    return magnitude, C, mel

def analyze_onset_characteristics(audio):
    """Analyze onset characteristics to understand temporal patterns"""
    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        units='frames'
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=SAMPLE_RATE, hop_length=HOP_SIZE)
    
    # Calculate intervals between onsets
    if len(onset_times) > 1:
        intervals = np.diff(onset_times)
        avg_interval = np.mean(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
    else:
        intervals = []
        avg_interval = min_interval = max_interval = 0
    
    return onset_frames, onset_times, intervals, avg_interval, min_interval, max_interval

def create_spectrogram_comparison():
    """Create comprehensive spectrogram comparison of test files"""
    
    # File paths
    base_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(base_dir, 'audio')
    
    chromatic_path = os.path.join(audio_dir, 'test_simple_chord.wav')
    chords_path = os.path.join(audio_dir, 'test_sustained.wav')
    
    print("🎼 Spectrogram Analysis Tool")
    print("=" * 50)
    
    # Load audio files
    audio_chromatic = load_audio(chromatic_path)
    audio_chords = load_audio(chords_path)
    
    if audio_chromatic is None or audio_chords is None:
        print("Failed to load one or both audio files")
        return
    
    # Compute features for both files
    print("\n📊 Computing spectral features...")
    
    # Chromatic run analysis
    print("  Analyzing chromatic run...")
    chroma_chromatic = compute_chroma_features(audio_chromatic)
    stft_chromatic, cqt_chromatic, mel_chromatic = compute_spectral_features(audio_chromatic)
    onsets_chr, onset_times_chr, intervals_chr, avg_int_chr, min_int_chr, max_int_chr = analyze_onset_characteristics(audio_chromatic)
    
    # Chords analysis
    print("  Analyzing chords...")
    chroma_chords = compute_chroma_features(audio_chords)
    stft_chords, cqt_chords, mel_chords = compute_spectral_features(audio_chords)
    onsets_ch, onset_times_ch, intervals_ch, avg_int_ch, min_int_ch, max_int_ch = analyze_onset_characteristics(audio_chords)
    
    # Print onset analysis
    print(f"\n🎵 Onset Analysis:")
    print(f"  Chromatic run: {len(onsets_chr)} onsets, avg interval: {avg_int_chr:.3f}s ({min_int_chr:.3f}-{max_int_chr:.3f}s)")
    print(f"  Chords:        {len(onsets_ch)} onsets, avg interval: {avg_int_ch:.3f}s ({min_int_ch:.3f}-{max_int_ch:.3f}s)")
    
    # Create comprehensive visualization
    print("\n📈 Creating visualizations...")
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Spectrogram Comparison: Chromatic Run vs Chords', fontsize=16, fontweight='bold')
    
    # Time axes for plotting
    time_chromatic = librosa.frames_to_time(np.arange(chroma_chromatic.shape[1]), sr=SAMPLE_RATE, hop_length=HOP_SIZE)
    time_chords = librosa.frames_to_time(np.arange(chroma_chords.shape[1]), sr=SAMPLE_RATE, hop_length=HOP_SIZE)
    
    # Row 1: Waveforms
    ax1 = plt.subplot(6, 2, 1)
    time_samples_chr = np.arange(len(audio_chromatic)) / SAMPLE_RATE
    plt.plot(time_samples_chr, audio_chromatic, alpha=0.7)
    plt.title('Chromatic Run - Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    # Mark onsets
    for onset_time in onset_times_chr:
        plt.axvline(x=onset_time, color='red', alpha=0.6, linestyle='--', linewidth=1)
    
    ax2 = plt.subplot(6, 2, 2)
    time_samples_ch = np.arange(len(audio_chords)) / SAMPLE_RATE
    plt.plot(time_samples_ch, audio_chords, alpha=0.7)
    plt.title('Chords - Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    # Mark onsets
    for onset_time in onset_times_ch:
        plt.axvline(x=onset_time, color='red', alpha=0.6, linestyle='--', linewidth=1)
    
    # Row 2: STFT Spectrograms
    ax3 = plt.subplot(6, 2, 3)
    librosa.display.specshow(
        librosa.amplitude_to_db(stft_chromatic, ref=np.max),
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        x_axis='time',
        y_axis='hz',
        cmap='magma'
    )
    plt.title('Chromatic Run - STFT Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    ax4 = plt.subplot(6, 2, 4)
    librosa.display.specshow(
        librosa.amplitude_to_db(stft_chords, ref=np.max),
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        x_axis='time',
        y_axis='hz',
        cmap='magma'
    )
    plt.title('Chords - STFT Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # Row 3: CQT Spectrograms
    ax5 = plt.subplot(6, 2, 5)
    librosa.display.specshow(
        librosa.amplitude_to_db(cqt_chromatic, ref=np.max),
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        x_axis='time',
        y_axis='cqt_note',
        cmap='magma'
    )
    plt.title('Chromatic Run - CQT Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    ax6 = plt.subplot(6, 2, 6)
    librosa.display.specshow(
        librosa.amplitude_to_db(cqt_chords, ref=np.max),
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        x_axis='time',
        y_axis='cqt_note',
        cmap='magma'
    )
    plt.title('Chords - CQT Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # Row 4: Chroma Features
    ax7 = plt.subplot(6, 2, 7)
    librosa.display.specshow(
        chroma_chromatic,
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        x_axis='time',
        y_axis='chroma',
        cmap='Blues'
    )
    plt.title('Chromatic Run - Chroma Features')
    plt.colorbar()
    
    ax8 = plt.subplot(6, 2, 8)
    librosa.display.specshow(
        chroma_chords,
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        x_axis='time',
        y_axis='chroma',
        cmap='Blues'
    )
    plt.title('Chords - Chroma Features')
    plt.colorbar()
    
    # Row 5: Chroma Analysis - Peak distribution
    ax9 = plt.subplot(6, 2, 9)
    # Analyze chroma peak distribution over time
    peak_ratios_chr = []
    for i in range(chroma_chromatic.shape[1]):
        frame = chroma_chromatic[:, i]
        sorted_vals = np.sort(frame)[::-1]
        if sorted_vals[0] > 0:
            peak_ratios_chr.append(sorted_vals[1] / sorted_vals[0])
        else:
            peak_ratios_chr.append(0)
    
    plt.plot(time_chromatic, peak_ratios_chr, alpha=0.7, linewidth=2)
    plt.title('Chromatic Run - Chroma Peak Ratio (2nd/1st peak)')
    plt.xlabel('Time (s)')
    plt.ylabel('Peak Ratio')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chord threshold (0.5)')
    plt.legend()
    
    ax10 = plt.subplot(6, 2, 10)
    peak_ratios_ch = []
    for i in range(chroma_chords.shape[1]):
        frame = chroma_chords[:, i]
        sorted_vals = np.sort(frame)[::-1]
        if sorted_vals[0] > 0:
            peak_ratios_ch.append(sorted_vals[1] / sorted_vals[0])
        else:
            peak_ratios_ch.append(0)
    
    plt.plot(time_chords, peak_ratios_ch, alpha=0.7, linewidth=2)
    plt.title('Chords - Chroma Peak Ratio (2nd/1st peak)')
    plt.xlabel('Time (s)')
    plt.ylabel('Peak Ratio')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chord threshold (0.5)')
    plt.legend()
    
    # Row 6: Spectral characteristics comparison
    ax11 = plt.subplot(6, 2, 11)
    # Spectral centroid and bandwidth
    spectral_centroids_chr = librosa.feature.spectral_centroid(y=audio_chromatic, sr=SAMPLE_RATE, hop_length=HOP_SIZE)[0]
    spectral_bandwidth_chr = librosa.feature.spectral_bandwidth(y=audio_chromatic, sr=SAMPLE_RATE, hop_length=HOP_SIZE)[0]
    
    plt.plot(time_chromatic, spectral_centroids_chr, label='Spectral Centroid', alpha=0.8)
    plt.plot(time_chromatic, spectral_bandwidth_chr, label='Spectral Bandwidth', alpha=0.8)
    plt.title('Chromatic Run - Spectral Characteristics')
    plt.xlabel('Time (s)')
    plt.ylabel('Hz')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax12 = plt.subplot(6, 2, 12)
    spectral_centroids_ch = librosa.feature.spectral_centroid(y=audio_chords, sr=SAMPLE_RATE, hop_length=HOP_SIZE)[0]
    spectral_bandwidth_ch = librosa.feature.spectral_bandwidth(y=audio_chords, sr=SAMPLE_RATE, hop_length=HOP_SIZE)[0]
    
    plt.plot(time_chords, spectral_centroids_ch, label='Spectral Centroid', alpha=0.8)
    plt.plot(time_chords, spectral_bandwidth_ch, label='Spectral Bandwidth', alpha=0.8)
    plt.title('Chords - Spectral Characteristics')
    plt.xlabel('Time (s)')
    plt.ylabel('Hz')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for main title
    
    # Save the plot
    output_path = os.path.join(base_dir, 'spectrogram_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Show statistics
    print(f"\n📈 Statistical Analysis:")
    print(f"Chromatic Run:")
    print(f"  Peak ratio mean: {np.mean(peak_ratios_chr):.3f} ± {np.std(peak_ratios_chr):.3f}")
    print(f"  Peak ratio >0.5: {np.sum(np.array(peak_ratios_chr) > 0.5)}/{len(peak_ratios_chr)} frames ({np.sum(np.array(peak_ratios_chr) > 0.5)/len(peak_ratios_chr)*100:.1f}%)")
    print(f"  Spectral centroid mean: {np.mean(spectral_centroids_chr):.0f} Hz")
    print(f"  Spectral bandwidth mean: {np.mean(spectral_bandwidth_chr):.0f} Hz")
    
    print(f"\nChords:")
    print(f"  Peak ratio mean: {np.mean(peak_ratios_ch):.3f} ± {np.std(peak_ratios_ch):.3f}")
    print(f"  Peak ratio >0.5: {np.sum(np.array(peak_ratios_ch) > 0.5)}/{len(peak_ratios_ch)} frames ({np.sum(np.array(peak_ratios_ch) > 0.5)/len(peak_ratios_ch)*100:.1f}%)")
    print(f"  Spectral centroid mean: {np.mean(spectral_centroids_ch):.0f} Hz")
    print(f"  Spectral bandwidth mean: {np.mean(spectral_bandwidth_ch):.0f} Hz")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_spectrogram_comparison()
