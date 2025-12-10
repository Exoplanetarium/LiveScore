import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import React, { useEffect, useRef, useState } from 'react';
import { Alert, PermissionsAndroid, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import AudioRecord from 'react-native-audio-record';
import PianoSheetMusic from '../components/PianoSheetMusic';
import { ThemedText } from '../components/ThemedText';
import { ThemedView } from '../components/ThemedView';

const BACKEND_URL = 'https://livescore-production-4dfa.up.railway.app';

// Types for analysis results
interface NoteResult {
  time_seconds: number;
  frame_index: number;
  midi_note: number;
  note_name: string;
  frequency_hz: number;
  method: string;
  confidence: number;
}

interface ChordResult {
  time_seconds: number;
  frame_index: number;
  type: string;
  chord_quality: string;
  label: string;
  inversion: string;
  confidence: number;
}

interface AnalysisResult {
  onsets: { time_seconds: number; frame_index: number }[];
  notes: NoteResult[];
  chords: ChordResult[];
  analysis_summary: {
    total_onsets: number;
    total_notes: number;
    total_chords: number;
    duration_seconds: number;
    sample_rate: number;
    bass_notes?: number;  // Optional: number of bass notes when using split analysis
    treble_notes?: number;  // Optional: number of treble notes when using split analysis
    detected_bpm?: number;  // Detected tempo from audio analysis
    tempo_confidence?: number;  // Confidence of tempo detection (0-1)
    beat_interval?: number;  // Beat interval in seconds
  };
}

export default function AnalyzeScreen() {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [recordedAudioPath, setRecordedAudioPath] = useState<string | null>(null);
  
  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<{
    onsets: number[];
    notes: string[];
    confidence: number;
    method: string;
    details?: AnalysisResult;
  } | null>(null);
  
  // Playback state
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  
  // Tempo multiplier state (1 = normal, 2 = double tempo/half note values, 0.5 = half tempo/double note values)
  const [tempoMultiplier, setTempoMultiplier] = useState(1);
  
  // Refs
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioChunksRef = useRef<string[]>([]);

  // Initialize AudioRecord on mount
  useEffect(() => {
    const audioOptions = {
      sampleRate: 44100,
      channels: 1,
      bitsPerSample: 16,
      audioSource: 6,
      wavFile: 'temp_audio.wav',
    };
    AudioRecord.init(audioOptions);

    return () => {
      try {
        AudioRecord.stop();
      } catch (error) {
        console.warn('Failed to stop AudioRecord:', error);
      }
    };
  }, []);

  // Request microphone permissions
  const requestPermissions = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: 'Microphone Permission',
            message: 'This app needs access to your microphone to record audio.',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          }
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        return false;
      }
    }
    return true;
  };

  // Start recording
  const startRecording = async () => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        Alert.alert('Permission Required', 'Please grant microphone permissions to record audio.');
        return;
      }

      setIsRecording(true);
      setDuration(0);
      audioChunksRef.current = [];
      setAnalysisResults(null);
      setRecordedAudioPath(null);
      
      if (sound) {
        await sound.unloadAsync();
        setSound(null);
      }
      
      AudioRecord.start();
      console.log('🎙️ Recording started');

      intervalRef.current = setInterval(() => {
        setDuration(prev => prev + 0.1);
      }, 100) as unknown as NodeJS.Timeout;
      
    } catch (err) {
      console.error('Failed to start recording:', err);
      Alert.alert('Error', 'Failed to start recording. Please try again.');
      setIsRecording(false);
    }
  };

  // Stop recording and analyze
  const stopRecording = async () => {
    try {
      console.log('🛑 Stopping recording...');
      setIsRecording(false);
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      const audioFile = await AudioRecord.stop();
      console.log('📦 Recording stopped, AudioRecord returned:', audioFile);
      console.log('📂 FileSystem.documentDirectory:', FileSystem.documentDirectory);
      console.log('📂 FileSystem.cacheDirectory:', FileSystem.cacheDirectory);

      // Find the actual file location (AudioRecord.stop() may return just filename or full path)
      let actualFilePath = audioFile;
      
      // Check if it's already a full path (starts with file://)
      if (!audioFile.startsWith('file://')) {
        // Try multiple possible locations
        const possiblePaths = [
          `file://${audioFile}`, // Direct file path
          `${FileSystem.documentDirectory}${audioFile}`, // Documents directory
          `${FileSystem.cacheDirectory}${audioFile}`, // Cache directory
        ];
        
        console.log('🔍 Checking possible file locations...');
        for (const path of possiblePaths) {
          try {
            const fileInfo = await FileSystem.getInfoAsync(path);
            if (fileInfo.exists) {
              console.log('✅ Found audio file at:', path);
              actualFilePath = path;
              break;
            } else {
              console.log('❌ Not found at:', path);
            }
          } catch {
            console.log('❌ Error checking:', path);
          }
        }
      }
      
      // Verify the file exists
      const fileInfo = await FileSystem.getInfoAsync(actualFilePath);
      if (!fileInfo.exists) {
        throw new Error(`Recording file not found at: ${actualFilePath}`);
      }
      
      console.log('✅ Verified recording file exists, size:', fileInfo.size);

      // Save a permanent copy for playback
      const outputPath = `${FileSystem.documentDirectory}recording_${Date.now()}.wav`;
      await FileSystem.copyAsync({
        from: actualFilePath,
        to: outputPath,
      });
      
      setRecordedAudioPath(outputPath);
      console.log('✅ Recording saved to:', outputPath);
      
      // Analyze the recording
      await analyzeRecording(outputPath);
      
    } catch (err) {
      console.error('Failed to stop recording:', err);
      Alert.alert('Error', 'Failed to stop recording.');
    }
  };

  // Analyze the recorded audio
  const analyzeRecording = async (audioPath: string) => {
    setIsAnalyzing(true);
    try {
      console.log('🔍 Analyzing recording:', audioPath);
      
      const formData = new FormData();
      formData.append('file', {
        uri: audioPath,
        name: 'recording.wav',
        type: 'audio/wave',
      } as any);
      
      console.log('📤 Sending to backend...');
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error:', errorText);
        throw new Error(`Server error: ${response.status}`);
      }

      const results: AnalysisResult = await response.json();
      console.log('✅ Analysis complete:', results.analysis_summary);

      setAnalysisResults({
        onsets: results.onsets.map(onset => onset.time_seconds),
        notes: results.notes.map(note => note.note_name),
        confidence: results.notes.length > 0 
          ? results.notes.reduce((acc, note) => acc + note.confidence, 0) / results.notes.length 
          : 0,
        method: results.notes.length > 0 ? results.notes[0].method : 'No detection',
        details: results,
      });
      
      Alert.alert('Success', `Detected ${results.notes.length} notes and ${results.chords.length} chords!`);
    } catch (err) {
      console.error('Analysis error:', err);
      Alert.alert('Error', 'Failed to analyze recording. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Play the recorded audio
  const playRecording = async () => {
    if (!recordedAudioPath) return;
    
    try {
      if (sound) {
        await sound.unloadAsync();
      }
      
      const { sound: newSound } = await Audio.Sound.createAsync(
        { uri: recordedAudioPath },
        { shouldPlay: true }
      );
      
      setSound(newSound);
      setIsPlaying(true);
      
      newSound.setOnPlaybackStatusUpdate((status: any) => {
        if (status.isLoaded) {
          setPlaybackPosition(status.positionMillis);
          setPlaybackDuration(status.durationMillis || 0);
          
          if (status.didJustFinish) {
            setIsPlaying(false);
          }
        }
      });
      
      await newSound.playAsync();
    } catch (error) {
      console.error('Error playing audio:', error);
      Alert.alert('Error', 'Failed to play recording.');
    }
  };

  // Pause playback
  const pauseRecording = async () => {
    if (sound) {
      await sound.pauseAsync();
      setIsPlaying(false);
    }
  };

  // Stop playback
  const stopPlayback = async () => {
    if (sound) {
      await sound.stopAsync();
      await sound.setPositionAsync(0);
      setIsPlaying(false);
      setPlaybackPosition(0);
    }
  };

  // Format duration display
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };

  // Format milliseconds to mm:ss
  const formatMilliseconds = (ms: number) => {
    const totalSeconds = ms / 1000;
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <ScrollView 
      style={{ flex: 1 }}
      contentContainerStyle={{ flexGrow: 1 }}
      showsVerticalScrollIndicator={true}
      keyboardShouldPersistTaps="handled"
    >
      <ThemedView style={styles.container}>
        <ThemedText type="title" style={styles.title}>
          LiveScore Piano
        </ThemedText>
        <ThemedText style={styles.subtitle}>
          Record piano music and generate sheet music
        </ThemedText>

        {/* Recording Control */}
        <View style={styles.recordingSection}>
          {!isRecording && !recordedAudioPath && (
            <TouchableOpacity 
              style={[styles.recordButton, styles.startButton]}
              onPress={startRecording}
            >
              <Ionicons name="mic" size={32} color="white" />
              <Text style={styles.recordButtonText}>Start Recording</Text>
            </TouchableOpacity>
          )}
          
          {isRecording && (
            <View style={styles.recordingActive}>
              <View style={styles.recordingIndicator}>
                <View style={styles.recordingDot} />
                <Text style={styles.recordingText}>Recording...</Text>
              </View>
              <Text style={styles.durationText}>{formatDuration(duration)}</Text>
              <TouchableOpacity 
                style={[styles.recordButton, styles.stopButton]}
                onPress={stopRecording}
              >
                <Ionicons name="stop" size={32} color="white" />
                <Text style={styles.recordButtonText}>Stop Recording</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Analysis Progress */}
          {isAnalyzing && (
            <View style={styles.analyzingSection}>
              <Ionicons name="analytics" size={24} color="#2196F3" />
              <Text style={styles.analyzingText}>Analyzing recording...</Text>
            </View>
          )}
        </View>

        {/* Playback Controls */}
        {recordedAudioPath && !isRecording && (
          <View style={styles.playbackSection}>
            <View style={styles.playbackControls}>
              {!isPlaying ? (
                <TouchableOpacity style={styles.playButton} onPress={playRecording}>
                  <Ionicons name="play" size={32} color="white" />
                </TouchableOpacity>
              ) : (
                <TouchableOpacity style={styles.playButton} onPress={pauseRecording}>
                  <Ionicons name="pause" size={32} color="white" />
                </TouchableOpacity>
              )}
              
              <TouchableOpacity style={styles.stopSmallButton} onPress={stopPlayback}>
                <Ionicons name="stop" size={20} color="white" />
              </TouchableOpacity>
              
              <View style={styles.playbackInfo}>
                <Text style={styles.playbackTime}>
                  {formatMilliseconds(playbackPosition)} / {formatMilliseconds(playbackDuration)}
                </Text>
              </View>
            </View>

            {/* Action Buttons */}
            <View style={styles.actionButtonsRow}>
              <TouchableOpacity 
                style={styles.newRecordingButton}
                onPress={() => {
                  setRecordedAudioPath(null);
                  setAnalysisResults(null);
                  setPlaybackPosition(0);
                  setPlaybackDuration(0);
                  if (sound) {
                    sound.unloadAsync();
                    setSound(null);
                  }
                }}
              >
                <Ionicons name="add-circle-outline" size={20} color="#2196F3" />
                <Text style={styles.newRecordingText}>New Recording</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.deleteRecordingButton}
                onPress={async () => {
                  if (recordedAudioPath) {
                    try {
                      await FileSystem.deleteAsync(recordedAudioPath);
                      console.log('🗑️ Deleted recording:', recordedAudioPath);
                      setRecordedAudioPath(null);
                      setAnalysisResults(null);
                      setPlaybackPosition(0);
                      setPlaybackDuration(0);
                      if (sound) {
                        await sound.unloadAsync();
                        setSound(null);
                      }
                      Alert.alert('Deleted', 'Recording deleted successfully');
                    } catch (error) {
                      console.error('Failed to delete recording:', error);
                      Alert.alert('Error', 'Failed to delete recording');
                    }
                  }
                }}
              >
                <Ionicons name="trash-outline" size={20} color="#f44336" />
                <Text style={styles.deleteRecordingText}>Delete</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Sheet Music Display */}
        {/* Analysis Summary - Compact */}
        {analysisResults && (
          <View style={styles.resultsSummary}>
            <View style={styles.statsRowCompact}>
              <View style={styles.statBoxCompact}>
                <Ionicons name="musical-notes" size={18} color="#4CAF50" />
                <Text style={styles.statValueCompact}>{analysisResults.notes.length}</Text>
                <Text style={styles.statLabelCompact}>Notes</Text>
              </View>
              <View style={styles.statBoxCompact}>
                <Ionicons name="pulse" size={18} color="#2196F3" />
                <Text style={styles.statValueCompact}>{analysisResults.details?.analysis_summary.total_onsets}</Text>
                <Text style={styles.statLabelCompact}>Onsets</Text>
              </View>
              {analysisResults.details?.analysis_summary.bass_notes !== undefined && (
                <>
                  <View style={styles.statBoxCompact}>
                    <Text style={styles.statValueCompact}>{analysisResults.details.analysis_summary.bass_notes}</Text>
                    <Text style={styles.statLabelCompact}>Bass</Text>
                  </View>
                  <View style={styles.statBoxCompact}>
                    <Text style={styles.statValueCompact}>{analysisResults.details.analysis_summary.treble_notes}</Text>
                    <Text style={styles.statLabelCompact}>Treble</Text>
                  </View>
                </>
              )}
            </View>
          </View>
        )}

        {/* Sheet Music - Main Focus */}
        {analysisResults && analysisResults.details && analysisResults.notes.length > 0 && (
          <View style={styles.sheetMusicSection}>
            <ThemedText style={styles.sectionTitle}>Sheet Music</ThemedText>
            
            {/* Detected Tempo Display */}
            {analysisResults.details?.analysis_summary?.detected_bpm && (
              <View style={styles.detectedTempoSection}>
                <Text style={styles.detectedTempoLabel}>
                  Detected Tempo: {Math.round(analysisResults.details.analysis_summary.detected_bpm)} BPM
                  {analysisResults.details.analysis_summary.tempo_confidence !== undefined && (
                    <Text style={styles.tempoConfidence}>
                      {' '}({Math.round(analysisResults.details.analysis_summary.tempo_confidence * 100)}% confidence)
                    </Text>
                  )}
                </Text>
              </View>
            )}
            
            {/* Tempo Control */}
            <View style={styles.tempoControlSection}>
              <Text style={styles.tempoLabel}>Note Values:</Text>
              <View style={styles.tempoButtonsRow}>
                <TouchableOpacity 
                  style={[styles.tempoButton, tempoMultiplier === 0.5 && styles.tempoButtonActive]}
                  onPress={() => setTempoMultiplier(0.5)}
                >
                  <Text style={[styles.tempoButtonText, tempoMultiplier === 0.5 && styles.tempoButtonTextActive]}>
                    ×2 (Slower)
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.tempoButton, tempoMultiplier === 1 && styles.tempoButtonActive]}
                  onPress={() => setTempoMultiplier(1)}
                >
                  <Text style={[styles.tempoButtonText, tempoMultiplier === 1 && styles.tempoButtonTextActive]}>
                    Normal
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.tempoButton, tempoMultiplier === 2 && styles.tempoButtonActive]}
                  onPress={() => setTempoMultiplier(2)}
                >
                  <Text style={[styles.tempoButtonText, tempoMultiplier === 2 && styles.tempoButtonTextActive]}>
                    ÷2 (Faster)
                  </Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.tempoHint}>
                {
                  tempoMultiplier === 0.5 ? '♪ → 𝅝 (eighth → quarter)' :
                  tempoMultiplier === 2 ? '♪ → ♫ (quarter → eighth)' :
                  'Original note values'
                }
              </Text>
            </View>
            
            <PianoSheetMusic 
              results={analysisResults.details}
              tempoMultiplier={tempoMultiplier}
            />
          </View>
        )}

        {/* Info Section */}
        {!recordedAudioPath && !isRecording && (
          <View style={styles.infoSection}>
            <ThemedText style={styles.infoTitle}>Recording Tips:</ThemedText>
            <ThemedText style={styles.infoText}>
              • Place device close to the piano{'\n'}
              • Play clearly and at a moderate tempo{'\n'}
              • Minimize background noise{'\n'}
              • Record at least a few notes for best results
            </ThemedText>
          </View>
        )}
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    paddingTop: 80,
    paddingBottom: 50,
  },
  title: {
    marginBottom: 8,
    textAlign: 'center',
    fontSize: 28,
    fontWeight: 'bold',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    marginBottom: 40,
    fontSize: 14,
  },

  // Recording Section
  recordingSection: {
    marginBottom: 30,
    alignItems: 'center',
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    gap: 12,
    minWidth: 200,
  },
  startButton: {
    backgroundColor: '#4CAF50',
  },
  stopButton: {
    backgroundColor: '#f44336',
    marginTop: 20,
  },
  recordButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  recordingActive: {
    alignItems: 'center',
    gap: 12,
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    borderRadius: 20,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#f44336',
  },
  recordingText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f44336',
  },
  durationText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  analyzingSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 16,
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    borderRadius: 8,
    marginTop: 20,
  },
  analyzingText: {
    fontSize: 16,
    color: '#2196F3',
    fontWeight: '600',
  },

  // Playback Section
  playbackSection: {
    marginBottom: 30,
    padding: 16,
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    borderRadius: 12,
  },
  playbackControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
  },
  playButton: {
    backgroundColor: '#2196F3',
    width: 60,
    height: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopSmallButton: {
    backgroundColor: '#f44336',
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  playbackInfo: {
    flex: 1,
    marginLeft: 12,
  },
  playbackTime: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
  },
  actionButtonsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  newRecordingButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2196F3',
    borderStyle: 'dashed',
  },
  newRecordingText: {
    color: '#2196F3',
    fontSize: 16,
    fontWeight: '600',
  },
  deleteRecordingButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#f44336',
    borderStyle: 'dashed',
  },
  deleteRecordingText: {
    color: '#f44336',
    fontSize: 16,
    fontWeight: '600',
  },

  // Sheet Music Section
  sheetMusicSection: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },

  // Compact Analysis Summary
  resultsSummary: {
    marginBottom: 16,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(128, 128, 128, 0.05)',
    borderRadius: 8,
  },
  statsRowCompact: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  statBoxCompact: {
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  statValueCompact: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#555',
  },
  statLabelCompact: {
    fontSize: 10,
    color: '#888',
    marginTop: 2,
  },

  // Results Section
  resultsSection: {
    marginBottom: 30,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  statBox: {
    alignItems: 'center',
    padding: 16,
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    borderRadius: 8,
    minWidth: 100,
  },
  splitInfoRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 20,
  },
  splitInfoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(128, 128, 128, 0.05)',
    borderRadius: 8,
  },
  splitInfoText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 8,
    color: '#888',
  },
  statLabel: {
    fontSize: 12,
    opacity: 0.7,
    marginTop: 4,
    color: '#666',
  },
  noteList: {
    maxHeight: 300,
  },
  noteItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: 'rgba(128, 128, 128, 0.05)',
    borderRadius: 8,
    marginBottom: 8,
  },
  noteIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  noteContent: {
    flex: 1,
  },
  noteText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  noteMeta: {
    fontSize: 12,
    color: '#666',
  },

  // Info Section
  infoSection: {
    padding: 20,
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    borderRadius: 8,
    marginTop: 20,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#2196F3',
  },
  infoText: {
    fontSize: 14,
    lineHeight: 22,
    opacity: 0.8,
  },
  
  // Tempo Control
  detectedTempoSection: {
    marginBottom: 12,
    padding: 10,
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#2196F3',
  },
  detectedTempoLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1976D2',
  },
  tempoConfidence: {
    fontSize: 13,
    fontWeight: '400',
    color: '#666',
  },
  tempoControlSection: {
    marginBottom: 16,
    padding: 12,
    backgroundColor: 'rgba(128, 128, 128, 0.05)',
    borderRadius: 8,
  },
  tempoLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
  },
  tempoButtonsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
  },
  tempoButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: '#e0e0e0',
    alignItems: 'center',
  },
  tempoButtonActive: {
    backgroundColor: '#2196F3',
  },
  tempoButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
  },
  tempoButtonTextActive: {
    color: 'white',
  },
  tempoHint: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
    marginTop: 8,
    fontStyle: 'italic',
  },
});
