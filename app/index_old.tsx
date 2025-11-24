import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
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
  };
}

type AnalysisMode = 'file' | 'realtime';

export default function AnalyzeScreen() {
  // Mode selection
  const [mode, setMode] = useState<AnalysisMode>('file');
  
  // File upload states
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<{
    onsets: number[];
    notes: string[];
    confidence: number;
    method: string;
    details?: AnalysisResult;
  } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Real-time recording states
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [detectedNotes, setDetectedNotes] = useState<NoteResult[]>([]);
  const [detectedChords, setDetectedChords] = useState<ChordResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  
  // Refs for intervals
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const analysisTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isRecordingRef = useRef<boolean>(false); // Track recording state for logic
  const sessionIdRef = useRef<string>(""); // streaming session id for overlap

  useEffect(() => {
    // Initialize AudioRecord when component mounts
    try {
      const options = {
        sampleRate: 44100,
        channels: 1,
        bitsPerSample: 16,
        audioSource: 6, // VOICE_RECOGNITION
        wavFile: 'temp_audio.wav'
      };
      
      AudioRecord.init(options);
    } catch (error) {
      console.warn('Failed to initialize AudioRecord:', error);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (analysisTimeoutRef.current) {
        clearTimeout(analysisTimeoutRef.current);
      }
      try {
        AudioRecord.stop();
      } catch (error) {
        console.warn('Failed to stop AudioRecord:', error);
      }
    };
  }, []);

  // Request permissions for recording
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
    return true; // iOS permissions handled by react-native-audio-record
  };

  // Send audio file to backend for real-time analysis
  const sendAudioFileForAnalysis = async (filePath: string) => {
    try {
      setConnectionStatus('connecting');
      
      // Construct proper file path
      let fullPath = filePath;
      if (!filePath.startsWith('file://')) {
        // If it's just a filename, construct the full path
        fullPath = `file://${filePath}`;
      }
      
      console.log('Checking file path:', fullPath);
      
      // Read the audio file
      const fileInfo = await FileSystem.getInfoAsync(fullPath);
      if (!fileInfo.exists) {
        console.log('File does not exist at:', fullPath);
        throw new Error('Audio file does not exist');
      }
      
      console.log('File exists, size:', fileInfo.size);
      
      // Create FormData to send the WAV chunk to streaming endpoint with session id
      const formData = new FormData();
      // @ts-ignore React Native FormData supports file objects
      formData.append('file', { uri: fullPath, type: 'audio/wav', name: 'chunk.wav' });
      formData.append('session_id', sessionIdRef.current);
      
      // Send to streaming chunk endpoint (server maintains overlap/state)
      const response = await fetch(`${BACKEND_URL}/stream/chunk`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: AnalysisResult = await response.json();
      setConnectionStatus('connected');
      console.log(`🎵 Backend returned: ${result.notes?.length || 0} notes, ${result.chords?.length || 0} chords`);
      
      // Add ALL detected notes from this chunk (not just the latest one)
      if (result.notes && result.notes.length > 0) {
        console.log('📝 Adding notes:', result.notes.map(n => `${n.note_name} at ${n.time_seconds.toFixed(2)}s`));
        
        // Add ALL notes from this chunk to the running list
        setDetectedNotes(prev => [...prev, ...result.notes].slice(-100)); // Keep last 100 notes
      }
      
      // Add ALL detected chords from this chunk
      if (result.chords && result.chords.length > 0) {
        console.log('🎼 Adding chords:', result.chords.map(c => `${c.label} at ${c.time_seconds.toFixed(2)}s`));
        
        // Add ALL chords from this chunk to the running list
        setDetectedChords(prev => [...prev, ...result.chords].slice(-100)); // Keep last 100 chords
      }
      
      // Log running totals
      setDetectedNotes(currentNotes => {
        console.log(`📊 Total notes accumulated: ${currentNotes.length + (result.notes?.length || 0)}`);
        return currentNotes;
      });
      
      setDetectedChords(currentChords => {
        console.log(`📊 Total chords accumulated: ${currentChords.length + (result.chords?.length || 0)}`);
        return currentChords;
      });

      // Mirror the file-upload analysisResults shape so UI and sheet music can use it
      try {
        setAnalysisResults({
          onsets: result.onsets ? result.onsets.map(o => o.time_seconds) : [],
          notes: result.notes ? result.notes.map(n => n.note_name) : [],
          confidence: result.notes && result.notes.length > 0
            ? result.notes.reduce((acc, n) => acc + n.confidence, 0) / result.notes.length
            : 0,
          method: result.notes && result.notes.length > 0 ? result.notes[0].method : 'No detection',
          details: result,
        });
      } catch (err) {
        console.warn('Failed to populate analysisResults from stream response', err);
      }
      
    } catch (error) {
      console.error('Analysis error:', error);
      setConnectionStatus('error');
    }
  };

  // Start real-time analysis
  const startRealTimeAnalysis = async () => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        Alert.alert('Permission Required', 'Please grant microphone permissions to record audio.');
        return;
      }

      setIsRecording(true);
      isRecordingRef.current = true; // Update ref immediately
      console.log('isRecording state:', isRecording); // This will still show false
      console.log('isRecordingRef.current:', isRecordingRef.current); // This will show true
      setIsAnalyzing(true);
      setDuration(0);
      setDetectedNotes([]);
      setDetectedChords([]);
      setAnalysisResults(null); // Clear analysis results to reset sheet music
      setConnectionStatus('connecting');
      
      // Start a new streaming session (for overlap continuity on the backend)
      sessionIdRef.current = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      try {
        await fetch(`${BACKEND_URL}/stream/reset`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionIdRef.current })
        });
      } catch (e) {
        console.warn('Failed to reset streaming session (continuing):', e);
      }
      
      // Start recording
      AudioRecord.start();

      // Start duration timer
      intervalRef.current = setInterval(() => {
        setDuration(prev => prev + 0.1);
      }, 100) as unknown as NodeJS.Timeout;

      // Start periodic analysis - stop and restart recording to get chunks
      const analyzeRecordingChunk = async () => {
        if (!isRecordingRef.current) {
          console.log('🛑 isRecordingRef.current is false, stopping chunk analysis');
          return;
        }
        
        const chunkStartTime = Date.now();
        console.log('🎵 Starting analysis chunk at:', new Date(chunkStartTime).toLocaleTimeString());
        
        try {
          // Stop current recording to get the file
          const audioFile = await AudioRecord.stop();

          console.log('AudioRecord.stop() returned:', audioFile);
          console.log('FileSystem.documentDirectory:', FileSystem.documentDirectory);
          console.log('FileSystem.cacheDirectory:', FileSystem.cacheDirectory);

          // Check multiple possible locations
          const possiblePaths = [
            audioFile, // Direct path returned
            `${FileSystem.documentDirectory}${audioFile}`, // Documents + filename
            `${FileSystem.cacheDirectory}${audioFile}`, // Cache + filename  
            `${FileSystem.documentDirectory}temp_audio.wav`, // Our expected path
            `${FileSystem.cacheDirectory}temp_audio.wav`, // Cache version
          ];

          console.log('Checking possible file locations:');
          for (const path of possiblePaths) {
            try {
              const fileInfo = await FileSystem.getInfoAsync(path);
              console.log(`  ${path}: exists=${fileInfo.exists}`);
              if (fileInfo.exists) {
                // Found the file! Use this path
                console.log('✓ Found audio file at:', path);
                await sendAudioFileForAnalysis(path);
                break;
              }
            } catch (error : any) {
              console.log(`  ${path}: ERROR - ${error.message}`);
            }
          }
          
          // Restart recording for next chunk if still recording
          if (isRecordingRef.current) {
            console.log('🔄 Restarting recording for next chunk...');
            AudioRecord.start();
            // Schedule next analysis
            const nextChunkTime = Date.now() + 2000;
            console.log('⏰ Next chunk scheduled for:', new Date(nextChunkTime).toLocaleTimeString());
            analysisTimeoutRef.current = setTimeout(analyzeRecordingChunk, 2000) as any; // Analyze every 2 seconds
          } else {
            console.log('🛑 Recording stopped, no more chunks scheduled');
          }
        } catch (error) {
          console.error('Chunk analysis error:', error);
          if (isRecordingRef.current) {
            console.log('🔄 Error occurred, restarting recording and trying again...');
            // Restart recording and try again
            AudioRecord.start();
            analysisTimeoutRef.current = setTimeout(analyzeRecordingChunk, 2000) as any;
          }
        }
        
        const chunkEndTime = Date.now();
        const chunkDuration = chunkEndTime - chunkStartTime;
        console.log(`✅ Chunk analysis completed in ${chunkDuration}ms`);
      };
      
      // Start the analysis cycle after a short delay
      console.log('🚀 Starting real-time analysis with 2-second intervals');
      
      analysisTimeoutRef.current = setTimeout(analyzeRecordingChunk, 2000) as any;
      
    } catch (err) {
      console.error('Failed to start real-time analysis', err);
      Alert.alert('Error', 'Failed to start real-time analysis. Please try again.');
      setIsRecording(false);
      setIsAnalyzing(false);
    }
  };

  // Stop real-time analysis
  const stopRealTimeAnalysis = async () => {
    try {
      console.log('🛑 Stopping real-time analysis...');
      const wasRecording = isRecordingRef.current; // Use ref value
      setIsRecording(false);
      isRecordingRef.current = false; // Update ref immediately
      setIsAnalyzing(false);
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      
      if (analysisTimeoutRef.current) {
        console.log('⏰ Clearing scheduled analysis timeout');
        clearTimeout(analysisTimeoutRef.current);
        analysisTimeoutRef.current = null;
      }

      // Stop recording and get final analysis if we were recording
      if (wasRecording) {
        console.log('🎵 Processing final audio chunk...');
        const finalAudioFile = await AudioRecord.stop();
        console.log('AudioRecord.stop() returned:', finalAudioFile);
        console.log('FileSystem.documentDirectory:', FileSystem.documentDirectory);
        console.log('FileSystem.cacheDirectory:', FileSystem.cacheDirectory);
        
        // Analyze the final chunk
        try {
          await sendAudioFileForAnalysis(finalAudioFile);
          console.log('✅ Final chunk analysis completed');
        } catch (error) {
          console.warn('❌ Failed to analyze final chunk:', error);
        }
      }
      
      setConnectionStatus('disconnected');
      console.log('🔌 Real-time analysis stopped and disconnected');
      
    } catch (err) {
      console.error('Failed to stop analysis', err);
      Alert.alert('Error', 'Failed to stop analysis.');
    }
  };

  // Clear real-time results
  const clearResults = () => {
    setDetectedNotes([]);
    setDetectedChords([]);
    setDuration(0);
  };

  // Helper functions for connection status
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#4CAF50';
      case 'connecting': return '#FF9800';
      case 'error': return '#f44336';
      default: return '#9E9E9E';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'error': return 'Connection Error';
      default: return 'Disconnected';
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };

  const pickAudioFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['audio/*'],
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedFile(result.assets[0].uri);
        setAnalysisResults(null);
      }
    } catch (err) {
      console.error('Error picking file:', err);
      Alert.alert('Error', 'Failed to pick audio file');
    }
  };

  const analyzeAudio = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    try {
      console.log('Preparing file upload:', selectedFile);
      
      // Create proper FormData with file
      const formData = new FormData();
      
      // This is the correct way to append a file in React Native
      formData.append('file', {
        uri: selectedFile,
        name: 'audio.wav',
        type: 'audio/wave',
      } as any);
      
      console.log('Sending file to server...');
      
      const response = await fetch(`${BACKEND_URL}/analyze-stream`, {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error:', errorText);
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }

      const results = await response.json();
      console.log('Selected file:', selectedFile);

      // Transform results for display
      setAnalysisResults({
        onsets: results.onsets.map((onset: any) => onset.time_seconds),
        notes: results.notes.map((note: any) => note.note_name),
        confidence: results.notes.length > 0 
          ? results.notes.reduce((acc: number, note: any) => acc + note.confidence, 0) / results.notes.length 
          : 0,
        method: results.notes.length > 0 ? results.notes[0].method : 'No detection',
        details: results, // Store full results for debugging
      });
    } catch (err) {
      console.error('Analysis error:', err);
      Alert.alert('Error', 'Failed to analyze audio');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <ScrollView 
      style={{ flex: 1 }}
      contentContainerStyle={{ flexGrow: 1 }}
      showsVerticalScrollIndicator={true}
      keyboardShouldPersistTaps="handled"
    >
    <ThemedView style={styles.container} >
      <ThemedText type="title" style={styles.title}>
        Piano Audio Analysis
      </ThemedText>
      
      <ThemedText style={styles.subtitle}>
        Upload audio files or record live for real-time analysis
      </ThemedText>

      {/* Mode Selection */}
      <View style={styles.modeSelector}>
        <TouchableOpacity
          style={[styles.modeButton, mode === 'file' && styles.modeButtonActive]}
          onPress={() => setMode('file')}
        >
          <Ionicons name="cloud-upload" size={20} color={mode === 'file' ? 'white' : '#666'} />
          <ThemedText style={[styles.modeButtonText, mode === 'file' && styles.modeButtonTextActive]}>
            File Upload
          </ThemedText>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.modeButton, mode === 'realtime' && styles.modeButtonActive]}
          onPress={() => setMode('realtime')}
        >
          <Ionicons name="musical-notes" size={20} color={mode === 'realtime' ? 'white' : '#666'} />
          <ThemedText style={[styles.modeButtonText, mode === 'realtime' && styles.modeButtonTextActive]}>
            Live Recording
          </ThemedText>
        </TouchableOpacity>
      </View>

      {/* File Upload Mode */}
      {mode === 'file' && (
        <>
          <View style={styles.uploadSection}>
            <TouchableOpacity
              style={styles.uploadButton}
              onPress={pickAudioFile}
            >
              <Ionicons name="cloud-upload" size={24} color="white" />
              <ThemedText style={styles.uploadButtonText}>
                Choose Audio File
              </ThemedText>
            </TouchableOpacity>

            {selectedFile && (
              <View style={styles.fileInfo}>
                <Ionicons name="musical-note" size={20} color="#666" />
                <ThemedText style={styles.fileName}>
                  Audio file selected
                </ThemedText>
              </View>
            )}
          </View>

          {selectedFile && (
            <TouchableOpacity
              style={[styles.analyzeButton, isAnalyzing && styles.disabledButton]}
              onPress={analyzeAudio}
              disabled={isAnalyzing}
            >
              <Ionicons 
                name={isAnalyzing ? "hourglass" : "analytics"} 
                size={20} 
                color="white" 
              />
              <ThemedText style={styles.analyzeButtonText}>
                {isAnalyzing ? 'Analyzing...' : 'Analyze Audio'}
              </ThemedText>
            </TouchableOpacity>
          )}

          {analysisResults && (
            <ScrollView style={styles.resultsContainer}>
              <ThemedText type="subtitle" style={styles.resultsTitle}>
                Analysis Results
              </ThemedText>
              
              <View style={styles.resultItem}>
                <ThemedText style={styles.resultLabel}>Detected Notes:</ThemedText>
                <ThemedText style={styles.resultValue}>
                  {analysisResults.notes.length > 0 ? analysisResults.notes.join(', ') : 'None detected'}
                </ThemedText>
              </View>
              
              <View style={styles.resultItem}>
                <ThemedText style={styles.resultLabel}>Onsets Found:</ThemedText>
                <ThemedText style={styles.resultValue}>
                  {analysisResults.onsets.length}
                </ThemedText>
              </View>
              
              <View style={styles.resultItem}>
                <ThemedText style={styles.resultLabel}>Average Confidence:</ThemedText>
                <ThemedText style={styles.resultValue}>
                  {(analysisResults.confidence * 100).toFixed(1)}%
                </ThemedText>
              </View>
              
              <View style={styles.resultItem}>
                <ThemedText style={styles.resultLabel}>Detection Method:</ThemedText>
                <ThemedText style={styles.resultValue}>
                  {analysisResults.method}
                </ThemedText>
              </View>

              {analysisResults.details?.analysis_summary && (
                <>
                  <View style={styles.resultItem}>
                    <ThemedText style={styles.resultLabel}>Duration:</ThemedText>
                    <ThemedText style={styles.resultValue}>
                      {analysisResults.details.analysis_summary.duration_seconds.toFixed(2)}s
                    </ThemedText>
                  </View>
                  
                  {/* Note: detected_fundamental property removed as it's not in the AnalysisResult type */}
                </>
              )}

              {/* Show individual note timings */}
              {analysisResults?.details?.notes && analysisResults.details.notes.length > 0 && (
                <View style={styles.noteTimings}>
                  <ThemedText style={styles.resultLabel}>Note Timings:</ThemedText>
                  {analysisResults.details.notes.map((note: any, index: number) => (
                    <View key={index} style={styles.noteItem}>
                      <ThemedText style={styles.noteText}>
                        {note.time_seconds.toFixed(2)}s: {note.note_name} ({(note.confidence * 100).toFixed(0)}%)
                      </ThemedText>
                    </View>
                  ))}
                </View>
              )}
            </ScrollView>
          )}

          {/* Sheet Music Display for File Analysis */}
          <PianoSheetMusic results={analysisResults?.details}/>
        </>
      )}

      {/* Real-time Recording Mode */}
      {mode === 'realtime' && (
        <View style={styles.realtimeContainer}>
          {/* Connection Status */}
          <View style={styles.statusContainer}>
            <View style={[styles.statusDot, { backgroundColor: getConnectionStatusColor() }]} />
            <ThemedText style={styles.statusText}>
              {getConnectionStatusText()}
            </ThemedText>
          </View>

          <View style={styles.recordingArea}>
            <View style={styles.durationContainer}>
              <ThemedText type="subtitle" style={styles.duration}>
                {formatDuration(duration)}
              </ThemedText>
            </View>

            <View style={styles.controlsContainer}>
              {!isRecording ? (
                <TouchableOpacity
                  style={[styles.recordButton, styles.recordButtonInactive]}
                  onPress={startRealTimeAnalysis}
                  disabled={isRecording}
                >
                  <Ionicons name="musical-notes" size={32} color="white" />
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={[styles.recordButton, styles.recordButtonActive]}
                  onPress={stopRealTimeAnalysis}
                >
                  <Ionicons name="stop" size={32} color="white" />
                </TouchableOpacity>
              )}
            </View>

            {isRecording && (
              <View style={styles.recordingIndicator}>
                <View style={styles.pulsingDot} />
                <ThemedText style={styles.recordingText}>
                  {isAnalyzing ? 'Analyzing...' : 'Recording...'}
                </ThemedText>
              </View>
            )}
          </View>

          {/* Live Sheet Music Display */}
          <PianoSheetMusic results={analysisResults?.details}/>

          {/* Real-time Results Display */}
          {(detectedNotes.length > 0 || detectedChords.length > 0) && (
            <View style={styles.realtimeResultsArea}>
              <View style={styles.resultsHeader}>
                <ThemedText type="subtitle" style={styles.resultsTitle}>
                  Detection Results ({detectedNotes.length + detectedChords.length} total)
                </ThemedText>
                <TouchableOpacity
                  style={styles.clearButton}
                  onPress={clearResults}
                >
                  <Ionicons name="refresh" size={16} color="white" />
                  <Text style={styles.clearButtonText}>Clear</Text>
                </TouchableOpacity>
              </View>
              
              <ScrollView style={styles.resultsList} nestedScrollEnabled={true}>
                {/* Show ALL detected notes (most recent first) */}
                {detectedNotes.slice().reverse().map((note, index) => (
                  <View key={`note-${detectedNotes.length - index - 1}`} style={styles.realtimeResultItem}>
                    <View style={styles.resultIcon}>
                      <Ionicons name="musical-note" size={16} color="#4CAF50" />
                    </View>
                    <View style={styles.resultContent}>
                      <Text style={styles.resultText}>
                        {note.note_name} ({note.frequency_hz.toFixed(1)}Hz)
                      </Text>
                      <Text style={styles.resultMeta}>
                        {note.time_seconds.toFixed(1)}s • {note.method} • {(note.confidence * 100).toFixed(0)}%
                      </Text>
                    </View>
                  </View>
                ))}
                
                {/* Show ALL detected chords (most recent first) */}
                {detectedChords.slice().reverse().map((chord, index) => (
                  <View key={`chord-${detectedChords.length - index - 1}`} style={styles.realtimeResultItem}>
                    <View style={styles.resultIcon}>
                      <Ionicons name="library" size={16} color="#2196F3" />
                    </View>
                    <View style={styles.resultContent}>
                      <Text style={styles.resultText}>
                        {chord.label} ({chord.inversion})
                      </Text>
                      <Text style={styles.resultMeta}>
                        {chord.time_seconds.toFixed(1)}s • {(chord.confidence * 100).toFixed(0)}%
                      </Text>
                    </View>
                  </View>
                ))}
              </ScrollView>
            </View>
          )}
        </View>
      )}

      {/* Only show info section if in file mode OR in real-time mode with no detection results */}
      {(mode === 'file' || (mode === 'realtime' && detectedNotes.length === 0 && detectedChords.length === 0)) && (
        <View style={styles.infoSection}>
          <ThemedText style={styles.infoTitle}>
            {mode === 'file' ? 'Supported Formats:' : 'Analysis Tips:'}
          </ThemedText>
          <ThemedText style={styles.infoText}>
            {mode === 'file' 
              ? 'WAV, MP3, M4A • 44.1kHz recommended • Mono/Stereo'
              : '• Hold device close to piano for best quality\n• Minimize background noise\n• Ensure stable internet connection\n• Analysis happens in real-time with ~1 second delay'
            }
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
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    marginBottom: 30,
  },
  
  // Mode Selector
  modeSelector: {
    flexDirection: 'row',
    marginBottom: 30,
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    borderRadius: 8,
    padding: 4,
  },
  modeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 6,
    gap: 8,
  },
  modeButtonActive: {
    backgroundColor: '#2196F3',
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  modeButtonTextActive: {
    color: 'white',
  },
  
  // File Upload Mode
  uploadSection: {
    alignItems: 'center',
    marginBottom: 30,
  },
  uploadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2196F3',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 8,
    gap: 10,
  },
  uploadButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  fileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 15,
    gap: 8,
  },
  fileName: {
    opacity: 0.7,
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#4CAF50',
    paddingVertical: 15,
    borderRadius: 8,
    marginBottom: 20,
    gap: 10,
  },
  disabledButton: {
    backgroundColor: '#999',
  },
  analyzeButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    borderRadius: 8,
    padding: 15,
    marginBottom: 20,
  },
  resultsTitle: {
    marginBottom: 15,
    textAlign: 'center',
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(128, 128, 128, 0.2)',
  },
  resultLabel: {
    fontWeight: 'bold',
  },
  resultValue: {
    opacity: 0.8,
  },
  noteTimings: {
    marginTop: 15,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: 'rgba(128, 128, 128, 0.2)',
  },
  noteItem: {
    paddingVertical: 4,
    paddingLeft: 10,
  },
  noteText: {
    fontSize: 14,
    opacity: 0.8,
    fontFamily: 'monospace',
  },
  
  // Real-time Mode
  realtimeContainer: {
    flex: 1,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    borderRadius: 16,
    alignSelf: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
  },
  recordingArea: {
    alignItems: 'center',
    marginBottom: 20,
  },
  durationContainer: {
    marginBottom: 20,
  },
  duration: {
    fontSize: 36,
    fontFamily: 'monospace',
    textAlign: 'center',
  },
  controlsContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  recordButtonInactive: {
    backgroundColor: '#4CAF50',
  },
  recordButtonActive: {
    backgroundColor: '#ff0000',
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
  },
  pulsingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#ff0000',
    marginRight: 8,
  },
  recordingText: {
    color: '#ff0000',
    fontWeight: 'bold',
  },
  realtimeResultsArea: {
    width: '100%',
    maxHeight: 300,
    marginBottom: 20,
  },
  resultsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#9E9E9E',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    gap: 4,
  },
  clearButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  resultsList: {
    maxHeight: 240,
    paddingBottom: 10,
  },
  realtimeResultItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(128, 128, 128, 0.05)',
    borderRadius: 8,
    marginBottom: 4,
  },
  resultIcon: {
    marginRight: 12,
    width: 24,
    alignItems: 'center',
  },
  resultContent: {
    flex: 1,
  },
  resultText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#aaaaaa',
  },
  resultMeta: {
    fontSize: 11,
    opacity: 0.6,
    marginTop: 2,
  },
  
  // Shared
  infoSection: {
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    padding: 15,
    borderRadius: 8,
    marginTop: 'auto',
  },
  infoTitle: {
    fontWeight: 'bold',
    marginBottom: 8,
  },
  infoText: {
    opacity: 0.8,
    lineHeight: 20,
    fontSize: 12,
  },
});
