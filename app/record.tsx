import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import React, { useEffect, useRef, useState } from 'react';
import { Alert, PermissionsAndroid, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import AudioRecord from 'react-native-audio-record';
import { ThemedText } from '../components/ThemedText';
import { ThemedView } from '../components/ThemedView';

export default function RecordScreen() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingUri, setRecordingUri] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const wavPath = `${FileSystem.documentDirectory}audio.wav`;

  useEffect(() => {
    // Initialize AudioRecord
    const options = {
      sampleRate: 44100,
      channels: 1,
      bitsPerSample: 16,
      audioSource: 6, // VOICE_RECOGNITION
      wavFile: wavPath
    };
    
    AudioRecord.init(options);
    
    return () => {
      AudioRecord.stop();
    };
  }, []);

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

  const startRecording = async () => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        Alert.alert('Permission Required', 'Please grant microphone permissions to record audio.');
        return;
      }

      setIsRecording(true);
      setDuration(0);
      setRecordingUri(null);

      // Start recording
      AudioRecord.start();

      // Start duration timer
      intervalRef.current = setInterval(() => {
        setDuration(prev => prev + 0.1);
      }, 100) as unknown as NodeJS.Timeout;

    } catch (err) {
      console.error('Failed to start recording', err);
      Alert.alert('Error', 'Failed to start recording. Please try again.');
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    try {
      setIsRecording(false);
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      // Stop recording and get the file path
      const audioFile = await AudioRecord.stop();
      console.log('Recording saved to:', audioFile);
      setRecordingUri(audioFile);

      console.log("recordingUri raw:", recordingUri);
      const info = await FileSystem.getInfoAsync(recordingUri as any, { size: true });
      console.log("FileSystem.getInfoAsync:", info);
      
    } catch (err) {
      console.error('Failed to stop recording', err);
      Alert.alert('Error', 'Failed to stop recording.');
    }
  };

  const playRecording = async () => {
    if (!recordingUri) return;

    try {
      if (sound && isPlaying) {
        await sound.pauseAsync();
        setIsPlaying(false);
        return;
      }

      if (sound && !isPlaying) {
        await sound.playAsync();
        setIsPlaying(true);
        return;
      }

      const { sound: newSound } = await Audio.Sound.createAsync(
        { uri: recordingUri },
        { shouldPlay: true }
      );
      
      setSound(newSound);
      setIsPlaying(true);

      newSound.setOnPlaybackStatusUpdate((status: any) => {
        if (status.isLoaded && status.didJustFinish) {
          setIsPlaying(false);
        }
      });

    } catch (err) {
      console.error('Failed to play recording', err);
      Alert.alert('Error', 'Failed to play recording.');
    }
  };

  const shareRecording = async () => {
    if (!recordingUri) return;

    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const fileName = `piano_recording_${timestamp}.wav`;
  
      const fullRecordingPath = `${recordingUri}`;
      console.log('Full recording path:', fullRecordingPath);    
      const documentsDir = FileSystem.documentDirectory;
      const newUri = `${documentsDir}${fileName}`;
      
      await FileSystem.copyAsync({
        from: fullRecordingPath,
        to: newUri,
      });

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(newUri, {
          mimeType: 'audio/wav',
          dialogTitle: 'Share Piano Recording',
        });
      } else {
        Alert.alert('Success', `Recording saved to: ${newUri}`);
      }
    } catch (err) {
      console.error('Failed to share recording', err);
      Alert.alert('Error', 'Failed to share recording.');
    }
  };

  const deleteRecording = () => {
    Alert.alert(
      'Delete Recording',
      'Are you sure you want to delete this recording?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            setRecordingUri(null);
            setDuration(0);
            if (sound) {
              sound.unloadAsync();
              setSound(null);
            }
            setIsPlaying(false);
          },
        },
      ]
    );
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title" style={styles.title}>
        Piano Recorder
      </ThemedText>
      
      <ThemedText style={styles.subtitle}>
        Record high-quality audio snippets for pitch detection analysis
      </ThemedText>

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
              onPress={startRecording}
              disabled={isRecording}
            >
              <Ionicons name="mic" size={32} color="white" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.recordButton, styles.recordButtonActive]}
              onPress={stopRecording}
            >
              <Ionicons name="stop" size={32} color="white" />
            </TouchableOpacity>
          )}
        </View>

        {isRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.pulsingDot} />
            <ThemedText style={styles.recordingText}>Recording...</ThemedText>
          </View>
        )}
      </View>

      {recordingUri && (
        <View style={styles.playbackArea}>
          <ThemedText type="subtitle" style={styles.playbackTitle}>
            Recorded Audio
          </ThemedText>
          
          <View style={styles.playbackControls}>
            <TouchableOpacity
              style={styles.playButton}
              onPress={playRecording}
            >
              <Ionicons 
                name={isPlaying ? "pause" : "play"} 
                size={24} 
                color="white" 
              />
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.shareButton}
              onPress={shareRecording}
            >
              <Ionicons name="share" size={20} color="white" />
              <Text style={styles.buttonText}>Export</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.deleteButton}
              onPress={deleteRecording}
            >
              <Ionicons name="trash" size={20} color="white" />
              <Text style={styles.buttonText}>Delete</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      <View style={styles.infoArea}>
        <ThemedText style={styles.infoTitle}>Recording Tips:</ThemedText>
        <ThemedText style={styles.infoText}>
          • Hold device close to piano for best quality{'\n'}
          • Minimize background noise{'\n'}
          • Record in a quiet environment{'\n'}
          • Records in 44.1kHz WAV format for optimal pitch detection
        </ThemedText>
      </View>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    paddingTop: 80,
    alignItems: 'center',
  },
  title: {
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    marginBottom: 40,
  },
  recordingArea: {
    alignItems: 'center',
    marginBottom: 20,
  },
  durationContainer: {
    marginBottom: 30,
  },
  duration: {
    fontSize: 48,
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
    backgroundColor: '#ff4444',
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
  playbackArea: {
    width: '100%',
    alignItems: 'center',
    marginBottom: 30,
  },
  playbackTitle: {
    marginBottom: 20,
  },
  playbackControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 15,
  },
  playButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2196F3',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 8,
    gap: 5,
  },
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f44336',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 8,
    gap: 5,
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  infoArea: {
    width: '100%',
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    padding: 15,
    borderRadius: 8,
  },
  infoTitle: {
    fontWeight: 'bold',
    marginBottom: 8,
  },
  infoText: {
    opacity: 0.8,
    lineHeight: 20,
  },
});
