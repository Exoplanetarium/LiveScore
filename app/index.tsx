import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import React, { useState } from 'react';
import { Alert, ScrollView, StyleSheet, TouchableOpacity, View } from 'react-native';
import { ThemedText } from '../components/ThemedText';
import { ThemedView } from '../components/ThemedView';

export default function AnalyzeScreen() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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
      // TODO: Connect to your Python backend for analysis
      // For now, this is a placeholder
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate analysis
      
      setAnalysisResults({
        onsets: [1, 2, 3],
        notes: ['C4', 'E4', 'G4'],
        confidence: 0.87,
        method: 'Robust (YIN)',
      });
    } catch (err) {
      console.error('Analysis error:', err);
      Alert.alert('Error', 'Failed to analyze audio');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title" style={styles.title}>
        Piano Audio Analysis
      </ThemedText>
      
      <ThemedText style={styles.subtitle}>
        Upload or record audio to detect piano notes and onsets
      </ThemedText>

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
              {analysisResults.notes.join(', ')}
            </ThemedText>
          </View>
          
          <View style={styles.resultItem}>
            <ThemedText style={styles.resultLabel}>Onsets Found:</ThemedText>
            <ThemedText style={styles.resultValue}>
              {analysisResults.onsets.length}
            </ThemedText>
          </View>
          
          <View style={styles.resultItem}>
            <ThemedText style={styles.resultLabel}>Confidence:</ThemedText>
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
        </ScrollView>
      )}

      <View style={styles.infoSection}>
        <ThemedText style={styles.infoTitle}>Supported Formats:</ThemedText>
        <ThemedText style={styles.infoText}>
          WAV, MP3, M4A • 44.1kHz recommended • Mono/Stereo
        </ThemedText>
      </View>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
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
  infoSection: {
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    padding: 15,
    borderRadius: 8,
  },
  infoTitle: {
    fontWeight: 'bold',
    marginBottom: 5,
  },
  infoText: {
    opacity: 0.7,
    fontSize: 12,
  },
});
