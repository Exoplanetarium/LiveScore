import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { ThemedText } from './ThemedText';
import { ThemedView } from './ThemedView';

// Types for musical events
interface MusicalEvent {
  id: string;
  time_seconds: number;
  type: 'note' | 'chord';
  note_name?: string;
  chord_label?: string;
  duration_seconds: number;
  midi_note?: number;
  confidence: number;
}

interface AdvancedVexFlowMusicProps {
  detectedNotes: any[];
  detectedChords: any[];
  isRecording: boolean;
}

export default function AdvancedVexFlowMusic({ 
  detectedNotes, 
  detectedChords, 
  isRecording 
}: AdvancedVexFlowMusicProps) {
  const [musicalEvents, setMusicalEvents] = useState<MusicalEvent[]>([]);
  const [measures, setMeasures] = useState<MusicalEvent[][]>([]);
  const [totalDuration, setTotalDuration] = useState(0);
  const [currentBeat, setCurrentBeat] = useState(0);
  
  // Constants for musical timing
  const QUARTER_NOTE_DURATION = 0.5; // 0.5 seconds = 1 quarter note
  const MEASURE_DURATION = 4 * QUARTER_NOTE_DURATION; // 4/4 time = 2 seconds per measure
  const BEATS_PER_MEASURE = 4;
  
  // Convert detected notes/chords to musical events
  useEffect(() => {
    const events: MusicalEvent[] = [];
    
    // Process notes
    detectedNotes.forEach((note, index) => {
      events.push({
        id: `note-${index}`,
        time_seconds: note.time_seconds,
        type: 'note',
        note_name: note.note_name,
        midi_note: note.midi_note,
        duration_seconds: note.duration_seconds || 0.5, // Default to quarter note
        confidence: note.confidence
      });
    });
    
    // Process chords
    detectedChords.forEach((chord, index) => {
      events.push({
        id: `chord-${index}`,
        time_seconds: chord.time_seconds,
        type: 'chord',
        chord_label: chord.label,
        duration_seconds: chord.duration_seconds || 1.0, // Default to half note
        confidence: chord.confidence
      });
    });
    
    // Sort by time
    events.sort((a, b) => a.time_seconds - b.time_seconds);
    setMusicalEvents(events);
    
    // Calculate total duration
    if (events.length > 0) {
      const lastEvent = events[events.length - 1];
      setTotalDuration(lastEvent.time_seconds + lastEvent.duration_seconds);
    }
  }, [detectedNotes, detectedChords]);
  
  // Group events into measures
  useEffect(() => {
    if (musicalEvents.length === 0) return;
    
    const measureCount = Math.ceil(totalDuration / MEASURE_DURATION);
    const newMeasures: MusicalEvent[][] = [];
    
    for (let i = 0; i < measureCount; i++) {
      const measureStart = i * MEASURE_DURATION;
      const measureEnd = measureStart + MEASURE_DURATION;
      
      const measureEvents = musicalEvents.filter(event => 
        event.time_seconds >= measureStart && event.time_seconds < measureEnd
      );
      
      newMeasures.push(measureEvents);
    }
    
    setMeasures(newMeasures);
  }, [MEASURE_DURATION, musicalEvents, totalDuration]);
  
  // Update current beat for live performance
  useEffect(() => {
    if (!isRecording) return;
    
    const interval = setInterval(() => {
      setCurrentBeat(prev => (prev + 1) % (BEATS_PER_MEASURE * measures.length));
    }, QUARTER_NOTE_DURATION * 1000);
    
    return () => clearInterval(interval);
  }, [isRecording, measures.length]);
  
  // Convert note names to VexFlow format
  const convertNoteName = (noteName: string) => {
    // Remove octave number and convert to VexFlow format
    const note = noteName.replace(/\d/g, '');
    return note;
  };
  
  // Convert duration to VexFlow note type
  const getNoteType = (durationSeconds: number) => {
    if (durationSeconds <= 0.125) return '8'; // Eighth note
    if (durationSeconds <= 0.25) return '4'; // Quarter note
    if (durationSeconds <= 0.5) return '2'; // Half note
    if (durationSeconds <= 1.0) return '1'; // Whole note
    return '4'; // Default to quarter note
  };
  
  // Get note position on staff (enhanced)
  const getNotePosition = (noteName: string) => {
    const noteMap: { [key: string]: number } = {
      'C': 0, 'C#': 0, 'D': 1, 'D#': 1, 'E': 2, 'F': 3, 'F#': 3,
      'G': 4, 'G#': 4, 'A': 5, 'A#': 5, 'B': 6
    };
    return noteMap[noteName] || 0;
  };
  
  // Get note color based on confidence
  const getNoteColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50'; // Green for high confidence
    if (confidence >= 0.6) return '#FF9800'; // Orange for medium confidence
    return '#F44336'; // Red for low confidence
  };
  
  // Render a single measure with enhanced notation
  const renderMeasure = (measureEvents: MusicalEvent[], measureIndex: number) => {
    const isCurrentMeasure = Math.floor(currentBeat / BEATS_PER_MEASURE) === measureIndex;
    
    return (
      <View key={`measure-${measureIndex}`} style={[
        styles.measureContainer,
        isCurrentMeasure && styles.currentMeasure
      ]}>
        <View style={styles.measureHeader}>
          <ThemedText style={styles.measureNumber}>M{measureIndex + 1}</ThemedText>
          {measureIndex === 0 && (
            <ThemedText style={styles.timeSignature}>4/4</ThemedText>
          )}
          {isCurrentMeasure && (
            <View style={styles.currentIndicator}>
              <ThemedText style={styles.currentText}>▶</ThemedText>
            </View>
          )}
        </View>
        
        <View style={styles.staffContainer}>
          {/* Draw staff lines */}
          {[0, 1, 2, 3, 4].map(line => (
            <View key={line} style={[styles.staffLine, { top: line * 8 }]} />
          ))}
          
          {/* Draw beat markers */}
          {[0, 1, 2, 3].map(beat => (
            <View key={beat} style={[styles.beatMarker, { left: 20 + (beat * 30) }]} />
          ))}
          
          {/* Render notes */}
          {measureEvents.map((event, index) => {
            const noteColor = getNoteColor(event.confidence);
            const notePosition = getNotePosition(
              event.note_name || event.chord_label?.split(':')[0] || 'C'
            );
            
            return (
              <View 
                key={event.id} 
                style={[
                  styles.noteContainer, 
                  { 
                    left: 20 + (index * 60), // Space notes horizontally
                    top: notePosition * 8
                  }
                ]}
              >
                {event.type === 'note' ? (
                  <View style={styles.note}>
                    <View style={[styles.noteHead, { backgroundColor: noteColor }]} />
                    <View style={[styles.noteStem, { backgroundColor: noteColor }]} />
                    <ThemedText style={[styles.noteText, { color: noteColor }]}>
                      {convertNoteName(event.note_name || 'C')}
                    </ThemedText>
                    <View style={styles.noteDuration}>
                      <ThemedText style={styles.durationText}>
                        {getNoteType(event.duration_seconds)}
                      </ThemedText>
                    </View>
                    <View style={styles.confidenceIndicator}>
                      <ThemedText style={styles.confidenceText}>
                        {(event.confidence * 100).toFixed(0)}%
                      </ThemedText>
                    </View>
                  </View>
                ) : (
                  <View style={styles.chord}>
                    <View style={[styles.chordSymbol, { backgroundColor: noteColor }]}>
                      <ThemedText style={styles.chordText}>
                        {event.chord_label?.split(':')[0] || 'C'}
                      </ThemedText>
                      <ThemedText style={styles.chordQuality}>
                        {event.chord_label?.split(':')[1] || 'maj'}
                      </ThemedText>
                    </View>
                    <View style={styles.confidenceIndicator}>
                      <ThemedText style={styles.confidenceText}>
                        {(event.confidence * 100).toFixed(0)}%
                      </ThemedText>
                    </View>
                  </View>
                )}
              </View>
            );
          })}
        </View>
        
        {/* Measure footer with timing info */}
        <View style={styles.measureFooter}>
          <ThemedText style={styles.measureTiming}>
            {(measureIndex * MEASURE_DURATION).toFixed(1)}s - {((measureIndex + 1) * MEASURE_DURATION).toFixed(1)}s
          </ThemedText>
        </View>
      </View>
    );
  };
  
  return (
    <ThemedView style={styles.container}>
      <ThemedText type="subtitle" style={styles.title}>
        Live Sheet Music
      </ThemedText>
      
      <View style={styles.infoRow}>
        <ThemedText style={styles.infoText}>
          Total Events: {musicalEvents.length}
        </ThemedText>
        <ThemedText style={styles.infoText}>
          Duration: {totalDuration.toFixed(1)}s
        </ThemedText>
        <ThemedText style={styles.infoText}>
          Measures: {measures.length}
        </ThemedText>
        {isRecording && (
          <ThemedText style={styles.infoText}>
            Beat: {currentBeat % BEATS_PER_MEASURE + 1}/{BEATS_PER_MEASURE}
          </ThemedText>
        )}
      </View>
      
      {measures.length === 0 ? (
        <View style={styles.emptyMeasure}>
          <ThemedText style={styles.emptyMeasureText}>
            {isRecording ? 'Waiting for notes...' : 'No notes detected'}
          </ThemedText>
        </View>
      ) : (
        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false}
          style={styles.measuresScrollView}
        >
          <View style={styles.measuresContainer}>
            {measures.map((measureEvents, index) => 
              renderMeasure(measureEvents, index)
            )}
          </View>
        </ScrollView>
      )}
      
      <View style={styles.legend}>
        <ThemedText style={styles.legendTitle}>Legend:</ThemedText>
        <ThemedText style={styles.legendText}>
          • 0.5s = Quarter Note • 1.0s = Half Note • 2.0s = Whole Note
        </ThemedText>
        <ThemedText style={styles.legendText}>
          • Colors indicate confidence: 🟢 High (≥80%) 🟠 Medium (60-79%) 🔴 Low (60%)
        </ThemedText>
        <ThemedText style={styles.legendText}>
          • ▶ indicates current measure during recording
        </ThemedText>
      </View>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    marginBottom: 20,
  },
  title: {
    textAlign: 'center',
    marginBottom: 15,
    color: '#333',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    flexWrap: 'wrap',
    gap: 10,
  },
  infoText: {
    fontSize: 12,
    opacity: 0.7,
  },
  measuresScrollView: {
    maxHeight: 200,
  },
  measuresContainer: {
    flexDirection: 'row',
    gap: 15,
    paddingHorizontal: 10,
  },
  measureContainer: {
    borderWidth: 2,
    borderColor: '#333',
    borderRadius: 8,
    padding: 15,
    backgroundColor: 'white',
    minWidth: 120,
    maxWidth: 150,
  },
  currentMeasure: {
    borderColor: '#4CAF50',
    borderWidth: 3,
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
  },
  measureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  measureNumber: {
    fontWeight: 'bold',
    fontSize: 14,
    color: '#333',
  },
  timeSignature: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
  },
  currentIndicator: {
    backgroundColor: '#4CAF50',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  currentText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  staffContainer: {
    position: 'relative',
    height: 40,
    marginBottom: 10,
  },
  staffLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: '#333',
  },
  beatMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: '#ddd',
    opacity: 0.5,
  },
  noteContainer: {
    position: 'absolute',
    alignItems: 'center',
  },
  note: {
    alignItems: 'center',
    position: 'relative',
  },
  noteHead: {
    width: 12,
    height: 8,
    borderRadius: 4,
    marginBottom: 2,
  },
  noteStem: {
    width: 2,
    height: 20,
    position: 'absolute',
    top: -10,
    right: 5,
  },
  noteText: {
    fontSize: 10,
    fontWeight: 'bold',
    marginTop: 2,
  },
  noteDuration: {
    marginTop: 2,
  },
  durationText: {
    fontSize: 8,
    color: '#666',
  },
  confidenceIndicator: {
    marginTop: 2,
    paddingHorizontal: 4,
    paddingVertical: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
    borderRadius: 3,
  },
  confidenceText: {
    fontSize: 8,
    color: '#666',
    fontWeight: 'bold',
  },
  chord: {
    alignItems: 'center',
  },
  chordSymbol: {
    alignItems: 'center',
    padding: 4,
    borderRadius: 4,
    minWidth: 30,
  },
  chordText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: 'white',
  },
  chordQuality: {
    fontSize: 8,
    color: 'white',
    opacity: 0.9,
  },
  measureFooter: {
    alignItems: 'center',
    marginTop: 5,
  },
  measureTiming: {
    fontSize: 10,
    color: '#999',
    fontFamily: 'monospace',
  },
  emptyMeasure: {
    height: 100,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#ddd',
    borderStyle: 'dashed',
    borderRadius: 8,
    backgroundColor: '#f9f9f9',
  },
  emptyMeasureText: {
    color: '#999',
    fontSize: 14,
    textAlign: 'center',
  },
  legend: {
    marginTop: 10,
    padding: 10,
    backgroundColor: 'rgba(128, 128, 128, 0.1)',
    borderRadius: 6,
  },
  legendTitle: {
    fontWeight: 'bold',
    fontSize: 12,
    marginBottom: 5,
    color: '#666',
  },
  legendText: {
    fontSize: 11,
    color: '#888',
    lineHeight: 16,
  },
});
