import React, { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';
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

interface VexFlowSheetMusicProps {
  detectedNotes: any[];
  detectedChords: any[];
  isRecording: boolean;
}

export default function VexFlowSheetMusic({ 
  detectedNotes, 
  detectedChords, 
  isRecording 
}: VexFlowSheetMusicProps) {
  const [musicalEvents, setMusicalEvents] = useState<MusicalEvent[]>([]);
  const [measures, setMeasures] = useState<MusicalEvent[][]>([]);
  const [totalDuration, setTotalDuration] = useState(0);
  
  // Constants for musical timing
  const QUARTER_NOTE_DURATION = 0.5; // 0.5 seconds = 1 quarter note
  const MEASURE_DURATION = 4 * QUARTER_NOTE_DURATION; // 4/4 time = 2 seconds per measure
  
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
  
  // Get note position on staff (simplified)
  const getNotePosition = (noteName: string) => {
    const noteMap: { [key: string]: number } = {
      'C': 0, 'C#': 0, 'D': 1, 'D#': 1, 'E': 2, 'F': 3, 'F#': 3,
      'G': 4, 'G#': 4, 'A': 5, 'A#': 5, 'B': 6
    };
    return noteMap[noteName] || 0;
  };
  
  // Render a single measure
  const renderMeasure = (measureEvents: MusicalEvent[], measureIndex: number) => {
    return (
      <View key={`measure-${measureIndex}`} style={styles.measureContainer}>
        <View style={styles.measureHeader}>
          <ThemedText style={styles.measureNumber}>M{measureIndex + 1}</ThemedText>
          {measureIndex === 0 && (
            <ThemedText style={styles.timeSignature}>4/4</ThemedText>
          )}
        </View>
        
        <View style={styles.staffContainer}>
          {/* Draw staff lines */}
          {[0, 1, 2, 3, 4].map(line => (
            <View key={line} style={[styles.staffLine, { top: line * 8 }]} />
          ))}
          
          {/* Render notes */}
          {measureEvents.map((event, index) => (
            <View 
              key={event.id} 
              style={[
                styles.noteContainer, 
                { 
                  left: 20 + (index * 60), // Space notes horizontally
                  top: getNotePosition(event.note_name || event.chord_label?.split(':')[0] || 'C') * 8
                }
              ]}
            >
              {event.type === 'note' ? (
                <View style={styles.note}>
                  <View style={styles.noteHead} />
                  <View style={styles.noteStem} />
                  <ThemedText style={styles.noteText}>
                    {convertNoteName(event.note_name || 'C')}
                  </ThemedText>
                  <View style={styles.noteDuration}>
                    <ThemedText style={styles.durationText}>
                      {getNoteType(event.duration_seconds)}
                    </ThemedText>
                  </View>
                </View>
              ) : (
                <View style={styles.chord}>
                  <View style={styles.chordSymbol}>
                    <ThemedText style={styles.chordText}>
                      {event.chord_label?.split(':')[0] || 'C'}
                    </ThemedText>
                    <ThemedText style={styles.chordQuality}>
                      {event.chord_label?.split(':')[1] || 'maj'}
                    </ThemedText>
                  </View>
                </View>
              )}
            </View>
          ))}
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
      </View>
      
      {measures.length === 0 ? (
        <View style={styles.emptyMeasure}>
          <ThemedText style={styles.emptyMeasureText}>
            {isRecording ? 'Waiting for notes...' : 'No notes detected'}
          </ThemedText>
        </View>
      ) : (
        <View style={styles.measuresContainer}>
          {measures.map((measureEvents, index) => 
            renderMeasure(measureEvents, index)
          )}
        </View>
      )}
      
      <View style={styles.legend}>
        <ThemedText style={styles.legendTitle}>Legend:</ThemedText>
        <ThemedText style={styles.legendText}>
          • 0.5s = Quarter Note • 1.0s = Half Note • 2.0s = Whole Note
        </ThemedText>
        <ThemedText style={styles.legendText}>
          • Notes are positioned on the staff based on pitch
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
  },
  infoText: {
    fontSize: 12,
    opacity: 0.7,
  },
  measuresContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 15,
    marginBottom: 15,
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
    backgroundColor: '#4CAF50',
    borderRadius: 4,
    marginBottom: 2,
  },
  noteStem: {
    width: 2,
    height: 20,
    backgroundColor: '#4CAF50',
    position: 'absolute',
    top: -10,
    right: 5,
  },
  noteText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginTop: 2,
  },
  noteDuration: {
    marginTop: 2,
  },
  durationText: {
    fontSize: 8,
    color: '#666',
  },
  chord: {
    alignItems: 'center',
  },
  chordSymbol: {
    alignItems: 'center',
    padding: 4,
    backgroundColor: '#2196F3',
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
