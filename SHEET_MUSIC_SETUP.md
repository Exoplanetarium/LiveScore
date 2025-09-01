# Live Sheet Music Setup Guide

This guide will help you set up the live sheet music functionality in your LiveScore app.

## Required Dependencies

Install the following packages to enable sheet music rendering:

```bash
# Install VexFlow for musical notation
npm install vexflow

# Install SVG support for React Native
npm install react-native-svg

# If using yarn:
yarn add vexflow react-native-svg
```

## What's Been Added

### 1. VexFlowSheetMusic Component (`components/VexFlowSheetMusic.tsx`)
- Renders live musical notation as you play
- Automatically groups notes into measures (4/4 time signature)
- Converts detected notes/chords to proper musical symbols
- Updates in real-time as new audio is analyzed

### 2. Integration with Main App
- Added to both real-time recording mode and file analysis mode
- Displays above the detection results for easy viewing
- Automatically scales and positions notes on the staff

## How It Works

### Musical Timing
- **0.5 seconds = 1 Quarter Note** (as requested)
- **1.0 second = 1 Half Note**
- **2.0 seconds = 1 Whole Note**
- **4/4 time signature** (4 quarter notes per measure)

### Note Positioning
- Notes are automatically positioned on the staff based on pitch
- C notes appear on the bottom line
- Higher notes appear on higher lines/spaces
- Chords are displayed as chord symbols above the staff

### Real-Time Updates
- As you play, new measures are created automatically
- Each measure shows the notes/chords detected in that time window
- The display scrolls horizontally to show multiple measures

## Features

✅ **Live Updates**: Sheet music updates as you play  
✅ **Automatic Measure Creation**: Groups notes into proper musical measures  
✅ **Note Duration Detection**: Automatically determines note lengths  
✅ **Chord Support**: Displays both individual notes and chord symbols  
✅ **Responsive Design**: Works on different screen sizes  
✅ **Visual Staff**: Traditional 5-line musical staff  

## Usage

### Real-Time Mode
1. Start recording
2. Play your piano
3. Watch the sheet music appear in real-time
4. Each measure shows 2 seconds of music (4/4 time)

### File Analysis Mode
1. Upload an audio file
2. Analyze the audio
3. View the complete sheet music for the entire file

## Customization

You can modify the musical timing by changing these constants in `VexFlowSheetMusic.tsx`:

```typescript
const QUARTER_NOTE_DURATION = 0.5; // Change this to adjust tempo
const MEASURE_DURATION = 4 * QUARTER_NOTE_DURATION; // 4/4 time
```

## Troubleshooting

### Notes Not Appearing
- Check that `detectedNotes` and `detectedChords` arrays are being populated
- Verify the audio analysis is working correctly
- Check console for any JavaScript errors

### Performance Issues
- The component automatically limits to the last 100 events
- Consider reducing the update frequency if needed
- Monitor memory usage on older devices

### Styling Issues
- All styles are defined in the component
- Modify the `styles` object to change appearance
- Colors, sizes, and layouts can be easily customized

## Next Steps

To enhance the sheet music further, consider:

1. **Adding VexFlow Integration**: Replace the custom rendering with actual VexFlow for professional notation
2. **MIDI Export**: Allow users to export the detected music as MIDI files
3. **Sheet Music PDF**: Generate printable sheet music
4. **Tempo Detection**: Automatically detect and adjust to the actual playing tempo
5. **Key Signature Detection**: Automatically determine the musical key

## Support

If you encounter any issues:
1. Check the console for error messages
2. Verify all dependencies are installed correctly
3. Ensure your React Native environment is properly configured
4. Test with simple audio files first

The sheet music component is designed to be robust and handle various edge cases, but let me know if you need any adjustments or have questions!
