export const OSMD_HTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
  <style>
    html, body { margin:0; padding:0; background:#fff; }
    
    /* Portrait mode: horizontal scroll container */
    #osmd-container {
      width: 100vw;
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
    }
    #osmd-container.portrait-mode {
      /* Enable horizontal scrolling in portrait */
      white-space: nowrap;
    }
    #osmd-container.landscape-mode {
      /* In landscape/fullscreen, use normal layout */
      overflow-x: hidden;
      overflow-y: auto;
    }
    #osmd { 
      display: inline-block;
      min-width: 100vw;
      padding-right: 60vw; /* Extra space at the end so last notes are visible */
    }
    #osmd.portrait-mode {
      /* Force single-line horizontal layout */
      width: max-content;
    }
    #osmd.landscape-mode {
      width: 100vw;
      padding-right: 0;
    }
    
    /* Fullscreen playback controls overlay */
    #fullscreen-controls {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 1000;
      background: rgba(248, 249, 250, 0.95);
      padding: 32px 16px 0px 16px;
      flex-direction: row;
      justify-content: center;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid #ddd;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    #fullscreen-controls.visible {
      display: flex;
    }
    #fullscreen-controls button {
      padding: 8px 16px;
      font-size: 16px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-weight: bold;
      margin-top: 6px; margin-bottom: 6px;
    }
    #fullscreen-controls .play-btn { background: #27ae60; color: white; }
    #fullscreen-controls .play-btn.playing { background: #e67e22; }
    #fullscreen-controls .stop-btn { background: #c0392b; color: white; }
    #fullscreen-controls .exit-btn { background: #555; color: white; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.2/build/opensheetmusicdisplay.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.min.js"></script>
</head>
<body>
  <!-- Fullscreen playback controls -->
  <div id="fullscreen-controls">
    <button class="play-btn" id="fs-play-btn" onclick="handleFsPlay()">▶</button>
    <button class="stop-btn" onclick="handleFsStop()">⏹</button>
    <button class="exit-btn" onclick="handleFsExit()">✕</button>
  </div>
  
  <div id="osmd-container" class="portrait-mode">
    <div id="osmd" class="portrait-mode"></div>
  </div>
  <script>
    const post = (m) => {
      try { window.ReactNativeWebView && window.ReactNativeWebView.postMessage(JSON.stringify(m)); } catch(e){}
    };
    let osmd;
    let currentXml = null;
    let fullscreenMode = false;
    
    // ─── Scroll Cursor Into View (Portrait Mode) ───
    function scrollCursorIntoView() {
      if (fullscreenMode) return; // Don't scroll in landscape/fullscreen mode
      
      const container = document.getElementById('osmd-container');
      if (!container || !osmd || !osmd.cursor) return;
      
      try {
        // Get the cursor element
        const cursorElement = osmd.cursor.cursorElement;
        if (!cursorElement) return;
        
        // Get cursor position
        const cursorRect = cursorElement.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        
        // Calculate where cursor is relative to the visible area
        const cursorLeft = cursorRect.left - containerRect.left;
        const cursorRight = cursorRect.right - containerRect.left;
        const viewportWidth = containerRect.width;
        
        // Keep cursor in the left 40% of the viewport - scroll immediately when it passes 40%
        const targetPosition = viewportWidth * 0.3; // Keep cursor at 30% from left
        const triggerPosition = viewportWidth * 0.4; // Start scrolling when cursor passes 40%
        
        if (cursorLeft > triggerPosition) {
          // Scroll instantly to put cursor at target position
          const scrollAmount = cursorLeft - targetPosition;
          container.scrollLeft += scrollAmount;
        }
      } catch (e) {
        console.warn('Scroll error:', e);
      }
    }
    
    // ─── Fullscreen Control Handlers ───
    function updateFsControls() {
      const playBtn = document.getElementById('fs-play-btn');
      const bpmDisplay = document.getElementById('fs-bpm-display');
      if (playBtn) {
        playBtn.textContent = isPlaying ? (isPaused ? '▶' : '⏸') : '▶';
        playBtn.className = 'play-btn' + (isPlaying && !isPaused ? ' playing' : '');
      }
      if (bpmDisplay) {
        bpmDisplay.textContent = playbackBPM;
      }
    }
    
    function handleFsPlay() {
      if (!isPlaying) {
        startPlayback();
      } else if (isPaused) {
        resumePlayback();
      } else {
        pausePlayback();
      }
    }
    
    function handleFsStop() {
      stopPlayback();
    }
    
    function handleFsExit() {
      post({ type: 'exitFullscreen' });
    }
    
    function setFullscreenMode(enabled) {
      fullscreenMode = enabled;
      const controls = document.getElementById('fullscreen-controls');
      const container = document.getElementById('osmd-container');
      const osmdEl = document.getElementById('osmd');
      
      if (controls) {
        controls.className = enabled ? 'visible' : '';
      }
      
      // Toggle portrait/landscape mode classes
      if (container) {
        container.className = enabled ? 'landscape-mode' : 'portrait-mode';
        // Reset scroll position when switching modes
        if (!enabled) {
          container.scrollLeft = 0;
        }
      }
      if (osmdEl) {
        osmdEl.className = enabled ? 'landscape-mode' : 'portrait-mode';
        osmdEl.style.paddingTop = enabled ? '50px' : '0';
      }
      
      // Update OSMD rendering mode and re-render
      if (osmd && currentXml) {
        // Toggle horizontal scrolling mode
        osmd.setOptions({
          renderSingleHorizontalStaffline: !enabled, // true for portrait, false for landscape
          autoResize: enabled // Auto-resize only in landscape
        });
        
        // Re-render with new settings
        setTimeout(async () => {
          try {
            await osmd.load(currentXml);
            await osmd.render();
          } catch (e) {
            console.warn('Re-render error:', e);
          }
        }, 100);
      }
    }
    
    // ─── Handle Visibility Changes (app backgrounding/foregrounding) ───
    document.addEventListener('visibilitychange', async () => {
      if (document.visibilityState === 'visible') {
        // App came back to foreground - ensure audio context is ready
        console.log('[Audio] App visible, checking audio context...');
        try {
          const ctx = Tone.context;
          if (ctx && ctx.state === 'suspended') {
            await ctx.resume();
            console.log('[Audio] Context resumed after visibility change');
          }
        } catch (e) {
          console.warn('[Audio] Failed to resume context:', e);
        }
      }
    });
    
    // ─── Handle Touch to Unlock Audio ───
    // Mobile browsers require user interaction to start audio
    let audioUnlocked = false;
    document.addEventListener('touchstart', async () => {
      if (!audioUnlocked) {
        try {
          await Tone.start();
          audioUnlocked = true;
          console.log('[Audio] Unlocked via touch');
        } catch (e) {
          console.warn('[Audio] Failed to unlock:', e);
        }
      }
    }, { once: false });
    
    // ─── Playback State ───
    let sampler = null;
    let isPlaying = false;
    let isPaused = false;
    let scheduledEvents = [];
    let playbackStartTime = 0;
    let pausedAtTime = 0;
    let playbackBPM = 120;
    
    // ─── Ensure Audio Context is Ready ───
    async function ensureAudioContext() {
      const ctx = Tone.context;
      
      // Check if context is closed or suspended
      if (ctx.state === 'closed') {
        console.log('[Audio] Context closed, reinitializing...');
        // Force Tone.js to create a new context
        sampler = null;
        await Tone.start();
        await initSampler();
        return true;
      }
      
      if (ctx.state === 'suspended') {
        console.log('[Audio] Context suspended, resuming...');
        await ctx.resume();
        await Tone.start();
      }
      
      // Verify the context is now running
      if (ctx.state !== 'running') {
        console.warn('[Audio] Context state:', ctx.state);
        post({ type: "playbackError", error: "Audio system unavailable. Tap to retry." });
        return false;
      }
      
      return true;
    }
    
    // ─── Initialize Piano Sampler ───
    async function initSampler() {
      if (sampler) return sampler;
      
      // Use free piano samples from a CDN
      sampler = new Tone.Sampler({
        urls: {
          A0: "A0.mp3",
          C1: "C1.mp3",
          "D#1": "Ds1.mp3",
          "F#1": "Fs1.mp3",
          A1: "A1.mp3",
          C2: "C2.mp3",
          "D#2": "Ds2.mp3",
          "F#2": "Fs2.mp3",
          A2: "A2.mp3",
          C3: "C3.mp3",
          "D#3": "Ds3.mp3",
          "F#3": "Fs3.mp3",
          A3: "A3.mp3",
          C4: "C4.mp3",
          "D#4": "Ds4.mp3",
          "F#4": "Fs4.mp3",
          A4: "A4.mp3",
          C5: "C5.mp3",
          "D#5": "Ds5.mp3",
          "F#5": "Fs5.mp3",
          A5: "A5.mp3",
          C6: "C6.mp3",
          "D#6": "Ds6.mp3",
          "F#6": "Fs6.mp3",
          A6: "A6.mp3",
          C7: "C7.mp3",
          "D#7": "Ds7.mp3",
          "F#7": "Fs7.mp3",
          A7: "A7.mp3",
          C8: "C8.mp3"
        },
        release: 1,
        baseUrl: "https://tonejs.github.io/audio/salamander/"
      }).toDestination();
      
      // Wait for samples to load
      await Tone.loaded();
      post({ type: "samplerReady" });
      return sampler;
    }
    
    // ─── MIDI to Note Name ───
    function midiToNoteName(midi) {
      const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      const octave = Math.floor(midi / 12) - 1;
      const note = noteNames[midi % 12];
      return note + octave;
    }
    
    // ─── Parse MusicXML for Playback ───
    function parseMusicXMLForPlayback(xmlString) {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xmlString, "text/xml");
      const notes = [];
      
      // Get divisions (typically 8 in our case)
      const divisionsEl = doc.querySelector('divisions');
      const divisions = divisionsEl ? parseInt(divisionsEl.textContent) : 8;
      
      // Get time signature
      const beatsEl = doc.querySelector('beats');
      const beatTypeEl = doc.querySelector('beat-type');
      const beatsPerMeasure = beatsEl ? parseInt(beatsEl.textContent) : 4;
      const beatType = beatTypeEl ? parseInt(beatTypeEl.textContent) : 4;
      
      // Quarter note duration in seconds at current BPM
      const quarterNoteDuration = 60.0 / playbackBPM;
      
      // Parse each measure
      const measures = doc.querySelectorAll('measure');
      let currentTime = 0; // in seconds
      
      measures.forEach((measure, measureIndex) => {
        let voice1Time = currentTime;
        let voice2Time = currentTime;
        let lastVoice1NoteTime = currentTime;  // Track last note start for chords
        let lastVoice2NoteTime = currentTime;
        
        const elements = measure.children;
        let activeVoice = 1;
        
        for (let i = 0; i < elements.length; i++) {
          const el = elements[i];
          
          if (el.tagName === 'backup') {
            const backupDur = parseInt(el.querySelector('duration')?.textContent || '0');
            const backupSeconds = (backupDur / divisions) * quarterNoteDuration;
            // Backup typically means we're switching from treble to bass staff
            // Reset voice2Time to go back to the start of the measure
            voice2Time = currentTime;  // Reset to measure start
            lastVoice2NoteTime = currentTime;  // Reset chord tracking too
            // Note: we don't use the backupSeconds directly because our measure structure
            // always backs up to the start of the measure for bass staff
          } else if (el.tagName === 'forward') {
            const forwardDur = parseInt(el.querySelector('duration')?.textContent || '0');
            const forwardSeconds = (forwardDur / divisions) * quarterNoteDuration;
            if (activeVoice === 1) {
              voice1Time += forwardSeconds;
            } else {
              voice2Time += forwardSeconds;
            }
          } else if (el.tagName === 'note') {
            const isRest = el.querySelector('rest') !== null;
            const isChord = el.querySelector('chord') !== null;
            const durationEl = el.querySelector('duration');
            const duration = durationEl ? parseInt(durationEl.textContent) : divisions;
            const durationSeconds = (duration / divisions) * quarterNoteDuration;
            
            // Get voice
            const voiceEl = el.querySelector('voice');
            activeVoice = voiceEl ? parseInt(voiceEl.textContent) : 1;
            const voiceTime = activeVoice === 1 ? voice1Time : voice2Time;
            
            if (!isRest) {
              // Get pitch
              const pitchEl = el.querySelector('pitch');
              if (pitchEl) {
                const step = pitchEl.querySelector('step')?.textContent || 'C';
                const alterEl = pitchEl.querySelector('alter');
                const alter = alterEl ? parseInt(alterEl.textContent) : 0;
                const octave = parseInt(pitchEl.querySelector('octave')?.textContent || '4');
                
                // Convert to MIDI
                const stepToSemitone = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
                const midi = (octave + 1) * 12 + stepToSemitone[step] + alter;
                const noteName = midiToNoteName(midi);
                
                // If chord, use same start time as previous note in this voice
                const lastNoteTime = activeVoice === 1 ? lastVoice1NoteTime : lastVoice2NoteTime;
                const noteStartTime = isChord ? lastNoteTime : voiceTime;
                
                notes.push({
                  time: noteStartTime,
                  note: noteName,
                  duration: Math.max(0.1, durationSeconds * 0.9), // slightly shorter for separation
                  midi: midi
                });
                
                // Update last note time for this voice (for chord detection)
                if (!isChord) {
                  if (activeVoice === 1) {
                    lastVoice1NoteTime = noteStartTime;
                  } else {
                    lastVoice2NoteTime = noteStartTime;
                  }
                }
              }
            }
            
            // Advance time only for non-chord notes
            if (!isChord) {
              if (activeVoice === 1) {
                voice1Time += durationSeconds;
              } else {
                voice2Time += durationSeconds;
              }
            }
          }
        }
        
        // Move to next measure (use the furthest voice position)
        currentTime = Math.max(voice1Time, voice2Time);
      });
      
      // Sort by time
      notes.sort((a, b) => a.time - b.time);
      return notes;
    }
    
    // ─── Start Playback ───
    async function startPlayback(bpm) {
      if (isPlaying && !isPaused) return;
      
      if (!currentXml) {
        post({ type: "playbackError", error: "No music to play" });
        return;
      }
      
      playbackBPM = bpm || 120;
      
      // Ensure audio context is ready (handles disconnection/suspension)
      const audioReady = await ensureAudioContext();
      if (!audioReady) return;
      
      // Initialize sampler if needed
      await initSampler();
      
      // Double-check audio context after sampler init
      await Tone.start();
      await Tone.start();
      
      if (isPaused) {
        // Resume from paused position
        Tone.Transport.start();
        isPlaying = true;
        isPaused = false;
        updateFsControls();
        post({ type: "playbackResumed" });
        return;
      }
      
      // Parse the XML
      const notes = parseMusicXMLForPlayback(currentXml);
      
      if (notes.length === 0) {
        post({ type: "playbackError", error: "No notes found in score" });
        return;
      }
      
      // Clear any existing scheduled events
      Tone.Transport.cancel();
      scheduledEvents = [];
      
      // Reset cursor and scroll to start
      if (osmd && osmd.cursor) {
        osmd.cursor.reset();
        osmd.cursor.show();
      }
      // Scroll container to start in portrait mode
      const container = document.getElementById('osmd-container');
      if (container && !fullscreenMode) {
        container.scrollLeft = 0;
      }
      
      // Schedule all notes
      const totalDuration = notes[notes.length - 1].time + notes[notes.length - 1].duration + 0.5;
      
      notes.forEach((noteEvent, index) => {
        const eventId = Tone.Transport.schedule((time) => {
          sampler.triggerAttackRelease(noteEvent.note, noteEvent.duration, time);
        }, noteEvent.time);
        scheduledEvents.push(eventId);
      });
      
      // Schedule cursor advances based on OSMD's internal structure
      // Use the actual graphical note entries from OSMD for accurate sync
      if (osmd && osmd.cursor && osmd.Sheet) {
        try {
          // Get all the timestamp positions from the cursor iterator
          osmd.cursor.reset();
          let iteratorPositions = [];
          let safetyCounter = 0;
          const maxIterations = 10000; // Prevent infinite loop
          
          while (!osmd.cursor.iterator.EndReached && safetyCounter < maxIterations) {
            const timestamp = osmd.cursor.iterator.CurrentSourceTimestamp;
            if (timestamp) {
              // Convert OSMD timestamp (in fractions) to seconds
              // OSMD uses quarter notes as the base, so timestamp.RealValue * 4 = beats
              const beats = timestamp.RealValue * 4;
              const timeInSeconds = beats * (60.0 / playbackBPM);
              iteratorPositions.push(timeInSeconds);
            }
            osmd.cursor.next();
            safetyCounter++;
          }
          
          // Reset cursor to start
          osmd.cursor.reset();
          osmd.cursor.show();
          
          // Schedule cursor movements at each position
          let cursorIndex = 0;
          iteratorPositions.forEach((posTime, idx) => {
            if (idx > 0) { // Skip first position (cursor already there)
              Tone.Transport.schedule((time) => {
                Tone.Draw.schedule(() => {
                  if (osmd && osmd.cursor) {
                    osmd.cursor.next();
                    scrollCursorIntoView();
                  }
                }, time);
              }, posTime);
            }
          });
        } catch (e) {
          console.warn('Cursor sync error, falling back to beat-based:', e);
          // Fallback: advance cursor on each beat
          const quarterNoteDuration = 60.0 / playbackBPM;
          let cursorTime = quarterNoteDuration;
          while (cursorTime < totalDuration) {
            const t = cursorTime;
            Tone.Transport.schedule((time) => {
              Tone.Draw.schedule(() => {
                if (osmd && osmd.cursor) {
                  osmd.cursor.next();
                  scrollCursorIntoView();
                }
              }, time);
            }, cursorTime);
            cursorTime += quarterNoteDuration;
          }
        }
      }
      
      // Schedule end of playback
      Tone.Transport.schedule((time) => {
        stopPlayback();
        post({ type: "playbackEnded" });
      }, totalDuration);
      
      // Start transport
      Tone.Transport.start();
      isPlaying = true;
      isPaused = false;
      playbackStartTime = Tone.now();
      updateFsControls();
      
      post({ type: "playbackStarted", noteCount: notes.length, duration: totalDuration });
    }
    
    // ─── Pause Playback ───
    function pausePlayback() {
      if (!isPlaying || isPaused) return;
      
      Tone.Transport.pause();
      isPaused = true;
      pausedAtTime = Tone.Transport.seconds;
      updateFsControls();
      post({ type: "playbackPaused", pausedAt: pausedAtTime });
    }
    
    // ─── Stop Playback ───
    function stopPlayback() {
      Tone.Transport.stop();
      Tone.Transport.cancel();
      scheduledEvents = [];
      isPlaying = false;
      isPaused = false;
      updateFsControls();
      
      // Reset cursor
      if (osmd && osmd.cursor) {
        osmd.cursor.reset();
      }
      
      post({ type: "playbackStopped" });
    }
    
    // ─── Set Playback BPM ───
    function setPlaybackBPM(bpm) {
      playbackBPM = Math.max(40, Math.min(240, bpm));
      updateFsControls();
      post({ type: "bpmSet", bpm: playbackBPM });
    }

    async function init(options) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmd", Object.assign({
        backend: "svg",
        autoResize: false, // We'll control sizing manually
        drawTitle: false,
        drawPartNames: false,
        // Render on a single horizontal line for portrait scrolling
        renderSingleHorizontalStaffline: true
      }, options||{}));
      
      // Configure engraving rules for better spacing with many short notes
      if (osmd.EngravingRules) {
        // Minimum note spacing to prevent squishing
        osmd.EngravingRules.MinSkyBottomDistBetweenStaves = 3;
        osmd.EngravingRules.StaffDistance = 8;
        osmd.EngravingRules.BetweenStaffDistance = 5;
        
        // Note spacing - ensure minimum distance between notes
        osmd.EngravingRules.MinNoteDistance = 2.0;
        osmd.EngravingRules.VoiceSpacingMultiplierVexflow = 1.0;
        osmd.EngravingRules.VoiceSpacingAddendVexflow = 3.0;
        
        // Measure width settings - critical for preventing squishing
        osmd.EngravingRules.MeasureMinimumWidth = 150;
        osmd.EngravingRules.FixedMeasureWidth = false;
        osmd.EngravingRules.FixedMeasureWidthFixedValue = 0;
        osmd.EngravingRules.FixedMeasureWidthUseForPickupMeasure = false;
        
        // Allow measures to expand based on content
        osmd.EngravingRules.LastSystemMaxScalingFactor = 1.5;
        osmd.EngravingRules.NewSystemAtXMLNewSystemAttribute = true;
        osmd.EngravingRules.NewPageAtXMLNewPageAttribute = true;
        
        // Better handling of beaming and note grouping
        osmd.EngravingRules.AutoBeamNotes = true;
        osmd.EngravingRules.AutoBeamOptions = {
          beam_rests: false,
          beam_middle_rests_only: false,
          maintain_stem_directions: true
        };
      }
      
      // Pre-load the sampler in the background
      initSampler().catch(e => console.warn('Sampler init failed:', e));
      
      post({ type: "ready" });
    }

    async function renderXml(xml) {
      try {
        currentXml = xml; // Store for playback
        await osmd.load(xml);
        await osmd.render();
        post({ type: "rendered", measures: osmd.Sheet?.Measures?.length || 0 });
      } catch (e) {
        post({ type: "error", error: String(e) });
      }
    }

    function setZoom(z){ osmd.Zoom = Math.max(0.3, Math.min(3, z)); osmd.render(); }
    function toggleCursor(show){
      if (!osmd) return;
      if (show) osmd.cursor.show(); else osmd.cursor.hide();
    }
    function cursorNext(){ osmd?.cursor?.next(); }
    function cursorReset(){ osmd?.cursor?.reset(); }

    function onMessage(e){
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "init") return init(msg.options);
        if (msg.type === "renderXml") return renderXml(msg.xml);
        if (msg.type === "setZoom") return setZoom(msg.zoom);
        if (msg.type === "toggleCursor") return toggleCursor(msg.show);
        if (msg.type === "cursorNext") return cursorNext();
        if (msg.type === "cursorReset") return cursorReset();
        // Playback controls
        if (msg.type === "play") return startPlayback(msg.bpm);
        if (msg.type === "pause") return pausePlayback();
        if (msg.type === "stop") return stopPlayback();
        if (msg.type === "setBPM") return setPlaybackBPM(msg.bpm);
        // Fullscreen mode
        if (msg.type === "setFullscreenMode") return setFullscreenMode(msg.enabled);
      } catch {}
    }
    window.addEventListener("message", onMessage);
    document.addEventListener("message", onMessage);
    // forward clicks from the webview to the React Native host so it can enter fullscreen
    // Only send click when NOT in fullscreen mode (exit is handled by the exit button)
    document.addEventListener('click', function(e){
      // Don't trigger fullscreen toggle if clicking on the controls
      if (e.target.closest('#fullscreen-controls')) return;
      // Only send click to enter fullscreen, not to exit
      if (!fullscreenMode) {
        post({ type: 'webview-click' });
      }
    });

    // auto-init
    init();
  </script>
</body>
</html>
`;