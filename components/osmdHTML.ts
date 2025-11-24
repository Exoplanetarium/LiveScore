export const OSMD_HTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
  <style>
    html, body { margin:0; padding:0; background:#fff; }
    #osmd { width: 100vw; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.2/build/opensheetmusicdisplay.min.js"></script>
</head>
<body>
  <div id="osmd"></div>
  <script>
    const post = (m) => {
      try { window.ReactNativeWebView && window.ReactNativeWebView.postMessage(JSON.stringify(m)); } catch(e){}
    };
    let osmd;

    async function init(options) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmd", Object.assign({
        backend: "svg",
        autoResize: true,
        drawTitle: false,
        drawPartNames: false
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
      
      post({ type: "ready" });
    }

    async function renderXml(xml) {
      try {
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
      } catch {}
    }
    window.addEventListener("message", onMessage);
    document.addEventListener("message", onMessage);
    // forward clicks from the webview to the React Native host so it can toggle orientation
    document.addEventListener('click', function(){ post({ type: 'webview-click' }); });

    // auto-init
    init();
  </script>
</body>
</html>
`;