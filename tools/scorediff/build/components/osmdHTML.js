"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OSMD_HTML = void 0;
exports.OSMD_HTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      min-height: 100%;
      background: linear-gradient(180deg, #ffffff 0%, #f5f7fb 100%);
      overflow: hidden;
    }
    body {
      overscroll-behavior: none;
      touch-action: manipulation;
    }

    #osmd-container {
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 560px;
      overflow-x: hidden;
      overflow-y: auto;
      background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%);
      box-sizing: border-box;
      overscroll-behavior-y: contain;
      -webkit-overflow-scrolling: touch;
      touch-action: pan-y pinch-zoom;
    }
    #osmd-container.landscape-mode {
      overflow-x: hidden;
      overflow-y: auto;
      touch-action: pan-y pinch-zoom;
    }

    #osmd-stage {
      position: relative;
      display: block;
      width: 100%;
      min-height: 100%;
      box-sizing: border-box;
      will-change: transform;
      transform: translate3d(0px, 0px, 0px);
    }
    #osmd-stage.portrait-mode {
      padding: 16px 12px 24px 12px;
      transform: translate3d(0px, 0px, 0px) !important;
    }
    #osmd-stage.landscape-mode {
      display: block;
      width: auto;
      padding: 68px 24px 24px 24px;
      transform: translate3d(0px, 0px, 0px) !important;
    }

    #osmd {
      display: block;
      width: 100%;
    }
    #osmd.portrait-mode {
      width: 100%;
      min-width: 0;
    }
    #osmd.landscape-mode {
      width: 100%;
      min-width: 0;
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
  <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.2/build/opensheetmusicdisplay.min.js"><\/script>
  <script src="https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.min.js"><\/script>
</head>
<body>
  <!-- Fullscreen playback controls -->
  <div id="fullscreen-controls">
    <button class="play-btn" id="fs-play-btn" onclick="handleFsPlay()">▶</button>
    <button class="stop-btn" onclick="handleFsStop()">⏹</button>
    <button class="exit-btn" onclick="handleFsExit()">✕</button>
  </div>
  
  <div id="osmd-container" class="portrait-mode">
    <div id="osmd-stage" class="portrait-mode">
      <div id="osmd" class="portrait-mode"></div>
    </div>
  </div>
  <script>
    const post = (m) => {
      try { window.ReactNativeWebView && window.ReactNativeWebView.postMessage(JSON.stringify(m)); } catch(e){}
    };
    let osmd;
    let currentXml = null;
    let fullscreenMode = false;
    let followTail = false;
    let followTailScheduleFrame = null;
    let cameraAnimationFrame = null;
    let cameraMode = 'smooth';
    let cameraSuspendUntil = 0;
    let suppressScoreTapUntil = 0;
    let scoreScrollActive = false;
    let scoreScrollReleaseTimer = null;
    const PORTRAIT_SVG_SCALE = 0.42;
    const PORTRAIT_LAYOUT_BOX_WIDTH_MULTIPLIER = 1.08;
    const PORTRAIT_MEASURE_MIN_WIDTH = 110;
    const LANDSCAPE_MEASURE_MIN_WIDTH = 190;
    const PORTRAIT_MIN_NOTE_DISTANCE = 3.1;
    const LANDSCAPE_MIN_NOTE_DISTANCE = 2.8;
    const PORTRAIT_VOICE_SPACING_ADDEND = 4.6;
    const LANDSCAPE_VOICE_SPACING_ADDEND = 4.5;
    const cameraState = {
      currentX: 0,
      targetX: 0,
      maxX: 0,
      viewportWidth: 0,
      viewportHeight: 0,
      contentWidth: 0,
      contentHeight: 0,
      paddingLeft: 0,
      paddingRight: 0,
      paddingTop: 0,
      paddingBottom: 0,
      measureWidths: [],
      measureAnchors: [],
      averageMeasureWidth: 0,
    };
    const dragState = {
      active: false,
      startX: 0,
      startCameraX: 0,
    };

    function getContainer() {
      return document.getElementById('osmd-container');
    }

    function getStage() {
      return document.getElementById('osmd-stage');
    }

    function getScoreRoot() {
      return document.getElementById('osmd');
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function parsePixels(value) {
      const parsed = parseFloat(value || '0');
      return Number.isFinite(parsed) ? parsed : 0;
    }

    function firstFinite() {
      for (let index = 0; index < arguments.length; index++) {
        const value = arguments[index];
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value;
        }
      }
      return null;
    }

    function clearScheduledFollowTail() {
      if (followTailScheduleFrame != null) {
        cancelAnimationFrame(followTailScheduleFrame);
        followTailScheduleFrame = null;
      }
    }

    function setScoreScrollActive(active) {
      if (scoreScrollReleaseTimer != null) {
        clearTimeout(scoreScrollReleaseTimer);
        scoreScrollReleaseTimer = null;
      }

      if (scoreScrollActive === active) {
        return;
      }

      scoreScrollActive = active;
      post({ type: 'scoreScrollActive', active: active });
    }

    function scheduleScoreScrollInactive() {
      if (scoreScrollReleaseTimer != null) {
        clearTimeout(scoreScrollReleaseTimer);
      }

      scoreScrollReleaseTimer = setTimeout(function() {
        scoreScrollReleaseTimer = null;
        setScoreScrollActive(false);
      }, 140);
    }

    function beginScoreScrollGesture() {
      if (fullscreenMode || !usesWrappedPortraitLayout()) return;
      setScoreScrollActive(true);
    }

    function markScoreScrollInteraction() {
      if (fullscreenMode || !usesWrappedPortraitLayout()) return;
      suppressScoreTapUntil = Date.now() + 250;
      setScoreScrollActive(true);
      scheduleScoreScrollInactive();
    }

    function usesWrappedPortraitLayout() {
      return !fullscreenMode;
    }

    function getRenderedSvgScale() {
      return fullscreenMode ? 1 : PORTRAIT_SVG_SCALE;
    }

    function getPortraitVirtualLayoutWidthPercent() {
      const scaledWidthFactor = PORTRAIT_SVG_SCALE * PORTRAIT_LAYOUT_BOX_WIDTH_MULTIPLIER;
      if (!Number.isFinite(scaledWidthFactor) || scaledWidthFactor <= 0) {
        return 100;
      }
      return 100 / scaledWidthFactor;
    }

    function applyLayoutViewportWidth(useVirtualWidth) {
      const scoreRoot = getScoreRoot();
      if (!scoreRoot) return;

      if (useVirtualWidth && usesWrappedPortraitLayout()) {
        // Let OSMD line-break against a wider virtual page, then shrink the SVG after render.
        scoreRoot.style.width = getPortraitVirtualLayoutWidthPercent().toFixed(2) + '%';
        scoreRoot.style.maxWidth = 'none';
        return;
      }

      scoreRoot.style.width = '100%';
      scoreRoot.style.maxWidth = '100%';
    }

    function applyEngravingDensity() {
      if (!osmd || !osmd.EngravingRules) return;

      const portraitWrapped = usesWrappedPortraitLayout();

      osmd.EngravingRules.MinSkyBottomDistBetweenStaves = 3;
      osmd.EngravingRules.StaffDistance = 8;
      osmd.EngravingRules.BetweenStaffDistance = 5;
      osmd.EngravingRules.MinNoteDistance = portraitWrapped
        ? PORTRAIT_MIN_NOTE_DISTANCE
        : LANDSCAPE_MIN_NOTE_DISTANCE;
      osmd.EngravingRules.VoiceSpacingMultiplierVexflow = 1.0;
      osmd.EngravingRules.VoiceSpacingAddendVexflow = portraitWrapped
        ? PORTRAIT_VOICE_SPACING_ADDEND
        : LANDSCAPE_VOICE_SPACING_ADDEND;
      osmd.EngravingRules.MeasureMinimumWidth = portraitWrapped
        ? PORTRAIT_MEASURE_MIN_WIDTH
        : LANDSCAPE_MEASURE_MIN_WIDTH;
      osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem = portraitWrapped ? 3 : 0;
      osmd.EngravingRules.FixedMeasureWidth = false;
      osmd.EngravingRules.FixedMeasureWidthFixedValue = 0;
      osmd.EngravingRules.FixedMeasureWidthUseForPickupMeasure = false;
      osmd.EngravingRules.AutoGenerateMultipleRestMeasuresFromRestMeasures = false;
      osmd.EngravingRules.RenderMultipleRestMeasures = false;
      osmd.EngravingRules.LastSystemMaxScalingFactor = 2.2;
      osmd.EngravingRules.NewSystemAtXMLNewSystemAttribute = false;
      osmd.EngravingRules.NewPageAtXMLNewPageAttribute = false;
      osmd.EngravingRules.AutoBeamNotes = true;
      osmd.EngravingRules.AutoBeamOptions = {
        beam_rests: false,
        beam_middle_rests_only: false,
        maintain_stem_directions: true
      };
    }

    function applyLayoutOptions() {
      if (!osmd) return;

      applyLayoutViewportWidth(true);

      osmd.setOptions({
        renderSingleHorizontalStaffline: false,
        autoResize: fullscreenMode,
        newSystemFromXML: false,
        newSystemFromNewPageInXML: false,
        newPageFromXML: false,
      });
      applyEngravingDensity();
    }

    function applyRenderedSvgScale() {
      const scoreRoot = getScoreRoot();
      if (!scoreRoot) return;

      applyLayoutViewportWidth(false);

      const scale = getRenderedSvgScale();
      const portraitWrapped = usesWrappedPortraitLayout();
      const svgNodes = scoreRoot.querySelectorAll('svg');
      svgNodes.forEach((svg) => {
        let baseWidth = parseFloat(svg.getAttribute('width') || '0');
        let baseHeight = parseFloat(svg.getAttribute('height') || '0');

        if ((!Number.isFinite(baseWidth) || baseWidth <= 0 || !Number.isFinite(baseHeight) || baseHeight <= 0) && svg.viewBox && svg.viewBox.baseVal) {
          baseWidth = svg.viewBox.baseVal.width;
          baseHeight = svg.viewBox.baseVal.height;
        }

        let contentWidth = baseWidth;
        let contentHeight = baseHeight;
        try {
          if (typeof svg.getBBox === 'function') {
            const bbox = svg.getBBox();
            if (bbox) {
              const bboxRight = Number.isFinite(bbox.x) && Number.isFinite(bbox.width)
                ? bbox.x + bbox.width
                : bbox.width;
              const bboxBottom = Number.isFinite(bbox.y) && Number.isFinite(bbox.height)
                ? bbox.y + bbox.height
                : bbox.height;

              if (Number.isFinite(bbox.width) && bbox.width > 0) {
                contentWidth = Math.max(contentWidth || 0, bbox.width, bboxRight || 0);
              }
              if (Number.isFinite(bbox.height) && bbox.height > 0) {
                contentHeight = Math.max(contentHeight || 0, bbox.height, bboxBottom || 0);
              }
            }
          }
        } catch (e) {}

        let shell = svg.parentElement;
        if (!shell || !shell.classList || !shell.classList.contains('osmd-svg-scale-shell')) {
          shell = document.createElement('div');
          shell.className = 'osmd-svg-scale-shell';
          shell.style.position = 'relative';
          shell.style.overflow = 'visible';
          shell.style.display = 'block';
          if (svg.parentNode) {
            svg.parentNode.insertBefore(shell, svg);
            shell.appendChild(svg);
          }
        }

        if (scale === 1) {
          shell.style.width = '';
          shell.style.height = '';
          svg.style.width = '';
          svg.style.height = '';
          svg.style.maxWidth = '';
          svg.style.display = '';
          svg.style.transform = '';
          svg.style.transformOrigin = '';
          return;
        }

        const layoutWidth = Number.isFinite(contentWidth) && contentWidth > 0
          ? Math.max(1, contentWidth * scale * PORTRAIT_LAYOUT_BOX_WIDTH_MULTIPLIER)
          : 0;
        const layoutHeight = Number.isFinite(contentHeight) && contentHeight > 0
          ? Math.max(1, contentHeight * scale)
          : 0;

        if (layoutWidth > 0) {
          shell.style.width = layoutWidth.toFixed(2) + 'px';
        }
        if (layoutHeight > 0) {
          shell.style.height = layoutHeight.toFixed(2) + 'px';
        }

        if (Number.isFinite(contentWidth) && contentWidth > 0) {
          svg.style.width = Math.max(1, contentWidth).toFixed(2) + 'px';
          svg.style.maxWidth = 'none';
        }
        if (Number.isFinite(contentHeight) && contentHeight > 0) {
          svg.style.height = Math.max(1, contentHeight).toFixed(2) + 'px';
        }
        svg.style.display = 'block';
        svg.style.transformOrigin = 'top left';
        svg.style.transform = 'scale(' + scale.toFixed(4) + ')';

        if (portraitWrapped) {
          const renderedRect = svg.getBoundingClientRect();
          const renderedHeight = Number.isFinite(renderedRect.height)
            ? renderedRect.height
            : 0;
          if (renderedHeight > 0) {
            shell.style.height = Math.max(layoutHeight, renderedHeight).toFixed(2) + 'px';
          }
        }
      });

      syncPortraitScrollExtent();
    }

    function syncPortraitScrollExtent() {
      const scoreRoot = getScoreRoot();
      const stage = getStage();
      if (!scoreRoot || !stage) return;

      if (!usesWrappedPortraitLayout()) {
        scoreRoot.style.minHeight = '';
        stage.style.minHeight = '';
        return;
      }

      const shells = scoreRoot.querySelectorAll('.osmd-svg-scale-shell');
      let maxBottom = 0;

      for (let index = 0; index < shells.length; index++) {
        const shell = shells[index];
        const rect = shell.getBoundingClientRect();
        const shellHeight = Math.max(
          shell.offsetHeight || 0,
          Number.isFinite(rect.height) ? rect.height : 0,
        );
        maxBottom = Math.max(maxBottom, shell.offsetTop + shellHeight);
      }

      const fallbackHeight = Math.max(
        scoreRoot.scrollHeight || 0,
        Math.ceil(scoreRoot.getBoundingClientRect().height || 0),
      );
      const targetHeight = Math.max(
        fallbackHeight,
        Math.ceil(maxBottom + (cameraState.paddingBottom || 24)),
      );

      if (targetHeight > 0) {
        const targetHeightPx = targetHeight + 'px';
        scoreRoot.style.minHeight = targetHeightPx;
        stage.style.minHeight = targetHeightPx;
      }
    }

    function cancelCameraAnimation() {
      if (cameraAnimationFrame != null) {
        cancelAnimationFrame(cameraAnimationFrame);
        cameraAnimationFrame = null;
      }
    }

    function getCameraPaddingPreset(viewportWidth) {
      if (fullscreenMode) {
        return { left: 24, right: 24, top: 68, bottom: 24 };
      }

      return {
        left: 12,
        right: 12,
        top: 16,
        bottom: 24,
      };
    }

    function updateStagePadding() {
      const container = getContainer();
      const stage = getStage();
      if (!container || !stage) return;

      const padding = getCameraPaddingPreset(container.clientWidth || window.innerWidth || 0);
      stage.style.paddingLeft = padding.left + 'px';
      stage.style.paddingRight = padding.right + 'px';
      stage.style.paddingTop = padding.top + 'px';
      stage.style.paddingBottom = padding.bottom + 'px';
      cameraState.paddingLeft = padding.left;
      cameraState.paddingRight = padding.right;
      cameraState.paddingTop = padding.top;
      cameraState.paddingBottom = padding.bottom;
    }

    function buildFallbackMeasureMetrics(contentWidth, measureCount) {
      if (!measureCount || contentWidth <= 0) {
        return {
          measureWidths: [],
          measureAnchors: [],
          averageMeasureWidth: 0,
        };
      }

      const averageMeasureWidth = contentWidth / measureCount;
      const measureWidths = [];
      const measureAnchors = [];

      for (let index = 0; index < measureCount; index++) {
        const left = averageMeasureWidth * index;
        const width = averageMeasureWidth;
        measureWidths.push({
          index: index,
          left: left,
          width: width,
          right: left + width,
          center: left + width / 2,
        });
        measureAnchors.push(left + width / 2);
      }

      return {
        measureWidths: measureWidths,
        measureAnchors: measureAnchors,
        averageMeasureWidth: averageMeasureWidth,
      };
    }

    function readMeasureBounds(measure) {
      try {
        const left = firstFinite(
          measure && measure.PositionAndShape && measure.PositionAndShape.AbsolutePosition && measure.PositionAndShape.AbsolutePosition.x,
          measure && measure.boundingBox && measure.boundingBox.absolutePosition && measure.boundingBox.absolutePosition.x,
          measure && measure.boundingBox && measure.boundingBox.relativePosition && measure.boundingBox.relativePosition.x,
          measure && measure.boundingBox && measure.boundingBox.x,
        );
        const width = firstFinite(
          measure && measure.PositionAndShape && measure.PositionAndShape.Size && measure.PositionAndShape.Size.width,
          measure && measure.boundingBox && measure.boundingBox.size && measure.boundingBox.size.width,
          measure && measure.boundingBox && measure.boundingBox.w,
          measure && measure.boundingBox && measure.boundingBox.width,
        );
        if (left != null && width != null && width > 0) {
          return { left: left, width: width };
        }
      } catch (e) {}
      return null;
    }

    function collectMeasureMetrics(contentWidth) {
      const rawMeasures = [];
      const measureCount = osmd && osmd.Sheet && osmd.Sheet.Measures ? osmd.Sheet.Measures.length : 0;

      try {
        const graphicMeasureList = osmd && osmd.GraphicSheet && osmd.GraphicSheet.MeasureList;

        if (graphicMeasureList && graphicMeasureList.length) {
          for (let systemIndex = 0; systemIndex < graphicMeasureList.length; systemIndex++) {
            const systemMeasures = graphicMeasureList[systemIndex];
            if (!systemMeasures) continue;

            // MeasureList[i] is normally an array of staff-measures, but be defensive
            // in case a build returns a single measure object directly.
            if (typeof systemMeasures.length === 'number') {
              for (let measureIndex = 0; measureIndex < systemMeasures.length; measureIndex++) {
                const bounds = readMeasureBounds(systemMeasures[measureIndex]);
                if (bounds) rawMeasures.push(bounds);
              }
            } else {
              const bounds = readMeasureBounds(systemMeasures);
              if (bounds) rawMeasures.push(bounds);
            }
          }
        }
      } catch (e) {
        // OSMD shape mismatch — fall through to fallback.
      }

      if (!rawMeasures.length) {
        return buildFallbackMeasureMetrics(contentWidth, measureCount);
      }

      let rawRight = 0;
      for (let index = 0; index < rawMeasures.length; index++) {
        rawRight = Math.max(rawRight, rawMeasures[index].left + rawMeasures[index].width);
      }

      const scale = rawRight > 0 ? contentWidth / rawRight : 1;
      const measureWidths = rawMeasures.map((measure, index) => {
        const left = measure.left * scale;
        const width = measure.width * scale;
        return {
          index: index,
          left: left,
          width: width,
          right: left + width,
          center: left + width / 2,
        };
      });
      const measureAnchors = measureWidths.map((measure) => measure.center);
      const averageMeasureWidth =
        measureWidths.reduce((sum, measure) => sum + measure.width, 0) /
        measureWidths.length;

      return {
        measureWidths: measureWidths,
        measureAnchors: measureAnchors,
        averageMeasureWidth: averageMeasureWidth,
      };
    }

    function updateCameraMetrics() {
      const container = getContainer();
      const stage = getStage();
      const scoreRoot = getScoreRoot();
      if (!container || !stage || !scoreRoot) return;

      updateStagePadding();

      const stageStyle = window.getComputedStyle(stage);
      const paddingLeft = parsePixels(stageStyle.paddingLeft);
      const paddingRight = parsePixels(stageStyle.paddingRight);
      const contentWidth = Math.max(
        scoreRoot.scrollWidth,
        Math.ceil(scoreRoot.getBoundingClientRect().width || 0),
      );
      const contentHeight = Math.max(
        scoreRoot.scrollHeight,
        Math.ceil(scoreRoot.getBoundingClientRect().height || 0),
      );
      const totalWidth = paddingLeft + contentWidth + paddingRight;
      const measureMetrics = collectMeasureMetrics(contentWidth);

      cameraState.viewportWidth = container.clientWidth;
      cameraState.viewportHeight = container.clientHeight;
      cameraState.contentWidth = contentWidth;
      cameraState.contentHeight = contentHeight;
      cameraState.maxX = usesWrappedPortraitLayout()
        ? 0
        : Math.max(0, totalWidth - container.clientWidth);
      cameraState.measureWidths = measureMetrics.measureWidths;
      cameraState.measureAnchors = measureMetrics.measureAnchors;
      cameraState.averageMeasureWidth = measureMetrics.averageMeasureWidth;
      cameraState.currentX = clamp(cameraState.currentX, 0, cameraState.maxX);
      cameraState.targetX = clamp(cameraState.targetX, 0, cameraState.maxX);

      if (fullscreenMode || usesWrappedPortraitLayout()) {
        const stageNode = getStage();
        if (stageNode) {
          stageNode.style.transition = 'none';
          stageNode.style.transform = 'translate3d(0px, 0px, 0px)';
        }
      }
    }

    function setStageTransform(nextX, immediate) {
      const stage = getStage();
      if (!stage) return;

      const clampedX = clamp(nextX, 0, cameraState.maxX || 0);
      cameraState.currentX = clampedX;

      if (fullscreenMode || usesWrappedPortraitLayout()) {
        stage.style.transition = 'none';
        stage.style.transform = 'translate3d(0px, 0px, 0px)';
        return;
      }

      stage.style.transition =
        immediate || cameraMode === 'smooth'
          ? 'none'
          : 'transform 240ms cubic-bezier(0.22, 1, 0.36, 1)';
      stage.style.transform = 'translate3d(' + (-clampedX).toFixed(2) + 'px, 0px, 0px)';
    }

    function animateCameraToTarget() {
      cancelCameraAnimation();

      if (fullscreenMode) return;

      const step = () => {
        const delta = cameraState.targetX - cameraState.currentX;
        if (Math.abs(delta) < 0.5) {
          setStageTransform(cameraState.targetX, true);
          cameraAnimationFrame = null;
          return;
        }

        setStageTransform(cameraState.currentX + delta * 0.16, true);
        cameraAnimationFrame = requestAnimationFrame(step);
      };

      cameraAnimationFrame = requestAnimationFrame(step);
    }

    function moveCameraTo(nextX, immediate) {
      const clampedTarget = clamp(nextX, 0, cameraState.maxX || 0);
      cameraState.targetX = clampedTarget;

      if (fullscreenMode || usesWrappedPortraitLayout()) {
        setStageTransform(0, true);
        return;
      }

      if (cameraMode === 'smooth' && !immediate) {
        animateCameraToTarget();
        return;
      }

      cancelCameraAnimation();
      setStageTransform(clampedTarget, !!immediate);
    }

    function getRelativeFocusX(node, edge) {
      const scoreRoot = getScoreRoot();
      if (!scoreRoot || !node) return null;

      const rootRect = scoreRoot.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      const focusX =
        edge === 'right'
          ? nodeRect.right - rootRect.left
          : nodeRect.left - rootRect.left + nodeRect.width / 2;

      if (!Number.isFinite(focusX)) return null;

      return clamp(focusX, 0, cameraState.contentWidth || focusX);
    }

    function getCursorFocusX() {
      if (!osmd || !osmd.cursor || !osmd.cursor.cursorElement) return null;
      return getRelativeFocusX(osmd.cursor.cursorElement, 'center');
    }

    function getLatestContentFocusX() {
      const scoreRoot = getScoreRoot();
      if (!scoreRoot) return null;

      const svgNodes = scoreRoot.querySelectorAll('svg');
      const lastVisualNode = svgNodes.length > 0 ? svgNodes[svgNodes.length - 1] : scoreRoot.lastElementChild || scoreRoot;
      const focusX = getRelativeFocusX(lastVisualNode, 'right');
      if (focusX == null) return null;

      return Math.max(0, focusX - Math.min(cameraState.viewportWidth * 0.18, 96));
    }

    function getLeadOffset() {
      return Math.round(cameraState.viewportWidth * 0.34);
    }

    function findNearestMeasure(focusX) {
      const measures = cameraState.measureWidths;
      if (!measures || !measures.length) return null;

      for (let index = 0; index < measures.length; index++) {
        const measure = measures[index];
        if (focusX >= measure.left && focusX <= measure.right) {
          return measure;
        }
      }

      let nearestMeasure = measures[0];
      let nearestDistance = Math.abs(measures[0].center - focusX);
      for (let index = 1; index < measures.length; index++) {
        const distance = Math.abs(measures[index].center - focusX);
        if (distance < nearestDistance) {
          nearestMeasure = measures[index];
          nearestDistance = distance;
        }
      }

      return nearestMeasure;
    }

    function getIdealCameraTarget(focusX) {
      const focusInStage = cameraState.paddingLeft + focusX;
      return clamp(focusInStage - getLeadOffset(), 0, cameraState.maxX || 0);
    }

    function getSnapCameraTarget(focusX) {
      const focusInViewport = cameraState.paddingLeft + focusX - cameraState.currentX;
      const leftBand = cameraState.viewportWidth * 0.24;
      const rightBand = cameraState.viewportWidth * 0.62;

      if (focusInViewport >= leftBand && focusInViewport <= rightBand) {
        return cameraState.currentX;
      }

      const measure = findNearestMeasure(focusX);
      if (measure) {
        return clamp(
          cameraState.paddingLeft + measure.left - cameraState.viewportWidth * 0.18,
          0,
          cameraState.maxX || 0,
        );
      }

      return getIdealCameraTarget(focusX);
    }

    function updateCameraForFocus(focusX, immediate) {
      if (focusX == null || fullscreenMode || usesWrappedPortraitLayout()) return;

      updateCameraMetrics();
      const nextX =
        cameraMode === 'smooth'
          ? getIdealCameraTarget(focusX)
          : getSnapCameraTarget(focusX);
      moveCameraTo(nextX, !!immediate);
    }

    function followLatestContent(immediate) {
      const focusX = getLatestContentFocusX();
      if (focusX == null) return;
      updateCameraForFocus(focusX, immediate);
    }

    function scheduleFollowTailScroll(smooth) {
      if (!followTail || fullscreenMode || usesWrappedPortraitLayout() || Date.now() < cameraSuspendUntil) return;

      clearScheduledFollowTail();
      followTailScheduleFrame = requestAnimationFrame(() => {
        followTailScheduleFrame = requestAnimationFrame(() => {
          followTailScheduleFrame = null;
          followLatestContent(!smooth);
        });
      });
    }

    function setFollowTail(enabled) {
      followTail = !!enabled;
      if (!followTail) {
        clearScheduledFollowTail();
        return;
      }

      if (usesWrappedPortraitLayout()) {
        clearScheduledFollowTail();
        moveCameraTo(0, true);
        return;
      }

      cameraSuspendUntil = 0;
      scheduleFollowTailScroll(false);
    }

    function setCameraMode(mode) {
      cameraMode = mode === 'snap' ? 'snap' : 'smooth';
      const stage = getStage();
      if (stage) {
        stage.dataset.cameraMode = cameraMode;
      }

      if (fullscreenMode || usesWrappedPortraitLayout()) {
        moveCameraTo(0, true);
        return cameraMode;
      }

      if (followTail) {
        scheduleFollowTailScroll(true);
      } else {
        moveCameraTo(cameraState.currentX, true);
      }

      return cameraMode;
    }

    function beginCameraDrag(clientX) {
      if (fullscreenMode || usesWrappedPortraitLayout() || cameraState.maxX <= 1) return;

      dragState.active = true;
      dragState.startX = clientX;
      dragState.startCameraX = cameraState.currentX;
      cancelCameraAnimation();
      clearScheduledFollowTail();
    }

    function updateCameraDrag(clientX) {
      if (!dragState.active || fullscreenMode || usesWrappedPortraitLayout() || cameraState.maxX <= 1) return false;

      const deltaX = clientX - dragState.startX;
      if (Math.abs(deltaX) < 3) return false;

      cameraSuspendUntil = Date.now() + 1800;
      suppressScoreTapUntil = Date.now() + 250;
      moveCameraTo(dragState.startCameraX - deltaX, true);
      return true;
    }

    function endCameraDrag() {
      if (!dragState.active) return;

      dragState.active = false;
      if (followTail) {
        window.setTimeout(() => {
          if (followTail && Date.now() >= cameraSuspendUntil) {
            scheduleFollowTailScroll(true);
          }
        }, 30);
      }
    }

    function bindCameraGestureHandlers() {
      const container = getContainer();
      if (!container || container.dataset.cameraBound === 'true') return;

      container.dataset.cameraBound = 'true';
      container.addEventListener('touchstart', function(evt) {
        if (!evt.touches || evt.touches.length !== 1) return;
        beginCameraDrag(evt.touches[0].clientX);
      }, { passive: true });
      container.addEventListener('touchmove', function(evt) {
        if (!evt.touches || evt.touches.length !== 1) return;
        const handled = updateCameraDrag(evt.touches[0].clientX);
        if (handled) {
          evt.preventDefault();
        }
      }, { passive: false });
      container.addEventListener('touchend', endCameraDrag, { passive: true });
      container.addEventListener('touchcancel', endCameraDrag, { passive: true });
    }

    function bindScoreScrollHandlers() {
      const container = getContainer();
      if (!container || container.dataset.scoreScrollBound === 'true') return;

      container.dataset.scoreScrollBound = 'true';
      container.addEventListener('touchstart', function(evt) {
        if (!evt.touches || evt.touches.length !== 1) return;
        beginScoreScrollGesture();
      }, { passive: true });
      container.addEventListener('touchmove', function(evt) {
        if (!evt.touches || evt.touches.length !== 1) return;
        markScoreScrollInteraction();
      }, { passive: true });
      container.addEventListener('touchend', function() {
        if (fullscreenMode || !usesWrappedPortraitLayout()) return;
        scheduleScoreScrollInactive();
      }, { passive: true });
      container.addEventListener('touchcancel', function() {
        if (fullscreenMode || !usesWrappedPortraitLayout()) return;
        scheduleScoreScrollInactive();
      }, { passive: true });
      container.addEventListener('scroll', function() {
        markScoreScrollInteraction();
      }, { passive: true });
    }

    function scrollCursorIntoView() {
      if (fullscreenMode || usesWrappedPortraitLayout() || Date.now() < cameraSuspendUntil) return;

      const focusX = getCursorFocusX();
      if (focusX == null) return;
      updateCameraForFocus(focusX, false);
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
      const stage = document.getElementById('osmd-stage');
      const osmdEl = document.getElementById('osmd');
      
      if (controls) {
        controls.className = enabled ? 'visible' : '';
      }
      
      // Toggle portrait/landscape mode classes
      if (container) {
        container.className = enabled ? 'landscape-mode' : 'portrait-mode';
      }
      if (stage) {
        stage.className = enabled ? 'landscape-mode' : 'portrait-mode';
      }
      if (osmdEl) {
        osmdEl.className = enabled ? 'landscape-mode' : 'portrait-mode';
      }
      updateStagePadding();
      cameraSuspendUntil = 0;
      setScoreScrollActive(false);
      cancelCameraAnimation();
      
      // Update OSMD rendering mode and re-render
      if (osmd && currentXml) {
        applyLayoutOptions();
        
        // Re-render with new settings
        setTimeout(async () => {
          try {
            await osmd.load(currentXml);
            await osmd.render();
            applyRenderedSvgScale();
            updateCameraMetrics();
            if (!enabled) {
              moveCameraTo(0, true);
              scheduleFollowTailScroll(true);
            } else {
              moveCameraTo(0, true);
            }
          } catch (e) {
            console.warn('Re-render error:', e);
          }
        }, 100);
      }
    }

    window.addEventListener('resize', function() {
      updateCameraMetrics();

      if (fullscreenMode) {
        moveCameraTo(0, true);
        return;
      }

      if (followTail) {
        scheduleFollowTailScroll(false);
      } else {
        moveCameraTo(cameraState.currentX, true);
      }
    });
    
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
    let cursorAnimationFrameId = null;
    let cursorPositions = [];
    let currentCursorIndex = 0;
    
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
    
    // ─── Expand Ornaments into Notes ───
    // Generates the individual notes that make up an ornament
    function expandOrnament(ornamentType, baseMidi, startTime, totalDuration, bpm) {
      const notes = [];
      const ornamentNoteDuration = Math.min(0.08, totalDuration / 8); // Fast ornament notes (~80ms or faster)
      
      switch (ornamentType) {
        case 'trill': {
          // Rapid alternation between base note and note above (usually whole step)
          const auxMidi = baseMidi + 2; // Whole step up (could be 1 for half step)
          const trillCount = Math.max(4, Math.floor(totalDuration / ornamentNoteDuration));
          const actualNoteDuration = totalDuration / trillCount;
          
          for (let i = 0; i < trillCount; i++) {
            const midi = i % 2 === 0 ? baseMidi : auxMidi;
            notes.push({
              time: startTime + i * actualNoteDuration,
              note: midiToNoteName(midi),
              duration: actualNoteDuration * 0.85,
              midi: midi,
              isOrnament: true
            });
          }
          break;
        }
        
        case 'mordent': {
          // Upper mordent: main -> upper -> main
          const auxMidi = baseMidi + 2;
          const mordentTime = Math.min(0.15, totalDuration * 0.3); // Mordent takes ~30% of note
          const mordentNoteDur = mordentTime / 3;
          const mainDuration = totalDuration - mordentTime;
          
          // Three quick notes at start
          notes.push({ time: startTime, note: midiToNoteName(baseMidi), duration: mordentNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          notes.push({ time: startTime + mordentNoteDur, note: midiToNoteName(auxMidi), duration: mordentNoteDur * 0.9, midi: auxMidi, isOrnament: true });
          notes.push({ time: startTime + mordentNoteDur * 2, note: midiToNoteName(baseMidi), duration: mordentNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          // Hold main note for rest
          notes.push({ time: startTime + mordentTime, note: midiToNoteName(baseMidi), duration: mainDuration * 0.9, midi: baseMidi });
          break;
        }
        
        case 'inverted-mordent': {
          // Lower mordent: main -> lower -> main
          const auxMidi = baseMidi - 2;
          const mordentTime = Math.min(0.15, totalDuration * 0.3);
          const mordentNoteDur = mordentTime / 3;
          const mainDuration = totalDuration - mordentTime;
          
          notes.push({ time: startTime, note: midiToNoteName(baseMidi), duration: mordentNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          notes.push({ time: startTime + mordentNoteDur, note: midiToNoteName(auxMidi), duration: mordentNoteDur * 0.9, midi: auxMidi, isOrnament: true });
          notes.push({ time: startTime + mordentNoteDur * 2, note: midiToNoteName(baseMidi), duration: mordentNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          notes.push({ time: startTime + mordentTime, note: midiToNoteName(baseMidi), duration: mainDuration * 0.9, midi: baseMidi });
          break;
        }
        
        case 'turn': {
          // Turn: upper -> main -> lower -> main
          const upperMidi = baseMidi + 2;
          const lowerMidi = baseMidi - 2;
          const turnTime = Math.min(0.2, totalDuration * 0.4);
          const turnNoteDur = turnTime / 4;
          const mainDuration = totalDuration - turnTime;
          
          notes.push({ time: startTime, note: midiToNoteName(upperMidi), duration: turnNoteDur * 0.9, midi: upperMidi, isOrnament: true });
          notes.push({ time: startTime + turnNoteDur, note: midiToNoteName(baseMidi), duration: turnNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          notes.push({ time: startTime + turnNoteDur * 2, note: midiToNoteName(lowerMidi), duration: turnNoteDur * 0.9, midi: lowerMidi, isOrnament: true });
          notes.push({ time: startTime + turnNoteDur * 3, note: midiToNoteName(baseMidi), duration: turnNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          // Hold main note for rest
          notes.push({ time: startTime + turnTime, note: midiToNoteName(baseMidi), duration: mainDuration * 0.9, midi: baseMidi });
          break;
        }
        
        case 'inverted-turn': {
          // Inverted turn: lower -> main -> upper -> main
          const upperMidi = baseMidi + 2;
          const lowerMidi = baseMidi - 2;
          const turnTime = Math.min(0.2, totalDuration * 0.4);
          const turnNoteDur = turnTime / 4;
          const mainDuration = totalDuration - turnTime;
          
          notes.push({ time: startTime, note: midiToNoteName(lowerMidi), duration: turnNoteDur * 0.9, midi: lowerMidi, isOrnament: true });
          notes.push({ time: startTime + turnNoteDur, note: midiToNoteName(baseMidi), duration: turnNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          notes.push({ time: startTime + turnNoteDur * 2, note: midiToNoteName(upperMidi), duration: turnNoteDur * 0.9, midi: upperMidi, isOrnament: true });
          notes.push({ time: startTime + turnNoteDur * 3, note: midiToNoteName(baseMidi), duration: turnNoteDur * 0.9, midi: baseMidi, isOrnament: true });
          notes.push({ time: startTime + turnTime, note: midiToNoteName(baseMidi), duration: mainDuration * 0.9, midi: baseMidi });
          break;
        }
        
        default:
          // No ornament, just return the plain note
          notes.push({
            time: startTime,
            note: midiToNoteName(baseMidi),
            duration: totalDuration * 0.9,
            midi: baseMidi
          });
      }
      
      return notes;
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

      // Tracks notes that have an open tie ("start"/"continue") so the tied
      // continuation can extend the original note's duration instead of being
      // re-articulated. Keyed by voice + midi; persists across measures since
      // ties commonly span barlines. Value is a reference to the pushed note.
      const tiedOpen = {};

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
            const isGrace = el.querySelector('grace') !== null;
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
                
                // Check for ornaments in notations
                const notationsEl = el.querySelector('notations');
                let ornamentType = null;
                
                if (notationsEl) {
                  const ornamentsEl = notationsEl.querySelector('ornaments');
                  if (ornamentsEl) {
                    if (ornamentsEl.querySelector('trill-mark')) ornamentType = 'trill';
                    else if (ornamentsEl.querySelector('inverted-mordent')) ornamentType = 'inverted-mordent';
                    else if (ornamentsEl.querySelector('mordent')) ornamentType = 'mordent';
                    else if (ornamentsEl.querySelector('inverted-turn')) ornamentType = 'inverted-turn';
                    else if (ornamentsEl.querySelector('turn')) ornamentType = 'turn';
                  }
                }
                
                // Detect tie state for this note. <tie> elements (not <tied>,
                // which is purely visual) drive playback sustain.
                let tieStart = false;
                let tieStop = false;
                const tieEls = el.querySelectorAll('tie');
                for (let ti = 0; ti < tieEls.length; ti++) {
                  const ty = tieEls[ti].getAttribute('type');
                  if (ty === 'start') tieStart = true;
                  else if (ty === 'stop') tieStop = true;
                }
                const tieKey = activeVoice + ':' + midi;

                // Handle grace notes - play very quickly before the main note
                if (isGrace) {
                  const graceNoteDuration = 0.08; // 80ms grace note
                  notes.push({
                    time: noteStartTime - graceNoteDuration,
                    note: noteName,
                    duration: graceNoteDuration * 0.9,
                    midi: midi,
                    isOrnament: true
                  });
                } else if (ornamentType) {
                  // Expand ornament into multiple notes
                  const expandedNotes = expandOrnament(ornamentType, midi, noteStartTime, durationSeconds, playbackBPM);
                  notes.push(...expandedNotes);
                } else if (tieStop && tiedOpen[tieKey]) {
                  // Continuation of a tie: extend the original note's duration
                  // rather than re-striking it. Use the full (un-shortened)
                  // duration so the sustain has no gap.
                  tiedOpen[tieKey].duration += durationSeconds;
                  if (!tieStart) {
                    // End of the tie chain — close it out.
                    delete tiedOpen[tieKey];
                  }
                } else {
                  // Regular note (or a tie-start). A tie-start sustains into the
                  // next segment, so don't apply the separation shortening; a
                  // standalone note gets the usual slight shortening.
                  const noteObj = {
                    time: noteStartTime,
                    note: noteName,
                    duration: tieStart
                      ? durationSeconds
                      : Math.max(0.1, durationSeconds * 0.9), // slightly shorter for separation
                    midi: midi
                  };
                  notes.push(noteObj);
                  if (tieStart) {
                    // Open a tie so following continuations extend this note.
                    tiedOpen[tieKey] = noteObj;
                  }
                }
                
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
            
            // Advance time only for non-chord, non-grace notes
            if (!isChord && !isGrace) {
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
      updateCameraMetrics();
      moveCameraTo(0, true);
      cameraSuspendUntil = 0;
      
      // Schedule all notes
      const totalDuration = notes[notes.length - 1].time + notes[notes.length - 1].duration + 0.5;
      
      notes.forEach((noteEvent, index) => {
        const eventId = Tone.Transport.schedule((time) => {
          sampler.triggerAttackRelease(noteEvent.note, noteEvent.duration, time);
        }, noteEvent.time);
        scheduledEvents.push(eventId);
      });
      
      // Schedule cursor advances based on OSMD's internal structure
      // Use continuous position tracking instead of discrete scheduling for responsiveness
      if (osmd && osmd.cursor && osmd.Sheet) {
        try {
          // Get all the timestamp positions from the cursor iterator
          osmd.cursor.reset();
          cursorPositions = [];
          let safetyCounter = 0;
          const maxIterations = 10000; // Prevent infinite loop
          
          while (!osmd.cursor.iterator.EndReached && safetyCounter < maxIterations) {
            const timestamp = osmd.cursor.iterator.CurrentSourceTimestamp;
            if (timestamp) {
              // Convert OSMD timestamp (in fractions) to seconds
              // OSMD uses quarter notes as the base, so timestamp.RealValue * 4 = beats
              const beats = timestamp.RealValue * 4;
              const timeInSeconds = beats * (60.0 / playbackBPM);
              cursorPositions.push(timeInSeconds);
            }
            osmd.cursor.next();
            safetyCounter++;
          }
          
          // Reset cursor to start
          osmd.cursor.reset();
          osmd.cursor.show();
          currentCursorIndex = 0;
          
          // Use requestAnimationFrame for smooth cursor tracking
          function updateCursor() {
            if (!isPlaying) {
              return; // Stop loop when not playing
            }
            
            if (!isPaused) {
              const currentTime = Tone.Transport.seconds;
              
              // Advance cursor while we're past the next position (with small lookahead)
              while (currentCursorIndex < cursorPositions.length - 1 && 
                     currentTime >= cursorPositions[currentCursorIndex + 1] - 0.015) {
                if (osmd && osmd.cursor) {
                  osmd.cursor.next();
                  currentCursorIndex++;
                  scrollCursorIntoView();
                }
              }
            }
            
            // Continue animation loop while playing
            cursorAnimationFrameId = requestAnimationFrame(updateCursor);
          }
          
          // Start the cursor tracking loop
          cursorAnimationFrameId = requestAnimationFrame(updateCursor);
          
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
      // Cancel cursor animation
      if (cursorAnimationFrameId) {
        cancelAnimationFrame(cursorAnimationFrameId);
        cursorAnimationFrameId = null;
      }
      currentCursorIndex = 0;
      
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

    function buildDebugSnapshot(reason, requestId) {
      const container = document.getElementById('osmd-container');
      const stage = document.getElementById('osmd');

      return {
        reason: reason,
        requestId: requestId == null ? null : requestId,
        initCompleted: !!osmd,
        initInFlight: false,
        dependenciesReady: !!window.opensheetmusicdisplay && !!window.Tone,
        hasOsmd: !!osmd,
        currentXmlLength: currentXml ? currentXml.length : 0,
        renderedMeasureCount: osmd && osmd.Sheet && osmd.Sheet.Measures ? osmd.Sheet.Measures.length : 0,
        stageChildElementCount: stage ? stage.children.length : 0,
        stageInnerHtmlLength: stage ? stage.innerHTML.length : 0,
        stageSvgCount: stage ? stage.querySelectorAll('svg').length : 0,
        stageClientWidth: stage ? stage.clientWidth : 0,
        stageClientHeight: stage ? stage.clientHeight : 0,
        containerClientWidth: container ? container.clientWidth : 0,
        containerClientHeight: container ? container.clientHeight : 0,
        containerScrollWidth: container ? container.scrollWidth : 0,
        containerScrollHeight: container ? container.scrollHeight : 0,
        containerClassName: container ? container.className : '',
        cameraMode: cameraMode,
        cameraX: cameraState.currentX,
        cameraTargetX: cameraState.targetX,
        cameraMaxX: cameraState.maxX,
        cameraViewportWidth: cameraState.viewportWidth,
        cameraContentWidth: cameraState.contentWidth,
        cameraMeasureCount: cameraState.measureWidths.length,
        cameraAverageMeasureWidth: cameraState.averageMeasureWidth,
        cameraPaddingLeft: cameraState.paddingLeft,
        cameraPaddingRight: cameraState.paddingRight,
        osmdScriptStatus: null,
        marker: 'inline-html',
        timestamp: Date.now()
      };
    }

    function postDebugSnapshot(reason, requestId) {
      post({
        type: 'debugState',
        snapshot: buildDebugSnapshot(reason, requestId)
      });
    }

    async function init(options) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmd", Object.assign({
        backend: "svg",
        autoResize: false, // We'll control sizing manually
        drawTitle: false,
        drawPartNames: false,
        // Portrait preview should wrap instead of using horizontal camera scrolling
        renderSingleHorizontalStaffline: false
      }, options||{}));

      applyLayoutOptions();

      bindCameraGestureHandlers();
      bindScoreScrollHandlers();
      updateCameraMetrics();
      setCameraMode(options && options.cameraMode ? options.cameraMode : cameraMode);
      
      // Pre-load the sampler in the background
      initSampler().catch(e => console.warn('Sampler init failed:', e));
      
      post({ type: "ready" });
      postDebugSnapshot('init-ready', null);
    }

    async function renderXml(xml, requestId) {
      try {
        currentXml = xml; // Store for playback
        applyLayoutOptions();
        await osmd.load(xml);
        await osmd.render();
        applyRenderedSvgScale();
        updateCameraMetrics();
        moveCameraTo(0, true);
        scheduleFollowTailScroll(true);
        post({ type: "rendered", requestId: requestId == null ? null : requestId, measures: osmd.Sheet?.Measures?.length || 0 });
        postDebugSnapshot('rendered', requestId);
      } catch (e) {
        post({ type: "error", requestId: requestId == null ? null : requestId, error: String(e) });
      }
    }

    function toggleCursor(show){
      if (!osmd) return;
      if (show) osmd.cursor.show(); else osmd.cursor.hide();
    }
    function cursorNext(){ osmd?.cursor?.next(); }
    function cursorReset(){ osmd?.cursor?.reset(); }

    window.__OSMD_INIT = function(options) { return init(options); };
    window.__OSMD_RENDER_XML = function(xml, requestId) { return renderXml(xml, requestId); };
    window.__OSMD_SET_FOLLOW_TAIL = function(enabled) { return setFollowTail(enabled); };
    window.__OSMD_TOGGLE_CURSOR = function(show) { return toggleCursor(show); };
    window.__OSMD_CURSOR_NEXT = function() { return cursorNext(); };
    window.__OSMD_CURSOR_RESET = function() { return cursorReset(); };
    window.__OSMD_PLAY = function(bpm) { return startPlayback(bpm); };
    window.__OSMD_PAUSE = function() { return pausePlayback(); };
    window.__OSMD_STOP = function() { return stopPlayback(); };
    window.__OSMD_SET_BPM = function(bpm) { return setPlaybackBPM(bpm); };
    window.__OSMD_SET_FULLSCREEN = function(enabled) { return setFullscreenMode(enabled); };
    window.__OSMD_SET_CAMERA_MODE = function(mode) { return setCameraMode(mode); };
    window.__OSMD_DEBUG_SNAPSHOT = function(reason, requestId) { return postDebugSnapshot(reason, requestId); };

    function onMessage(e){
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "init") return init(msg.options);
        if (msg.type === "renderXml") return renderXml(msg.xml, msg.requestId);
        if (msg.type === "setFollowTail") return setFollowTail(msg.enabled);
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
        if (msg.type === "setCameraMode") return setCameraMode(msg.mode);
        if (msg.type === "debugSnapshot") return postDebugSnapshot(msg.reason, msg.requestId);
      } catch {}
    }
    window.addEventListener("message", onMessage);
    document.addEventListener("message", onMessage);
    // forward clicks from the webview to the React Native host so it can enter fullscreen
    // Only send click when NOT in fullscreen mode (exit is handled by the exit button)
    document.addEventListener('click', function(e){
      // Don't trigger fullscreen toggle if clicking on the controls
      if (e.target.closest('#fullscreen-controls')) return;
      if (Date.now() < suppressScoreTapUntil) return;
      // Only send click to enter fullscreen, not to exit
      if (!fullscreenMode) {
        post({ type: 'webview-click' });
      }
    });

    // ─── Dependency Bootstrap ───
    // The two <script src="..."> tags above load OSMD and Tone from jsdelivr.
    // Android WebView occasionally fails to load CDN scripts cleanly, so we
    // retry from unpkg before giving up. init() is gated until OSMD is live.
    function loadFallbackScript(src, isReady) {
      return new Promise(function(resolve, reject){
        if (isReady()) { resolve(); return; }
        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.crossOrigin = 'anonymous';
        script.onload = function(){ resolve(); };
        script.onerror = function(){ reject(new Error('fallback script failed: ' + src)); };
        document.head.appendChild(script);
      });
    }

    async function ensureDependencies() {
      const osmdReady = function(){ return !!(window.opensheetmusicdisplay && window.opensheetmusicdisplay.OpenSheetMusicDisplay); };
      const toneReady = function(){ return !!window.Tone; };

      // Give the inline <script> tags a brief window to finish before we intervene.
      const deadline = Date.now() + 2500;
      while (Date.now() < deadline && (!osmdReady() || !toneReady())) {
        await new Promise(function(r){ setTimeout(r, 50); });
      }

      if (!osmdReady()) {
        try {
          await loadFallbackScript('https://unpkg.com/opensheetmusicdisplay@1.9.2/build/opensheetmusicdisplay.min.js', osmdReady);
        } catch (e) {
          post({ type: 'error', error: 'OSMD CDN unreachable: ' + String(e) });
          return false;
        }
      }
      if (!toneReady()) {
        try {
          await loadFallbackScript('https://unpkg.com/tone@14.7.77/build/Tone.js', toneReady);
        } catch (e) {
          // Tone is only needed for playback — log but allow rendering to proceed.
          console.warn('Tone CDN unreachable:', e);
        }
      }
      return osmdReady();
    }

    async function bootstrap() {
      const ok = await ensureDependencies();
      if (!ok) return;
      try {
        await init();
      } catch (e) {
        post({ type: 'error', error: 'init failed: ' + String(e) });
      }
    }

    bootstrap();
  <\/script>
</body>
</html>
`;
