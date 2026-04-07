"""
Live/Streaming Rhythm Detection Module

Provides low-latency rhythm quantization for real-time transcription:
1. IncrementalTempoTracker - builds BPM estimate incrementally from onset times
2. Coarse quantization - fast, per-note rhythm assignment without lookahead
3. Deferred refinement - periodically re-quantizes recent notes with more context

Architecture:
- Stage 1 (real-time, <50ms): Onset + pitch + coarse rhythm -> immediate display
- Stage 2 (deferred, ~1s behind): Re-quantize with tempo/context -> update display
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
NOTE_TYPE_BEATS = {
    'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 
    'eighth': 0.5, '16th': 0.25, '32nd': 0.125
}
BEATS_TO_TYPE = {v: k for k, v in NOTE_TYPE_BEATS.items()}

# Standard note durations in beats (no triplets for coarse pass)
COARSE_CANDIDATES = [0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Refinement adds triplets and dotted variations
REFINED_CANDIDATES = [
    (0.125, '32nd', False, False),
    (0.1667, '32nd', False, True),   # 32nd triplet
    (0.1875, '32nd', True, False),   # Dotted 32nd
    (0.25, '16th', False, False),
    (0.333, '16th', False, True),    # 16th triplet
    (0.375, '16th', True, False),    # Dotted 16th
    (0.5, 'eighth', False, False),
    (0.667, 'eighth', False, True),  # Eighth triplet
    (0.75, 'eighth', True, False),   # Dotted eighth
    (1.0, 'quarter', False, False),
    (1.333, 'quarter', False, True), # Quarter triplet
    (1.5, 'quarter', True, False),   # Dotted quarter
    (2.0, 'half', False, False),
    (2.667, 'half', False, True),    # Half triplet
    (3.0, 'half', True, False),      # Dotted half
    (4.0, 'whole', False, False),
    (6.0, 'whole', True, False),     # Dotted whole
]


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Tempo Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IncrementalTempoTracker:
    """
    Builds BPM estimate incrementally from onset times.
    
    Uses a histogram-based approach:
    - Collects inter-onset intervals (IOIs)
    - Finds dominant interval clusters
    - Maps to likely beat duration
    
    Attributes:
        initial_bpm: Starting BPM estimate (default 120)
        min_bpm: Minimum allowed BPM
        max_bpm: Maximum allowed BPM
        window_iois: How many IOIs to keep in the sliding window
    """
    initial_bpm: float = 120.0
    min_bpm: float = 40.0
    max_bpm: float = 240.0
    window_iois: int = 32
    
    # Internal state
    ioi_buffer: deque = field(default_factory=lambda: deque(maxlen=32))
    last_onset: float = -1.0
    current_bpm: float = 120.0
    confidence: float = 0.0
    _update_count: int = 0
    
    def __post_init__(self):
        self.ioi_buffer = deque(maxlen=self.window_iois)
        self.current_bpm = self.initial_bpm
    
    def reset(self):
        """Reset tracker state."""
        self.ioi_buffer.clear()
        self.last_onset = -1.0
        self.current_bpm = self.initial_bpm
        self.confidence = 0.0
        self._update_count = 0
    
    def add_onset(self, time_seconds: float) -> Tuple[float, float]:
        """
        Add a new onset time and update BPM estimate.
        
        Args:
            time_seconds: Onset time in seconds
            
        Returns:
            Tuple of (current_bpm, confidence)
        """
        if self.last_onset >= 0:
            ioi = time_seconds - self.last_onset
            # Filter out very short or very long intervals
            min_ioi = 60.0 / self.max_bpm / 4  # 16th note at max tempo
            max_ioi = 60.0 / self.min_bpm * 4   # Whole note at min tempo
            
            if min_ioi <= ioi <= max_ioi:
                self.ioi_buffer.append(ioi)
                self._update_count += 1
                
                # Update BPM every few onsets (avoid thrashing)
                if len(self.ioi_buffer) >= 4 and self._update_count % 2 == 0:
                    self._update_bpm()
        
        self.last_onset = time_seconds
        return (self.current_bpm, self.confidence)
    
    def _update_bpm(self):
        """Update BPM estimate from IOI histogram with octave disambiguation."""
        if len(self.ioi_buffer) < 4:
            return

        iois = np.array(self.ioi_buffer)

        # Find clusters of similar IOIs using a simple histogram approach
        # Convert IOIs to potential beat durations (could be 1/4, 1/2, 1, 2, 4 beats)

        candidates = []
        for ioi in iois:
            # Try each possible beat mapping
            for divisor in [0.25, 0.5, 1.0, 2.0, 4.0]:
                beat_dur = ioi / divisor
                bpm = 60.0 / beat_dur
                if self.min_bpm <= bpm <= self.max_bpm:
                    candidates.append(bpm)

        if not candidates:
            return

        candidates = np.array(candidates)

        # Finer histogram (80 bins instead of 40) for better resolution
        hist, bin_edges = np.histogram(candidates, bins=80,
                                        range=(self.min_bpm, self.max_bpm))
        peak_idx = np.argmax(hist)
        peak_bpm = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2

        # Confidence based on how concentrated the histogram is
        total = hist.sum()
        peak_count = hist[peak_idx]
        # Include adjacent bins for tolerance
        if peak_idx > 0:
            peak_count += hist[peak_idx - 1]
        if peak_idx < len(hist) - 1:
            peak_count += hist[peak_idx + 1]

        self.confidence = min(1.0, peak_count / max(total, 1) * 1.5)

        # Octave disambiguation: check if 0.5x or 2x gives better IOI alignment
        best_bpm = peak_bpm
        best_score = self._alignment_score(iois, 60.0 / peak_bpm)
        for mult in [0.5, 2.0]:
            alt_bpm = peak_bpm * mult
            if self.min_bpm <= alt_bpm <= self.max_bpm:
                alt_score = self._alignment_score(iois, 60.0 / alt_bpm)
                if alt_score > best_score * 1.05:
                    best_bpm = alt_bpm
                    best_score = alt_score

        # Smooth BPM transitions (don't jump suddenly)
        alpha = 0.3 if self.confidence > 0.5 else 0.1
        self.current_bpm = self.current_bpm * (1 - alpha) + best_bpm * alpha

        # Snap to common tempos if very close
        common_tempos = [60, 72, 80, 90, 100, 108, 120, 132, 140, 160, 180, 200]
        for ct in common_tempos:
            if abs(self.current_bpm - ct) < 3:
                self.current_bpm = ct
                break

    @staticmethod
    def _alignment_score(iois, beat_period):
        """Score how well IOIs align to integer/half-integer multiples of beat."""
        import math
        score = 0.0
        for ioi in iois:
            ratio = ioi / beat_period
            nearest = round(ratio * 2) / 2
            if nearest < 0.25:
                continue
            dist = abs(ratio - nearest)
            score += math.exp(-(dist ** 2) / (2 * 0.08 ** 2))
        return score / len(iois)
    
    def get_beat_duration(self) -> float:
        """Get current beat duration in seconds."""
        return 60.0 / self.current_bpm


# ─────────────────────────────────────────────────────────────────────────────
# Coarse Quantization (Stage 1 - Real-time)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_coarse(note: Dict, bpm: float, prev_notes: Optional[List[Dict]] = None) -> Dict:
    """
    Fast, single-note quantization for real-time display.
    
    Uses only past context (no lookahead). Snaps to nearest standard
    note value without triplet detection.
    
    Args:
        note: Dict with 'duration_seconds' (or 'time_seconds' + next onset)
        bpm: Current tempo estimate
        prev_notes: Optional list of recent notes for context
        
    Returns:
        Dict with added rhythm fields: note_value, note_divisions, dotted, is_triplet
    """
    beat_dur = 60.0 / bpm
    duration = note.get('duration_seconds', 0.5)
    
    # If no explicit duration, use a default quarter note
    if duration <= 0:
        duration = beat_dur
    
    beats = duration / beat_dur
    
    # Find closest standard duration
    best_beats = min(COARSE_CANDIDATES, key=lambda c: abs(beats - c))
    
    # Determine note type
    is_dotted = best_beats in [1.5, 3.0, 0.375, 0.1875, 0.75]
    base_beats = best_beats / 1.5 if is_dotted else best_beats
    
    note_type = 'quarter'
    for nt, nb in NOTE_TYPE_BEATS.items():
        if abs(base_beats - nb) < 0.01:
            note_type = nt
            break
    
    # Add rhythm fields
    note['note_value'] = note_type
    note['note_divisions'] = best_beats
    note['dotted'] = is_dotted
    note['is_triplet'] = False  # Never triplet in coarse pass
    note['triplet'] = False
    note['quantization_method'] = 'coarse_live'
    note['quantization_confidence'] = max(0.5, 1.0 - abs(beats - best_beats) / best_beats)
    note['raw_beats'] = beats
    
    return note


def quantize_batch_coarse(notes: List[Dict], bpm: float) -> List[Dict]:
    """
    Batch coarse quantization for multiple notes.
    
    Args:
        notes: List of note dicts
        bpm: Current tempo estimate
        
    Returns:
        Notes with coarse rhythm assignments
    """
    for i, note in enumerate(notes):
        prev_notes = notes[max(0, i-4):i] if i > 0 else None
        quantize_coarse(note, bpm, prev_notes)
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Deferred Refinement (Stage 2 - Background)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeferredRefinementState:
    """
    Tracks state for deferred rhythm refinement.
    
    Maintains a buffer of notes and periodically triggers refinement
    of notes that are old enough (past the lookahead window).
    """
    # Configuration
    refinement_delay_sec: float = 1.0  # Wait this long before refining a note
    min_notes_for_refinement: int = 4   # Need at least this many notes
    lookahead_notes: int = 4            # Use this many future notes for context
    
    # State
    pending_notes: List[Dict] = field(default_factory=list)
    refined_notes: List[Dict] = field(default_factory=list)
    last_refinement_time: float = 0.0
    _refinement_version: int = 0
    
    def __post_init__(self):
        self.pending_notes = []
        self.refined_notes = []
    
    def add_note(self, note: Dict, current_time: float) -> None:
        """Add a new note to the pending buffer."""
        note['_arrival_time'] = current_time
        note['_refined'] = False
        self.pending_notes.append(note)
    
    def add_notes(self, notes: List[Dict], current_time: float) -> None:
        """Add multiple notes to the pending buffer."""
        for note in notes:
            self.add_note(note, current_time)
    
    def check_refinement(self, current_time: float, bpm: float) -> Optional[List[Dict]]:
        """
        Check if any notes are ready for refinement.
        
        Returns:
            List of newly refined notes if any, None otherwise
        """
        if len(self.pending_notes) < self.min_notes_for_refinement:
            return None
        
        # Find notes that are old enough to refine
        cutoff_time = current_time - self.refinement_delay_sec
        ready_indices = []
        
        for i, note in enumerate(self.pending_notes):
            arrival = note.get('_arrival_time', 0)
            if arrival <= cutoff_time and not note.get('_refined', False):
                ready_indices.append(i)
        
        if not ready_indices:
            return None
        
        # Refine notes with surrounding context
        newly_refined = []
        for idx in ready_indices:
            note = self.pending_notes[idx]
            
            # Get context window
            start_ctx = max(0, idx - 4)
            end_ctx = min(len(self.pending_notes), idx + self.lookahead_notes + 1)
            context = self.pending_notes[start_ctx:end_ctx]
            note_in_context = idx - start_ctx
            
            # Refine with context
            refined = refine_note_with_context(note, context, note_in_context, bpm)
            refined['_refined'] = True
            newly_refined.append(refined)
        
        if newly_refined:
            self._refinement_version += 1
            self.last_refinement_time = current_time
            self.refined_notes.extend(newly_refined)
        
        return newly_refined if newly_refined else None
    
    def get_refinement_version(self) -> int:
        """Get current refinement version for cache invalidation."""
        return self._refinement_version
    
    def get_all_notes(self) -> List[Dict]:
        """Get all notes (both pending and refined)."""
        # Merge refined into pending
        result = []
        refined_times = {n.get('time_seconds', 0) for n in self.refined_notes}
        
        for note in self.pending_notes:
            t = note.get('time_seconds', 0)
            if t in refined_times:
                # Find the refined version
                for rn in self.refined_notes:
                    if abs(rn.get('time_seconds', 0) - t) < 0.001:
                        result.append(rn)
                        break
            else:
                result.append(note)
        
        return sorted(result, key=lambda n: n.get('time_seconds', 0))
    
    def clear(self):
        """Clear all state."""
        self.pending_notes.clear()
        self.refined_notes.clear()
        self._refinement_version = 0


def refine_note_with_context(note: Dict, context: List[Dict], 
                              note_idx: int, bpm: float) -> Dict:
    """
    Refine a single note's rhythm using surrounding context.
    
    This is the "Stage 2" refinement that runs ~1s behind real-time.
    It can detect triplets, improve quantization, and smooth gaps.
    
    Args:
        note: The note to refine
        context: List of surrounding notes
        note_idx: Index of note within context
        bpm: Current tempo estimate
        
    Returns:
        Refined note dict
    """
    beat_dur = 60.0 / bpm
    duration = note.get('duration_seconds', note.get('raw_beats', 1.0) * beat_dur)
    beats = duration / beat_dur
    
    # Check for triplet patterns in context
    is_likely_triplet = _detect_triplet_in_context(context, note_idx, bpm)
    
    # Find best match from refined candidates
    best_match = None
    best_distance = float('inf')
    
    for (cand_beats, cand_type, cand_dotted, cand_triplet) in REFINED_CANDIDATES:
        # Skip triplets if not detected in context
        if cand_triplet and not is_likely_triplet:
            continue
        
        # Use logarithmic distance for better musical perception
        if beats > 0 and cand_beats > 0:
            ratio = beats / cand_beats
            log_dist = abs(math.log2(ratio))
        else:
            log_dist = abs(beats - cand_beats)
        
        if log_dist < best_distance:
            best_distance = log_dist
            best_match = (cand_beats, cand_type, cand_dotted, cand_triplet)
    
    if best_match:
        best_beats, note_type, is_dotted, is_triplet = best_match
        note['note_value'] = note_type
        note['note_divisions'] = best_beats
        note['dotted'] = is_dotted
        note['is_triplet'] = is_triplet
        note['triplet'] = is_triplet
        note['quantization_method'] = 'refined_deferred'
        note['quantization_confidence'] = max(0.6, 1.0 - best_distance)
    
    return note


def _detect_triplet_in_context(context: List[Dict], note_idx: int, bpm: float) -> bool:
    """
    Detect if the note at note_idx is likely part of a triplet.
    
    Looks for groups of 3 similarly-timed notes that together span
    a simple beat duration (1, 2, or 4 beats).
    """
    if len(context) < 3:
        return False
    
    beat_dur = 60.0 / bpm
    
    # Check triplet groups: [i-2, i-1, i], [i-1, i, i+1], [i, i+1, i+2]
    for start in range(max(0, note_idx - 2), min(len(context) - 2, note_idx + 1)):
        if start + 2 >= len(context):
            continue
        
        group = context[start:start + 3]
        durations = [n.get('duration_seconds', 0.5) for n in group]
        
        # Check if all durations are similar
        if max(durations) > 0 and min(durations) / max(durations) > 0.7:
            # Check if total spans a simple beat
            total_beats = sum(durations) / beat_dur
            for target in [1.0, 2.0, 4.0]:
                if abs(total_beats - target) / target < 0.15:
                    return True
    
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Live Transcription Session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass  
class LiveTranscriptionSession:
    """
    Manages state for a live transcription session.
    
    Coordinates tempo tracking, coarse quantization, and deferred refinement.
    Tracks when the frontend needs to be notified of updates.
    """
    session_id: str
    
    # Sub-components
    tempo_tracker: IncrementalTempoTracker = field(
        default_factory=IncrementalTempoTracker
    )
    refinement_state: DeferredRefinementState = field(
        default_factory=DeferredRefinementState
    )
    
    # Session state
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    _last_notified_version: int = 0
    
    # Results
    coarse_notes: List[Dict] = field(default_factory=list)
    coarse_chords: List[Dict] = field(default_factory=list)
    
    def process_notes(self, notes: List[Dict], chords: List[Dict] = None) -> Dict:
        """
        Process newly detected notes and chords.
        
        Returns:
            Dict with:
                - coarse_notes: Immediately quantized notes
                - bpm: Current tempo estimate
                - needs_refresh: Whether frontend should reload score
                - refined_notes: If any refinements are ready
        """
        current_time = time.time()
        self.last_update = current_time
        
        # Update tempo from new onsets
        for note in notes:
            onset = note.get('time_seconds', 0)
            self.tempo_tracker.add_onset(onset)
        
        bpm = self.tempo_tracker.current_bpm
        
        # Coarse quantization (immediate)
        quantize_batch_coarse(notes, bpm)
        self.coarse_notes.extend(notes)
        
        if chords:
            quantize_batch_coarse(chords, bpm)
            self.coarse_chords.extend(chords)
        
        # Add to refinement buffer
        self.refinement_state.add_notes(notes, current_time)
        
        # Check for ready refinements
        refined = self.refinement_state.check_refinement(current_time, bpm)
        
        # Determine if frontend needs refresh
        needs_refresh = False
        current_version = self.refinement_state.get_refinement_version()
        if current_version > self._last_notified_version:
            needs_refresh = True
            self._last_notified_version = current_version
        
        return {
            'coarse_notes': notes,
            'coarse_chords': chords or [],
            'bpm': bpm,
            'bpm_confidence': self.tempo_tracker.confidence,
            'needs_refresh': needs_refresh,
            'refined_notes': refined,
            'refinement_version': current_version,
        }
    
    def get_all_notes(self) -> List[Dict]:
        """Get all notes with best available quantization."""
        return self.refinement_state.get_all_notes()
    
    def get_current_bpm(self) -> Tuple[float, float]:
        """Get current BPM and confidence."""
        return (self.tempo_tracker.current_bpm, self.tempo_tracker.confidence)
    
    def force_refinement(self) -> Optional[List[Dict]]:
        """Force refinement of all pending notes (e.g., on recording stop)."""
        bpm = self.tempo_tracker.current_bpm
        current_time = time.time()
        
        # Temporarily reduce delay to refine everything
        old_delay = self.refinement_state.refinement_delay_sec
        self.refinement_state.refinement_delay_sec = 0
        
        refined = self.refinement_state.check_refinement(current_time + 10, bpm)
        
        self.refinement_state.refinement_delay_sec = old_delay
        return refined
    
    def reset(self):
        """Reset session state."""
        self.tempo_tracker.reset()
        self.refinement_state.clear()
        self.coarse_notes.clear()
        self.coarse_chords.clear()
        self._last_notified_version = 0


# ─────────────────────────────────────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────────────────────────────────────

_live_sessions: Dict[str, LiveTranscriptionSession] = {}


def get_or_create_session(session_id: str) -> LiveTranscriptionSession:
    """Get existing session or create a new one."""
    if session_id not in _live_sessions:
        _live_sessions[session_id] = LiveTranscriptionSession(session_id=session_id)
    return _live_sessions[session_id]


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    if session_id in _live_sessions:
        del _live_sessions[session_id]
        return True
    return False


def cleanup_stale_sessions(max_age_seconds: float = 3600.0) -> int:
    """Remove sessions older than max_age_seconds."""
    current = time.time()
    stale = [
        sid for sid, sess in _live_sessions.items()
        if current - sess.last_update > max_age_seconds
    ]
    for sid in stale:
        del _live_sessions[sid]
    return len(stale)
