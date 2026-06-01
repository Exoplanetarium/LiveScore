"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateMusicXML = generateMusicXML;
exports.default = PianoSheetMusic;
const ScreenOrientation = __importStar(require("expo-screen-orientation"));
const react_1 = __importStar(require("react"));
const react_native_1 = require("react-native");
const react_native_webview_1 = require("react-native-webview");
const ThemedText_1 = require("./ThemedText");
const osmdHTML_1 = require("./osmdHTML");
function midiToStepOctaveForKey(midi, fifths = 0) {
    const map = [
        ["C", 0],
        ["C", 1],
        ["D", 0],
        ["D", 1],
        ["E", 0],
        ["F", 0],
        ["F", 1],
        ["G", 0],
        ["G", 1],
        ["A", 0],
        ["A", 1],
        ["B", 0],
    ];
    const flatOrder = [
        [10, "B", -1],
        [3, "E", -1],
        [8, "A", -1],
        [1, "D", -1],
        [6, "G", -1],
        [11, "C", -1],
        [4, "F", -1],
    ];
    const sharpExtra = [
        [5, "E", 1],
        [0, "B", 1],
    ];
    if (fifths < 0) {
        const n = Math.min(Math.abs(fifths), 7);
        for (let i = 0; i < n; i++) {
            const [pc, step, alter] = flatOrder[i];
            map[pc] = [step, alter];
        }
    }
    else if (fifths > 5) {
        const extra = Math.min(fifths - 5, 2);
        for (let i = 0; i < extra; i++) {
            const [pc, step, alter] = sharpExtra[i];
            map[pc] = [step, alter];
        }
    }
    const pc = midi % 12;
    const [step, alter] = map[pc];
    let octave = Math.floor(midi / 12) - 1;
    if (step === "C" && alter === -1)
        octave += 1;
    if (step === "B" && alter === 1)
        octave -= 1;
    return { step, alter, octave };
}
function generateMeasureXmls(notes, chords, timeSignature = "4/4", bpm = 120, fifths = 0) {
    var _a, _b;
    const measures = [];
    const getBeatsPerMeasure = (ts) => {
        switch (ts) {
            case "4/4":
                return 4;
            case "3/4":
                return 3;
            case "6/8":
                return 6 * 0.5;
            default:
                return 4;
        }
    };
    const BEATS_PER_MEASURE = getBeatsPerMeasure(timeSignature);
    const getAuthoritativeStartBeat = (event) => {
        var _a;
        if (typeof event.start_beat === "number" &&
            Number.isFinite(event.start_beat)) {
            return Math.round(event.start_beat * 24) / 24;
        }
        return Math.round((((_a = event.time_seconds) !== null && _a !== void 0 ? _a : 0) / 60) * bpm * 24) / 24;
    };
    const getNoteBeats = (noteType, dotted, triplet) => {
        let beats = 1;
        switch (noteType) {
            case "whole":
                beats = 4;
                break;
            case "half":
                beats = 2;
                break;
            case "quarter":
                beats = 1;
                break;
            case "eighth":
                beats = 0.5;
                break;
            case "16th":
                beats = 0.25;
                break;
            case "32nd":
                beats = 0.125;
                break;
            default:
                beats = 1;
                break;
        }
        if (dotted)
            beats *= 1.5;
        if (triplet)
            beats = Math.round(beats * (2 / 3) * 24) / 24;
        return beats;
    };
    const getNoteDuration = (noteType, dotted, triplet) => {
        let duration = 24;
        switch (noteType) {
            case "whole":
                duration = 96;
                break;
            case "half":
                duration = 48;
                break;
            case "quarter":
                duration = 24;
                break;
            case "eighth":
                duration = 12;
                break;
            case "16th":
                duration = 6;
                break;
            case "32nd":
                duration = 3;
                break;
            default:
                duration = 24;
                break;
        }
        if (dotted)
            duration = duration * 1.5;
        if (duration !== Math.floor(duration))
            duration = Math.floor(duration);
        if (triplet)
            duration = (duration * 2) / 3;
        return duration;
    };
    const getFallbackDurationSpec = (noteType, dotted, triplet) => {
        const resolvedNoteType = noteType || "quarter";
        const resolvedDotted = dotted || false;
        const resolvedTriplet = triplet || false;
        return {
            beats: getNoteBeats(resolvedNoteType, resolvedDotted, resolvedTriplet),
            duration: getNoteDuration(resolvedNoteType, resolvedDotted, resolvedTriplet),
            noteType: resolvedNoteType,
            dotted: resolvedDotted,
            triplet: resolvedTriplet,
        };
    };
    const getDurationSpec = (preferredBeats, noteType, dotted, triplet) => {
        const fallback = getFallbackDurationSpec(noteType, dotted, triplet);
        if (preferredBeats === undefined ||
            !Number.isFinite(preferredBeats) ||
            preferredBeats <= 0) {
            return fallback;
        }
        const canonicalSpecs = [
            {
                beats: 4,
                duration: 96,
                noteType: "whole",
                dotted: false,
                triplet: false,
            },
            {
                beats: 3,
                duration: 72,
                noteType: "half",
                dotted: true,
                triplet: false,
            },
            {
                beats: 8 / 3,
                duration: 64,
                noteType: "whole",
                dotted: false,
                triplet: true,
            },
            {
                beats: 2,
                duration: 48,
                noteType: "half",
                dotted: false,
                triplet: false,
            },
            {
                beats: 1.5,
                duration: 36,
                noteType: "quarter",
                dotted: true,
                triplet: false,
            },
            {
                beats: 4 / 3,
                duration: 32,
                noteType: "half",
                dotted: false,
                triplet: true,
            },
            {
                beats: 1,
                duration: 24,
                noteType: "quarter",
                dotted: false,
                triplet: false,
            },
            {
                beats: 0.75,
                duration: 18,
                noteType: "eighth",
                dotted: true,
                triplet: false,
            },
            {
                beats: 2 / 3,
                duration: 16,
                noteType: "quarter",
                dotted: false,
                triplet: true,
            },
            {
                beats: 0.5,
                duration: 12,
                noteType: "eighth",
                dotted: false,
                triplet: false,
            },
            {
                beats: 0.375,
                duration: 9,
                noteType: "16th",
                dotted: true,
                triplet: false,
            },
            {
                beats: 1 / 3,
                duration: 8,
                noteType: "eighth",
                dotted: false,
                triplet: true,
            },
            {
                beats: 0.25,
                duration: 6,
                noteType: "16th",
                dotted: false,
                triplet: false,
            },
            {
                beats: 1 / 6,
                duration: 4,
                noteType: "16th",
                dotted: false,
                triplet: true,
            },
            {
                beats: 0.125,
                duration: 3,
                noteType: "32nd",
                dotted: false,
                triplet: false,
            },
            {
                beats: 1 / 12,
                duration: 2,
                noteType: "32nd",
                dotted: false,
                triplet: true,
            },
        ];
        const matched = canonicalSpecs.find((spec) => Math.abs(spec.beats - preferredBeats) <= 1 / 48);
        return matched || fallback;
    };
    const splitBeatsIntoNoteTypes = (beats) => {
        const result = [];
        const noteValues = [
            { beats: 4, noteType: "whole", duration: 96, dotted: false },
            { beats: 3, noteType: "half", duration: 72, dotted: true },
            { beats: 2, noteType: "half", duration: 48, dotted: false },
            { beats: 1.5, noteType: "quarter", duration: 36, dotted: true },
            { beats: 1, noteType: "quarter", duration: 24, dotted: false },
            { beats: 0.75, noteType: "eighth", duration: 18, dotted: true },
            { beats: 0.5, noteType: "eighth", duration: 12, dotted: false },
            { beats: 0.25, noteType: "16th", duration: 6, dotted: false },
            { beats: 0.125, noteType: "32nd", duration: 3, dotted: false },
        ];
        let remaining = Math.round(beats * 24) / 24;
        while (remaining >= 0.125 - 0.001) {
            let found = false;
            for (const nv of noteValues) {
                if (remaining >= nv.beats - 0.001) {
                    result.push({
                        noteType: nv.noteType,
                        duration: nv.duration,
                        beats: nv.beats,
                        dotted: nv.dotted,
                    });
                    remaining = Math.round((remaining - nv.beats) * 24) / 24;
                    found = true;
                    break;
                }
            }
            if (!found)
                break;
        }
        return result;
    };
    const generateNoteXmlWithTie = (pitchXml, duration, noteType, staff, tieType, isChord, dotted) => {
        const chordTag = isChord ? "<chord/>" : "";
        const dotXml = dotted ? "<dot/>" : "";
        let tieXml = "";
        let notationsXml = "";
        if (tieType === "start") {
            tieXml = '<tie type="start"/>';
            notationsXml = '<notations><tied type="start"/></notations>';
        }
        else if (tieType === "stop") {
            tieXml = '<tie type="stop"/>';
            notationsXml = '<notations><tied type="stop"/></notations>';
        }
        else if (tieType === "continue") {
            tieXml = '<tie type="stop"/><tie type="start"/>';
            notationsXml =
                '<notations><tied type="stop"/><tied type="start"/></notations>';
        }
        return `<note>${chordTag}${pitchXml}<duration>${duration}</duration>${tieXml}<voice>${staff}</voice><type>${noteType}</type>${dotXml}<staff>${staff}</staff>${notationsXml}</note>`;
    };
    const getSegmentTieType = (outerTieType, segIndex, totalSegments) => {
        if (totalSegments <= 1)
            return outerTieType;
        const isFirst = segIndex === 0;
        const isLast = segIndex === totalSegments - 1;
        if (isFirst) {
            return outerTieType === "stop" ? "continue" : outerTieType;
        }
        else if (isLast) {
            return outerTieType === "start" ? "continue" : outerTieType;
        }
        else {
            return "continue";
        }
    };
    const getAdjustedNoteType = (noteType) => noteType || "quarter";
    const getTripletNotations = (tripletPosition, actualNotes = 3, normalNotes = 2) => {
        if (!tripletPosition)
            return "";
        if (tripletPosition === "start") {
            return `<notations><tuplet type="start" bracket="yes" number="1"/></notations>`;
        }
        else if (tripletPosition === "end") {
            return `<notations><tuplet type="stop" number="1"/></notations>`;
        }
        return "";
    };
    const getOrnamentNotations = (ornament, trillTo) => {
        if (!ornament)
            return "";
        switch (ornament) {
            case "trill":
                let trillXml = '<ornaments><trill-mark placement="above"/>';
                if (trillTo !== undefined) {
                    const trillPitch = trillTo % 12;
                    const needsAccidental = [1, 3, 6, 8, 10].includes(trillPitch);
                    if (needsAccidental) {
                        trillXml += "<accidental-mark>sharp</accidental-mark>";
                    }
                }
                trillXml += "</ornaments>";
                return trillXml;
            case "mordent_upper":
                return "<ornaments><inverted-mordent/></ornaments>";
            case "mordent_lower":
                return "<ornaments><mordent/></ornaments>";
            case "turn_upper":
                return '<ornaments><turn placement="above"/></ornaments>';
            case "turn_inverted":
                return '<ornaments><inverted-turn placement="above"/></ornaments>';
            case "grace":
                return "";
            default:
                return "";
        }
    };
    const getTimeModification = (triplet, actualNotes = 3, normalNotes = 2) => {
        if (!triplet)
            return "";
        return `<time-modification><actual-notes>${actualNotes}</actual-notes><normal-notes>${normalNotes}</normal-notes></time-modification>`;
    };
    const generateRestXml = (beats, staff) => {
        const rests = [];
        let remaining = Math.round(beats * 24) / 24;
        const totalRounded = remaining;
        const restValues = [
            { beats: 4, type: "whole", duration: 96 },
            { beats: 2, type: "half", duration: 48 },
            { beats: 1, type: "quarter", duration: 24 },
            { beats: 0.5, type: "eighth", duration: 12 },
            { beats: 0.25, type: "16th", duration: 6 },
            { beats: 0.125, type: "32nd", duration: 3 },
        ];
        while (remaining >= 0.125 - 0.001) {
            let found = false;
            for (const rv of restValues) {
                if (remaining >= rv.beats - 0.001) {
                    rests.push(`<note><rest/><duration>${rv.duration}</duration><voice>${staff}</voice><type>${rv.type}</type><staff>${staff}</staff></note>`);
                    remaining = Math.round((remaining - rv.beats) * 24) / 24;
                    found = true;
                    break;
                }
            }
            if (!found)
                break;
        }
        if (remaining > 0.001) {
            const fwdDiv = Math.round(remaining * 24);
            if (fwdDiv > 0) {
                rests.push(`<forward><duration>${fwdDiv}</duration></forward>`);
                remaining = Math.round((remaining - fwdDiv / 24) * 24) / 24;
            }
        }
        return { xml: rests, beatsEmitted: totalRounded - remaining };
    };
    const getStaff = (midi) => {
        const octave = Math.floor(midi / 12) - 1;
        return octave < 4 ? 2 : 1;
    };
    const noteToXmlWithVoice = (n, isChordNote = false) => {
        const { step: baseStep, alter, octave, } = midiToStepOctaveForKey(n.midi_note, fifths);
        const staff = getStaff(n.midi_note);
        const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
        const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
        const durationSpec = getDurationSpec(n.note_divisions, n.note_value, n.dotted, n.triplet);
        const adjustedNoteType = getAdjustedNoteType(durationSpec.noteType);
        const dotXml = durationSpec.dotted ? "<dot/>" : "";
        const chordTag = isChordNote ? "<chord/>" : "";
        const timeModXml = getTimeModification(durationSpec.triplet, n.actual_notes || 3, n.normal_notes || 2);
        const tupletXml = getTripletNotations(n.triplet_position, n.actual_notes || 3, n.normal_notes || 2);
        const ornamentXml = getOrnamentNotations(n.ornament, n.trill_to);
        let notationsXml = "";
        if (tupletXml || ornamentXml) {
            let notationsContent = "";
            if (tupletXml) {
                const tupletMatch = tupletXml.match(/<notations>(.*)<\/notations>/);
                if (tupletMatch)
                    notationsContent += tupletMatch[1];
            }
            if (ornamentXml)
                notationsContent += ornamentXml;
            if (notationsContent) {
                notationsXml = `<notations>${notationsContent}</notations>`;
            }
        }
        if (n.ornament === "grace") {
            const graceType = n.grace_type === "appoggiatura" ? "<grace/>" : '<grace slash="yes"/>';
            return `<note>${graceType}${chordTag}${pitchXml}<voice>${staff}</voice><type>eighth</type><staff>${staff}</staff></note>`;
        }
        return `<note>${chordTag}${pitchXml}<duration>${durationSpec.duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${dotXml}${timeModXml}<staff>${staff}</staff>${notationsXml}</note>`;
    };
    const chordToMidiList = (c) => {
        var _a;
        if (c.midi_notes && c.midi_notes.length)
            return c.midi_notes.slice();
        const nameToSemitone = {
            C: 0,
            "C#": 1,
            DB: 1,
            D: 2,
            "D#": 3,
            EB: 3,
            E: 4,
            FB: 4,
            F: 5,
            "F#": 6,
            GB: 6,
            G: 7,
            "G#": 8,
            AB: 8,
            A: 9,
            "A#": 10,
            BB: 10,
            B: 11,
            CB: 11,
        };
        const buildFromRoot = (root, quality) => {
            const q = (quality || "").toLowerCase();
            if (q.includes("maj7") || q === "maj7")
                return [root, root + 4, root + 7, root + 11];
            if (q === "7" || q.includes("dom7") || q === "dom")
                return [root, root + 4, root + 7, root + 10];
            if (q === "m7" || q === "min7")
                return [root, root + 3, root + 7, root + 10];
            if (q === "m" || q === "min")
                return [root, root + 3, root + 7];
            if (q === "dim")
                return [root, root + 3, root + 6];
            if (q === "aug")
                return [root, root + 4, root + 8];
            if (q === "sus2")
                return [root, root + 2, root + 7];
            if (q === "sus4")
                return [root, root + 5, root + 7];
            return [root, root + 4, root + 7];
        };
        if (typeof c.root_midi === "number")
            return buildFromRoot(c.root_midi, c.chord_quality);
        if (c.label) {
            const m = String(c.label)
                .toUpperCase()
                .match(/^([A-G][#B]?)/);
            if (m) {
                const rootName = m[1].replace("B", "B");
                const semitone = (_a = nameToSemitone[rootName]) !== null && _a !== void 0 ? _a : 0;
                const octave = typeof c.octave === "number" ? c.octave : 4;
                const rootMidi = (octave + 1) * 12 + semitone;
                return buildFromRoot(rootMidi, c.chord_quality);
            }
        }
        return [];
    };
    const chordMidiToXml = (midiList, noteType, dotted, staff, preferredBeats, triplet, tripletPosition, actualNotes = 3, normalNotes = 2) => {
        const durationSpec = getDurationSpec(preferredBeats, noteType, dotted, triplet);
        const adjustedNoteType = getAdjustedNoteType(durationSpec.noteType);
        const dotXml = durationSpec.dotted ? "<dot/>" : "";
        const timeModXml = getTimeModification(durationSpec.triplet, actualNotes, normalNotes);
        const staffNotes = midiList.filter((m) => getStaff(m) === staff);
        if (staffNotes.length === 0)
            return [];
        const xmlParts = [];
        let first = true;
        for (const midi of staffNotes) {
            const { step: baseStep, alter, octave, } = midiToStepOctaveForKey(midi, fifths);
            const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
            const pitchInner = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
            const chordTag = first ? "" : "<chord/>";
            const tripletNotationsXml = first
                ? getTripletNotations(tripletPosition, actualNotes, normalNotes)
                : "";
            const noteXml = `<note>${chordTag}${pitchInner}<duration>${durationSpec.duration}</duration><voice>${staff}</voice><type>${adjustedNoteType}</type>${dotXml}${timeModXml}<staff>${staff}</staff>${tripletNotationsXml}</note>`;
            xmlParts.push(noteXml);
            first = false;
        }
        return xmlParts;
    };
    const timeline = [];
    const BEAT_KEY_TOLERANCE = 1 / 48;
    const notesInChords = new Set();
    for (const c of chords) {
        const beatStart = getAuthoritativeStartBeat(c);
        const midiList = chordToMidiList(c);
        for (const midi of midiList) {
            notesInChords.add(`${Math.round(beatStart / BEAT_KEY_TOLERANCE)}:${midi}`);
        }
    }
    for (const n of notes) {
        const time = (_a = n.time_seconds) !== null && _a !== void 0 ? _a : 0;
        const beatStart = getAuthoritativeStartBeat(n);
        const noteKey = `${Math.round(beatStart / BEAT_KEY_TOLERANCE)}:${n.midi_note}`;
        if (notesInChords.has(noteKey)) {
            continue;
        }
        const staff = getStaff(n.midi_note);
        const isGrace = n.ornament === "grace";
        const durationSpec = getDurationSpec(n.note_divisions, n.note_value, n.dotted, n.triplet);
        const beats = isGrace ? 0 : durationSpec.beats;
        const xml = [noteToXmlWithVoice(n, false)];
        timeline.push({
            time,
            beatStart,
            staff,
            beats,
            xml,
            midiNotes: [n.midi_note],
            triplet: isGrace ? false : n.triplet,
            tripletPosition: isGrace ? undefined : n.triplet_position,
            tripletType: isGrace ? undefined : n.triplet_type || n.note_value,
            actualNotes: isGrace ? undefined : n.actual_notes,
            normalNotes: isGrace ? undefined : n.normal_notes,
        });
    }
    for (const c of chords) {
        const time = (_b = c.time_seconds) !== null && _b !== void 0 ? _b : 0;
        const beatStart = getAuthoritativeStartBeat(c);
        let midiList = chordToMidiList(c);
        if (midiList.length === 0)
            continue;
        const inversionToIndex = (inv, chordLen) => {
            if (typeof inv === "number")
                return Math.max(0, Math.floor(inv));
            if (!inv)
                return 0;
            const s = String(inv).toLowerCase();
            if (s === "root")
                return 0;
            if (s === "first")
                return 1;
            if (s === "second")
                return 2;
            if (s === "third")
                return 3;
            if (s === "slash")
                return chordLen >= 4 ? 3 : 1;
            return 0;
        };
        const inversion = inversionToIndex(c.inversion, midiList.length);
        for (let i = 0; i < inversion; i++) {
            const n = midiList.shift();
            if (typeof n === "number")
                midiList.push(n + 12);
        }
        const noteType = c.note_value || "quarter";
        const dotted = c.dotted || false;
        const triplet = c.triplet || false;
        const durationSpec = getDurationSpec(c.note_divisions, noteType, dotted, triplet);
        const beats = durationSpec.beats;
        const trebleXml = chordMidiToXml(midiList, noteType, dotted, 1, c.note_divisions, triplet, c.triplet_position, c.actual_notes, c.normal_notes);
        const bassXml = chordMidiToXml(midiList, noteType, dotted, 2, c.note_divisions, triplet, c.triplet_position, c.actual_notes, c.normal_notes);
        if (trebleXml.length > 0) {
            const trebleMidiNotes = midiList.filter((m) => getStaff(m) === 1);
            timeline.push({
                time,
                beatStart,
                staff: 1,
                beats,
                xml: trebleXml,
                midiNotes: trebleMidiNotes,
                triplet,
                tripletPosition: c.triplet_position,
                tripletType: c.triplet_type || noteType,
                actualNotes: c.actual_notes,
                normalNotes: c.normal_notes,
            });
        }
        if (bassXml.length > 0) {
            const bassMidiNotes = midiList.filter((m) => getStaff(m) === 2);
            timeline.push({
                time,
                beatStart,
                staff: 2,
                beats,
                xml: bassXml,
                midiNotes: bassMidiNotes,
                triplet,
                tripletPosition: c.triplet_position,
                tripletType: c.triplet_type || noteType,
                actualNotes: c.actual_notes,
                normalNotes: c.normal_notes,
            });
        }
    }
    timeline.sort((a, b) => a.beatStart - b.beatStart || a.time - b.time);
    const CROSS_STAFF_TIME_TOLERANCE = 0.025;
    const SAME_STAFF_TOLERANCE = 0.005;
    const BEAT_GROUP_TOLERANCE = 1 / 48;
    const SAME_STAFF_BEAT_TOLERANCE = 1 / 96;
    const timeGroups = [];
    for (const ev of timeline) {
        let group = timeGroups.find((g) => {
            const beatDelta = Math.abs(g.beatStart - ev.beatStart);
            const timeDelta = Math.abs(g.time - ev.time);
            if (beatDelta < BEAT_GROUP_TOLERANCE &&
                timeDelta < CROSS_STAFF_TIME_TOLERANCE) {
                const sameStaffEvents = ev.staff === 1 ? g.treble : g.bass;
                if (sameStaffEvents.length > 0) {
                    return (beatDelta < SAME_STAFF_BEAT_TOLERANCE &&
                        timeDelta < SAME_STAFF_TOLERANCE);
                }
                return true;
            }
            return false;
        });
        if (!group) {
            group = { time: ev.time, beatStart: ev.beatStart, treble: [], bass: [] };
            timeGroups.push(group);
        }
        else {
            group.time = Math.min(group.time, ev.time);
            group.beatStart = Math.min(group.beatStart, ev.beatStart);
        }
        if (ev.staff === 1) {
            group.treble.push(ev);
        }
        else {
            group.bass.push(ev);
        }
    }
    timeGroups.sort((a, b) => a.beatStart - b.beatStart || a.time - b.time);
    for (const group of timeGroups) {
        for (const staffKey of ["treble", "bass"]) {
            const events = group[staffKey];
            if (events.length <= 1)
                continue;
            const seenMidi = new Set();
            const deduped = [];
            for (const ev of events) {
                const uniqueMidi = (ev.midiNotes || []).filter((m) => {
                    if (seenMidi.has(m))
                        return false;
                    seenMidi.add(m);
                    return true;
                });
                if (uniqueMidi.length === 0)
                    continue;
                if (uniqueMidi.length === ev.midiNotes.length) {
                    deduped.push(ev);
                }
                else {
                    const filteredXml = ev.xml.filter((xml) => {
                        const pitchMatch = xml.match(/<pitch><step>(\w+)<\/step>(?:<alter>(-?\d)<\/alter>)?<octave>(\d+)<\/octave><\/pitch>/);
                        if (!pitchMatch)
                            return true;
                        const step = pitchMatch[1];
                        const alter = pitchMatch[2] ? parseInt(pitchMatch[2]) : 0;
                        const octave = parseInt(pitchMatch[3]);
                        const noteNames = [
                            "C",
                            "C#",
                            "D",
                            "D#",
                            "E",
                            "F",
                            "F#",
                            "G",
                            "G#",
                            "A",
                            "A#",
                            "B",
                        ];
                        const stepIdx = noteNames.indexOf(step + (alter === 1 ? "#" : alter === -1 ? "b" : ""));
                        const midi = stepIdx >= 0 ? (octave + 1) * 12 + stepIdx : -1;
                        return midi < 0 || uniqueMidi.includes(midi);
                    });
                    if (filteredXml.length > 0) {
                        deduped.push({
                            ...ev,
                            midiNotes: uniqueMidi,
                            xml: filteredXml,
                        });
                    }
                }
            }
            group[staffKey] = deduped;
        }
    }
    const CHORD_MERGE_TOLERANCE = 0;
    for (const group of timeGroups) {
        if (group.treble.length > 1) {
            const subGroups = [];
            for (const ev of group.treble) {
                let found = false;
                for (const sg of subGroups) {
                    if (Math.abs(sg[0].time - ev.time) < CHORD_MERGE_TOLERANCE) {
                        sg.push(ev);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    subGroups.push([ev]);
                }
            }
            const newTreble = [];
            for (const sg of subGroups) {
                if (sg.length === 1) {
                    newTreble.push(sg[0]);
                }
                else {
                    const mergedXml = [];
                    const mergedMidiNotes = [];
                    let maxBeats = 0;
                    let first = true;
                    for (const ev of sg) {
                        mergedMidiNotes.push(...(ev.midiNotes || []));
                        for (const xml of ev.xml) {
                            if (first) {
                                mergedXml.push(xml);
                                first = false;
                            }
                            else {
                                const chordXml = xml.replace("<note>", "<note><chord/>");
                                mergedXml.push(chordXml);
                            }
                        }
                        maxBeats = Math.max(maxBeats, ev.beats);
                    }
                    newTreble.push({
                        time: sg[0].time,
                        beatStart: sg[0].beatStart,
                        staff: 1,
                        beats: maxBeats,
                        xml: mergedXml,
                        midiNotes: mergedMidiNotes,
                        triplet: sg[0].triplet,
                        tripletPosition: sg[0].tripletPosition,
                        tripletType: sg[0].tripletType,
                        actualNotes: sg[0].actualNotes,
                        normalNotes: sg[0].normalNotes,
                    });
                }
            }
            group.treble = newTreble;
        }
        if (group.bass.length > 1) {
            const subGroups = [];
            for (const ev of group.bass) {
                let found = false;
                for (const sg of subGroups) {
                    if (Math.abs(sg[0].time - ev.time) < CHORD_MERGE_TOLERANCE) {
                        sg.push(ev);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    subGroups.push([ev]);
                }
            }
            const newBass = [];
            for (const sg of subGroups) {
                if (sg.length === 1) {
                    newBass.push(sg[0]);
                }
                else {
                    const mergedXml = [];
                    const mergedMidiNotes = [];
                    let maxBeats = 0;
                    let first = true;
                    for (const ev of sg) {
                        mergedMidiNotes.push(...(ev.midiNotes || []));
                        for (const xml of ev.xml) {
                            if (first) {
                                mergedXml.push(xml);
                                first = false;
                            }
                            else {
                                const chordXml = xml.replace("<note>", "<note><chord/>");
                                mergedXml.push(chordXml);
                            }
                        }
                        maxBeats = Math.max(maxBeats, ev.beats);
                    }
                    newBass.push({
                        time: sg[0].time,
                        beatStart: sg[0].beatStart,
                        staff: 2,
                        beats: maxBeats,
                        xml: mergedXml,
                        midiNotes: mergedMidiNotes,
                        triplet: sg[0].triplet,
                        tripletPosition: sg[0].tripletPosition,
                        tripletType: sg[0].tripletType,
                        actualNotes: sg[0].actualNotes,
                        normalNotes: sg[0].normalNotes,
                    });
                }
            }
            group.bass = newBass;
        }
    }
    const stripTripletFromXml = (xmlArr) => {
        return xmlArr.map((xml) => {
            let result = xml.replace(/<time-modification>.*?<\/time-modification>/g, "");
            result = result.replace(/<tuplet[^>]*\/>/g, "");
            result = result.replace(/<notations>\s*<\/notations>/g, "");
            return result;
        });
    };
    const stripTripletFromEvent = (ev) => {
        ev.xml = stripTripletFromXml(ev.xml);
        ev.triplet = false;
        ev.tripletPosition = undefined;
    };
    for (const staff of [1, 2]) {
        const events = staff === 1
            ? timeGroups.flatMap((g) => g.treble)
            : timeGroups.flatMap((g) => g.bass);
        let tripletStart = null;
        let tripletMiddle = null;
        for (const ev of events) {
            if (ev.tripletPosition === "start") {
                if (tripletStart)
                    stripTripletFromEvent(tripletStart);
                if (tripletMiddle)
                    stripTripletFromEvent(tripletMiddle);
                tripletStart = ev;
                tripletMiddle = null;
            }
            else if (ev.tripletPosition === "middle" && tripletStart) {
                tripletMiddle = ev;
            }
            else if (ev.tripletPosition === "end") {
                if (tripletStart && tripletMiddle) {
                    const sameType = tripletStart.tripletType === tripletMiddle.tripletType &&
                        tripletMiddle.tripletType === ev.tripletType;
                    const startBeat = tripletStart.beatStart;
                    const middleBeat = tripletMiddle.beatStart;
                    const endBeat = ev.beatStart;
                    const startMeasure = Math.floor(startBeat / BEATS_PER_MEASURE);
                    const middleMeasure = Math.floor(middleBeat / BEATS_PER_MEASURE);
                    const endMeasure = Math.floor(endBeat / BEATS_PER_MEASURE);
                    const sameMeasure = startMeasure === middleMeasure && middleMeasure === endMeasure;
                    if (!sameType || !sameMeasure) {
                        stripTripletFromEvent(tripletStart);
                        stripTripletFromEvent(tripletMiddle);
                        stripTripletFromEvent(ev);
                    }
                }
                else {
                    if (tripletStart)
                        stripTripletFromEvent(tripletStart);
                    if (tripletMiddle)
                        stripTripletFromEvent(tripletMiddle);
                    stripTripletFromEvent(ev);
                }
                tripletStart = null;
                tripletMiddle = null;
            }
        }
        if (tripletStart)
            stripTripletFromEvent(tripletStart);
        if (tripletMiddle)
            stripTripletFromEvent(tripletMiddle);
    }
    const measuresData = [];
    let currentMeasure = { trebleEvents: [], bassEvents: [] };
    let currentBeatPos = 0;
    let pendingTrebleTies = [];
    let pendingBassTies = [];
    const addPendingTiesToMeasure = () => {
        for (const tie of pendingTrebleTies) {
            const beatsThisMeasure = Math.min(tie.remainingBeats, BEATS_PER_MEASURE);
            const tieType = tie.remainingBeats > BEATS_PER_MEASURE ? "continue" : "stop";
            const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
            const xml = [];
            for (let si = 0; si < segments.length; si++) {
                const seg = segments[si];
                const segTie = getSegmentTieType(tieType, si, segments.length);
                let first = xml.length === 0;
                for (const midi of tie.midiNotes) {
                    const { step: baseStep, alter, octave, } = midiToStepOctaveForKey(midi, fifths);
                    const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
                    const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
                    xml.push(generateNoteXmlWithTie(pitchXml, seg.duration, seg.noteType, 1, segTie, !first, seg.dotted));
                    first = false;
                }
            }
            currentMeasure.trebleEvents.push({
                beatPos: 0,
                xml,
                beats: beatsThisMeasure,
                midiNotes: tie.midiNotes,
                staff: 1,
                time: -1,
            });
            tie.remainingBeats -= beatsThisMeasure;
        }
        pendingTrebleTies = pendingTrebleTies.filter((t) => t.remainingBeats > 0.001);
        for (const tie of pendingBassTies) {
            const beatsThisMeasure = Math.min(tie.remainingBeats, BEATS_PER_MEASURE);
            const tieType = tie.remainingBeats > BEATS_PER_MEASURE ? "continue" : "stop";
            const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
            const xml = [];
            for (let si = 0; si < segments.length; si++) {
                const seg = segments[si];
                const segTie = getSegmentTieType(tieType, si, segments.length);
                let first = xml.length === 0;
                for (const midi of tie.midiNotes) {
                    const { step: baseStep, alter, octave, } = midiToStepOctaveForKey(midi, fifths);
                    const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
                    const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
                    xml.push(generateNoteXmlWithTie(pitchXml, seg.duration, seg.noteType, 2, segTie, !first, seg.dotted));
                    first = false;
                }
            }
            currentMeasure.bassEvents.push({
                beatPos: 0,
                xml,
                beats: beatsThisMeasure,
                midiNotes: tie.midiNotes,
                staff: 2,
                time: -1,
            });
            tie.remainingBeats -= beatsThisMeasure;
        }
        pendingBassTies = pendingBassTies.filter((t) => t.remainingBeats > 0.001);
    };
    const addEventToMeasure = (ev, events, pendingTies) => {
        const remainingInMeasure = BEATS_PER_MEASURE - currentBeatPos;
        if (ev.beats <= remainingInMeasure + 0.001) {
            events.push({
                beatPos: currentBeatPos,
                xml: ev.xml,
                beats: ev.beats,
                midiNotes: ev.midiNotes || [],
                staff: ev.staff,
                time: ev.time,
            });
        }
        else {
            const beatsThisMeasure = remainingInMeasure;
            const beatsRemaining = ev.beats - beatsThisMeasure;
            if (beatsThisMeasure > 0.001 && ev.midiNotes && ev.midiNotes.length > 0) {
                const segments = splitBeatsIntoNoteTypes(beatsThisMeasure);
                const xml = [];
                for (let si = 0; si < segments.length; si++) {
                    const seg = segments[si];
                    const segTie = getSegmentTieType("start", si, segments.length);
                    let first = xml.length === 0;
                    for (const midi of ev.midiNotes) {
                        const { step: baseStep, alter, octave, } = midiToStepOctaveForKey(midi, fifths);
                        const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
                        const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
                        xml.push(generateNoteXmlWithTie(pitchXml, seg.duration, seg.noteType, ev.staff, segTie, !first, seg.dotted));
                        first = false;
                    }
                }
                events.push({
                    beatPos: currentBeatPos,
                    xml,
                    beats: beatsThisMeasure,
                    midiNotes: ev.midiNotes,
                    staff: ev.staff,
                    time: ev.time,
                });
                pendingTies.push({
                    midiNotes: ev.midiNotes,
                    remainingBeats: beatsRemaining,
                    staff: ev.staff,
                });
            }
            else {
                pendingTies.push({
                    midiNotes: ev.midiNotes || [],
                    remainingBeats: ev.beats,
                    staff: ev.staff,
                });
            }
        }
    };
    let currentMeasureIndex = 0;
    for (const group of timeGroups) {
        const groupMeasureIndex = Math.max(0, Math.floor(group.beatStart / BEATS_PER_MEASURE));
        while (currentMeasureIndex < groupMeasureIndex) {
            if (currentMeasure.trebleEvents.length > 0 ||
                currentMeasure.bassEvents.length > 0) {
                measuresData.push(currentMeasure);
            }
            currentMeasure = { trebleEvents: [], bassEvents: [] };
            currentBeatPos = 0;
            currentMeasureIndex++;
            addPendingTiesToMeasure();
        }
        const beatPosFromGrid = group.beatStart - currentMeasureIndex * BEATS_PER_MEASURE;
        currentBeatPos = Math.min(Math.max(0, Math.round(beatPosFromGrid * 24) / 24), BEATS_PER_MEASURE - 1 / 24);
        for (const ev of group.treble) {
            addEventToMeasure(ev, currentMeasure.trebleEvents, pendingTrebleTies);
        }
        for (const ev of group.bass) {
            addEventToMeasure(ev, currentMeasure.bassEvents, pendingBassTies);
        }
    }
    while (pendingTrebleTies.length > 0 || pendingBassTies.length > 0) {
        if (currentMeasure.trebleEvents.length > 0 ||
            currentMeasure.bassEvents.length > 0) {
            measuresData.push(currentMeasure);
        }
        currentMeasure = { trebleEvents: [], bassEvents: [] };
        currentBeatPos = 0;
        addPendingTiesToMeasure();
    }
    if (currentMeasure.trebleEvents.length > 0 ||
        currentMeasure.bassEvents.length > 0) {
        measuresData.push(currentMeasure);
    }
    for (let mIdx = 0; mIdx < measuresData.length; mIdx++) {
        const mData = measuresData[mIdx];
        const measureNum = mIdx + 1;
        let measureContent = "";
        if (measureNum === 1) {
            const [beats, beatType] = timeSignature === "6/8"
                ? ["6", "8"]
                : timeSignature === "3/4"
                    ? ["3", "4"]
                    : ["4", "4"];
            measureContent += `<attributes><divisions>24</divisions><key><fifths>${fifths}</fifths></key><time><beats>${beats}</beats><beat-type>${beatType}</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>`;
            measureContent += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`;
        }
        mData.trebleEvents.sort((a, b) => a.beatPos - b.beatPos);
        mData.bassEvents.sort((a, b) => a.beatPos - b.beatPos);
        const MIN_REST_GAP = 0.12;
        const midiFromXml = (xml) => {
            var _a;
            const m = xml.match(/<pitch><step>(\w+)<\/step>(?:<alter>(-?\d)<\/alter>)?<octave>(\d+)<\/octave><\/pitch>/);
            if (!m)
                return null;
            const stepNames = {
                C: 0,
                D: 2,
                E: 4,
                F: 5,
                G: 7,
                A: 9,
                B: 11,
            };
            const base = (_a = stepNames[m[1]]) !== null && _a !== void 0 ? _a : 0;
            const alter = m[2] ? parseInt(m[2]) : 0;
            const octave = parseInt(m[3]);
            return (octave + 1) * 12 + base + alter;
        };
        const consolidateEvents = (events) => {
            if (events.length === 0)
                return events;
            events.sort((a, b) => a.beatPos - b.beatPos);
            const consolidated = [];
            let currentGroup = [events[0]];
            const finalizeGroup = (group) => {
                const maxBeats = Math.max(...group.map((e) => e.beats));
                const canonicalDuration = Math.round(maxBeats * 24);
                const mergedXml = [];
                const mergedMidi = [];
                const seenMidi = new Set();
                let first = true;
                for (const groupEv of group) {
                    for (let xi = 0; xi < groupEv.xml.length; xi++) {
                        let xml = groupEv.xml[xi];
                        const midi = xi < groupEv.midiNotes.length
                            ? groupEv.midiNotes[xi]
                            : midiFromXml(xml);
                        if (midi !== null && seenMidi.has(midi))
                            continue;
                        if (midi !== null)
                            seenMidi.add(midi);
                        xml = xml.replace(/<duration>\d+<\/duration>/, `<duration>${canonicalDuration}</duration>`);
                        if (first) {
                            mergedXml.push(xml);
                            first = false;
                        }
                        else if (!xml.includes("<chord/>")) {
                            mergedXml.push(xml.replace("<note>", "<note><chord/>"));
                        }
                        else {
                            mergedXml.push(xml);
                        }
                        if (midi !== null)
                            mergedMidi.push(midi);
                    }
                }
                consolidated.push({
                    beatPos: group[0].beatPos,
                    xml: mergedXml,
                    beats: maxBeats,
                    midiNotes: mergedMidi,
                    staff: group[0].staff,
                    time: group[0].time,
                });
            };
            for (let i = 1; i < events.length; i++) {
                const ev = events[i];
                const sameBeatPos = Math.abs(ev.beatPos - currentGroup[0].beatPos) < 0.001;
                const sameTime = Math.abs(ev.time - currentGroup[0].time) < SAME_STAFF_TOLERANCE;
                if (sameBeatPos && sameTime) {
                    currentGroup.push(ev);
                }
                else {
                    finalizeGroup(currentGroup);
                    currentGroup = [ev];
                }
            }
            if (currentGroup.length > 0) {
                finalizeGroup(currentGroup);
            }
            return consolidated;
        };
        const clampEventsToMeasure = (events) => {
            if (events.length === 0)
                return { events, overflows: [] };
            const result = [];
            const overflows = [];
            let currentBeatEnd = 0;
            for (const ev of events) {
                const startPos = Math.max(ev.beatPos, currentBeatEnd);
                if (startPos >= BEATS_PER_MEASURE - 0.001) {
                    if (ev.midiNotes.length > 0) {
                        overflows.push({
                            midiNotes: ev.midiNotes,
                            remainingBeats: ev.beats,
                            staff: ev.staff,
                        });
                    }
                    continue;
                }
                const remaining = BEATS_PER_MEASURE - startPos;
                if (ev.beats <= remaining + 0.001) {
                    result.push({ ...ev, beatPos: startPos });
                    currentBeatEnd = startPos + ev.beats;
                }
                else {
                    const truncatedBeats = remaining;
                    if (truncatedBeats < 0.125) {
                        if (ev.midiNotes.length > 0) {
                            overflows.push({
                                midiNotes: ev.midiNotes,
                                remainingBeats: ev.beats,
                                staff: ev.staff,
                            });
                        }
                        continue;
                    }
                    const segments = splitBeatsIntoNoteTypes(truncatedBeats);
                    const xml = [];
                    const outerTie = ev.xml.some((x) => x.includes('tie type="stop"'))
                        ? "continue"
                        : "start";
                    for (let si = 0; si < segments.length; si++) {
                        const seg = segments[si];
                        const segTie = getSegmentTieType(outerTie, si, segments.length);
                        let first = xml.length === 0;
                        for (const midi of ev.midiNotes) {
                            const { step: baseStep, alter, octave, } = midiToStepOctaveForKey(midi, fifths);
                            const alterXml = alter !== 0 ? `<alter>${alter}</alter>` : "";
                            const pitchXml = `<pitch><step>${baseStep}</step>${alterXml}<octave>${octave}</octave></pitch>`;
                            xml.push(generateNoteXmlWithTie(pitchXml, seg.duration, seg.noteType, ev.staff, segTie, !first, seg.dotted));
                            first = false;
                        }
                    }
                    result.push({
                        beatPos: startPos,
                        xml,
                        beats: truncatedBeats,
                        midiNotes: ev.midiNotes,
                        staff: ev.staff,
                        time: ev.time,
                    });
                    const overflowBeats = ev.beats - truncatedBeats;
                    if (overflowBeats > 0.001 && ev.midiNotes.length > 0) {
                        overflows.push({
                            midiNotes: ev.midiNotes,
                            remainingBeats: overflowBeats,
                            staff: ev.staff,
                        });
                    }
                    currentBeatEnd = BEATS_PER_MEASURE;
                }
            }
            return { events: result, overflows };
        };
        const consolidatedTreble = consolidateEvents([...mData.trebleEvents]);
        const trebleClampResult = clampEventsToMeasure(consolidatedTreble);
        const clampedTreble = trebleClampResult.events;
        for (const ov of trebleClampResult.overflows) {
            pendingTrebleTies.push(ov);
        }
        let trebleBeatPos = 0;
        for (let evIdx = 0; evIdx < clampedTreble.length; evIdx++) {
            const ev = clampedTreble[evIdx];
            const gap = ev.beatPos - trebleBeatPos;
            if (gap < -0.001) {
                trebleBeatPos = ev.beatPos;
            }
            else if (gap > MIN_REST_GAP) {
                const restResult = generateRestXml(gap, 1);
                measureContent += restResult.xml.join("");
                trebleBeatPos += restResult.beatsEmitted;
            }
            else if (gap > 0.001) {
                const forwardDivisions = Math.round(gap * 24);
                if (forwardDivisions > 0) {
                    measureContent += `<forward><duration>${forwardDivisions}</duration></forward>`;
                    trebleBeatPos += forwardDivisions / 24;
                }
            }
            measureContent += ev.xml.join("");
            trebleBeatPos += ev.beats;
            trebleBeatPos = Math.round(trebleBeatPos * 24) / 24;
        }
        const trebleShortfall = Math.round((BEATS_PER_MEASURE - trebleBeatPos) * 24) / 24;
        if (trebleShortfall > 0.001) {
            const padResult = generateRestXml(trebleShortfall, 1);
            measureContent += padResult.xml.join("");
            trebleBeatPos += padResult.beatsEmitted;
        }
        const backupDuration = Math.round(trebleBeatPos * 24);
        if (backupDuration > 0) {
            measureContent += `<backup><duration>${backupDuration}</duration></backup>`;
        }
        const consolidatedBass = consolidateEvents([...mData.bassEvents]);
        const bassClampResult = clampEventsToMeasure(consolidatedBass);
        const clampedBass = bassClampResult.events;
        for (const ov of bassClampResult.overflows) {
            pendingBassTies.push(ov);
        }
        let bassBeatPos = 0;
        for (let evIdx = 0; evIdx < clampedBass.length; evIdx++) {
            const ev = clampedBass[evIdx];
            const gap = ev.beatPos - bassBeatPos;
            if (gap < -0.001) {
                bassBeatPos = ev.beatPos;
            }
            else if (gap > MIN_REST_GAP) {
                const restResult = generateRestXml(gap, 2);
                measureContent += restResult.xml.join("");
                bassBeatPos += restResult.beatsEmitted;
            }
            else if (gap > 0.001) {
                const forwardDivisions = Math.round(gap * 24);
                if (forwardDivisions > 0) {
                    measureContent += `<forward><duration>${forwardDivisions}</duration></forward>`;
                    bassBeatPos += forwardDivisions / 24;
                }
            }
            measureContent += ev.xml.join("");
            bassBeatPos += ev.beats;
            bassBeatPos = Math.round(bassBeatPos * 24) / 24;
        }
        const bassShortfall = Math.round((BEATS_PER_MEASURE - bassBeatPos) * 24) / 24;
        if (bassShortfall > 0.001) {
            const padResult = generateRestXml(bassShortfall, 2);
            measureContent += padResult.xml.join("");
        }
        measures.push(`<measure number="${measureNum}">${measureContent}</measure>`);
    }
    return measures;
}
function generateMusicXML(notes, chords, timeSignature = "4/4", bpm = 120, fifths = 0) {
    const measures = generateMeasureXmls(notes, chords, timeSignature, bpm, fifths);
    const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n<score-partwise version="3.1">\n  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>\n  <part id="P1">${measures.join("")}</part></score-partwise>`;
    return xml;
}
function generateBlankPageMusicXML(timeSignature = "4/4", bpm = 120, fifths = 0, measureCount = 12) {
    const timeConfig = timeSignature === "6/8"
        ? {
            beats: "6",
            beatType: "8",
            measureDuration: 72,
        }
        : timeSignature === "3/4"
            ? {
                beats: "3",
                beatType: "4",
                measureDuration: 72,
            }
            : {
                beats: "4",
                beatType: "4",
                measureDuration: 96,
            };
    const measures = Array.from({ length: measureCount }, (_, measureIndex) => {
        const measureNumber = measureIndex + 1;
        let measureContent = "";
        if (measureNumber === 1) {
            measureContent += `<attributes><divisions>24</divisions><key><fifths>${fifths}</fifths></key><time><beats>${timeConfig.beats}</beats><beat-type>${timeConfig.beatType}</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>`;
            measureContent += `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${Math.round(bpm)}</per-minute></metronome></direction-type><sound tempo="${Math.round(bpm)}"/></direction>`;
        }
        measureContent += `<note><rest measure="yes"/><duration>${timeConfig.measureDuration}</duration><voice>1</voice><staff>1</staff></note>`;
        measureContent += `<backup><duration>${timeConfig.measureDuration}</duration></backup>`;
        measureContent += `<note><rest measure="yes"/><duration>${timeConfig.measureDuration}</duration><voice>2</voice><staff>2</staff></note>`;
        return `<measure number="${measureNumber}">${measureContent}</measure>`;
    }).join("");
    return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">${measures}</part>
</score-partwise>`;
}
const FALLBACK_XML = generateBlankPageMusicXML();
function PianoSheetMusic({ results, timeSignature = "4/4", keySignature = 0, compact = false, viewportHeight, refinementVersion, onScoreRendered, onScoreScrollActiveChange, }) {
    var _a, _b, _c;
    const [accumulatedNotes, setAccumulatedNotes] = (0, react_1.useState)([]);
    const [accumulatedChords, setAccumulatedChords] = (0, react_1.useState)([]);
    const hasReceivedDataRef = (0, react_1.useRef)(false);
    const lastRefinementVersionRef = (0, react_1.useRef)(undefined);
    (0, react_1.useEffect)(() => {
        var _a, _b;
        if (!results) {
            if (hasReceivedDataRef.current) {
                setAccumulatedNotes([]);
                setAccumulatedChords([]);
                hasReceivedDataRef.current = false;
            }
            return;
        }
        const incomingNotes = (_a = results.notes) !== null && _a !== void 0 ? _a : [];
        const incomingChords = (_b = results.chords) !== null && _b !== void 0 ? _b : [];
        const isRefinementUpdate = refinementVersion !== undefined &&
            refinementVersion !== lastRefinementVersionRef.current;
        lastRefinementVersionRef.current = refinementVersion;
        if (isRefinementUpdate &&
            (incomingNotes.length > 0 || incomingChords.length > 0)) {
            hasReceivedDataRef.current = true;
            setAccumulatedNotes(incomingNotes);
            setAccumulatedChords(incomingChords);
            return;
        }
        if (incomingNotes.length === 0 && incomingChords.length === 0)
            return;
        hasReceivedDataRef.current = true;
        setAccumulatedNotes((prev) => {
            var _a;
            const seen = new Set();
            for (const n of prev)
                seen.add(`${n.time_seconds.toFixed(6)}:${n.midi_note}`);
            const toAdd = [];
            for (const n of incomingNotes) {
                const key = `${((_a = n.time_seconds) !== null && _a !== void 0 ? _a : 0).toFixed(6)}:${n.midi_note}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    toAdd.push(n);
                }
            }
            return [...prev, ...toAdd];
        });
        setAccumulatedChords((prev) => {
            var _a, _b;
            const seen = new Set();
            for (const c of prev)
                seen.add(`${((_a = c.time_seconds) !== null && _a !== void 0 ? _a : 0).toFixed(6)}:${(c.midi_notes || []).join("-")}`);
            const toAdd = [];
            for (const c of incomingChords) {
                const key = `${((_b = c.time_seconds) !== null && _b !== void 0 ? _b : 0).toFixed(6)}:${(c.midi_notes || []).join("-")}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    toAdd.push(c);
                }
            }
            return [...prev, ...toAdd];
        });
    }, [results, refinementVersion]);
    const detectedBPM = (_b = (_a = results === null || results === void 0 ? void 0 : results.analysis_summary) === null || _a === void 0 ? void 0 : _a.detected_bpm) !== null && _b !== void 0 ? _b : 120;
    const score = (0, react_1.useMemo)(() => {
        if ((!accumulatedNotes || accumulatedNotes.length === 0) &&
            (!accumulatedChords || accumulatedChords.length === 0))
            return FALLBACK_XML;
        return generateMusicXML(accumulatedNotes, accumulatedChords, timeSignature, detectedBPM, keySignature);
    }, [
        accumulatedNotes,
        accumulatedChords,
        timeSignature,
        detectedBPM,
        keySignature,
    ]);
    const hasPlayableScore = score !== FALLBACK_XML;
    const webRef = (0, react_1.useRef)(null);
    const source = (0, react_1.useMemo)(() => ({
        html: osmdHTML_1.OSMD_HTML,
        baseUrl: "https://osmd.local/",
    }), []);
    const shouldFollowLatest = ((_c = results === null || results === void 0 ? void 0 : results.analysis_summary) === null || _c === void 0 ? void 0 : _c.method) === "live";
    const measuresSentRef = (0, react_1.useRef)(0);
    const lastXmlRef = (0, react_1.useRef)(null);
    const pendingXmlRef = (0, react_1.useRef)(null);
    const playAfterRenderRef = (0, react_1.useRef)(false);
    const pendingSentAtRef = (0, react_1.useRef)(0);
    const pendingRenderIdRef = (0, react_1.useRef)(null);
    const nextRenderIdRef = (0, react_1.useRef)(1);
    const renderProbeTimeoutRef = (0, react_1.useRef)(null);
    const renderRefinementRef = (0, react_1.useRef)(undefined);
    const [, setDebugSnapshot] = (0, react_1.useState)(null);
    const [, setDebugEvents] = (0, react_1.useState)([]);
    const appendDebugEvent = (0, react_1.useCallback)((message) => {
        const timestamp = new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
        const line = `${timestamp} ${message}`;
        setDebugEvents((prev) => [line, ...prev].slice(0, 8));
    }, []);
    const injectWebCommand = (0, react_1.useCallback)((script, description) => {
        if (!webRef.current) {
            return;
        }
        webRef.current.injectJavaScript(`
        try {
          ${script}
        } catch (e) {
          try {
            if (window.ReactNativeWebView) {
              window.ReactNativeWebView.postMessage(JSON.stringify({
                type: 'error',
                error: 'Injected command failed (${description}): ' + String(e),
              }));
            }
          } catch (_bridgeError) {}
        }
        true;
      `);
    }, []);
    const requestDebugSnapshot = (0, react_1.useCallback)((reason, requestId) => {
        injectWebCommand(`
          if (window.__OSMD_DEBUG_SNAPSHOT) {
            window.__OSMD_DEBUG_SNAPSHOT(${JSON.stringify(reason)}, ${requestId !== null && requestId !== void 0 ? requestId : null});
          }
        `, `debug-snapshot-${reason}`);
    }, [injectWebCommand]);
    const sendRenderXml = (0, react_1.useCallback)((xml, description, extraScript = "") => {
        const requestId = nextRenderIdRef.current;
        nextRenderIdRef.current += 1;
        pendingXmlRef.current = xml;
        pendingSentAtRef.current = Date.now();
        pendingRenderIdRef.current = requestId;
        appendDebugEvent(`render request #${requestId} (${description}) xml=${xml.length}`);
        if (renderProbeTimeoutRef.current) {
            clearTimeout(renderProbeTimeoutRef.current);
        }
        renderProbeTimeoutRef.current = setTimeout(() => {
            renderProbeTimeoutRef.current = null;
            requestDebugSnapshot(`render-timeout:${description}`, requestId);
        }, 1500);
        injectWebCommand(`
          if (window.__OSMD_RENDER_XML) window.__OSMD_RENDER_XML(${JSON.stringify(xml)}, ${requestId});
          ${extraScript}
        `, description);
    }, [appendDebugEvent, injectWebCommand, requestDebugSnapshot]);
    const [isLandscape, setIsLandscape] = (0, react_1.useState)(false);
    const [isPlaying, setIsPlaying] = (0, react_1.useState)(false);
    const [isPaused, setIsPaused] = (0, react_1.useState)(false);
    const [playbackBPM, setPlaybackBPM] = (0, react_1.useState)(detectedBPM);
    const [cameraMotionMode, setCameraMotionMode] = (0, react_1.useState)("smooth");
    const compactViewportHeight = Math.max(220, Math.round(viewportHeight !== null && viewportHeight !== void 0 ? viewportHeight : 280));
    const lastDetectedBPMRef = (0, react_1.useRef)(undefined);
    const [webViewReady, setWebViewReady] = (0, react_1.useState)(false);
    const webViewReadyRef = (0, react_1.useRef)(false);
    const scoreScrollActiveRef = (0, react_1.useRef)(false);
    const updateScoreScrollActive = (0, react_1.useCallback)((active) => {
        if (scoreScrollActiveRef.current === active) {
            return;
        }
        scoreScrollActiveRef.current = active;
        onScoreScrollActiveChange === null || onScoreScrollActiveChange === void 0 ? void 0 : onScoreScrollActiveChange(active);
    }, [onScoreScrollActiveChange]);
    (0, react_1.useEffect)(() => {
        return () => {
            updateScoreScrollActive(false);
        };
    }, [updateScoreScrollActive]);
    const requestScorePlayback = (0, react_1.useCallback)(() => {
        if (score === FALLBACK_XML) {
            appendDebugEvent("playback blocked: fallback score");
            return;
        }
        if (webViewReadyRef.current &&
            !pendingXmlRef.current &&
            lastXmlRef.current) {
            injectWebCommand(`if (window.__OSMD_PLAY) window.__OSMD_PLAY(${JSON.stringify(playbackBPM)});`, "play-score");
            return;
        }
        playAfterRenderRef.current = true;
        appendDebugEvent("queued playback until score render completes");
    }, [appendDebugEvent, injectWebCommand, playbackBPM, score]);
    const handlePlayPause = (0, react_1.useCallback)(() => {
        if (!isPlaying || isPaused) {
            requestScorePlayback();
            return;
        }
        injectWebCommand(`if (window.__OSMD_PAUSE) window.__OSMD_PAUSE();`, "pause-score");
    }, [injectWebCommand, isPaused, isPlaying, requestScorePlayback]);
    const handleStopPlayback = (0, react_1.useCallback)(() => {
        playAfterRenderRef.current = false;
        injectWebCommand(`if (window.__OSMD_STOP) window.__OSMD_STOP();`, "stop-score");
    }, [injectWebCommand]);
    const updatePlaybackTempo = (0, react_1.useCallback)((delta) => {
        const nextBpm = Math.max(40, Math.min(240, playbackBPM + delta));
        if (nextBpm === playbackBPM) {
            return;
        }
        setPlaybackBPM(nextBpm);
        injectWebCommand(`if (window.__OSMD_SET_BPM) window.__OSMD_SET_BPM(${JSON.stringify(nextBpm)});`, delta > 0 ? "increase-bpm" : "decrease-bpm");
    }, [injectWebCommand, playbackBPM]);
    (0, react_1.useEffect)(() => {
        if (detectedBPM && detectedBPM !== lastDetectedBPMRef.current) {
            lastDetectedBPMRef.current = detectedBPM;
            setPlaybackBPM(detectedBPM);
            if (webViewReadyRef.current) {
                injectWebCommand(`if (window.__OSMD_SET_BPM) window.__OSMD_SET_BPM(${JSON.stringify(detectedBPM)});`, "set-bpm-detected");
            }
        }
    }, [detectedBPM, injectWebCommand]);
    const onWebMessage = (0, react_1.useCallback)(async (e) => {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j;
        try {
            const msg = JSON.parse(e.nativeEvent.data);
            if (msg.type === "webview-click") {
                if (!isLandscape) {
                    try {
                        await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.LANDSCAPE);
                        setIsLandscape(true);
                    }
                    catch (err) {
                        console.warn("Orientation lock failed", err);
                    }
                }
                return;
            }
            if (msg.type === "scoreScrollActive") {
                updateScoreScrollActive(Boolean(msg.active));
                return;
            }
            if (msg.type === "ready") {
                webViewReadyRef.current = true;
                setWebViewReady(true);
                appendDebugEvent("webview ready");
                injectWebCommand(`
              if (window.__OSMD_SET_FOLLOW_TAIL) window.__OSMD_SET_FOLLOW_TAIL(${shouldFollowLatest ? "true" : "false"});
              if (window.__OSMD_SET_CAMERA_MODE) window.__OSMD_SET_CAMERA_MODE(${JSON.stringify(cameraMotionMode)});
              if (window.__OSMD_TOGGLE_CURSOR) window.__OSMD_TOGGLE_CURSOR(true);
              if (window.__OSMD_SET_BPM) window.__OSMD_SET_BPM(${JSON.stringify(playbackBPM !== null && playbackBPM !== void 0 ? playbackBPM : 120)});
            `, "sync-ready-state");
                requestDebugSnapshot("ready");
            }
            if (msg.type === "rendered") {
                if (pendingRenderIdRef.current !== null &&
                    msg.requestId !== pendingRenderIdRef.current) {
                    return;
                }
                if (renderProbeTimeoutRef.current) {
                    clearTimeout(renderProbeTimeoutRef.current);
                    renderProbeTimeoutRef.current = null;
                }
                if (typeof msg.measures === "number") {
                    measuresSentRef.current = msg.measures;
                    onScoreRendered === null || onScoreRendered === void 0 ? void 0 : onScoreRendered(msg.measures);
                    appendDebugEvent(`rendered request #${(_a = msg.requestId) !== null && _a !== void 0 ? _a : "?"} measures=${msg.measures}`);
                }
                if (pendingXmlRef.current) {
                    lastXmlRef.current = pendingXmlRef.current;
                    pendingXmlRef.current = null;
                    pendingSentAtRef.current = 0;
                    pendingRenderIdRef.current = null;
                }
                if (playAfterRenderRef.current && lastXmlRef.current) {
                    playAfterRenderRef.current = false;
                    appendDebugEvent("starting queued playback after render");
                    injectWebCommand(`if (window.__OSMD_PLAY) window.__OSMD_PLAY(${JSON.stringify(playbackBPM)});`, "play-score-after-render");
                }
            }
            if (msg.type === "appended") {
                if (typeof msg.appended === "number") {
                    measuresSentRef.current += msg.appended;
                }
            }
            if (msg.type === "error") {
                appendDebugEvent(`error ${String(msg.error).slice(0, 120)}`);
                console.warn("OSMD error:", msg.error);
            }
            if (msg.type === "debugState") {
                setDebugSnapshot((_b = msg.snapshot) !== null && _b !== void 0 ? _b : null);
                appendDebugEvent(`snapshot ${(_d = (_c = msg.snapshot) === null || _c === void 0 ? void 0 : _c.reason) !== null && _d !== void 0 ? _d : "unknown"} measures=${(_f = (_e = msg.snapshot) === null || _e === void 0 ? void 0 : _e.renderedMeasureCount) !== null && _f !== void 0 ? _f : "?"} svg=${(_h = (_g = msg.snapshot) === null || _g === void 0 ? void 0 : _g.stageSvgCount) !== null && _h !== void 0 ? _h : "?"}`);
            }
            if (msg.type === "playbackStarted") {
                setIsPlaying(true);
                setIsPaused(false);
                console.log("Playback started:", msg.noteCount, "notes,", msg.duration.toFixed(1), "seconds");
            }
            if (msg.type === "playbackPaused") {
                setIsPaused(true);
            }
            if (msg.type === "playbackResumed") {
                setIsPaused(false);
            }
            if (msg.type === "playbackStopped" || msg.type === "playbackEnded") {
                setIsPlaying(false);
                setIsPaused(false);
            }
            if (msg.type === "playbackError") {
                console.warn("Playback error:", msg.error);
                playAfterRenderRef.current = false;
                requestDebugSnapshot("playback-error");
                react_native_1.Alert.alert("Playback Error", String((_j = msg.error) !== null && _j !== void 0 ? _j : "Unknown error"));
                setIsPlaying(false);
                setIsPaused(false);
            }
            if (msg.type === "bpmSet") {
                setPlaybackBPM(msg.bpm);
            }
            if (msg.type === "exitFullscreen") {
                try {
                    await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT_UP);
                    setIsLandscape(false);
                }
                catch (err) {
                    console.warn("Exit fullscreen failed", err);
                }
            }
            if (msg.type === "bpmChanged") {
                setPlaybackBPM(msg.bpm);
            }
        }
        catch (err) {
            console.warn("webview message parse error", err);
        }
    }, [
        appendDebugEvent,
        injectWebCommand,
        isLandscape,
        onScoreRendered,
        cameraMotionMode,
        playbackBPM,
        requestDebugSnapshot,
        shouldFollowLatest,
        updateScoreScrollActive,
    ]);
    const onWebLoadStart = (0, react_1.useCallback)(() => {
        appendDebugEvent("webview load start");
        updateScoreScrollActive(false);
        setWebViewReady(false);
        webViewReadyRef.current = false;
        measuresSentRef.current = 0;
        lastXmlRef.current = null;
        pendingXmlRef.current = null;
        playAfterRenderRef.current = false;
        pendingSentAtRef.current = 0;
        pendingRenderIdRef.current = null;
        renderRefinementRef.current = undefined;
        setDebugSnapshot(null);
        if (renderProbeTimeoutRef.current) {
            clearTimeout(renderProbeTimeoutRef.current);
            renderProbeTimeoutRef.current = null;
        }
    }, [appendDebugEvent, updateScoreScrollActive]);
    const onWebLoadEnd = (0, react_1.useCallback)(() => {
        appendDebugEvent("webview load end");
        requestDebugSnapshot("load-end");
    }, [appendDebugEvent, requestDebugSnapshot]);
    const onWebError = (0, react_1.useCallback)((event) => {
        var _a, _b;
        appendDebugEvent(`webview error ${String((_b = (_a = event === null || event === void 0 ? void 0 : event.nativeEvent) === null || _a === void 0 ? void 0 : _a.description) !== null && _b !== void 0 ? _b : "unknown")}`);
        console.warn("[PianoSheetMusic] WebView error", event === null || event === void 0 ? void 0 : event.nativeEvent);
    }, [appendDebugEvent]);
    const onWebHttpError = (0, react_1.useCallback)((event) => {
        var _a, _b;
        appendDebugEvent(`webview http error ${String((_b = (_a = event === null || event === void 0 ? void 0 : event.nativeEvent) === null || _a === void 0 ? void 0 : _a.statusCode) !== null && _b !== void 0 ? _b : "unknown")}`);
        console.warn("[PianoSheetMusic] WebView HTTP error", event === null || event === void 0 ? void 0 : event.nativeEvent);
    }, [appendDebugEvent]);
    (0, react_1.useEffect)(() => {
        if (!webRef.current)
            return;
        if (!webViewReady)
            return;
        if (refinementVersion !== undefined &&
            refinementVersion !== renderRefinementRef.current) {
            renderRefinementRef.current = refinementVersion;
            measuresSentRef.current = 0;
            lastXmlRef.current = null;
            pendingXmlRef.current = null;
            pendingSentAtRef.current = 0;
            pendingRenderIdRef.current = null;
        }
        if (pendingXmlRef.current &&
            pendingSentAtRef.current > 0 &&
            Date.now() - pendingSentAtRef.current > 8000) {
            pendingXmlRef.current = null;
            pendingSentAtRef.current = 0;
            measuresSentRef.current = 0;
            lastXmlRef.current = null;
            pendingRenderIdRef.current = null;
        }
        const measures = generateMeasureXmls(accumulatedNotes, accumulatedChords, timeSignature, undefined, keySignature);
        const currentScoreUsesFallback = score === FALLBACK_XML;
        const pendingScoreUsesFallback = pendingXmlRef.current === FALLBACK_XML;
        const lastScoreUsesFallback = lastXmlRef.current === FALLBACK_XML;
        if (!currentScoreUsesFallback &&
            (pendingScoreUsesFallback || lastScoreUsesFallback)) {
            pendingXmlRef.current = null;
            pendingSentAtRef.current = 0;
            measuresSentRef.current = 0;
            lastXmlRef.current = null;
            pendingRenderIdRef.current = null;
        }
        if (measuresSentRef.current === 0 || !lastXmlRef.current) {
            try {
                sendRenderXml(score, "render-full-score", "if (window.__OSMD_TOGGLE_CURSOR) window.__OSMD_TOGGLE_CURSOR(true);");
            }
            catch (e) {
                console.warn("renderXml post failed", e);
            }
            return;
        }
        if (measures.length > measuresSentRef.current) {
            if (!lastXmlRef.current) {
                sendRenderXml(score, "render-full-score-retry");
                return;
            }
            const newMeasures = measures.slice(measuresSentRef.current);
            const existingCount = measuresSentRef.current;
            const adjusted = newMeasures.map((m) => {
                const noAttrs = m.replace(/<attributes>[\s\S]*?<\/attributes>/i, "");
                return noAttrs.replace(/number\s*=\s*"(\d+)"/i, function (_, p1) {
                    return 'number="' + (existingCount + parseInt(p1, 10)) + '"';
                });
            });
            let base = lastXmlRef.current || "";
            const closingPart = "</part>";
            let newXml;
            const idx = base.lastIndexOf(closingPart);
            if (idx !== -1) {
                newXml = base.slice(0, idx) + adjusted.join("") + base.slice(idx);
            }
            else {
                const closingScore = "</score-partwise>";
                const idx2 = base.lastIndexOf(closingScore);
                if (idx2 !== -1)
                    newXml = base.slice(0, idx2) + adjusted.join("") + base.slice(idx2);
                else
                    newXml = base + adjusted.join("");
            }
            try {
                sendRenderXml(newXml, "render-appended-score");
            }
            catch (e) {
                console.warn("Failed posting appended renderXml", e);
            }
        }
    }, [
        accumulatedNotes,
        accumulatedChords,
        score,
        timeSignature,
        keySignature,
        refinementVersion,
        injectWebCommand,
        sendRenderXml,
        webViewReady,
    ]);
    (0, react_1.useEffect)(() => {
        return () => {
            if (renderProbeTimeoutRef.current) {
                clearTimeout(renderProbeTimeoutRef.current);
            }
            ScreenOrientation.unlockAsync().catch(() => { });
        };
    }, []);
    (0, react_1.useEffect)(() => {
        injectWebCommand(`if (window.__OSMD_SET_FULLSCREEN) window.__OSMD_SET_FULLSCREEN(${isLandscape ? "true" : "false"});`, "set-fullscreen-mode");
    }, [injectWebCommand, isLandscape]);
    (0, react_1.useEffect)(() => {
        if (!webViewReady)
            return;
        injectWebCommand(`if (window.__OSMD_SET_FOLLOW_TAIL) window.__OSMD_SET_FOLLOW_TAIL(${shouldFollowLatest ? "true" : "false"});`, "set-follow-tail");
    }, [injectWebCommand, shouldFollowLatest, webViewReady]);
    (0, react_1.useEffect)(() => {
        if (!webViewReady)
            return;
        injectWebCommand(`if (window.__OSMD_SET_CAMERA_MODE) window.__OSMD_SET_CAMERA_MODE(${JSON.stringify(cameraMotionMode)});`, "set-camera-mode");
    }, [cameraMotionMode, injectWebCommand, webViewReady]);
    return (react_1.default.createElement(react_native_1.View, { style: [styles.container, compact ? styles.compactContainer : null] },
        react_1.default.createElement(react_native_1.View, { style: [
                styles.mainContainer,
                compact ? styles.compactMainContainer : null,
            ] },
            compact ? null : (react_1.default.createElement(react_native_1.View, { style: styles.playbackSection },
                react_1.default.createElement(react_native_1.View, { style: styles.playbackControls },
                    react_1.default.createElement(react_native_1.View, { style: styles.playbackButtonRow },
                        react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: [
                                styles.controlButton,
                                styles.playbackButton,
                                isPlaying && !isPaused
                                    ? styles.warningControlButton
                                    : styles.successControlButton,
                            ], onPress: handlePlayPause },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.controlButtonText }, isPlaying ? (isPaused ? "Resume" : "Pause") : "Play")),
                        react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: [
                                styles.controlButton,
                                styles.playbackButton,
                                styles.dangerControlButton,
                            ], onPress: handleStopPlayback },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.controlButtonText }, "Stop"))),
                    react_1.default.createElement(react_native_1.View, { style: styles.tempoControl },
                        react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: styles.stepperButton, onPress: () => updatePlaybackTempo(-10) },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.stepperButtonText }, "-")),
                        react_1.default.createElement(react_native_1.View, { style: styles.bpmDisplay },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.bpmValue }, playbackBPM),
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.bpmLabel }, "Tempo")),
                        react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: styles.stepperButton, onPress: () => updatePlaybackTempo(10) },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.stepperButtonText }, "+")))))),
            react_1.default.createElement(react_native_1.View, { style: [
                    styles.scoreSection,
                    compact
                        ? {
                            flex: 1,
                            minHeight: compactViewportHeight,
                            marginBottom: 0,
                            borderWidth: 0,
                            borderRadius: 0,
                        }
                        : null,
                ] },
                react_1.default.createElement(react_native_1.View, { style: [
                        styles.webviewFrame,
                        compact
                            ? {
                                flex: 1,
                                minHeight: compactViewportHeight,
                            }
                            : null,
                    ] },
                    react_1.default.createElement(react_native_webview_1.WebView, { ref: webRef, originWhitelist: ["*"], source: source, onMessage: onWebMessage, onLoadStart: onWebLoadStart, onLoadEnd: onWebLoadEnd, onError: onWebError, onHttpError: onWebHttpError, javaScriptEnabled: true, allowFileAccess: true, allowUniversalAccessFromFileURLs: true, mixedContentMode: "always", style: [
                            isLandscape ? styles.landscapeWebview : styles.webview,
                            compact
                                ? {
                                    flex: 1,
                                    height: undefined,
                                }
                                : null,
                        ], nestedScrollEnabled: true, scrollEnabled: true }),
                    compact && !isLandscape ? (react_1.default.createElement(react_native_1.View, { pointerEvents: "box-none", style: styles.compactPlaybackOverlay },
                        react_1.default.createElement(react_native_1.View, { style: styles.compactPlaybackBar },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.compactPlaybackBpm },
                                playbackBPM,
                                " BPM"),
                            react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: [
                                    styles.compactPlaybackButton,
                                    isPlaying && !isPaused
                                        ? styles.compactPlaybackButtonActive
                                        : styles.compactPlaybackButtonPrimary,
                                    !hasPlayableScore && !isPlaying && !isPaused
                                        ? styles.compactPlaybackButtonDisabled
                                        : null,
                                ], onPress: handlePlayPause, disabled: !hasPlayableScore && !isPlaying && !isPaused },
                                react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.compactPlaybackButtonText }, isPlaying ? (isPaused ? "Resume" : "Pause") : "Play")),
                            react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: [
                                    styles.compactPlaybackButton,
                                    styles.compactPlaybackButtonSecondary,
                                    !isPlaying && !isPaused
                                        ? styles.compactPlaybackButtonDisabled
                                        : null,
                                ], onPress: handleStopPlayback, disabled: !isPlaying && !isPaused },
                                react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.compactPlaybackButtonText }, "Stop"))))) : null)),
            compact ? null : (react_1.default.createElement(react_native_1.View, { style: styles.viewControlsSection },
                react_1.default.createElement(react_native_1.View, { style: styles.viewControls },
                    react_1.default.createElement(react_native_1.View, { style: styles.cameraModeGroup },
                        react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.controlGroupLabel }, "Camera"),
                        react_1.default.createElement(react_native_1.View, { style: styles.cameraModeChips }, ["smooth", "snap"].map((mode) => (react_1.default.createElement(react_native_1.TouchableOpacity, { key: mode, activeOpacity: 0.85, style: [
                                styles.cameraChip,
                                cameraMotionMode === mode && styles.cameraChipActive,
                            ], onPress: () => setCameraMotionMode(mode) },
                            react_1.default.createElement(ThemedText_1.ThemedText, { style: [
                                    styles.cameraChipText,
                                    cameraMotionMode === mode &&
                                        styles.cameraChipTextActive,
                                ] }, mode === "smooth" ? "Smooth" : "Reposition")))))),
                    react_1.default.createElement(react_native_1.TouchableOpacity, { activeOpacity: 0.85, style: [
                            styles.controlButton,
                            styles.clearButton,
                            styles.ghostControlButton,
                        ], onPress: () => {
                            setAccumulatedNotes([]);
                            setAccumulatedChords([]);
                            lastXmlRef.current = null;
                            measuresSentRef.current = 0;
                            pendingXmlRef.current = null;
                            playAfterRenderRef.current = false;
                            pendingSentAtRef.current = 0;
                            pendingRenderIdRef.current = null;
                            sendRenderXml(FALLBACK_XML, "clear-score", "if (window.__OSMD_STOP) window.__OSMD_STOP();");
                        } },
                        react_1.default.createElement(ThemedText_1.ThemedText, { style: styles.ghostControlButtonText }, "Clear"))))))));
}
const styles = react_native_1.StyleSheet.create({
    container: {
        paddingVertical: 10,
        paddingHorizontal: 8,
        backgroundColor: "rgba(255,255,255,0.95)",
        borderRadius: 12,
        marginBottom: 20,
        width: "100%",
        overflow: "hidden",
    },
    compactContainer: {
        flex: 1,
        paddingVertical: 0,
        paddingHorizontal: 0,
        marginBottom: 0,
        backgroundColor: "transparent",
        borderRadius: 0,
    },
    mainContainer: {
        width: "100%",
        maxWidth: "100%",
    },
    compactMainContainer: {
        flex: 1,
        minHeight: 0,
    },
    title: {
        textAlign: "center",
        marginBottom: 12,
        color: "#333",
        fontSize: 18,
        fontWeight: "bold",
    },
    playbackSection: {
        backgroundColor: "#f8fafc",
        borderRadius: 12,
        padding: 12,
        marginBottom: 12,
        borderWidth: 1,
        borderColor: "#d8e1ea",
    },
    sectionLabel: {
        fontSize: 14,
        fontWeight: "600",
        color: "#666",
        marginBottom: 8,
        textAlign: "center",
    },
    playbackControls: {
        width: "100%",
        alignItems: "stretch",
        gap: 10,
    },
    playbackButtonRow: {
        flexDirection: "row",
        flexWrap: "wrap",
        width: "100%",
        gap: 10,
    },
    controlButton: {
        minHeight: 46,
        borderRadius: 12,
        borderWidth: 1,
        justifyContent: "center",
        alignItems: "center",
        paddingHorizontal: 16,
        paddingVertical: 12,
        maxWidth: "100%",
    },
    playbackButton: {
        flexGrow: 1,
        flexBasis: 0,
        minWidth: 124,
    },
    successControlButton: {
        backgroundColor: "#1f8f5f",
        borderColor: "#1f8f5f",
    },
    warningControlButton: {
        backgroundColor: "#c56a1f",
        borderColor: "#c56a1f",
    },
    dangerControlButton: {
        backgroundColor: "#c24135",
        borderColor: "#c24135",
    },
    ghostControlButton: {
        backgroundColor: "#ffffff",
        borderColor: "#cbd5e1",
    },
    controlButtonText: {
        fontSize: 14,
        fontWeight: "700",
        color: "#ffffff",
    },
    ghostControlButtonText: {
        fontSize: 14,
        fontWeight: "700",
        color: "#52606d",
    },
    tempoControl: {
        flexDirection: "row",
        alignItems: "center",
        width: "100%",
        backgroundColor: "#fff",
        borderRadius: 12,
        padding: 6,
        borderWidth: 1,
        borderColor: "#d8e1ea",
        gap: 8,
    },
    stepperButton: {
        width: 44,
        minHeight: 44,
        borderRadius: 10,
        backgroundColor: "#e8eef5",
        justifyContent: "center",
        alignItems: "center",
        flexShrink: 0,
    },
    stepperButtonText: {
        fontSize: 22,
        lineHeight: 24,
        fontWeight: "700",
        color: "#334155",
    },
    bpmDisplay: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
        paddingHorizontal: 8,
    },
    bpmValue: {
        fontSize: 19,
        fontWeight: "bold",
        color: "#1f2937",
    },
    bpmLabel: {
        fontSize: 11,
        fontWeight: "600",
        color: "#64748b",
        textTransform: "uppercase",
        letterSpacing: 0.6,
    },
    scoreSection: {
        flex: 1,
        minHeight: 560,
        backgroundColor: "#fff",
        borderRadius: 12,
        overflow: "hidden",
        marginBottom: 12,
        borderWidth: 1,
        borderColor: "#d8e1ea",
    },
    webviewFrame: {
        width: "100%",
        minHeight: 560,
        maxWidth: "100%",
        overflow: "hidden",
    },
    compactPlaybackOverlay: {
        position: "absolute",
        top: 12,
        right: 12,
        zIndex: 3,
        elevation: 6,
    },
    compactPlaybackBar: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        paddingHorizontal: 8,
        paddingVertical: 8,
        borderRadius: 999,
        backgroundColor: "rgba(15,23,42,0.78)",
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.16)",
    },
    compactPlaybackBpm: {
        fontSize: 10,
        fontWeight: "700",
        color: "rgba(226,232,240,0.88)",
        textTransform: "uppercase",
        letterSpacing: 0.8,
    },
    compactPlaybackButton: {
        minHeight: 32,
        borderRadius: 999,
        paddingHorizontal: 12,
        alignItems: "center",
        justifyContent: "center",
    },
    compactPlaybackButtonPrimary: {
        backgroundColor: "#0f766e",
    },
    compactPlaybackButtonActive: {
        backgroundColor: "#c56a1f",
    },
    compactPlaybackButtonSecondary: {
        backgroundColor: "rgba(255,255,255,0.16)",
    },
    compactPlaybackButtonDisabled: {
        opacity: 0.48,
    },
    compactPlaybackButtonText: {
        fontSize: 12,
        fontWeight: "700",
        color: "#ffffff",
    },
    debugSection: {
        backgroundColor: "#f7f9fb",
        borderRadius: 10,
        padding: 10,
        marginBottom: 12,
        borderWidth: 1,
        borderColor: "#d7dde3",
    },
    debugHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 6,
    },
    debugTitle: {
        fontSize: 13,
        fontWeight: "700",
        color: "#37474f",
    },
    debugButtonRow: {
        minWidth: 72,
    },
    debugSummary: {
        fontSize: 11,
        color: "#455a64",
        marginBottom: 2,
    },
    debugLog: {
        marginTop: 6,
        borderTopWidth: 1,
        borderTopColor: "#e0e6eb",
        paddingTop: 6,
    },
    debugLogLine: {
        fontSize: 11,
        color: "#607d8b",
        marginBottom: 2,
    },
    webview: {
        height: 560,
        borderRadius: 8,
        width: "100%",
        overflow: "hidden",
        backgroundColor: "#fff",
    },
    landscapeWebview: {
        height: 400,
        borderRadius: 0,
        width: "100%",
        overflow: "hidden",
        backgroundColor: "#fff",
    },
    viewControlsSection: {
        backgroundColor: "#f8fafc",
        borderRadius: 12,
        padding: 12,
        borderWidth: 1,
        borderColor: "#d8e1ea",
    },
    viewControls: {
        flexDirection: "row",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 10,
    },
    cameraModeGroup: {
        flexGrow: 1,
        flexShrink: 1,
        flexBasis: 220,
        maxWidth: "100%",
        gap: 8,
    },
    controlGroupLabel: {
        fontSize: 12,
        fontWeight: "700",
        color: "#52606d",
        textTransform: "uppercase",
        letterSpacing: 0.5,
    },
    cameraModeChips: {
        flexDirection: "row",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 8,
        maxWidth: "100%",
    },
    cameraChip: {
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: "#cbd5e1",
        backgroundColor: "#ffffff",
    },
    cameraChipActive: {
        backgroundColor: "#1f2937",
        borderColor: "#1f2937",
    },
    cameraChipText: {
        fontSize: 13,
        fontWeight: "600",
        color: "#475569",
    },
    cameraChipTextActive: {
        color: "#ffffff",
    },
    clearButton: {
        flexGrow: 1,
        flexBasis: 108,
        minWidth: 108,
    },
    toolbar: {
        flexDirection: "row",
        justifyContent: "space-between",
        gap: 8,
        marginBottom: 8,
    },
    playbackToolbar: {
        flexDirection: "row",
        justifyContent: "center",
        alignItems: "center",
        gap: 8,
        marginBottom: 8,
        paddingVertical: 4,
        backgroundColor: "rgba(0,0,0,0.05)",
        borderRadius: 8,
    },
    bpmText: {
        fontSize: 16,
        fontWeight: "bold",
        color: "#333",
        minWidth: 40,
        textAlign: "center",
    },
});
