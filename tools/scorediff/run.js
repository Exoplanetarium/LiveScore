#!/usr/bin/env node
/*
 * scorediff — compare the rendered score (generateMusicXML) against the raw
 * MIDI events, both derived from the SAME app payload.
 *
 * The app builds the on-screen score from generateMusicXML(notes, chords,
 * "4/4", bpm, 0) and the exported MIDI directly from the same notes/chords at
 * their raw time_seconds. Any pitch that the score adds, drops, or shifts to a
 * different beat relative to the raw events is therefore caused entirely by the
 * grouping/quantization inside generateMeasureXmls. This harness runs the REAL
 * production function (compiled with the project's own tsc, React Native imports
 * stubbed at load — zero logic drift) and reports those differences per onset.
 *
 * Usage:
 *   node tools/scorediff/run.js backend/_tmp_app_payload_clip_017.json [...more]
 *   node tools/scorediff/run.js --ref reference.musicxml backend/_tmp_app_payload_clip_017.json
 *   node tools/scorediff/run.js --ref-dir references/ backend/_tmp_app_payload_clip_017.json [...more]
 *   node tools/scorediff/run.js --xml-pair predicted.musicxml reference.musicxml
 *   node tools/scorediff/run.js --ref-dir references/ --json-out score_metrics.json backend/_tmp_app_payload_*.json
 *   node tools/scorediff/run.js            # globs backend/_tmp_app_payload_*.json
 *   node tools/scorediff/run.js --self-test
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const HERE = __dirname;
const REPO = path.resolve(HERE, "..", "..");
const BUILD_OUT = path.join(HERE, "build", "components", "PianoSheetMusic.js");

// ── 1. React Native / Expo stubs so the pure XML function can load in Node ──
function installRequireStubs() {
  const Module = require("module");
  const origLoad = Module._load;

  // A proxy that survives `StyleSheet.create({...})` at module scope and any
  // other member access from the (never-invoked) React component body.
  const rnProxy = new Proxy(function () {}, {
    get(_t, prop) {
      if (prop === "StyleSheet") return { create: (s) => s };
      if (prop === "__esModule") return true;
      if (prop === "default") return rnProxy;
      return rnProxy;
    },
    apply() {
      return rnProxy;
    },
  });

  const STUBS = {
    "react-native": rnProxy,
    "react-native-webview": rnProxy,
    "expo-screen-orientation": rnProxy,
    "./ThemedText": { ThemedText: () => null },
    "./osmdHTML": { OSMD_HTML: "" },
  };

  Module._load = function (request, parent, isMain) {
    if (Object.prototype.hasOwnProperty.call(STUBS, request)) {
      return STUBS[request];
    }
    return origLoad.apply(this, arguments);
  };
}

// ── 2. Compile the real component with the project's tsc, then load it ──
function loadGenerateMusicXML() {
  let tscJs;
  try {
    tscJs = path.join(path.dirname(require.resolve("typescript", { paths: [REPO] })), "tsc.js");
  } catch (e) {
    throw new Error("Could not resolve the project's 'typescript' devDependency. Run `npm install` first.\n" + e.message);
  }
  process.stderr.write("[scorediff] compiling components/PianoSheetMusic.tsx ...\n");
  try {
    execFileSync(process.execPath, [tscJs, "-p", path.join(HERE, "tsconfig.json")], {
      cwd: HERE,
      stdio: ["ignore", "ignore", "pipe"],
    });
  } catch (e) {
    // tsc exits non-zero on type errors; noEmitOnError:false still emits JS.
    // Only fail if the output file is missing.
    if (!fs.existsSync(BUILD_OUT)) {
      throw new Error("tsc did not emit " + BUILD_OUT + "\n" + (e.stderr ? e.stderr.toString() : e.message));
    }
  }
  if (!fs.existsSync(BUILD_OUT)) {
    throw new Error("Compiled output missing: " + BUILD_OUT);
  }

  installRequireStubs();
  const mod = require(BUILD_OUT);
  const fn = mod.generateMusicXML || (mod.default && mod.default.generateMusicXML);
  if (typeof fn !== "function") {
    throw new Error("generateMusicXML export not found in compiled module.");
  }
  return fn;
}

// ── 3. MusicXML walker: pitch + absolute onset (in divisions) per note ──
const STEP_SEMITONE = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const CANON_DIVISIONS = 24;

function pitchToMidi(step, alter, octave) {
  const base = STEP_SEMITONE[step];
  if (base === undefined) return null;
  return (octave + 1) * 12 + base + (alter || 0);
}

function readTag(body, tag) {
  const m = body.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`));
  return m ? m[1].trim() : null;
}

function readIntTag(body, tag, fallback = null) {
  const value = readTag(body, tag);
  if (value == null) return fallback;
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizePitchSet(pitches) {
  return [...new Set(pitches)].sort((a, b) => a - b);
}

// Returns one canonical event per sounding note, with absolute quarter-note
// position plus legacy divAbs/beatAbs fields for the old raw diff path.
function parseScoreXml(xml) {
  const out = [];
  const parts = xml.match(/<part\b[\s\S]*?<\/part>/g) || [xml];
  for (let partIndex = 0; partIndex < parts.length; partIndex++) {
    const measures = parts[partIndex].split(/<measure\b/).slice(1);
    let measureIndex = 0;
    let measureBaseQuarter = 0;
    let divisions = CANON_DIVISIONS;
    let beats = 4;
    let beatType = 4;

    for (const rawMeasure of measures) {
      const attrs = rawMeasure.match(/<attributes\b[\s\S]*?<\/attributes>/);
      if (attrs) {
        divisions = readIntTag(attrs[0], "divisions", divisions) || divisions;
        beats = readIntTag(attrs[0], "beats", beats) || beats;
        beatType = readIntTag(attrs[0], "beat-type", beatType) || beatType;
      }

      let cursorQuarter = 0;
      let lastOnsetQuarter = 0;
      const measureLenQuarter = (beats * 4) / beatType;

      const tokenRe = /<(note|backup|forward)\b[\s\S]*?<\/\1>/g;
      let m;
      while ((m = tokenRe.exec(rawMeasure)) !== null) {
        const tag = m[1];
        const body = m[0];
        const durationDivs = readIntTag(body, "duration", 0) || 0;
        const durationQuarter = durationDivs / divisions;

        if (tag === "backup") {
          cursorQuarter -= durationQuarter;
          continue;
        }
        if (tag === "forward") {
          cursorQuarter += durationQuarter;
          continue;
        }

        const isChord = /<chord\s*\/>/.test(body);
        const isRest = /<rest\b/.test(body);
        const onsetQuarter = isChord ? lastOnsetQuarter : cursorQuarter;
        const staff = readIntTag(body, "staff", 1) || 1;
        const voice = readTag(body, "voice") || "1";

        if (!isRest) {
          const p = body.match(/<step>([A-G])<\/step>(?:\s*<alter>(-?\d+)<\/alter>)?\s*<octave>(-?\d+)<\/octave>/);
          if (p) {
            const midi = pitchToMidi(p[1], p[2] ? parseInt(p[2], 10) : 0, parseInt(p[3], 10));
            if (midi != null) {
              const quarterAbs = measureBaseQuarter + onsetQuarter;
              out.push({
                midi,
                quarterAbs,
                durationQuarter,
                divAbs: Math.round(quarterAbs * CANON_DIVISIONS),
                durationDivs: Math.round(durationQuarter * CANON_DIVISIONS),
                beatAbs: quarterAbs,
                staff,
                voice,
                part: partIndex + 1,
                measure: measureIndex + 1,
                localQuarter: onsetQuarter,
              });
            }
          }
        }

        if (!isChord) {
          lastOnsetQuarter = cursorQuarter;
          cursorQuarter += durationQuarter;
        }
      }
      measureBaseQuarter += measureLenQuarter;
      measureIndex += 1;
    }
  }
  return { notes: out, divisions: CANON_DIVISIONS };
}

// ── 4. Build the raw (MIDI-equivalent) event list from the app payload ──
function rawEventsFromPayload(payload) {
  const events = [];
  for (const n of payload.notes || []) {
    if (typeof n.midi_note !== "number") continue;
    events.push({ t: n.time_seconds || 0, midi: n.midi_note });
  }
  for (const c of payload.chords || []) {
    const t = c.time_seconds || 0;
    for (const midi of c.midi_notes || []) {
      if (typeof midi === "number") events.push({ t, midi });
    }
  }
  events.sort((a, b) => a.t - b.t || a.midi - b.midi);
  return events;
}

// Cluster a list of {t, midi} into onset groups within `tol` seconds.
function clusterByTime(events, tol) {
  const clusters = [];
  for (const ev of events) {
    const last = clusters[clusters.length - 1];
    if (last && Math.abs(ev.t - last.t) <= tol) {
      last.pitches.add(ev.midi);
      last.tSum += ev.t;
      last.n += 1;
      last.t = last.tSum / last.n;
    } else {
      clusters.push({ t: ev.t, tSum: ev.t, n: 1, pitches: new Set([ev.midi]) });
    }
  }
  return clusters;
}

// ── 5. Diff one payload: align raw clusters to printed (XML) clusters ──
function diffPayload(payload, generateMusicXML) {
  const bpm = payload.bpm && payload.bpm > 1 ? payload.bpm : 120;
  const xml = generateMusicXML(payload.notes || [], payload.chords || [], "4/4", bpm, 0);
  const { notes: xmlNotes } = parseScoreXml(xml);

  const secPerQuarter = 60 / bpm;
  // XML clusters: notes sharing the same absolute division = one printed onset.
  const byDiv = new Map();
  for (const n of xmlNotes) {
    if (!byDiv.has(n.divAbs)) byDiv.set(n.divAbs, new Set());
    byDiv.get(n.divAbs).add(n.midi);
  }
  const xmlClusters = [...byDiv.entries()]
    .map(([divAbs, pitches]) => ({ t: (divAbs / 24) * secPerQuarter, pitches }))
    .sort((a, b) => a.t - b.t);

  const rawClusters = clusterByTime(rawEventsFromPayload(payload), 0.03);

  // Pitch-aware alignment: among printed onsets within the time window, prefer
  // the one that shares the most pitches with the raw onset, breaking ties by
  // nearest time. This keeps rows musically comparable instead of pairing, say,
  // a raw G chord against an unrelated printed E that merely happens to be close.
  const matchTol = Math.max(0.08, secPerQuarter / 2);
  const usedXml = new Set();
  const rows = [];
  for (const rc of rawClusters) {
    let best = -1;
    let bestScore = -1;
    let bestDt = Infinity;
    for (let i = 0; i < xmlClusters.length; i++) {
      if (usedXml.has(i)) continue;
      const dt = Math.abs(xmlClusters[i].t - rc.t);
      if (dt > matchTol) continue;
      let overlap = 0;
      for (const p of xmlClusters[i].pitches) if (rc.pitches.has(p)) overlap += 1;
      if (overlap > bestScore || (overlap === bestScore && dt < bestDt)) {
        bestScore = overlap;
        bestDt = dt;
        best = i;
      }
    }
    // Require at least one shared pitch to treat it as the same onset; a
    // zero-overlap "match" is really a raw onset whose pitches were scattered
    // elsewhere, so report it as absent rather than as a confusing swap.
    if (best >= 0 && bestScore >= 1) {
      usedXml.add(best);
      const xc = xmlClusters[best];
      const dropped = [...rc.pitches].filter((p) => !xc.pitches.has(p));
      const added = [...xc.pitches].filter((p) => !rc.pitches.has(p));
      rows.push({ kind: "matched", rawT: rc.t, xmlT: xc.t, raw: [...rc.pitches], xml: [...xc.pitches], dropped, added });
    } else {
      rows.push({ kind: "raw_only", rawT: rc.t, raw: [...rc.pitches], xml: [], dropped: [...rc.pitches], added: [] });
    }
  }
  for (let i = 0; i < xmlClusters.length; i++) {
    if (!usedXml.has(i)) {
      const xc = xmlClusters[i];
      rows.push({ kind: "xml_only", xmlT: xc.t, raw: [], xml: [...xc.pitches], dropped: [], added: [...xc.pitches] });
    }
  }
  rows.sort((a, b) => (a.rawT ?? a.xmlT) - (b.rawT ?? b.xmlT));

  // Tag dropped pitches that resurface in a LATER printed cluster (the
  // "appears later in the MIDI" symptom) vs. genuinely missing.
  for (const row of rows) {
    if (!row.dropped.length) continue;
    row.shiftedLater = row.dropped.filter((p) =>
      xmlClusters.some((xc) => xc.t > (row.rawT ?? row.xmlT) + 1e-6 && xc.pitches.has(p))
    );
  }

  const totals = rows.reduce(
    (acc, r) => {
      acc.dropped += r.dropped.length;
      acc.added += r.added.length;
      acc.shifted += (r.shiftedLater || []).length;
      if (r.kind === "raw_only") acc.rawOnly += 1;
      if (r.kind === "xml_only") acc.xmlOnly += 1;
      return acc;
    },
    { dropped: 0, added: 0, shifted: 0, rawOnly: 0, xmlOnly: 0 }
  );

  return { bpm, rows, totals, rawClusterCount: rawClusters.length, xmlClusterCount: xmlClusters.length };
}

// ── 6. Reporting ──
function tokenKey(token, includeVoice = true) {
  const voicePart = includeVoice ? `|v${token.voice}` : "";
  return `${token.q}|p${token.part}|s${token.staff}${voicePart}|d${token.durationDivs}|${token.pitches.join(",")}`;
}

function buildScoreTokens(xml) {
  const { notes } = parseScoreXml(xml);
  const bySlot = new Map();
  for (const n of notes) {
    const q = Math.round(n.quarterAbs * CANON_DIVISIONS);
    const durationDivs = Math.max(1, Math.round(n.durationQuarter * CANON_DIVISIONS));
    const key = `${q}|${n.part}|${n.staff}|${n.voice}`;
    if (!bySlot.has(key)) {
      bySlot.set(key, {
        q,
        quarterAbs: q / CANON_DIVISIONS,
        part: n.part,
        staff: n.staff,
        voice: n.voice,
        pitches: [],
        durationDivs,
      });
    }
    const token = bySlot.get(key);
    token.pitches.push(n.midi);
    token.durationDivs = Math.max(token.durationDivs, durationDivs);
  }
  return [...bySlot.values()]
    .map((t) => ({ ...t, pitches: normalizePitchSet(t.pitches) }))
    .sort((a, b) => a.q - b.q || a.part - b.part || a.staff - b.staff || String(a.voice).localeCompare(String(b.voice)) || a.pitches[0] - b.pitches[0]);
}

function multisetExactF1(predTokens, refTokens) {
  const refCounts = new Map();
  for (const token of refTokens) {
    const key = tokenKey(token);
    refCounts.set(key, (refCounts.get(key) || 0) + 1);
  }
  let matched = 0;
  for (const token of predTokens) {
    const key = tokenKey(token);
    const count = refCounts.get(key) || 0;
    if (count > 0) {
      matched += 1;
      refCounts.set(key, count - 1);
    }
  }
  const precision = predTokens.length ? matched / predTokens.length : 0;
  const recall = refTokens.length ? matched / refTokens.length : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { matched, predicted: predTokens.length, reference: refTokens.length, precision, recall, f1 };
}

function pitchJaccard(a, b) {
  const aa = new Set(a);
  const bb = new Set(b);
  let intersection = 0;
  for (const p of aa) if (bb.has(p)) intersection += 1;
  const union = new Set([...a, ...b]).size;
  return union ? intersection / union : 1;
}

function tokenInsDelCost(token) {
  return 1.0 + Math.min(0.35, Math.max(0, token.pitches.length - 1) * 0.08);
}

// Cost model: pitch and duration are weighted equally; onset, staff, and voice
// are excluded. Onset shifts cascade from duration errors and would double-penalise
// a single upstream mistake; staff/voice are notation preferences, not musical content.
// Duration earns partial credit inside a free band (≤1/3 off is free).
// All knobs are env-overridable (SCOREDIFF_*) so the model can be retuned without code changes.
const envNum = (name, fallback) => {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? v : fallback;
};
const PITCH_WEIGHT = envNum("SCOREDIFF_PITCH_WEIGHT", 1.0);
const DURATION_WEIGHT = envNum("SCOREDIFF_DURATION_WEIGHT", 1.0); // equal to pitch: wrong length is as bad as wrong note
const ONSET_WEIGHT = envNum("SCOREDIFF_ONSET_WEIGHT", 0.0); // 0: onset shifts cascade from duration errors, double-penalising
const STAFF_WEIGHT = envNum("SCOREDIFF_STAFF_WEIGHT", 0.0); // 0: staff assignment is a notation preference, not musical content
const VOICE_WEIGHT = envNum("SCOREDIFF_VOICE_WEIGHT", 0.0); // 0: voice assignment is a notation preference, not musical content
// "Close enough" bands earn full credit (zero cost) before the linear ramp:
const DURATION_FREE_RATIO = envNum("SCOREDIFF_DURATION_FREE_RATIO", 0.34); // ≤ ~1/3 off the note value is free
const DURATION_FULL_RATIO = envNum("SCOREDIFF_DURATION_FULL_RATIO", 1.0); // off by ≥ the whole value ⇒ max duration cost

// Zero cost up to `free`, then ramps linearly to 1 at `full` (same units).
function bandedCost(error, free, full) {
  if (error <= free) return 0;
  if (error >= full) return 1;
  return (error - free) / (full - free);
}

function tokenSubstitutionCost(pred, ref) {
  const pitchCost = PITCH_WEIGHT * (1.0 - pitchJaccard(pred.pitches, ref.pitches));
  const durationError =
    Math.abs(pred.durationDivs - ref.durationDivs) / Math.max(pred.durationDivs, ref.durationDivs, 1);
  const durationCost = DURATION_WEIGHT * bandedCost(durationError, DURATION_FREE_RATIO, DURATION_FULL_RATIO);
  const cost = pitchCost + durationCost;
  return Math.min(cost, tokenInsDelCost(pred) + tokenInsDelCost(ref));
}

function rhythmicEditDistance(predTokens, refTokens) {
  const n = refTokens.length;
  const m = predTokens.length;
  const width = m + 1;
  const size = (n + 1) * (m + 1);
  const dp = new Float64Array(size);
  const op = new Uint8Array(size);

  for (let i = 1; i <= n; i++) {
    const idx = i * width;
    dp[idx] = dp[(i - 1) * width] + tokenInsDelCost(refTokens[i - 1]);
    op[idx] = 1;
  }
  for (let j = 1; j <= m; j++) {
    dp[j] = dp[j - 1] + tokenInsDelCost(predTokens[j - 1]);
    op[j] = 2;
  }

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const idx = i * width + j;
      const sub = dp[(i - 1) * width + (j - 1)] + tokenSubstitutionCost(predTokens[j - 1], refTokens[i - 1]);
      const del = dp[(i - 1) * width + j] + tokenInsDelCost(refTokens[i - 1]);
      const ins = dp[i * width + (j - 1)] + tokenInsDelCost(predTokens[j - 1]);
      if (sub <= del && sub <= ins) {
        dp[idx] = sub;
        op[idx] = 0;
      } else if (del <= ins) {
        dp[idx] = del;
        op[idx] = 1;
      } else {
        dp[idx] = ins;
        op[idx] = 2;
      }
    }
  }

  let i = n;
  let j = m;
  const counts = { exact: 0, substitutions: 0, deletions: 0, insertions: 0 };
  const examples = [];
  while (i > 0 || j > 0) {
    const choice = op[i * width + j];
    if (i > 0 && j > 0 && choice === 0) {
      const cost = tokenSubstitutionCost(predTokens[j - 1], refTokens[i - 1]);
      if (cost <= 1e-9) {
        counts.exact += 1;
      } else {
        counts.substitutions += 1;
        if (examples.length < 8) examples.push({ kind: "substitute", ref: refTokens[i - 1], pred: predTokens[j - 1], cost });
      }
      i -= 1;
      j -= 1;
    } else if (i > 0 && (j === 0 || choice === 1)) {
      counts.deletions += 1;
      if (examples.length < 8) examples.push({ kind: "delete", ref: refTokens[i - 1], cost: tokenInsDelCost(refTokens[i - 1]) });
      i -= 1;
    } else {
      counts.insertions += 1;
      if (examples.length < 8) examples.push({ kind: "insert", pred: predTokens[j - 1], cost: tokenInsDelCost(predTokens[j - 1]) });
      j -= 1;
    }
  }

  const referenceCost = refTokens.reduce((total, token) => total + tokenInsDelCost(token), 0);
  const editCost = dp[n * width + m];
  return {
    editCost,
    referenceCost,
    normalizedAccuracy: referenceCost ? Math.max(0, 1 - editCost / referenceCost) : m === 0 ? 1 : 0,
    counts,
    examples: examples.reverse(),
  };
}

// Forgive over-segmentation: collapse consecutive same-voice tokens that carry
// the IDENTICAL pitch set and are temporally contiguous — a held note/chord the
// renderer split into fragments or tied across a barline. Applied symmetrically
// to both streams, so a genuine rearticulation is only "forgiven" when one side
// splits and the other doesn't (exactly the tie-vs-repeat ambiguity a human
// reading the score also collapses). Env-gated (default on); set
// SCOREDIFF_MERGE_OVERSEG=0 to restore strict 1-token-per-slot scoring.
const MERGE_OVERSEG = process.env.SCOREDIFF_MERGE_OVERSEG !== "0";
const MERGE_GAP_DIVS = envNum("SCOREDIFF_MERGE_GAP_DIVS", CANON_DIVISIONS / 8); // ≤ a 32nd-note gap still counts as "held"

function pitchSetEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function mergeOverSegmented(tokens) {
  if (!MERGE_OVERSEG || tokens.length < 2) return tokens;
  const groups = new Map();
  for (const t of tokens) {
    const k = `${t.part}|${t.staff}|${t.voice}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(t);
  }
  const out = [];
  for (const group of groups.values()) {
    group.sort((a, b) => a.q - b.q);
    let cur = null;
    for (const t of group) {
      if (cur && pitchSetEqual(cur.pitches, t.pitches) && t.q <= cur.q + cur.durationDivs + MERGE_GAP_DIVS) {
        cur.durationDivs = Math.max(cur.q + cur.durationDivs, t.q + t.durationDivs) - cur.q; // extend held span
      } else {
        cur = { ...t };
        out.push(cur);
      }
    }
  }
  return out.sort(
    (a, b) =>
      a.q - b.q ||
      a.part - b.part ||
      a.staff - b.staff ||
      String(a.voice).localeCompare(String(b.voice)) ||
      a.pitches[0] - b.pitches[0]
  );
}

function compareScoreXml(predXml, refXml) {
  const predTokens = mergeOverSegmented(buildScoreTokens(predXml));
  const refTokens = mergeOverSegmented(buildScoreTokens(refXml));
  return {
    predTokens,
    refTokens,
    exactTokenF1: multisetExactF1(predTokens, refTokens),
    rhythmicEdit: rhythmicEditDistance(predTokens, refTokens),
  };
}

function compactScoreComparison(label, refPath, comparison) {
  const exact = comparison.exactTokenF1;
  const edit = comparison.rhythmicEdit;
  return {
    label,
    reference_path: refPath || null,
    predicted_tokens: comparison.predTokens.length,
    reference_tokens: comparison.refTokens.length,
    exact_token_matched: exact.matched,
    exact_token_precision: exact.precision,
    exact_token_recall: exact.recall,
    exact_token_f1: exact.f1,
    score_edit_accuracy: edit.normalizedAccuracy,
    score_edit_cost: edit.editCost,
    score_reference_cost: edit.referenceCost,
    score_edit_exact_ops: edit.counts.exact,
    score_edit_substitutions: edit.counts.substitutions,
    score_edit_deletions: edit.counts.deletions,
    score_edit_insertions: edit.counts.insertions,
  };
}

const noteName = (m) => {
  const N = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return N[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
};
const names = (arr) => "[" + arr.map(noteName).join(" ") + "]";

function report(label, result, limit) {
  const { totals, rows } = result;
  console.log("\n========================================================");
  console.log(`CLIP ${label}   bpm=${result.bpm.toFixed(1)}`);
  console.log(
    `  raw onsets=${result.rawClusterCount}  printed onsets=${result.xmlClusterCount}  ` +
      `| dropped=${totals.dropped} (shifted-later=${totals.shifted}) added=${totals.added} ` +
      `raw-only onsets=${totals.rawOnly} fabricated onsets=${totals.xmlOnly}`
  );
  const bad = rows.filter((r) => r.dropped.length || r.added.length);
  if (!bad.length) {
    console.log("  (no pitch differences)");
    return;
  }
  console.log("  worst onsets (raw -> printed):");
  bad
    .map((r) => ({ r, score: r.dropped.length + r.added.length }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .forEach(({ r }) => {
      const t = (r.rawT ?? r.xmlT).toFixed(3);
      const tags = [];
      if (r.dropped.length) tags.push(`drop ${names(r.dropped)}`);
      if (r.shiftedLater && r.shiftedLater.length) tags.push(`(of which shifted-later ${names(r.shiftedLater)})`);
      if (r.added.length) tags.push(`add ${names(r.added)}`);
      if (r.kind === "raw_only") tags.push("[onset entirely absent from score]");
      if (r.kind === "xml_only") tags.push("[onset fabricated by score]");
      console.log(`   t=${t}s  raw=${names(r.raw)}  printed=${names(r.xml)}  ${tags.join("  ")}`);
    });
}

function fmtPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function tokenLabel(token) {
  if (!token) return "(none)";
  return `q=${token.quarterAbs.toFixed(3)} p${token.part}/s${token.staff}/v${token.voice} d=${(token.durationDivs / CANON_DIVISIONS).toFixed(3)} ${names(token.pitches)}`;
}

function reportScoreReference(label, comparison, limit) {
  const exact = comparison.exactTokenF1;
  const edit = comparison.rhythmicEdit;
  console.log("\n---------------- SCORE REFERENCE ----------------");
  console.log(`  ${label}`);
  console.log(
    `  score tokens: predicted=${comparison.predTokens.length} reference=${comparison.refTokens.length} ` +
      `exact=${exact.matched} exact_f1=${exact.f1.toFixed(4)}`
  );
  console.log(
    `  rhythmic_edit_accuracy=${fmtPct(edit.normalizedAccuracy)} ` +
      `edit_cost=${edit.editCost.toFixed(3)} reference_cost=${edit.referenceCost.toFixed(3)}`
  );
  console.log(
    `  ops: exact=${edit.counts.exact} substitute=${edit.counts.substitutions} ` +
      `delete=${edit.counts.deletions} insert=${edit.counts.insertions}`
  );
  if (!edit.examples.length || limit <= 0) return;
  console.log("  edit examples:");
  edit.examples.slice(0, limit).forEach((example) => {
    if (example.kind === "substitute") {
      console.log(`   sub cost=${example.cost.toFixed(3)}  ref ${tokenLabel(example.ref)}  -> pred ${tokenLabel(example.pred)}`);
    } else if (example.kind === "delete") {
      console.log(`   del cost=${example.cost.toFixed(3)}  ref ${tokenLabel(example.ref)}`);
    } else {
      console.log(`   ins cost=${example.cost.toFixed(3)}  pred ${tokenLabel(example.pred)}`);
    }
  });
}

// ── 7. Self-test (no GPU / no payload needed) ──
function selfTestPayload() {
  // Two same-staff notes ~12 ms apart at the very start: a real chord in the
  // raw events, but past the 5 ms same-staff merge window + 1/24-beat snap, so
  // the score should split them across beats (drop one, resurface it later).
  return {
    clip_id: "SELFTEST",
    bpm: 120,
    notes: [
      { time_seconds: 0.0, midi_note: 60, duration_seconds: 0.5 },
      { time_seconds: 0.012, midi_note: 64, duration_seconds: 0.5 },
      { time_seconds: 0.5, midi_note: 67, duration_seconds: 0.5 },
    ],
    chords: [{ time_seconds: 1.0, midi_notes: [48, 55, 64], duration_seconds: 0.5 }],
  };
}

// ── main ──
function parseArgs(argv) {
  const args = {
    selfTest: false,
    limit: 12,
    ref: null,
    refDir: null,
    refPayload: null,
    xmlPair: null,
    jsonOut: null,
    files: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--self-test") {
      args.selfTest = true;
    } else if (a.startsWith("--limit=")) {
      args.limit = parseInt(a.split("=")[1], 10);
    } else if (a === "--limit") {
      args.limit = parseInt(argv[++i], 10);
    } else if (a.startsWith("--ref=")) {
      args.ref = a.slice("--ref=".length);
    } else if (a === "--ref") {
      args.ref = argv[++i];
    } else if (a.startsWith("--ref-dir=")) {
      args.refDir = a.slice("--ref-dir=".length);
    } else if (a === "--ref-dir") {
      args.refDir = argv[++i];
    } else if (a.startsWith("--ref-payload=")) {
      args.refPayload = a.slice("--ref-payload=".length);
    } else if (a === "--ref-payload") {
      args.refPayload = argv[++i];
    } else if (a === "--xml-pair") {
      args.xmlPair = [argv[++i], argv[++i]];
    } else if (a.startsWith("--json-out=")) {
      args.jsonOut = a.slice("--json-out=".length);
    } else if (a === "--json-out") {
      args.jsonOut = argv[++i];
    } else if (a.startsWith("--")) {
      throw new Error(`Unknown option: ${a}`);
    } else {
      args.files.push(a);
    }
  }
  if (!Number.isFinite(args.limit)) args.limit = 12;
  return args;
}

function findReferenceForPayload(payload, payloadFile, args) {
  const explicit =
    payload.gt_musicxml_path ||
    payload.reference_musicxml_path ||
    payload.ref_musicxml_path ||
    payload.musicxml_path ||
    null;
  if (explicit) return path.resolve(path.dirname(payloadFile), explicit);
  if (args.ref) return path.resolve(args.ref);
  if (!args.refDir) return null;

  const dir = path.resolve(args.refDir);
  const candidates = [];
  const clipId = payload.clip_id ? String(payload.clip_id) : "";
  if (clipId) {
    candidates.push(`${clipId}.musicxml`, `${clipId}.xml`);
    candidates.push(`reference_${clipId}.musicxml`, `reference_${clipId}.xml`);
  }
  const stem = path.basename(payloadFile, path.extname(payloadFile));
  candidates.push(`${stem}.musicxml`, `${stem}.xml`);
  if (stem.startsWith("_tmp_app_payload_")) {
    const shortStem = stem.slice("_tmp_app_payload_".length);
    candidates.push(`${shortStem}.musicxml`, `${shortStem}.xml`);
  }
  for (const name of candidates) {
    const candidate = path.join(dir, name);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

// Index a GT-payload source by clip id so the reference score can be
// regenerated per clip. Accepts the oracle `{payloads:[...]}` wrapper, a bare
// array of payloads, or a single payload object.
function loadRefPayloadMap(refPayloadPath) {
  const raw = JSON.parse(fs.readFileSync(path.resolve(refPayloadPath), "utf-8"));
  const list = Array.isArray(raw) ? raw : Array.isArray(raw.payloads) ? raw.payloads : [raw];
  const map = new Map();
  for (const gt of list) {
    if (!gt || typeof gt !== "object") continue;
    for (const key of [gt.benchmark_clip_id, gt.clip_id]) {
      if (key) map.set(String(key), gt);
    }
  }
  return map;
}

// Strip a trailing experiment-arm suffix ("clip_002__control" -> "clip_002").
function baseClipId(label) {
  return String(label).split("__")[0];
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.xmlPair) {
    const [predPath, refPath] = args.xmlPair.map((p) => path.resolve(p));
    const comparison = compareScoreXml(fs.readFileSync(predPath, "utf-8"), fs.readFileSync(refPath, "utf-8"));
    reportScoreReference(`${path.basename(predPath)} vs ${path.basename(refPath)}`, comparison, args.limit);
    if (args.jsonOut) {
      fs.writeFileSync(
        path.resolve(args.jsonOut),
        JSON.stringify(
          {
            mode: "xml_pair",
            comparisons: [compactScoreComparison(`${path.basename(predPath)} vs ${path.basename(refPath)}`, refPath, comparison)],
          },
          null,
          2
        ),
        "utf-8"
      );
    }
    return;
  }

  const generateMusicXML = loadGenerateMusicXML();

  if (args.selfTest) {
    const payload = selfTestPayload();
    const result = diffPayload(payload, generateMusicXML);
    report("SELFTEST", result, args.limit);
    const predXml = generateMusicXML(payload.notes, payload.chords, "4/4", payload.bpm, 0);
    const comparison = compareScoreXml(predXml, predXml);
    reportScoreReference("SELFTEST generated XML vs itself", comparison, args.limit);
    if (args.jsonOut) {
      fs.writeFileSync(
        path.resolve(args.jsonOut),
        JSON.stringify(
          {
            mode: "self_test",
            raw_summaries: [{ clip: "SELFTEST", ...result.totals }],
            comparisons: [compactScoreComparison("SELFTEST generated XML vs itself", null, comparison)],
          },
          null,
          2
        ),
        "utf-8"
      );
    }
    return;
  }

  let payloadFiles = args.files;
  if (!payloadFiles.length) {
    const backend = path.join(REPO, "backend");
    payloadFiles = fs
      .readdirSync(backend)
      .filter((f) => /^_tmp_app_payload_.*\.json$/.test(f))
      .map((f) => path.join(backend, f));
  }
  if (!payloadFiles.length) {
    console.error("No payloads. Run backend/dump_app_payloads.py first, or pass --self-test.");
    process.exit(1);
  }

  const refPayloadMap = args.refPayload ? loadRefPayloadMap(args.refPayload) : null;

  const summary = [];
  const scoreSummary = [];
  for (const file of payloadFiles) {
    const payload = JSON.parse(fs.readFileSync(file, "utf-8"));
    const label = payload.clip_id || path.basename(file);
    const bpm = payload.bpm && payload.bpm > 1 ? payload.bpm : 120;
    const predXml = generateMusicXML(payload.notes || [], payload.chords || [], "4/4", bpm, 0);
    const result = diffPayload(payload, generateMusicXML);
    report(label, result, args.limit);
    summary.push({ clip: label, ...result.totals });

    // Tempo-normalized reference: regenerate the GT score at the SAME bpm the
    // prediction used, so a global tempo difference can't drift onsets apart and
    // split matched notes into insert+delete pairs. Falls back to the static
    // reference .musicxml when no GT payload is supplied.
    if (refPayloadMap) {
      const gt = refPayloadMap.get(baseClipId(label)) || refPayloadMap.get(String(label));
      if (gt) {
        const refXml = generateMusicXML(gt.notes || [], gt.chords || [], "4/4", bpm, 0);
        const comparison = compareScoreXml(predXml, refXml);
        reportScoreReference(`${label} vs GT@${bpm.toFixed(1)}bpm`, comparison, args.limit);
        const metrics = compactScoreComparison(label, `gt_payload@${bpm.toFixed(1)}bpm`, comparison);
        metrics.reference_bpm = bpm;
        metrics.tempo_normalized = true;
        scoreSummary.push({
          clip: label,
          accuracy: comparison.rhythmicEdit.normalizedAccuracy,
          exactF1: comparison.exactTokenF1.f1,
          editCost: comparison.rhythmicEdit.editCost,
          referenceCost: comparison.rhythmicEdit.referenceCost,
          metrics,
        });
        continue;
      }
      console.warn(`[scorediff] no GT payload for ${label} (base=${baseClipId(label)})`);
    }

    const refPath = findReferenceForPayload(payload, file, args);
    if (refPath && fs.existsSync(refPath)) {
      const comparison = compareScoreXml(predXml, fs.readFileSync(refPath, "utf-8"));
      reportScoreReference(`${label} vs ${path.basename(refPath)}`, comparison, args.limit);
      scoreSummary.push({
        clip: label,
        accuracy: comparison.rhythmicEdit.normalizedAccuracy,
        exactF1: comparison.exactTokenF1.f1,
        editCost: comparison.rhythmicEdit.editCost,
        referenceCost: comparison.rhythmicEdit.referenceCost,
        metrics: compactScoreComparison(label, refPath, comparison),
      });
    } else if (args.ref || args.refDir || payload.gt_musicxml_path || payload.reference_musicxml_path || payload.ref_musicxml_path) {
      console.warn(`[scorediff] reference MusicXML not found for ${label}`);
    }
  }

  summary.sort((a, b) => b.dropped + b.added - (a.dropped + a.added));
  console.log("\n=================== RANKED SUMMARY ===================");
  for (const s of summary) {
    console.log(
      `  ${String(s.clip).padEnd(16)} dropped=${s.dropped} (shifted=${s.shifted}) added=${s.added} ` +
        `raw-only=${s.rawOnly} fabricated=${s.xmlOnly}`
    );
  }

  if (scoreSummary.length) {
    console.log("\n============== SCORE-REFERENCE SUMMARY ==============");
    scoreSummary.sort((a, b) => a.accuracy - b.accuracy);
    for (const s of scoreSummary) {
      console.log(
        `  ${String(s.clip).padEnd(16)} rhythmic_edit_accuracy=${fmtPct(s.accuracy)} ` +
          `exact_token_f1=${s.exactF1.toFixed(4)} edit=${s.editCost.toFixed(3)}/${s.referenceCost.toFixed(3)}`
      );
    }
  }

  if (args.jsonOut) {
    fs.writeFileSync(
      path.resolve(args.jsonOut),
      JSON.stringify(
        {
          mode: "payload_batch",
          raw_summaries: summary,
          comparisons: scoreSummary.map((row) => row.metrics),
        },
        null,
        2
      ),
      "utf-8"
    );
  }
}

main();
