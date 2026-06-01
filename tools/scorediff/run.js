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

function pitchToMidi(step, alter, octave) {
  const base = STEP_SEMITONE[step];
  if (base === undefined) return null;
  return (octave + 1) * 12 + base + (alter || 0);
}

// Returns [{ midi, beatAbs, divAbs, staff }] for every sounding note.
function parseScoreXml(xml) {
  const out = [];
  const divsMatch = xml.match(/<divisions>(\d+)<\/divisions>/);
  const divisions = divsMatch ? parseInt(divsMatch[1], 10) : 24;
  const timeMatch = xml.match(/<beats>(\d+)<\/beats>\s*<beat-type>(\d+)<\/beat-type>/);
  const beats = timeMatch ? parseInt(timeMatch[1], 10) : 4;
  const beatType = timeMatch ? parseInt(timeMatch[2], 10) : 4;
  const measureDivs = Math.round((beats * 4) / beatType * divisions); // quarter = `divisions`

  const measures = xml.split(/<measure\b/).slice(1);
  let measureIndex = 0;
  for (const rawMeasure of measures) {
    const measureBase = measureIndex * measureDivs;
    let cursor = 0; // divisions from measure start
    let lastOnset = 0;

    // Tokenize the measure into <note>, <backup>, <forward> elements in order.
    const tokenRe = /<(note|backup|forward)\b[\s\S]*?<\/\1>/g;
    let m;
    while ((m = tokenRe.exec(rawMeasure)) !== null) {
      const tag = m[1];
      const body = m[0];
      const durMatch = body.match(/<duration>(\d+)<\/duration>/);
      const duration = durMatch ? parseInt(durMatch[1], 10) : 0;

      if (tag === "backup") {
        cursor -= duration;
        continue;
      }
      if (tag === "forward") {
        cursor += duration;
        continue;
      }

      // <note>
      const isChord = /<chord\s*\/>/.test(body);
      const isRest = /<rest\b/.test(body);
      const staffMatch = body.match(/<staff>(\d+)<\/staff>/);
      const staff = staffMatch ? parseInt(staffMatch[1], 10) : 1;

      const onset = isChord ? lastOnset : cursor;

      if (!isRest) {
        const p = body.match(/<step>([A-G])<\/step>(?:\s*<alter>(-?\d+)<\/alter>)?\s*<octave>(\d+)<\/octave>/);
        if (p) {
          const midi = pitchToMidi(p[1], p[2] ? parseInt(p[2], 10) : 0, parseInt(p[3], 10));
          if (midi != null) {
            const divAbs = measureBase + onset;
            out.push({ midi, divAbs, beatAbs: divAbs / divisions, staff });
          }
        }
      }

      if (!isChord) {
        lastOnset = cursor;
        cursor += duration;
      }
    }
    measureIndex += 1;
  }
  return { notes: out, divisions };
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
function main() {
  const argv = process.argv.slice(2);
  const selfTest = argv.includes("--self-test");
  const limitArg = argv.find((a) => a.startsWith("--limit="));
  const limit = limitArg ? parseInt(limitArg.split("=")[1], 10) : 12;
  const files = argv.filter((a) => !a.startsWith("--"));

  const generateMusicXML = loadGenerateMusicXML();

  if (selfTest) {
    const result = diffPayload(selfTestPayload(), generateMusicXML);
    report("SELFTEST", result, limit);
    return;
  }

  let payloadFiles = files;
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

  const summary = [];
  for (const file of payloadFiles) {
    const payload = JSON.parse(fs.readFileSync(file, "utf-8"));
    const result = diffPayload(payload, generateMusicXML);
    report(payload.clip_id || path.basename(file), result, limit);
    summary.push({ clip: payload.clip_id || path.basename(file), ...result.totals });
  }

  summary.sort((a, b) => b.dropped + b.added - (a.dropped + a.added));
  console.log("\n=================== RANKED SUMMARY ===================");
  for (const s of summary) {
    console.log(
      `  ${String(s.clip).padEnd(16)} dropped=${s.dropped} (shifted=${s.shifted}) added=${s.added} ` +
        `raw-only=${s.rawOnly} fabricated=${s.xmlOnly}`
    );
  }
}

main();
