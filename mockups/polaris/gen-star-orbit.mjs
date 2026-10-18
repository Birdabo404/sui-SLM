// Generates star-orbit.css: the path and the time-critical keyframes of the header star's triggered wrap.
//   node gen-star-orbit.mjs
// Idle motion, the glow swell and the glints are hand-authored in polaris.css; everything that has to agree
// with the path geometry or the travel easing is computed here, so the two can't drift apart.
//
// Path: two coils around the word, a stadium (level passes, semicircular ends) whose turn between the coils
// swings wider on the right, resampled by arc length and emitted as Catmull-Rom cubic Béziers
// (C1-continuous, true curves) for offset-path. offset-distance runs it on a single cubic-bezier, so speed comes from one slow-in/slow-out.
// Scale: the band leaves ~12px between the ink and the 40px band's edge, so the star's size in transit is
// the largest that keeps GAP_INK / GAP_BAND everywhere (measured on the star's cross-shaped silhouette),
// eroded and blurred in time so it changes gently. The star stays upright and banks into turns.
// Every sample is re-checked from the emitted keyframes; the script throws on contact.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const WRAP = 4.4;               // s, whole triggered wrap; polaris.css uses the same duration
const T_LAUNCH = 0.38;          // s, anticipation ends, travel begins
const T_ARRIVE = 3.9;           // s, back at rest; follow-through runs on to WRAP
const TRAVEL = [0.42, 0, 0.3, 1]; // travel easing: unhurried launch, long settle
const COILS = 2;
const SWING = 0.7;              // em, extra reach on the right between the two coils
const REST_X = 2.75;            // em from the word's center
const INK_X = 2.07;             // em, measured ink half-width of "polaris", JetBrains Mono 600
const BANK = 8;                 // deg, maximum lean into a turn
const SWAY = 2.5;               // deg, idle sway amplitude (polaris.css), stacked on the bank
const RMAX = 11;                // deg, tilt budget of the clearance model
const RAY_V = 0.5, RAY_H = 0.4167, WAIST = 0.07; // em, star silhouette (#star-lux): ray half-lengths, half-waist
const BREATHE = 1.03;           // idle breathing scale (polaris.css), counted in every clearance
const GAP_INK = 1.5, GAP_BAND = 3; // px
const STEP = 0.5;               // % of WRAP between body keyframes
const SEGMENTS = 96;            // cubic Béziers in the path

// Measured in the mock, px from the word's center. Slim: the mark sits 1px above the band's center.
const bands = [
  { suffix: "",      sel: ".band",      fs: 15, inkTop: -8.5, inkBottom: 7.5, top: -27, bottom: 27, left: -52.75 },
  { suffix: "-slim", sel: ".band.slim", fs: 14, inkTop: -6,   inkBottom: 8,   top: -18, bottom: 20, left: -46.3 },
];

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const r2 = (v) => (Math.abs(v) < 0.005 ? 0 : +v.toFixed(2));
const r4 = (v) => (Math.abs(v) < 5e-5 ? 0 : +v.toFixed(4));
const pct = (t) => r4((t / WRAP) * 100);

function cubicBezier([x1, y1, x2, y2]) {
  const bx = (t) => 3 * x1 * t * (1 - t) ** 2 + 3 * x2 * t * t * (1 - t) + t ** 3;
  const by = (t) => 3 * y1 * t * (1 - t) ** 2 + 3 * y2 * t * t * (1 - t) + t ** 3;
  return (x) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let lo = 0, hi = 1;
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; (bx(m) < x ? (lo = m) : (hi = m)); }
    return by((lo + hi) / 2);
  };
}
const travel = cubicBezier(TRAVEL);
const inOut = cubicBezier([0.45, 0, 0.55, 1]);
const release = cubicBezier([0.25, 0.6, 0.3, 1]);
const bump = (t, t0, t1) => (t <= t0 || t >= t1 ? 0 : Math.sin(Math.PI * inOut((t - t0) / (t1 - t0))));

function polyline(pts) {
  const len = [0];
  for (let i = 1; i < pts.length; i++) len.push(len[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const at = (d) => {
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; (len[m] < d ? (lo = m) : (hi = m)); }
    const k = (d - len[lo]) / (len[hi] - len[lo] || 1);
    return [pts[lo][0] + k * (pts[hi][0] - pts[lo][0]), pts[lo][1] + k * (pts[hi][1] - pts[lo][1])];
  };
  return { len, total: len[len.length - 1], at };
}

function build(b) {
  const { fs } = b, inkX = INK_X * fs, restX = REST_X * fs;
  // centre each pass in the free space between ink and band edge, weighted by the two gaps
  const yTop = (b.inkTop - GAP_INK + b.top + GAP_BAND) / 2, yBot = (b.inkBottom + GAP_INK + b.bottom - GAP_BAND) / 2;
  const cy = (yTop + yBot) / 2, B = (yBot - yTop) / 2;
  // stadium: level passes over and under the word, semicircular ends sweeping around the ink corners;
  // the turn between the coils is pushed SWING further right, into the band's open space
  const R = B, xR = restX - R, xL = Math.min(restX, -b.left - GAP_BAND - 0.5 * fs * 0.6) - R, swing = SWING * fs;
  const dense = [], Q = Math.PI / 2, EXT = 0.35;                       // EXT: rad of arc past both ends, for end tangents
  const arc = (cx, a0, a1) => { const k = Math.ceil((Math.abs(a1 - a0) * R) / 0.05);
    for (let i = 0; i < k; i++) { const t = a0 + ((a1 - a0) * i) / k; dense.push([cx + R * Math.cos(t), cy + R * Math.sin(t)]); } };
  const line = (x0, x1, y) => { const k = Math.ceil(Math.abs(x1 - x0) / 0.05); for (let i = 0; i < k; i++) dense.push([x0 + ((x1 - x0) * i) / k, y]); };
  arc(xR, EXT, -Q);                                                    // rest → up the right end
  for (let c = 0; c < COILS; c++) {
    const last = c === COILS - 1, xEnd = last ? xR : xR + swing;
    line(c ? xR + swing : xR, -xL, cy - R);                            // over the word
    arc(-xL, -Q, -3 * Q);                                              // round the left end
    line(-xL, xEnd, cy + R);                                           // under the word
    arc(xEnd, Q, last ? -EXT : -Q);                                    // round the right end (last one: back to rest)
  }
  dense.push([xR + R * Math.cos(-EXT), cy + R * Math.sin(-EXT)]);
  const dl = polyline(dense), startD = EXT * R, endD = dl.total - EXT * R;
  const step = (endD - startD) / SEGMENTS;
  const P = [];
  for (let i = -1; i <= SEGMENTS + 1; i++) P.push(dl.at(startD + i * step));
  P[1] = [restX, cy]; P[SEGMENTS + 1] = [restX, cy];              // exact rest point at both ends
  let d = `M${r2(P[1][0])} ${r2(P[1][1])}`;
  const bez = [];
  for (let i = 1; i <= SEGMENTS; i++) {
    const p0 = P[i], p1 = P[i + 1], c1 = [p0[0] + (p1[0] - P[i - 1][0]) / 6, p0[1] + (p1[1] - P[i - 1][1]) / 6];
    const c2 = [p1[0] - (P[i + 2][0] - p0[0]) / 6, p1[1] - (P[i + 2][1] - p0[1]) / 6];
    d += `C${r2(c1[0])} ${r2(c1[1])} ${r2(c2[0])} ${r2(c2[1])} ${r2(p1[0])} ${r2(p1[1])}`;
    bez.push([p0, c1, c2, p1].map(([x, y]) => [r2(x), r2(y)]));
  }
  // the path as the browser will see it (rounded control points), for timing and clearance
  const fine = [];
  for (const [p0, c1, c2, p1] of bez) for (let k = 0; k < 80; k++) {
    const t = k / 80, u = 1 - t;
    fine.push([0, 1].map((j) => u ** 3 * p0[j] + 3 * u * u * t * c1[j] + 3 * u * t * t * c2[j] + t ** 3 * p1[j]));
  }
  fine.push(bez[bez.length - 1][3]);
  const path = polyline(fine);
  const posAt = (t) => path.at(path.total * travel(clamp01((t - T_LAUNCH) / (T_ARRIVE - T_LAUNCH))));

  // star silhouette as two bars (long vertical rays, shorter horizontal ones), widened for up to RMAX of tilt
  const tilt = Math.sin((RMAX * Math.PI) / 180);
  const bars = [[WAIST + RAY_V * tilt, RAY_V], [RAY_H, WAIST + RAY_H * tilt]];
  const inkC = (b.inkTop + b.inkBottom) / 2, inkH = (b.inkBottom - b.inkTop) / 2;
  const air = ([x, y], s) => Math.min(...bars.map(([hx, hy]) => {
    const w = hx * fs * s * BREATHE, h = hy * fs * s * BREATHE;
    const dx = Math.max(0, Math.abs(x) - inkX - w), dy = Math.max(0, Math.abs(y - inkC) - inkH - h);
    return Math.min((dx || dy ? Math.hypot(dx, dy) : -1) - GAP_INK, y - h - b.top - GAP_BAND, b.bottom - y - h - GAP_BAND, x - w - b.left - GAP_BAND);
  }));
  const fit = (p) => { let lo = 0, hi = 1; if (air(p, 1) >= 0.25) return 1;
    for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; (air(p, m) >= 0.25 ? (lo = m) : (hi = m)); } return lo; };

  // size limit over time: erode (window min) then blur, so the result never exceeds the raw limit
  const DT = 0.005, n = Math.round(WRAP / DT), raw = [], ero = [];
  for (let i = 0; i <= n; i++) raw.push(fit(posAt(i * DT)));
  const W = Math.round(0.1 / DT), H = Math.round(0.05 / DT);
  const blur = (arr, h) => arr.map((_, i) => { let s = 0; for (let j = i - h; j <= i + h; j++) s += arr[Math.min(n, Math.max(0, j))]; return s / (2 * h + 1); });
  for (let i = 0; i <= n; i++) { let m = Infinity; for (let j = i - W; j <= i + W; j++) m = Math.min(m, raw[Math.min(n, Math.max(0, j))]); ero.push(m); }
  const lim = blur(blur(ero, H), H);
  const limitAt = (t) => lim[Math.min(n, Math.round(t / DT))];

  // banking: tilt into the turn in proportion to how fast the heading changes, smoothed
  const heading = [];
  for (let i = 0; i <= n; i++) { const [x0, y0] = posAt(Math.max(0, i - 1) * DT), [x1, y1] = posAt(Math.min(n, i + 1) * DT); heading.push(Math.atan2(y1 - y0, x1 - x0)); }
  let bank = heading.map((h, i) => { if (!i) return 0; let d = h - heading[i - 1]; d -= 2 * Math.PI * Math.round(d / (2 * Math.PI)); return d / DT; });
  bank = blur(blur(bank, 16), 16).map((w, i) => Math.max(-BANK, Math.min(BANK, w * 0.7)));
  const bankAt = (t) => bank[Math.min(n, Math.round(t / DT))];

  const pose = (t) => {
    // anticipation: dip down-right, gather (squash), lean back against the launch
    const a = t < 0.3 ? inOut(t / 0.3) : t < T_LAUNCH ? 1 : 1 - release(clamp01((t - T_LAUNCH) / 0.22));
    const stretch = 0.07 * bump(t, T_LAUNCH, T_LAUNCH + 0.36);     // launch stretch along the (upward) motion
    const lead = bump(t, T_ARRIVE - 0.35, WRAP);                     // follow-through: carries on up, settles back
    const bloom = 0.045 * bump(t, T_ARRIVE - 0.1, WRAP);            // settle bloom
    const s = Math.min(1, limitAt(t)) * (1 + bloom);
    const [, y] = posAt(t);
    const far = clamp01((cy - y) / B);                                // top pass reads as the far side
    return {
      x: a * 0.04 * fs, y: a * 0.1 * fs - lead * 0.04 * fs, r: 7 * a + bankAt(t),
      sx: s * (1 + 0.05 * a) * (1 - 0.5 * stretch), sy: s * (1 - 0.08 * a) * (1 + stretch),
      o: 1 - 0.16 * far * far,
    };
  };

  const keys = [];
  for (let p = 0; p <= 100 + 1e-9; p += STEP) keys.push([p, pose((p / 100) * WRAP)]);
  keys[keys.length - 1][1] = { x: 0, y: 0, r: 0, sx: 1, sy: 1, o: 1 };
  const tf = (k) => `translate(${r2(k.x)}px,${r2(k.y)}px) rotate(${r2(k.r)}deg) scale(${r4(k.sx)},${r4(k.sy)})`;

  // re-check from the emitted keyframes, linearly interpolated as the browser does
  const lerp = (t) => {
    const q = (t / WRAP) * 100 / STEP, i = Math.min(keys.length - 2, Math.floor(q)), k = q - i, A = keys[i][1], Z = keys[i + 1][1];
    return Object.fromEntries(Object.keys(A).map((p) => [p, A[p] + k * (Z[p] - A[p])]));
  };
  let worst = [Infinity], minS = 1, peak = 0, maxR = 0, prev = posAt(0);
  for (let i = 1; i <= n; i++) {
    const t = i * DT, k = lerp(t), [x, y] = posAt(t);
    const v = air([x + k.x, y + k.y], Math.max(k.sx, k.sy));
    if (v < worst[0]) worst = [v, t];
    maxR = Math.max(maxR, Math.abs(k.r));
    if (t > T_LAUNCH + 0.3 && t < T_ARRIVE - 0.3) minS = Math.min(minS, Math.max(k.sx, k.sy));
    peak = Math.max(peak, Math.hypot(x - prev[0], y - prev[1]) / DT); prev = [x, y];
  }
  if (maxR + SWAY > RMAX) throw new Error(`${b.sel}: tilt ${(maxR + SWAY).toFixed(1)}° exceeds RMAX`);
  if (worst[0] < 0) throw new Error(`${b.sel}: star comes ${(-worst[0]).toFixed(3)}px inside its clearance at t=${worst[1].toFixed(3)}s`);

  const crossings = [];                                               // when the star passes over / under the word's centre
  for (let i = 1; i <= n; i++) { const [x0, y0] = posAt((i - 1) * DT), [x1] = posAt(i * DT); if (Math.sign(x0) !== Math.sign(x1)) crossings.push(`${(i * DT).toFixed(2)}s ${y0 < cy ? "over" : "under"}`); }
  console.log(`${b.sel}: path ${path.total.toFixed(0)}px (${COILS} coils, ${SEGMENTS} Béziers), peak ${peak.toFixed(0)}px/s; ` +
    `star ${fs}px at rest, ≥${(minS * fs).toFixed(1)}px in transit; min air beyond gaps ${worst[0].toFixed(2)}px, max tilt ${maxR.toFixed(1)}°; crossings ${crossings.join(", ")}`);

  return `${b.sel} .star-fx{--rest-x:${r2(restX)}px;--rest-y:${r2(cy)}px}
${b.sel} .rig,${b.sel} .trail>*{offset-path:path("${d}")}
/* Body, ${fs}px word, sampled every ${STEP}%:
   anticipation 0–${T_LAUNCH}s: dips down-right, squashes, leans back against the launch;
   squash & stretch: brief vertical stretch on the (upward) launch;
   secondary action: banks into each turn, up to ${BANK}°; far side: dims slightly on the passes over the word;
   staging: size held to the clearance between ink and band (≥${(minS * fs).toFixed(1)}px here);
   follow-through ${(T_ARRIVE - 0.35).toFixed(2)}–${WRAP}s: carries on upward past rest and blooms ${4.5}% on landing, then settles to identity. */
@keyframes star-wrap-body${b.suffix}{
${keys.map(([p, k]) => `  ${r4(p)}%{transform:${tf(k)};opacity:${r4(k.o)}}`).join("\n")}
}`;
}

const trail = (fadeIn, fadeOut) => `  0%,${pct(T_LAUNCH + fadeIn[0])}%{opacity:0;animation-timing-function:cubic-bezier(.3,0,.2,1)}
  ${pct(T_LAUNCH + fadeIn[1])}%,${pct(T_ARRIVE - fadeOut[0])}%{opacity:1;animation-timing-function:cubic-bezier(.5,0,.7,.4)}
  ${pct(T_ARRIVE - fadeOut[1])}%,100%{opacity:0}`;
const sparkle = () => {
  const t0 = T_LAUNCH + 0.2, t1 = T_ARRIVE - 1.0, t2 = T_ARRIVE - 0.3;
  const k = [`  0%,${pct(t0)}%{opacity:0;animation-timing-function:cubic-bezier(.3,0,.2,1)}`];
  for (let t = t0 + 0.3, hi = true; t < t1; t += hi ? 0.11 : 0.17) {
    k.push(`  ${pct(t)}%{opacity:${hi ? 1 : 0.45};animation-timing-function:cubic-bezier(.45,0,.55,1)}`);
    hi = !hi;
  }
  k.push(`  ${pct(t1)}%{opacity:.9;animation-timing-function:cubic-bezier(.5,0,.7,.4)}`, `  ${pct(t2)}%,100%{opacity:0}`);
  return k.join("\n");
};

const css = `/* star-orbit.css — generated by gen-star-orbit.mjs; edit the generator, not this file.
   Triggered wrap, ${WRAP}s: ${T_LAUNCH}s anticipation, ${(T_ARRIVE - T_LAUNCH).toFixed(2)}s travel (${COILS} coils), ${(WRAP - T_ARRIVE).toFixed(1)}s follow-through. */
${bands.map(build).join("\n")}

/* Arcs + slow in/slow out: one eased trip along the coil path. Held at the start through the anticipation. */
@keyframes star-wrap-path{
  0%,${pct(T_LAUNCH)}%{offset-distance:0%;animation-timing-function:cubic-bezier(${TRAVEL.join(",")})}
  ${pct(T_ARRIVE)}%,100%{offset-distance:100%}
}
/* Overlapping action: the trail rides the same path on a delay, so its spacing stretches with speed.
   It appears once the star is moving and is gone before each piece reaches rest. */
@keyframes star-trail{
${trail([0.08, 0.4], [0.8, 0.12])}
}
/* Sparks: arrive later and twinkle on an uneven 0.17s/0.11s beat, so they never pulse in step with the star. */
@keyframes star-trail-spark{
${sparkle()}
}
`;
writeFileSync(join(dirname(fileURLToPath(import.meta.url)), "star-orbit.css"), css);
