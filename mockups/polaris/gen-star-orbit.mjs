// Generates star-orbit.css: the path and the time-critical keyframes of the header star's triggered wrap.
//   node gen-star-orbit.mjs [--debug]       (--debug also writes /tmp/star-orbit-debug.json)
// Idle motion and the glints are hand-authored in polaris.css; everything that has to agree with the path,
// the depth or the travel timing is computed here, so the pieces can't drift apart.
//
// The wrap is an orbit around the word seen nearly edge-on: two turns keyed every quarter turn
// (rest → behind → left end → under → right swing → front → left end → behind → rest), with x, y and depth
// each a periodic cubic spline through the keys, so the curve is C2 and closes exactly on the rest point.
// Projected like a real orbit, the star is fastest crossing the word and slows into every end.
// Depth drives size, light and layering. The word is drawn twice — .base under the star, .ink over it — and
// the switch between them is a stepped opacity change placed where the star and every trail piece are clear
// of all glyphs, so nothing ever changes layer over a letter. The glow rides the same path in its own layer
// under both copies, so it lights the band around the letters and never swaps.
// Every sample is re-checked from the emitted keyframes; the script throws on contact or a dirty switch.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const WRAP = 4.4;               // s, whole triggered wrap; polaris.css uses the same duration
const T_LAUNCH = 0.38;          // s, anticipation ends, travel begins
const T_ARRIVE = 3.9;           // s, back at rest; follow-through runs on to WRAP
const TRAVEL = [0.35, 0, 0.5, 1];   // orbit phase easing: slow in, long slow out
const REST_X = 2.75;            // em from the word's center
const SWING_X = 2.95;           // em, right-hand end of the swing between the turns
// depth per quarter turn, -1 far side … +1 near side: rest, behind, left end, under, swing, front, left end, behind
const DEPTH = [0, -1, -0.3, 0.25, 0.6, 1, 0, -1];
const NEAR = 1.1, FAR = 0.6;    // star size on the near / far side (1 at depth 0)
const DIM = 0.28;               // opacity lost on the far side
const BANK = 8;                 // deg, maximum lean into a turn
const SWAY = 2.5;               // deg, idle sway amplitude (polaris.css), stacked on the bank
const RMAX = 11;                // deg, tilt budget of the clearance model
const RAY_V = 0.5, RAY_H = 0.4167, WAIST = 0.07; // em, star silhouette (#star-lux): ray half-lengths, half-waist
const BREATHE = 1.03;           // idle breathing scale (polaris.css), counted in every clearance
const GAP_INK = 1.5, GAP_BAND = 3, GAP_SWITCH = 0.25; // px; the switch gap is to glyph boxes, the pixel audit checks the frames
const STEP = 0.5;               // % of WRAP between path and body keyframes
const STEP_FX = 1;              // % of WRAP between glow and trail keyframes
const SEGMENTS = 128;           // cubic Béziers in the path
// polaris.css: halo opacity at rest, the glow's alpha by radius (fraction of the halo's 1em radius), its idle peak
const HALO_O = 0.62, HALO_STOPS = [[0, 0.62], [0.38, 0.3], [0.66, 0.09], [1, 0]], GLOW_PEAK = 1.03;
const SEEN = 0.039;             // glow alpha that registers against the band (≈16/255 summed over channels)
// trail pieces (polaris.css): [delay s, radius px]; dots are drawn out 2.8× along the path, sparks .3em × 1.5
const DOTS = [...Array(10)].map((_, k) => [0.014 * (k + 1), 1.4 * (3.2 - 0.2 * k)]);
const SPARKS = [0.07, 0.12, 0.18];

// Measured in the mock, px from the word's center: the band's inner edges, the rest point, the x-height and a
// box per glyph (the p's descender, l and i ascenders separately), rounded outwards to 0.25px.
const bands = [
  { suffix: "", sel: ".band", fs: 15, restY: -0.25, top: -27, bottom: 27, left: -52.75, xTop: -3.5, base: 5.25,
    glyphs: [[-30.75, -21.75, -3.5, 5.25], [-30.75, -28.75, -3.5, 7.75], [-21.75, -14.75, -3.5, 5.25], [-13.75, -5.25, -6, 5],
      [-4.25, 2.75, -3.5, 5.25], [5.25, 12.25, -3.5, 5], [13.75, 21.75, -7, 5], [22.75, 30.25, -3.5, 5.25]] },
  { suffix: "-slim", sel: ".band.slim", fs: 14, restY: 1, top: -18, bottom: 20, left: -46.3, xTop: -2.75, base: 5.25,
    glyphs: [[-28.75, -22.25, -2.75, 5.25], [-28.75, -26.75, -2.75, 7.5], [-20.75, -13.75, -2.75, 5.25], [-12.75, -5.25, -5, 5],
      [-4.25, 2.75, -2.75, 5.25], [4.75, 11.75, -2.75, 5], [13.25, 20.25, -5.75, 5], [21.25, 28.25, -2.75, 5.25]] },
];

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const r2 = (v) => (Math.abs(v) < 0.005 ? 0 : +v.toFixed(2));
const r4 = (v) => (Math.abs(v) < 5e-5 ? 0 : +v.toFixed(4));
const pct = (t) => r4((t / WRAP) * 100);
const steps = (step) => { const a = []; for (let p = 0; p <= 100 + 1e-9; p += step) a.push(+p.toFixed(4)); return a; };

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

// periodic cubic spline through equally spaced keys, parameter in key units
function periodicSpline(v) {
  const n = v.length, A = v.map((_, i) => v.map((_, j) => (j === i ? 4 : j === (i + 1) % n || j === (i + n - 1) % n ? 1 : 0)));
  const r = v.map((_, i) => 6 * (v[(i + 1) % n] - 2 * v[i] + v[(i + n - 1) % n]));
  for (let c = 0; c < n; c++) for (let i = 0; i < n; i++) if (i !== c) {
    const f = A[i][c] / A[c][c];
    for (let j = 0; j < n; j++) A[i][j] -= f * A[c][j];
    r[i] -= f * r[c];
  }
  const M = r.map((x, i) => x / A[i][i]);
  return (s) => {
    s = ((s % n) + n) % n;
    const i = Math.floor(s) % n, j = (i + 1) % n, t = s - Math.floor(s), u = 1 - t;
    return u * v[i] + t * v[j] + ((u ** 3 - u) * M[i] + (t ** 3 - t) * M[j]) / 6;
  };
}
const KEYS = DEPTH.length;      // quarter turns in the whole wrap
const depthAt = periodicSpline(DEPTH);
const depthScale = (z) => 1 + ((NEAR - FAR) / 2) * z + ((NEAR + FAR) / 2 - 1) * z * z;
const depthLight = (z) => 1 - DIM * Math.min(0, z) ** 2;

// orbit phase (quarter turns, 0…KEYS) at time t, shared by both bands
const phaseAt = (t) => KEYS * travel(clamp01((t - T_LAUNCH) / (T_ARRIVE - T_LAUNCH)));
const timeOf = (s) => { let lo = T_LAUNCH, hi = T_ARRIVE; for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; (phaseAt(m) < s ? (lo = m) : (hi = m)); } return (lo + hi) / 2; };
const zAt = (t) => depthAt(phaseAt(t));

// trail and sparks: present once the star is moving, gone before it lands, dimmer on the far side
const presence = (t) => {
  const a = T_LAUNCH + 0.08, b = T_LAUNCH + 0.4, c = T_ARRIVE - 0.8, d = T_ARRIVE - 0.12;
  return t <= a || t >= d ? 0 : t < b ? cubicBezier([0.3, 0, 0.2, 1])((t - a) / (b - a)) : t <= c ? 1 : 1 - cubicBezier([0.5, 0, 0.7, 0.4])((t - c) / (d - c));
};
const trailLight = (t) => presence(t) * (1 - 0.45 * Math.min(0, zAt(t)) ** 2);
const trailScale = (t) => Math.min(1, depthScale(zAt(t)));
// sparks twinkle on an uneven 0.11s/0.17s beat, so they never pulse in step with the star
const twinkleKeys = (() => {
  const t0 = T_LAUNCH + 0.2, t1 = T_ARRIVE - 1.0, t2 = T_ARRIVE - 0.3, k = [[0, 0], [t0, 0]];
  for (let t = t0 + 0.3, hi = true; t < t1; t += hi ? 0.11 : 0.17, hi = !hi) k.push([t, hi ? 1 : 0.45]);
  k.push([t1, 0.9], [t2, 0], [WRAP, 0]);
  return k;
})();
const twinkle = (t) => {
  const i = twinkleKeys.findIndex(([kt]) => kt > t);
  if (i <= 0) return 0;
  const [ta, va] = twinkleKeys[i - 1], [tb, vb] = twinkleKeys[i];
  return va + (vb - va) * inOut((t - ta) / (tb - ta));
};
const sparkLight = (t) => twinkle(t) * (1 - 0.45 * Math.min(0, zAt(t)) ** 2);

// halo alpha → the radius (fraction of the halo's radius) beyond which it is fainter than `alpha`
const haloReach = (alpha, opacity) => {
  for (let i = 1; i < HALO_STOPS.length; i++) {
    const [r0, a0] = HALO_STOPS[i - 1], [r1, a1] = HALO_STOPS[i];
    if (a1 * opacity <= alpha) return a0 * opacity <= alpha ? r0 : r0 + ((a0 * opacity - alpha) / ((a0 - a1) * opacity)) * (r1 - r0);
  }
  return 1;
};

function polyline(pts) {
  const len = [0];
  for (let i = 1; i < pts.length; i++) len.push(len[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = len[len.length - 1];
  const at = (d) => {
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; (len[m] < d ? (lo = m) : (hi = m)); }
    const k = (d - len[lo]) / (len[hi] - len[lo] || 1);
    return [pts[lo][0] + k * (pts[hi][0] - pts[lo][0]), pts[lo][1] + k * (pts[hi][1] - pts[lo][1])];
  };
  return { len, total, at };
}
const blur = (arr, h) => arr.map((_, i) => { let s = 0; for (let j = i - h; j <= i + h; j++) s += arr[Math.min(arr.length - 1, Math.max(0, j))]; return s / (2 * h + 1); });
const erode = (arr, w) => arr.map((_, i) => { let m = Infinity; for (let j = i - w; j <= i + w; j++) m = Math.min(m, arr[Math.min(arr.length - 1, Math.max(0, j))]); return m; });

const debug = {};

function build(b) {
  const { fs, glyphs } = b;
  const xc = (b.xTop + b.base) / 2;
  const restX = REST_X * fs, swingX = SWING_X * fs, endX = b.left + GAP_BAND + RAY_H * fs * BREATHE + 0.3;
  // two loops turning the same way. First: behind across the upper half of the letters (tight), then a wide
  // swing under the word. Second: in front, riding high so the near-side star covers the letters, then behind
  // across their lower half, rising into rest. Keys per quarter turn: rest, behind, left end, under, swing,
  // front, left end, behind.
  const yUnder = (b.base + GAP_INK + b.bottom - GAP_BAND) / 2;
  const X = [restX, (restX + endX) / 2, endX, (endX + swingX) / 2 + 0.3 * fs, swingX, (swingX + endX) / 2, endX, (endX + restX) / 2];
  const Y = [b.restY, xc - 0.17 * fs, xc + 0.45 * fs, yUnder, xc + 0.3 * fs, xc - 0.45 * fs, xc - 0.3 * fs, xc + 0.2 * fs];
  const fx = periodicSpline(X), fy = periodicSpline(Y);

  // dense curve by phase, then resampled by arc length into Catmull-Rom cubic Béziers (closed, so the
  // tangents at the rest point come from the curve itself)
  const N = 8000, dense = [];
  for (let i = 0; i <= N; i++) { const s = (KEYS * i) / N; dense.push([fx(s), fy(s)]); }
  const dl = polyline(dense);
  const lenAt = (s) => { const q = (s / KEYS) * N, i = Math.min(N - 1, Math.floor(q)); return dl.len[i] + (q - i) * (dl.len[i + 1] - dl.len[i]); };
  const atLen = (d) => dl.at(((d % dl.total) + dl.total) % dl.total);
  const step = dl.total / SEGMENTS, P = [];
  for (let i = -1; i <= SEGMENTS + 1; i++) P.push(atLen(i * step));
  P[1] = P[SEGMENTS + 1] = [restX, b.restY];
  let d = `M${r2(P[1][0])} ${r2(P[1][1])}`;
  const bez = [];
  for (let i = 1; i <= SEGMENTS; i++) {
    const p0 = P[i], p1 = P[i + 1], c1 = [p0[0] + (p1[0] - P[i - 1][0]) / 6, p0[1] + (p1[1] - P[i - 1][1]) / 6];
    const c2 = [p1[0] - (P[i + 2][0] - p0[0]) / 6, p1[1] - (P[i + 2][1] - p0[1]) / 6];
    d += `C${r2(c1[0])} ${r2(c1[1])} ${r2(c2[0])} ${r2(c2[1])} ${r2(p1[0])} ${r2(p1[1])}`;
    bez.push([p0, c1, c2, p1].map(([x, y]) => [r2(x), r2(y)]));
  }
  // the path as the browser will see it (rounded control points)
  const fine = [];
  for (const [p0, c1, c2, p1] of bez) for (let k = 0; k < 80; k++) {
    const t = k / 80, u = 1 - t;
    fine.push([0, 1].map((j) => u ** 3 * p0[j] + 3 * u * u * t * c1[j] + 3 * u * t * t * c2[j] + t ** 3 * p1[j]));
  }
  fine.push(bez[bez.length - 1][3]);
  const path = polyline(fine);

  // offset-distance keyframes: the orbit phase mapped to arc length, sampled every STEP and run linearly
  const distKeys = steps(STEP).map((p) => [p, lenAt(phaseAt((p / 100) * WRAP)) / dl.total]);
  distKeys[distKeys.length - 1][1] = 1;
  const keyLerp = (keys, t) => {
    const q = ((t / WRAP) * 100) / (keys[1][0] - keys[0][0]), i = Math.min(keys.length - 2, Math.max(0, Math.floor(q))), k = q - i;
    const A = keys[i][1], Z = keys[i + 1][1];
    return typeof A === "number" ? A + k * (Z - A) : Object.fromEntries(Object.keys(A).map((p) => [p, A[p] + k * (Z[p] - A[p])]));
  };
  const posAt = (t) => path.at(path.total * keyLerp(distKeys, Math.min(WRAP, Math.max(0, t))));

  // clearance: the star as two bars (long vertical rays, shorter horizontal ones), widened for RMAX of tilt
  const tilt = Math.sin((RMAX * Math.PI) / 180);
  const bars = [[WAIST + RAY_V * tilt, RAY_V], [RAY_H, WAIST + RAY_H * tilt]];
  const glyphGap = (x0, x1, y0, y1) => Math.min(...glyphs.map(([gx0, gx1, gy0, gy1]) => {
    const dx = Math.max(0, gx0 - x1, x0 - gx1), dy = Math.max(0, gy0 - y1, y0 - gy1);
    return dx || dy ? Math.hypot(dx, dy) : -1;
  }));
  const boxes = ([x, y], s) => bars.map(([hx, hy]) => [x - hx * fs * s * BREATHE, x + hx * fs * s * BREATHE, y - hy * fs * s * BREATHE, y + hy * fs * s * BREATHE]);
  const inkGap = (p, s) => Math.min(...boxes(p, s).map((q) => glyphGap(...q)));
  const bandGap = (p, s) => Math.min(...boxes(p, s).map(([x0, , y0, y1]) => Math.min(y0 - b.top, b.bottom - y1, x0 - b.left))) - GAP_BAND;
  
  // the moments the word changes layer: star and trail clear of every glyph for ±20ms at the largest they can be
  // then, nearest to where the depth crosses zero (the glow has its own layer under the word, so it never swaps)
  const DT = 0.005, n = Math.round(WRAP / DT), T = [...Array(n + 1)].map((_, i) => i * DT);
  // trail pieces are ellipses drawn out along the path: bounding box of the rotated ellipse
  const pieceGap = (t, along, across) => {
    const [x, y] = posAt(t), [x0, y0] = posAt(t - 0.004), [x1, y1] = posAt(t + 0.004), a = Math.atan2(y1 - y0, x1 - x0);
    const hx = Math.hypot(along * Math.cos(a), across * Math.sin(a)), hy = Math.hypot(along * Math.sin(a), across * Math.cos(a));
    return glyphGap(x - hx, x + hx, y - hy, y + hy);
  };
  const switchGap = (t, size) => {
    let g = inkGap(posAt(t), size(t));
    for (const [dd, r] of DOTS) if (trailLight(t - dd) > 0.02) g = Math.min(g, pieceGap(t - dd, r * trailScale(t - dd), (r / 2.8) * trailScale(t - dd)));
    for (const dd of SPARKS) if (sparkLight(t - dd) > 0.02) g = Math.min(g, pieceGap(t - dd, 0.225 * fs * trailScale(t - dd), 0.12 * fs * trailScale(t - dd)));
    return g;
  };
  const windowGap = (t, size) => Math.min(...[-0.02, -0.01, 0, 0.01, 0.02].map((o) => switchGap(t + o, size)));
  const findSwitch = (s0, s1, size, label) => {
    const t0 = timeOf(s0), t1 = timeOf(s1);
    let cross = null;
    for (let t = t0; t < t1; t += DT) if (Math.sign(zAt(t)) !== Math.sign(zAt(t + DT))) { cross = t; break; }
    cross ??= (t0 + t1) / 2;
    let best = null, bestGap = -Infinity, at = null;
    for (let t = t0; t <= t1; t += DT) {
      const g = windowGap(t, size);
      if (g > bestGap) { bestGap = g; at = t; }
      if (g >= GAP_SWITCH && (best === null || Math.abs(t - cross) < Math.abs(best - cross))) best = t;
    }
    if (best === null) throw new Error(`${b.sel}: no clean ${label} switch between ${t0.toFixed(2)}s and ${t1.toFixed(2)}s (best gap ${bestGap.toFixed(2)}px at ${at.toFixed(2)}s)`);
    return +best.toFixed(3);
  };
  // size: the depth size, held to what fits — the band always, and the ink across the middle of the under pass
  // (past the p's descender, before the rise in front of the right-hand letters) — eroded then blurred in time
  // so it changes gently and never exceeds the raw limit
  const tSwing = timeOf(4), UNDER = [3.1, 3.45];
  let inkFrom = timeOf(UNDER[0]);
  const inkTo = timeOf(UNDER[1]);
  const needInk = (t) => t >= inkFrom - 0.02 && t <= inkTo;
  const fit = (t) => {
    const p = posAt(t), ok = (s) => bandGap(p, s) >= 0.25 && (!needInk(t) || inkGap(p, s) - GAP_INK >= 0.25);
    if (ok(NEAR * 1.06)) return NEAR * 1.06;
    let lo = 0, hi = NEAR * 1.06;
    for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; (ok(m) ? (lo = m) : (hi = m)); }
    return lo;
  };
  let lim;
  const fitAll = () => (lim = blur(blur(erode(T.map(fit), 20), 10), 10));
  const limitAt = (t) => lim[Math.min(n, Math.max(0, Math.round(t / DT)))];
  fitAll();
  const sizeUpper = (t) => Math.min(depthScale(zAt(t)), limitAt(t)) * 1.06;
  const S1 = findSwitch(UNDER[0], UNDER[1], sizeUpper, "behind → front");
  const S2 = findSwitch(5.4, 6.8, sizeUpper, "front → behind");

  // banking: tilt into the turn in proportion to how fast the heading changes, smoothed
  const heading = T.map((t) => { const [x0, y0] = posAt(t - DT), [x1, y1] = posAt(t + DT); return Math.atan2(y1 - y0, x1 - x0); });
  let bank = heading.map((h, i) => { if (!i) return 0; let dd = h - heading[i - 1]; dd -= 2 * Math.PI * Math.round(dd / (2 * Math.PI)); return Math.hypot(...posAt(T[i] + DT).map((v, j) => v - posAt(T[i] - DT)[j])) > 0.01 ? dd / DT : 0; });
  bank = blur(blur(bank, 16), 16).map((w) => Math.max(-BANK, Math.min(BANK, w * 0.7)));
  const bankAt = (t) => bank[Math.min(n, Math.max(0, Math.round(t / DT)))];
  const [ax0, ay0] = posAt(T_ARRIVE - 0.12), [ax1, ay1] = posAt(T_ARRIVE), al = Math.hypot(ax1 - ax0, ay1 - ay0) || 1;
  const arrive = [(ax1 - ax0) / al, (ay1 - ay0) / al];

  const pose = (t) => {
    // anticipation: dip down and back against the leftward launch, gather (squash), lean back
    const a = t < 0.3 ? inOut(t / 0.3) : t < T_LAUNCH ? 1 : 1 - release(clamp01((t - T_LAUNCH) / 0.22));
    const stretch = 0.08 * bump(t, T_LAUNCH, T_LAUNCH + 0.4);       // launch: drawn out along the glide
    const lead = bump(t, T_ARRIVE - 0.35, WRAP);                     // follow-through: carries on past rest, settles back
    const bloom = 0.045 * bump(t, T_ARRIVE - 0.1, WRAP);
    const z = zAt(t), s = Math.min(depthScale(z), limitAt(t)) * (1 + bloom);
    return {
      x: a * 0.05 * fs + lead * 0.05 * fs * arrive[0], y: a * 0.08 * fs + lead * 0.05 * fs * arrive[1], r: 7 * a + bankAt(t),
      sx: s * (1 + 0.05 * a) * (1 + stretch), sy: s * (1 - 0.08 * a) * (1 - 0.5 * stretch), o: depthLight(z),
    };
  };
  const keys = steps(STEP).map((p) => [p, pose((p / 100) * WRAP)]);
  keys[keys.length - 1][1] = { x: 0, y: 0, r: 0, sx: 1, sy: 1, o: 1 };
  const tf = (k) => `translate(${r2(k.x)}px,${r2(k.y)}px) rotate(${r2(k.r)}deg) scale(${r4(k.sx)},${r4(k.sy)})`;

  // reactive glow: draws in with the anticipation, brightens and swells on the near side, dims behind, eases
  // back after landing; its visible edge is held inside the band
  const haloRaw = (t) => {
    const a = t < 0.3 ? inOut(t / 0.3) : t < T_LAUNCH ? 1 : 1 - release(clamp01((t - T_LAUNCH) / 0.3));
    const z = zAt(t), near = Math.max(0, z), settle = bump(t, T_ARRIVE - 0.2, WRAP);
    return { o: HALO_O * (1 + 0.4 * z) * (1 - 0.25 * a) * (1 + 0.15 * settle), s: (1 + 0.14 * near * near) * (1 - 0.12 * a) * (1 + 0.05 * settle) };
  };
  const haloCap = T.map((t) => {
    const k = keyLerp(keys, t), [x, y] = posAt(t), cx = x + k.x, cy = y + k.y, h = haloRaw(t);
    const reach = haloReach(SEEN, h.o) * fs * Math.max(k.sx, k.sy) * GLOW_PEAK;
    return reach > 0 ? (Math.min(cy - b.top, b.bottom - cy, cx - b.left) - 1.25) / reach : 9;
  });
  const capS = blur(blur(erode(haloCap, 16), 8), 8);
  const halo = (t) => { const h = haloRaw(t); return { o: h.o, s: Math.min(h.s, capS[Math.min(n, Math.max(0, Math.round(t / DT)))]) }; };
  const haloKeys = steps(STEP_FX).map((p) => [p, halo((p / 100) * WRAP)]);
  haloKeys[haloKeys.length - 1][1] = { o: HALO_O, s: 1 };

  // re-check everything from the emitted keyframes, linearly interpolated as the browser does
  let worstBand = [Infinity], worstInk = [Infinity], worstGlow = [Infinity], maxR = 0, peak = 0, prev = posAt(0);
  const size = { behind: Infinity, front: 0, under: Infinity };
  for (let i = 1; i <= n; i++) {
    const t = T[i], k = keyLerp(keys, t), [x, y] = posAt(t), p = [x + k.x, y + k.y], s = Math.max(k.sx, k.sy), z = zAt(t);
    const vb = bandGap(p, s); if (vb < worstBand[0]) worstBand = [vb, t];
    if (needInk(t)) { const vi = inkGap(p, s) - GAP_INK; if (vi < worstInk[0]) worstInk = [vi, t]; }
    const h = keyLerp(haloKeys, t), reach = haloReach(SEEN, h.o) * fs * s * h.s * GLOW_PEAK;
    const vg = Math.min(p[1] - b.top, b.bottom - p[1], p[0] - b.left) - reach; if (vg < worstGlow[0]) worstGlow = [vg, t];
    maxR = Math.max(maxR, Math.abs(k.r));
    peak = Math.max(peak, Math.hypot(x - prev[0], y - prev[1]) / DT); prev = [x, y];
    if (t > T_LAUNCH + 0.3 && t < T_ARRIVE - 0.3) {
      if (z < -0.8) size.behind = Math.min(size.behind, s);
      if (z > 0.8) size.front = Math.max(size.front, s);
      if (needInk(t)) size.under = Math.min(size.under, s);
    }
  }
  const sizeAt = (t) => { const k = keyLerp(keys, t); return Math.max(k.sx, k.sy); };
  const sw = [S1, S2].map((t) => windowGap(t, sizeAt));
  if (maxR + SWAY > RMAX) throw new Error(`${b.sel}: tilt ${(maxR + SWAY).toFixed(1)}° exceeds RMAX`);
  if (worstBand[0] < 0) throw new Error(`${b.sel}: star comes ${(-worstBand[0]).toFixed(2)}px inside the band gap at t=${worstBand[1].toFixed(3)}s`);
  if (worstInk[0] < 0) throw new Error(`${b.sel}: star comes ${(-worstInk[0]).toFixed(2)}px inside the ink gap at t=${worstInk[1].toFixed(3)}s`);
  if (worstGlow[0] < 1) throw new Error(`${b.sel}: glow reaches ${worstGlow[0].toFixed(2)}px from the band edge at t=${worstGlow[1].toFixed(3)}s`);
  if (Math.min(...sw) < GAP_SWITCH) throw new Error(`${b.sel}: layer switch not clear (${sw.map((g) => g.toFixed(2)).join(", ")}px)`);

  // when the star's center crosses the word's middle on each pass
  const across = (s0, s1) => { const t0 = timeOf(s0), t1 = timeOf(s1); for (let t = t0; t < t1; t += 0.001) if (Math.sign(posAt(t)[0]) !== Math.sign(posAt(t + 0.001)[0])) return +t.toFixed(3); return null; };
  const tBehind = across(0.5, 1.6), tUnder = across(2.5, 3.6), tFront = across(4.5, 5.6), tBehind2 = across(6.5, 7.6);
  const timeline = { launch: T_LAUNCH, behind: tBehind, leftEnd: +timeOf(2).toFixed(3), under: tUnder, toFront: S1, swing: +tSwing.toFixed(3),
    front: tFront, leftEnd2: +timeOf(6).toFixed(3), toBehind: S2, behind2: tBehind2, arrive: T_ARRIVE };
  console.log(`${b.sel}: path ${path.total.toFixed(0)}px (${SEGMENTS} Béziers), peak ${peak.toFixed(0)}px/s; star ${fs}px at rest, ` +
    `${(size.behind * fs).toFixed(1)}px behind, ${(size.front * fs).toFixed(1)}px in front, ≥${(size.under * fs).toFixed(1)}px under; ` +
    `air: band ${worstBand[0].toFixed(2)}px, ink ${worstInk[0].toFixed(2)}px, glow ${worstGlow[0].toFixed(2)}px, layer switches ${sw.map((g) => g.toFixed(2)).join("/")}px; tilt ${maxR.toFixed(1)}°`);
  console.log(`  ${Object.entries(timeline).map(([k, v]) => `${k} ${v}s`).join(" · ")}`);
  debug[b.sel] = { glyphs, band: b, dense: dense.filter((_, i) => i % 10 === 0), timeline,
    frames: T.filter((_, i) => i % 2 === 0).map((t) => ({ t, p: posAt(t), z: zAt(t), s: sizeAt(t), behind: t < S1 || t >= S2 })) };

  return { timeline, css: `${b.sel} .star-fx{--rest-x:${r2(restX)}px;--rest-y:${r2(b.restY)}px;` +
    `--t-behind:${tBehind}s;--t-front:${tFront}s;--t-behind2:${tBehind2}s;--t-switch:${S1}s ${S2}s}
${b.sel} .rig,${b.sel} .trail>*{offset-path:path("${d}")}
${b.suffix ? `.band.slim .mark.is-wrapping .rig{animation-name:star-wrap-path-slim}
.band.slim .mark.is-wrapping .body{animation-name:star-wrap-body-slim}
.band.slim .mark.is-wrapping .halo{animation-name:star-halo-slim}
.band.slim .mark.is-wrapping .ink{animation-name:star-depth-ink-slim}
.band.slim .mark.is-wrapping .base{animation-name:star-depth-base-slim}
.band.slim .mark.is-wrapping .trail>*{animation-name:star-wrap-path-slim,star-trail}
.band.slim .mark.is-wrapping .trail b{animation-name:star-wrap-path-slim,star-trail-spark}
` : ""}/* Arcs + slow in/slow out, ${fs}px word: the orbit phase runs on cubic-bezier(${TRAVEL.join(",")}) and is mapped to
   arc length here, so the projected speed peaks crossing the word and falls away into each end, like a real
   orbit seen edge-on. Held through the anticipation. */
@keyframes star-wrap-path${b.suffix}{
${distKeys.map(([p, f]) => `  ${r4(p)}%{offset-distance:${r4(f * 100)}%}`).join("\n")}
}
/* Depth, ${fs}px word: the word over the star (.ink) while it is behind, under it (.base) while it is in front.
   Stepped at ${S1}s (under the word) and ${S2}s (past the left end), when nothing is over a glyph. */
@keyframes star-depth-ink${b.suffix}{
  0%{opacity:1;animation-timing-function:steps(1,end)}
  ${pct(S1)}%{opacity:0;animation-timing-function:steps(1,end)}
  ${pct(S2)}%,100%{opacity:1}
}
@keyframes star-depth-base${b.suffix}{
  0%{opacity:0;animation-timing-function:steps(1,end)}
  ${pct(S1)}%{opacity:1;animation-timing-function:steps(1,end)}
  ${pct(S2)}%,100%{opacity:0}
}
/* Body, ${fs}px word, sampled every ${STEP}%:
   anticipation 0–${T_LAUNCH}s: dips down and back, squashes, leans back against the launch;
   squash & stretch: drawn out along the glide as it launches;
   depth: ${(FAR * fs).toFixed(1)}px and ${Math.round(DIM * 100)}% dimmer on the far side, up to ${(size.front * fs).toFixed(1)}px on the near side (perspective);
   secondary action: banks into each turn, up to ${BANK}°;
   staging: under the word the size is held to the clearance (≥${(size.under * fs).toFixed(1)}px here);
   follow-through ${(T_ARRIVE - 0.35).toFixed(2)}–${WRAP}s: carries on past rest, blooms ${4.5}% on landing, settles to identity. */
@keyframes star-wrap-body${b.suffix}{
${keys.map(([p, k]) => `  ${r4(p)}%{transform:${tf(k)};opacity:${r4(k.o)}}`).join("\n")}
}
/* Reactive glow, ${fs}px word, every ${STEP_FX}%: draws in with the anticipation, swells and brightens as the star
   comes near, dims behind the word, and eases back after the star lands (follow-through). */
@keyframes star-halo${b.suffix}{
${haloKeys.map(([p, h]) => `  ${r4(p)}%{opacity:${r4(h.o)};transform:scale(${r4(h.s)})}`).join("\n")}
}` };
}

const built = bands.map(build);
const fx = steps(STEP_FX);
const css = `/* star-orbit.css — generated by gen-star-orbit.mjs; edit the generator, not this file.
   Triggered wrap, ${WRAP}s: ${T_LAUNCH}s anticipation, ${(T_ARRIVE - T_LAUNCH).toFixed(2)}s orbit (behind, under, front, behind), ${(WRAP - T_ARRIVE).toFixed(1)}s follow-through. */
${built.map((x) => x.css).join("\n")}

/* Overlapping action: the trail rides the same path on a delay, so it lengthens with speed and shortens into
   the ends. Each piece replays the star's depth where it is (smaller, dimmer behind) and sits in the same
   layer, so the word hides it exactly as it hides the star. Gone before each piece reaches rest. */
@keyframes star-trail{
${fx.map((p) => { const t = (p / 100) * WRAP; return `  ${r4(p)}%{opacity:${r4(trailLight(t))};transform:scale(${r4(trailScale(t))})}`; }).join("\n")}
}
/* Sparks: later and on their own uneven twinkle, drawn out along the path, with the same depth. */
@keyframes star-trail-spark{
${fx.map((p) => { const t = (p / 100) * WRAP, k = trailScale(t); return `  ${r4(p)}%{opacity:${r4(sparkLight(t))};transform:scale(${r4(1.5 * k)},${r4(0.8 * k)})}`; }).join("\n")}
}
`;
writeFileSync(join(dirname(fileURLToPath(import.meta.url)), "star-orbit.css"), css);
if (process.argv.includes("--debug")) writeFileSync("/tmp/star-orbit-debug.json", JSON.stringify(debug));
