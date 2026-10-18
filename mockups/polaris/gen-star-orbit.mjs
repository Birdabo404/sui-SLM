// Generates star-orbit.css: the header star's dwell + loop around the "polaris" wordmark.
//   node gen-star-orbit.mjs
// The loop is a rounded rectangle around the word's ink box, sampled by arc length so the only speed
// profile is one ease-in-out-sine over the whole trip. Keyframes are dense (every 0.5 % of the cycle) with
// linear timing between them, and use transform only: translate + rotate + scale.
// A 14px star cannot circle a 14px word inside the 40px slim band at full size, so it eases down to the
// largest scale that fits while in transit and back to full size at rest.
// Every sample is checked (as bounding boxes) against the measured ink box and the band edges; the script
// throws on contact.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const CYCLE = 7.2;         // s, one dwell + one loop
const DWELL = 1.2;         // s at rest beside the wordmark before each loop
const STEP = 0.5;          // % of the cycle between keyframes
const A = 2.75;            // em, half-width of the loop
const REST_Y = -0.1;       // em, rest point on the right side, slightly above the word's center
const GAP = 0.05;          // em, minimum air between star and ink
const BOB = 0.06;          // em, star-float amplitude (polaris.css); it sits inside the orbit's scale
const BAND_AIR = 3.5;      // px, minimum air between the star's tips and the band's inner edge
const TURN = 90;           // deg per loop; the 4-point star is 90°-symmetric, so the loop is seamless
const WORD_HALF = 2.1;     // em, half of the 7-character mono word box
const INK_X = 2.07;        // em, measured ink half-width of "polaris" in JetBrains Mono 600
const MARK_PAD_LEFT = 0.35;// em, .band .mark padding-left that leaves room for the left-hand pass

// Bands, measured in the mock: font size, band height, horizontal band padding, vertical ink half-extent
// (slim: the mark is nudged up 1px, which also centers the ink optically).
const bands = [
  { name: "star-orbit",      fs: 15, height: 56, pad: 16, inkY: 0.567, sCap: 0.8 },
  { name: "star-orbit-slim", fs: 14, height: 40, pad: 12, inkY: 0.5,   sCap: 1 },
];

const smoothstep = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
const easeInOutSine = (u) => (1 - Math.cos(Math.PI * u)) / 2;
const f = (v, d = 4) => (Math.abs(v) < 5e-5 ? 0 : +v.toFixed(d));

function build({ name, fs, height, pad, inkY, sCap }) {
  const px = 1 / fs;
  const innerHalf = height / 2 - 1;                                   // inside the 1px border
  const reach = (s) => s * (0.5 + BOB);                               // star tip + float, both scaled with the orbit
  const sFit = (innerHalf * px - BAND_AIR * px - inkY - GAP) / (2 * (0.5 + BOB));
  const sMin = Math.min(sCap, Math.floor(sFit * 100) / 100);
  const B = inkY + GAP + reach(sMin);

  // Rounded rectangle; the corner radius is 3/4 of the vertical air so the arcs clear the ink box corners.
  // Starts at the rest point and runs up the right side first (counter-clockwise on screen).
  const R = 0.75 * (B - inkY), cx = A - R, cy = B - R, pts = [];
  const line = (x0, y0, x1, y1) => { const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 0.002);
    for (let i = 0; i < n; i++) pts.push([x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * i) / n]); };
  const arc = (ox, oy, a0, a1) => { const n = Math.ceil((R * Math.abs(a1 - a0)) / 0.002);
    for (let i = 0; i < n; i++) { const a = a0 + ((a1 - a0) * i) / n; pts.push([ox + R * Math.cos(a), oy + R * Math.sin(a)]); } };
  const Q = Math.PI / 2;
  line(A, REST_Y, A, -cy);   arc(cx, -cy, 0, -Q);
  line(cx, -B, -cx, -B);     arc(-cx, -cy, -Q, -2 * Q);
  line(-A, -cy, -A, cy);     arc(-cx, cy, -2 * Q, -3 * Q);
  line(-cx, B, cx, B);       arc(cx, cy, -3 * Q, -4 * Q);
  line(A, cy, A, REST_Y);    pts.push([A, REST_Y]);
  const SAMPLES = pts.length - 1, len = [0];
  for (let i = 1; i <= SAMPLES; i++) len.push(len[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = len[SAMPLES];
  const at = (d) => {
    let lo = 0, hi = SAMPLES;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; (len[mid] < d ? (lo = mid) : (hi = mid)); }
    const k = (d - len[lo]) / (len[hi] - len[lo] || 1);
    return [pts[lo][0] + k * (pts[hi][0] - pts[lo][0]), pts[lo][1] + k * (pts[hi][1] - pts[lo][1])];
  };
  const pose = (u) => {                                               // u = 0..1 through the travel phase
    const p = easeInOutSine(u);
    const [x, y] = at(p * total);
    const s = 1 - (1 - sMin) * smoothstep(0, 0.16, u) * (1 - smoothstep(0.84, 1, u));
    return { x, y, r: TURN * p, s };
  };

  // clearance: the star's tips sit at 0.5·s em from its center at any rotation; band air is checked net of BAND_AIR
  const leftLimit = -(WORD_HALF + MARK_PAD_LEFT + pad * px) + px;
  const worst = { ink: [Infinity], band: [Infinity], left: [Infinity] };
  const note = (k, v, u) => { if (v < worst[k][0]) worst[k] = [v, u]; };
  for (let i = 0; i <= 4000; i++) {
    const u = i / 4000, { x, y, s } = pose(u), h = reach(s);
    note("ink", Math.max(Math.abs(x) - (INK_X + h), Math.abs(y) - (inkY + h)), u);
    note("band", innerHalf * px - BAND_AIR * px - (Math.abs(y) + h), u);
    note("left", x - h - leftLimit, u);
  }
  const rest = pose(0);
  for (const [k, [v, u]] of Object.entries(worst)) {
    if (v < 0) {
      const { x, y, s } = pose(u);
      throw new Error(`${name}: star touches ${k} by ${(-v * fs).toFixed(2)}px at u=${u.toFixed(3)} (x ${x.toFixed(2)}em, y ${y.toFixed(2)}em, scale ${s.toFixed(2)})`);
    }
  }

  const frames = [`  0%,${f((DWELL / CYCLE) * 100, 3)}%{transform:${tf(rest)}}`];
  for (let pct = Math.ceil((DWELL / CYCLE) * 100 / STEP) * STEP; pct <= 100 + 1e-9; pct += STEP) {
    const u = ((pct / 100) * CYCLE - DWELL) / (CYCLE - DWELL);
    frames.push(`  ${f(pct, 3)}%{transform:${tf(pct >= 100 ? rest : pose(u))}}`);
  }
  console.log(`${name}: ${fs}px word, ${height}px band, loop ${(2 * A * fs).toFixed(1)}×${(2 * B * fs).toFixed(1)}px, ` +
    `star ${fs}px at rest → ${(sMin * fs).toFixed(1)}px in transit; min air: ink ${(worst.ink[0] * fs).toFixed(2)}px, ` +
    `band edge ${((worst.band[0] + BAND_AIR * px) * fs).toFixed(2)}px, left ${(worst.left[0] * fs).toFixed(2)}px; ${frames.length} keyframes`);
  return { rest, css: `@keyframes ${name}{\n${frames.join("\n")}\n}` };
}

function tf({ x, y, r, s }) {
  return `translate(${f(x)}em,${f(y)}em) rotate(${f(r, 2)}deg) scale(${f(s)})`;
}

const out = bands.map(build);
const restTf = tf(out[0].rest);
const css = `/* star-orbit.css — generated by gen-star-orbit.mjs (${CYCLE}s cycle: ${DWELL}s dwell, then one loop). Edit the generator, not this file. */
.band .orbit{transform:${restTf}}
${out.map((o) => o.css).join("\n")}
`;
writeFileSync(join(dirname(fileURLToPath(import.meta.url)), "star-orbit.css"), css);
