// Proof of the header star motion, both band sizes (56px empty state, 40px slim):
//   star-motion-v2.webm  9 s at 60 fps: idle (with a glint), wrap triggered at 2.0 s, settle, still
//   star-motion-v2.png   6 frames: idle, anticipation, first coil, swing, second coil, settle
// Frames come from the live CSS animations, seeked with the Web Animations API on one shared timeline.
// Checks (non-zero exit on failure):
//   pixel audit over the whole timeline every 40 ms at 2×: the star, trail and sparks stay ≥2px inside the
//   band and ≥1.5px off the letters; the glow never reaches the band's border;
//   live run of one wrap: frame pacing and layout count, timed with requestAnimationFrame.
//   CHROME_PATH=/usr/bin/chromium node shoot-motion.mjs [outdir]     (webm needs ffmpeg with libvpx-vp9)
import { chromium } from "playwright-core";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.argv[2] ?? join(here, "screens"));
const url = (route) => `${pathToFileURL(join(here, "index.html")).href}#${route}`;
const TRIGGER = 2.0, WRAP = 4.4, LENGTH = 9.0, FPS = 60, SCALE = 4, CLIP_W = 216;
const STRIP = [[0.75, "idle · glint"], [2.3, "anticipation"], [2.95, "first coil"], [3.75, "swing"], [4.65, "second coil"], [5.95, "settle"]];
const bands = [{ route: "empty", label: "56px band · empty state" }, { route: "answer", label: "40px slim band" }];
const WRAP_ANIM = /^star-(wrap|halo|trail)/;

mkdirSync(outDir, { recursive: true });
const work = mkdtempSync(join(tmpdir(), "star-motion-"));
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
let failed = false;

// One timeline for idle and wrap: idle animations run from 0, the wrap class goes on at TRIGGER and off at
// TRIGGER + WRAP (as the page script does when the glow finishes), transitions follow their own start.
async function stage(page, route) {
  await page.goto(url(route));
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate((id) => document.querySelector(`#${id} .band .mark`).classList.remove("is-wrapping"), route);
  await page.waitForTimeout(150);
  const r = await page.evaluate((id) => document.querySelector(`#${id} .band`).getBoundingClientRect().toJSON(), route);
  let phase = 0;
  const seek = async (t) => {
    const want = t < TRIGGER ? 0 : t < TRIGGER + WRAP ? 1 : 2;
    if (want !== phase) {
      await page.evaluate(([id, on]) => document.querySelector(`#${id} .band .mark`).classList.toggle("is-wrapping", on), [route, want === 1]);
      phase = want;
    }
    await page.evaluate(([t, trig, wrap, re]) => {
      for (const a of document.getAnimations()) {
        a.pause();
        const wrapAnim = new RegExp(re).test(a.animationName ?? "");
        const tr = a instanceof CSSTransition;
        a.currentTime = 1000 * (wrapAnim ? t - trig : tr ? (t < trig + wrap ? t - trig : t - trig - wrap) : t);
      }
    }, [t, TRIGGER, WRAP, WRAP_ANIM.source]);
  };
  return { band: r, seek };
}

// ---- pixel audit ----
const audit = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
for (const { route, label } of bands) {
  const { band, seek } = await stage(audit, route);
  const clip = { x: band.x, y: band.y, width: CLIP_W, height: band.height };
  const hideAll = await audit.addStyleTag({ content: ".band .star-fx *{opacity:0!important}" });
  const mask = (await audit.screenshot({ clip })).toString("base64");
  await hideAll.evaluate((n) => n.remove());
  const noHalo = await audit.addStyleTag({ content: ".band .halo{opacity:0!important}" });
  const core = [], light = [];
  for (let ms = 0; ms < LENGTH * 1000; ms += 40) { await seek(ms / 1000); core.push([ms, (await audit.screenshot({ clip })).toString("base64")]); }
  await noHalo.evaluate((n) => n.remove());
  for (let ms = 0; ms < LENGTH * 1000; ms += 40) { await seek(ms / 1000); light.push([ms, (await audit.screenshot({ clip })).toString("base64")]); }
  const res = await audit.evaluate(async ({ mask, core, light, h }) => {
    const load = async (b64) => { const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
      const c = new OffscreenCanvas(img.width, img.height), g = c.getContext("2d"); g.drawImage(img, 0, 0); return g.getImageData(0, 0, img.width, img.height); };
    const m = await load(mask), W = m.width, H = m.height, dpr = 2;
    // distance (device px) from every pixel to the nearest letter pixel, letters = clearly brighter than the band
    const text = [];
    for (let i = 0; i < W * H; i++) if (Math.max(m.data[4 * i], m.data[4 * i + 1], m.data[4 * i + 2]) > 90) text.push([i % W, (i / W) | 0]);
    const toText = new Float32Array(W * H).fill(1e9);
    for (let i = 0; i < W * H; i++) { const x = i % W, y = (i / W) | 0; for (const [tx, ty] of text) { const d = (tx - x) ** 2 + (ty - y) ** 2; if (d < toText[i]) toText[i] = d; } }
    const diff = (d, i) => Math.abs(d.data[4 * i] - m.data[4 * i]) + Math.abs(d.data[4 * i + 1] - m.data[4 * i + 1]) + Math.abs(d.data[4 * i + 2] - m.data[4 * i + 2]);
    let edge = [1e9], ink = [1e9], glow = [1e9];
    for (const [ms, b64] of core) {
      const d = await load(b64);
      for (let i = 0; i < W * H; i++) {
        if (diff(d, i) <= 48) continue;
        const x = i % W, y = (i / W) | 0;
        const e = Math.min(y - 2, H - 3 - y, x - 2) / dpr, k = Math.sqrt(toText[i]) / dpr;
        if (e < edge[0]) edge = [e, ms];
        if (k < ink[0]) ink = [k, ms];
      }
    }
    for (const [ms, b64] of light) {
      const d = await load(b64);
      for (let i = 0; i < W * H; i++) {
        if (diff(d, i) <= 16) continue;
        const x = i % W, y = (i / W) | 0, e = Math.min(y - 2, H - 3 - y, x - 2) / dpr;
        if (e < glow[0]) glow = [e, ms];
      }
    }
    return { edge, ink, glow };
  }, { mask, core, light, h: band.height });
  const ok = res.edge[0] >= 2 && res.ink[0] >= 1.5 && res.glow[0] >= 0.5;
  failed ||= !ok;
  const at = ([v, ms]) => `${v.toFixed(1)}px (t=${(ms / 1000).toFixed(2)}s)`;
  console.log(`${label}: ${ok ? "ok" : "FAIL"} — star/trail to band edge ${at(res.edge)}, to letters ${at(res.ink)}; glow to band border ${at(res.glow)}`);
}
await audit.close();

// ---- live pacing: one wrap in real time ----
{
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Performance.enable");
  for (const { route, label } of bands) {
    await page.goto(url(route));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForFunction((id) => !document.querySelector(`#${id} .band .mark`).classList.contains("is-wrapping"), route, { timeout: 8000 });
    await page.waitForTimeout(600);
    const metric = async () => Object.fromEntries((await cdp.send("Performance.getMetrics")).metrics.map((m) => [m.name, m.value]));
    // the trigger frame may take one layout (seen intermittently, also with every star animation disabled);
    // the motion frames after it must take none
    const m0 = await metric();
    const t0 = await page.evaluate((id) => new Promise((done) => {
      document.querySelector(`#${id} .band .mark`).dispatchEvent(new PointerEvent("pointerenter"));
      requestAnimationFrame(() => requestAnimationFrame(done));
    }), route);
    const m1 = await metric();
    const gaps = await page.evaluate(([id, t0]) => new Promise((done) => {
      const mark = document.querySelector(`#${id} .band .mark`), ts = [t0];
      const tick = (t) => { ts.push(t); mark.classList.contains("is-wrapping") ? requestAnimationFrame(tick) : done(ts.slice(1).map((v, i) => v - ts[i])); };
      requestAnimationFrame(tick);
    }), [route, t0]);
    const m2 = await metric();
    const worst = Math.max(...gaps), slow = gaps.filter((g) => g > 20).length;
    const atTrigger = m1.LayoutCount - m0.LayoutCount, during = m2.LayoutCount - m1.LayoutCount;
    const styleMs = ((m2.RecalcStyleDuration - m1.RecalcStyleDuration) * 1000) / gaps.length;
    const ok = atTrigger <= 1 && during === 0 && slow <= 2;
    failed ||= !ok;
    console.log(`${label} live wrap: ${ok ? "ok" : "FAIL"} — ${gaps.length} frames, ${(1000 / (gaps.reduce((a, b) => a + b, 0) / gaps.length)).toFixed(1)} fps, ` +
      `worst gap ${worst.toFixed(1)}ms, ${slow} over 20ms; layouts ${atTrigger} at trigger, ${during} during the wrap; style recalc ${styleMs.toFixed(3)}ms/frame`);
  }
  await page.close();
}

// ---- frames: strip and video ----
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: SCALE });
const strip = {};
for (const { route } of bands) {
  const { band, seek } = await stage(page, route);
  const clip = { x: band.x - 4, y: band.y - 4, width: CLIP_W, height: band.height + 8 };
  mkdirSync(join(work, route));
  const want = new Map(STRIP.map(([t]) => [Math.round(t * FPS), t]));
  strip[route] = [];
  for (let i = 0; i < LENGTH * FPS; i++) {
    await seek(i / FPS);
    const buf = await page.screenshot({ clip, path: join(work, route, `${String(i).padStart(4, "0")}.png`) });
    if (want.has(i)) strip[route].push(buf.toString("base64"));
  }
}

const font = (file) => pathToFileURL(join(here, "fonts", file)).href;
const stripFile = join(work, "strip.html");
writeFileSync(stripFile, `<!doctype html><meta charset="utf-8"><style>
@font-face{font-family:M;src:url("${font("JetBrainsMono-Regular.woff2")}")}
@font-face{font-family:M;font-weight:600;src:url("${font("JetBrainsMono-SemiBold.woff2")}")}
body{margin:0;background:#0B1220;color:#AEB7C8;font:400 12px/16px M}
.wrap{display:inline-block;padding:20px 24px 24px}
h1{margin:0 0 4px;font:600 13px/20px M;color:#E6EAF2}
p{margin:0 0 4px}
.row{display:grid;grid-template-columns:repeat(6,${CLIP_W * 1.5}px);gap:10px;margin-top:10px}
.lbl{margin:16px 0 0;font-size:11px;letter-spacing:.06em;text-transform:uppercase}
img{display:block;width:100%;border:1px solid #263352;border-radius:6px}
.t{margin-top:6px}.t b{font-weight:600;color:#E6EAF2}
</style><div class="wrap"><h1>polaris · header star, motion v2</h1>
<p>Idle: 2.5% breath and glow on 5.2s, ±2.5° sway on 8.6s, glints on 7.3s / 11.9s. Wrap (on load, hover, focus, click), 4.4s:
0.38s anticipation, two coils on offset-path with a lagging tail and sparks, 0.5s follow-through. Frames at 1.5×; t from page start, wrap triggered at ${TRIGGER}s.</p>
${bands.map(({ route, label }) => `<div class="lbl">${label}</div><div class="row">${strip[route].map((b64, i) =>
  `<div><img src="data:image/png;base64,${b64}"><div class="t"><b>t = ${STRIP[i][0].toFixed(2)} s</b> · ${STRIP[i][1]}</div></div>`).join("")}</div>`).join("")}
</div>`);
const sheet = await browser.newPage({ viewport: { width: 2200, height: 400 }, deviceScaleFactor: 2 });
await sheet.goto(pathToFileURL(stripFile).href);
await sheet.evaluate(() => document.fonts.ready);
await sheet.locator(".wrap").screenshot({ path: join(outDir, "star-motion-v2.png") });
await browser.close();

execFileSync("ffmpeg", ["-y", "-loglevel", "error",
  "-framerate", String(FPS), "-i", join(work, bands[0].route, "%04d.png"),
  "-framerate", String(FPS), "-i", join(work, bands[1].route, "%04d.png"),
  "-filter_complex", "[0]pad=iw:ih+16:0:0:color=0x0B1220[a];[a][1]vstack=inputs=2,format=yuv420p",
  "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "22", "-row-mt", "1", "-r", String(FPS), join(outDir, "star-motion-v2.webm")]);
rmSync(work, { recursive: true, force: true });
console.log(`wrote ${join(outDir, "star-motion-v2.png")} and star-motion-v2.webm`);
process.exit(failed ? 1 : 0);
