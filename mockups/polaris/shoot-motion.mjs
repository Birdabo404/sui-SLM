// Proof of the header star motion, for both band sizes (56px empty state, 40px slim):
//   star-motion.png   4 frames at t = 0, 1.5, 3, 4.5 s
//   star-motion.webm  4 s at 30 fps, t = 1.0–5.0 s (end of the dwell and most of one loop)
// Frames come from the live CSS animations seeked with the Web Animations API, captured at 4× around the band.
// Before that, a pixel audit renders one whole cycle every 40 ms at 2× and fails if rendered amber comes
// within 2px of the band's inner edge or 1.5px of the letters.
//   CHROME_PATH=/usr/bin/chromium node shoot-motion.mjs [outdir]     (webm needs ffmpeg with libvpx-vp9)
import { chromium } from "playwright-core";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.argv[2] ?? join(here, "screens"));
const CYCLE = 7.2, DWELL = 1.2, SCALE = 4, FPS = 30;
const STRIP_T = [0, 1.5, 3, 4.5];
const VIDEO_T0 = 1.0, VIDEO_T1 = 5.0;
const bands = [{ route: "empty", label: "56px band · empty state" }, { route: "answer", label: "40px slim band" }];

const phase = (t) => {
  const c = t % CYCLE;
  if (c <= DWELL) return "dwell";
  return `loop ${Math.round(((1 - Math.cos(Math.PI * ((c - DWELL) / (CYCLE - DWELL)))) / 2) * 100)}%`;
};

mkdirSync(outDir, { recursive: true });
const work = mkdtempSync(join(tmpdir(), "star-motion-"));
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: SCALE });

// Pixel check over one whole cycle: how close rendered amber gets to the band's inner edge and to the letters.
const audit = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
let auditFailed = false;
for (const { route, label } of bands) {
  await audit.goto(`${pathToFileURL(join(here, "index.html")).href}#${route}`);
  await audit.evaluate(() => document.fonts.ready);
  const r = await audit.evaluate((id) => document.querySelector(`#${id} .band`).getBoundingClientRect().toJSON(), route);
  const clip = { x: r.x, y: r.y, width: 176, height: r.height };
  const seek = (ms) => audit.evaluate((t) => document.getAnimations().forEach((a) => { a.pause(); a.currentTime = t; }), ms);
  await audit.addStyleTag({ content: ".band .float svg{opacity:0}" });
  const textMask = (await audit.screenshot({ clip })).toString("base64");
  await audit.addStyleTag({ content: ".band .float svg{opacity:1}" });
  const frames = [];
  for (let ms = 0; ms < CYCLE * 1000; ms += 40) { await seek(ms); frames.push([ms, (await audit.screenshot({ clip })).toString("base64")]); }
  const res = await audit.evaluate(async ({ textMask, frames, height }) => {
    const load = async (b64) => { const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
      const c = new OffscreenCanvas(img.width, img.height), g = c.getContext("2d"); g.drawImage(img, 0, 0);
      return g.getImageData(0, 0, img.width, img.height); };
    // letters (with their subpixel fringes) = anything clearly brighter than the band in the star-less frame;
    // star = any pixel that differs from that frame
    const m = await load(textMask), text = [];
    for (let i = 0; i < m.data.length; i += 4) {
      if (Math.max(m.data[i], m.data[i + 1], m.data[i + 2]) > 90) text.push([(i / 4) % m.width, Math.floor(i / 4 / m.width)]);
    }
    const dpr = 2, top = 1 * dpr, bottom = (height - 1) * dpr;
    let edge = [Infinity], ink = [Infinity];
    for (const [ms, b64] of frames) {
      const d = await load(b64);
      for (let i = 0; i < d.data.length; i += 4) {
        const diff = Math.abs(d.data[i] - m.data[i]) + Math.abs(d.data[i + 1] - m.data[i + 1]) + Math.abs(d.data[i + 2] - m.data[i + 2]);
        if (diff <= 48) continue;
        const x = (i / 4) % d.width, y = Math.floor(i / 4 / d.width);
        const e = Math.min(y - top, bottom - 1 - y) / dpr;
        if (e < edge[0]) edge = [e, ms];
        for (const [tx, ty] of text) { const k = Math.hypot(tx - x, ty - y) / dpr; if (k < ink[0]) ink = [k, ms]; }
      }
    }
    return { edge, ink };
  }, { textMask, frames, height: r.height });
  const ok = res.edge[0] >= 2 && res.ink[0] >= 1.5;
  auditFailed ||= !ok;
  console.log(`${label}: ${ok ? "ok" : "FAIL"} — closest amber pixel to the band's inner edge ${res.edge[0].toFixed(1)}px ` +
    `(t=${(res.edge[1] / 1000).toFixed(2)}s), to the letters ${res.ink[0].toFixed(1)}px (t=${(res.ink[1] / 1000).toFixed(2)}s)`);
}
await audit.close();

const shots = {};
for (const { route } of bands) {
  await page.goto(`${pathToFileURL(join(here, "index.html")).href}#${route}`);
  await page.evaluate(() => document.fonts.ready);
  const r = await page.evaluate((id) => document.querySelector(`#${id} .band`).getBoundingClientRect().toJSON(), route);
  const clip = { x: r.x - 4, y: r.y - 4, width: 176, height: r.height + 8 };
  const frame = async (t, path) => {
    await page.evaluate((ms) => document.getAnimations().forEach((a) => { a.pause(); a.currentTime = ms; }), t * 1000);
    return page.screenshot({ clip, path });
  };
  shots[route] = [];
  for (const t of STRIP_T) shots[route].push((await frame(t)).toString("base64"));
  mkdirSync(join(work, route));
  const n = Math.round((VIDEO_T1 - VIDEO_T0) * FPS);
  for (let i = 0; i < n; i++) await frame(VIDEO_T0 + i / FPS, join(work, route, `${String(i).padStart(4, "0")}.png`));
}

const font = (file) => pathToFileURL(join(here, "fonts", file)).href;
const strip = `<!doctype html><meta charset="utf-8"><style>
@font-face{font-family:M;src:url("${font("JetBrainsMono-Regular.woff2")}")}
@font-face{font-family:M;font-weight:600;src:url("${font("JetBrainsMono-SemiBold.woff2")}")}
body{margin:0;background:#0B1220;color:#AEB7C8;font:400 12px/16px M}
.wrap{display:inline-block;padding:20px 24px 24px}
h1{margin:0 0 4px;font:600 13px/20px M;color:#E6EAF2}
p{margin:0 0 16px}
.row{display:grid;grid-template-columns:repeat(4,${176 * 2}px);gap:12px;margin-top:12px}
.lbl{margin:16px 0 0;font-size:11px;letter-spacing:.06em;text-transform:uppercase}
img{display:block;width:100%;border:1px solid #263352;border-radius:6px}
.t{margin-top:6px}.t b{font-weight:600;color:#E6EAF2}
</style><div class="wrap"><h1>polaris · header star motion</h1>
<p>${CYCLE} s cycle: ${DWELL} s dwell beside the wordmark, then one ${CYCLE - DWELL} s ease-in-out loop around it, with a 2.4 s float throughout. Frames at 2×.</p>
${bands.map(({ route, label }) => `<div class="lbl">${label}</div><div class="row">${shots[route].map((b64, i) =>
  `<div><img src="data:image/png;base64,${b64}"><div class="t"><b>t = ${STRIP_T[i].toFixed(1)} s</b> · ${phase(STRIP_T[i])}</div></div>`).join("")}</div>`).join("")}
</div>`;
const stripFile = join(work, "strip.html");
writeFileSync(stripFile, strip);
const sheet = await browser.newPage({ viewport: { width: 1600, height: 400 }, deviceScaleFactor: 2 });
await sheet.goto(pathToFileURL(stripFile).href);
await sheet.evaluate(() => document.fonts.ready);
await sheet.locator(".wrap").screenshot({ path: join(outDir, "star-motion.png") });
await browser.close();

execFileSync("ffmpeg", ["-y", "-loglevel", "error",
  "-framerate", String(FPS), "-i", join(work, bands[0].route, "%04d.png"),
  "-framerate", String(FPS), "-i", join(work, bands[1].route, "%04d.png"),
  "-filter_complex", "[0]pad=iw:ih+16:0:0:color=0x0B1220[a];[a][1]vstack=inputs=2,format=yuv420p",
  "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "24", "-row-mt", "1", join(outDir, "star-motion.webm")]);
rmSync(work, { recursive: true, force: true });
console.log(`wrote ${join(outDir, "star-motion.png")} (${readFileSync(join(outDir, "star-motion.png")).length} B) and star-motion.webm`);
process.exit(auditFailed ? 1 : 0);
