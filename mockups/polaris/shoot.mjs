// Screenshots every route at 1366×768 and checks fonts, network, overflow, 11 px minimum and 4.5:1 text contrast.
//   npm i --no-save playwright-core
//   CHROME_PATH=/usr/bin/chromium node shoot.mjs [outdir]     (outdir defaults to ./screens)
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.argv[2] ?? join(here, "screens"));
const routes = ["empty", "answer", "idk", "compare", "daylight"];
const page_url = pathToFileURL(join(here, "index.html")).href;

mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });

const external = [];
page.on("request", (r) => { if (!r.url().startsWith("file:")) external.push(r.url()); });

let failed = false;
for (const route of routes) {
  await page.goto(`${page_url}#${route}`);
  await page.evaluate(() => document.fonts.ready);

  const report = await page.evaluate((id) => {
    const screen = document.getElementById(id);
    const faces = [...document.fonts].map((f) => `${f.family} ${f.weight} ${f.status}`);
    const overflow = [...screen.querySelectorAll(".thread, .side, .col")]
      .filter((el) => el.scrollHeight > el.clientHeight + 1)
      .map((el) => `${el.className} +${el.scrollHeight - el.clientHeight}px`);
    const amber = new Set(["rgb(245, 185, 66)", "rgb(146, 64, 14)"]);
    const amberEls = [...screen.querySelectorAll("*")].filter((el) => {
      if (el.closest("symbol")) return false;
      const cs = getComputedStyle(el);
      const box = el.getBoundingClientRect();
      if (!box.width || !box.height) return false;
      const ownText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
      return (ownText && amber.has(cs.color)) || amber.has(cs.backgroundColor) ||
        (el.tagName === "svg" && amber.has(cs.fill)) ||
        (amber.has(cs.borderTopColor) && cs.borderTopStyle !== "none" && parseFloat(cs.borderTopWidth) > 0);
    }).map((el) => el.className.baseVal ?? el.className);
    const textEls = [...screen.querySelectorAll("*")].filter((el) =>
      [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()) && el.getClientRects().length);
    const tiny = textEls.filter((el) => parseFloat(getComputedStyle(el).fontSize) < 11).map((el) => el.className);

    const rgb = (s) => s.match(/[\d.]+/g).map(Number);
    const lum = ([r, g, b]) => [r, g, b].map((v) => (v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
      .reduce((acc, v, i) => acc + v * [0.2126, 0.7152, 0.0722][i], 0);
    const backdrop = (el) => {
      for (let n = el; n; n = n.parentElement) {
        const c = rgb(getComputedStyle(n).backgroundColor);
        if (c.length < 4 || c[3] > 0) return c.slice(0, 3);
      }
      return [255, 255, 255];
    };
    const contrast = textEls.map((el) => {
      const a = lum(rgb(getComputedStyle(el).color)), b = lum(backdrop(el));
      return { el: el.className || el.tagName.toLowerCase(), text: el.textContent.trim().slice(0, 24),
        ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) };
    });
    const minContrast = contrast.reduce((m, c) => (c.ratio < m.ratio ? c : m));
    const lowContrast = contrast.filter((c) => c.ratio < 4.5).map((c) => `${c.el} "${c.text}" ${c.ratio.toFixed(2)}`);
    return { faces, overflow, amberEls, tiny, minContrast, lowContrast, scrollY: document.documentElement.scrollHeight - innerHeight };
  }, route);

  const unloaded = report.faces.filter((f) => !f.endsWith("loaded"));
  const ok = unloaded.length === 0 && report.overflow.length === 0 && report.tiny.length === 0 &&
    report.lowContrast.length === 0 && report.scrollY <= 0;
  failed ||= !ok;
  const mc = report.minContrast;
  console.log(`#${route.padEnd(9)} ${ok ? "ok  " : "FAIL"} min contrast ${mc.ratio.toFixed(2)} (${mc.el} "${mc.text}") ` +
    `amber=${report.amberEls.length} [${report.amberEls.join(", ")}]`);
  if (unloaded.length) console.log(`  fonts not loaded: ${unloaded.join("; ")}`);
  if (report.overflow.length) console.log(`  overflow: ${report.overflow.join("; ")}`);
  if (report.tiny.length) console.log(`  text under 11px: ${report.tiny.join("; ")}`);
  if (report.lowContrast.length) console.log(`  contrast under 4.5:1: ${report.lowContrast.join("; ")}`);
  if (report.scrollY > 0) console.log(`  page scrolls by ${report.scrollY}px`);

  await page.screenshot({ path: join(outDir, `${route}.png`) });
}

console.log(`fonts: ${(await page.evaluate(() => [...document.fonts].map((f) => `${f.family} ${f.weight}`))).join(", ")}`);
console.log(`external requests: ${external.length ? external.join(", ") : "none"}`);
await browser.close();
process.exit(failed || external.length ? 1 : 0);
