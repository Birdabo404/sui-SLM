// Screenshots every route at 1366×768 and checks the mock against the spec's hard rules.
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
    const tiny = [...screen.querySelectorAll("*")].filter((el) => {
      const ownText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
      const fs = parseFloat(getComputedStyle(el).fontSize);
      return ownText && fs < 16 && !el.classList.contains("footer-license");
    }).map((el) => el.className);
    return { faces, overflow, amberEls, tiny, scrollY: document.documentElement.scrollHeight - innerHeight };
  }, route);

  const unloaded = report.faces.filter((f) => !f.endsWith("loaded"));
  const ok = unloaded.length === 0 && report.overflow.length === 0 && report.tiny.length === 0 && report.scrollY <= 0;
  failed ||= !ok;
  console.log(`#${route.padEnd(9)} ${ok ? "ok  " : "FAIL"} amber=${report.amberEls.length} [${report.amberEls.join(", ")}]`);
  if (unloaded.length) console.log(`  fonts not loaded: ${unloaded.join("; ")}`);
  if (report.overflow.length) console.log(`  overflow: ${report.overflow.join("; ")}`);
  if (report.tiny.length) console.log(`  text under 16px: ${report.tiny.join("; ")}`);
  if (report.scrollY > 0) console.log(`  page scrolls by ${report.scrollY}px`);

  await page.screenshot({ path: join(outDir, `${route}.png`) });
}

console.log(`fonts: ${(await page.evaluate(() => [...document.fonts].map((f) => `${f.family} ${f.weight}`))).join(", ")}`);
console.log(`external requests: ${external.length ? external.join(", ") : "none"}`);
await browser.close();
process.exit(failed || external.length ? 1 : 0);
