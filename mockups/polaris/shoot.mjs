// Screenshots every route at 1366×768 (header star frozen at rest) and checks fonts, network, overflow, 11 px minimum,
// 4.5:1 text contrast, and that the only animation is the header star: transform-only, layout-free, off under reduced motion.
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
const cdp = await page.context().newCDPSession(page);
await cdp.send("Performance.enable");

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

  // motion: only the header star animates, only transform/opacity, live without layout, off under reduced motion
  const anims = () => page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running").map((a) => ({
    name: a.animationName, inBand: !!a.effect.target.closest(".band"),
    props: [...new Set(a.effect.getKeyframes().flatMap((k) => Object.keys(k)))]
      .filter((p) => !["offset", "computedOffset", "easing", "composite"].includes(p)),
  })));
  const starTf = () => page.evaluate((id) => getComputedStyle(document.querySelector(`#${id} .band .orbit`)).transform, route);
  const layoutCount = async () => (await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === "LayoutCount").value;
  const running = await anims();
  const [tf0, l0] = [await starTf(), await layoutCount()];
  await page.waitForTimeout(1600);
  const [tf1, l1] = [await starTf(), await layoutCount()];
  await page.emulateMedia({ reducedMotion: "reduce" });
  const reduced = (await anims()).length;
  await page.emulateMedia({ reducedMotion: "no-preference" });
  const stray = running.filter((a) => !a.inBand || a.props.some((p) => !["transform", "opacity"].includes(p)));
  const motionOk = running.length === 2 && stray.length === 0 && tf0 !== tf1 && l1 === l0 && reduced === 0;
  failed ||= !motionOk;
  console.log(`  motion ${motionOk ? "ok  " : "FAIL"} ${running.map((a) => `${a.name}(${a.props})`).join(" + ")}; ` +
    `moved while live: ${tf0 !== tf1}; layouts during 1.6s: ${l1 - l0}; animations under reduced motion: ${reduced}` +
    (stray.length ? `; outside band or non-transform: ${stray.map((a) => a.name).join(", ")}` : ""));

  await page.evaluate(() => document.getAnimations().forEach((a) => { a.pause(); a.currentTime = 0; }));
  await page.screenshot({ path: join(outDir, `${route}.png`) });
}

console.log(`fonts: ${(await page.evaluate(() => [...document.fonts].map((f) => `${f.family} ${f.weight}`))).join(", ")}`);
console.log(`external requests: ${external.length ? external.join(", ") : "none"}`);
await browser.close();
process.exit(failed || external.length ? 1 : 0);
