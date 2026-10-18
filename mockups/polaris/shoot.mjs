// Screenshots every route at 1366×768 (header star frozen at rest) and checks fonts, network, overflow, 11 px minimum,
// 4.5:1 text contrast, and the header star's motion contract (see the motion block below).
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

  // motion: only the header star animates (transform, opacity, offset-distance) and never triggers layout; the wrap
  // runs once on load in the empty state and never by itself during answers; hover starts one; it settles
  // exactly on the rest pose; nothing runs, and hover does nothing, under reduced motion
  const sel = `#${route} .band .mark`;
  const anims = () => page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running").map((a) => ({
    name: a.animationName ?? `transition:${a.transitionProperty}`, inBand: !!a.effect.target.closest(".band"),
    props: [...new Set(a.effect.getKeyframes().flatMap((k) => Object.keys(k)))]
      .filter((p) => !["offset", "computedOffset", "easing", "composite"].includes(p)),
  })));
  const starBox = () => page.evaluate((s) => JSON.stringify(document.querySelector(`${s} .sway svg`).getBoundingClientRect()), sel);
  const isWrapping = () => page.evaluate((s) => document.querySelector(s).classList.contains("is-wrapping"), sel);
  const layoutCount = async () => (await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === "LayoutCount").value;
  const running = await anims();
  const wrapped = running.some((a) => a.name === "star-wrap-path");
  const [box0, l0] = [await starBox(), await layoutCount()];
  await page.waitForTimeout(1600);
  const [box1, l1] = [await starBox(), await layoutCount()];
  await page.emulateMedia({ reducedMotion: "reduce" });
  const reduced = (await anims()).length;
  if (!wrapped) { await page.hover(sel); await page.mouse.move(1360, 760); }
  const reducedHover = !wrapped && (await isWrapping());
  await page.emulateMedia({ reducedMotion: "no-preference" });
  let hovered = null;
  if (route === "answer") {
    await page.hover(sel); await page.mouse.move(1360, 760);
    hovered = (await anims()).some((a) => a.name === "star-wrap-path");
  }
  if (await isWrapping()) await page.waitForFunction((s) => !document.querySelector(s).classList.contains("is-wrapping"), sel, { timeout: 8000 });
  const rest = await page.evaluate((s) => {
    const m = document.querySelector(s), w = m.querySelector(".word").getBoundingClientRect(), g = m.querySelector(".rig").getBoundingClientRect();
    const v = getComputedStyle(m.querySelector(".star-fx")), px = (n) => parseFloat(v.getPropertyValue(n));
    return { off: Math.hypot(g.x + g.width / 2 - (w.x + w.width / 2) - px("--rest-x"), g.y + g.height / 2 - (w.y + w.height / 2) - px("--rest-y")),
      body: getComputedStyle(m.querySelector(".body")).transform,
      trail: Math.max(...[...m.querySelectorAll(".trail>*")].map((e) => +getComputedStyle(e).opacity)) };
  }, sel);
  const settled = rest.off < 0.01 && rest.body === "none" && rest.trail === 0;
  const stray = running.filter((a) => !a.inBand || a.props.some((p) => !["transform", "opacity", "offsetDistance"].includes(p)));
  const expectWrap = route === "empty";
  const motionOk = stray.length === 0 && box0 !== box1 && l1 === l0 && reduced === 0 && !reducedHover &&
    wrapped === expectWrap && hovered !== false && settled;
  failed ||= !motionOk;
  const names = [...new Set(running.map((a) => `${a.name}(${a.props})`))];
  console.log(`  motion ${motionOk ? "ok  " : "FAIL"} ${running.length} running: ${names.join(" ")}`);
  console.log(`         wrap on arrival: ${wrapped} (want ${expectWrap})${hovered === null ? "" : `; hover starts a wrap: ${hovered}`}; ` +
    `settled on rest: ${settled} (off by ${rest.off.toFixed(3)}px, body ${rest.body}, trail opacity ${rest.trail}); moved while live: ${box0 !== box1}; ` +
    `layouts during 1.6s: ${l1 - l0}; under reduced motion: ${reduced} animations, hover wraps: ${reducedHover}` +
    (stray.length ? `; outside band or other properties: ${stray.map((a) => a.name).join(", ")}` : ""));

  await page.evaluate(() => document.getAnimations().forEach((a) => { a.pause(); a.currentTime = 0; }));
  await page.screenshot({ path: join(outDir, `${route}.png`) });
}

console.log(`fonts: ${(await page.evaluate(() => [...document.fonts].map((f) => `${f.family} ${f.weight}`))).join(", ")}`);
console.log(`external requests: ${external.length ? external.join(", ") : "none"}`);
await browser.close();
process.exit(failed || external.length ? 1 : 0);
