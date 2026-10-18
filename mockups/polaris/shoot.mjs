// Screenshots every route at 1366×768 (header star frozen at rest) and checks fonts, network, overflow, 11 px minimum,
// 4.5:1 text contrast, the header star's motion contract and its occlusion by the word (see the blocks below).
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
  const wrapped = running.some((a) => /^star-wrap-path/.test(a.name));
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
    hovered = (await anims()).some((a) => /^star-wrap-path/.test(a.name));
  }
  if (await isWrapping()) await page.waitForFunction((s) => !document.querySelector(s).classList.contains("is-wrapping"), sel, { timeout: 8000 });
  const rest = await page.evaluate((s) => {
    const m = document.querySelector(s), w = m.querySelector(".word").getBoundingClientRect(), g = m.querySelector(".rig").getBoundingClientRect();
    const v = getComputedStyle(m.querySelector(".star-fx")), px = (n) => parseFloat(v.getPropertyValue(n));
    return { off: Math.hypot(g.x + g.width / 2 - (w.x + w.width / 2) - px("--rest-x"), g.y + g.height / 2 - (w.y + w.height / 2) - px("--rest-y")),
      body: getComputedStyle(m.querySelector(".body")).transform,
      trail: Math.max(...[...m.querySelectorAll(".trail>*")].map((e) => +getComputedStyle(e).opacity)),
      ink: +getComputedStyle(m.querySelector(".ink")).opacity, base: +getComputedStyle(m.querySelector(".base")).opacity };
  }, sel);
  const settled = rest.off < 0.01 && rest.body === "none" && rest.trail === 0 && rest.ink === 1 && rest.base === 0;
  const stray = running.filter((a) => !a.inBand || a.props.some((p) => !["transform", "opacity", "offsetDistance"].includes(p)));
  const expectWrap = route === "empty";
  const motionOk = stray.length === 0 && box0 !== box1 && l1 === l0 && reduced === 0 && !reducedHover &&
    wrapped === expectWrap && hovered !== false && settled;
  failed ||= !motionOk;
  const names = [...new Set(running.map((a) => `${a.name}(${a.props})`))];
  console.log(`  motion ${motionOk ? "ok  " : "FAIL"} ${running.length} running: ${names.join(" ")}`);
  console.log(`         wrap on arrival: ${wrapped} (want ${expectWrap})${hovered === null ? "" : `; hover starts a wrap: ${hovered}`}; ` +
    `settled on rest: ${settled} (off by ${rest.off.toFixed(3)}px, body ${rest.body}, trail opacity ${rest.trail}, ink/base ${rest.ink}/${rest.base}); moved while live: ${box0 !== box1}; ` +
    `layouts during 1.6s: ${l1 - l0}; under reduced motion: ${reduced} animations, hover wraps: ${reducedHover}` +
    (stray.length ? `; outside band or other properties: ${stray.map((a) => a.name).join(", ")}` : ""));

  await page.evaluate(() => document.getAnimations().forEach((a) => { a.pause(); a.currentTime = 0; }));
  await page.screenshot({ path: join(outDir, `${route}.png`) });
}

// occlusion: the word is drawn twice, .base under the star layer and .ink over it, and the wrap shows one or the
// other. Frozen at 2× in the middle of each pass (--t-behind, --t-front, --t-behind2 from star-orbit.css), each
// frame must be the layering it claims: behind, no letter pixel changes although the star overlaps the letters
// (shown by the same frame with the other copy) and is visible around them; in front, the star covers letter
// pixels. At each layer switch (--t-switch, ±20ms) both layerings must render identically, so the swap can't
// be seen; at rest the two copies must render identically too.
const occ = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
const LAYER = { behind: ".band .ink{opacity:1!important}.band .base{opacity:0!important}",
  front: ".band .ink{opacity:0!important}.band .base{opacity:1!important}" };
const NO_GLOW = ".band .halo{opacity:0!important}";
for (const route of ["empty", "answer"]) {
  const sel = `#${route} .band .mark`;
  await occ.goto(`${page_url}#${route}`);
  await occ.evaluate(() => document.fonts.ready);
  const clip = await occ.evaluate((s) => { const r = document.querySelector(s).closest(".band").getBoundingClientRect(); return { x: r.x, y: r.y, width: 216, height: r.height }; }, sel);
  const vars = await occ.evaluate((s) => { const v = getComputedStyle(document.querySelector(`${s} .star-fx`)), n = (k) => v.getPropertyValue(k).trim().split(/\s+/).map(parseFloat);
    return { behind: n("--t-behind")[0], front: n("--t-front")[0], behind2: n("--t-behind2")[0], switch: n("--t-switch") }; }, sel);
  const freeze = (t) => occ.evaluate(([s, t]) => {
    document.querySelector(s).classList.toggle("is-wrapping", t !== null);
    for (const a of document.getAnimations()) { a.pause(); a.currentTime = t !== null && /^star-(wrap|halo|trail|depth)/.test(a.animationName ?? "") ? t * 1000 : 0; }
  }, [sel, t]);
  const shot = async (css) => {
    const tag = await occ.addStyleTag({ content: css });
    const b64 = (await occ.screenshot({ clip })).toString("base64");
    await tag.evaluate((n) => n.remove());
    return b64;
  };
  const measure = (imgs) => occ.evaluate(async ({ mask, a, b }) => {
    const load = async (b64) => { const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
      const c = new OffscreenCanvas(img.width, img.height), g = c.getContext("2d"); g.drawImage(img, 0, 0); return g.getImageData(0, 0, img.width, img.height).data; };
    const [m, A, B] = await Promise.all([mask, a, b].map(load));
    const diff = (P, Q, i) => Math.abs(P[i] - Q[i]) + Math.abs(P[i + 1] - Q[i + 1]) + Math.abs(P[i + 2] - Q[i + 2]);
    const r = { differ: 0, leak: 0, hitA: 0, hitB: 0, around: 0 };
    for (let i = 0; i < m.length; i += 4) {
      const core = (m[i + 2] - 44) / (242 - 44) >= 0.97;  // letter pixels at ≥97% coverage, band #111A2C → ink #E6EAF2
      if (diff(A, B, i) > 8) r.differ++;
      if (core && diff(A, m, i) > 24) r.leak++;
      if (core && diff(A, m, i) > 48) r.hitA++;
      if (core && diff(B, m, i) > 48) r.hitB++;
      if (!core && diff(A, m, i) > 48) r.around++;
    }
    return r;
  }, imgs);

  await freeze(null);
  const mask = await shot(".band .star-fx *{opacity:0!important}");
  const swap = await measure({ mask, a: await shot(LAYER.behind), b: await shot(LAYER.front) });
  const passes = [];
  for (const [name, t, want] of [["behind", vars.behind, "behind"], ["front", vars.front, "front"], ["behind again", vars.behind2, "behind"]]) {
    await freeze(t);
    const actual = await shot(NO_GLOW), forced = await shot(NO_GLOW + LAYER[want]);
    const other = await shot(NO_GLOW + LAYER[want === "behind" ? "front" : "behind"]);
    const same = (await measure({ mask, a: actual, b: forced })).differ;
    const m = await measure({ mask, a: actual, b: other });
    const ok = same === 0 && (want === "behind" ? m.leak === 0 && m.hitB >= 20 && m.around >= 20 : m.hitA >= 20);
    passes.push({ name, t, ok, text: want === "behind"
      ? `${name} t=${t}s: ${m.leak} letter px changed, ${m.hitB} letter px the star passes under, ${m.around} star px visible`
      : `${name} t=${t}s: ${m.hitA} letter px covered by the star` + (same ? `; ${same} px off the forced layering` : "") });
  }
  const switches = [];
  for (const s of vars.switch) for (const o of [-0.02, 0, 0.02]) {
    await freeze(s + o);
    switches.push((await measure({ mask, a: await shot(LAYER.behind), b: await shot(LAYER.front) })).differ);
  }
  const ok = swap.differ === 0 && passes.every((p) => p.ok) && switches.every((v) => v === 0);
  failed ||= !ok;
  console.log(`#${route.padEnd(9)} occlusion ${ok ? "ok  " : "FAIL"} copies identical at rest: ${swap.differ === 0} (${swap.differ} px differ); ` +
    `layer switches at ${vars.switch.join("s, ")}s ±20ms: ${switches.join("/")} px differ between layerings`);
  for (const p of passes) console.log(`         ${p.ok ? "" : "FAIL "}${p.text}`);
}
await occ.close();

console.log(`fonts: ${(await page.evaluate(() => [...document.fonts].map((f) => `${f.family} ${f.weight}`))).join(", ")}`);
console.log(`external requests: ${external.length ? external.join(", ") : "none"}`);
await browser.close();
process.exit(failed || external.length ? 1 : 0);
