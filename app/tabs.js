// The chat tab strip scrolls like a browser's once its tabs reach their narrowest. This adds the two chevrons over
// the strip's ends (shown only while tabs are out of sight that way), lets a vertical wheel scroll it sideways, and
// scrolls the open tab into view when it changes, when a tab is added and when the strip is resized; otherwise the
// reader's own scrolling is kept. It also marks the strip data-more-left / data-more-right for the edge fade.
// Runs once in the page and finds the strip again after every redraw.
(function (factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory;
  } else {
    factory(window, document);
  }
})(function (win, doc) {
  "use strict";
  const STRIP = ".st-key-tabs";
  const ACTIVE = '[class*="st-key-tab-on-"]';
  const TAB = ':scope > [data-testid="stVerticalBlockBorderWrapper"]';
  const EDGE_PX = 32;

  let strip = null;
  let seen = { active: null, count: -1, width: -1 };
  let queued = false;
  const wired = new WeakSet();

  function chevron(dir) {
    const b = doc.createElement("button");
    b.type = "button";
    b.className = "polaris-tab-scroll";
    b.setAttribute("data-dir", dir);
    b.setAttribute("aria-label", dir === "left" ? "Scroll tabs left" : "Scroll tabs right");
    b.tabIndex = -1;
    b.hidden = true;
    b.addEventListener("click", function () {
      if (strip !== null) {
        const step = Math.max(strip.clientWidth * 0.7, 120);
        strip.scrollBy({ left: dir === "left" ? -step : step, behavior: "smooth" });
      }
    });
    doc.body.appendChild(b);
    return b;
  }
  const left = chevron("left");
  const right = chevron("right");

  function mark(name, on) {
    if (on) {
      strip.setAttribute(name, "");
    } else {
      strip.removeAttribute(name);
    }
  }

  function edges() {
    if (strip === null) {
      left.hidden = right.hidden = true;
      return;
    }
    const max = strip.scrollWidth - strip.clientWidth;
    const more = { left: max > 1 && strip.scrollLeft > 1, right: max > 1 && strip.scrollLeft < max - 1 };
    mark("data-more-left", more.left);
    mark("data-more-right", more.right);
    const box = strip.getBoundingClientRect();
    const top = box.top + Math.max(box.height - 38, 0) + 19 - 12 + "px";
    left.style.top = right.style.top = top;
    left.style.left = box.left + "px";
    right.style.left = box.right - 24 + "px";
    left.hidden = !more.left;
    right.hidden = !more.right;
  }

  function reveal(tab) {
    const box = strip.getBoundingClientRect();
    const r = tab.getBoundingClientRect();
    const pad = strip.scrollWidth > strip.clientWidth + 1 ? EDGE_PX : 0;
    if (r.left < box.left + pad) {
      strip.scrollLeft -= box.left + pad - r.left;
    } else if (r.right > box.right - pad) {
      strip.scrollLeft += r.right - (box.right - pad);
    }
  }

  function onWheel(event) {
    if (strip.scrollWidth <= strip.clientWidth + 1 || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
      return;
    }
    strip.scrollLeft += event.deltaY;
    event.preventDefault();
  }

  function check() {
    queued = false;
    strip = doc.querySelector(STRIP);
    if (strip === null) {
      edges();
      return;
    }
    if (!wired.has(strip)) {
      wired.add(strip);
      strip.addEventListener("scroll", edges, { passive: true });
      strip.addEventListener("wheel", onWheel, { passive: false });
      seen = { active: null, count: -1, width: -1 };
    }
    const active = strip.querySelector(ACTIVE);
    const key = active && Array.from(active.classList).find(function (c) { return c.startsWith("st-key-tab-on-"); });
    const now = { active: key, count: strip.querySelectorAll(TAB).length, width: strip.clientWidth };
    if (active !== null && (now.active !== seen.active || now.count > seen.count || now.width !== seen.width)) {
      reveal(active.parentElement);
    }
    seen = now;
    edges();
  }

  function queue() {
    if (!queued) {
      queued = true;
      win.requestAnimationFrame(check);
    }
  }

  new win.MutationObserver(queue).observe(doc.body, { childList: true, subtree: true });
  win.addEventListener("resize", queue);
  check();
  return { check: check };
});
