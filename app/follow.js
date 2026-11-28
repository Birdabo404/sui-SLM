// Keeps a streaming answer in view, the way a chat does. The thread scrolls inside Streamlit's main section, not
// the window. While a sent question is being answered (the page carries .polaris-generating, which only a send
// raises), every change to the page pins that section to its bottom: right after the send and with each token, on
// one card or both compare sides. Scrolling up stops the following; scrolling back within NEAR_PX of the bottom, or
// the "Jump to latest" button, resumes it. Without a send nothing moves, so switching tabs, Settings and the system
// check keep the reader's place. Runs once in the page and finds the section again after every redraw.
(function (factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory;
  } else {
    factory(window, document);
  }
})(function (win, doc) {
  "use strict";
  const NEAR_PX = 80;
  const LIVE = ".polaris-generating";
  const SCROLLER = '[data-testid="stMain"]';
  const COMPOSER = ".st-key-composer-dock";

  let scroller = null;
  let live = false;
  let following = false;
  let lastTop = 0;
  let queued = false;

  const jump = doc.createElement("button");
  jump.type = "button";
  jump.className = "polaris-jump";
  jump.textContent = "Jump to latest";
  jump.hidden = true;
  jump.addEventListener("click", function () {
    following = true;
    pin();
  });
  doc.body.appendChild(jump);

  function distance(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight;
  }

  function pin() {
    if (scroller !== null) {
      scroller.scrollTop = scroller.scrollHeight;
      lastTop = scroller.scrollTop;
    }
    showJump();
  }

  function showJump() {
    const show = live && !following && scroller !== null && distance(scroller) > NEAR_PX;
    if (show) {
      const dock = doc.querySelector(COMPOSER);
      const box = (dock || scroller).getBoundingClientRect();
      jump.style.left = box.left + box.width / 2 + "px";
      jump.style.bottom = (dock ? win.innerHeight - box.top : 16) + 8 + "px";
    }
    jump.hidden = !show;
  }

  function onScroll() {
    const top = scroller.scrollTop;
    // Growing content never moves scrollTop and shrinking content clamps it to the bottom, so a move up that
    // leaves the bottom is the reader's.
    if (top < lastTop - 1 && distance(scroller) > 2) {
      following = false;
    } else if (live && distance(scroller) <= NEAR_PX) {
      following = true;
    }
    lastTop = top;
    showJump();
  }

  function attach() {
    const found = doc.querySelector(SCROLLER) || doc.scrollingElement;
    if (found !== scroller) {
      if (scroller !== null) {
        scroller.removeEventListener("scroll", onScroll);
      }
      scroller = found;
      if (scroller !== null) {
        scroller.addEventListener("scroll", onScroll, { passive: true });
        lastTop = scroller.scrollTop;
      }
    }
  }

  function check() {
    queued = false;
    attach();
    const was = live;
    live = doc.querySelector(LIVE) !== null;
    if (live && !was) {
      following = true;
    }
    // A run's last change can land in the same batch that drops the marker, so it still gets pinned.
    if ((live || was) && following) {
      pin();
    } else {
      showJump();
    }
  }

  function queue() {
    if (!queued) {
      queued = true;
      win.requestAnimationFrame(check);
    }
  }

  new win.MutationObserver(queue).observe(doc.body, { childList: true, subtree: true, characterData: true });
  win.addEventListener("resize", queue);
  check();
  return { check: check, state: function () { return { live: live, following: following }; } };
});
