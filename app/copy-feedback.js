// Streamlit's code copy button copies without any sign that it did. This marks the clicked button
// data-copied for COPIED_MS so style.css can swap its icon for a check and "Copied", then puts it back.
// It runs in the page (not the loader iframe), once, and listens on the document, so every code box gets it:
// answers, compare sides, replayed chats and boxes Streamlit redraws later.
(function () {
  "use strict";
  const COPIED_MS = 1500;
  const BUTTON = '[data-testid="stCodeCopyButton"]';
  const live = document.createElement("div");
  live.className = "polaris-copy-live";
  live.setAttribute("aria-live", "polite");
  document.body.appendChild(live);

  document.addEventListener("click", function (event) {
    const button = event.target instanceof Element ? event.target.closest(BUTTON) : null;
    if (button === null) {
      return;
    }
    clearTimeout(button.polarisCopied);
    button.setAttribute("data-copied", "");
    button.title = "Copied";
    live.textContent = "Copied";
    button.polarisCopied = setTimeout(function () {
      button.removeAttribute("data-copied");
      button.title = "Copy to clipboard";
      live.textContent = "";
    }, COPIED_MS);
  }, true);
})();
