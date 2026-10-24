"""Polaris demo helpers: feature flags, run records, and HTML fragments.

Render functions that need Streamlit live in app.py. Generation is always
app.engine.answer_question.
"""
from __future__ import annotations

import csv
import http.client
import io
import json
import os
import random
import re
import socket
import threading
from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

import psutil
import requests

from app.engine import MODE_LABELS, MODE_SUBS, OLLAMA_HOST, mode_choices, ollama_chat_body, uses_rag
from shared.frozen import format_threshold

STAR_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M12 0L14.2 9.8 24 12 14.2 14.2 12 24 9.8 14.2 0 12 9.8 9.8Z"/>'
    "</svg>"
)
GLINT_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path fill="#FFEBC2" d="M12 0C12.4 9 15 11.6 24 12C15 12.4 12.4 15 12 24C11.6 15 9 12.4 0 12C9 11.6 11.6 9 12 0Z"/>'
    '<circle cx="12" cy="12" r="2.2" fill="#FFF8EA"/>'
    "</svg>"
)


def star_lux_svg(uid):
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true">'
        f'<path fill="url(#polaris-star-lux-{uid})" d="M12 0C12.45 7.1 14.7 11.2 22 12C14.7 12.8 12.45 16.9 12 24C11.55 16.9 9.3 12.8 2 12C9.3 11.2 11.55 7.1 12 0Z"/>'
        '<path fill="#FFF9EC" opacity=".55" d="M12 6C12.2 9.6 12.9 11.3 15.5 12C12.9 12.7 12.2 14.4 12 18C11.8 14.4 11.1 12.7 8.5 12C11.1 11.3 11.8 9.6 12 6Z"/>'
        "</svg>"
    )


def star_defs(uid):
    """Each wordmark carries its own gradient id: a shared id breaks once the first copy is display:none."""
    return (
        '<svg width="0" height="0" aria-hidden="true">'
        "<defs>"
        f'<radialGradient id="polaris-star-lux-{uid}" cx="12" cy="12" r="11.5" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#FFF7E2"/>'
        '<stop offset=".16" stop-color="#FFE3A2"/>'
        '<stop offset=".46" stop-color="#F5B942"/>'
        '<stop offset="1" stop-color="#E29A2E"/>'
        "</radialGradient>"
        "</defs>"
        "</svg>"
    )


SNIPPET_LEN = 200
CHIP_CAP = 4
EXPORT_FIELDS = (
    "ts",
    "question",
    "config",
    "mode",
    "model",
    "answer",
    "latency_s",
    "app_rss_mb",
    "ollama_rss_mb",
    "total_mb",
    "retrieval_max_dense",
    "docs",
    "abstained",
    "index_path",
    "threshold",
    "k",
    "git_commit",
    "hits_json",
)
EXAMPLES = (
    "extract a .tar.gz file",
    "make a file readable only by me",
    "connect to the lab printer",
    "what's the lab wi-fi password?",
)
MODE_NAMES = {
    "base": "Base",
    "cot": "CoT",
    "rag": "RAG",
    "full": "Full",
}
MODE_HINTS = {
    "base": "Model only",
    "cot": "Reasons step by step",
    "rag": "Answers from the lab docs",
    "full": "Lab docs, reasoned step by step",
}
TAGLINE = "Offline help for the Linux lab"
STILL_WORKING = "The model runs on this machine, so this can take a moment."
# Shown one after another while an answer is running, after the stage label (always 7: the CSS cycle is built for 8).
THINKING_WORDS = (
    "Grepping…",
    "Reading man pages…",
    "Untarring thoughts…",
    "Piping it through…",
    "Charting a course…",
    "Stargazing…",
    "Checking permissions…",
)
_WIFI_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 8.5a15 15 0 0 1 20 0M5 12a10.5 10.5 0 0 1 14 0'
    'M8.5 15.5a5.5 5.5 0 0 1 7 0"/><circle class="d" cx="12" cy="19" r="1.2"/></svg>'
)
_WIFI_OFF_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 15.5a5.5 5.5 0 0 1 7 0M5 12a10.5 10.5 0 0 1 4.2-2.3'
    'M2 8.5a15 15 0 0 1 5-3M3 3l18 18"/><circle class="d" cx="12" cy="19" r="1.2"/></svg>'
)
_ALERT_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5"/>'
    '<circle class="d" cx="12" cy="16.5" r="1"/></svg>'
)
SOURCE_KEY_RULES = (
    ("tldr", "TLDR"),
    ("man", "MAN"),
    ("gnome", "GNOME"),
    ("ubuntu", "UBUNTU-HELP"),
    ("help", "UBUNTU-HELP"),
)


class ChatCancelled(Exception):
    """The user stopped the answer before Ollama replied."""


class CancellableChat:
    """chat_fn for answer_question that the UI can stop.

    Sends the same body as engine.ollama_chat. cancel() shuts the socket down, which unblocks the waiting
    thread and makes Ollama abandon the generation instead of finishing it in the background.
    """

    def __init__(self, host=None, timeout=300, temperature=0.2, seed=None, keep_alive=None):
        self._url = urlsplit((host or OLLAMA_HOST).rstrip("/") + "/api/chat")
        self._timeout = timeout
        self._temperature = temperature
        self._seed = seed
        self._keep_alive = keep_alive
        self._lock = threading.Lock()
        self._conn = None
        self._committed = False
        self.cancelled = False

    def configure(self, *, temperature, seed, keep_alive):
        with self._lock:
            self._temperature = temperature
            self._seed = seed
            self._keep_alive = keep_alive

    def __call__(self, messages, model):
        conn = http.client.HTTPConnection(self._url.hostname, self._url.port or 80, timeout=self._timeout)
        conn.auto_open = 0
        body = json.dumps(
            ollama_chat_body(
                messages,
                model,
                temperature=self._temperature,
                seed=self._seed,
                keep_alive=self._keep_alive,
            )
        ).encode("utf-8")
        try:
            conn.connect()
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            if self.cancelled:
                raise ChatCancelled() from e
            raise requests.ConnectionError(str(e)) from e
        with self._lock:
            if self.cancelled:
                conn.close()
                raise ChatCancelled()
            self._conn = conn
        try:
            conn.request("POST", self._url.path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
        except (OSError, http.client.HTTPException) as e:
            if self.cancelled:
                raise ChatCancelled() from e
            raise requests.ConnectionError(str(e)) from e
        finally:
            with self._lock:
                if self._conn is conn:
                    self._conn = None
            conn.close()
        if self.cancelled:
            raise ChatCancelled()
        if resp.status >= 400:
            raise requests.HTTPError(f"Ollama returned HTTP {resp.status}")
        return json.loads(data)["message"]["content"]

    @contextmanager
    def commit(self):
        with self._lock:
            if self.cancelled:
                raise ChatCancelled()
            if self._committed:
                raise RuntimeError("chat result already committed")
            self._committed = True
        yield

    def cancel(self):
        with self._lock:
            if self._committed:
                return
            self.cancelled = True
            conn = self._conn
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if conn is not None:
            conn.close()


SKY_COLS, SKY_ROWS = 220, 56
SKY_LEVELS = (1.0, 0.55, 0.3, 0.12, 0.0)
# The Little Dipper, Polaris at the tip of the handle: (col, row) offsets, two columns per unit for the
# monospace cell's aspect ratio, and the glyph for each star by rough brightness.
LITTLE_DIPPER = (
    (0, 0, "*"), (5, 2, "."), (9, 4, "'"), (13, 6, "."), (15, 9, "."), (22, 9, "+"), (20, 6, "*"),
)


def ascii_sky(cols=SKY_COLS, rows=SKY_ROWS, seed=7):
    """A sparse ASCII starfield, the same every time, with the Little Dipper up and to the right."""
    rng = random.Random(seed)
    grid = [[" "] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.011:
                grid[r][c] = rng.choices(".'+*", weights=(70, 14, 9, 7))[0]
    c0, r0 = int(cols * 0.66), int(rows * 0.14)
    bright = set()
    for dc, dr, glyph in LITTLE_DIPPER:
        grid[r0 + dr][c0 + dc] = glyph
        bright.add((r0 + dr, c0 + dc))
    lines = []
    for r, row in enumerate(grid):
        end = len("".join(row).rstrip())
        cells = (
            f"<b>{escape(ch, quote=False)}</b>" if (r, c) in bright else escape(ch, quote=False)
            for c, ch in enumerate(row[:end])
        )
        lines.append("".join(cells))
    return "<br>".join(lines)


# One line of HTML on purpose: Streamlit replaces <pre>, and a blank line would end the HTML block.
_SKY_HTML = f'<div class="sky" aria-hidden="true"><div class="sky-field">{ascii_sky()}</div></div>'


def sky_html():
    return _SKY_HTML


def sky_level(turns):
    """0 on an empty chat, one step fainter per question asked, gone after the fourth."""
    return max(0, min(int(turns), len(SKY_LEVELS) - 1))


def polaris_on():
    raw = os.environ.get("POLARIS_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


class HitScore(TypedDict):
    rank: int
    source: str
    title: str
    path: str
    dense: float | None
    rrf: float | None
    bm25: float | None
    snippet: str


class RunRecord(TypedDict):
    ts: str
    question: str
    config: str
    mode: str
    model: str
    answer: str
    latency_s: float
    app_rss_mb: float
    ollama_rss_mb: float
    total_mb: float
    retrieval_max_dense: float | None
    hits: list[HitScore]
    docs: list[str]
    abstained: bool
    index_path: str
    threshold: float
    k: int
    git_commit: str


def _opt_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_dense(hits):
    sims = [_opt_float(h.get("dense")) for h in hits]
    sims = [s for s in sims if s is not None]
    if not sims:
        return None
    return max(sims)


def snippet_of(text):
    compact = " ".join((text or "").split())
    return compact[:SNIPPET_LEN]


def hit_scores(hits):
    rows: list[HitScore] = []
    for i, hit in enumerate(hits or [], 1):
        rows.append(
            {
                "rank": i,
                "source": str(hit.get("source") or ""),
                "title": str(hit.get("title") or ""),
                "path": str(hit.get("path") or ""),
                "dense": _opt_float(hit.get("dense")),
                "rrf": _opt_float(hit.get("rrf")),
                "bm25": _opt_float(hit.get("bm25")),
                "snippet": snippet_of(hit.get("text") or ""),
            }
        )
    return rows


def build_run_record(result, index_path=""):
    hits = list(result.get("hits") or [])
    scored = hit_scores(hits)
    rag = uses_rag(result.get("mode") or "")
    abstained = bool(result.get("weak")) if rag else False
    docs = [label for h in scored if (label := doc_label(h))]
    return {
        "ts": str(result.get("ts") or ""),
        "question": str(result.get("question") or ""),
        "config": str(result.get("config") or ""),
        "mode": str(result.get("mode") or ""),
        "model": str(result.get("model") or ""),
        "answer": str(result.get("text") or ""),
        "latency_s": float(result.get("latency_s") or 0.0),
        "app_rss_mb": float(result.get("app_rss_mb") or 0.0),
        "ollama_rss_mb": float(result.get("ollama_rss_mb") or 0.0),
        "total_mb": float(result.get("total_mb") or 0.0),
        "retrieval_max_dense": max_dense(hits) if rag else None,
        "hits": scored,
        "docs": docs,
        "abstained": abstained,
        "index_path": str(index_path or ""),
        "threshold": float(result.get("threshold") or 0.0),
        "k": int(result.get("k") or 0),
        "git_commit": str(result.get("git_commit") or ""),
    }


def records_to_csv(records):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(EXPORT_FIELDS), extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        row = {k: rec.get(k, "") for k in EXPORT_FIELDS}
        row["latency_s"] = f"{float(rec.get('latency_s') or 0):.4f}"
        row["app_rss_mb"] = f"{float(rec.get('app_rss_mb') or 0):.1f}"
        row["ollama_rss_mb"] = f"{float(rec.get('ollama_rss_mb') or 0):.1f}"
        row["total_mb"] = f"{float(rec.get('total_mb') or 0):.1f}"
        dense = rec.get("retrieval_max_dense")
        row["retrieval_max_dense"] = "" if dense is None else f"{float(dense):.4f}"
        row["docs"] = "; ".join(rec.get("docs") or [])
        row["hits_json"] = json.dumps(rec.get("hits") or [], ensure_ascii=False)
        writer.writerow(row)
    return buf.getvalue()


def records_to_json(records):
    return json.dumps(list(records), indent=2, ensure_ascii=False)


def default_compare_modes():
    modes = mode_choices()
    if "base" in modes and "full" in modes:
        return "base", "full"
    if len(modes) < 2:
        raise ValueError("need two modes to compare")
    return modes[0], modes[-1]


def pair_compare_modes(left, right):
    modes = mode_choices()
    if not modes:
        raise ValueError("no modes")
    if left not in modes:
        left = modes[0]
    if right not in modes or right == left:
        right = next(m for m in modes if m != left)
    return left, right


def doc_label(hit):
    title = str(hit.get("title") or "").strip()
    if title:
        return title
    path = str(hit.get("path") or "").strip()
    if path:
        return path
    source = str(hit.get("source") or "").strip()
    if source:
        return source
    rank = hit.get("rank")
    if rank is not None and rank != "":
        return f"hit {rank}"
    return ""


def source_key(source):
    raw = (source or "").strip()
    low = raw.lower()
    for needle, key in SOURCE_KEY_RULES:
        if needle in low:
            return key
    if not raw:
        return "DOC"
    return raw.upper()[:12]


def chip_name(hit):
    title = (hit.get("title") or "").strip()
    if title:
        return title
    path = (hit.get("path") or "").strip()
    return Path(path).name if path else (hit.get("source") or "")


def _best_hit(hits):
    if not hits:
        return None
    ranked = sorted(
        hits,
        key=lambda h: (_opt_float(h.get("dense")) is not None, _opt_float(h.get("dense")) or 0.0),
        reverse=True,
    )
    return ranked[0]


def short_doc_name(hit):
    title = (hit.get("title") or "").strip()
    if title:
        word = title.split()[0]
        return word.split("(")[0] or word
    path = (hit.get("path") or "").strip()
    if path:
        return Path(path).stem
    return ""


def best_match_label(hits):
    top = _best_hit(hits)
    if top is None:
        return ""
    key = source_key(top.get("source"))
    name = chip_name(top)
    if name:
        return f"{key.lower()} {name}"
    return key.lower()


def evidence_source_phrase(hits):
    top = _best_hit(hits)
    if top is None:
        return ""
    key = source_key(top.get("source")).lower()
    name = short_doc_name(top)
    return f"{key} {name}".strip()


def source_count(record):
    if not uses_rag(record.get("mode") or ""):
        return 0
    hits = record.get("hits") or []
    if hits:
        return len(hits)
    return len(record.get("docs") or [])


def source_count_label(n):
    return f"{n} source" if n == 1 else f"{n} sources"


def format_total_ram(mb):
    return f"{float(mb or 0.0) / 1024.0:.1f} GB"


def split_reasoning(text):
    body = text or ""
    for sep in ("\n\nAnswer:", "\nAnswer:", "\n\n**Answer:**"):
        if sep in body:
            head, tail = body.split(sep, 1)
            return head.strip(), tail.strip()
    return "", body


def first_line(text, limit=88):
    line = " ".join((text or "").split())
    if len(line) <= limit:
        return line
    return line[: limit - 1] + "…"


def index_chunk_count(path):
    p = Path(path) / "chunks.jsonl"
    if not p.is_file():
        return 0
    n = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def yn(ok):
    return "yes" if ok else "no"


def mode_label(mode):
    return MODE_LABELS.get(mode, (mode or "").upper())


def mode_sub(mode):
    return MODE_SUBS.get(mode, "")


def mode_name(mode):
    return MODE_NAMES.get(mode, mode_label(mode))


def mode_hint(mode):
    return MODE_HINTS.get(mode, "")


def mode_from_name(name, fallback="full"):
    return next((m for m, n in MODE_NAMES.items() if n == name), fallback)


def _fmt_score(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def empty_slot_html(kind="", label=""):
    klass = f"slot-empty {kind}".strip()
    inner = escape(label) if label else "&nbsp;"
    return f'<div class="{klass}">{inner}</div>'


def _header_star_html(uid):
    glints = f'<span class="glints"><b>{GLINT_SVG}</b><b>{GLINT_SVG}</b></span>'
    trail = "<span class=\"trail\">" + ("<i></i>" * 10) + f"<b>{GLINT_SVG}</b>" * 3 + "</span>"
    glow = (
        '<span class="star-fx star-glow" aria-hidden="true">'
        '<span class="rig"><span class="body"><span class="halo"><span class="glow"></span></span></span></span>'
        "</span>"
    )
    star = (
        f'<span class="star-fx" aria-hidden="true">{glints}{trail}'
        f'<span class="rig"><span class="body"><span class="breathe"><span class="sway">{star_lux_svg(uid)}</span></span></span></span>'
        "</span>"
    )
    return (
        glow
        + '<span class="base" aria-hidden="true">polaris</span>'
        + star
        + '<span class="ink">polaris</span>'
    )


def network_online(route_dir="/proc/net"):
    """True when this machine has a usable network route. Passive: reads routing state, sends nothing."""
    base = Path(route_dir)
    v4, v6 = base / "route", base / "ipv6_route"
    if v4.is_file() or v6.is_file():
        try:
            if v4.is_file():
                for line in v4.read_text().splitlines()[1:]:
                    f = line.split()
                    if len(f) > 3 and f[1] == "00000000" and int(f[3], 16) & 1:
                        return True
            if v6.is_file():
                for line in v6.read_text().splitlines():
                    f = line.split()
                    if len(f) > 9 and f[0] == "0" * 32 and f[1] == "00" and f[9] != "lo":
                        return True
            return False
        except (OSError, ValueError):
            pass
    try:
        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            if not stats.get(name) or not stats[name].isup or name.lower().startswith(("lo", "docker", "veth", "virbr")):
                continue
            for a in addrs:
                ip = str(a.address)
                if ip.count(".") == 3 and not ip.startswith(("127.", "169.254.")):
                    return True
    except Exception:
        pass
    return False


def connection_html(online, ollama_up):
    if not ollama_up:
        return (
            '<span class="conn down" title="Start it with: ollama serve">'
            f"{_ALERT_SVG}<span>Ollama stopped</span></span>"
        )
    if online:
        return (
            '<span class="conn online" title="A network is connected. Answers still come from the model on this machine.">'
            f"{_WIFI_SVG}<span>Online</span></span>"
        )
    return (
        '<span class="conn offline" title="No network. Everything runs on this machine.">'
        f"{_WIFI_OFF_SVG}<span>Offline</span></span>"
    )


def header_band_html(slim=False, animate=False, status="", uid=None):
    uid = uid or ("slim" if slim else "hero")
    classes = ["band"]
    if slim:
        classes.append("slim")
        classes.append("rest")
    if animate:
        classes.append("wrap-on")
    mark_class = "mark is-wrapping" if animate else "mark rest"
    label = "polaris, replay the star" if animate else "polaris"
    return (
        f'<div class="{" ".join(classes)}">'
        + star_defs(uid)
        + f'<span class="{mark_class}" tabindex="0" role="img" aria-label="{label}"><span class="word">'
        + _header_star_html(uid)
        + "</span></span>"
        + (f'<span class="spacer"></span>{status}' if status else "")
        + "</div>"
    )


def side_logo_html(play=False):
    """Sidebar wordmark: no idle loops; wraps once when a chat starts and again on hover or focus."""
    mark_class = "mark is-wrapping" if play else "mark"
    return (
        '<div class="side-logo"><div class="band side calm wrap-on">'
        + star_defs("side")
        + f'<span class="{mark_class}" tabindex="0" role="img" aria-label="polaris, replay the star"><span class="word">'
        + _header_star_html("side")
        + "</span></span></div></div>"
    )


def empty_hero_html():
    return (
        '<div class="hero">'
        + header_band_html(animate=True)
        + f'<p class="tagline">{escape(TAGLINE)}</p>'
        "</div>"
    )


def card_head_html(mode):
    return (
        '<div class="card-head">'
        f'<span class="mode">{escape(mode_name(mode))}</span>'
        f'<span class="sub">{escape(mode_hint(mode))}</span>'
        "</div>"
    )


def retrieval_bar_html(score, threshold, retrieval_off=False, mode=""):
    if retrieval_off:
        text = f"No retrieval in {mode_name(mode)}" if mode else "No retrieval"
        return f'<div class="retr off">{escape(text)}</div>'
    n = 0.0 if score is None else max(0.0, min(1.0, float(score)))
    t = 0.30 if threshold is None else float(threshold)
    t = max(0.0, min(1.0, t))
    ok = score is not None and float(score) >= t
    klass = "retr ok" if ok else "retr low"
    width = f"{n * 100:.1f}%"
    tick = f"{t * 100:.1f}%"
    verdict = "clears" if ok else "below"
    return (
        f'<div class="{klass}">'
        '<span class="lbl">Retrieval</span>'
        f'<span class="bar" role="meter" aria-valuemin="0" aria-valuemax="1" aria-valuenow="{n:.2f}"'
        f' aria-label="Best match {_fmt_score(score)}, threshold {format_threshold(t)}">'
        f'<span class="fill" style="width:{width}"></span>'
        f'<span class="tick" style="left:{tick}"></span>'
        "</span>"
        f'<span class="val"><b>{_fmt_score(score)}</b> {verdict} {format_threshold(t)}</span>'
        "</div>"
    )


def source_chips_html(hits, retrieval_off=False):
    if retrieval_off:
        return '<div class="no-sources">no sources · retrieval off in this mode</div>'
    if not hits:
        return '<div class="no-sources">no sources</div>'
    shown = hits[:CHIP_CAP]
    extra = len(hits) - len(shown)
    parts = ['<div class="chips">']
    for hit in shown:
        key = escape(source_key(hit.get("source")))
        name = escape(chip_name(hit))
        dense = _opt_float(hit.get("dense"))
        path = escape(str(hit.get("path") or ""))
        title = f"score {_fmt_score(dense)} · {path}"
        parts.append(
            f'<span class="chip" title="{title}">'
            f'<span class="k">{key}</span>'
            f'<span class="n">{name}</span>'
            f"</span>"
        )
    if extra > 0:
        parts.append(f'<span class="chip more">+{extra}</span>')
    parts.append("</div>")
    return "".join(parts)


def metrics_html(record):
    n = source_count(record)
    bits = [
        f'<b>{float(record.get("latency_s") or 0):.1f}</b> s',
        f'<b>{format_total_ram(record.get("total_mb"))}</b> RAM total',
    ]
    if record.get("abstained"):
        unit = "candidate checked" if n == 1 else "candidates checked"
        bits.append(f"<b>{n}</b> {unit}")
        bits.append("abstained")
    else:
        count, _, unit = source_count_label(n).partition(" ")
        bits.append(f"<b>{count}</b> {unit}")
    return '<div class="metrics">' + "".join(f"<span>{b}</span>" for b in bits) + "</div>"


def ram_split_html(record):
    app_mb = float(record.get("app_rss_mb") or 0)
    ollama_mb = float(record.get("ollama_rss_mb") or 0)
    return (
        f'<p class="ram-split">app <b>{app_mb:.0f}</b> MB'
        f' · ollama <b>{ollama_mb:.0f}</b> MB</p>'
    )


_TICK_RE = re.compile(r"`([^`\n]+)`")


def idk_card_html(record):
    _, answer = split_reasoning(record.get("answer") or "")
    body = _TICK_RE.sub(r"<code>\1</code>", escape(answer)).replace("\n", "<br>")
    return (
        '<div class="idk-body">'
        f'<span class="idk-star">{STAR_SVG}</span>'
        "<h4>No fix on this one</h4>"
        f"<p>{body}</p>"
        "</div>"
        '<div class="polaris-idk-flag" hidden></div>'
    )


def best_match_html(record):
    hits = record.get("hits") or []
    top = _best_hit(hits)
    if top is None:
        return empty_slot_html("best")
    src = source_key(top.get("source")).lower()
    title = (top.get("title") or "").strip()
    if title and "(" not in title:
        name = title
    else:
        name = short_doc_name(top)
    gap = " " if name else ""
    return f'<p class="best">Best match: <code>{escape(src)}{gap}{escape(name)}</code></p>'


def evidence_summary_text(record):
    n = source_count(record)
    phrase = evidence_source_phrase(record.get("hits") or [])
    label = f"Evidence, {source_count_label(n)}"
    if phrase:
        label += f", best {phrase}"
    return label


def evidence_summary_html(record):
    n = source_count(record)
    phrase = evidence_source_phrase(record.get("hits") or [])
    src_html = f' <span class="best-src">{escape(phrase)}</span>' if phrase else ""
    return (
        '<span class="chev" aria-hidden="true"></span>'
        f'<span class="ev-title">Evidence</span> <span class="count">{source_count_label(n)}</span>{src_html}'
    )


def evidence_rows_html(hits):
    if not hits:
        return '<p class="no-sources">no hits</p>'
    parts = ['<ol class="ev">']
    for hit in hits:
        dense = _fmt_score(hit.get("dense"))
        src = source_key(hit.get("source"))
        parts.append(
            "<li>"
            f'<span class="rank">{int(hit["rank"])}</span>'
            f'<span class="src">{escape(src)}</span>'
            f'<span class="title">{escape(str(hit.get("title") or ""))}</span>'
            f'<span class="sc">cosine {dense}</span>'
            f'<p class="snip">{escape(str(hit.get("snippet") or ""))}</p>'
            "</li>"
        )
    parts.append("</ol>")
    return "".join(parts)


def debug_rows_html(hits):
    if not hits:
        return '<p class="no-sources">no hits</p>'
    parts = ['<ol class="dbg-ev">']
    for hit in hits:
        parts.append(
            "<li>"
            f'<span class="rank">{int(hit["rank"])}</span>'
            f'<span class="path">{escape(str(hit.get("path") or ""))}</span>'
            f'<span class="sc">rrf {_fmt_score(hit.get("rrf"), 4)}'
            f' · bm25 {_fmt_score(hit.get("bm25"), 3)}'
            f' · cos {_fmt_score(hit.get("dense"))}</span>'
            "</li>"
        )
    parts.append("</ol>")
    return "".join(parts)


def evidence_panel_html(record):
    hits = record.get("hits") or []
    thr_raw = record.get("threshold")
    thr = format_threshold(0.30 if thr_raw is None else thr_raw)
    parts = []
    top = _best_hit(hits)
    if top is not None:
        src = source_key(top.get("source")).lower()
        dense = _fmt_score(top.get("dense"))
        parts.append(
            f'<p class="best-panel">Best match <span class="src">{escape(src)}</span>'
            f" scored cosine {dense}, threshold {thr}</p>"
        )
        parts.append(evidence_rows_html(hits))
    else:
        parts.append('<p class="no-sources">no hits</p>')
    parts.append(ram_split_html(record))
    parts.append('<details class="dbg"><summary>Debug</summary>')
    parts.append(debug_rows_html(hits))
    parts.append("</details>")
    return "".join(parts)


def evidence_block_html(record, collapsed=True):
    open_attr = "" if collapsed else " open"
    return (
        f'<details class="ev-box"{open_attr}>'
        f"<summary>{evidence_summary_html(record)}</summary>"
        f"{evidence_panel_html(record)}"
        "</details>"
    )


def answer_body_html(text):
    parts = []
    for kind, chunk in split_code_blocks(text):
        if kind == "code":
            parts.append(f'<pre class="cmd"><code>{escape(chunk)}</code></pre>')
        elif chunk.strip():
            parts.append(f'<div class="prose">{escape(chunk).replace(chr(10), "<br>")}</div>')
    if not parts:
        return '<div class="prose"></div>'
    return "".join(parts)


def card_html(record, align_slots=False):
    mode = record.get("mode") or ""
    rag = uses_rag(mode)
    abstained = bool(record.get("abstained"))
    klass = "card idk-card" if abstained else "card"
    parts = [f'<div class="{klass}">']
    if abstained:
        parts.append('<div class="polaris-idk-flag" hidden></div>')
    parts.append(card_head_html(mode))
    if abstained:
        parts.append(idk_card_html(record))
    else:
        parts.append(answer_body_html(record.get("answer") or ""))
    if rag:
        parts.append(evidence_block_html(record, collapsed=True))
    elif align_slots:
        parts.append(empty_slot_html("ev", f"No sources in {mode_name(mode)}"))
    parts.append(
        retrieval_bar_html(
            record.get("retrieval_max_dense"),
            record.get("threshold"),
            retrieval_off=not rag,
            mode=mode,
        )
    )
    if abstained:
        parts.append(best_match_html(record))
    elif align_slots:
        parts.append(empty_slot_html("best"))
    parts.append(metrics_html(record))
    parts.append("</div>")
    return "".join(parts)


def _dots_html():
    return '<span class="dots" aria-hidden="true">' + "<i></i>" * 9 + "</span>"


def thinking_html(label, still=STILL_WORKING):
    """Loader for a running answer: a 3x3 constellation breathing out from its center star, the stage label,
    then a cycle of thinking phrases. Pure CSS, so it keeps moving while the main thread is busy."""
    words = "".join(f"<span>{escape(w)}</span>" for w in (label + "…", *THINKING_WORDS))
    return (
        f'<div class="thinking" role="status" aria-label="{escape(label)}">'
        + _dots_html()
        + '<span class="thinking-text" aria-hidden="true">'
        f'<span class="thinking-label">{words}</span>'
        f'<span class="thinking-note">{escape(still)}</span>'
        "</span></div>"
    )


def _css_string(text):
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def mode_menu_css(model, modes):
    """Per-run CSS: the model name on the mode dropdown, and each mode's hint inside its open menu."""
    open_menu = ':root:has([class*="st-key-mode-"] [aria-expanded="true"]) [data-baseweb="popover"] li[role="option"]'
    rules = [f":root{{--polaris-model:{_css_string(model)}}}"]
    for i, mode in enumerate(modes, 1):
        rules.append(f"{open_menu}:nth-child({i})::after{{content:{_css_string(mode_hint(mode))}}}")
    return "<style>" + "".join(rules) + "</style>"


def stopped_html():
    return '<p class="stopped">Stopped before an answer came back.</p>'


def running_card_html(mode, label):
    return '<div class="card running-card">' + card_head_html(mode) + thinking_html(label) + "</div>"


def thinking_label(mode, loading_index=False):
    if loading_index:
        return "Loading the lab docs index"
    if uses_rag(mode):
        return "Searching the lab docs"
    if mode == "cot":
        return "Thinking it through"
    return "Writing an answer"


def waiting_html(mode, after_label):
    return (
        '<div class="card waiting-card">'
        + card_head_html(mode)
        + '<div class="slot-empty body"><div class="thinking queued">'
        + _dots_html()
        + f'<span class="thinking-text"><span class="queued-label">Starts when {escape(after_label)} finishes</span></span>'
        + "</div></div>"
        + empty_slot_html("ev", "No sources yet")
        + '<div class="retr off">No retrieval yet</div>'
        + empty_slot_html("best")
        + '<div class="metrics slot-empty foot"></div>'
        + "</div>"
    )


def user_turn_html(question):
    return f'<div class="q" role="heading" aria-level="2">{escape(question)}</div>'


def preview_text(answer):
    _, body = split_reasoning(answer or "")
    prose = " ".join(chunk for kind, chunk in split_code_blocks(body) if kind == "md")
    return first_line(prose.replace("`", ""))


def compare_compact_html(left, right):
    def block(rec):
        mode = rec.get("mode") or ""
        preview = preview_text(rec.get("answer") or "")
        return (
            '<div class="cmp-cell">'
            + card_head_html(mode)
            + f'<p class="preview">{escape(preview)}</p>'
            + "</div>"
        )

    return '<div class="cmp-compact">' + block(left) + block(right) + "</div>"


def avg_latency(records):
    if not records:
        return None
    return sum(float(r.get("latency_s") or 0.0) for r in records) / len(records)


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", re.DOTALL)


def split_code_blocks(text):
    """Yield ('md'|'code', chunk) pairs so command fences can use st.code."""
    body = text or ""
    pos = 0
    for m in _FENCE_RE.finditer(body):
        if m.start() > pos:
            yield "md", body[pos : m.start()]
        yield "code", m.group(1).strip("\n")
        pos = m.end()
    if pos < len(body):
        yield "md", body[pos:]


def sidebar_kv_html(rows):
    parts = ['<div class="kv-list">']
    for key, value in rows:
        parts.append(
            f'<div class="kv"><span class="k">{escape(str(key))}</span>'
            f'<span class="v">{escape(str(value))}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def warm_label(warmed):
    return "ready" if warmed else "idle"


def style_path():
    return Path(__file__).with_name("style.css")


def orbit_style_path():
    return Path(__file__).with_name("star-orbit.css")


def injected_css():
    return (
        style_path().read_text(encoding="utf-8")
        + "\n"
        + orbit_style_path().read_text(encoding="utf-8")
    )
