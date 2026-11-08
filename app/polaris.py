"""Polaris demo helpers: feature flags, run records, and HTML fragments.

Render functions that need Streamlit live in app.py. Generation is always
app.engine.answer_question.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
from html import escape
from pathlib import Path
from typing import TypedDict

import psutil

from app.engine import (
    MODE_LABELS,
    MODE_SUBS,
    MODE_SUFFIX,
    mode_choices,
    uses_rag,
)
from app.answer_format import parse_answer, split_reasoning
from app.models import is_installed
from app.streaming import CancellableChat, ChatCancelled
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
    "model_key",
    "context_turns",
    "answer",
    "latency_s",
    "eval_count",
    "eval_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "load_duration",
    "tok_s",
    "gen_s",
    "prompt_s",
    "load_s",
    "done_reason",
    "ollama_version",
    "model_digest",
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
# The thesis config names (base, base+CoT, base+RAG, base+RAG+CoT), so the UI and the eval CSVs say the same thing.
MODE_NAMES = {mode: suffix or "BASE" for mode, suffix in MODE_SUFFIX.items()}
# CoT here is a prompt that asks for written steps; requests send think=false, so nothing is hidden reasoning.
MODE_HINTS = {
    "base": "Plain model, no documents",
    "cot": "Step-by-step prompt, no documents",
    "rag": "Retrieved lab docs, short cited answer",
    "full": "Retrieved lab docs, step-by-step prompt",
}
TAGLINE = "Offline help for the Linux lab"
STILL_WORKING = "The model runs on this machine, so this can take a moment."
ENGLISH_NOTE = "English questions only"
HISTORY_EMPTY = "Chats you start are saved on this machine."
DELETE_ASK = "Delete this chat? This can't be undone."
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


RAIL_ART_COLS, RAIL_ART_ROWS = 44, 9
_BAYER4 = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))
_DITHER_RAMP = " ˙·:░▒▓"
# The Little Dipper in (col, row) cells: Polaris, the handle (delta, epsilon, zeta), then the bowl (Kochab, gamma, eta).
_RAIL_DIPPER = ((37, 1), (31, 1), (26, 2), (22, 3), (15, 3), (14, 5), (21, 5))


def _rail_ridge(c, cols, rows):
    return rows - 2.2 - 1.1 * math.sin(c / cols * math.pi * 1.25 + 0.2) - 0.45 * math.sin(c / 3.1 + 1)


def _rail_field(c, r, cols, rows):
    """Brightness 0-1: a glow around the pole star, haze along the horizon, and the ridge below it."""
    px, py = _RAIL_DIPPER[0]
    dx, dy = (c - px) / 6.5, (r - py) / 2.0
    ridge = _rail_ridge(c, cols, rows)
    if r >= ridge:
        return min(1.0, 0.62 + 0.12 * (r - ridge))
    return max(0.62 * math.exp(-(dx * dx + dy * dy)), 0.34 * math.exp(-(ridge - r) / 1.3))


def rail_art_html(cols=RAIL_ART_COLS, rows=RAIL_ART_ROWS):
    """The rail's header art: an ordered (Bayer) dither in text, toned in three bands by row the way a terminal
    banner is. Static and decorative, so it's hidden from screen readers."""
    levels = len(_DITHER_RAMP) - 1
    stars = {cell: i for i, cell in enumerate(_RAIL_DIPPER)}
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            if (c, r) in stars:
                star = stars[(c, r)]
                glyph = "*" if star == 0 else ("•" if star == 4 else "∙")
                cells.append(f'<b class="{"pole" if star == 0 else "dip"}">{glyph}</b>')
                continue
            v = _rail_field(c, r, cols, rows) * levels + (_BAYER4[r % 4][c % 4] / 16.0 - 0.5)
            cells.append(escape(_DITHER_RAMP[max(0, min(levels, int(round(v))))], quote=False))
        band = "hi" if r < rows // 3 else ("mid" if r < 2 * rows // 3 else "lo")
        lines.append(f'<span class="{band}">{"".join(cells)}</span>')
    return '<div class="rail-art" aria-hidden="true">' + "".join(lines) + "</div>"


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
    model_key: str
    context_turns: int
    answer: str
    latency_s: float
    eval_count: int | None
    eval_duration: int | None
    prompt_eval_count: int | None
    prompt_eval_duration: int | None
    load_duration: int | None
    tok_s: float | None
    gen_s: float | None
    prompt_s: float | None
    load_s: float | None
    done_reason: str | None
    ollama_version: str
    model_digest: str
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


def _opt_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
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
        "model_key": str(result.get("model_key") or "base"),
        "context_turns": int(result.get("context_turns") or 0),
        "answer": str(result.get("text") or ""),
        "latency_s": float(result.get("latency_s") or 0.0),
        "eval_count": _opt_int(result.get("eval_count")),
        "eval_duration": _opt_int(result.get("eval_duration")),
        "prompt_eval_count": _opt_int(result.get("prompt_eval_count")),
        "prompt_eval_duration": _opt_int(result.get("prompt_eval_duration")),
        "load_duration": _opt_int(result.get("load_duration")),
        "tok_s": _opt_float(result.get("tok_s")),
        "gen_s": _opt_float(result.get("gen_s")),
        "prompt_s": _opt_float(result.get("prompt_s")),
        "load_s": _opt_float(result.get("load_s")),
        "done_reason": str(result["done_reason"]) if result.get("done_reason") else None,
        "ollama_version": str(result.get("ollama_version") or ""),
        "model_digest": str(result.get("model_digest") or ""),
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
        for field in ("tok_s", "gen_s", "prompt_s", "load_s"):
            value = rec.get(field)
            row[field] = "" if value is None else f"{float(value):.4f}"
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


def card_head_html(mode, model=""):
    """model is set only when two answers side by side come from different models, so the heads tell them apart."""
    tag = f'<span class="model">{escape(model)}</span>' if model else ""
    return (
        '<div class="card-head">'
        f'<span class="mode">{escape(mode_name(mode))}</span>'
        f"{tag}"
        f'<span class="sub">{escape(mode_hint(mode))}</span>'
        "</div>"
    )


SCORES_NOTE = "cos is the MiniLM score the threshold (tick) gates. rrf is a rank score and bm25 is raw; neither is a confidence."
RATE_NOTE = "chars/s is counted in the app per 0.5 s, not tokens. tok/s is Ollama's eval_count / eval_duration."


def hits_from_chart(chart):
    """The streaming backend's RetrievalChart rows as hit dicts, so the live view and a saved record draw alike."""
    return [
        {
            "rank": row.rank,
            "source": row.path,
            "title": row.label,
            "path": row.path,
            "dense": row.dense_cosine,
            "rrf": row.rrf_rank_score,
            "bm25": row.bm25_raw,
        }
        for row in chart.rows
    ]


class RateTrace:
    """Characters per second while an answer streams, from the backend's live transport counts.

    Ollama picks its NDJSON chunk sizes, so these are characters, never tokens. Windows start at the first text, so
    the wait before it (model load, prompt evaluation) is reported once as the first-text time, not as a slow start.
    """

    def __init__(self, every=0.5):
        self.every = every
        self.samples = []
        self.first_s = None
        self.last = None
        self._mark = None

    def add(self, live):
        self.last = live
        if self._mark is None:
            self.first_s = live.elapsed_s
            self._mark = (live.elapsed_s, live.character_count)
            return
        t0, c0 = self._mark
        if live.elapsed_s - t0 >= self.every:
            self.samples.append((live.elapsed_s, (live.character_count - c0) / (live.elapsed_s - t0)))
            self._mark = (live.elapsed_s, live.character_count)

    def finish(self):
        if self.last is None or self._mark is None:
            return
        t0, c0 = self._mark
        if self.last.elapsed_s - t0 > 0.05 and self.last.character_count > c0:
            self.samples.append((self.last.elapsed_s, (self.last.character_count - c0) / (self.last.elapsed_s - t0)))
            self._mark = (self.last.elapsed_s, self.last.character_count)

    def frozen(self):
        """What the session keeps once the answer is done: the samples and the first-text time."""
        return {"samples": tuple(self.samples), "first_s": self.first_s}


def _trace_peak(samples):
    return max((rate for _, rate in samples), default=None)


def rate_svg(traces, width=240, height=48):
    """One or two traces (left solid with its area, right dashed) on shared axes: seconds since each answer's first
    text across, chars/s up."""
    drawn = [(cls, samples) for cls, samples in traces if len(samples) >= 2]
    if not drawn:
        return ""
    span = max(max(t for t, _ in samples) - samples[0][0] for _, samples in drawn) or 1e-9
    top = max(max(rate for _, rate in samples) for _, samples in drawn) * 1.15 or 1.0
    parts = []
    for cls, samples in drawn:
        t0 = samples[0][0]
        pts = [
            (round((t - t0) / span * width, 1), round(height - rate / top * (height - 2), 1)) for t, rate in samples
        ]
        line = " ".join(f"{x},{y}" for x, y in pts)
        if cls == "l":
            area = f"M0,{height} L" + " L".join(f"{x},{y}" for x, y in pts) + f" L{pts[-1][0]},{height} Z"
            parts.append(f'<path class="area" d="{area}"/>')
        parts.append(f'<polyline class="line {cls}" points="{line}" vector-effect="non-scaling-stroke"/>')
    peak = max(_trace_peak(samples) for _, samples in drawn)
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Characters per second while streaming, peak {peak:.0f}">' + "".join(parts) + "</svg>"
    )


def _run_label(run):
    side = run.get("side") or ""
    tag = f'<span class="side {side.lower()}">{escape(side)}</span>' if side else ""
    return (
        f'<p class="mon-cfg">{tag}<span class="m">{escape(run.get("model") or "")}</span>'
        f'<span class="md">{escape(mode_name(run.get("mode") or ""))}</span></p>'
    )


def _run_cells(run):
    """A run's numbers. Streaming: what the app can count (seconds, characters, chunks, chars/s), never tok/s.
    Finished: the response time and Ollama's own tok/s, or why there is none."""
    state = run.get("state")
    if state == "streaming":
        live = run.get("live")
        if live is None:
            return '<p class="mon-cells pending">waiting for the first text</p>'
        samples = run.get("samples") or ()
        current = samples[-1][1] if samples else None
        return (
            '<p class="mon-cells">'
            f'<span><b>{live.elapsed_s:.1f}</b> s</span>'
            f'<span><b>{live.character_count:,}</b> chars</span>'
            f'<span><b>{live.chunk_count:,}</b> chunks</span>'
            f'<span><b>{"—" if current is None else f"{current:.0f}"}</b> chars/s</span>'
            "</p>"
            '<p class="mon-cells pending tok-later">tok/s comes from Ollama when the answer ends</p>'
        )
    if state == "waiting":
        return '<p class="mon-cells pending">starts when the left answer finishes</p>'
    if state in ("stopped", "error"):
        return f'<p class="mon-cells pending">{"stopped" if state == "stopped" else "failed"}, no stats</p>'
    record = run.get("record") or {}
    extra = []
    first_s = run.get("first_s")
    peak = _trace_peak(run.get("samples") or ())
    if first_s is not None:
        extra.append(f"first text <b>{first_s:.1f}</b> s")
    if peak is not None:
        extra.append(f"peak <b>{peak:.0f}</b>")
    return (
        '<p class="mon-cells">' + answer_stats_html(record) + "</p>"
        + (f'<p class="mon-cells dim">{" · ".join(extra)}</p>' if extra else "")
    )


def _rate_view(runs, streaming):
    tag = '<span class="mon-tag live">streaming</span>' if streaming else '<span class="mon-tag">last answer</span>'
    if not runs:
        return (
            '<div class="mon-view mon-rate">'
            '<p class="mon-head"><span class="mon-tag">idle</span></p>'
            '<div class="mon-chart empty"><span>No answer in this chat yet</span></div>'
            f'<p class="mon-note">{escape(RATE_NOTE)}</p>'
            "</div>"
        )
    traces = [((run.get("side") or "l").lower(), tuple(run.get("samples") or ())) for run in runs]
    chart = rate_svg(traces)
    if not chart:
        restored = not streaming and all(not (run.get("samples") or ()) for run in runs)
        empty = "No trace: the rate is only recorded while an answer streams in this session" if restored else "collecting samples"
        chart_html = f'<div class="mon-chart empty"><span>{escape(empty)}</span></div>'
    else:
        chart_html = f'<div class="mon-chart">{chart}</div>'
    body = "".join(_run_label(run) + _run_cells(run) for run in runs)
    return (
        '<div class="mon-view mon-rate">'
        f'<p class="mon-head">{tag}<span class="unit">chars/s</span></p>'
        + chart_html
        + body
        + f'<p class="mon-note">{escape(RATE_NOTE)}</p>'
        "</div>"
    )


def _score(value, digits):
    """Leading-dot fixed point (.742), the compact form a narrow score column needs."""
    text = _fmt_score(value, digits)
    return text[1:] if text.startswith("0.") else text


def score_table_html(hits, threshold):
    t = 0.30 if threshold is None else float(threshold)
    tick = f"{max(0.0, min(1.0, t)) * 100:.1f}%"
    rows = []
    for hit in hits:
        dense = _opt_float(hit.get("dense"))
        width = 0.0 if dense is None else max(0.0, min(1.0, dense)) * 100
        ok = dense is not None and dense >= t
        rows.append(
            f'<li class="{"ok" if ok else "low"}">'
            f'<span class="rk">{int(hit.get("rank") or 0)}</span>'
            f'<span class="ttl" title="{escape(str(hit.get("path") or ""))}">{escape(chip_name(hit))}</span>'
            f'<span class="bar" aria-hidden="true"><span class="fill" style="width:{width:.1f}%"></span>'
            f'<span class="tick" style="left:{tick}"></span></span>'
            f'<span class="cos"><span class="sr">cosine </span>{_score(dense, 3)}</span>'
            f'<span class="raw"><span class="sr">rrf </span>{_score(hit.get("rrf"), 4)}</span>'
            f'<span class="raw"><span class="sr">bm25 </span>{_fmt_score(hit.get("bm25"), 2)}</span>'
            "</li>"
        )
    return (
        '<p class="sc-cols" aria-hidden="true"><span>#</span><span>doc</span><span>cos</span><span>rrf</span>'
        '<span>bm25</span></p><ol class="sc-rows">' + "".join(rows) + "</ol>"
    )


def _scores_view(runs):
    rag = [run for run in runs if uses_rag(run.get("mode") or "")]
    plain = [run for run in runs if not uses_rag(run.get("mode") or "")]
    if not runs:
        body = '<div class="mon-chart empty"><span>No retrieval yet in this chat</span></div>'
    elif not rag:
        body = (
            f'<div class="mon-chart empty"><span>No retrieval in {escape(mode_name(plain[0].get("mode") or ""))}, '
            "so there are no scores</span></div>"
        )
    else:
        run = next((r for r in reversed(rag) if r.get("hits") is not None), None)
        if run is None:
            body = '<div class="mon-chart empty"><span>Searching the lab docs</span></div>'
        else:
            hits = run.get("hits") or []
            t = 0.30 if run.get("threshold") is None else float(run["threshold"])
            best = max_dense(hits)
            verdict = "clears" if best is not None and best >= t else "below"
            side = run.get("side") or ""
            who = f'<span class="side {side.lower()}">{escape(side)}</span>' if side else ""
            shared = [r for r in rag if r is not run and [h.get("path") for h in (r.get("hits") or [])] == [h.get("path") for h in hits]]
            note = []
            if shared:
                note.append("Both sides retrieved the same chunks.")
            for p in plain:
                if p.get("side"):
                    note.append(f'{p["side"]} ({mode_name(p.get("mode") or "")}) retrieves nothing.')
            body = (
                f'<p class="mon-head mon-best"><span>{who}best cos <b>{_fmt_score(best, 3)}</b> {verdict} '
                f"{format_threshold(t)}</span>"
                + ("" if verdict == "clears" else '<span class="unit">model not called</span>')
                + "</p>"
                + (score_table_html(hits, t) if hits else '<p class="mon-cells pending">no hits</p>')
                + (f'<p class="mon-note">{escape(" ".join(note))}</p>' if note else "")
            )
    return (
        '<div class="mon-view mon-scores">'
        + body
        + f'<p class="mon-note">{escape(SCORES_NOTE)}</p>'
        "</div>"
    )


def monitor_html(runs, streaming=False):
    """The rail monitor: both views, one shown by the Rate/Scores switch, so flipping views never reruns the app."""
    return '<div class="mon">' + _rate_view(runs, streaming) + _scores_view(runs) + "</div>"


# No `checked` attribute: Streamlit renders this HTML through React, which would turn a checked radio into a
# controlled input and undo every click. Rate is the view whenever Scores isn't picked.
MONITOR_SWITCH = (
    '<div class="mon-switch" role="radiogroup" aria-label="Monitor view">'
    '<input type="radio" name="mon-view" id="mon-v-rate"><label for="mon-v-rate">Rate</label>'
    '<input type="radio" name="mon-view" id="mon-v-scores"><label for="mon-v-scores">Scores</label>'
    "</div>"
)


def finished_runs(message, index, traces):
    """The monitor's runs for the finished assistant message at `index`, with any stream trace this session kept."""

    def run(rec, side, key, missing=None):
        trace = (traces or {}).get(key) or {}
        if rec is None:
            return {
                "side": side,
                "model": (missing or {}).get("model") or "",
                "mode": (missing or {}).get("mode") or "",
                "state": "stopped" if (missing or {}).get("state") == "cancelled" else "error",
            }
        return {
            "side": side,
            "model": rec.get("model") or "",
            "mode": rec.get("mode") or "",
            "state": "done",
            "record": rec,
            "samples": tuple(trace.get("samples") or ()),
            "first_s": trace.get("first_s"),
            "hits": list(rec.get("hits") or []),
            "threshold": rec.get("threshold"),
        }

    if message.get("compare"):
        left, right = message["compare"]
        miss_l, miss_r = message.get("compare_unanswered") or (None, None)
        return [
            run(left, "L", f"answer-h{index}-l", miss_l),
            run(right, "R", f"answer-h{index}-r", miss_r),
        ]
    if message.get("record"):
        return [run(message["record"], "", f"answer-{index}")]
    return []


def last_runs(messages, traces):
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and (msg.get("record") or msg.get("compare")):
            return finished_runs(msg, i, traces)
    return []


def live_phase_label(mode, retrieved=False, weak=False):
    """What runs before the first text: the hybrid search in a RAG config, then the wait on Ollama."""
    if uses_rag(mode) and not retrieved:
        return "Searching the lab docs"
    if weak:
        return "No doc cleared the threshold, so the model is not called"
    return "Waiting for the model"


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


def timing_breakdown(record):
    parts = []
    for label, field in (("model load", "load_s"), ("prompt", "prompt_s"), ("generation", "gen_s")):
        value = _opt_float(record.get(field))
        if value is not None:
            parts.append(f"{label} {value:.2f} s")
    return ", ".join(parts)


TOK_S_NOT_GENERATED = "Retrieval was below the threshold, so the model was not called and no tokens were generated."
TOK_S_UNAVAILABLE = "Ollama reported no eval_count and eval_duration for this answer."


def answer_stats_html(record):
    """Response time and generation speed, always both. tok/s is Ollama's terminal eval_count / eval_duration;
    an abstained answer never calls the model, and an answer without those fields says so, never a made-up 0."""
    latency = float(record.get("latency_s") or 0.0)
    detail = timing_breakdown(record)
    title = "Question to answer" + (f": {detail}" if detail else "")
    chips = [
        f'<span class="stat" title="{escape(title)}"><span class="sr">Response time </span>'
        f"<b>{latency:.1f}</b> s</span>"
    ]
    tok_s = _opt_float(record.get("tok_s"))
    if tok_s is not None:
        tokens = _opt_int(record.get("eval_count"))
        gen_s = _opt_float(record.get("gen_s"))
        counted = tokens is not None and gen_s is not None
        speed = f"{tokens} tokens in {gen_s:.2f} s of generation" if counted else "Generation speed"
        chips.append(
            f'<span class="stat" title="{escape(speed)}"><span class="sr">Speed </span>'
            f"<b>{tok_s:.1f}</b> tok/s</span>"
        )
        if counted:
            chips.append(f'<span class="stat calc">{tokens} tok / {gen_s:.2f} s</span>')
    elif record.get("abstained"):
        chips.append(
            f'<span class="stat na" title="{escape(TOK_S_NOT_GENERATED)}"><span class="sr">Speed </span>'
            "tok/s not generated</span>"
        )
    else:
        chips.append(
            f'<span class="stat na" title="{escape(TOK_S_UNAVAILABLE)}"><span class="sr">Speed </span>'
            "tok/s unavailable</span>"
        )
    return '<span class="stats">' + "".join(chips) + "</span>"


RAM_TITLE = "Peak resident memory (RSS) of this app plus Ollama's processes, sampled every 0.25 s while answering"


def ram_html(record):
    """Peak RSS while the answer ran. Ollama's share is only counted when its processes are visible to this app."""
    total = _opt_float(record.get("total_mb")) or 0.0
    if total <= 0:
        return '<span class="na" title="No memory sample was recorded for this answer.">RAM not measured</span>'
    app_mb = _opt_float(record.get("app_rss_mb")) or 0.0
    ollama_mb = _opt_float(record.get("ollama_rss_mb")) or 0.0
    if ollama_mb <= 0:
        title = f"{RAM_TITLE}. No Ollama process was visible, so this is the app alone."
        return f'<span title="{escape(title)}"><b>{format_total_ram(total)}</b> peak RSS, app only</span>'
    title = f"{RAM_TITLE}: app {app_mb:.0f} MB, Ollama {ollama_mb:.0f} MB."
    return f'<span title="{escape(title)}"><b>{format_total_ram(total)}</b> peak RSS app+Ollama</span>'


def metrics_html(record):
    n = source_count(record)
    bits = []
    if record.get("abstained"):
        unit = "candidate checked" if n == 1 else "candidates checked"
        bits.append(f"<b>{n}</b> {unit}")
        bits.append("abstained")
    else:
        count, _, unit = source_count_label(n).partition(" ")
        bits.append(f"<b>{count}</b> {unit}")
    return (
        '<div class="metrics">'
        + answer_stats_html(record)
        + ram_html(record)
        + "".join(f"<span>{b}</span>" for b in bits)
        + cut_off_html(record)
        + "</div>"
    )


CUT_OFF_TITLE = ("Ollama ended this answer with done_reason=length: generation hit a token or context limit, so the "
                 "text may stop mid-way. The app shows everything Ollama sent.")


def cut_off_html(record):
    """Only when Ollama itself says it stopped at a limit; an answer that ended on its own says nothing here."""
    if record.get("done_reason") != "length":
        return ""
    return f'<span class="stat warn" title="{escape(CUT_OFF_TITLE)}">cut off at a length limit</span>'


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


# The loader: a 3x3 cube in oblique text projection (front face, the top above it, the right side beside it) turning
# its top and right faces. Each move is two frames at rest and one mid-turn frame in which the turning row or column
# squashes toward the way it travels, so every sticker keeps its own cells and the frame never changes size.
CUBE_COLS, CUBE_ROWS = 12, 6
CUBE_MOVES = ("U", "R", "U'", "R'", "R", "U", "R'", "U'")  # a commutator, then its inverse: back to solved
CUBE_FRAMES = 3 * len(CUBE_MOVES)
CUBE_SECONDS = 3.6
_CUBE_FACES = {(0, 1, 0): "w", (0, -1, 0): "y", (0, 0, 1): "g", (0, 0, -1): "b", (1, 0, 0): "r", (-1, 0, 0): "o"}
# A quarter turn of the U (top) or R (right) layer, clockwise as seen from that side, on positions and normals alike.
_CUBE_TURNS = {"U": (1, lambda x, y, z: (-z, y, x)), "R": (0, lambda x, y, z: (x, z, -y))}
_CUBE_GLYPHS = {"f": "██", "t": "▄▄", "s": "▌"}
_CUBE_MID = {"U": {"f": "█ "}, "U'": {"f": " █"}, "R": {"f": "▀▀", "t": "▀▀"}, "R'": {"f": "▄▄", "t": "██"}}


def _cube_solved():
    cube = {}
    for normal, color in _CUBE_FACES.items():
        axis = next(i for i, n in enumerate(normal) if n)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                pos = [a, b]
                pos.insert(axis, normal[axis])
                cube[tuple(pos), normal] = color
    return cube


def _cube_turn(cube, move):
    axis, quarter = _CUBE_TURNS[move[0]]
    turns = 3 if move.endswith("'") else 1
    out = {}
    for (pos, normal), color in cube.items():
        if pos[axis] == 1:
            for _ in range(turns):
                pos, normal = quarter(*pos), quarter(*normal)
        out[pos, normal] = color
    return out


def _cube_cells(cube, move=None):
    """Rows of (glyph, classes) cells: front stickers two columns wide, the top's rows stepping one column right per
    row of depth, the right side one column wide stepping up."""
    grid = [[(" ", "")] * CUBE_COLS for _ in range(CUBE_ROWS)]
    for ((x, y, z), normal), color in cube.items():
        depth = 1 - z
        if normal == (0, 0, 1):
            row, col, face = 4 - y, 3 * x + 3, "f"
        elif normal == (0, 1, 0):
            row, col, face = 2 - depth, 3 * x + 4 + depth, "t"
        elif normal == (1, 0, 0):
            row, col, face = 4 - y - depth, 9 + depth, "s"
        else:
            continue
        glyphs = _CUBE_GLYPHS[face]
        if move and (x, y, z)[_CUBE_TURNS[move[0]][0]] == 1:
            glyphs = _CUBE_MID[move].get(face, glyphs)
        for i, glyph in enumerate(glyphs):
            if glyph != " ":
                grid[row][col + i] = (glyph, f"{color} {face}")
    return grid


def _cube_frames():
    frames, cube = [], _cube_solved()
    for move in CUBE_MOVES:
        rest = _cube_cells(cube)
        frames += [rest, rest, _cube_cells(cube, move)]
        cube = _cube_turn(cube, move)
    return frames


_CUBE_CELLS = _cube_frames()


def cube_frame(t):
    """Frame t of the loop as plain rows of text."""
    return ["".join(glyph for glyph, _ in row) for row in _CUBE_CELLS[t % CUBE_FRAMES]]


def _cube_lines(frame):
    lines = []
    for row in frame:
        line, i = "", 0
        while i < len(row):
            classes = row[i][1]
            j = i
            while j < len(row) and row[j][1] == classes:
                j += 1
            run = "".join(g for g, _ in row[i:j])
            line += f'<i class="{classes}">{run}</i>' if classes else run
            i = j
        lines.append(line)
    return lines


# One line of HTML, every frame stacked in a reel that one CSS steps() animation slides through.
_CUBE_REEL = "<br>".join(line for frame in _CUBE_CELLS for line in _cube_lines(frame))
_CUBE_REST = "<br>".join(_cube_lines(_CUBE_CELLS[0]))


def cube_html(still=False, elapsed=0.0):
    """The turning cube, or its first (solved) frame when still. `elapsed` seconds into the loop: a loader drawn again
    with a new label picks the turn up where the last one was rather than at its first frame."""
    if still:
        return f'<span class="cube still" aria-hidden="true"><span class="cube-reel">{_CUBE_REST}</span></span>'
    delay = f' style="animation-delay:-{elapsed % CUBE_SECONDS:.2f}s"' if elapsed else ""
    return f'<span class="cube" aria-hidden="true"><span class="cube-reel"{delay}>{_CUBE_REEL}</span></span>'


def thinking_html(label, elapsed=0.0, note=None):
    """The loader: the turning cube beside one plain label for what runs. Without a note, STILL_WORKING fades in once
    the wait passes 15 s; a note given is shown at once."""
    if note is None:
        wait = f' style="animation-delay:{max(0.0, 15 - elapsed):.1f}s"' if elapsed else ""
        note_html = f'<span class="thinking-note"{wait}>{escape(STILL_WORKING)}</span>'
    else:
        note_html = f'<span class="thinking-note now">{escape(note)}</span>'
    return (
        '<div class="thinking" role="status">'
        + cube_html(elapsed=elapsed)
        + '<span class="thinking-text">'
        f'<span class="thinking-label">{escape(label)}…</span>'
        + note_html
        + "</span></div>"
    )


REASONING_LABEL = "Writing its reasoning, under Show reasoning"
REASONING_NOTE = "The answer shows from its Answer: line, or when the reply ends."


def flags_html(sky, daylight=False, generating=False):
    """The hidden marker the stylesheet reads through :root:has(); polaris-generating holds the ambient motion."""
    flags = ["polaris-flags", f"polaris-sky-{sky}"]
    if daylight:
        flags.append("polaris-daylight")
    if generating:
        flags.append("polaris-generating")
    return f'<div class="{" ".join(flags)}"></div>'


def composer_note_html(conn):
    return f'<div class="composer-conn">{conn}<span class="composer-note">{escape(ENGLISH_NOTE)}</span></div>'


def stopped_html():
    return '<p class="stopped">Stopped. The partial answer was not kept.</p>'


def cancelled_side_html():
    return '<p class="stopped">Cancelled</p>'


def thinking_label(mode, loading_index=False):
    """Names only work that runs: BASE and +CoT never touch the docs, and no mode has a hidden thinking phase."""
    if loading_index:
        return "Loading the lab docs index"
    if uses_rag(mode):
        return "Generating an answer from the lab docs"
    return "Generating an answer"


def waiting_html(mode, after_label, model=""):
    return (
        '<div class="card waiting-card">'
        + card_head_html(mode, model)
        + '<div class="slot-empty body"><div class="thinking queued">'
        + cube_html(still=True)
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


def preview_text(answer, mode=""):
    parsed = parse_answer(answer or "", mode=mode)
    prose = " ".join(b.text for b in parsed.blocks if b.kind == "md")
    return first_line(prose.replace("`", ""))


def loose_commands_html(count):
    """Above commands the model listed after its steps without tying them to one: the app doesn't guess the step."""
    what = "this command" if count == 1 else f"these {count} commands"
    return (f'<p class="cmd-loose">The model listed {what} after its steps without saying which step each belongs '
            "to.</p>")


def source_line_html(answer, mode, hits):
    """Under an answer: the excerpts it cites, by name, or that there were none to cite. A [n] that points at no
    retrieved excerpt was taken out of the text and is named here, so nothing is quietly dropped."""
    dropped = ", ".join(f"[{n}]" for n in answer.dropped)
    if not uses_rag(mode):
        line = f"No documents in {escape(mode_name(mode))}, so nothing to cite"
        if dropped:
            line += f" · removed {dropped} the model wrote"
        return f'<p class="src-line none">{line}</p>'
    names = []
    for n in answer.cited:
        hit = hits[n - 1] if n - 1 < len(hits) else {}
        names.append(f"<b>[{n}]</b> {escape(doc_label(hit) or f'excerpt {n}')}")
    line = ("Source " + " · ".join(names)) if names else "Source not cited by the model"
    if dropped:
        line += f" · removed {dropped}, not among the {len(hits)} retrieved"
    return f'<p class="src-line">{line}</p>'


def compare_model_tags(left, right):
    """Model names for the two heads of a compare, or blanks when both sides ran the same model."""
    left_model = str((left or {}).get("model") or "")
    right_model = str((right or {}).get("model") or "")
    if left and right and left_model != right_model:
        return left_model, right_model
    return "", ""


def compare_compact_html(left, right):
    left_tag, right_tag = compare_model_tags(left, right)

    def block(rec, tag):
        mode = rec.get("mode") or ""
        preview = preview_text(rec.get("answer") or "", mode)
        return (
            '<div class="cmp-cell">'
            + '<div class="cmp-top">'
            + card_head_html(mode, tag)
            + answer_stats_html(rec)
            + "</div>"
            + f'<p class="preview">{escape(preview)}</p>'
            + "</div>"
        )

    return '<div class="cmp-compact">' + block(left, left_tag) + block(right, right_tag) + "</div>"


def compare_summary_html(left_model, left_mode, right_model, right_mode):
    """The composer's line while compare is on: which config answers on each side. The model is named once when
    both sides share it, and per side when they don't."""
    left = f"<b>{escape(mode_name(left_mode))}</b>"
    right = f"<b>{escape(mode_name(right_mode))}</b>"
    if left_model == right_model:
        body = f'<span class="m">{escape(left_model)}</span> {left}<span class="vs">vs</span>{right}'
    else:
        body = (
            f'<span class="m">{escape(left_model)}</span> {left}<span class="vs">vs</span>'
            f'<span class="m">{escape(right_model)}</span> {right}'
        )
    return f'<p class="composer-cmp">{body}</p>'


def config_summary_html(model, mode):
    """The composer's line outside compare: the model and config the next question goes to."""
    return f'<p class="composer-cmp"><span class="m">{escape(model)}</span> <b>{escape(mode_name(mode))}</b></p>'


def model_choice_label(option, names):
    return option.name + (" · not installed" if is_installed(option.name, names) is False else "")


def model_status_html(option, names, ollama_up):
    """One line under the model picker: what this model is, and whether Ollama has it."""
    only = " · runs +RAG+CoT only" if option.key == "distilled" else ""
    if not ollama_up or names is None:
        return f'<p class="side-note model-st">{escape(option.role)}{only} · can\'t check, Ollama is not running</p>'
    if is_installed(option.name, names):
        return f'<p class="side-note model-st"><span class="ok">installed</span> · {escape(option.role)}{only}</p>'
    return (
        '<p class="side-note model-st err">not installed · '
        f"<code>{escape(option.create)}</code></p>"
    )


def models_table_html(options, names, ollama_up):
    rows = []
    for option in options:
        state = is_installed(option.name, names) if ollama_up else None
        label = {True: "installed", False: "not installed", None: "unknown"}[state]
        fix = "" if state else f"<code>{escape(option.create)}</code>"
        rows.append(
            f'<li class="{label.replace(" ", "-")}"><span class="n">{escape(option.name)}</span>'
            f'<span class="r">{escape(option.role)}</span><span class="s">{label}</span>{fix}</li>'
        )
    return (
        '<ul class="models">' + "".join(rows) + "</ul>"
        '<p class="dlg-note">OLLAMA_MODEL sets the default base and OLLAMA_DISTILLED_MODEL names the student. '
        "A missing model is never swapped for another one.</p>"
    )


def records_in(messages):
    """Run records in the order they were answered, compare pairs left then right."""
    records = []
    for message in messages:
        if message.get("record"):
            records.append(message["record"])
        for rec in message.get("compare") or ():
            if rec:
                records.append(rec)
    return records


_MD_PUNCT = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$])")


def plain_label(text):
    """Widget labels are Markdown; escape a saved question so it shows exactly as typed ($HOME, *.tar.gz, `ls`).
    Streamlit swaps :name: for an emoji or icon even when the colons are escaped, so each colon gets a zero-width
    space after it instead."""
    flat = " ".join(str(text or "").split())
    return _MD_PUNCT.sub(r"\\\1", flat).replace(":", ":\u200b")


def history_note_html(text, error=False):
    klass = "side-note err" if error else "side-note"
    return f'<p class="{klass}">{escape(text)}</p>'


def history_error_text(path):
    home = str(Path.home())
    shown = str(path)
    if shown == home or shown.startswith(home + os.sep):
        shown = "~" + shown[len(home):]
    return f"Couldn't open chat history in {shown}, so chats aren't being saved. Check that the folder is writable."


def avg_latency(records):
    if not records:
        return None
    return sum(float(r.get("latency_s") or 0.0) for r in records) / len(records)


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
