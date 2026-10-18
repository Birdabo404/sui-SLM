"""Polaris demo helpers: feature flags, run records, and HTML fragments.

Render functions that need Streamlit live in app.py. Generation is always
app.engine.answer_question.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
from html import escape
from pathlib import Path
from typing import TypedDict

from app.engine import MODE_LABELS, MODE_SUBS, mode_choices, uses_rag
from shared.frozen import format_threshold

STAR_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M12 0L14.2 9.8 24 12 14.2 14.2 12 24 9.8 14.2 0 12 9.8 9.8Z"/>'
    "</svg>"
)
STAR_LUX_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path fill="url(#polaris-star-lux-fill)" d="M12 0C12.45 7.1 14.7 11.2 22 12C14.7 12.8 12.45 16.9 12 24C11.55 16.9 9.3 12.8 2 12C9.3 11.2 11.55 7.1 12 0Z"/>'
    '<path fill="#FFF9EC" opacity=".55" d="M12 6C12.2 9.6 12.9 11.3 15.5 12C12.9 12.7 12.2 14.4 12 18C11.8 14.4 11.1 12.7 8.5 12C11.1 11.3 11.8 9.6 12 6Z"/>'
    "</svg>"
)
GLINT_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path fill="#FFEBC2" d="M12 0C12.4 9 15 11.6 24 12C15 12.4 12.4 15 12 24C11.6 15 9 12.4 0 12C9 11.6 11.6 9 12 0Z"/>'
    '<circle cx="12" cy="12" r="2.2" fill="#FFF8EA"/>'
    "</svg>"
)
STAR_DEFS = (
    '<svg width="0" height="0" aria-hidden="true">'
    "<defs>"
    '<radialGradient id="polaris-star-lux-fill" cx="12" cy="12" r="11.5" gradientUnits="userSpaceOnUse">'
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
SOURCE_KEY_RULES = (
    ("tldr", "TLDR"),
    ("man", "MAN"),
    ("gnome", "GNOME"),
    ("ubuntu", "UBUNTU-HELP"),
    ("help", "UBUNTU-HELP"),
)


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


def _fmt_score(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def empty_slot_html(kind="", label=""):
    klass = f"slot-empty {kind}".strip()
    inner = escape(label) if label else "&nbsp;"
    return f'<div class="{klass}">{inner}</div>'


def _header_star_html():
    glints = f'<span class="glints"><b>{GLINT_SVG}</b><b>{GLINT_SVG}</b></span>'
    trail = "<span class=\"trail\">" + ("<i></i>" * 10) + f"<b>{GLINT_SVG}</b>" * 3 + "</span>"
    glow = (
        '<span class="star-fx star-glow" aria-hidden="true">'
        '<span class="rig"><span class="body"><span class="halo"><span class="glow"></span></span></span></span>'
        "</span>"
    )
    star = (
        f'<span class="star-fx" aria-hidden="true">{glints}{trail}'
        f'<span class="rig"><span class="body"><span class="breathe"><span class="sway">{STAR_LUX_SVG}</span></span></span></span>'
        "</span>"
    )
    return (
        glow
        + '<span class="base" aria-hidden="true">polaris</span>'
        + star
        + '<span class="ink">polaris</span>'
    )


def header_band_html(slim=False, animate=False):
    classes = ["band"]
    if slim:
        classes.append("slim")
        classes.append("rest")
    if animate:
        classes.append("wrap-on")
    mark_class = "mark is-wrapping" if animate else "mark rest"
    return (
        f'<div class="{" ".join(classes)}">'
        + STAR_DEFS
        + f'<span class="{mark_class}" tabindex="0"><span class="word">'
        + _header_star_html()
        + "</span></span>"
        '<span class="tag">offline help for the linux lab</span>'
        '<span class="spacer"></span>'
        '<span class="pill-offline">offline</span>'
        "</div>"
    )


def empty_hero_html():
    return (
        '<div class="empty">'
        f'<div class="mark">polaris{STAR_SVG}</div>'
        '<div class="const"></div>'
        '<p class="tag">offline help for the linux lab</p>'
        "</div>"
    )


def card_head_html(mode):
    return (
        '<div class="card-head">'
        f'<span class="mode">[ {escape(mode_label(mode))} ]</span>'
        f'<span class="sub">{escape(mode_sub(mode))}</span>'
        "</div>"
    )


def retrieval_bar_html(score, threshold, retrieval_off=False, mode=""):
    if retrieval_off:
        name = mode_label(mode)
        text = f"no retrieval in {name}" if mode else "no retrieval"
        return f'<div class="retr off">{escape(text)}</div>'
    n = 0.0 if score is None else max(0.0, min(1.0, float(score)))
    t = 0.30 if threshold is None else float(threshold)
    t = max(0.0, min(1.0, t))
    ok = score is not None and float(score) >= t
    klass = "retr ok" if ok else "retr low"
    width = f"{n * 100:.1f}%"
    tick = f"{t * 100:.1f}%"
    score_lbl = _fmt_score(score)
    return (
        f'<div class="{klass}">'
        f'<div class="retr-top"><span class="lbl">retrieval</span> <b>{score_lbl}</b></div>'
        f'<div class="bar">'
        f'<span class="fill" style="width:{width}"></span>'
        f'<span class="tick" style="left:{tick}"></span>'
        f'<span class="knob" style="left:{width}"></span>'
        f"</div>"
        f'<div class="scale">'
        f'<span class="mark start" style="left:0%">0</span>'
        f'<span class="mark thr" style="left:{tick}">{format_threshold(t)}</span>'
        f'<span class="mark score" style="left:{width}">{score_lbl}</span>'
        f'<span class="mark end" style="left:100%">1</span>'
        f"</div>"
        f"</div>"
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
        f'<b>{format_total_ram(record.get("total_mb"))}</b> total',
    ]
    if record.get("abstained"):
        unit = "candidate checked" if n == 1 else "candidates checked"
        bits.append(f"{n} {unit}")
        bits.append("abstain")
    else:
        bits.append(source_count_label(n))
    return f'<div class="metrics">{" · ".join(bits)}</div>'


def ram_split_html(record):
    app_mb = float(record.get("app_rss_mb") or 0)
    ollama_mb = float(record.get("ollama_rss_mb") or 0)
    return (
        f'<p class="ram-split">app <b>{app_mb:.0f}</b> MB'
        f' · ollama <b>{ollama_mb:.0f}</b> MB</p>'
    )


def idk_card_html(record):
    body = escape(record.get("answer") or "").replace("\n", "<br>")
    return (
        '<div class="idk-body">'
        f'<span class="idk-star">{STAR_SVG}</span>'
        "<h4>no fix on this one</h4>"
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
    return f'<p class="best">best match: {escape(src)}{gap}{escape(name)}</p>'


def evidence_summary_text(record):
    n = source_count(record)
    phrase = evidence_source_phrase(record.get("hits") or [])
    label = f"Evidence · {source_count_label(n)}"
    if phrase:
        label += f" · {phrase}"
    return label


def evidence_summary_html(record):
    n = source_count(record)
    phrase = evidence_source_phrase(record.get("hits") or [])
    src_html = ""
    if phrase:
        key, _, rest = phrase.partition(" ")
        src_html = f' · <span class="src">{escape(key)}</span>'
        if rest:
            src_html += f" {escape(rest)}"
    return f"Evidence · {source_count_label(n)}{src_html}"


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
            f'<p class="best-panel">best match: '
            f'<span class="src">{escape(src)}</span>'
            f" · cosine {dense} of {thr}</p>"
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
        parts.append(empty_slot_html("ev", f"no sources in {mode_label(mode)}"))
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


def waiting_html(mode, after_label):
    return (
        '<div class="card waiting-card">'
        + card_head_html(mode)
        + f'<div class="slot-empty body">waiting · runs after {escape(after_label)} finishes</div>'
        + empty_slot_html("ev", "no sources yet")
        + '<div class="retr off">no retrieval yet</div>'
        + empty_slot_html("best")
        + '<div class="metrics slot-empty foot"></div>'
        + "</div>"
    )


def user_turn_html(question):
    return (
        f'<div class="you"><span class="who">you ›</span> {escape(question)}</div>'
    )


def compare_compact_html(left, right):
    def block(rec):
        mode = rec.get("mode") or ""
        preview = first_line(rec.get("answer") or "")
        return (
            "<div>"
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
