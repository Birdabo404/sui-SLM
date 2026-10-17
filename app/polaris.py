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


def _env_flag(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no")


# Flip a key to hide one piece without rewriting the page.
POLARIS = {
    "enabled": _env_flag("POLARIS_ENABLED", True),
    "header_band": True,
    "empty_state": True,
    "mode_pills": True,
    "answer_card": True,
    "reasoning": True,
    "retrieval_bar": True,
    "source_chips": True,
    "idk_card": True,
    "evidence_panel": True,
    "run_records": True,
    "export": True,
    "read_only_sidebar": True,
    "compare": True,
    "daylight": True,
    "model_digest": True,
}


def feature(name):
    if name == "enabled":
        return bool(POLARIS.get("enabled"))
    return bool(POLARIS.get("enabled")) and bool(POLARIS.get(name, False))


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
    docs = [h["title"] or h["path"] or h["source"] for h in scored if h["title"] or h["path"]]
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


def run_compare_pair(question, left_mode, right_mode, answer_fn, **kwargs):
    """Run left then right through answer_fn. Never overlapping."""
    left_mode, right_mode = pair_compare_modes(left_mode, right_mode)
    left = answer_fn(question, left_mode, **kwargs)
    right = answer_fn(question, right_mode, **kwargs)
    return left, right


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


def best_match_label(hits):
    if not hits:
        return ""
    ranked = sorted(
        hits,
        key=lambda h: (_opt_float(h.get("dense")) is not None, _opt_float(h.get("dense")) or 0.0),
        reverse=True,
    )
    top = ranked[0]
    key = source_key(top.get("source"))
    name = chip_name(top)
    if name:
        return f"{key.lower()} {name}"
    return key.lower()


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


def header_band_html(slim=False):
    klass = "band slim" if slim else "band"
    return (
        f'<div class="{klass}">'
        f'<span class="mark">polaris{STAR_SVG}</span>'
        f'<span class="tag">offline help for the linux lab</span>'
        f'<span class="spacer"></span>'
        f'<span class="pill-offline">offline</span>'
        f"</div>"
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


def retrieval_bar_html(score, threshold, retrieval_off=False):
    if retrieval_off:
        return (
            '<div class="retr off">'
            '<span class="lbl">retrieval</span> off'
            "</div>"
        )
    n = 0.0 if score is None else max(0.0, min(1.0, float(score)))
    t = 0.30 if threshold is None else float(threshold)
    ok = score is not None and float(score) >= t
    klass = "retr ok" if ok else "retr low"
    width = f"{n * 100:.1f}%"
    tick = f"{max(0.0, min(1.0, t)) * 100:.1f}%"
    return (
        f'<div class="{klass}">'
        f'<span class="lbl">retrieval</span> <b>{_fmt_score(score)}</b>'
        f" · threshold <b>{format_threshold(t)}</b>"
        f'<span class="bar"><span class="fill" style="width:{width}"></span>'
        f'<span class="tick" style="left:{tick}"></span></span>'
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
    n_docs = len(record.get("docs") or [])
    if not uses_rag(record.get("mode") or ""):
        n_docs = 0
    bits = [
        f'<b>{record["latency_s"]:.1f}</b> s',
        f'app <b>{record["app_rss_mb"]:.0f}</b>',
        f'ollama <b>{record["ollama_rss_mb"]:.0f}</b>',
        f'total <b>{record["total_mb"]:.0f}</b>',
        f'docs <b>{n_docs}</b>',
    ]
    if record.get("abstained"):
        bits.append("abstain")
    if "`" in (record.get("answer") or "") and not record.get("abstained"):
        bits.append('<span class="copy-hint">| copy</span>')
    return f'<div class="metrics">{" · ".join(bits)}</div>'


def idk_card_html(record):
    hits = record.get("hits") or []
    best = best_match_label(hits)
    best_line = f'<p class="best">best match: {escape(best)}</p>' if best else ""
    body = escape(record.get("answer") or "").replace("\n", "<br>")
    return (
        '<div class="idk">'
        f"{STAR_SVG}"
        "<h4>no fix on this one</h4>"
        f"<p>{body}</p>"
        + retrieval_bar_html(record.get("retrieval_max_dense"), record.get("threshold"))
        + best_line
        + "</div>"
    )


def evidence_rows_html(hits):
    if not hits:
        return '<p class="no-sources">no hits</p>'
    parts = ['<ol class="ev">']
    for hit in hits:
        dense = _fmt_score(hit.get("dense"))
        rrf = _fmt_score(hit.get("rrf"), 4)
        bm25 = _fmt_score(hit.get("bm25"), 3)
        parts.append(
            "<li>"
            f'<span class="rank">{int(hit["rank"])}</span>'
            f'<span class="src">{escape(str(hit.get("source") or ""))}</span>'
            f'<span class="title">{escape(str(hit.get("title") or ""))}</span>'
            f'<span class="path">{escape(str(hit.get("path") or ""))}</span>'
            f'<span class="sc">cos {dense} · rrf {rrf} · bm25 {bm25}</span>'
            f'<p class="snip">{escape(str(hit.get("snippet") or ""))}</p>'
            "</li>"
        )
    parts.append("</ol>")
    return "".join(parts)


def waiting_html(mode, after_label):
    return (
        card_head_html(mode)
        + f'<div class="waiting">waiting · runs after {escape(after_label)} finishes</div>'
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
