#!/usr/bin/env python3
"""Offline Streamlit chat app for the CCIS Linux lab assistant."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

from app.engine import (
    BASE_MODEL,
    DEFAULT_INDEX,
    DISTILLED_MODEL,
    MODE_LABELS,
    OLLAMA_HOST,
    answer_question,
    config_id,
    git_commit,
    index_ok,
    model_digest,
    model_name_for,
    mode_choices,
    ollama_reachable,
    uses_rag,
)
from app.polaris import (
    EXAMPLES,
    avg_latency,
    build_run_record,
    card_head_html,
    compare_compact_html,
    default_compare_modes,
    empty_hero_html,
    evidence_rows_html,
    feature,
    header_band_html,
    idk_card_html,
    index_chunk_count,
    metrics_html,
    pair_compare_modes,
    records_to_csv,
    records_to_json,
    retrieval_bar_html,
    sidebar_kv_html,
    source_chips_html,
    split_code_blocks,
    split_reasoning,
    style_path,
    user_turn_html,
    waiting_html,
    warm_label,
    yn,
)
from shared.frozen import format_threshold, load_frozen


@st.cache_resource
def load_retriever(index_dir):
    from rag.query_hybrid import HybridRetriever

    return HybridRetriever(index_dir)


@st.cache_data(ttl=15)
def _ollama_up():
    return ollama_reachable()


@st.cache_data(ttl=30)
def _model_digest(model):
    if not feature("model_digest"):
        return ""
    return model_digest(model)


@st.cache_data(ttl=60)
def _chunk_count(index_dir):
    return index_chunk_count(index_dir)


def main():
    if feature("enabled"):
        st.set_page_config(page_title="Polaris", layout="centered")
        run_polaris()
        return
    st.set_page_config(page_title="CCIS Linux lab assistant", layout="centered")
    run_legacy()


def run_legacy():
    st.title("CCIS Linux lab assistant")
    st.caption("Runs offline. Docs come from the local index; the model is local Ollama.")

    with st.sidebar:
        st.header("Settings")
        model_key = st.radio("Model", ["base", "distilled"], index=0)
        base_model = st.text_input("Base model", value=BASE_MODEL)
        distilled_model = st.text_input(
            "Distilled model",
            value=DISTILLED_MODEL,
            help="Ollama model name for the distilled GGUF. Leave empty until it is registered.",
        )
        mode_label = st.radio("Mode", list(MODE_LABELS.values()), index=3)
        mode = next(k for k, v in MODE_LABELS.items() if v == mode_label)
        index_dir = st.text_input("RAG index", value=DEFAULT_INDEX)
        st.caption(f"Config: {config_id(model_key, mode)}")

    model = model_name_for(model_key, base_model=base_model, distilled_model=distilled_model)
    if model_key == "distilled" and not model:
        st.warning("Set the distilled model name (an Ollama model) before using that config.")

    retriever = None
    if uses_rag(mode):
        # Load MiniLM/Chroma only when a RAG mode is selected (8G mem cap).
        if not index_ok(index_dir):
            st.warning(
                f"No index at {index_dir}. Fetch the corpus and run rag/build_index.py first."
            )
        else:
            retriever = load_retriever(index_dir)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _show_sources(msg.get("sources") or [])
            if msg.get("stats"):
                st.caption(msg["stats"])

    question = st.chat_input("Ask a Linux or Ubuntu lab question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if model_key == "distilled" and not model:
            text = "The distilled model name is empty."
            hits, weak, stats = [], False, ""
        elif uses_rag(mode) and retriever is None:
            text = "The RAG index is missing, so I cannot answer in this mode."
            hits, weak, stats = [], False, ""
        else:
            try:
                result = answer_question(
                    question, mode, model, retriever=retriever, model_key=model_key
                )
                text = result["text"]
                hits = result["hits"]
                weak = result["weak"]
                stats = (
                    f"{result['latency_s']:.2f} s, "
                    f"app {result['app_rss_mb']:.0f} MB, "
                    f"ollama {result['ollama_rss_mb']:.0f} MB, "
                    f"total {result['total_mb']:.0f} MB"
                    + (" (weak retrieval, abstained)" if weak else "")
                )
            except requests.RequestException as e:
                text = (
                    f"Could not reach Ollama at {OLLAMA_HOST}. "
                    f"Start it with `ollama serve`.\n\n{e}"
                )
                hits, weak, stats = [], False, ""
        st.markdown(text)
        sources = hits if uses_rag(mode) else []
        _show_sources(sources)
        if stats:
            st.caption(stats)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "sources": sources if uses_rag(mode) else [],
            "stats": stats,
        }
    )


def _show_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for i, s in enumerate(sources, 1):
            score = f", rrf {s['rrf']:.4f}" if "rrf" in s else ""
            st.markdown(f"**[{i}] {s['title']}** ({s['source']}{score})")
            st.caption(s.get("path", ""))
            st.text(s["text"][:800])


def _inject_css():
    css = style_path().read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _init_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("run_records", [])
    st.session_state.setdefault("warmed", False)
    if "cmp_left" not in st.session_state or "cmp_right" not in st.session_state:
        left, right = default_compare_modes()
        st.session_state.setdefault("cmp_left", left)
        st.session_state.setdefault("cmp_right", right)


def run_polaris():
    _inject_css()
    _init_session()
    index_dir = DEFAULT_INDEX
    model = BASE_MODEL
    frozen = load_frozen()

    _render_sidebar(model, index_dir, frozen)

    if feature("daylight") and st.session_state.get("daylight"):
        st.markdown('<div class="polaris-daylight"></div>', unsafe_allow_html=True)
    compare_on = bool(feature("compare") and st.session_state.get("compare"))
    if compare_on:
        st.markdown('<div class="polaris-compare"></div>', unsafe_allow_html=True)

    slim = bool(st.session_state.messages)
    if feature("header_band"):
        st.markdown(header_band_html(slim=slim), unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        _replay_message(msg, i, compact_compare=True)

    empty = not st.session_state.messages
    if empty and feature("empty_state"):
        st.markdown(empty_hero_html(), unsafe_allow_html=True)

    mode = "full"
    left_mode, right_mode = default_compare_modes()
    if compare_on:
        left_mode, right_mode = pair_compare_modes(
            st.session_state.get("cmp_left"),
            st.session_state.get("cmp_right"),
        )
    elif feature("mode_pills"):
        mode = _render_mode_pills()

    if empty and feature("empty_state"):
        _render_examples()
        with st.container(key="composer-empty"):
            typed = st.chat_input("ask a linux or lab question…")
    else:
        typed = st.chat_input("ask a linux or lab question…")

    question = st.session_state.pop("pending_q", None) or typed
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(user_turn_html(question), unsafe_allow_html=True)

    if compare_on:
        need_rag = uses_rag(left_mode) or uses_rag(right_mode)
    else:
        need_rag = uses_rag(mode)
    retriever = None
    if need_rag:
        if not index_ok(index_dir):
            st.error(
                f"No index at {index_dir}. Fetch the corpus and run rag/build_index.py first."
            )
            return
        retriever = load_retriever(index_dir)

    if compare_on:
        recs = _run_compare(question, left_mode, right_mode, model, retriever, index_dir)
        if recs[0] is not None or recs[1] is not None:
            st.session_state.messages.append({"role": "assistant", "compare": recs})
        return

    rec = _answer_record(question, mode, model, retriever, index_dir)
    if rec is None:
        return
    _render_card(rec, key=f"answer-{len(st.session_state.run_records)}")
    st.session_state.messages.append({"role": "assistant", "record": rec})


def _render_sidebar(model, index_dir, frozen):
    with st.sidebar:
        if feature("read_only_sidebar"):
            digest = _model_digest(model) if feature("model_digest") else ""
            model_value = model if not digest else f"{model} · {digest}"
            loaded = index_ok(index_dir)
            chunks = _chunk_count(index_dir) if loaded else 0
            index_value = Path(index_dir).name
            if loaded:
                index_value = f"{index_value} · {chunks} chunks"
            sha = git_commit()
            st.markdown('<div class="chrome">STATUS</div>', unsafe_allow_html=True)
            st.markdown(
                sidebar_kv_html(
                    [
                        ("model", model_value),
                        ("index", index_value),
                        ("threshold", f"cos {format_threshold(frozen['threshold'])}"),
                        ("k", str(frozen["k"])),
                        ("git", sha[:7] if sha else "—"),
                        ("ollama", yn(_ollama_up())),
                        ("index loaded", yn(loaded)),
                        ("warm-up", warm_label(st.session_state.get("warmed"))),
                    ]
                ),
                unsafe_allow_html=True,
            )
        else:
            st.header("Settings")

        st.markdown('<div class="chrome">VIEW</div>', unsafe_allow_html=True)
        if feature("compare"):
            st.toggle("Compare modes", key="compare")
        if feature("daylight"):
            st.toggle("Daylight", key="daylight")

        if feature("compare") and st.session_state.get("compare"):
            st.markdown('<div class="chrome">COMPARE</div>', unsafe_allow_html=True)
            modes = mode_choices()
            st.selectbox(
                "Left",
                modes,
                format_func=lambda m: MODE_LABELS.get(m, m),
                key="cmp_left",
            )
            right_opts = [m for m in modes if m != st.session_state.get("cmp_left")]
            current_right = st.session_state.get("cmp_right")
            if current_right not in right_opts and right_opts:
                st.session_state.cmp_right = right_opts[0]
            st.selectbox(
                "Right",
                right_opts,
                format_func=lambda m: MODE_LABELS.get(m, m),
                key="cmp_right",
            )

        st.markdown('<div class="chrome">IDK THRESHOLD</div>', unsafe_allow_html=True)
        st.caption(f"cos {format_threshold(frozen['threshold'])} (frozen)")

        records = st.session_state.get("run_records") or []
        st.markdown('<div class="chrome">SESSION</div>', unsafe_allow_html=True)
        avg = avg_latency(records)
        st.caption("avg —" if avg is None else f"avg {avg:.1f} s · {len(records)} runs")
        if feature("export") and feature("run_records") and records:
            st.download_button(
                "CSV",
                data=records_to_csv(records),
                file_name="polaris-runs.csv",
                mime="text/csv",
                key="export-csv",
            )
            st.download_button(
                "JSON",
                data=records_to_json(records),
                file_name="polaris-runs.json",
                mime="application/json",
                key="export-json",
            )

        if st.button("Clear chat", key="clear-chat"):
            st.session_state.messages = []
            st.session_state.run_records = []
            st.session_state.warmed = False
            st.rerun()

        st.markdown(
            '<p class="footer-license">Inter and JetBrains Mono — SIL OFL 1.1</p>',
            unsafe_allow_html=True,
        )


def _render_mode_pills():
    modes = mode_choices()
    labels = [MODE_LABELS[m] for m in modes]
    default = MODE_LABELS.get("full", labels[-1])
    with st.container(key="mode"):
        picked = st.segmented_control(
            "Mode",
            labels,
            default=default,
            key="mode-ctl",
            label_visibility="collapsed",
        )
    if picked is None:
        picked = default
    return next(m for m, lab in MODE_LABELS.items() if lab == picked)


def _render_examples():
    with st.container(key="examples"):
        cols = st.columns(2)
        for i, text in enumerate(EXAMPLES):
            if cols[i % 2].button(text, key=f"ex-{i}"):
                st.session_state.pending_q = text
                st.rerun()


def _answer_record(question, mode, model, retriever, index_dir):
    try:
        result = answer_question(
            question, mode, model, retriever=retriever, model_key="base"
        )
    except requests.RequestException as e:
        st.error(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            f"Start it with `ollama serve`.\n\n{e}"
        )
        return None
    rec = build_run_record(result, index_path=index_dir)
    if feature("run_records"):
        st.session_state.run_records.append(rec)
    st.session_state.warmed = True
    return rec


def _run_compare(question, left_mode, right_mode, model, retriever, index_dir):
    left_mode, right_mode = pair_compare_modes(left_mode, right_mode)
    rec_l = rec_r = None
    with st.container(key="cmp"):
        col_l, col_r = st.columns(2, gap="medium")
        wait = col_r.empty()
        wait.markdown(
            waiting_html(right_mode, MODE_LABELS.get(left_mode, left_mode)),
            unsafe_allow_html=True,
        )
        with col_l:
            rec_l = _answer_record(question, left_mode, model, retriever, index_dir)
            if rec_l is not None:
                _render_card(rec_l, key="answer-cmp-left")
        rec_r = _answer_record(question, right_mode, model, retriever, index_dir)
        wait.empty()
        with col_r:
            if rec_r is not None:
                _render_card(rec_r, key="answer-cmp-right")
    return rec_l, rec_r


def _replay_message(msg, index, compact_compare=False):
    role = msg.get("role")
    if role == "user":
        st.markdown(user_turn_html(msg.get("content") or ""), unsafe_allow_html=True)
        return
    if msg.get("compare"):
        left, right = msg["compare"]
        if compact_compare and left and right:
            st.markdown(compare_compact_html(left, right), unsafe_allow_html=True)
            return
        with st.container(key=f"cmp-hist-{index}"):
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                if left:
                    _render_card(left, key=f"answer-h{index}-l")
            with col_r:
                if right:
                    _render_card(right, key=f"answer-h{index}-r")
        return
    rec = msg.get("record")
    if rec:
        _render_card(rec, key=f"answer-{index}")
        return
    st.markdown(msg.get("content") or "")


def _render_card(record, key):
    if not feature("answer_card"):
        st.markdown(record.get("answer") or "")
        return
    mode = record.get("mode") or ""
    rag = uses_rag(mode)
    with st.container(key=key):
        st.markdown(card_head_html(mode), unsafe_allow_html=True)
        if record.get("abstained") and feature("idk_card"):
            st.markdown(idk_card_html(record), unsafe_allow_html=True)
            _render_evidence(record)
        else:
            if feature("reasoning") and mode in ("cot", "full"):
                reason, answer = split_reasoning(record.get("answer") or "")
                if reason:
                    with st.expander("REASONING", expanded=False):
                        st.markdown(reason)
                    _render_answer_body(answer)
                else:
                    _render_answer_body(record.get("answer") or "")
            else:
                _render_answer_body(record.get("answer") or "")
            if feature("retrieval_bar"):
                st.markdown(
                    retrieval_bar_html(
                        record.get("retrieval_max_dense"),
                        record.get("threshold"),
                        retrieval_off=not rag,
                    ),
                    unsafe_allow_html=True,
                )
            if feature("source_chips"):
                raw_hits = record.get("hits") or []
                st.markdown(
                    source_chips_html(raw_hits, retrieval_off=not rag),
                    unsafe_allow_html=True,
                )
            if rag:
                _render_evidence(record)
        st.markdown(metrics_html(record), unsafe_allow_html=True)


def _render_answer_body(text):
    chunks = list(split_code_blocks(text))
    if not chunks:
        st.markdown(text or "")
        return
    for kind, chunk in chunks:
        if kind == "code":
            st.code(chunk, language="bash")
        elif chunk.strip():
            st.markdown(chunk)


def _render_evidence(record):
    if not feature("evidence_panel"):
        return
    hits = record.get("hits") or []
    with st.expander("Evidence", expanded=False):
        st.markdown(evidence_rows_html(hits), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
