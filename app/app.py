#!/usr/bin/env python3
"""Offline Streamlit chat app for the CCIS Linux lab assistant."""
import os
import queue
import sqlite3
import threading
import time
from functools import partial
from html import escape
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import requests
import streamlit as st

from app.answer_format import app_messages, parse_answer, safe_markdown
from app.chat_memory import ChatStore, build_chat_context
from app.engine import (
    BASE_MODEL,
    DEFAULT_INDEX,
    DISTILLED_MODEL,
    MODE_LABELS,
    NUM_CTX,
    OLLAMA_HOST,
    answer_question,
    config_id,
    git_commit,
    index_ok,
    keep_alive_setting,
    model_name_for,
    ollama_info,
    ollama_reachable,
    uses_rag,
    warm_model,
)
from app.models import (
    compare_picks,
    installed_names,
    is_installed,
    missing_text,
    model_options,
    modes_for,
    option_named,
    pair_compare_ids,
    pick_by_id,
)
from app.polaris import (
    DELETE_ASK,
    EXAMPLES,
    HISTORY_EMPTY,
    MONITOR_SWITCH,
    RAM_TITLE,
    RATE_NOTE,
    REASONING_LABEL,
    REASONING_NOTE,
    SCORES_NOTE,
    RateTrace,
    avg_latency,
    best_match_html,
    build_run_record,
    cancelled_side_html,
    card_head_html,
    compare_compact_html,
    compare_model_tags,
    compare_summary_html,
    composer_note_html,
    config_summary_html,
    connection_html,
    empty_hero_html,
    empty_slot_html,
    evidence_block_html,
    flags_html,
    header_band_html,
    history_error_text,
    history_note_html,
    hits_from_chart,
    idk_card_html,
    index_chunk_count,
    injected_css,
    last_runs,
    live_phase_label,
    metrics_html,
    mode_hint,
    mode_name,
    model_choice_label,
    model_status_html,
    models_table_html,
    monitor_html,
    network_online,
    plain_label,
    polaris_on,
    rail_art_html,
    records_in,
    records_to_csv,
    records_to_json,
    side_logo_html,
    sky_html,
    sky_level,
    sidebar_kv_html,
    loose_commands_html,
    source_line_html,
    stopped_html,
    thinking_html,
    thinking_label,
    user_turn_html,
    waiting_html,
    warm_label,
)
from app.streaming import ChatCancelled, stream_answer_question
from shared.frozen import format_threshold, load_frozen


@st.cache_resource(show_spinner=False)
def load_retriever(index_dir):
    from rag.query_hybrid import HybridRetriever

    return HybridRetriever(index_dir)


@st.cache_data(ttl=15, show_spinner=False)
def _ollama_up():
    return ollama_reachable()


@st.cache_data(ttl=30, show_spinner=False)
def _model_info(model):
    return ollama_info(model)


@st.cache_data(ttl=10, show_spinner=False)
def _network_online():
    return network_online()


@st.cache_data(ttl=60, show_spinner=False)
def _chunk_count(index_dir):
    return index_chunk_count(index_dir)


@st.cache_resource(show_spinner=False)
def _warmed_models():
    return set()


def _warm_model_at_startup(model, keep_alive):
    warmed = _warmed_models()
    if model in warmed:
        return True
    if not warm_model(model, keep_alive=keep_alive):
        return False
    warmed.add(model)
    return True


def _session_git_commit():
    if "git_sha" not in st.session_state:
        st.session_state["git_sha"] = git_commit()
    return st.session_state["git_sha"]


def main():
    if polaris_on():
        st.set_page_config(page_title="Polaris", layout="centered")
        run_polaris()
        return
    st.set_page_config(page_title="CCIS Linux lab assistant", layout="centered")
    run_legacy()


def run_legacy():
    st.title("CCIS Linux lab assistant")
    st.caption("Runs offline. Docs come from the local index; the model is local Ollama.")
    session_git_sha = _session_git_commit()

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

    context = build_chat_context(st.session_state.messages, question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    failed = False
    with st.chat_message("assistant"):
        if model_key == "distilled" and not model:
            text = "The distilled model name is empty."
            hits, weak, stats = [], False, ""
            failed = True
        elif uses_rag(mode) and retriever is None:
            text = "The RAG index is missing, so I cannot answer in this mode."
            hits, weak, stats = [], False, ""
            failed = True
        else:
            try:
                result = answer_question(
                    question,
                    mode,
                    model,
                    retriever=retriever,
                    model_key=model_key,
                    keep_alive=keep_alive_setting(),
                    git_sha=session_git_sha,
                    model_info=_model_info(model),
                    context=context,
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
                failed = True
        st.markdown(text)
        sources = hits if uses_rag(mode) else []
        _show_sources(sources)
        if stats:
            st.caption(stats)

    message = {
        "role": "assistant",
        "content": text,
        "sources": sources if uses_rag(mode) else [],
        "stats": stats,
    }
    if failed:
        message["error"] = text
    st.session_state.messages.append(message)


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
    st.markdown(f"<style>{injected_css()}</style>", unsafe_allow_html=True)


# Tabs open at once; older ones stay one click away in History.
MAX_TABS = 6


def _init_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("run_records", [])
    st.session_state.setdefault("warmed", False)
    st.session_state.setdefault("mode", "full")
    st.session_state.setdefault("compare", False)
    st.session_state.setdefault("model_pick", BASE_MODEL)
    st.session_state.setdefault("chat", _chat_state())
    _session_git_commit()
    if "tabs" not in st.session_state:
        st.session_state["tabs"] = _recent_chat_ids()


@st.cache_resource
def _loaded_indexes():
    return set()


@st.cache_data(ttl=15, show_spinner=False)
def _installed_names():
    return installed_names(OLLAMA_HOST)


def _still_missing(option, names):
    """The not-installed message for option, or None. A cached "missing" is checked once more against Ollama, so a
    model created a moment ago is picked up without waiting for the cache."""
    if is_installed(option.name, names) is not False:
        return None
    _installed_names.clear()
    if is_installed(option.name, _installed_names()) is not False:
        return None
    return missing_text(option)


def _compare_pair(options):
    """Both compare sides as picks, made valid together and written back to the cmp_*_pick keys."""
    configured = bool(DISTILLED_MODEL.strip())
    left, right = pair_compare_ids(
        st.session_state.get("cmp_left_pick"), st.session_state.get("cmp_right_pick"), options, configured
    )
    st.session_state.cmp_left_pick, st.session_state.cmp_right_pick = left, right
    return pick_by_id(options, left), pick_by_id(options, right)


def run_polaris():
    _inject_css()
    _init_session()
    index_dir = DEFAULT_INDEX
    frozen = load_frozen()
    up = _ollama_up()
    names = _installed_names() if up else None
    options = model_options(BASE_MODEL, DISTILLED_MODEL)
    picked = option_named(options, st.session_state.model_pick)
    modes = modes_for(picked)
    mode = st.session_state.mode if st.session_state.mode in modes else modes[-1]
    if up and is_installed(picked.name, names) is not False:
        st.session_state.warmed = _warm_model_at_startup(picked.name, keep_alive_setting())
    else:
        st.session_state.warmed = False
        if not up:
            _warmed_models().discard(picked.name)
    online = _network_online()
    # A question from the landing (typed or an example chip) is answered in the run it arrives in. The layout is
    # chosen here, before anything renders, so this run builds the thread exactly where every later run does.
    landing_q = st.session_state.get("pending_q") or st.session_state.get("chat-landing")
    incoming = landing_q or st.session_state.get("chat-dock")
    msgs = st.session_state.messages
    chat = st.session_state.chat
    if incoming and chat["id"] is None:
        # Stored with its question before the tabs draw, so a new chat is listed as current in this same run.
        _sync_chat(chat, [*msgs, {"role": "user", "content": incoming}])
    compare_on = bool(st.session_state.get("compare"))
    left, right = _compare_pair(options)
    asked = sum(1 for m in msgs if m.get("role") == "user") + (1 if incoming else 0)

    flags = st.empty()

    def mark(generating):
        flags.markdown(
            flags_html(
                sky_level(asked),
                daylight=bool(st.session_state.get("daylight")),
                compare=compare_on,
                generating=generating,
            ),
            unsafe_allow_html=True,
        )

    # Before the rail, so the star's first wrap starts out paused while this run generates.
    mark(generating=bool(incoming))
    status = dict(options=options, picked=picked, mode=mode, names=names, up=up, online=online,
                  index_dir=index_dir, frozen=frozen)
    rail = _render_rail(status, compare_on, left, right, busy=bool(incoming), chat=bool(msgs or incoming))
    st.markdown(sky_html(), unsafe_allow_html=True)
    _render_tabs()

    if compare_on:
        summary = compare_summary_html(left.option.name, left.mode, right.option.name, right.mode)
    else:
        summary = config_summary_html(picked.name, mode)
    composer = dict(summary=summary, conn=connection_html(online, up))

    if not (msgs or incoming):
        with st.container(key="landing"):
            st.markdown(empty_hero_html(), unsafe_allow_html=True)
            _render_composer("landing", **composer)
            _render_examples()
        return

    st.markdown(header_band_html(slim=True), unsafe_allow_html=True)
    thread = st.container(key="thread")
    with thread:
        for i, msg in enumerate(msgs):
            _replay_message(msg, i, compact_compare=True)
            if msg.get("role") == "user" and (i + 1 == len(msgs) or msgs[i + 1].get("role") == "user"):
                st.markdown(stopped_html(), unsafe_allow_html=True)
    typed, stop_slot = _render_composer("dock", generating=bool(incoming), **composer)
    st.session_state.pop("pending_q", None)
    question = landing_q or typed
    if not question:
        if incoming:
            rail.refresh()
        return

    context = build_chat_context(msgs, question)
    msgs.append({"role": "user", "content": question})
    try:
        with thread:
            _answer_turn(
                question,
                picked,
                mode,
                compare_on,
                left,
                right,
                index_dir,
                monitor=rail.monitor,
                names=names,
                context=context,
            )
    finally:
        # Stored before the next st call: a click can end this run at any st call, and reading session state counts
        # as one. A stopped or failed turn still keeps its question.
        _sync_chat(chat, msgs)
        # A turn that raises still takes the Stop buttons away and lets the sky move again.
        stop_slot.empty()
        rail.foot.empty()
        mark(generating=False)
    rail.refresh()


def _answer_turn(question, option, mode, compare_on, left, right, index_dir, *, monitor, names, context=None):
    st.session_state.pop("last_cancelled", None)
    st.markdown(user_turn_html(question), unsafe_allow_html=True)
    slot = st.empty()

    if not compare_on:
        missing = _still_missing(option, names)
        if missing:
            slot.error(missing)
            st.session_state.messages.append({"role": "assistant", "error": missing})
            return False

    need_rag = (uses_rag(left.mode) or uses_rag(right.mode)) if compare_on else uses_rag(mode)
    retriever = None
    if need_rag:
        if not index_ok(index_dir):
            message = f"No index at {index_dir}. Fetch the corpus and run rag/build_index.py first."
            slot.error(message)
            st.session_state.messages.append({"role": "assistant", "error": message})
            return False
        retriever = _retriever_with_loader(index_dir, slot)

    if compare_on:
        slot.empty()
        recs, _ = _run_compare(question, left, right, retriever, index_dir, monitor=monitor, names=names,
                               context=context)
        return any(recs)

    key = f"answer-{len(st.session_state.messages)}"
    monitor.plan([("", option.name, mode)])
    rec, view = _answer_record(
        question,
        mode,
        option,
        retriever,
        index_dir,
        slot,
        key=key,
        monitor=monitor,
        names=names,
        context=context,
    )
    if rec is None:
        if st.session_state.pop("last_cancelled", False):
            return False
        st.session_state.messages.append({"role": "assistant", "error": st.session_state.pop("last_error", "")})
        return False
    st.session_state.messages.append({"role": "assistant", "record": rec})
    view.finish(rec)
    return True


def _retriever_with_loader(index_dir, slot):
    loaded = _loaded_indexes()
    if index_dir in loaded:
        return load_retriever(index_dir)
    slot.markdown(thinking_html(thinking_label("", loading_index=True)), unsafe_allow_html=True)
    retriever = load_retriever(index_dir)
    loaded.add(index_dir)
    return retriever


def _chat_state(chat_id=None, saved=0):
    """The open chat's history-store id (None until its first question, so an empty chat is never stored), how many
    of its messages the store has, and the last history error. Changed in place, so a turn's end can record a save
    without an st call."""
    return {"id": chat_id, "saved": saved, "error": None}


def _recent_chat_ids():
    """The most recent chats, oldest first, as the tabs a new session opens with."""
    try:
        chats = ChatStore().list_chats()
    except (sqlite3.Error, OSError):
        return []
    return [c.chat_id for c in chats[: MAX_TABS - 1]][::-1]


def _new_chat():
    st.session_state.messages = []
    st.session_state.run_records = []
    st.session_state.chat = _chat_state()
    st.session_state.pop("hist_confirm", None)
    st.session_state.pop("live_traces", None)


def _sync_chat(chat, messages):
    """Store the messages the history store doesn't have yet. Makes no st call, so a click can't interrupt it.

    Only new messages are appended, so another tab adding turns to the same chat keeps them. A chat deleted
    elsewhere since the last save is stored again as a new chat with the whole thread.
    """
    if not messages or (chat["id"] is not None and chat["saved"] >= len(messages)):
        return
    store = ChatStore()
    try:
        if chat["id"] is not None:
            try:
                store.append_messages(chat["id"], messages[chat["saved"]:])
            except KeyError:
                chat.update(id=None, saved=0)
            else:
                chat.update(saved=len(messages), error=None)
                return
        chat["id"] = store.new_chat().chat_id
        store.save_chat(chat["id"], messages)
    except (sqlite3.Error, OSError):
        chat["error"] = history_error_text(store.root)
        return
    chat.update(saved=len(messages), error=None)


def _open_chat(chat_id):
    store = ChatStore()
    st.session_state.pop("hist_confirm", None)
    try:
        stored = store.load_chat(chat_id)
    except KeyError:
        return
    except (sqlite3.Error, OSError, ValueError):
        st.session_state.chat["error"] = history_error_text(store.root)
        return
    st.session_state.messages = list(stored.messages)
    st.session_state.run_records = records_in(st.session_state.messages)
    st.session_state.chat = _chat_state(stored.chat_id, len(stored.messages))
    st.session_state.pop("pending_q", None)
    st.session_state.pop("live_traces", None)
    if chat_id not in st.session_state.tabs:
        st.session_state.tabs.append(chat_id)


def _close_tab(chat_id):
    """Close a tab; the chat stays in History. Closing the open tab moves to its neighbour, or to a new chat."""
    tabs = st.session_state.tabs
    st.session_state.pop("hist_confirm", None)
    if chat_id is None:
        if tabs:
            _open_chat(tabs[-1])
        return
    at = tabs.index(chat_id) if chat_id in tabs else len(tabs)
    if chat_id in tabs:
        tabs.remove(chat_id)
    if chat_id != st.session_state.chat["id"]:
        return
    if tabs:
        _open_chat(tabs[min(at, len(tabs) - 1)])
    else:
        _new_chat()


def _ask_delete(chat_id):
    st.session_state.hist_confirm = chat_id


def _cancel_delete():
    st.session_state.pop("hist_confirm", None)


def _delete_chat(chat_id):
    store = ChatStore()
    st.session_state.pop("hist_confirm", None)
    try:
        store.delete_chat(chat_id)
    except (sqlite3.Error, OSError):
        st.session_state.chat["error"] = history_error_text(store.root)
        return
    if chat_id in st.session_state.tabs:
        st.session_state.tabs.remove(chat_id)
    if chat_id == st.session_state.chat["id"]:
        _new_chat()


def _visible_tabs(chats, current):
    """Open tabs that still exist (another browser tab may have deleted one), the open chat always among them,
    capped at MAX_TABS by dropping the oldest."""
    known = {c.chat_id for c in chats}
    tabs = [cid for cid in st.session_state.tabs if cid in known or cid == current]
    if current is not None and current not in tabs:
        tabs.append(current)
    room = MAX_TABS - (1 if current is None else 0)
    while len(tabs) > room:
        tabs.remove(next(cid for cid in tabs if cid != current))
    st.session_state.tabs = tabs
    return tabs


def _render_tabs():
    """Browser-style tabs over the thread: one per open chat, a draft tab for a new one, History and New on the
    right. Only the open chat's thread is ever drawn."""
    store = ChatStore()
    try:
        chats = store.list_chats()
        failed = False
    except (sqlite3.Error, OSError):
        chats, failed = [], True
    current = st.session_state.chat["id"]
    titles = {c.chat_id: c.title for c in chats}
    if current is not None and current not in titles:
        titles[current] = next((m.get("content") for m in st.session_state.messages if m.get("role") == "user"), "")
    tabs = _visible_tabs(chats, current)
    with st.container(key="tabbar"):
        with st.container(key="tabs"):
            for cid in tabs:
                _tab(cid, titles.get(cid) or "Untitled", active=cid == current)
            if current is None:
                _tab(None, "New chat", active=True, closable=bool(tabs))
        with st.container(key="tab-history"):
            with st.popover("History", icon=":material/history:", help=f"All saved chats ({len(chats)})"):
                _history_list(chats, store, failed)
        st.button("New chat", key="new-chat", icon=":material/add:", help="New chat", on_click=_new_chat)


def _tab(chat_id, title, active, closable=True):
    name = "draft" if chat_id is None else chat_id
    label = plain_label(title)
    with st.container(key=f"tab-{'on' if active else 'off'}-{name}"):
        # The open tab is disabled: clicking it would only re-run, and that stops an answer mid-flight.
        st.button(label, key=f"tab-open-{name}", disabled=active, on_click=_open_chat, args=(chat_id,),
                  use_container_width=True)
        if closable:
            st.button(f"Close {label}", key=f"tab-close-{name}", icon=":material/close:", help="Close tab",
                      on_click=_close_tab, args=(chat_id,))


def _history_list(chats, store, failed):
    if failed:
        st.markdown(history_note_html(history_error_text(store.root), error=True), unsafe_allow_html=True)
        return
    error = st.session_state.chat["error"]
    if error:
        st.markdown(history_note_html(error, error=True), unsafe_allow_html=True)
    if not chats:
        if not error:
            st.markdown(history_note_html(HISTORY_EMPTY), unsafe_allow_html=True)
        return
    current = st.session_state.chat["id"]
    confirm = st.session_state.get("hist_confirm")
    with st.container(key="history"):
        for chat in chats:
            cid = chat.chat_id
            if cid == confirm:
                with st.container(key=f"hist-confirm-{cid}"):
                    st.markdown(history_note_html(DELETE_ASK), unsafe_allow_html=True)
                    st.button("Delete", key=f"hist-yes-{cid}", on_click=_delete_chat, args=(cid,))
                    st.button("Cancel", key=f"hist-no-{cid}", on_click=_cancel_delete)
                continue
            row = "hist-current" if cid == current else "hist-item"
            with st.container(key=f"{row}-{cid}"):
                st.button(
                    plain_label(chat.title),
                    key=f"hist-open-{cid}",
                    use_container_width=True,
                    disabled=cid == current,
                    on_click=_open_chat,
                    args=(cid,),
                )
                st.button(
                    "Delete",
                    key=f"hist-del-{cid}",
                    icon=":material/delete:",
                    on_click=_ask_delete,
                    args=(cid,),
                )


class _Rail:
    """The sidebar: run controls, the monitor and the footer. While an answer streams the controls are drawn
    disabled under their own keys, since any click reruns the script and would stop the answer; refresh() draws
    them again under the usual keys once it ends."""

    def __init__(self, status, compare_on, left, right, controls, foot, monitor):
        self.status, self.compare_on, self.left, self.right = status, compare_on, left, right
        self.controls, self.foot, self.monitor = controls, foot, monitor

    def draw(self, busy):
        suffix = "-busy" if busy else ""
        with self.controls.container(key=f"rail-run{suffix}"):
            _run_controls(self.status, self.compare_on, self.left, self.right, suffix, busy)
        with self.foot.container(key=f"rail-foot{suffix}"):
            if busy:
                # Settings can't open mid-answer (it would rerun and stop it), so Stop takes its place.
                st.button("Stop", key="stop-rail", icon=":material/stop:", help="Stop generating")
            elif st.button("Settings", key="settings", icon=":material/tune:"):
                _settings_dialog(self.status)
            if busy:
                st.toggle("Daylight", value=bool(st.session_state.get("daylight")), key="daylight-busy",
                          disabled=True)
            else:
                st.toggle("Daylight", key="daylight")

    def refresh(self):
        self.draw(busy=False)
        self.monitor.show_last()


def _render_rail(status, compare_on, left, right, busy, chat=False):
    with st.sidebar:
        with st.container(key="side-top"):
            st.markdown(side_logo_html(play=chat), unsafe_allow_html=True)
            st.markdown(rail_art_html(), unsafe_allow_html=True)
        controls = st.empty()
        with st.container(key="rail-mon"):
            st.markdown(
                '<div class="mon-top"><p class="side-h">Monitor</p>' + MONITOR_SWITCH + "</div>",
                unsafe_allow_html=True,
            )
            monitor = _Monitor(st.empty())
        foot = st.empty()
    rail = _Rail(status, compare_on, left, right, controls, foot, monitor)
    rail.draw(busy)
    monitor.show_last()
    return rail


def _sync_pick(key, target):
    picked = st.session_state.get(key)
    if picked:
        st.session_state[target] = picked


def _run_controls(status, compare_on, left, right, suffix, busy):
    options, names = status["options"], status["names"]
    st.markdown('<p class="side-h">Run</p>', unsafe_allow_html=True)
    if compare_on:
        _compare_pickers(options, names, left, right, suffix, busy)
    else:
        picked, mode = status["picked"], status["mode"]
        labels = [o.name for o in options]
        st.selectbox(
            "Model",
            labels,
            index=labels.index(picked.name),
            format_func=lambda name: model_choice_label(option_named(options, name), names),
            key=f"model-select{suffix}",
            on_change=_sync_pick,
            args=(f"model-select{suffix}", "model_pick"),
            disabled=busy,
        )
        st.markdown(model_status_html(picked, names, status["up"]), unsafe_allow_html=True)
        modes = modes_for(picked)
        st.radio(
            "Mode",
            modes,
            index=modes.index(mode),
            format_func=mode_name,
            key=f"mode-pick{suffix}",
            horizontal=True,
            on_change=_sync_pick,
            args=(f"mode-pick{suffix}", "mode"),
            disabled=busy,
        )
        st.markdown(f'<p class="side-note mode-hint">{escape(mode_hint(mode))}</p>', unsafe_allow_html=True)
    if busy:
        st.toggle("Compare two configs", value=compare_on, key="compare-busy", disabled=True)
    else:
        st.toggle("Compare two configs", key="compare")


def _compare_pickers(options, names, left, right, suffix, busy):
    """One picker per side over every model and config the machine can offer, written back to cmp_*_pick."""
    picks = {p.id: p for p in compare_picks(options)}

    def label(pick_id):
        pick = picks[pick_id]
        missing = is_installed(pick.option.name, names) is False
        return f"{pick.option.name} · {mode_name(pick.mode)}" + (" · not installed" if missing else "")

    # Pass index rather than seeding the key through session state: the browser shows a seeded value once, then
    # falls back to the widget's default, so the default has to be the current pick.
    ids = list(picks)
    st.selectbox("Left", ids, index=ids.index(left.id), format_func=label, key=f"cmp-left-config{suffix}",
                 on_change=_sync_pick, args=(f"cmp-left-config{suffix}", "cmp_left_pick"), disabled=busy)
    right_ids = [i for i in ids if i != left.id]
    st.selectbox("Right", right_ids, index=right_ids.index(right.id), format_func=label,
                 key=f"cmp-right-config{suffix}", on_change=_sync_pick,
                 args=(f"cmp-right-config{suffix}", "cmp_right_pick"), disabled=busy)


@st.dialog("Settings", width="large")
def _settings_dialog(status):
    records = st.session_state.get("run_records") or []
    avg = avg_latency(records)
    model = status["picked"].name
    info = _model_info(model)
    index_dir = status["index_dir"]
    loaded = index_ok(index_dir)
    chunks = _chunk_count(index_dir) if loaded else 0
    sha = st.session_state.get("git_sha", "")
    session_col, system_col = st.columns(2, gap="large")
    with session_col:
        st.markdown('<p class="dlg-h">Session</p>', unsafe_allow_html=True)
        st.markdown(
            sidebar_kv_html(
                [("Answers", str(len(records))), ("Average time", "—" if avg is None else f"{avg:.1f} s")]
            ),
            unsafe_allow_html=True,
        )
        if records:
            n = len(records)
            csv_col, json_col = st.columns(2, gap="small")
            csv_col.download_button("CSV", data=records_to_csv(records), file_name="polaris-runs.csv",
                                    mime="text/csv", key=f"export-csv-{n}", icon=":material/download:",
                                    use_container_width=True)
            json_col.download_button("JSON", data=records_to_json(records), file_name="polaris-runs.json",
                                     mime="application/json", key=f"export-json-{n}", icon=":material/download:",
                                     use_container_width=True)
        st.markdown('<p class="dlg-h">Models</p>', unsafe_allow_html=True)
        st.markdown(models_table_html(status["options"], status["names"], status["up"]), unsafe_allow_html=True)
    with system_col:
        st.markdown('<p class="dlg-h">System</p>', unsafe_allow_html=True)
        st.markdown(
            sidebar_kv_html(
                [
                    ("Model", model),
                    ("Digest", info.model_digest or "—"),
                    ("Ollama", f"running {info.version}".strip() if status["up"] else "stopped"),
                    ("Keep alive", keep_alive_setting()),
                    ("Context", f"{NUM_CTX:,} tokens"),
                    ("Index", Path(index_dir).name),
                    ("Chunks", f"{chunks:,}" if loaded else "not loaded"),
                    ("IDK threshold", f"cos {format_threshold(status['frozen']['threshold'])}"),
                    ("Top-k", str(status["frozen"]["k"])),
                    ("Network", "connected" if status["online"] else "offline"),
                    ("Warm-up", warm_label(st.session_state.get("warmed"))),
                    ("Commit", sha[:7] if sha else "—"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<p class="dlg-h">About the numbers</p>', unsafe_allow_html=True)
        st.markdown(
            "".join(f'<p class="dlg-note">{escape(note)}</p>' for note in (RATE_NOTE, SCORES_NOTE, RAM_TITLE + "."))
            + '<p class="dlg-note">Ubuntu Sans, Ubuntu Font Licence 1.0 · JetBrains Mono, SIL OFL 1.1</p>',
            unsafe_allow_html=True,
        )


def _render_composer(where, summary, conn, generating=False):
    with st.container(key=f"composer-{where}"):
        typed = st.chat_input("Ask about Linux or the lab", key=f"chat-{where}")
        conn_col, ctl_col = st.columns([1, 1], gap="small", vertical_alignment="center")
        conn_col.markdown(composer_note_html(conn), unsafe_allow_html=True)
        ctl_col.markdown(summary, unsafe_allow_html=True)
        stop_slot = st.empty()
        if generating:
            stop_slot.button("Stop", key="stop-gen", help="Stop generating")
    return typed, stop_slot


def _pick_example():
    picked = st.session_state.get("ex-pick")
    if picked:
        st.session_state.pending_q = picked
    st.session_state["ex-pick"] = None


def _render_examples():
    with st.container(key="examples"):
        st.pills(
            "Examples",
            list(EXAMPLES),
            key="ex-pick",
            label_visibility="collapsed",
            on_change=_pick_example,
        )


def _check_for_stop():
    """Let a Stop click or a new question interrupt this run without sending anything to the browser.

    Reading session state runs the same interrupt check as any st.* call that draws something, so the
    loader is drawn once instead of being re-sent every tick.
    """
    st.session_state.get("stop-gen")


def _pump(answer, events):
    """Drive the streaming answer on a worker thread, so the script thread is free to see a Stop click even while
    Ollama is still evaluating the prompt and has sent nothing."""
    try:
        for delta in answer:
            events.put(("text", delta))
        events.put(("done", None))
    except BaseException as e:
        events.put(("error", e))
    finally:
        answer.close()


def _stream_text(events, on_event, heartbeat=_check_for_stop, tick=0.2, every=0.1):
    """Text for st.write_stream, batched to at most one redraw per `every` seconds so the browser doesn't re-render
    the growing answer for every chunk while Ollama holds the CPU. Retrieval and progress go to on_event; waiting
    runs the heartbeat, where a Stop click or a new question interrupts this script."""
    pending = []
    flushed = time.monotonic()
    while True:
        wait = max(0.0, every - (time.monotonic() - flushed)) if pending else tick
        try:
            kind, value = events.get(timeout=wait)
        except queue.Empty:
            kind, value = None, None
        if kind == "error":
            raise value
        if kind == "text":
            pending.append(value)
        elif kind in ("retrieval", "progress"):
            on_event(kind, value)
        if pending and (kind in (None, "done") or time.monotonic() - flushed >= every):
            yield "".join(pending)
            pending.clear()
            flushed = time.monotonic()
        if kind == "done":
            return
        if kind is None:
            heartbeat()


class _Monitor:
    """The rail's live view of the current turn: each side's character trace and retrieval scores while it streams,
    then the last finished answer. Redrawn at most every `every` seconds."""

    def __init__(self, slot, every=0.5):
        self.slot = slot
        self.every = every
        self.runs = []
        self._drawn = None

    def show_last(self):
        self.runs = []
        runs = last_runs(st.session_state.get("messages") or [], st.session_state.get("live_traces") or {})
        self.slot.markdown(monitor_html(runs), unsafe_allow_html=True)

    def plan(self, sides):
        self.runs = [
            {"side": side, "model": model, "mode": mode, "state": "waiting", "trace": RateTrace(self.every)}
            for side, model, mode in sides
        ]
        self.draw(force=True)

    def begin(self, side):
        run = next(r for r in self.runs if r["side"] == side)
        run["state"] = "streaming"
        self.draw(force=True)
        return run

    def retrieval(self, run, chart):
        run.update(hits=hits_from_chart(chart), threshold=chart.dense_threshold)
        self.draw(force=True)

    def progress(self, run, live):
        run["trace"].add(live)
        self.draw()

    def end(self, run, state, record=None):
        run["trace"].finish()
        run.update(state=state, record=record)
        self.draw(force=True)

    def draw(self, force=False):
        now = time.monotonic()
        if not force and self._drawn is not None and now - self._drawn < self.every:
            return
        self._drawn = now
        views = []
        for run in self.runs:
            trace = run["trace"]
            views.append({**run, "samples": tuple(trace.samples), "first_s": trace.first_s, "live": trace.last})
        streaming = any(r["state"] in ("streaming", "waiting") for r in self.runs)
        self.slot.markdown(monitor_html(views, streaming=streaming), unsafe_allow_html=True)


def _ollama_error_text(error):
    if isinstance(error, requests.ConnectionError):
        return f"Could not reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`.\n\n{error}"
    return f"Ollama stopped this answer with an error.\n\n{error}"


def _answer_record(
    question,
    mode,
    option,
    retriever,
    index_dir,
    slot,
    *,
    key,
    monitor,
    names=None,
    side="",
    context=None,
    tag="",
    show_error=True,
):
    """Stream one answer into slot. Returns its run record and the view it streamed into, or (None, None) if it was
    stopped or failed. The caller stores the record, then calls view.finish, so the card that streamed is the card
    that stays; `key` is its key."""
    run = monitor.begin(side)
    missing = _still_missing(option, names)
    if missing:
        st.session_state.last_error = missing
        monitor.end(run, "error")
        if show_error:
            slot.error(missing)
        else:
            slot.empty()
        return None, None
    events = queue.Queue()
    answer = stream_answer_question(
        question,
        mode,
        option.name,
        retriever=retriever,
        model_key=option.key,
        keep_alive=keep_alive_setting(),
        git_sha=_session_git_commit(),
        model_info=_model_info(option.name),
        context=context,
        messages_fn=partial(app_messages, step_note=context.note_for(side) if context is not None else ""),
        on_retrieval=lambda chart: events.put(("retrieval", chart)),
        on_progress=lambda live: events.put(("progress", live)),
    )
    worker = threading.Thread(target=_pump, args=(answer, events), daemon=True)
    done = False
    try:
        with slot.container(key=f"live-{key}"):
            view = _AnswerView(mode, tag)
        view.loading(live_phase_label(mode))
        weak = False

        def on_event(kind, value):
            nonlocal weak
            if kind == "retrieval":
                monitor.retrieval(run, value)
                weak = value.weak
                if weak:
                    # No generation: the abstain card is drawn when the turn finishes, so the cube stops here.
                    view.stop_loading()
                else:
                    view.loading(live_phase_label(mode, retrieved=True))
                return
            monitor.progress(run, value)

        worker.start()
        text = ""
        for piece in _stream_text(events, on_event):
            if weak:
                continue
            text += piece
            view.show(parse_answer(text, mode, n_hits=len(run.get("hits") or ()), streaming=True))
        result = answer.result
        done = True
    except ChatCancelled:
        st.session_state.last_cancelled = True
        monitor.end(run, "stopped")
        slot.empty()
        return None, None
    except requests.RequestException as e:
        message = _ollama_error_text(e)
        st.session_state.last_error = message
        monitor.end(run, "error")
        if show_error:
            slot.error(message)
        else:
            slot.empty()
        return None, None
    finally:
        if not done:
            answer.cancel()
        if worker.is_alive():
            worker.join(timeout=1.0)
        if not worker.is_alive():
            answer.close()
    rec = build_run_record(result, index_path=index_dir)
    monitor.end(run, "done", rec)
    st.session_state.setdefault("live_traces", {})[key] = run["trace"].frozen()
    st.session_state.run_records.append(rec)
    st.session_state.warmed = True
    return rec, view


def _run_compare(question, left, right, retriever, index_dir, *, monitor, names=None, context=None):
    left_model, right_model = left.option.name, right.option.name
    left_mode, right_mode = left.mode, right.mode
    left_tag, right_tag = (left_model, right_model) if left_model != right_model else ("", "")
    messages = st.session_state.messages
    message_index = len(messages)
    message = {"role": "assistant"}
    recs = [None, None]
    # A side counts as cancelled until it answers or fails, so a click that ends this run mid-compare (Stop, a new
    # question) leaves the side that was running, and one not started yet, cancelled.
    unanswered = [
        {"mode": left_mode, "model": left_model, "state": "cancelled"},
        {"mode": right_mode, "model": right_model, "state": "cancelled"},
    ]
    monitor.plan([("L", left_model, left_mode), ("R", right_model, right_mode)])
    try:
        with st.container(key="cmp"):
            col_l, col_r = st.columns(2, gap="medium")
            wait = col_r.empty()
            wait.markdown(
                waiting_html(right_mode, "the left answer", right_tag),
                unsafe_allow_html=True,
            )
            with col_l:
                left_slot = st.empty()
                recs[0], view = _answer_record(
                    question,
                    left_mode,
                    left.option,
                    retriever,
                    index_dir,
                    left_slot,
                    key=f"answer-h{message_index}-l",
                    monitor=monitor,
                    names=names,
                    side="L",
                    context=context,
                    tag=left_tag,
                    show_error=False,
                )
                if recs[0] is not None:
                    unanswered[0] = None
                    _keep_compare(messages, message, recs, unanswered)
                    view.finish(recs[0], align_slots=True)
                else:
                    unanswered[0] = _side_outcome(left_mode, left_model)
                    _render_unanswered_side(unanswered[0], key=f"answer-h{message_index}-l", model=left_tag)
            wait.empty()
            if unanswered[0] and unanswered[0]["state"] == "cancelled":
                with col_r:
                    _render_unanswered_side(unanswered[1], key=f"answer-h{message_index}-r", model=right_tag)
                return tuple(recs), tuple(unanswered)
            with col_r:
                right_slot = st.empty()
                recs[1], view = _answer_record(
                    question,
                    right_mode,
                    right.option,
                    retriever,
                    index_dir,
                    right_slot,
                    key=f"answer-h{message_index}-r",
                    monitor=monitor,
                    names=names,
                    side="R",
                    context=context,
                    tag=right_tag,
                    show_error=False,
                )
                if recs[1] is not None:
                    unanswered[1] = None
                    _keep_compare(messages, message, recs, unanswered)
                    view.finish(recs[1], align_slots=True)
                else:
                    unanswered[1] = _side_outcome(right_mode, right_model)
                    _render_unanswered_side(unanswered[1], key=f"answer-h{message_index}-r", model=right_tag)
    finally:
        _keep_compare(messages, message, recs, unanswered)
    return tuple(recs), tuple(unanswered)


def _side_outcome(mode, model):
    """Why a compare side has no record, kept with its config so a replay can say which side it was."""
    if st.session_state.pop("last_cancelled", False):
        return {"mode": mode, "model": model, "state": "cancelled"}
    return {"mode": mode, "model": model, "state": "error", "error": st.session_state.pop("last_error", "")}


def _keep_compare(messages, message, recs, unanswered):
    """Put a compare turn in the thread as `message`, or bring it up to date there. Makes no st call, so it still runs
    while a click is ending this run.

    Called as soon as a side answers: a click starts the next run straight away, while this one is still unwinding,
    and that run draws the thread as it is then. With no answer on either side, the turn is kept the way a single
    answer's would be: nothing if it was stopped, so the thread shows it as stopped, or else its error.
    """
    if any(recs):
        message["compare"] = tuple(recs)
        if any(unanswered):
            message["compare_unanswered"] = tuple(unanswered)
        else:
            message.pop("compare_unanswered", None)
        if not any(kept is message for kept in messages):
            messages.append(message)
    elif not any(side["state"] == "cancelled" for side in unanswered):
        messages.append({"role": "assistant", "error": unanswered[0]["error"] or unanswered[1]["error"]})


def _render_unanswered_side(side, key, model=""):
    with st.container(key=key):
        st.markdown(card_head_html(side.get("mode") or "", model), unsafe_allow_html=True)
        if side.get("state") == "cancelled":
            st.markdown(cancelled_side_html(), unsafe_allow_html=True)
        else:
            st.error(side.get("error") or "That answer failed.")


def _replay_message(msg, index, compact_compare=False):
    role = msg.get("role")
    if role == "user":
        st.markdown(user_turn_html(msg.get("content") or ""), unsafe_allow_html=True)
        return
    if msg.get("compare"):
        left, right = msg["compare"]
        miss_l, miss_r = msg.get("compare_unanswered") or (None, None)
        if compact_compare and left and right:
            st.markdown(compare_compact_html(left, right), unsafe_allow_html=True)
            return
        left_tag, right_tag = compare_model_tags(left or miss_l, right or miss_r)
        with st.container(key=f"cmp-hist-{index}"):
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                if left:
                    _render_card(left, key=f"answer-h{index}-l", align_slots=True, model=left_tag)
                elif miss_l:
                    _render_unanswered_side(miss_l, key=f"answer-h{index}-l", model=left_tag)
            with col_r:
                if right:
                    _render_card(right, key=f"answer-h{index}-r", align_slots=True, model=right_tag)
                elif miss_r:
                    _render_unanswered_side(miss_r, key=f"answer-h{index}-r", model=right_tag)
        return
    rec = msg.get("record")
    if rec:
        _render_card(rec, key=f"answer-{index}")
        return
    if "error" in msg:
        st.error(msg["error"] or "That answer failed.")
        return
    st.markdown(msg.get("content") or "")


def _render_card(record, key, align_slots=False, model=""):
    with st.container(key=key):
        view = _AnswerView(record.get("mode") or "", model)
    view.finish(record, align_slots=align_slots)


def _answer_parts(parsed):
    """The answer's blocks as the values its slots show, in order: ("md", text), ("code", text, lang) or
    ("html", html) for the app's own note. Step numbers are the parser's, so a code box between two steps
    can't restart the list at 1."""
    parts, noted = [], False
    for block in parsed.blocks:
        if block.loose and not noted:
            noted = True
            parts.append(("html", loose_commands_html(len(parsed.loose))))
        if block.kind == "code":
            parts.append(("code", block.text, block.lang or "bash"))
        else:
            parts.append(("md", block.text))
    return parts


def _draw_part(slot, part):
    kind = part[0]
    if kind == "code":
        slot.code(part[1], language=part[2])
    elif kind == "md":
        slot.markdown(safe_markdown(part[1]))
    else:
        slot.markdown(part[1], unsafe_allow_html=True)


class _AnswerView:
    """One answer, streaming or replayed, in a container built once: head, reasoning, blocks, loader, footer.

    Streamlit keeps the children of a container that is sent again, so redrawing a whole container per chunk
    leaves rows of an earlier, longer draft under the new text until the run ends. Here every block has its own
    slot, rewritten only when its content changes, and slots the answer no longer needs are emptied. A replay
    builds the same slots and draws the finished answer once, so streamed, finished and replayed cards match.

    The loader stays through the search, the wait and any reasoning, and goes when the first answer block is drawn
    (or the turn ends). A new label keeps the cube's place in its turn."""

    def __init__(self, mode, model=""):
        self.mode = mode
        st.markdown(card_head_html(mode, model), unsafe_allow_html=True)
        self._reasoning = st.empty()
        self._reasoning_body = None
        self._reasoning_shown = None
        self._blocks = st.container()
        self._slots, self._shown = [], []
        self._status = st.empty()
        self._loading = None
        self._loading_since = None
        self._foot = st.container()

    def loading(self, label, note=None):
        if self._loading == (label, note):
            return
        now = time.monotonic()
        self._loading_since = self._loading_since or now
        self._loading = (label, note)
        self._status.markdown(thinking_html(label, elapsed=now - self._loading_since, note=note),
                              unsafe_allow_html=True)

    def stop_loading(self):
        if self._loading is not None:
            self._status.empty()
            self._loading = None

    def show(self, parsed):
        parts = _answer_parts(parsed)
        self._show_reasoning(parsed.reasoning)
        if parts:
            self.stop_loading()
        elif parsed.reasoning_open and self._loading is not None:
            self.loading(REASONING_LABEL, REASONING_NOTE)
        self._show_parts(parts)

    def _show_reasoning(self, text):
        if not text:
            if self._reasoning_body is not None:
                self._reasoning.empty()
                self._reasoning_body = self._reasoning_shown = None
            return
        if self._reasoning_body is None:
            with self._reasoning.container():
                self._reasoning_body = st.expander("Show reasoning", expanded=False).empty()
        if text != self._reasoning_shown:
            self._reasoning_body.markdown(safe_markdown(text))
            self._reasoning_shown = text

    def _show_parts(self, parts):
        for i, part in enumerate(parts):
            if i == len(self._slots):
                self._slots.append(self._blocks.empty())
                self._shown.append(None)
            if self._shown[i] != part:
                _draw_part(self._slots[i], part)
                self._shown[i] = part
        for i in range(len(parts), len(self._slots)):
            if self._shown[i] is not None:
                self._slots[i].empty()
                self._shown[i] = None

    def finish(self, record, align_slots=False):
        """The finished answer from its record: the final parse (or the abstain card), then sources and metrics."""
        mode = self.mode
        hits = record.get("hits") or []
        abstained = bool(record.get("abstained"))
        if abstained:
            self._show_reasoning("")
            self._show_parts([("html", idk_card_html(record))])
        else:
            parsed = parse_answer(record.get("answer") or "", mode, n_hits=len(hits))
            self.show(parsed)
        self.stop_loading()
        with self._foot:
            if not abstained:
                st.markdown(source_line_html(parsed, mode, hits), unsafe_allow_html=True)
            if uses_rag(mode):
                st.markdown(evidence_block_html(record, collapsed=True), unsafe_allow_html=True)
            elif align_slots:
                st.markdown(empty_slot_html("ev", f"No sources in {mode_name(mode)}"), unsafe_allow_html=True)
            if abstained:
                st.markdown(best_match_html(record), unsafe_allow_html=True)
            elif align_slots:
                st.markdown(empty_slot_html("best"), unsafe_allow_html=True)
            st.markdown(metrics_html(record), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
