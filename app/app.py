#!/usr/bin/env python3
"""Offline Streamlit chat app for the CCIS Linux lab assistant."""
import os
import sqlite3
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import requests
import streamlit as st

from app.chat_memory import ChatStore, build_chat_context
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
    keep_alive_setting,
    model_name_for,
    mode_choices,
    ollama_info,
    ollama_reachable,
    uses_rag,
    warm_model,
)
from app.polaris import (
    DELETE_ASK,
    EXAMPLES,
    HISTORY_EMPTY,
    CancellableChat,
    ChatCancelled,
    avg_latency,
    best_match_html,
    build_run_record,
    card_head_html,
    compare_compact_html,
    compare_configs,
    compare_model_tags,
    compare_summary_html,
    composer_note_html,
    connection_html,
    default_compare_modes,
    default_compare_model_keys,
    empty_hero_html,
    empty_slot_html,
    evidence_block_html,
    flags_html,
    header_band_html,
    history_error_text,
    history_note_html,
    idk_card_html,
    index_chunk_count,
    injected_css,
    metrics_html,
    mode_menu_css,
    mode_name,
    model_key_choices,
    network_online,
    pair_compare_config_modes,
    pair_compare_model_keys,
    plain_label,
    polaris_on,
    records_in,
    records_to_csv,
    records_to_json,
    retrieval_bar_html,
    running_card_html,
    side_logo_html,
    sky_html,
    sky_level,
    sidebar_kv_html,
    split_code_blocks,
    split_reasoning,
    stopped_html,
    thinking_html,
    thinking_label,
    user_turn_html,
    waiting_html,
    warm_label,
)
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


def _init_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("run_records", [])
    st.session_state.setdefault("warmed", False)
    st.session_state.setdefault("mode", "full")
    st.session_state.setdefault("compare", False)
    st.session_state.setdefault("chat", _chat_state())
    _session_git_commit()
    if "cmp_left" not in st.session_state or "cmp_right" not in st.session_state:
        if len(model_key_choices(DISTILLED_MODEL)) > 1:
            left, right = "full", "full"
        else:
            left, right = default_compare_modes()
        st.session_state.setdefault("cmp_left", left)
        st.session_state.setdefault("cmp_right", right)
    if "cmp_left_model" not in st.session_state or "cmp_right_model" not in st.session_state:
        left, right = default_compare_model_keys(DISTILLED_MODEL)
        st.session_state.setdefault("cmp_left_model", left)
        st.session_state.setdefault("cmp_right_model", right)


@st.cache_resource
def _loaded_indexes():
    return set()


def run_polaris():
    _inject_css()
    _init_session()
    index_dir = DEFAULT_INDEX
    model = BASE_MODEL
    frozen = load_frozen()
    up = _ollama_up()
    if up and not st.session_state.warmed:
        st.session_state.warmed = _warm_model_at_startup(model, keep_alive_setting())
    elif not up:
        st.session_state.warmed = False
        _warmed_models().discard(model)
    online = _network_online()
    # A question from the landing (typed or an example chip) is answered in the run it arrives in. The layout is
    # chosen here, before anything renders, so this run builds the thread exactly where every later run does.
    landing_q = st.session_state.get("pending_q") or st.session_state.get("chat-landing")
    incoming = landing_q or st.session_state.get("chat-dock")
    msgs = st.session_state.messages
    chat = st.session_state.chat
    if incoming and chat["id"] is None:
        # Stored with its question before the sidebar draws, so a new chat is listed as current in this same run.
        _sync_chat(chat, [*msgs, {"role": "user", "content": incoming}])
    compare_on = bool(st.session_state.get("compare"))
    asked = sum(1 for m in msgs if m.get("role") == "user") + (1 if incoming else 0)

    flags = st.empty()

    def mark(generating):
        flags.markdown(
            flags_html(
                sky_level(asked),
                daylight=bool(st.session_state.get("daylight")),
                compare=compare_on,
                generating=generating,
            )
            + mode_menu_css(model, mode_choices()),
            unsafe_allow_html=True,
        )

    # Before the sidebar, so the star's first wrap starts out paused while this run generates.
    mark(generating=bool(incoming))
    refresh_sidebar = _render_sidebar(model, index_dir, frozen, up, online, chat=bool(msgs or incoming))
    st.markdown(sky_html(), unsafe_allow_html=True)

    left_model_key, left_mode, right_model_key, right_mode = _compare_sides(
        st.session_state.get("cmp_left_model"),
        st.session_state.get("cmp_left"),
        st.session_state.get("cmp_right_model"),
        st.session_state.get("cmp_right"),
    )
    conn = connection_html(online, up)
    composer = dict(
        compare_on=compare_on,
        compare_line=compare_summary_html(
            model_name_for(left_model_key, base_model=model, distilled_model=DISTILLED_MODEL),
            left_mode,
            model_name_for(right_model_key, base_model=model, distilled_model=DISTILLED_MODEL),
            right_mode,
        ),
        conn=conn,
    )

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
        return

    mode = st.session_state.get("mode", "full")
    context = build_chat_context(msgs, question)
    msgs.append({"role": "user", "content": question})
    try:
        with thread:
            answered = _answer_turn(
                question,
                mode,
                compare_on,
                left_mode,
                right_mode,
                model,
                index_dir,
                left_model_key=left_model_key,
                right_model_key=right_model_key,
                context=context,
            )
    finally:
        # Stored before the next st call: a click can end this run at any st call, and reading session state counts
        # as one. A stopped or failed turn still keeps its question.
        _sync_chat(chat, msgs)
        # A turn that raises still takes the Stop button away and lets the sky move again.
        stop_slot.empty()
        mark(generating=False)
    if answered:
        refresh_sidebar()


def _answer_turn(
    question,
    mode,
    compare_on,
    left_mode,
    right_mode,
    model,
    index_dir,
    *,
    left_model_key="base",
    right_model_key="base",
    context=None,
):
    st.session_state.pop("last_cancelled", None)
    st.markdown(user_turn_html(question), unsafe_allow_html=True)
    slot = st.empty()

    need_rag = (uses_rag(left_mode) or uses_rag(right_mode)) if compare_on else uses_rag(mode)
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
        recs, failures = _run_compare(
            question,
            left_mode,
            right_mode,
            model,
            retriever,
            index_dir,
            left_model_key=left_model_key,
            right_model_key=right_model_key,
            context=context,
        )
        if recs[0] is None and recs[1] is None:
            if st.session_state.pop("last_cancelled", False):
                return False
            error = next((failure["error"] for failure in failures if failure), "")
            st.session_state.messages.append({"role": "assistant", "error": error})
            return False
        message = {"role": "assistant", "compare": recs}
        if any(failures):
            message["compare_errors"] = failures
        st.session_state.messages.append(message)
        return True

    rec = _answer_record(
        question,
        mode,
        model,
        retriever,
        index_dir,
        slot=slot,
        context=context,
    )
    if rec is None:
        if st.session_state.pop("last_cancelled", False):
            return False
        st.session_state.messages.append({"role": "assistant", "error": st.session_state.pop("last_error", "")})
        return False
    _render_card(rec, key=f"answer-{len(st.session_state.messages)}", slot=slot)
    st.session_state.messages.append({"role": "assistant", "record": rec})
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


def _new_chat():
    st.session_state.messages = []
    st.session_state.run_records = []
    st.session_state.chat = _chat_state()
    st.session_state.pop("hist_confirm", None)


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
    if chat_id == st.session_state.chat["id"]:
        _new_chat()


def _history_panel():
    st.markdown('<p class="side-h">Chats</p>', unsafe_allow_html=True)
    store = ChatStore()
    try:
        chats = store.list_chats()
    except (sqlite3.Error, OSError):
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
            # The open chat's row is disabled: clicking it would only re-run, and that stops an answer mid-flight.
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


def _compare_sides(left_key, left_mode, right_key, right_mode):
    """Each compare side's model key and config, made valid as a pair: the model keys first, then the configs for
    those models. Returns them in the order they came in."""
    left_key, right_key = pair_compare_model_keys(left_key, right_key, DISTILLED_MODEL)
    left_mode, right_mode = pair_compare_config_modes(left_mode, right_mode, left_key, right_key)
    return left_key, left_mode, right_key, right_mode


def _compare_pickers():
    """One picker per side over the thesis configs, written back to the cmp_* keys run_polaris pairs up."""
    configs = compare_configs(DISTILLED_MODEL)
    two_models = len(model_key_choices(DISTILLED_MODEL)) > 1

    def label(cid):
        return cid if two_models else mode_name(configs[cid][1])

    left_key, left_mode, right_key, right_mode = _compare_sides(
        st.session_state.get("cmp_left_model"),
        st.session_state.get("cmp_left"),
        st.session_state.get("cmp_right_model"),
        st.session_state.get("cmp_right"),
    )
    # Pass index rather than seeding the key through session state: the browser shows a seeded value once, then
    # falls back to the widget's default, so the default has to be the current pick.
    left_options = list(configs)
    left = st.selectbox(
        "Left",
        left_options,
        index=left_options.index(config_id(left_key, left_mode)),
        format_func=label,
        key="cmp-left-config",
    )
    right_options = [cid for cid in configs if cid != left]
    wanted = config_id(right_key, right_mode)
    right = st.selectbox(
        "Right",
        right_options,
        index=right_options.index(wanted) if wanted in right_options else 0,
        format_func=label,
        key="cmp-right-config",
    )
    st.session_state.cmp_left_model, st.session_state.cmp_left = configs[left]
    st.session_state.cmp_right_model, st.session_state.cmp_right = configs[right]


def _render_sidebar(model, index_dir, frozen, up, online, chat=False):
    """Draw the sidebar and return refresh(), which redraws the parts an answer changes in place."""
    with st.sidebar:
        with st.container(key="side-top"):
            st.markdown(side_logo_html(play=chat), unsafe_allow_html=True)
            st.button(
                "New chat",
                key="new-chat",
                icon=":material/edit_square:",
                use_container_width=True,
                on_click=_new_chat,
            )

        _history_panel()

        st.markdown('<p class="side-h">View</p>', unsafe_allow_html=True)
        st.toggle("Compare two configs", key="compare")
        if st.session_state.get("compare"):
            with st.container(key="cmp-pickers"):
                _compare_pickers()
        st.toggle("Daylight", key="daylight")

        session = st.empty()
        system = st.empty()
        st.markdown(
            '<p class="footer-license">Ubuntu Sans, Ubuntu Font Licence 1.0<br>'
            "JetBrains Mono, SIL OFL 1.1</p>",
            unsafe_allow_html=True,
        )

    def refresh():
        _session_panel(session)
        _system_panel(system, model, index_dir, frozen, up, online)

    refresh()
    return refresh


def _session_panel(slot):
    records = st.session_state.get("run_records") or []
    avg = avg_latency(records)
    with slot.container():
        st.markdown('<p class="side-h">Session</p>', unsafe_allow_html=True)
        st.markdown(
            sidebar_kv_html(
                [
                    ("Answers", str(len(records))),
                    ("Average time", "—" if avg is None else f"{avg:.1f} s"),
                ]
            ),
            unsafe_allow_html=True,
        )
        if not records:
            return
        # Keyed by the record count: an answer fills this slot a second time in the same run, and widget keys
        # must be unique per run.
        n = len(records)
        csv_col, json_col = st.columns(2, gap="small")
        csv_col.download_button(
            "CSV",
            data=records_to_csv(records),
            file_name="polaris-runs.csv",
            mime="text/csv",
            key=f"export-csv-{n}",
            icon=":material/download:",
            use_container_width=True,
        )
        json_col.download_button(
            "JSON",
            data=records_to_json(records),
            file_name="polaris-runs.json",
            mime="application/json",
            key=f"export-json-{n}",
            icon=":material/download:",
            use_container_width=True,
        )


def _system_panel(slot, model, index_dir, frozen, up, online):
    info = _model_info(model)
    loaded = index_ok(index_dir)
    chunks = _chunk_count(index_dir) if loaded else 0
    sha = st.session_state.get("git_sha", "")
    with slot.container():
        st.markdown('<p class="side-h">System</p>', unsafe_allow_html=True)
        st.markdown(
            sidebar_kv_html(
                [
                    ("Model", model),
                    ("Digest", info.model_digest or "—"),
                    ("Index", Path(index_dir).name),
                    ("Chunks", f"{chunks:,}" if loaded else "not loaded"),
                    ("IDK threshold", f"cos {format_threshold(frozen['threshold'])}"),
                    ("Top-k", str(frozen["k"])),
                    ("Ollama", "running" if up else "stopped"),
                    ("Network", "connected" if online else "offline"),
                    ("Warm-up", warm_label(st.session_state.get("warmed"))),
                    ("Commit", sha[:7] if sha else "—"),
                ]
            ),
            unsafe_allow_html=True,
        )


def _sync_mode(key):
    picked = st.session_state.get(key)
    if picked:
        st.session_state.mode = picked


def _render_composer(where, compare_on, compare_line, conn, generating=False):
    with st.container(key=f"composer-{where}"):
        typed = st.chat_input("Ask about Linux or the lab", key=f"chat-{where}")
        conn_col, ctl_col = st.columns([1, 1], gap="small", vertical_alignment="center")
        conn_col.markdown(composer_note_html(conn), unsafe_allow_html=True)
        if compare_on:
            ctl_col.markdown(compare_line, unsafe_allow_html=True)
        else:
            key = f"mode-{where}"
            modes = mode_choices()
            mode = st.session_state.get("mode", "full")
            if key in st.session_state and st.session_state[key] is None:
                del st.session_state[key]
            with ctl_col, st.container(key=f"mode-{where}-box"):
                st.selectbox(
                    "Mode",
                    modes,
                    index=modes.index(mode) if mode in modes else len(modes) - 1,
                    format_func=mode_name,
                    key=key,
                    label_visibility="collapsed",
                    on_change=_sync_mode,
                    args=(key,),
                )
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


def _run_stoppable(work, heartbeat=_check_for_stop, tick=0.2):
    """Run work(chat) on a worker thread and wait for it.

    The heartbeat is where a Stop click or a new question interrupts this script; the finally then
    cancels the request so Ollama stops generating.
    """
    chat = CancellableChat()
    box = {}

    def run():
        try:
            box["value"] = work(chat)
        except BaseException as e:
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(tick)
            if worker.is_alive():
                heartbeat()
    finally:
        if worker.is_alive():
            chat.cancel()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _answer_record(
    question,
    mode,
    model,
    retriever,
    index_dir,
    panel=False,
    slot=None,
    *,
    model_key="base",
    context=None,
    tag="",
):
    keep = slot is not None
    slot = slot if keep else st.empty()
    label = thinking_label(mode)
    loader = running_card_html(mode, label, tag) if panel else thinking_html(label)
    keep_alive = keep_alive_setting()
    git_sha = _session_git_commit()
    model_info = _model_info(model)
    slot.markdown(loader, unsafe_allow_html=True)
    try:
        result = _run_stoppable(
            lambda chat: answer_question(
                question,
                mode,
                model,
                retriever=retriever,
                chat_fn=chat,
                model_key=model_key,
                log_commit=chat.commit,
                keep_alive=keep_alive,
                git_sha=git_sha,
                model_info=model_info,
                context=context,
            ),
        )
    except ChatCancelled:
        st.session_state.last_cancelled = True
        slot.empty()
        return None
    except requests.RequestException as e:
        message = f"Could not reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`.\n\n{e}"
        st.session_state.last_error = message
        if keep:
            slot.error(message)
        else:
            slot.empty()
        return None
    if not keep:
        slot.empty()
    rec = build_run_record(result, index_path=index_dir)
    st.session_state.run_records.append(rec)
    st.session_state.warmed = True
    return rec


def _run_compare(
    question,
    left_mode,
    right_mode,
    model,
    retriever,
    index_dir,
    *,
    left_model_key="base",
    right_model_key="base",
    context=None,
):
    left_model_key, left_mode, right_model_key, right_mode = _compare_sides(
        left_model_key,
        left_mode,
        right_model_key,
        right_mode,
    )
    left_model = model_name_for(
        left_model_key,
        base_model=model,
        distilled_model=DISTILLED_MODEL,
    )
    right_model = model_name_for(
        right_model_key,
        base_model=model,
        distilled_model=DISTILLED_MODEL,
    )
    left_tag, right_tag = (left_model, right_model) if left_model != right_model else ("", "")
    message_index = len(st.session_state.messages)
    rec_l = rec_r = fail_l = fail_r = None
    with st.container(key="cmp"):
        col_l, col_r = st.columns(2, gap="medium")
        wait = col_r.empty()
        wait.markdown(
            waiting_html(right_mode, "the left answer", right_tag),
            unsafe_allow_html=True,
        )
        with col_l:
            rec_l = _answer_record(
                question,
                left_mode,
                left_model,
                retriever,
                index_dir,
                panel=True,
                model_key=left_model_key,
                context=context,
                tag=left_tag,
            )
            if rec_l is not None:
                _render_card(rec_l, key=f"answer-h{message_index}-l", align_slots=True, model=left_tag)
            else:
                fail_l = _side_failure(left_mode, left_model)
                if fail_l:
                    _render_failed_side(fail_l, key=f"answer-h{message_index}-l", model=left_tag)
        if st.session_state.get("last_cancelled") is True:
            wait.empty()
            return (rec_l, rec_r), (fail_l, fail_r)
        wait.empty()
        with col_r:
            rec_r = _answer_record(
                question,
                right_mode,
                right_model,
                retriever,
                index_dir,
                panel=True,
                model_key=right_model_key,
                context=context,
                tag=right_tag,
            )
            if rec_r is not None:
                _render_card(rec_r, key=f"answer-h{message_index}-r", align_slots=True, model=right_tag)
            else:
                fail_r = _side_failure(right_mode, right_model)
                if fail_r:
                    _render_failed_side(fail_r, key=f"answer-h{message_index}-r", model=right_tag)
    return (rec_l, rec_r), (fail_l, fail_r)


def _side_failure(mode, model):
    """The error a compare side came back with, kept with its config so a replay can say which side failed."""
    error = st.session_state.pop("last_error", None)
    return None if error is None else {"mode": mode, "model": model, "error": error}


def _render_failed_side(failure, key, model=""):
    with st.container(key=key):
        st.markdown(card_head_html(failure.get("mode") or "", model), unsafe_allow_html=True)
        st.error(failure.get("error") or "That answer failed.")


def _replay_message(msg, index, compact_compare=False):
    role = msg.get("role")
    if role == "user":
        st.markdown(user_turn_html(msg.get("content") or ""), unsafe_allow_html=True)
        return
    if msg.get("compare"):
        left, right = msg["compare"]
        fail_l, fail_r = msg.get("compare_errors") or (None, None)
        if compact_compare and left and right:
            st.markdown(compare_compact_html(left, right), unsafe_allow_html=True)
            return
        left_tag, right_tag = compare_model_tags(left or fail_l, right or fail_r)
        with st.container(key=f"cmp-hist-{index}"):
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                if left:
                    _render_card(left, key=f"answer-h{index}-l", align_slots=True, model=left_tag)
                elif fail_l:
                    _render_failed_side(fail_l, key=f"answer-h{index}-l", model=left_tag)
            with col_r:
                if right:
                    _render_card(right, key=f"answer-h{index}-r", align_slots=True, model=right_tag)
                elif fail_r:
                    _render_failed_side(fail_r, key=f"answer-h{index}-r", model=right_tag)
        return
    rec = msg.get("record")
    if rec:
        _render_card(rec, key=f"answer-{index}")
        return
    if "error" in msg:
        st.error(msg["error"] or "That answer failed.")
        return
    st.markdown(msg.get("content") or "")


def _render_card(record, key, align_slots=False, slot=None, model=""):
    mode = record.get("mode") or ""
    rag = uses_rag(mode)
    abstained = bool(record.get("abstained"))
    with (slot.container(key=key) if slot is not None else st.container(key=key)):
        st.markdown(card_head_html(mode, model), unsafe_allow_html=True)
        if abstained:
            st.markdown(idk_card_html(record), unsafe_allow_html=True)
        elif mode in ("cot", "full"):
            reason, answer = split_reasoning(record.get("answer") or "")
            if reason:
                with st.expander("Steps", expanded=False):
                    st.markdown(reason)
                _render_answer_body(answer)
            else:
                _render_answer_body(record.get("answer") or "")
        else:
            _render_answer_body(record.get("answer") or "")
        if rag:
            st.markdown(evidence_block_html(record, collapsed=True), unsafe_allow_html=True)
        elif align_slots:
            st.markdown(
                empty_slot_html("ev", f"No sources in {mode_name(mode)}"),
                unsafe_allow_html=True,
            )
        st.markdown(
            retrieval_bar_html(
                record.get("retrieval_max_dense"),
                record.get("threshold"),
                retrieval_off=not rag,
                mode=mode,
            ),
            unsafe_allow_html=True,
        )
        if abstained:
            st.markdown(best_match_html(record), unsafe_allow_html=True)
        elif align_slots:
            st.markdown(empty_slot_html("best"), unsafe_allow_html=True)
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


if __name__ == "__main__":
    main()
