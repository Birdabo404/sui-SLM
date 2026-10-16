"""Ollama chat parsing, think=false, and /api/show model matching."""
import os
import re

import requests

from shared.prompt import SYSTEM, SYSTEM_BASE, SYSTEM_COT, SYSTEM_RAG

THINK = False
REQUIRED_QUANT = "Q4_K_M"
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
SHARED_SYSTEMS = (SYSTEM, SYSTEM_BASE, SYSTEM_COT, SYSTEM_RAG)
_DIGEST_CACHE = {}


def format_think(value=THINK):
    return "true" if value else "false"


def normalize_quant(value):
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in {"15", "Q4_K_M"}:
        return REQUIRED_QUANT
    return text


def tag_names(model):
    name = (model or "").strip()
    names = {name}
    if not name:
        return names
    if ":" not in name:
        names.add(name + ":latest")
    elif name.endswith(":latest"):
        names.add(name[: -len(":latest")])
    return names


def has_think_tags(text):
    return bool(THINK_OPEN_RE.search(text or ""))


def strip_think_leaks(text):
    """Remove <think> blocks from model text. Unclosed tags drop the rest of the string."""
    text = text or ""
    leaked = has_think_tags(text)
    cleaned = THINK_BLOCK_RE.sub("", text)
    m = THINK_OPEN_RE.search(cleaned)
    if m:
        leaked = True
        cleaned = cleaned[: m.start()]
    return cleaned.strip(), leaked


def inspect_chat_response(payload, think=THINK):
    """Pull content/thinking out of an /api/chat JSON body."""
    msg = (payload or {}).get("message") or {}
    content = msg.get("content") or ""
    thinking = msg.get("thinking")
    if thinking is None:
        thinking = (payload or {}).get("thinking") or ""
    thinking = "" if thinking is None else str(thinking)
    stripped, tag_leak = strip_think_leaks(content)
    thinking_leak = bool(thinking.strip())
    return chat_result(
        stripped,
        raw_output=content,
        thinking=thinking,
        think=think,
        think_leak=int(tag_leak or thinking_leak),
    )


def chat_result(text, raw_output=None, thinking="", think=THINK, think_leak=0):
    return {
        "text": text,
        "raw_output": text if raw_output is None else raw_output,
        "thinking": thinking,
        "think": format_think(think),
        "think_leak": int(think_leak),
    }


def empty_chat_result(text=""):
    return {
        "text": text,
        "raw_output": "",
        "thinking": "",
        "think": "",
        "think_leak": "",
    }


def as_chat_result(out):
    return {
        "text": out["text"],
        "raw_output": out["raw_output"],
        "thinking": out["thinking"],
        "think": out["think"],
        "think_leak": out["think_leak"],
    }


def _system_from_modelfile(modelfile):
    text = modelfile or ""
    m = re.search(r'(?im)^SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?im)^SYSTEM\s+"(.*)"\s*$', text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?im)^SYSTEM\s+(.+)$", text)
    if m:
        return m.group(1).strip()
    return ""


def digest_from_show(show):
    show = show or {}
    if show.get("digest"):
        return str(show["digest"])
    details = show.get("details") or {}
    if details.get("digest"):
        return str(details["digest"])
    info = show.get("model_info") or {}
    for key in ("general.sha256", "sha256", "digest"):
        if info.get(key):
            return str(info[key])
    return ""


def quant_from_show(show):
    show = show or {}
    details = show.get("details") or {}
    q = details.get("quantization_level")
    if not q:
        info = show.get("model_info") or {}
        q = info.get("general.file_type")
    return normalize_quant(q)


def template_from_show(show):
    return ((show or {}).get("template") or "").strip()


def system_from_show(show):
    show = show or {}
    if show.get("system"):
        return str(show["system"])
    return _system_from_modelfile(show.get("modelfile") or "")


def model_fields(show, name=""):
    return {
        "name": name,
        "quant": quant_from_show(show),
        "template": template_from_show(show),
        "system": system_from_show(show),
        "digest": digest_from_show(show),
    }


def _norm_ws(text):
    return " ".join((text or "").split())


def _preview(text, n=160):
    text = text or ""
    if len(text) <= n:
        return repr(text)
    return repr(text[:n]) + f"... ({len(text)} chars)"


def matched_model_diffs(base, distilled):
    diffs = []
    for info in (base, distilled):
        if info["quant"] != REQUIRED_QUANT:
            diffs.append(
                f"{info['name']} quant is {info['quant']!r}, expected {REQUIRED_QUANT}"
            )
    if not base["template"]:
        diffs.append(f"{base['name']} chat template is empty")
    if not distilled["template"]:
        diffs.append(f"{distilled['name']} chat template is empty")
    if base["template"] != distilled["template"]:
        diffs.append("chat templates differ")
        diffs.append(f"  base: {_preview(base['template'])}")
        diffs.append(f"  distilled: {_preview(distilled['template'])}")
    if _norm_ws(base["system"]) != _norm_ws(distilled["system"]):
        diffs.append("Modelfile SYSTEM prompts differ")
        diffs.append(f"  base: {_preview(base['system'])}")
        diffs.append(f"  distilled: {_preview(distilled['system'])}")
    shared_norm = {_norm_ws(s) for s in SHARED_SYSTEMS}
    for info in (base, distilled):
        sys_n = _norm_ws(info["system"])
        if sys_n and sys_n not in shared_norm:
            diffs.append(
                f"{info['name']} Modelfile SYSTEM is not from shared/prompt.py"
            )
            diffs.append(f"  got: {_preview(info['system'])}")
    return diffs


def quant_diffs(info):
    if info["quant"] == REQUIRED_QUANT:
        return []
    return [f"{info['name']} quant is {info['quant']!r}, expected {REQUIRED_QUANT}"]


def format_match_failure(diffs):
    return "Matched-model check failed:\n" + "\n".join(diffs)


def assert_matched_models(base, distilled):
    diffs = matched_model_diffs(base, distilled)
    if diffs:
        raise SystemExit(format_match_failure(diffs))


def assert_quant(info):
    diffs = quant_diffs(info)
    if diffs:
        raise SystemExit(format_match_failure(diffs))


def remember_digest(model, digest):
    if model:
        _DIGEST_CACHE[model] = digest


def clear_digest_cache():
    _DIGEST_CACHE.clear()


def cached_model_digest(model, host=None):
    if not model:
        return ""
    if model in _DIGEST_CACHE:
        return _DIGEST_CACHE[model]
    try:
        digest = digest_from_show(ollama_show(model, host=host))
    except SystemExit:
        digest = ""
    _DIGEST_CACHE[model] = digest
    return digest


def _digest_from_tags_payload(model, payload):
    aliases = tag_names(model)
    for m in (payload or {}).get("models") or []:
        if m.get("name") in aliases or m.get("model") in aliases:
            return m.get("digest") or ""
    return ""


def ollama_show(model, host=None):
    host = host or DEFAULT_HOST
    url = host.rstrip("/") + "/api/show"
    try:
        r = requests.post(url, json={"model": model}, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Ollama /api/show failed for {model!r}: {e}") from e
    data = r.json()
    if not digest_from_show(data):
        try:
            tags = requests.get(host.rstrip("/") + "/api/tags", timeout=60)
            tags.raise_for_status()
            found = _digest_from_tags_payload(model, tags.json())
            if found:
                data["digest"] = found
        except requests.RequestException:
            pass
    return data
