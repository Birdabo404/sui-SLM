"""Choices kept between sessions in prefs.json beside the chat history."""
import json
import os
from pathlib import Path

from app.chat_memory import chat_data_dir
from app.polaris import theme_or_default

PREFS_FILE = "prefs.json"
NORMALIZE_ENV = "NORMALIZE_QUESTION_BEFORE_SEARCH"
_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


def _root(root):
    return Path(root) if root is not None else chat_data_dir()


def load_prefs(root=None):
    try:
        data = json.loads((_root(root) / PREFS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(prefs, root=None):
    """Write prefs whole, replacing the file in one step so a crash never leaves half of it. False when the folder
    isn't writable: the choice then lasts for this session only."""
    folder = _root(root)
    try:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        part = folder / (PREFS_FILE + ".part")
        part.write_text(json.dumps(prefs, sort_keys=True), encoding="utf-8")
        part.replace(folder / PREFS_FILE)
    except OSError:
        return False
    return True


def saved_theme(root=None):
    return theme_or_default(load_prefs(root).get("theme"))


def save_theme(theme, root=None):
    return save_prefs({**load_prefs(root), "theme": theme_or_default(theme)}, root)


def normalization_env_override(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get(NORMALIZE_ENV)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def saved_normalization(root=None, environ=None):
    override = normalization_env_override(environ)
    if override is not None:
        return override
    return load_prefs(root).get("normalize_question_before_search") is True


def save_normalization(enabled, root=None):
    return save_prefs(
        {**load_prefs(root), "normalize_question_before_search": bool(enabled)},
        root,
    )
