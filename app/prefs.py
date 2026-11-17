"""Choices kept between sessions on this machine: only the theme, in prefs.json beside the chat history."""
import json
from pathlib import Path

from app.chat_memory import chat_data_dir
from app.polaris import theme_or_default

PREFS_FILE = "prefs.json"


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
