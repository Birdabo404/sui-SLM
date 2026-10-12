"""Frozen retrieval settings used by eval and the chat app.

Tune threshold and k on --split dev only, then write the chosen values here.
The test split must use this file as-is.
"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "frozen.json"


def load_frozen(path=None):
    p = Path(path) if path else PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    try:
        threshold = float(data["threshold"])
        k = int(data["k"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{p} must contain numeric threshold and k") from e
    if k < 1:
        raise ValueError(f"{p} k must be >= 1")
    return {"threshold": threshold, "k": k}
