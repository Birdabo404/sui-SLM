"""Provisional retrieval settings used by eval and the chat app.

threshold 0.30 and k=5 are starting defaults, not a tuned result.
Run --split dev on the real index, pick values from the sweep CSV,
write them here, and commit that CSV. Do not retune on the test split.
"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "frozen.json"


def format_threshold(value):
    return f"{float(value):.2f}"


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
