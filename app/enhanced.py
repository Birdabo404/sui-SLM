"""The Enhanced config: answers retrieved and generated exactly as in the evaluated config, plus optional checks that
run after an answer and are not part of the reported eval. The only one is the grounding score (app/grounding.py),
off by default.

The evaluated config never imports app/grounding.py. It is imported here by name, only when Enhanced needs it, so
this module also loads on a branch without it, and Enhanced then has nothing to add.
"""
import importlib

FROZEN = "frozen"
ENHANCED = "enhanced"
CONFIG_LABELS = {FROZEN: "Evaluated config (frozen)", ENHANCED: "Enhanced"}
NOT_EVALUATED = "not part of reported eval"
GROUNDING_MODEL = "vectara/hallucination_evaluation_model"


class Unavailable(Exception):
    """app.grounding can't be imported here."""


def grounding_backend():
    # By name and on first use, never at import time: the evaluated config must not import app.grounding at all.
    try:
        return importlib.import_module("app.grounding")
    except ImportError as e:
        raise Unavailable(f"app/grounding.py can't be imported: {e}") from e


def config_for(requested, uses_retrieval):
    """Enhanced only adds checks on excerpts, so a mode without retrieval runs the evaluated config either way."""
    return ENHANCED if requested == ENHANCED and uses_retrieval else FROZEN


def excerpts_of(hits):
    """Each hit as the model saw it: title, then text."""
    return [f"{h.get('title') or ''}\n{h.get('text') or ''}".strip() for h in hits or ()]


def ground(answer, excerpts, grounding=None):
    """app.grounding's score for answer against excerpts, as a small dict for the run record."""
    module = grounding or grounding_backend()
    result = module.score(answer, excerpts)
    score = float(result.score)
    labels = getattr(result, "labels", ()) or ()
    label = str(labels[0].label) if labels else ("consistent" if score >= 0.5 else "hallucinated")
    return {"score": score, "label": label, "model": str(getattr(result, "model", "") or GROUNDING_MODEL)}
