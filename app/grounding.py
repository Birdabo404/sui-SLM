"""Optional Enhanced-mode answer grounding.

The UI-owned toggle is off by default and must avoid calling ``score`` in the
frozen configuration. Model loading is lazy and restricted to the local cache
unless an explicit online setup call opts out.
"""
from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Sequence

from app.backend import UserFacingBackendError


GROUNDING_MODEL = "vectara/hallucination_evaluation_model"
CONSISTENT_THRESHOLD = 0.5


class GroundingModelError(UserFacingBackendError):
    """The optional grounding model is unavailable or failed."""

    code = "grounding_model_unavailable"
    message = "The optional grounding check is not available in the local cache."
    action = "Turn off grounding, or cache the Enhanced models while online."


@dataclass(frozen=True)
class GroundingLabel:
    """A model label for all or part of the generated answer."""

    label: str
    score: float
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class GroundingResult:
    """Overall factual-consistency score and label information."""

    score: float
    labels: tuple[GroundingLabel, ...]
    model: str = GROUNDING_MODEL


@lru_cache(maxsize=2)
def _load_scorer(model_name: str, local_files_only: bool):
    """Load HHEM once on CPU only when Enhanced grounding is enabled."""

    try:
        module = importlib.import_module("transformers")
        model = module.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        return model.to("cpu")
    except Exception as error:
        raise GroundingModelError("grounding model could not be loaded") from error


def _first_score(values) -> float:
    value = values
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    elif isinstance(value, Sequence):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def score(
    answer: str,
    excerpts: Sequence[str],
    *,
    scorer=None,
    model_name: str = GROUNDING_MODEL,
    local_files_only: bool = True,
    consistent_threshold: float = CONSISTENT_THRESHOLD,
) -> GroundingResult:
    """Score whether ``answer`` is supported by the retrieved ``excerpts``.

    The result uses HHEM's consistency probability: 1 means fully supported
    and 0 means unsupported. Phase 1 returns a label for the complete answer;
    later work may add sentence or token spans without changing this signature.
    """

    premise = "\n\n".join(str(excerpt) for excerpt in excerpts if str(excerpt).strip())
    if not answer.strip():
        raise ValueError("answer must not be empty")
    if not premise:
        raise ValueError("at least one non-empty excerpt is required")

    model = scorer or _load_scorer(model_name, local_files_only)
    try:
        consistency = _first_score(model.predict([(premise, answer)]))
    except GroundingModelError:
        raise
    except Exception as error:
        raise GroundingModelError("grounding prediction failed") from error
    label = "consistent" if consistency >= consistent_threshold else "hallucinated"
    detail = GroundingLabel(label, consistency, 0, len(answer), answer)
    return GroundingResult(consistency, (detail,), model_name)
