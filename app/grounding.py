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
GROUNDING_REVISION = "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"
FLAN_TOKENIZER_MODEL = "google/flan-t5-base"
FLAN_TOKENIZER_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
CONSISTENT_THRESHOLD = 0.5
GROUNDING_FILES = (
    "config.json",
    "configuration_hhem_v2.py",
    "model.safetensors",
    "modeling_hhem_v2.py",
)
FLAN_TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


class GroundingModelError(UserFacingBackendError):
    """The optional grounding model is unavailable or failed."""

    code = "grounding_model_unavailable"
    message = "The optional grounding check is not available in the local cache."
    action = "Turn off grounding, or run `bash install.sh --enhanced` while online."


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


def _snapshot_download(
    repo_id: str,
    revision: str,
    *,
    local_files_only: bool,
    allow_patterns=None,
) -> str:
    try:
        hub = importlib.import_module("huggingface_hub")
        return str(
            hub.snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=local_files_only,
                allow_patterns=allow_patterns,
            )
        )
    except Exception as error:
        raise GroundingModelError("grounding assets are not cached") from error


def cache_grounding_assets(*, local_files_only: bool = True) -> tuple[str, str]:
    """Cache pinned HHEM custom code/weights and its pinned FLAN tokenizer."""

    hhem_path = _snapshot_download(
        GROUNDING_MODEL,
        GROUNDING_REVISION,
        local_files_only=local_files_only,
        allow_patterns=GROUNDING_FILES,
    )
    flan_path = _snapshot_download(
        FLAN_TOKENIZER_MODEL,
        FLAN_TOKENIZER_REVISION,
        local_files_only=local_files_only,
        allow_patterns=FLAN_TOKENIZER_FILES,
    )
    return hhem_path, flan_path


@lru_cache(maxsize=2)
def _load_scorer(
    model_name: str,
    revision: str,
    flan_revision: str,
    local_files_only: bool,
):
    """Load HHEM once on CPU only when Enhanced grounding is enabled."""

    try:
        module = importlib.import_module("transformers")
        if model_name == GROUNDING_MODEL and revision == GROUNDING_REVISION:
            hhem_path, flan_path = cache_grounding_assets(
                local_files_only=local_files_only,
            )
        else:
            hhem_path = _snapshot_download(
                model_name,
                revision,
                local_files_only=local_files_only,
            )
            flan_path = _snapshot_download(
                FLAN_TOKENIZER_MODEL,
                flan_revision,
                local_files_only=local_files_only,
                allow_patterns=FLAN_TOKENIZER_FILES,
            )
        config = module.AutoConfig.from_pretrained(
            hhem_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config.foundation = flan_path
        declared_model_type = getattr(type(config), "model_type", None)
        if declared_model_type:
            config.model_type = declared_model_type
        model = module.AutoModelForSequenceClassification.from_pretrained(
            hhem_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
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
    model_revision: str = GROUNDING_REVISION,
    flan_revision: str = FLAN_TOKENIZER_REVISION,
    local_files_only: bool = True,
    consistent_threshold: float = CONSISTENT_THRESHOLD,
) -> GroundingResult:
    """Score whether ``answer`` is supported by the retrieved ``excerpts``.

    The result uses HHEM's consistency probability: 1 means fully supported
    and 0 means unsupported. The label spans the complete answer; sentence or
    token spans can be added later without changing this signature.
    """

    premise = "\n\n".join(str(excerpt) for excerpt in excerpts if str(excerpt).strip())
    if not answer.strip():
        raise ValueError("answer must not be empty")
    if not premise:
        raise ValueError("at least one non-empty excerpt is required")

    model = scorer or _load_scorer(
        model_name,
        model_revision,
        flan_revision,
        local_files_only,
    )
    try:
        consistency = _first_score(model.predict([(premise, answer)]))
    except GroundingModelError:
        raise
    except Exception as error:
        raise GroundingModelError("grounding prediction failed") from error
    label = "consistent" if consistency >= consistent_threshold else "hallucinated"
    detail = GroundingLabel(label, consistency, 0, len(answer), answer)
    return GroundingResult(consistency, (detail,), model_name)
