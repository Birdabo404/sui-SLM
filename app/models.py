"""Which Ollama models the app can offer, and whether this machine has them.

The picker lists the default base model (OLLAMA_MODEL), the matched Q4_K_M base and the KD student that
kd/colab_distill.ipynb exports. Nothing here downloads, builds or substitutes a model: a missing one is shown as
missing, with the command that creates it.
"""
from dataclasses import dataclass

import requests

from app.engine import MODE_LABELS, THESIS_CONFIGS

MATCHED_BASE = "thesis-base-q4km"
STUDENT = "thesis-student"


@dataclass(frozen=True)
class ModelOption:
    key: str
    name: str
    role: str
    create: str


def model_options(base_model, distilled_model=""):
    """Default base first, then the matched base and the student; OLLAMA_DISTILLED_MODEL renames the student."""
    student = (distilled_model or "").strip() or STUDENT
    options = [ModelOption("base", base_model, "default base", f"ollama pull {base_model}")]
    if MATCHED_BASE != base_model:
        options.append(
            ModelOption("base", MATCHED_BASE, "Q4_K_M base", f"ollama create {MATCHED_BASE} -f Modelfile-base")
        )
    if student not in (o.name for o in options):
        options.append(ModelOption("distilled", student, "KD student", f"ollama create {student} -f Modelfile"))
    return tuple(options)


def option_named(options, name):
    return next((o for o in options if o.name == name), options[0])


def modes_for(option):
    """The thesis configs this model runs: every mode for a base model, +RAG+CoT only for the student."""
    return tuple(mode for key, mode in THESIS_CONFIGS if key == option.key and mode in MODE_LABELS)


def tag_names(payload):
    names = set()
    for item in (payload or {}).get("models") or []:
        for field in ("name", "model"):
            value = str(item.get(field) or "").strip()
            if value:
                names.add(value)
    return frozenset(names)


def is_installed(name, names):
    """Ollama lists an untagged name as name:latest."""
    if names is None:
        return None
    return name in names or (":" not in name and f"{name}:latest" in names)


def installed_names(host, timeout=1.0):
    """Model names /api/tags lists, or None when Ollama can't be asked."""
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        if not r.ok:
            return None
        return tag_names(r.json())
    except (requests.RequestException, ValueError):
        return None


def missing_text(option):
    return (
        f"{option.name} isn't installed in Ollama on this machine. "
        f"Create it with `{option.create}`"
        + (" from the KD export (kd/README.md)" if option.create.startswith("ollama create") else "")
        + ", then ask again."
    )


@dataclass(frozen=True)
class ComparePick:
    option: ModelOption
    mode: str

    @property
    def id(self):
        return f"{self.option.name}|{self.mode}"


def compare_picks(options):
    return tuple(ComparePick(o, mode) for o in options for mode in modes_for(o))


def default_compare_ids(options, distilled_configured):
    """Unchanged default: base model BASE vs +RAG+CoT, or base +RAG+CoT vs the student when one is configured."""
    base = options[0].name
    student = next((o.name for o in options if o.key == "distilled"), None)
    if distilled_configured and student:
        return f"{base}|full", f"{student}|full"
    return f"{base}|base", f"{base}|full"


def pair_compare_ids(left, right, options, distilled_configured):
    ids = [p.id for p in compare_picks(options)]
    default_left, default_right = default_compare_ids(options, distilled_configured)
    if left not in ids:
        left = default_left
    if right not in ids or right == left:
        right = default_right if default_right != left else next(i for i in ids if i != left)
    return left, right


def pick_by_id(options, pick_id):
    return next(p for p in compare_picks(options) if p.id == pick_id)
