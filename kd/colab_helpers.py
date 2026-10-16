"""Path and file checks used by kd/colab_distill.ipynb. No Colab imports."""
from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from shared.chatml import (
    ASSISTANT_PREFIX,
    CHATML_INSTRUCTION,
    CHATML_RESPONSE,
    IM_END,
    render_chatml,
)
from shared.paths import INDEX_NAME
from shared.prompt import SYSTEM

REPO_URL = "https://github.com/Birdabo404/sui-SLM.git"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/thesis-kd"
STUDENT_HF = "unsloth/Qwen3.5-2B"
TEACHER_9B = "qwen3.5:9b"
TEACHER_4B = "qwen3.5:4b"
OLLAMA_NAME = "thesis-student"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_CHAT_TEMPLATE = (
    "{{ if .System }}"
    "<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}{{ range .Messages }}{{ if eq .Role \"user\" }}"
    "<|im_start|>user\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ else if eq .Role \"assistant\" }}"
    f"{ASSISTANT_PREFIX}"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}{{ end }}"
    f"{ASSISTANT_PREFIX}"
)
HELDOUT_MISSING = (
    "Held-out questions file is missing: {path}\n"
    "Write 150-200 questions as JSONL with fields "
    "id, question, answerable, gold_answer, source.\n"
    "Upload that file to Drive as {hint} or set HELDOUT to its path, "
    "then re-run this cell.\n"
    "Do not use eval/test.jsonl. That file is a format placeholder, "
    "not the thesis test set."
)
SEEDS_MISSING = (
    "Seed chunks are missing: {seeds}\n"
    "On the laptop, after rag/{index}/ exists:\n"
    "  python kd/make_seed_chunks.py "
    "--chunks rag/{index}/chunks.jsonl --out kd/seed_chunks.jsonl\n"
    "Copy seed_chunks.jsonl or chunks.jsonl to {work}."
)


def teacher_ollama_name(use_4b):
    return TEACHER_4B if use_4b else TEACHER_9B


def count_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                n += 1
    return n


def resolve_heldout(candidates, work_dir):
    """Return the first existing held-out JSONL, or raise SystemExit."""
    checked = []
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        checked.append(str(p))
        if p.is_file():
            return p
    hint = Path(work_dir) / "your-heldout.jsonl"
    raise SystemExit(HELDOUT_MISSING.format(path="; ".join(checked) or hint, hint=hint))


def resolve_chunks(work_dir, repo_dir):
    work_dir = Path(work_dir)
    repo_dir = Path(repo_dir)
    for p in (
        work_dir / "chunks.jsonl",
        repo_dir / "rag" / INDEX_NAME / "chunks.jsonl",
    ):
        if p.is_file():
            return p
    return None


def resolve_seeds(work_dir, repo_dir):
    work_dir = Path(work_dir)
    repo_dir = Path(repo_dir)
    for p in (work_dir / "seed_chunks.jsonl", repo_dir / "kd" / "seed_chunks.jsonl"):
        if p.is_file() and count_jsonl(p) > 0:
            return p
    return None


def require_seeds(work_dir, repo_dir):
    seeds = resolve_seeds(work_dir, repo_dir)
    if seeds is not None:
        return seeds
    raise SystemExit(SEEDS_MISSING.format(
        seeds=Path(work_dir) / "seed_chunks.jsonl",
        work=work_dir,
        index=INDEX_NAME,
    ))


def latest_sft_checkpoint(out_dir):
    """Highest-numbered sft-out/checkpoint-* directory, or None."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    ckpts = [p for p in out_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None

    def step(p):
        tail = p.name.split("-", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return -1

    return max(ckpts, key=step)


def select_q4km_gguf(directory):
    """Return the *Q4_K_M*.gguf file. Stop if it is missing. Never invent a name."""
    directory = Path(directory)
    matches = sorted(
        {
            *directory.glob("*Q4_K_M*.gguf"),
            *directory.glob("*q4_k_m*.gguf"),
        }
    )
    if not matches:
        raise SystemExit(
            f"No *Q4_K_M*.gguf in {directory}. "
            "Re-run the GGUF export cell. Refusing to guess a filename."
        )
    return matches[0]


def ollama_tags_ok(url=OLLAMA_TAGS_URL, timeout=2):
    try:
        urlopen(url, timeout=timeout)
        return True
    except (URLError, OSError, TimeoutError):
        return False


def modelfile_text(gguf_filename, system=None, temperature=0, num_ctx=4096):
    system = SYSTEM if system is None else system
    return (
        f"FROM ./{Path(gguf_filename).name}\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        f'PARAMETER stop "{IM_END}"\n'
        f'TEMPLATE """{OLLAMA_CHAT_TEMPLATE}"""\n'
        f'SYSTEM """{system}"""\n'
    )


def ollama_create_cmd(model_name=OLLAMA_NAME):
    return f"ollama create {model_name} -f Modelfile"


def write_modelfile(dest, gguf_filename, system=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(modelfile_text(gguf_filename, system=system), encoding="utf-8")
    return dest
