"""How the app asks for answers and reads them back.

`app_messages` lays an app-only format on top of the thesis prompts: HOW-TO answers as numbered steps with one
fenced bash block per command, explanations as plain sentences, `Source: [n]` only when excerpts were given.
shared/prompt.py and engine.build_messages are what eval and KD use; they are read here, never changed, and
nothing under eval/, kd/ or shared/ imports this module.

`parse_answer` is the one reading of a raw answer. The card, the live stream, the chat memory sent back to the
model and a follow-up's "step 2" all come from it, so step 2 on screen is step 2 everywhere.
"""
import re
import textwrap
from dataclasses import dataclass

from app.engine import uses_rag
from shared.prompt import system_for_mode, user_msg

_FORMAT = (
    "Format your reply for a student at a Linux lab PC:\n"
    "- If the question asks how to do something, write one short intro line, then numbered steps (1., 2., 3.), "
    "one action per step. Put each command in its own fenced code block that starts with ```bash, directly "
    "under its step, one command per block.\n"
    "- If the question asks for an explanation or a fact, answer in two to four short sentences. Do not turn it "
    "into steps.\n"
    "- Write any value the student must fill in as <username> or <path/to/file>. Never guess a real value.\n"
    "- No greeting, no summary, no closing remarks."
)
_CITE = "- End with one line, Source: [n], giving the number of each excerpt you used, for example Source: [2]."
_NO_CITE = "- You were given no documents, so do not write a Source line or any [n] reference."
_REASON = (
    "- Start with a line Reasoning: and two to four short bullet lines that begin with '-', never with numbers. "
    "Then write a line Answer: and the reply in the format above."
)
_NO_REASON = "- Do not write a reasoning section."


def app_messages(question, hits, mode, step_note=""):
    """The app's messages for one turn: the frozen thesis system prompt for `mode` plus the app format rules.

    The thesis RAG prompt's "Do not write a step list" keeps reasoning out of +RAG; here that is said as "no
    reasoning section", so a HOW-TO answer can still be numbered steps.
    """
    rag = uses_rag(mode)
    rules = [_FORMAT, _CITE if rag else _NO_CITE, _REASON if mode in ("cot", "full") else _NO_REASON]
    system = system_for_mode(mode) + "\n\n" + "\n".join(rules)
    user = user_msg(question, hits) if rag else f"Question: {question}"
    if step_note:
        user += "\n\n" + step_note
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# tldr-style {{username}} or {{path/to/file}}. Not Jinja/Ansible "{{ item }}" (inner spaces), Go "{{.Name}}",
# GitHub "${{ x }}", tldr option alternatives "{{[-a|--all]}}", or anything with quotes or braces inside.
_PLACEHOLDER = re.compile(r"(?<![$\\{])\{\{([A-Za-z0-9_/~][A-Za-z0-9_./@:~+-]*)\}\}(?!\})")


def normalize_placeholders(text):
    return _PLACEHOLDER.sub(r"<\1>", text or "")


_ANSWER_MARK = re.compile(r"^[ \t]*(?:\*\*|__)?(?:final[ \t]+)?answer(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*",
                          re.IGNORECASE | re.MULTILINE)
_REASON_LABEL = re.compile(r"^\s*(?:\*\*|__)?(?:reasoning|steps|thinking)(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*",
                           re.IGNORECASE)
_FENCE_OPEN = re.compile(r"^([ \t]*)(`{3,}|~{3,})[ \t]*([\w+.-]*)[^`]*$")
_ITEM = re.compile(r"^ ?(\*\*|__)?(\d{1,2})[.)](\*\*|__)?[ \t]+(.*)$")
_STEP_HEAD = re.compile(r"^[ \t]*(?:#{1,4}[ \t]*)?(\*\*|__)?step[ \t]+(\d{1,2})[ \t]*[:.)\u2013-](\*\*|__)?[ \t]*(.*)$",
                        re.IGNORECASE)
_SOURCE = re.compile(r"^[ \t]*(?:\*\*|__)?sources?(?:\*\*|__)?[ \t]*:(?:\*\*|__)?[ \t]*(.*)$", re.IGNORECASE)
_CITATION = re.compile(r"\[(\d{1,2})\]")
_SHELL_LANGS = {"", "bash", "sh", "shell", "console", "terminal", "zsh"}


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    step: int | None = None
    lang: str = ""
    open: bool = False


@dataclass(frozen=True)
class Step:
    number: int
    text: str
    commands: tuple[str, ...]
    shown: str


@dataclass(frozen=True)
class Answer:
    reasoning: str
    blocks: tuple[Block, ...]
    steps: tuple[Step, ...]
    cited: tuple[int, ...]
    dropped: tuple[int, ...]
    source_line: bool
    reasoning_open: bool = False


def split_reasoning(text, streaming=False):
    """(reasoning, answer). Only an explicit `Answer:` line splits them. Without one, nothing is hidden, so a
    procedure is never mistaken for reasoning; the same holds when the commands sit above `Answer:` and none
    below it. While streaming, text that opens with a `Reasoning:` label is reasoning until `Answer:` arrives."""
    body = text or ""
    mark = _ANSWER_MARK.search(body)
    if mark is None:
        if streaming and _REASON_LABEL.match(body):
            return _REASON_LABEL.sub("", body, count=1).strip(), ""
        return "", body.strip()
    head, tail = body[: mark.start()], body[mark.end():]
    reasoning = _REASON_LABEL.sub("", head, count=1).strip()
    if not tail.strip() and not streaming:
        return "", body.strip()
    if "```" in head and "```" not in tail and not streaming:
        return "", body.strip()
    return reasoning, tail.strip()


def _settled(text, streaming):
    """While streaming, hold back a last partial line that may still become a fence, a step number or a label,
    so the view never flips from one element type to another under the reader."""
    if not streaming:
        return text
    head, sep, tail = text.rpartition("\n")
    if not sep:
        head, tail = "", text
    stripped = tail.strip()
    if stripped.startswith(("`", "~")) or (stripped and len(stripped) < 12):
        return head
    return text


def _clean_citations(md, n_hits):
    """Drop [n] that cites no retrieved excerpt: any [n] without retrieval, or past the last excerpt.
    Returns the text, the [n] kept and the [n] dropped."""
    kept, dropped = [], []

    def keep(m):
        n = int(m.group(1))
        if 1 <= n <= n_hits:
            kept.append(n)
            return m.group(0)
        dropped.append(n)
        return ""

    out = []
    for i, part in enumerate(re.split(r"(`[^`\n]*`)", md)):
        out.append(part if i % 2 else _CITATION.sub(keep, part))
    return re.sub(r"[ \t]+([.,;:])", r"\1", "".join(out)), kept, dropped


def _item(line):
    """A top-level numbered step: "1. x", "2) x", "**3.** x", "Step 4: x", "### Step 5 - x". Lines indented
    two spaces or more are nested lists and stay part of their step."""
    m = _ITEM.match(line) or _STEP_HEAD.match(line)
    if m is None:
        return None
    bold, n, closed, text = m.groups()
    text = text.strip()
    if bold and not closed and text.endswith(bold):
        text = text[: -len(bold)].rstrip()
    return int(n), text


def _dedent(lines, indent):
    width = len(indent.expandtabs(4))
    out = []
    for line in lines:
        expanded = line.expandtabs(4)
        lead = len(expanded) - len(expanded.lstrip(" "))
        out.append(expanded[min(lead, width):])
    return "\n".join(out).strip("\n")


def parse_answer(text, mode="", n_hits=0, streaming=False):
    """Read a raw answer into reasoning, display blocks and canonical numbered steps.

    Steps are numbered in the order they appear, whatever numbers the model wrote ("1. 1. 1." included), so the
    shown numbering survives the commands between them. A Source line is lifted out and checked against what
    was retrieved; BASE and +CoT retrieve nothing, so any [n] there is dropped rather than shown.
    """
    raw = normalize_placeholders(text)
    reasoning, body = split_reasoning(raw, streaming=streaming) if mode in ("cot", "full") else ("", raw.strip())
    reasoning_open = streaming and not body and bool(reasoning)
    body = _settled(body, streaming)
    hits = n_hits if uses_rag(mode) else 0
    blocks = []
    md, step_no = [], None
    cited, dropped, source_line = [], [], False

    def flush_md():
        if md and any(line.strip() for line in md):
            chunk, kept, gone = _clean_citations(textwrap.dedent("\n".join(md)).strip("\n"), hits)
            cited.extend(kept)
            dropped.extend(gone)
            if chunk.strip():
                blocks.append(Block("md", chunk, step_no))
        md.clear()

    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_OPEN.match(line)
        if fence:
            flush_md()
            indent, run, lang = fence.groups()
            closer = re.compile(rf"^[ \t]*{re.escape(run[0])}{{{len(run)},}}[ \t]*$")
            j = i + 1
            while j < len(lines) and not closer.match(lines[j]):
                j += 1
            code = _dedent(lines[i + 1:j], indent)
            is_open = j >= len(lines)
            lang = lang.lower()
            if code or is_open:
                blocks.append(Block("code", code, step_no, "bash" if lang in _SHELL_LANGS else lang, is_open))
            i = j + 1
            continue
        source = _SOURCE.match(line)
        if source:
            source_line = True
            for n in (int(x) for x in _CITATION.findall(source.group(1))):
                (cited if 1 <= n <= hits else dropped).append(n)
            i += 1
            continue
        item = _item(line)
        if item:
            flush_md()
            step_no = (step_no or 0) + 1
            md.append(f"{step_no}. {item[1]}")
        else:
            md.append(line)
        i += 1
    flush_md()
    return Answer(
        reasoning=reasoning,
        blocks=tuple(blocks),
        steps=_steps(blocks),
        cited=tuple(dict.fromkeys(cited)),
        dropped=tuple(dict.fromkeys(dropped)),
        source_line=source_line,
        reasoning_open=reasoning_open,
    )


def _block_text(block):
    return f"```{block.lang or 'bash'}\n{block.text}\n```" if block.kind == "code" else block.text


def _steps(blocks):
    steps = []
    for n in dict.fromkeys(b.step for b in blocks if b.step is not None):
        mine = [b for b in blocks if b.step == n]
        prose = "\n".join(b.text for b in mine if b.kind == "md")
        prose = re.sub(rf"^{n}\. ", "", prose, count=1).strip()
        commands = tuple(b.text for b in mine if b.kind == "code" and b.text)
        steps.append(Step(n, prose, commands, "\n".join(_block_text(b) for b in mine)))
    return tuple(steps)


def safe_markdown(md):
    """Markdown for st.markdown with `<` outside code shown as typed, so <username> is never taken for HTML."""
    parts = re.split(r"(`[^`\n]*`)", md or "")
    return "".join(part if i % 2 else part.replace("<", "&lt;") for i, part in enumerate(parts))


def canonical_text(answer):
    """The answer as the reader sees it, without reasoning or the Source line: steps carry their shown numbers
    and every command is a fenced block. This is what the chat memory sends back to the model."""
    return "\n".join(_block_text(b) for b in answer.blocks).strip()


def memory_text(raw, mode=""):
    """Old [n] are dropped too: the next turn numbers its own excerpts from [1]."""
    return canonical_text(parse_answer(raw, mode=mode, n_hits=0))


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10}
_STEP_NUMBER = re.compile(r"\bstep\s*(?:#|no\.?|number)?\s*(\d{1,2})\b", re.IGNORECASE)
_STEP_ORDINAL = re.compile(r"\b(" + "|".join(_ORDINALS) + r"|last)\s+step\b", re.IGNORECASE)


def step_reference(question):
    """The step number a follow-up names ("explain step 2", "the second step", "the last step"), or None.
    "last" comes back as -1."""
    q = question or ""
    m = _STEP_NUMBER.search(q)
    if m:
        return int(m.group(1))
    m = _STEP_ORDINAL.search(q)
    if m:
        word = m.group(1).lower()
        return -1 if word == "last" else _ORDINALS[word]
    return None


def step_note(question, previous_raw, previous_mode=""):
    """What the model is told when a follow-up names a step: the step as the reader saw it, so "step 2" is never
    read as excerpt [2] or a reasoning line. A step the last answer doesn't have is said to be missing."""
    n = step_reference(question)
    if n is None:
        return ""
    steps = parse_answer(previous_raw, mode=previous_mode).steps
    if n == -1 and steps:
        n = steps[-1].number
    step = next((s for s in steps if s.number == n), None)
    if step is None:
        count = f"only {len(steps)} numbered steps" if steps else "no numbered steps"
        return (f'The student refers to step {n if n > 0 else "(last)"} of your previous answer, which had {count}. '
                "Say so briefly instead of guessing.")
    return (
        f"In this question, step {n} means step {n} of your previous answer, shown to the student as:\n"
        f"{step.shown}\n"
        f"Answer about that step. It is not excerpt [{n}] and not a reasoning line."
    )


def step_query(question, previous_raw, previous_mode=""):
    """Extra retrieval words for a follow-up that names a step: that step's text and commands."""
    n = step_reference(question)
    if n is None:
        return ""
    steps = parse_answer(previous_raw, mode=previous_mode).steps
    if n == -1 and steps:
        n = steps[-1].number
    step = next((s for s in steps if s.number == n), None)
    if step is None:
        return ""
    words = " ".join([step.text, *step.commands])
    return " ".join(re.sub(r"[`*_<>#]", " ", words).split())
