"""How the app asks for answers and reads them back.

`app_messages` lays an app-only format on top of the thesis prompts: HOW-TO answers as numbered steps with one
fenced bash block per command, explanations as plain sentences, `Source: [n]` only when excerpts were given.
shared/prompt.py and engine.build_messages are what eval and KD use; they are read here, never changed, and
nothing under eval/, kd/ or shared/ imports this module.

`parse_answer` is the one reading of a raw answer. The card, the live stream, the chat memory sent back to the
model and a follow-up's "step 2" all come from it, so step 2 on screen is step 2 everywhere.
"""
import re
import shlex
import textwrap
from dataclasses import dataclass, replace

from app.engine import MODE_SUFFIX, uses_rag
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
_ITEM = re.compile(r"^ ?(?:[-*+][ \t]+)?(\*\*|__)?(\d{1,2})[.)](\*\*|__)?[ \t]+(.*)$")
_STEP_HEAD = re.compile(r"^[ \t]*(?:#{1,4}[ \t]*)?(\*\*|__)?step[ \t]+(\d{1,2})[ \t]*[:.)\u2013-](\*\*|__)?[ \t]*(.*)$",
                        re.IGNORECASE)
_SOURCE = re.compile(r"^[ \t]*(?:\*\*|__)?sources?(?:\*\*|__)?[ \t]*:(?:\*\*|__)?[ \t]*(.*)$", re.IGNORECASE)
_CITATION = re.compile(r"\[(\d{1,2})\]")
_SHELL_LANGS = {"", "bash", "sh", "shell", "console", "terminal", "zsh"}


_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_COMMAND_NAME = re.compile(r"^[a-z][a-z0-9_.+-]*$")
# Command names an inline `span` may start with to be shown as its own copyable command. Deliberately a fixed list,
# so what gets a copy box doesn't depend on what the machine running the app has installed.
_KNOWN_COMMANDS = frozenset("""
    adduser useradd usermod userdel deluser passwd chpasswd chage groupadd groupdel groupmod gpasswd groups id whoami
    su sudo visudo newgrp chsh chown chmod chgrp umask ls cd pwd mkdir rmdir rm cp mv ln touch cat less more head
    tail grep egrep fgrep find locate updatedb which whereis file stat du df free top htop ps kill killall pkill
    uptime uname hostname hostnamectl date timedatectl tar gzip gunzip zip unzip bzip2 bunzip2 xz unxz 7z apt
    apt-get apt-cache dpkg snap flatpak systemctl journalctl service reboot shutdown poweroff mount umount lsblk
    blkid fdisk parted mkfs fsck ip ifconfig ping traceroute nslookup dig host curl wget ssh ssh-keygen ssh-copy-id
    scp sftp rsync nmcli iwconfig ufw iptables nano vim vi gedit git python3 python pip pip3 make gcc g++ java javac
    node npm echo printf export source env alias history man tldr sort uniq wc cut awk sed tr tee xargs diff cmp
    crontab nohup jobs watch lsusb lspci lscpu xdg-open gsettings update-alternatives localectl setxkbmap lpstat lp
    lpr last lastlog w getent
""".split())
# Names that are also English words: a span starting with one needs an argument that looks like shell syntax.
_WORDLIKE = frozenset("""
    file find less more head tail sort cut watch last w free top kill make host man history date id groups touch
    env echo printf source export alias jobs which diff cat ln cd at lp
""".split())
_SHELLISH = re.compile(r"""^-|[/.<>=$*~|&;'"\d]""")


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    step: int | None = None
    lang: str = ""
    open: bool = False
    # A fence with a blank line before it.
    gap: bool = False
    indented: bool = False
    # A copy box for a command the model wrote inline, as `code`, in a step's sentence.
    derived: bool = False
    # A command written after the last step that the answer doesn't tie to any step.
    loose: bool = False
    # Which line of a fence split into one box per command; 0 for the first or an unsplit fence.
    piece: int = 0


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
    loose: tuple[str, ...] = ()


def split_reasoning(text, streaming=False):
    """(reasoning, answer) for a +CoT or +RAG+CoT reply, which the app asks for as `Reasoning:` then `Answer:`, or
    for any reply that opens with a `Reasoning:` or `Steps:` label.

    The `Answer:` line is the boundary. While streaming, nothing before it is answer: a reply can open with a
    numbered draft ("1. The excerpts show ...") and only then write `Answer:`, so until that line arrives the text
    is held as reasoning and the answer stays empty. A finished reply without `Answer:` hides only the block a
    `Reasoning:` label opens (see _lead_end), and only when more follows it; without the label nothing is hidden,
    so a procedure is never taken for reasoning. The same holds when the commands sit above `Answer:` and none
    below it."""
    body = text or ""
    mark = _ANSWER_MARK.search(body)
    if mark is None:
        if streaming:
            return _REASON_LABEL.sub("", body, count=1).strip(), ""
        if not _REASON_LABEL.match(body):
            return "", body.strip()
        lines = _REASON_LABEL.sub("", body, count=1).split("\n")
        end = _lead_end(lines)
        lead, rest = "\n".join(lines[:end]).strip(), "\n".join(lines[end:]).strip()
        return (lead, rest) if lead and rest else ("", body.strip())
    head, tail = body[: mark.start()], body[mark.end():]
    reasoning = _REASON_LABEL.sub("", head, count=1).strip()
    if not tail.strip() and not streaming:
        return "", body.strip()
    if "```" in head and "```" not in tail and not streaming:
        return "", body.strip()
    return reasoning, tail.strip()


def _settled(text, streaming):
    """While streaming, hold back a last partial line that may still become a fence, a step number or a label,
    so the view never flips from one element type to another under the reader, and an inline `span` until its
    closing backtick arrives."""
    if not streaming:
        return text
    head, sep, tail = text.rpartition("\n")
    if not sep:
        head, tail = "", text
    stripped = tail.strip()
    if stripped.startswith(("`", "~")) or (stripped and len(stripped) < 12):
        return head
    if tail.count("`") % 2:
        return text[: len(text) - len(tail) + tail.rindex("`")].rstrip()
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


def _lead_end(lines):
    """Where the block a `Reasoning:` label opens ends, as Markdown would end it: at a blank line or a fence, or at
    a numbered step after prose or bullets. A lead written as a numbered list ends where the numbering restarts."""
    started, last = False, None
    for i, line in enumerate(lines):
        if not line.strip():
            if started:
                return i
            continue
        if _FENCE_OPEN.match(line):
            return i
        item = _item(line)
        if item:
            if started and (last is None or item[0] <= last):
                return i
            last = item[0]
        started = True
    return len(lines)


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
    # BASE and +RAG are not asked for reasoning, but a reply that opens with the label is still a draft.
    split = mode in ("cot", "full") or _REASON_LABEL.match(raw)
    reasoning, body = split_reasoning(raw, streaming=streaming) if split else ("", raw.strip())
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
            lang = "bash" if lang.lower() in _SHELL_LANGS else lang.lower()
            if code or is_open:
                gap = i > 0 and not lines[i - 1].strip()
                pieces = (_command_lines(code) if lang == "bash" and not is_open else None) or [code]
                blocks.extend(Block("code", piece, step_no, lang, is_open, gap=gap and not k, indented=bool(indent),
                                    piece=k) for k, piece in enumerate(pieces))
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
    if not streaming:
        blocks = _place_trailing_commands(blocks)
    blocks = _lift_inline_commands(blocks, open_step=step_no if streaming else None)
    return Answer(
        reasoning=reasoning,
        blocks=tuple(blocks),
        steps=_steps(blocks),
        cited=tuple(sorted(set(cited))),
        dropped=tuple(sorted(set(dropped))),
        source_line=source_line,
        reasoning_open=reasoning_open,
        loose=tuple(b.text for b in blocks if b.loose),
    )


def _place_trailing_commands(blocks):
    """Fences right after the last step's text belong to it only when the answer's own layout says so: it is the
    only step, earlier steps carry their own fences, or a single fence follows directly or after a line ending in
    ':'. Otherwise (several steps without commands, then the commands in a row) which command goes with which step
    is the model's to say, not ours: they stay in order, marked loose, tied to no step."""
    last = max((b.step for b in blocks if b.step is not None), default=None)
    if last is None:
        return blocks
    head = next(i for i, b in enumerate(blocks) if b.step == last)
    end = head + 1
    while end < len(blocks) and blocks[end].kind == "code" and blocks[end].step == last:
        end += 1
    trailing = blocks[head + 1:end]
    if not trailing:
        return blocks
    earlier = any(b.kind == "code" and b.step is not None and b.step < last for b in blocks)
    leads_in = not trailing[0].gap or blocks[head].text.rstrip().endswith(":")
    fences = sum(1 for b in trailing if b.piece == 0)
    if last == 1 or earlier or (fences == 1 and leads_in):
        return blocks
    after = [replace(b, step=None) if b.step == last else b for b in blocks[end:]]
    return [*blocks[:head + 1], *(replace(b, step=None, loose=True) for b in trailing), *after]


_SHELL_WORDS = frozenset("if then else elif fi for while until do done case esac function select { } [[ ]]".split())


def _command_lines(code):
    """A fence of several commands, one per line, as those lines, so each gets its own copy box. Only when every
    line is a standalone command with a known name: a continuation, pipe or && at a line end, a heredoc, a
    comment, a prompt sign, a shell keyword or an indented line keeps the fence whole. None otherwise."""
    lines = [line for line in code.split("\n") if line.strip()]
    if len(lines) < 2:
        return None
    for line in lines:
        if line != line.strip() or "<<" in line or line.startswith(("#", "|", "&")):
            return None
        if line.endswith(("\\", "|", "&&", "||", "{", "(")):
            return None
        try:
            words = shlex.split(line)
        except ValueError:
            return None
        if not words or words[0] in _SHELL_WORDS:
            return None
        name = words[1] if words[0] == "sudo" and len(words) > 1 else words[0]
        if name not in _KNOWN_COMMANDS or name == "sudo":
            return None
    return lines


def is_inline_command(span):
    """Whether an inline `span` is a whole shell command to offer as its own copy box: it splits as shell words,
    starts with a known command name (after an optional bare sudo) and has an argument or follows sudo. Names alone
    (`adduser`), paths, options, config lines and anything that doesn't parse stay inline."""
    try:
        words = shlex.split(span)
    except ValueError:
        return False
    if len(words) < 2 or span != span.strip():
        return False
    name, args = (words[1], words[2:]) if words[0] == "sudo" else (words[0], words[1:])
    if not _COMMAND_NAME.match(name) or name not in _KNOWN_COMMANDS or name == "sudo":
        return False
    if name in _WORDLIKE:
        return any(_SHELLISH.search(arg) for arg in args)
    return bool(args) or words[0] == "sudo"


def _lift_inline_commands(blocks, open_step=None):
    """A step with no fenced command but a whole command written inline gets that command, byte for byte, as a copy
    box under its text. The sentence is left as written; nothing is lifted from text outside numbered steps.
    `open_step` is the step still streaming: its own fence may be on the way, so it gets no box yet."""
    fenced = {b.step for b in blocks if b.kind == "code" and b.step is not None} | {open_step}
    out = []
    for block in blocks:
        out.append(block)
        if block.kind != "md" or block.step is None or block.step in fenced:
            continue
        for span in _INLINE_CODE.findall(block.text):
            if is_inline_command(span):
                out.append(Block("code", span, block.step, "bash", derived=True))
    return out


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
    """The answer as the model wrote it, normalised, without reasoning or the Source line: steps carry their shown
    numbers and fenced commands stay fenced, a fence split into one box per command whole again. Copy boxes the app
    lifted from inline commands are left out, and loose commands stay after the steps, apart. This is what the chat memory sends back to the model."""
    parts = []
    for block in answer.blocks:
        if block.derived:
            continue
        if block.piece:
            parts[-1] = parts[-1].removesuffix("\n```") + f"\n{block.text}\n```"
            continue
        if block.loose and parts:
            parts.append("")
        parts.append(_block_text(block))
    return "\n".join(parts).strip()


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


def _named_step(question, raw, mode):
    """(n, step, answer) for the step a follow-up names, step None when that answer doesn't have it; None when the
    question names no step."""
    n = step_reference(question)
    if n is None:
        return None
    answer = parse_answer(raw or "", mode=mode)
    if n == -1 and answer.steps:
        n = answer.steps[-1].number
    return n, next((s for s in answer.steps if s.number == n), None), answer


def _step_shown(n, step, answer):
    """Step n as the reader saw it, or that the answer doesn't have it, plus any commands the answer listed after
    its steps without tying them to one."""
    if step is None:
        count = f"only {len(answer.steps)} numbered steps" if answer.steps else "no numbered steps"
        text = f"it had {count}, so there is no step {n if n > 0 else '(last)'}."
    else:
        text = f"step {n} was shown as:\n{step.shown}"
    if answer.loose:
        fences = "\n".join(f"```bash\n{c}\n```" for c in answer.loose)
        text += ("\nAfter its steps it also listed these commands without saying which step each belongs to:\n"
                 + fences)
    return text


_RULES = ("Explain step {n} itself, as worded above, in two to four sentences, without listing the other steps "
          "again. If you show a command, copy it exactly as shown. Commands listed after the steps are there only "
          "so you can tell whether one of them does what step {n} says; if none clearly does, say so instead of "
          "guessing. Step {n} is not excerpt [{n}] and not a reasoning line.")


def step_note(question, previous_raw, previous_mode=""):
    """What the model is told when a follow-up names a step: the step as the reader saw it, so "step 2" is never
    read as excerpt [2] or a reasoning line. A step the last answer doesn't have is said to be missing."""
    named = _named_step(question, previous_raw, previous_mode)
    if named is None:
        return ""
    n, step, answer = named
    if step is None and not answer.loose:
        return (f"The student refers to step {n if n > 0 else '(last)'} of your previous answer, but "
                f"{_step_shown(n, step, answer)} Say so briefly instead of guessing.")
    return (f"In this question, step {n} means step {n} of your previous answer. In that answer, "
            f"{_step_shown(n, step, answer)}\n{_RULES.format(n=n)}")


def side_label(side, mode, model=""):
    """How a compare side is named to the model: "Left answer (qwen3.5:2b · +CoT)"."""
    config = " · ".join(part for part in (model, MODE_SUFFIX.get(mode) or "BASE") if part)
    return f"{side} answer ({config})"


def compare_step_note(question, sides):
    """For a single follow-up after a compare: step n of each side, labelled, since the question doesn't say which
    answer it means. `sides` is ((label, raw or None, mode), ...); None is a side that never answered."""
    n = step_reference(question)
    if n is None:
        return ""
    parts = []
    for label, raw, mode in sides:
        if raw is None:
            parts.append(f"{label}: no answer, it was stopped or failed.")
            continue
        number, step, answer = _named_step(question, raw, mode)
        parts.append(f"{label}: {_step_shown(number, step, answer)}")
    shown = "\n\n".join(parts)
    return (f"In this question, step {n} could mean step {n} of either of the two answers the student saw side by "
            f"side. Answer for each separately, starting with its label.\n\n{shown}\n\n{_RULES.format(n=n)}")


def step_query(question, previous_raw, previous_mode=""):
    """Extra retrieval words for a follow-up that names a step: that step's text and commands."""
    named = _named_step(question, previous_raw, previous_mode)
    if named is None or named[1] is None:
        return ""
    step = named[1]
    words = " ".join([step.text, *step.commands])
    return " ".join(re.sub(r"[`*_<>#]", " ", words).split())
