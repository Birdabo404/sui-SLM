"""Which commands and flags in an answer never appear in the excerpts it was given. Pure Python, no model.

The answer side reads fenced shell blocks and inline code of two or more words. Each line is split the way a shell
would (quotes kept together, pipes, `;`, `&&`, `$(...)` and redirections as separators), and each part gives one
command: its first word after `sudo` and similar wrappers, environment assignments and placeholders. Its flags are
the words starting with `-` before a `--`. Single-word inline code is left alone: it is as often a file, a user or
a value as a command.

The excerpt side is lenient on purpose, to keep false alarms rare: a command counts as present when it appears as a
word anywhere in an excerpt, and a flag when it appears in an excerpt that mentions that command. Short flags are
compared one letter at a time, so `-xzf` is present when `-x`, `-z` and `-f` are, and tar's dashless `xzf` counts.

A warning means "not in what the model was shown", not "wrong".
"""
import re
import shlex
from dataclasses import dataclass, field

SHELL_LANGS = frozenset({"", "bash", "sh", "shell", "zsh", "console", "terminal", "shell-session", "sh-session",
                         "shellsession"})
# Shell syntax and builtins: always there, never in a lab page, so checking them would only raise false alarms.
BUILTINS = frozenset({".", ":", "[", "[[", "]]", "alias", "bg", "break", "builtin", "case", "cd", "command",
                      "continue", "declare", "dirs", "disown", "do", "done", "echo", "elif", "else", "esac", "eval",
                      "exec", "exit", "export", "false", "fc", "fg", "fi", "for", "function", "getopts", "hash",
                      "help", "history", "if", "in", "jobs", "kill", "let", "local", "logout", "popd", "printf",
                      "pushd", "pwd", "read", "readonly", "return", "select", "set", "shift", "shopt", "source",
                      "suspend", "test", "then", "time", "times", "trap", "true", "type", "typeset", "ulimit",
                      "umask", "unalias", "unset", "until", "wait", "while", "bash", "sh", "zsh", "dash"})
# Run the command after them: the wrapper itself is not what the answer is about.
PREFIXES = frozenset({"sudo", "doas", "pkexec", "time", "command", "builtin", "exec", "env", "nohup", "nice",
                      "ionice", "stdbuf", "unbuffer", "chronic"})
# Run a command named later on the line, and are commands worth checking themselves.
RUNNERS = frozenset({"xargs", "watch", "timeout", "strace", "ltrace", "flock", "parallel", "chroot", "su", "runuser",
                     "setsid", "taskset", "systemd-run"})
SUDO_ARG_FLAGS = frozenset({"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-T", "-U", "--user", "--group",
                            "--host", "--prompt", "--chdir", "--role", "--type", "--command-timeout",
                            "--other-user", "--close-from"})
# Spell long options with one dash, so `-name` is one flag, not -n -a -m -e.
SINGLE_DASH_LONG = frozenset({"find", "java", "javac", "gcc", "g++", "cc", "clang", "clang++", "ld", "ffmpeg",
                              "ffprobe", "ffplay", "xrandr", "convert", "magick", "mogrify", "identify", "montage",
                              "ip", "tc", "bridge", "openssl", "xset", "xterm", "qemu-img", "go", "dig"})
# Take a dashless bundle of short flags as their first argument: `tar xzf`, `ps aux`, `ar rcs`.
BUNDLE_FIRST = frozenset({"tar", "ps", "ar"})
# Flags whose next word is a command: `find . -exec rm {} \;`.
EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})

# Plumbing after a pipe (`dpkg -l | grep nginx`): checked when they lead a line, not when they only filter output.
PIPE_FILTERS = frozenset({"grep", "egrep", "fgrep", "head", "tail", "less", "more", "sort", "uniq", "wc", "cut", "tr",
                          "awk", "sed", "tee", "column", "nl", "cat", "fmt", "fold", "rev", "jq"})
SEPARATORS = frozenset({"|", "||", "&&", ";", ";;", "&", "(", ")", "|&", "{", "}", "!"})
REDIRECTS = frozenset({">", ">>", "<", "<<", "<<<", "&>", "&>>", ">&", "<&", ">|", "<>"})
PLACEHOLDER = "\x00"

_FENCE = re.compile(r"^[ \t]*(```+|~~~+)[ \t]*([\w+-]*)[^\n]*\n(.*?)^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)
_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_PROMPT = re.compile(r"^(?:[\w.-]+@[\w.-]+(?::[^\s$#]*)?\s*)?\$\s+")
_ROOT_PROMPT = re.compile(r"^(?:[\w.-]+@[\w.-]+(?::[^\s$#]*)?\s*)?[$#]\s+")
_ANGLE = re.compile(r"<[A-Za-z][^<>\s|;&]*>")
_TLDR_ALTS = re.compile(r"\{\{\[([^\]{}]+)\]\}\}")
_TLDR = re.compile(r"\{\{.*?\}\}")
_SQUARE = re.compile(r"\[(?:-{1,2}[\w-]+|options?|OPTIONS?|args?|ARGS?|flags?|FILE|file)[^\]]*\]")
_COMMAND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_WORD = re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._+-]*")
_FLAG_IN_TEXT = re.compile(r"(?<![\w/.\\-])(--?[A-Za-z][\w-]*)")
_SHORT_TAIL_VALUE = re.compile(r"^(-[A-Za-z]+?)\d+$")
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")


@dataclass(frozen=True)
class Use:
    """One command as the answer runs it, with the flags it passes."""

    command: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result:
    commands: tuple[str, ...]  # every command checked, in order
    missing_commands: tuple[str, ...]
    missing_flags: tuple[tuple[str, str], ...]  # (command, flag)
    flags_checked: int = 0

    @property
    def missing(self):
        """What to show, in order: `cmd` for a command, `cmd -x` for a flag."""
        return tuple(self.missing_commands) + tuple(f"{cmd} {flag}" for cmd, flag in self.missing_flags)

    @property
    def ok(self):
        return not self.missing_commands and not self.missing_flags

    def as_dict(self):
        return {"commands": list(self.commands), "missing": list(self.missing), "flags_checked": self.flags_checked}


# ---------- reading command lines ----------

def _normalize(line):
    line = _TLDR_ALTS.sub(lambda m: " ".join(m.group(1).split("|")), line)
    line = _TLDR.sub(PLACEHOLDER, line)
    line = _ANGLE.sub(PLACEHOLDER, line)
    line = _SQUARE.sub(" ", line)
    return line.replace("...", " ").replace("…", " ")


def _tokens(line):
    """Shell words and operators. Quoted words come back as one token marked quoted, so `grep "-v"`'s pattern is
    never read as a flag."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars="();<>|&")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        out = list(lexer)
    except ValueError:
        # An unbalanced quote: drop quoted spans and split on spaces, which is right for everything else.
        stripped = re.sub(r"(['\"]).*?(\1|$)", " " + PLACEHOLDER + " ", line)
        out = re.findall(r"\|\||&&|[|;&()<>]+|[^\s|;&()<>]+", stripped)
    return out


def _quoted_words(line):
    return {m.group(2) for m in re.finditer(r"(['\"])(.*?)\1", line)}


def _segments(tokens):
    """Split a line's tokens into simple commands, dropping redirection targets and `$(`. Yields (words, piped):
    piped when the command reads the previous one's output."""
    part, skip, piped = [], False, False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token in SEPARATORS or token == "$":
            if part:
                yield part, piped
            part = []
            piped = token in ("|", "|&")
            continue
        if token in REDIRECTS or (token[:1] in "<>&" and set(token) <= set("<>&|")):
            skip = True
            continue
        part.append(token)
    if part:
        yield part, piped


def _flag_word(token):
    return token.startswith("-") and len(token) > 1 and token != "--" and not re.fullmatch(r"-+\d[\w.]*", token)


def _uses(segment, quoted=frozenset(), piped=False):
    """The commands in one simple command: the command after its prefixes, and for runners like xargs both the
    runner and the command it runs. A filter reading a pipe gives nothing."""
    words = list(segment)
    out = []
    while words:
        while words and _ASSIGN.match(words[0]):
            words.pop(0)
        if not words:
            break
        head = words.pop(0)
        if head in quoted or not _COMMAND.match(head) or head.isdigit():
            break
        if head in PREFIXES:
            words = _skip_prefix_args(head, words)
            continue
        if head in BUILTINS or (piped and head in PIPE_FILTERS):
            break
        flags, rest = _flags_until_command(head, words, quoted)
        out.append(Use(head, tuple(flags)))
        if not rest:
            break
        words, piped = rest, False
    return out


def _skip_prefix_args(prefix, words):
    """The words after `sudo -u bob`, `nice -n 10`, `env A=1` and the like: where the real command starts."""
    words = list(words)
    while words:
        word = words[0]
        if word == "--":
            words.pop(0)
            break
        if _flag_word(word):
            words.pop(0)
            if words and _prefix_value(prefix, word, words[0]):
                words.pop(0)
            continue
        if _ASSIGN.match(word):
            words.pop(0)
            continue
        break
    return words


def _prefix_value(prefix, flag, word):
    """True when word is flag's value, as in `sudo -u bob` or `nice -n 10`."""
    if prefix in ("sudo", "doas"):
        return flag in SUDO_ARG_FLAGS
    return prefix in ("nice", "ionice", "stdbuf") and len(flag) == 2 and not _flag_word(word)


def _flags_until_command(head, words, quoted):
    """head's flags, and for a runner the words from the command it runs onward."""
    flags, rest = [], []
    for i, word in enumerate(words):
        if word == "--":
            rest = words[i + 1:] if head in RUNNERS else []
            break
        if word in quoted:
            continue
        if _flag_word(word):
            flags.append(word)
            if word in EXEC_FLAGS:
                rest = words[i + 1:]
                break
            continue
        if head in RUNNERS and re.match(r"^[A-Za-z][A-Za-z0-9._+-]*$", word) and not (i and _takes_value(words[i - 1])):
            rest = words[i:]
            break
    return flags, rest


def _takes_value(word):
    return _flag_word(word) and len(word) == 2


def _lines(text, lang):
    """Command lines from one fenced block: prompts stripped, continuations joined, heredoc bodies, comments and
    console output skipped."""
    raw = text.splitlines()
    console = lang in ("console", "shell-session", "sh-session", "shellsession", "terminal")
    if not console and any(_PROMPT.match(l) for l in raw if l.strip()) and not all(
            _PROMPT.match(l) for l in raw if l.strip()):
        console = True
    prompt = _ROOT_PROMPT if lang in ("console", "shell-session", "sh-session", "shellsession", "terminal") else _PROMPT
    out, buf, heredoc = [], "", None
    for line in raw:
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        stripped = line.strip()
        if not buf:
            prompted = prompt.match(stripped)
            if console and not prompted:
                continue
            if prompted:
                stripped = stripped[prompted.end():]
            if not stripped or stripped.startswith("#"):
                continue
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        full = buf + stripped
        buf = ""
        full = re.sub(r"\s#\s.*$", "", full)
        m = _HEREDOC.search(full)
        if m:
            heredoc = m.group(1)
            full = full[: m.start()]
        out.append(full)
    if buf:
        out.append(buf)
    return out


def _line_uses(line):
    normal = _normalize(line)
    quoted = _quoted_words(normal)
    uses = []
    for segment, piped in _segments(_tokens(normal)):
        uses.extend(_uses(segment, quoted, piped))
    return uses


def answer_uses(answer):
    """Every command the answer runs, in order, from shell code blocks and multi-word inline code."""
    uses = []
    rest = answer or ""
    for m in _FENCE.finditer(rest):
        lang = m.group(2).lower()
        if lang not in SHELL_LANGS:
            continue
        for line in _lines(m.group(3), lang):
            uses.extend(_line_uses(line))
    prose = _FENCE.sub(" ", rest)
    for m in _INLINE.finditer(prose):
        code = m.group(1).strip()
        code = _PROMPT.sub("", code)
        if len(code.split()) < 2:
            continue
        first = code.split()[0]
        if not re.match(r"^[a-z0-9][a-z0-9._+-]*$", first) and first not in PREFIXES:
            continue
        uses.extend(_line_uses(code))
    return uses


# ---------- what the excerpts show ----------

@dataclass
class _Page:
    words: set = field(default_factory=set)
    flags: set = field(default_factory=set)
    letters: set = field(default_factory=set)  # short-flag letters, from -xvf and tar's dashless xvf


def _expand(flag):
    """Short-flag letters in -xzf (or -p2222's -p), none for a long flag."""
    if flag.startswith("--"):
        return set()
    m = _SHORT_TAIL_VALUE.match(flag)
    body = (m.group(1) if m else flag)[1:]
    return {c for c in body if c.isalpha()}


def _base(flag):
    return flag.split("=", 1)[0]


def excerpts_seen(excerpts):
    """Per excerpt: its words, the flags anywhere in it (tldr's {{[-R|--recursive]}} included) and their letters."""
    pages = []
    for text in excerpts or ():
        text = str(text or "")
        page = _Page(set(_WORD.findall(text)), {_base(f) for f in _FLAG_IN_TEXT.findall(text)})
        for flag in page.flags:
            page.letters |= _expand(flag)
        codes = [m.group(1) for m in _INLINE.finditer(text)]
        codes += [line for m in _FENCE.finditer(text) for line in _lines(m.group(3), m.group(2).lower())]
        for code in codes:
            for segment, _ in _segments(_tokens(_normalize(code))):
                page.letters |= _bsd_bundle(segment)
        pages.append(page)
    return pages


def _bsd_bundle(segment):
    words = [w for w in segment if w not in PREFIXES]
    if len(words) > 1 and words[0] in BUNDLE_FIRST and re.fullmatch(r"[A-Za-z]{1,8}", words[1]):
        return set(words[1])
    return set()


# ---------- the check ----------

def check(answer, excerpts):
    """The commands and flags the answer uses that its excerpts never show."""
    pages = excerpts_seen(excerpts)
    words = set().union(*(p.words for p in pages)) if pages else set()
    commands, missing_commands, missing_flags, checked = [], [], [], 0
    for use in answer_uses(answer):
        if use.command not in commands:
            commands.append(use.command)
        if use.command not in words:
            if use.command not in missing_commands:
                missing_commands.append(use.command)
            continue
        for flag in use.flags:
            checked += 1
            for item in _unseen(use.command, _base(flag), pages):
                if (use.command, item) not in missing_flags:
                    missing_flags.append((use.command, item))
    return Result(tuple(commands), tuple(missing_commands), tuple(missing_flags), checked)


def _unseen(command, flag, pages):
    """The parts of one flag that no excerpt mentioning command shows: nothing, the flag, or the letters of a
    bundle (`-j` of `-xjf`)."""
    about = [p for p in pages if command in p.words]
    flags = set().union(*(p.flags for p in about)) if about else set()
    letters = set().union(*(p.letters for p in about)) if about else set()
    if flag in flags:
        return []
    if flag.startswith("--") or (command in SINGLE_DASH_LONG and len(flag) > 2):
        return [flag]
    m = _SHORT_TAIL_VALUE.match(flag)
    body = (m.group(1) if m else flag)[1:]
    missing = [c for c in dict.fromkeys(body) if c.isalpha() and c not in letters]
    if not missing:
        return []
    return [flag] if len(body) == 1 else [f"-{c}" for c in missing]
