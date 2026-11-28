"""The system check: everything Polaris needs from this machine, measured one line at a time.

Every probe is local. It reads platform data, /proc, /sys and psutil, or asks Ollama on OLLAMA_HOST, which is this
machine unless someone points it elsewhere. Nothing here makes an internet call. It reuses the app's own checks
(ollama_reachable, ollama_version, warm_model, index_ok, index_chunk_count, hf_cached, network_online), so the
system check and the answer path can't disagree about what "ready" means.
"""
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from html import escape
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import psutil
import requests
import streamlit

from app import engine, issues, polaris

PASS, WARN, FAIL = "pass", "warn", "fail"
MARKS = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}
SUPPORTED_PYTHON = ((3, 12), (3, 14))  # install.sh's range
PINNED_STREAMLIT = "1.45.0"  # requirements.txt
LOW_RAM_GB, LOW_DISK_GB, NO_DISK_GB = 2.0, 5.0, 1.0
SLOW_WARMUP_S = 10.0
# Each line holds its loading state at least this long, so the sequence reads as one check after another.
PACE_S = 0.12
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "ANONYMIZED_TELEMETRY": "False"}


@dataclass(frozen=True)
class Probe:
    label: str
    status: str
    value: str
    fix: str = ""
    ms: float = 0.0


def _gb(n):
    return f"{n / 1024 ** 3:.1f} GB"


def os_probe(os_release="/etc/os-release"):
    name = ""
    try:
        for line in Path(os_release).read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                name = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    name = name or f"{platform.system()} {platform.release()}".strip()
    linux = platform.system() == "Linux"
    return Probe("OS", PASS if linux else WARN, f"{name} · {platform.machine()}",
                 "" if linux else "Polaris is built and tested on Ubuntu")


def kernel_probe():
    return Probe("Kernel", PASS, f"{platform.system()} {platform.release()}")


def cpu_probe(cpuinfo="/proc/cpuinfo"):
    model = ""
    try:
        for line in Path(cpuinfo).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware", "processor\t: ")) and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value and not value.isdigit():
                    model = value
                    break
    except OSError:
        pass
    model = " ".join((model or platform.processor() or "unknown CPU").split())
    cores, threads = psutil.cpu_count(logical=False), psutil.cpu_count()
    counts = f"{cores} cores / {threads} threads" if cores and cores != threads else f"{threads or '?'} cores"
    return Probe("CPU", PASS, f"{model} · {counts}")


def ram_probe(memory=None):
    vm = memory or psutil.virtual_memory()
    low = vm.available / 1024 ** 3 < LOW_RAM_GB
    return Probe("RAM", WARN if low else PASS, f"{_gb(vm.total)} total · {_gb(vm.available)} free",
                 "close other apps: Ollama needs about 2 GB free to load a model" if low else "")


_GPU_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}


def gpu_probe(nvidia="/proc/driver/nvidia/gpus", drm="/sys/class/drm"):
    """A GPU is optional: the lab ThinkPads run every model on the CPU, so none found still passes."""
    found = []
    try:
        for info in sorted(Path(nvidia).glob("*/information")):
            for line in info.read_text(encoding="utf-8").splitlines():
                if line.startswith("Model:"):
                    found.append(line.split(":", 1)[1].strip())
    except OSError:
        pass
    if not found:
        for card in sorted(Path(drm).glob("card[0-9]")):
            try:
                vendor = (card / "device" / "vendor").read_text().strip()
                driver = next((line.split("=", 1)[1] for line in (card / "device" / "uevent").read_text().splitlines()
                               if line.startswith("DRIVER=")), "")
            except OSError:
                continue
            found.append(f"{_GPU_VENDORS.get(vendor, vendor)} ({driver})" if driver else _GPU_VENDORS.get(vendor, vendor))
    if not found:
        return Probe("GPU", PASS, "none · models run on the CPU")
    return Probe("GPU", PASS, " · ".join(dict.fromkeys(found)))


def disk_probe(path=engine.ROOT):
    usage = shutil.disk_usage(path)
    free = usage.free / 1024 ** 3
    status = FAIL if free < NO_DISK_GB else WARN if free < LOW_DISK_GB else PASS
    fix = "free some space: chat history and Ollama's model files live on this disk" if status != PASS else ""
    return Probe("Disk", status, f"{_gb(usage.free)} free of {_gb(usage.total)}", fix)


def python_probe(version=None):
    version = tuple((version or sys.version_info)[:3])
    low, high = SUPPORTED_PYTHON
    ok = low <= version[:2] <= high
    return Probe("Python", PASS if ok else WARN, ".".join(map(str, version)),
                 "" if ok else f"install.sh supports Python {low[0]}.{low[1]} to {high[0]}.{high[1]}")


def streamlit_probe(version=None):
    version = streamlit.__version__ if version is None else version
    ok = version == PINNED_STREAMLIT
    return Probe("Streamlit", PASS if ok else WARN, version,
                 "" if ok else f"requirements.txt pins {PINNED_STREAMLIT}: bash install.sh")


def ollama_probe(host=engine.OLLAMA_HOST):
    if not engine.ollama_reachable(host):
        return Probe("Ollama", FAIL, f"not running at {host}", "ollama serve")
    version = engine.ollama_version(host)
    return Probe("Ollama", PASS, f"{version or 'version unknown'} · {host}")


def ollama_tags(host=engine.OLLAMA_HOST, timeout=1.5):
    """/api/tags' model list, or None when Ollama can't be asked."""
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        return (r.json() or {}).get("models") or [] if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def _tag_for(name, tags):
    """Ollama lists an untagged name as name:latest."""
    wanted = {name} if ":" in name else {name, f"{name}:latest"}
    return next((t for t in tags if wanted & {t.get("name"), t.get("model")}), None)


def model_probes(options, tags):
    """One line per model the picker offers. The default base is required; the others only limit what can run."""
    probes = []
    for i, option in enumerate(options):
        missing = FAIL if i == 0 else WARN
        if tags is None:
            probes.append(Probe("Model", missing, f"{option.name} · can't check until Ollama runs"))
            continue
        item = _tag_for(option.name, tags)
        if item is None:
            probes.append(Probe("Model", missing, f"{option.name} · not installed · {option.role}", option.create))
            continue
        details = item.get("details") or {}
        parts = (option.name, _gb(int(item.get("size") or 0)), details.get("parameter_size"),
                 details.get("quantization_level"), option.role)
        probes.append(Probe("Model", PASS, " · ".join(filter(None, parts))))
    return probes


def index_probe(index_dir=engine.DEFAULT_INDEX):
    if not engine.index_ok(index_dir):
        return Probe("RAG index", FAIL, f"missing · {Path(index_dir).name}", issues.INDEX_BUILD)
    chunks = polaris.index_chunk_count(index_dir)
    if not chunks:
        return Probe("RAG index", FAIL, f"empty · {Path(index_dir).name}", issues.INDEX_BUILD)
    return Probe("RAG index", PASS, f"{chunks:,} chunks · {Path(index_dir).name}")


def embed_probe():
    cached = issues.hf_cached()
    return Probe("Embedding model", PASS if cached else FAIL, "MiniLM-L6-v2 cached locally" if cached else "not cached",
                 "" if cached else "bash install.sh")


def network_probe(online=None):
    online = polaris.network_online() if online is None else online
    return Probe("Network", PASS, "online · not used by the app" if online else "OFFLINE · no route out, none needed")


def _loopback(host):
    name = urlparse(host if "://" in host else f"http://{host}").hostname or ""
    if name == "localhost":
        return True
    try:
        return ip_address(name).is_loopback
    except ValueError:
        return False


def outbound_probe(host=engine.OLLAMA_HOST, environ=None, usage_stats=None):
    """What would make a call off this machine: Ollama on another host, a Hugging Face download, telemetry."""
    env = os.environ if environ is None else environ
    if usage_stats is None:
        usage_stats = streamlit.get_option("browser.gatherUsageStats")
    leaks = [name for name, value in OFFLINE_ENV.items() if env.get(name) != value]
    if usage_stats:
        leaks.append("browser.gatherUsageStats")
    if not _loopback(host):
        leaks.append(f"OLLAMA_HOST {host}")
    if leaks:
        return Probe("Outbound calls", WARN, "possible: " + ", ".join(leaks), "the app sets these itself; something overrode them")
    return Probe("Outbound calls", PASS, "none · Ollama on loopback, Hugging Face offline, telemetry off")


def warmup_probe(model, host=engine.OLLAMA_HOST, ollama_up=True):
    """Loads the model the way the app does at start, and times it. A warm model answers in milliseconds."""
    if not ollama_up:
        return Probe("Warm-up", FAIL, f"{model} · Ollama is not running", "ollama serve")
    start = time.perf_counter()
    ok = engine.warm_model(model, host=host)
    took = time.perf_counter() - start
    if not ok:
        return Probe("Warm-up", FAIL, f"{model} did not load", f"ollama run {model}")
    shown = f"{took * 1000:.0f} ms" if took < 1 else f"{took:.1f} s"
    slow = took > SLOW_WARMUP_S
    return Probe("Warm-up", WARN if slow else PASS, f"{model} ready in {shown}",
                 "the first answer waits while the model loads" if slow else "")


def plan(options, model, host=engine.OLLAMA_HOST, index_dir=engine.DEFAULT_INDEX):
    """The checks in the order they run: the machine, the runtime, Ollama and its models, the docs, the network,
    then the warm-up, which needs everything before it. Each step returns one Probe or a list of them."""
    state = {}

    def ollama():
        probe = ollama_probe(host)
        state["up"] = probe.status == PASS
        return probe

    def models():
        return model_probes(options, ollama_tags(host) if state.get("up") else None)

    return (
        ("OS", os_probe),
        ("Kernel", kernel_probe),
        ("CPU", cpu_probe),
        ("RAM", ram_probe),
        ("GPU", gpu_probe),
        ("Disk", disk_probe),
        ("Python", python_probe),
        ("Streamlit", streamlit_probe),
        ("Ollama", ollama),
        ("Models", models),
        ("RAG index", lambda: index_probe(index_dir)),
        ("Embedding model", embed_probe),
        ("Network", network_probe),
        ("Outbound calls", lambda: outbound_probe(host)),
        ("Warm-up", lambda: warmup_probe(model, host, ollama_up=state.get("up", False))),
    )


def run_step(step):
    """Run one step, timed. A probe that raises is a failed line with the error's name, never a traceback."""
    label, fn = step
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        result = Probe(label, FAIL, f"check failed: {type(e).__name__}")
    ms = (time.perf_counter() - start) * 1000
    probes = result if isinstance(result, list) else [result]
    return [Probe(p.label, p.status, p.value, p.fix, ms / len(probes)) for p in probes]


def counts(probes):
    return {status: sum(1 for p in probes if p.status == status) for status in (PASS, WARN, FAIL)}


def summary_text(probes, total_s):
    n = counts(probes)
    failing = [p.label for p in probes if p.status == FAIL]
    verdict = "not ready: " + ", ".join(failing) if failing else "ready to answer offline"
    return (f"{len(probes)} checks · {n[PASS]} pass · {n[WARN]} warn · {n[FAIL]} fail · {total_s:.1f} s · "
            f"{verdict}")


def line_html(label, probe=None):
    """One aligned line: [mark] label value time. Without a probe it is the running state."""
    if probe is None:
        return (f'<div class="diag-line run"><span class="diag-m"><i class="diag-spin"></i></span>'
                f'<span class="diag-l">{escape(label)}</span><span class="diag-v">measuring</span>'
                '<span class="diag-t"></span></div>')
    fix = f'<code class="diag-fix">{escape(probe.fix)}</code>' if probe.fix else ""
    return (f'<div class="diag-line {probe.status}"><span class="diag-m">{MARKS[probe.status]}</span>'
            f'<span class="diag-l">{escape(probe.label)}</span><span class="diag-v">{escape(probe.value)}</span>'
            f'<span class="diag-t">{probe.ms:,.0f} ms</span>{fix}</div>')


def summary_html(probes, total_s):
    state = FAIL if counts(probes)[FAIL] else WARN if counts(probes)[WARN] else PASS
    return f'<p class="diag-sum {state}">{escape(summary_text(probes, total_s))}</p>'


def header_html(stamp=""):
    when = f'<span class="diag-when">{escape(stamp)}</span>' if stamp else ""
    return ('<div class="diag-head"><p class="diag-title">System check</p>'
            '<p class="diag-sub">What Polaris needs from this machine, measured here. Nothing leaves it.</p>'
            f"{when}</div>")
