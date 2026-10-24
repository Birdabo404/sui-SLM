"""Launch the offline Streamlit app and its local Ollama service."""
import argparse
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

import requests

from app.engine import index_ok, ollama_reachable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_PORT = 8501


def pick_index(rag_dir):
    rag_dir = Path(rag_dir)
    selected = rag_dir / "rag-index-v2"
    version = 2
    for candidate in rag_dir.glob("rag-index-v*"):
        suffix = candidate.name.removeprefix("rag-index-v")
        if not suffix.isdigit() or int(suffix) <= version or not index_ok(candidate):
            continue
        selected = candidate
        version = int(suffix)
    return selected


def launch_environment(environ=None, root=None):
    root = ROOT if root is None else Path(root)
    env = dict(os.environ if environ is None else environ)
    env.setdefault("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    env.setdefault("OLLAMA_MODEL", "qwen3.5:2b")
    env.setdefault("INCLUDE_UBUNTU_SERVER", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("ANONYMIZED_TELEMETRY", "False")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    if not env.get("RAG_INDEX"):
        env["RAG_INDEX"] = str(pick_index(root / "rag"))
    return env


def streamlit_command(python, port=DEFAULT_PORT):
    return [
        python,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.fileWatcherType",
        "none",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def ollama_server_environment(env):
    child_env = dict(env)
    parsed = urlsplit(child_env.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    if parsed.scheme and parsed.hostname:
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = f"[{hostname}]"
        child_env["OLLAMA_HOST"] = f"{hostname}:{parsed.port or 11434}"
    return child_env


def ensure_ollama(
    env,
    check=None,
    spawn=None,
    sleep=None,
    attempts=60,
):
    host = env.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    check = ollama_reachable if check is None else check
    spawn = _spawn_ollama if spawn is None else spawn
    sleep = time.sleep if sleep is None else sleep
    if check(host):
        return True
    if not _loopback_url(host):
        return False
    try:
        spawn(["ollama", "serve"], env)
    except OSError:
        return False
    for _ in range(attempts):
        if check(host):
            return True
        sleep(0.25)
    return False


def app_reachable(port=DEFAULT_PORT):
    try:
        response = requests.get(
            f"http://127.0.0.1:{port}/_stcore/health",
            timeout=0.5,
        )
        return response.ok
    except requests.RequestException:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    env = launch_environment()
    command = streamlit_command(sys.executable, args.port)
    if args.dry_run:
        print(json.dumps({"command": command, "environment": env}, sort_keys=True))
        return 0

    if not ensure_ollama(env):
        print("Ollama is not available. Install it or start `ollama serve`.", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}/"
    if app_reachable(args.port):
        webbrowser.open(url)
        return 0

    process = subprocess.Popen(command, cwd=ROOT, env=env)
    for _ in range(120):
        if process.poll() is not None:
            return process.returncode or 1
        if app_reachable(args.port):
            webbrowser.open(url)
            return process.wait()
        time.sleep(0.25)
    process.terminate()
    return 1


def _spawn_ollama(command, env):
    return subprocess.Popen(
        command,
        env=ollama_server_environment(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _loopback_url(url):
    return urlsplit(url).hostname in {"127.0.0.1", "::1", "localhost"}


if __name__ == "__main__":
    raise SystemExit(main())
