#!/usr/bin/env python3
"""Offline launcher and desktop installation tests."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.launcher import (
    ensure_ollama,
    launch_environment,
    ollama_server_environment,
    pick_index,
    streamlit_command,
)
from app.engine import is_ollama_process

ROOT = Path(__file__).resolve().parent.parent


class LauncherTests(unittest.TestCase):
    def test_default_index_stays_v2_when_newer_indexes_exist(self):
        with tempfile.TemporaryDirectory() as td:
            rag = Path(td)
            (rag / "rag-index-v10").mkdir()
            self.assertEqual(pick_index(rag), rag / "rag-index-v2")

    def test_environment_sets_offline_defaults_without_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = launch_environment({}, root=root)
            self.assertEqual(env["OLLAMA_MODEL"], "qwen3.5:2b")
            self.assertEqual(env["RAG_INDEX"], str(root / "rag" / "rag-index-v2"))
            self.assertEqual(env["INCLUDE_UBUNTU_SERVER"], "1")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")
            custom = launch_environment(
                {
                    "OLLAMA_MODEL": "custom:latest",
                    "RAG_INDEX": "/tmp/custom-index",
                    "INCLUDE_UBUNTU_SERVER": "0",
                },
                root=root,
            )
            self.assertEqual(custom["OLLAMA_MODEL"], "custom:latest")
            self.assertEqual(custom["RAG_INDEX"], "/tmp/custom-index")
            self.assertEqual(custom["INCLUDE_UBUNTU_SERVER"], "0")
            self.assertEqual(
                ollama_server_environment(env)["OLLAMA_HOST"],
                "127.0.0.1:11434",
            )

    def test_streamlit_runs_on_loopback_without_file_watching(self):
        command = streamlit_command("/tmp/python", port=8765)
        self.assertEqual(command[:4], ["/tmp/python", "-m", "streamlit", "run"])
        self.assertEqual(command[4], "streamlit_app.py")
        self.assertIn("127.0.0.1", command)
        watcher = command.index("--server.fileWatcherType")
        self.assertEqual(command[watcher + 1], "none")
        self.assertIn("8765", command)

    def test_streamlit_script_entry_imports_every_app_module(self):
        entry = streamlit_command(sys.executable)[4]
        probe = """
import importlib
import pkgutil
import runpy
import sys
from pathlib import Path

entry = (Path(sys.argv[1]) / sys.argv[2]).resolve()
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(entry.parent))
runpy.run_path(str(entry), run_name="streamlit_entry")
for found in pkgutil.iter_modules([str(root / "app")]):
    importlib.import_module(f"app.{found.name}")
"""
        result = subprocess.run(
            [sys.executable, "-c", probe, str(ROOT), entry],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ensure_ollama_starts_once_and_waits_until_ready(self):
        checks = iter((False, False, True))
        spawned = []
        sleeps = []
        env = {"OLLAMA_HOST": "http://127.0.0.1:11434"}
        self.assertTrue(
            ensure_ollama(
                env,
                check=lambda host: next(checks),
                spawn=lambda command, child_env: spawned.append((command, child_env)),
                sleep=sleeps.append,
                attempts=3,
            )
        )
        self.assertEqual(spawned, [(["ollama", "serve"], env)])
        self.assertTrue(is_ollama_process("ollama", spawned[0][0]))
        self.assertEqual(sleeps, [0.25])

    def test_ensure_ollama_does_not_spawn_for_remote_host(self):
        spawned = []
        self.assertFalse(
            ensure_ollama(
                {"OLLAMA_HOST": "https://example.com"},
                check=lambda host: False,
                spawn=lambda command, env: spawned.append(command),
            )
        )
        self.assertEqual(spawned, [])

    def test_install_script_installs_launcher_desktop_and_icon_idempotently(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            data = Path(td) / "xdg"
            home.mkdir()
            env = os.environ.copy()
            env["HOME"] = str(home)
            env["XDG_DATA_HOME"] = str(data)
            command = ["bash", str(ROOT / "install.sh"), "--install-launcher"]
            first = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn(f"Warning: {home / '.local' / 'bin'} is not on PATH.", first.stderr)
            self.assertIn('export PATH="$HOME/.local/bin:$PATH"', first.stderr)
            launcher = home / ".local" / "bin" / "sui-slm"
            desktop = data / "applications" / "sui-slm.desktop"
            icon = data / "icons" / "hicolor" / "scalable" / "apps" / "sui-slm.svg"
            self.assertTrue(launcher.is_symlink())
            self.assertEqual(launcher.resolve(), ROOT / "bin" / "sui-slm")
            self.assertTrue(icon.is_file())
            before = (desktop.read_bytes(), icon.read_bytes())
            second = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((desktop.read_bytes(), icon.read_bytes()), before)
            text = desktop.read_text(encoding="utf-8")
            self.assertIn(f'Exec="{launcher}"', text)
            self.assertIn("Icon=sui-slm", text)

    def test_dry_run_is_machine_readable_and_has_no_network_step(self):
        python = ROOT / ".venv" / "bin" / "python"
        if not python.is_file():
            self.skipTest("repository virtualenv is not installed")
        env = os.environ.copy()
        env.pop("OLLAMA_MODEL", None)
        env.pop("RAG_INDEX", None)
        env["SUI_SLM_TEST_SECRET"] = "do-not-print"
        result = subprocess.run(
            [str(python), "-m", "app.launcher", "--dry-run"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["environment"]["OLLAMA_MODEL"], "qwen3.5:2b")
        self.assertIn("--server.fileWatcherType", plan["command"])
        self.assertNotIn("SUI_SLM_TEST_SECRET", plan["environment"])
        self.assertNotIn("do-not-print", result.stdout)

    def test_readme_documents_english_only_scope(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("English-only", readme)


if __name__ == "__main__":
    unittest.main()
