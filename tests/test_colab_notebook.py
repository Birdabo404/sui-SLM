#!/usr/bin/env python3
"""Syntax and small dry run for the Colab KD notebook and helpers. No GPU."""
import ast
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kd.colab_helpers import (
    TEACHER_4B,
    TEACHER_9B,
    count_jsonl,
    modelfile_text,
    ollama_create_cmd,
    questions_txt_from_jsonl,
    require_seeds,
    resolve_heldout,
    resolve_seeds,
    teacher_ollama_name,
    write_modelfile,
)
from shared.prompt import SYSTEM

NB = ROOT / "kd" / "colab_distill.ipynb"


def _chunk(src, title, path, text, distro="ubuntu"):
    body = text if len(text) >= 250 else (text + " " + ("lab docs. " * 40))
    return {
        "id": f"{path}#0",
        "text": f"{title}\n{body}",
        "source": src,
        "title": title,
        "path": path,
        "chunk": 0,
        "license": "CC-BY-4.0",
        "distro": distro,
    }


class NotebookSyntaxTests(unittest.TestCase):
    def test_notebook_exists_and_code_cells_parse(self):
        nb = json.loads(NB.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(nb["cells"]), 10)
        code_cells = 0
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source") or [])
            if cell["cell_type"] == "code":
                code_cells += 1
                ast.parse(src)
                for line in src.splitlines():
                    self.assertFalse(line.lstrip().startswith("!"), line)
            self.assertNotRegex(src, r"[\U0001F300-\U0001FAFF]")
        self.assertGreaterEqual(code_cells, 8)

    def test_notebook_covers_required_steps(self):
        text = NB.read_text(encoding="utf-8")
        for needle in (
            "USE_4B_TEACHER",
            "your-heldout.jsonl",
            "teacher_generate.py",
            "judge_rows.py",
            "build_sft.py",
            "--holdout",
            "qwen3.5:9b",
            "qwen3.5:4b",
            "QLoRA",
            "q4_k_m",
            "ollama create",
            "Ollama",
            "transformers 4-bit",
        ):
            self.assertIn(needle, text, needle)


class HelperTests(unittest.TestCase):
    def test_teacher_flag(self):
        self.assertEqual(teacher_ollama_name(False), TEACHER_9B)
        self.assertEqual(teacher_ollama_name(True), TEACHER_4B)

    def test_missing_heldout_message(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                resolve_heldout([Path(td) / "missing.jsonl"], td)
            msg = str(cm.exception)
            self.assertIn("Held-out questions file is missing", msg)
            self.assertIn("eval/test.jsonl", msg)
            self.assertIn("150-200", msg)

    def test_questions_txt_from_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "h.jsonl"
            src.write_text(
                "# comment\n"
                + json.dumps({"id": "a", "question": "how do I list files"})
                + "\n"
                + json.dumps({"id": "b", "question": "how do I use tar"})
                + "\n",
                encoding="utf-8",
            )
            dest = Path(td) / "q.txt"
            n = questions_txt_from_jsonl(src, dest)
            self.assertEqual(n, 2)
            self.assertEqual(dest.read_text(encoding="utf-8").splitlines(), [
                "how do I list files",
                "how do I use tar",
            ])

    def test_modelfile(self):
        text = modelfile_text("student.gguf")
        self.assertIn("FROM ./student.gguf", text)
        self.assertIn("think false", text)
        self.assertIn(SYSTEM[:40], text)
        self.assertEqual(ollama_create_cmd("thesis-student"), "ollama create thesis-student -f Modelfile")
        with tempfile.TemporaryDirectory() as td:
            p = write_modelfile(Path(td) / "Modelfile", "x.gguf")
            self.assertTrue(p.is_file())

    def test_require_seeds_stops(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                require_seeds(td, td)
            self.assertIn("Seed chunks are missing", str(cm.exception))


class MockOllama(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode()) if n else {}
        user = ""
        for m in body.get("messages") or []:
            user += m.get("content") or ""
        if "strict reviewer" in user:
            content = json.dumps({"verdict": "yes", "reason": "supported"})
        else:
            content = json.dumps({
                "question": "how do I list files in a folder",
                "steps": ["The excerpt names ls.", "ls lists files."],
                "answer": "Use `ls` in the folder.",
            })
        payload = json.dumps({"message": {"content": content}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class PipelineDryRunTests(unittest.TestCase):
    def test_seed_teacher_judge_sft_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            chunks = td / "chunks.jsonl"
            rows = [
                _chunk("tldr", "ls (tldr common)", "tldr/common/ls.md", "ls lists files. `ls` `ls -l`."),
                _chunk("tldr", "pwd (tldr common)", "tldr/common/pwd.md", "pwd prints the working directory. `pwd`."),
                _chunk("command-manpages", "ls(1)", "command-manpages/ls.1", "ls lists directory contents. `ls`."),
                _chunk("man-pages", "ls(1)", "man-pages/man/man1/ls.1", "NAME ls - list directory contents."),
                _chunk("gnome-user-docs", "Files", "gnome/files.page", "The Files app opens folders."),
                _chunk("ubuntu-docs", "Terminal", "ubuntu/terminal.page", "Open a terminal from the menu."),
                _chunk("ubuntu-server-docs", "ssh.md", "ubuntu-server-docs/ssh.md", "Install openssh-server. `sudo apt install`."),
            ]
            with open(chunks, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            seeds = td / "seed_chunks.jsonl"
            r = subprocess.run(
                [
                    sys.executable, str(ROOT / "kd" / "make_seed_chunks.py"),
                    "--chunks", str(chunks),
                    "--out", str(seeds),
                    "--mix", "tldr=2,command-manpages=1,man-pages=1,gnome-user-docs=1,ubuntu-docs=1,ubuntu-server-docs=1",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertGreaterEqual(count_jsonl(seeds), 6)
            self.assertEqual(resolve_seeds(td, ROOT), seeds)

            httpd = HTTPServer(("127.0.0.1", 0), MockOllama)
            port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                raw = td / "teacher_raw.jsonl"
                r = subprocess.run(
                    [
                        sys.executable, str(ROOT / "kd" / "teacher_generate.py"),
                        "--seeds", str(seeds),
                        "--out", str(raw),
                        "--model", "mock",
                        "--host", f"http://127.0.0.1:{port}",
                        "--limit", "3",
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                self.assertEqual(count_jsonl(raw), 3)
                r = subprocess.run(
                    [
                        sys.executable, str(ROOT / "kd" / "teacher_generate.py"),
                        "--seeds", str(seeds),
                        "--out", str(raw),
                        "--model", "mock",
                        "--host", f"http://127.0.0.1:{port}",
                        "--limit", "3",
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                self.assertIn("3 done", r.stdout)
                self.assertEqual(count_jsonl(raw), 6)

                judged = td / "teacher_judged.jsonl"
                r = subprocess.run(
                    [
                        sys.executable, str(ROOT / "kd" / "judge_rows.py"),
                        "--raw", str(raw),
                        "--seeds", str(seeds),
                        "--out", str(judged),
                        "--model", "mock",
                        "--host", f"http://127.0.0.1:{port}",
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                self.assertEqual(count_jsonl(judged), 6)
                self.assertTrue(all(json.loads(l)["judge"] == "yes" for l in judged.read_text().splitlines() if l.strip()))
            finally:
                httpd.shutdown()
                httpd.server_close()

            holdout = td / "your-heldout.jsonl"
            holdout.write_text(
                json.dumps({
                    "id": "h1",
                    "question": "how do I list files in a folder",
                    "answerable": True,
                    "gold_answer": "ls",
                    "source": "tldr ls",
                }) + "\n",
                encoding="utf-8",
            )
            qtxt = td / "holdout_questions.txt"
            questions_txt_from_jsonl(holdout, qtxt)
            train = td / "sft_train.jsonl"
            val = td / "sft_val.jsonl"
            r = subprocess.run(
                [
                    sys.executable, str(ROOT / "kd" / "build_sft.py"),
                    "--raw", str(judged),
                    "--seeds", str(seeds),
                    "--holdout", str(qtxt),
                    "--out-train", str(train),
                    "--out-val", str(val),
                    "--holdout-sim", "0.80",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("dropped_too_close_to_holdout", r.stdout)


if __name__ == "__main__":
    unittest.main()
