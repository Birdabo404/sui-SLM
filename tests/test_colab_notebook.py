#!/usr/bin/env python3
"""Syntax and small dry run for the Colab KD notebook and helpers. No GPU."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import index_ok
from kd.make_seed_chunks import DEFAULT_MIX
from rag.build_index import DEFAULT_SOURCES, LICENSE
from kd.colab_helpers import (
    OLLAMA_BASE_NAME,
    OLLAMA_CHAT_TEMPLATE,
    OLLAMA_NAME,
    TEACHER_4B,
    TEACHER_9B,
    count_jsonl,
    latest_sft_checkpoint,
    modelfile_text,
    ollama_create_cmd,
    require_seeds,
    resolve_chunks,
    resolve_heldout,
    resolve_seeds,
    select_q4km_gguf,
    teacher_ollama_name,
    write_modelfile,
)
from shared.chatml import ASSISTANT_PREFIX, render_chatml
from shared.holdout import gold_source_stem, load_holdout, path_matches_gold_source
from shared.paths import DEFAULT_INDEX, INDEX_NAME
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
            "USE_QLORA = False",
            "your-heldout.jsonl",
            "teacher_generate.py",
            "judge_rows.py",
            "build_sft.py",
            "--holdout",
            "qwen3.5:9b",
            "qwen3.5:4b",
            "fp16 LoRA",
            "q4_k_m",
            "ollama create",
            "Ollama",
            "transformers 4-bit",
            "transformers==5.3.0",
            "trl==0.28.0",
            "unsloth==2026.10.6",
            "unsloth_zoo==2026.10.6",
            "max_length",
            "processing_class",
            "train_on_responses_only",
            "resume_from_checkpoint",
            "FastLanguageModel",
            "zstd",
            "pciutils",
            "apt-get", "update",
            "/api/tags",
            "1,116",
            "tldr=460",
            "select_q4km_gguf",
            "SFTConfig",
            "thesis-base-q4km",
            "gguf-base",
            "Modelfile-base",
            "Q8_0",
            "differ only in the FROM line",
            "remove_columns",
        ):
            self.assertIn(needle, text, needle)
        self.assertGreaterEqual(text.count("save_pretrained_gguf"), 2)
        self.assertGreaterEqual(text.count("q4_k_m"), 2)
        self.assertNotIn("save_pretrained_merged", text)
        self.assertNotIn("PARAMETER think false", text)
        self.assertNotIn("FastVisionModel", text)
        self.assertNotIn("questions_txt_from_jsonl", text)
        self.assertNotIn("holdout_questions.txt", text)
        self.assertNotIn("assistant_only_loss=True", text)


class DefaultCorpusTests(unittest.TestCase):
    def test_default_seed_mix_target_is_1129_without_server_docs(self):
        mix = {k: int(v) for k, v in (p.split("=") for p in DEFAULT_MIX.split(","))}
        self.assertNotIn("ubuntu-server-docs", mix)
        self.assertEqual(sum(mix.values()), 1129)
        self.assertEqual(mix, {
            "tldr": 460,
            "command-manpages": 295,
            "man-pages": 115,
            "gnome-user-docs": 230,
            "ubuntu-docs": 29,
        })

    def test_default_index_sources_drop_server_docs(self):
        self.assertNotIn("ubuntu-server-docs", DEFAULT_SOURCES)
        self.assertEqual(LICENSE["ubuntu-docs"], "CC-BY-SA-3.0")
        self.assertEqual(LICENSE["ubuntu-server-docs"], "unlicensed")
        self.assertEqual(LICENSE["gnome-user-docs"], "CC-BY-SA-3.0")

    def test_fetch_manifest_labels(self):
        script = (ROOT / "corpus" / "fetch_corpus.sh").read_text(encoding="utf-8")
        self.assertIn('INCLUDE_UBUNTU_SERVER="${INCLUDE_UBUNTU_SERVER:-0}"', script)
        self.assertIn("get ubuntu-docs", script)
        self.assertIn('"CC-BY-SA-3.0"', script)
        self.assertIn("ubuntu-docs-copyright", script)
        self.assertIn("unlicensed (no LICENSE; conf.py TODO)", script)
        self.assertIn("changelogs.ubuntu.com/changelogs/pool/main/u/ubuntu-docs", script)
        legal_line = [
            ln for ln in script.splitlines()
            if "legalcode.txt" in ln and "CC-BY-SA-3.0" in ln
        ]
        self.assertTrue(legal_line)
        self.assertNotIn("|| true", legal_line[0])


class DefaultIndexTests(unittest.TestCase):
    def test_shared_default_is_lab_index(self):
        self.assertEqual(INDEX_NAME, "lab-index")
        self.assertTrue(DEFAULT_INDEX.endswith("rag/lab-index") or DEFAULT_INDEX.endswith("rag\\lab-index"))

    def test_listed_defaults_point_at_lab_index(self):
        from app import engine as engine_mod
        from kd import colab_helpers as helpers_mod

        self.assertTrue(str(engine_mod.DEFAULT_INDEX).endswith(INDEX_NAME))
        self.assertNotIn("rag-index-v2", engine_mod.DEFAULT_INDEX)
        self.assertIn(
            INDEX_NAME,
            helpers_mod.SEEDS_MISSING.format(seeds="x", work="y", index=INDEX_NAME),
        )
        files = {
            ROOT / "rag" / "query_hybrid.py": "lab-index",
            ROOT / "rag" / "query.py": "lab-index",
            ROOT / "kd" / "make_seed_chunks.py": "lab-index",
            ROOT / "install.sh": "rag/lab-index",
        }
        for path, needle in files.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(needle, text, path.name)
            if path.name != "install.sh":
                self.assertNotIn("rag-index-v2", text, path.name)

    def test_install_sh_does_not_reuse_v2_name(self):
        text = (ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn("rag/lab-index", text)
        self.assertNotIn("make_trimmed_index.py", text)
        self.assertNotIn("--out rag/rag-index-v2", text)
        self.assertNotIn("--out rag/rag-index\n", text)


class IndexOkTests(unittest.TestCase):
    def _index(self, folder, rows):
        folder = Path(folder)
        (folder / "chroma").mkdir(parents=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return folder

    def test_missing_files_are_false(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(index_ok(td))

    def test_clean_index_ok(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._index(td, [{"source": "tldr", "path": "tldr/ls.md"}])
            self.assertTrue(index_ok(path))

    def test_server_docs_fail_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._index(td, [
                {"source": "tldr", "path": "tldr/ls.md"},
                {"source": "ubuntu-server-docs", "path": "ubuntu-server-docs/ssh.md"},
            ])
            with self.assertRaises(SystemExit) as cm:
                index_ok(path)
            msg = str(cm.exception)
            self.assertIn("ubuntu-server-docs", msg)
            self.assertIn("INCLUDE_UBUNTU_SERVER=1", msg)

    def test_server_docs_opt_in_flag(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._index(td, [{"source": "ubuntu-server-docs", "path": "s.md"}])
            self.assertTrue(index_ok(path, allow_ubuntu_server=True))

    def test_server_docs_opt_in_env(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._index(td, [{"source": "ubuntu-server-docs", "path": "s.md"}])
            old = os.environ.get("INCLUDE_UBUNTU_SERVER")
            os.environ["INCLUDE_UBUNTU_SERVER"] = "1"
            try:
                self.assertTrue(index_ok(path))
            finally:
                if old is None:
                    os.environ.pop("INCLUDE_UBUNTU_SERVER", None)
                else:
                    os.environ["INCLUDE_UBUNTU_SERVER"] = old


class HoldoutPathTests(unittest.TestCase):
    def test_gold_source_stem(self):
        self.assertEqual(gold_source_stem("ls"), "ls")
        self.assertEqual(gold_source_stem("tldr ls"), "ls")
        self.assertEqual(gold_source_stem("man ls"), "ls")
        self.assertEqual(gold_source_stem("ssh(1)"), "ssh")
        self.assertEqual(gold_source_stem("tldr/common/ls.md"), "ls")
        self.assertEqual(gold_source_stem("command-manpages/ls.1"), "ls")
        self.assertEqual(gold_source_stem("ls.1.gz"), "ls")
        self.assertEqual(gold_source_stem("command-manpages/coreutils/ls.1.gz"), "ls")
        self.assertEqual(gold_source_stem("man-pages/man/man1/ls.1.gz"), "ls")
        self.assertEqual(gold_source_stem("command-manpages/openssh-client/ssh.1.gz"), "ssh")
        self.assertEqual(gold_source_stem("gzip.1.gz"), "gzip")

    def test_path_matches_gold_source(self):
        self.assertTrue(path_matches_gold_source("tldr/common/ls.md", ["tldr ls"]))
        self.assertTrue(path_matches_gold_source("tldr/common/ls.md", ["ls"]))
        self.assertTrue(path_matches_gold_source("command-manpages/ls.1", ["tldr ls"]))
        self.assertTrue(path_matches_gold_source("man-pages/man/man1/ls.1", ["man ls"]))
        self.assertTrue(path_matches_gold_source("command-manpages/ssh.1", ["ssh(1)"]))
        self.assertTrue(path_matches_gold_source("command-manpages/coreutils/ls.1.gz", ["tldr ls"]))
        self.assertTrue(path_matches_gold_source("command-manpages/coreutils/ls.1.gz", ["ls"]))
        self.assertTrue(path_matches_gold_source("man-pages/man/man1/ls.1.gz", ["man ls"]))
        self.assertFalse(path_matches_gold_source("tldr/common/pwd.md", ["tldr ls"]))
        self.assertFalse(path_matches_gold_source("tldr/common/tools.md", ["ls"]))
        self.assertFalse(path_matches_gold_source("tldr/common/false.md", ["ls"]))
        self.assertFalse(path_matches_gold_source("command-manpages/coreutils/tools.1.gz", ["ls"]))
        self.assertFalse(path_matches_gold_source("command-manpages/coreutils/false.1.gz", ["ls"]))

    def test_load_holdout_jsonl_and_txt(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "h.jsonl"
            src.write_text(
                json.dumps({
                    "id": "a",
                    "question": "how do I list files",
                    "source": "tldr ls",
                }) + "\n",
                encoding="utf-8",
            )
            questions, gold = load_holdout(src)
            self.assertEqual(questions, ["how do I list files"])
            self.assertEqual(gold, ["tldr ls"])
            txt = Path(td) / "q.txt"
            txt.write_text("how do I use tar\n", encoding="utf-8")
            questions, gold = load_holdout(txt)
            self.assertEqual(questions, ["how do I use tar"])
            self.assertEqual(gold, [])


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

    def test_modelfile(self):
        text = modelfile_text("student.gguf")
        self.assertIn("FROM ./student.gguf", text)
        self.assertNotIn("think false", text)
        self.assertIn("TEMPLATE", text)
        self.assertIn('PARAMETER stop "<|im_end|>"', text)
        self.assertIn(ASSISTANT_PREFIX, text)
        self.assertTrue(OLLAMA_CHAT_TEMPLATE.endswith(ASSISTANT_PREFIX))
        self.assertIn("<|im_start|>system", text)
        self.assertIn(SYSTEM[:40], text)
        self.assertEqual(OLLAMA_NAME, "thesis-student")
        self.assertEqual(OLLAMA_BASE_NAME, "thesis-base-q4km")
        self.assertEqual(ollama_create_cmd("thesis-student"), "ollama create thesis-student -f Modelfile")
        self.assertEqual(
            ollama_create_cmd(OLLAMA_BASE_NAME, "Modelfile-base"),
            "ollama create thesis-base-q4km -f Modelfile-base",
        )
        with tempfile.TemporaryDirectory() as td:
            p = write_modelfile(Path(td) / "Modelfile", "x.gguf")
            self.assertTrue(p.is_file())

    def test_base_and_student_modelfiles_differ_only_in_from(self):
        student = modelfile_text("student.Q4_K_M.gguf")
        base = modelfile_text("base.Q4_K_M.gguf")
        student_lines = student.splitlines()
        base_lines = base.splitlines()
        self.assertEqual(len(student_lines), len(base_lines))
        self.assertEqual(student_lines[0], "FROM ./student.Q4_K_M.gguf")
        self.assertEqual(base_lines[0], "FROM ./base.Q4_K_M.gguf")
        diffs = [(a, b) for a, b in zip(student_lines, base_lines) if a != b]
        self.assertEqual(len(diffs), 1, diffs)
        self.assertTrue(diffs[0][0].startswith("FROM "))
        self.assertTrue(diffs[0][1].startswith("FROM "))
        self.assertEqual(student_lines[1:], base_lines[1:])
        self.assertIn('PARAMETER stop "<|im_end|>"', student)
        self.assertIn("TEMPLATE", student)
        self.assertIn(SYSTEM[:40], student)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            written_s = write_modelfile(td / "Modelfile", "student.Q4_K_M.gguf")
            written_b = write_modelfile(td / "Modelfile-base", "base.Q4_K_M.gguf")
            s_lines = written_s.read_text(encoding="utf-8").splitlines()
            b_lines = written_b.read_text(encoding="utf-8").splitlines()
            self.assertEqual(s_lines[1:], b_lines[1:])
            self.assertNotEqual(s_lines[0], b_lines[0])
            self.assertTrue(s_lines[0].startswith("FROM "))
            self.assertTrue(b_lines[0].startswith("FROM "))

    def test_sft_text_matches_modelfile_prefix(self):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Question: how do I list files"},
            {"role": "assistant", "content": "Use `ls`."},
        ]
        rendered = render_chatml(messages)
        mf = modelfile_text("student.gguf")
        self.assertIn("<|im_end|>\n" + ASSISTANT_PREFIX, rendered)
        self.assertTrue(rendered.endswith("Use `ls`.<|im_end|>\n"))
        self.assertIn(ASSISTANT_PREFIX, mf)
        self.assertEqual(
            render_chatml(messages[:2], add_generation_prompt=True),
            render_chatml(messages[:2]) + ASSISTANT_PREFIX,
        )
        gen = render_chatml(messages[:2], add_generation_prompt=True)
        self.assertTrue(gen.endswith(ASSISTANT_PREFIX))
        self.assertTrue(OLLAMA_CHAT_TEMPLATE.endswith(ASSISTANT_PREFIX))

    def test_require_seeds_stops(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                require_seeds(td, td)
            self.assertIn("Seed chunks are missing", str(cm.exception))

    def test_resolve_chunks_ignores_old_index_names(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            old = td / "rag" / "rag-index-v2"
            old.mkdir(parents=True)
            (old / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertIsNone(resolve_chunks(td, td))
            fresh = td / "rag" / INDEX_NAME
            fresh.mkdir(parents=True)
            (fresh / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(resolve_chunks(td, td), fresh / "chunks.jsonl")

    def test_select_q4km_gguf(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with self.assertRaises(SystemExit) as cm:
                select_q4km_gguf(td)
            self.assertIn("Q4_K_M", str(cm.exception))
            (td / "other.Q5_K_M.gguf").write_bytes(b"nope")
            with self.assertRaises(SystemExit):
                select_q4km_gguf(td)
            wanted = td / "student.Q4_K_M.gguf"
            wanted.write_bytes(b"ok")
            self.assertEqual(select_q4km_gguf(td), wanted)

    def test_latest_sft_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self.assertIsNone(latest_sft_checkpoint(td))
            (td / "checkpoint-10").mkdir()
            (td / "checkpoint-50").mkdir()
            (td / "checkpoint-2").mkdir()
            self.assertEqual(latest_sft_checkpoint(td).name, "checkpoint-50")


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
    def test_seed_holdout_drops_matching_path(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            chunks = td / "chunks.jsonl"
            rows = [
                _chunk("tldr", "ls (tldr common)", "tldr/common/ls.md", "ls lists files. `ls` `ls -l`."),
                _chunk("tldr", "pwd (tldr common)", "tldr/common/pwd.md", "pwd prints the working directory. `pwd`."),
            ]
            with open(chunks, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            holdout = td / "your-heldout.jsonl"
            holdout.write_text(
                json.dumps({
                    "id": "h1",
                    "question": "how do I list files",
                    "source": "tldr ls",
                }) + "\n",
                encoding="utf-8",
            )
            seeds = td / "seed_chunks.jsonl"
            r = subprocess.run(
                [
                    sys.executable, str(ROOT / "kd" / "make_seed_chunks.py"),
                    "--chunks", str(chunks),
                    "--out", str(seeds),
                    "--mix", "tldr=2",
                    "--holdout", str(holdout),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("dropped_source_path_holdout 1", r.stdout)
            kept = [json.loads(l) for l in seeds.read_text().splitlines() if l.strip()]
            self.assertEqual([k["path"] for k in kept], ["tldr/common/pwd.md"])

    def test_seed_holdout_drops_gzipped_nested_manpage(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            chunks = td / "chunks.jsonl"
            rows = [
                _chunk(
                    "command-manpages",
                    "ls(1)",
                    "command-manpages/coreutils/ls.1.gz",
                    "ls lists files. `ls` `ls -l`.",
                ),
                _chunk(
                    "command-manpages",
                    "pwd(1)",
                    "command-manpages/coreutils/pwd.1.gz",
                    "pwd prints the working directory. `pwd`.",
                ),
            ]
            with open(chunks, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            holdout = td / "your-heldout.jsonl"
            holdout.write_text(
                json.dumps({
                    "id": "h1",
                    "question": "how do I list files",
                    "source": "tldr ls",
                }) + "\n",
                encoding="utf-8",
            )
            seeds = td / "seed_chunks.jsonl"
            r = subprocess.run(
                [
                    sys.executable, str(ROOT / "kd" / "make_seed_chunks.py"),
                    "--chunks", str(chunks),
                    "--out", str(seeds),
                    "--mix", "command-manpages=2",
                    "--holdout", str(holdout),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("dropped_source_path_holdout 1", r.stdout)
            kept = [json.loads(l) for l in seeds.read_text().splitlines() if l.strip()]
            self.assertEqual([k["path"] for k in kept], ["command-manpages/coreutils/pwd.1.gz"])

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
            train = td / "sft_train.jsonl"
            val = td / "sft_val.jsonl"
            r = subprocess.run(
                [
                    sys.executable, str(ROOT / "kd" / "build_sft.py"),
                    "--raw", str(judged),
                    "--seeds", str(seeds),
                    "--holdout", str(holdout),
                    "--out-train", str(train),
                    "--out-val", str(val),
                    "--holdout-sim", "0.80",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertIn("dropped_source_path_holdout", r.stdout)
            self.assertIn("dropped_too_close_to_holdout", r.stdout)
            for dest in (train, val):
                for line in dest.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.assertIn("text", row)
                    self.assertIn(ASSISTANT_PREFIX, row["text"])


if __name__ == "__main__":
    unittest.main()
