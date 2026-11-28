#!/usr/bin/env python3
"""The system check's probes, each against made-up machine state, and the lines and summary they draw."""
import socket
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import diagnostics
from app.diagnostics import FAIL, PASS, WARN, Probe
from app.models import model_options

Memory = namedtuple("Memory", "total available")
GB = 1024 ** 3


class ProbeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_os_reads_the_pretty_name(self):
        release = self._write("os-release", 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n')
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"):
            self.assertEqual(diagnostics.os_probe(release), Probe("OS", PASS, "Ubuntu 24.04.1 LTS · x86_64"))
        with patch("platform.system", return_value="Darwin"), patch("platform.release", return_value="24.0"):
            probe = diagnostics.os_probe(self.root / "missing")
        self.assertEqual((probe.status, probe.fix), (WARN, "Polaris is built and tested on Ubuntu"))
        self.assertTrue(probe.value.startswith("Darwin 24.0 · "))

    def test_cpu_names_the_model_and_counts_cores_and_threads(self):
        info = self._write("cpuinfo", "processor\t: 0\nmodel name\t: Intel(R) Core(TM)  i5-8250U CPU @ 1.60GHz\n")
        with patch("psutil.cpu_count", side_effect=lambda logical=True: 8 if logical else 4):
            probe = diagnostics.cpu_probe(info)
        self.assertEqual(probe, Probe("CPU", PASS, "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz · 4 cores / 8 threads"))
        with patch("psutil.cpu_count", return_value=2), patch("platform.processor", return_value=""):
            self.assertEqual(diagnostics.cpu_probe(self.root / "missing").value, "unknown CPU · 2 cores")

    def test_ram_warns_when_too_little_is_free_to_load_a_model(self):
        self.assertEqual(diagnostics.ram_probe(Memory(16 * GB, 9 * GB)), Probe("RAM", PASS, "16.0 GB total · 9.0 GB free"))
        low = diagnostics.ram_probe(Memory(8 * GB, int(1.5 * GB)))
        self.assertEqual((low.status, low.value), (WARN, "8.0 GB total · 1.5 GB free"))
        self.assertIn("2 GB free", low.fix)

    def test_gpu_is_optional_and_named_when_present(self):
        none = diagnostics.gpu_probe(self.root / "nvidia", self.root / "drm")
        self.assertEqual(none, Probe("GPU", PASS, "none · models run on the CPU"))
        self._write("drm/card0/device/vendor", "0x8086\n")
        self._write("drm/card0/device/uevent", "DRIVER=i915\nPCI_ID=8086:5917\n")
        self.assertEqual(diagnostics.gpu_probe(self.root / "nvidia", self.root / "drm").value, "Intel (i915)")
        self._write("nvidia/0000:01:00.0/information", "Model: \t GeForce MX150\nIRQ: 130\n")
        self.assertEqual(diagnostics.gpu_probe(self.root / "nvidia", self.root / "drm").value, "GeForce MX150")

    def test_disk_warns_low_and_fails_nearly_full(self):
        Usage = namedtuple("Usage", "total used free")
        for free, status in ((40, PASS), (3, WARN), (0.5, FAIL)):
            with patch("shutil.disk_usage", return_value=Usage(256 * GB, 0, int(free * GB))):
                probe = diagnostics.disk_probe(self.root)
            self.assertEqual(probe.status, status, free)
            self.assertEqual(bool(probe.fix), status != PASS)
        self.assertEqual(probe.value, "0.5 GB free of 256.0 GB")

    def test_python_and_streamlit_match_what_install_sh_sets_up(self):
        self.assertEqual(diagnostics.python_probe((3, 12, 3)), Probe("Python", PASS, "3.12.3"))
        old = diagnostics.python_probe((3, 10, 12))
        self.assertEqual((old.status, old.fix), (WARN, "install.sh supports Python 3.12 to 3.14"))
        self.assertEqual(diagnostics.streamlit_probe("1.45.0"), Probe("Streamlit", PASS, "1.45.0"))
        self.assertEqual(diagnostics.streamlit_probe("1.50.0").status, WARN)

    def test_ollama_reports_its_version_or_how_to_start_it(self):
        with patch("app.engine.ollama_reachable", return_value=True), \
                patch("app.engine.ollama_version", return_value="0.12.6"):
            self.assertEqual(diagnostics.ollama_probe("http://127.0.0.1:11434"),
                             Probe("Ollama", PASS, "0.12.6 · http://127.0.0.1:11434"))
        with patch("app.engine.ollama_reachable", return_value=False):
            self.assertEqual(diagnostics.ollama_probe("http://127.0.0.1:11434"),
                             Probe("Ollama", FAIL, "not running at http://127.0.0.1:11434", "ollama serve"))

    def test_each_model_line_has_its_size_or_the_command_that_makes_it(self):
        options = model_options("qwen3.5:2b")
        tags = [{"name": "qwen3.5:2b", "size": int(2.7 * GB), "details": {"parameter_size": "2.3B",
                                                                          "quantization_level": "Q8_0"}},
                {"name": "thesis-student:latest", "size": int(1.4 * GB), "details": {}}]
        probes = diagnostics.model_probes(options, tags)
        self.assertEqual([p.value for p in probes], [
            "qwen3.5:2b · 2.7 GB · 2.3B · Q8_0 · default base",
            "thesis-base-q4km · not installed · Q4_K_M base",
            "thesis-student · 1.4 GB · KD student",
        ])
        self.assertEqual([p.status for p in probes], [PASS, WARN, PASS])
        self.assertEqual(probes[1].fix, "ollama create thesis-base-q4km -f Modelfile-base")
        # The default base is the one the app can't run without.
        self.assertEqual(diagnostics.model_probes(options, [])[0].status, FAIL)
        self.assertEqual({p.value.split(" · ", 1)[1] for p in diagnostics.model_probes(options, None)},
                         {"can't check until Ollama runs"})

    def test_the_index_must_exist_and_hold_chunks(self):
        with patch("app.engine.index_ok", return_value=False):
            self.assertEqual(diagnostics.index_probe(self.root / "index").status, FAIL)
        with patch("app.engine.index_ok", return_value=True), patch("app.polaris.index_chunk_count", return_value=0):
            self.assertEqual(diagnostics.index_probe(self.root / "index").value, "empty · index")
        with patch("app.engine.index_ok", return_value=True), patch("app.polaris.index_chunk_count", return_value=4812):
            self.assertEqual(diagnostics.index_probe(self.root / "index"), Probe("RAG index", PASS, "4,812 chunks · index"))

    def test_offline_passes_and_says_nothing_is_needed(self):
        self.assertEqual(diagnostics.network_probe(False), Probe("Network", PASS, "OFFLINE · no route out, none needed"))
        self.assertEqual(diagnostics.network_probe(True), Probe("Network", PASS, "online · not used by the app"))

    def test_outbound_flags_anything_that_could_reach_off_the_machine(self):
        quiet = dict(diagnostics.OFFLINE_ENV)
        self.assertEqual(diagnostics.outbound_probe("http://127.0.0.1:11434", quiet, False).status, PASS)
        self.assertEqual(diagnostics.outbound_probe("localhost:11434", quiet, False).status, PASS)
        self.assertEqual(diagnostics.outbound_probe("http://[::1]:11434", quiet, False).status, PASS)
        leaky = diagnostics.outbound_probe("http://10.0.0.5:11434", {**quiet, "HF_HUB_OFFLINE": "0"}, True)
        self.assertEqual(leaky.status, WARN)
        self.assertEqual(leaky.value,
                         "possible: HF_HUB_OFFLINE, browser.gatherUsageStats, OLLAMA_HOST http://10.0.0.5:11434")

    def test_warm_up_is_timed_and_needs_ollama(self):
        with patch("app.engine.warm_model", return_value=True):
            probe = diagnostics.warmup_probe("thesis-student", "http://127.0.0.1:11434")
        self.assertEqual(probe.status, PASS)
        self.assertRegex(probe.value, r"^thesis-student ready in \d+ ms$")
        with patch("app.engine.warm_model", return_value=False):
            self.assertEqual(diagnostics.warmup_probe("thesis-student").fix, "ollama run thesis-student")
        self.assertEqual(diagnostics.warmup_probe("thesis-student", ollama_up=False).status, FAIL)


class RunTests(unittest.TestCase):
    def test_a_probe_that_raises_is_a_failed_line(self):
        def broken():
            raise PermissionError("no")

        (probe,) = diagnostics.run_step(("Disk", broken))
        self.assertEqual((probe.label, probe.status, probe.value), ("Disk", FAIL, "check failed: PermissionError"))

    def test_steps_that_return_several_lines_share_the_time(self):
        lines = diagnostics.run_step(("Models", lambda: [Probe("Model", PASS, "a"), Probe("Model", PASS, "b")]))
        self.assertEqual([p.value for p in lines], ["a", "b"])
        self.assertEqual(lines[0].ms, lines[1].ms)

    def test_the_plan_skips_model_checks_when_ollama_is_down_and_never_leaves_the_machine(self):
        options = model_options("qwen3.5:2b")
        blocked = []

        def no_sockets(*args, **kwargs):
            blocked.append(args)
            raise OSError("network disabled in this test")

        with patch("app.engine.ollama_reachable", return_value=False), patch("app.engine.index_ok", return_value=False), \
                patch("app.issues.hf_cached", return_value=True), patch.object(socket, "create_connection", no_sockets), \
                patch.object(socket.socket, "connect", no_sockets):
            steps = diagnostics.plan(options, "qwen3.5:2b", "http://127.0.0.1:11434", "/nonexistent")
            probes = [p for step in steps for p in diagnostics.run_step(step)]
        self.assertEqual(blocked, [])
        self.assertEqual([p.label for p in probes],
                         ["OS", "Kernel", "CPU", "RAM", "GPU", "Disk", "Python", "Streamlit", "Ollama", "Model",
                          "Model", "Model", "RAG index", "Embedding model", "Network", "Outbound calls", "Warm-up"])
        by_label = {p.label: p for p in probes}
        self.assertEqual(by_label["Warm-up"].value, "qwen3.5:2b · Ollama is not running")
        text = diagnostics.summary_text(probes, 1.84)
        self.assertRegex(text, r"^17 checks · \d+ pass · \d+ warn · \d+ fail · 1\.8 s · not ready: ")
        self.assertIn("Ollama, Model, RAG index, Warm-up", text)


class HtmlTests(unittest.TestCase):
    def test_a_running_line_then_its_result(self):
        self.assertEqual(diagnostics.line_html("RAM"),
                         '<div class="diag-line run"><span class="diag-m"><i class="diag-spin"></i></span>'
                         '<span class="diag-l">RAM</span><span class="diag-v">measuring</span>'
                         '<span class="diag-t"></span></div>')
        done = diagnostics.line_html("RAM", Probe("RAM", WARN, "8.0 GB total · 1.5 GB free", "close <apps>", 3.4))
        self.assertEqual(done, '<div class="diag-line warn"><span class="diag-m">WARN</span><span class="diag-l">RAM</span>'
                               '<span class="diag-v">8.0 GB total · 1.5 GB free</span><span class="diag-t">3 ms</span>'
                               '<code class="diag-fix">close &lt;apps&gt;</code></div>')

    def test_the_summary_takes_the_worst_status(self):
        ok = [Probe("OS", PASS, "x")]
        self.assertEqual(diagnostics.summary_html(ok, 0.5),
                         '<p class="diag-sum pass">1 checks · 1 pass · 0 warn · 0 fail · 0.5 s · ready to answer offline</p>')
        self.assertIn('class="diag-sum warn"', diagnostics.summary_html([*ok, Probe("RAM", WARN, "y")], 1))
        self.assertIn('class="diag-sum fail"', diagnostics.summary_html([*ok, Probe("Disk", FAIL, "z")], 1))

    def test_the_check_css_is_a_monospace_grid_with_no_web_fonts(self):
        css = (Path(diagnostics.__file__).with_name("style.css")).read_text(encoding="utf-8")
        line = css.split(".diag-line{", 1)[1].split("}", 1)[0]
        self.assertIn("display:grid", line)
        self.assertIn("var(--f-mono)", line)
        for status in ("pass", "warn", "fail"):
            self.assertIn(f".diag-line.{status} .diag-m", css)


if __name__ == "__main__":
    unittest.main()
