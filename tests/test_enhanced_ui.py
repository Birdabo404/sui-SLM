#!/usr/bin/env python3
"""The answer config toggle: the evaluated config by default; Enhanced answers the same way and only allows the
optional grounding score (app/grounding.py, PR #28), off by default. Here app.grounding is faked in sys.modules, and
the tests at the end run against the real one when it is present."""
import csv
import importlib.abc
import sys
import tempfile
import unittest
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_polaris_app as base
from app import enhanced, polaris
from tests import backend_fake

BACKEND = backend_fake.module()
HIT = base.HIT
TAR_PAGE = ("Archiving utility. Often combined with a compression method, such as `gzip` or `bzip2`.\n"
            "- E[x]tract a (compressed) archive [f]ile into the current directory [v]erbosely:\n"
            "`tar {{[-x|--extract]}} {{[-v|--verbose]}} {{[-f|--file]}} {{path/to/source.tar[.gz|.bz2|.xz]}}`\n"
            "- E[x]tract a [z]ipped archive [f]ile:\n`tar xzf {{path/to/source.tar.gz}}`")
REPLY = "Use `tar -xzf archive.tar.gz`."


@dataclass(frozen=True)
class _Label:  # app.grounding.GroundingLabel
    label: str
    score: float


@dataclass(frozen=True)
class _Grounding:  # app.grounding.GroundingResult
    score: float
    labels: tuple
    model: str = enhanced.GROUNDING_MODEL


def _grounding(score=0.82, label="consistent", error=None, on_call=None):
    module = ModuleType("app.grounding")
    module.calls = []

    def score_fn(answer, excerpts, **kw):
        module.calls.append({"answer": answer, "excerpts": list(excerpts)})
        if on_call is not None:
            on_call()
        if error is not None:
            raise error
        return _Grounding(score, (_Label(label, score),))

    module.score = score_fn
    return module


class _ImportSpy(importlib.abc.MetaPathFinder):
    """Records every attempt to import app.grounding, and lets the import go on."""

    def __init__(self):
        self.seen = []

    def find_spec(self, name, path=None, target=None):
        if name == "app.grounding":
            self.seen.append(name)


class ConfigTests(unittest.TestCase):
    def test_enhanced_applies_only_to_modes_with_retrieval_and_imports_nothing(self):
        spy = _ImportSpy()
        with patch.object(sys, "meta_path", [spy, *sys.meta_path]), patch.dict(sys.modules):
            sys.modules.pop("app.grounding", None)
            for requested, rag, want in ((None, True, "frozen"), ("frozen", True, "frozen"), ("x", True, "frozen"),
                                         ("enhanced", False, "frozen"), ("enhanced", True, "enhanced")):
                with self.subTest(requested=requested, rag=rag):
                    self.assertEqual(enhanced.config_for(requested, rag), want)
        self.assertEqual(spy.seen, [])

    def test_a_missing_grounding_module_is_named(self):
        with patch.dict(sys.modules, {"app.grounding": None}), self.assertRaisesRegex(enhanced.Unavailable,
                                                                                      "app/grounding.py"):
            enhanced.grounding_backend()


class GroundTests(unittest.TestCase):
    def test_score_and_label_come_from_the_backend_result(self):
        module = _grounding(0.31, "hallucinated")
        got = enhanced.ground("Use tar.", ["tar\npage"], module)
        self.assertEqual(got, {"score": 0.31, "label": "hallucinated", "model": enhanced.GROUNDING_MODEL})
        self.assertEqual(module.calls, [{"answer": "Use tar.", "excerpts": ["tar\npage"]}])

    def test_without_labels_the_threshold_decides(self):
        module = ModuleType("app.grounding")
        module.score = lambda answer, excerpts: _Grounding(0.5, ())
        self.assertEqual(enhanced.ground("a", ["b"], module)["label"], "consistent")

    def test_excerpts_are_what_the_model_saw(self):
        self.assertEqual(enhanced.excerpts_of([{"title": "tar", "text": "Archiving utility."}, {"text": "x"}, {}]),
                         ["tar\nArchiving utility.", "x", ""])


class BadgeHtmlTests(unittest.TestCase):
    def test_config_chip(self):
        self.assertEqual(polaris.config_chip_html({}), "")
        self.assertIn(">evaluated config</span>", polaris.config_chip_html({"config_mode": "frozen"}))
        chip = polaris.config_chip_html({"config_mode": "enhanced"})
        self.assertIn(">evaluated config + Enhanced checks</span>", chip)
        self.assertIn("not part of the reported eval", chip)

    def test_metrics_carry_the_chip_and_old_records_do_not(self):
        rec = {"latency_s": 1.0, "eval_count": 10, "tok_s": 10.0, "gen_s": 1.0}
        self.assertNotIn("cfg", polaris.metrics_html(rec))
        self.assertIn("Enhanced checks", polaris.metrics_html(dict(rec, config_mode="enhanced")))

    def test_command_check_badge(self):
        self.assertEqual(polaris.answer_checks_html({}), "")
        self.assertEqual(polaris.answer_checks_html({"command_check": {"commands": [], "missing": []}}), "")
        warn = polaris.answer_checks_html({"command_check": {"commands": ["apt"], "missing": ["apt-fast",
                                                                                              "tar -j", "<x>"]}})
        self.assertIn('class="ac-cmd warn"', warn)
        self.assertIn("<code>apt-fast</code><code>tar -j</code><code>&lt;x&gt;</code>", warn)
        ok = polaris.answer_checks_html({"command_check": {"commands": ["tar"], "missing": []}})
        self.assertIn(">1 command · all in the excerpts</span>", ok)
        self.assertIn("2 commands", polaris.answer_checks_html({"command_check": {"commands": ["a", "b"]}}))

    def test_grounding_badge(self):
        pending = polaris.answer_checks_html({}, grounding_pending=True)
        self.assertIn('class="ac-gr wait"', pending)
        self.assertIn("grounding · checking…", pending)
        ok = polaris.answer_checks_html({"grounding": {"score": 0.8234, "label": "consistent"}})
        self.assertIn('class="ac-gr ok"', ok)
        self.assertIn("grounding <b>0.82</b> · supported", ok)
        low = polaris.answer_checks_html({"grounding": {"score": 0.1, "label": "hallucinated"}})
        self.assertIn('class="ac-gr low"', low)
        self.assertIn("· not supported", low)
        self.assertEqual(polaris.answer_checks_html({"grounding": {"issue": {"kind": "x"}}}), "")

    def test_export_keeps_the_config(self):
        self.assertEqual(polaris.EXPORT_FIELDS[-1], "config_mode")


class EnhancedAppTests(unittest.TestCase):
    """The toggle in the running app, through the same AppTest harness as tests/test_polaris_app.py."""

    _Run = base.PolarisRunLoopTests
    setUpClass = _Run.__dict__["setUpClass"]
    tearDownClass = _Run.__dict__["tearDownClass"]
    setUp = _Run.setUp
    _html = _Run._html
    _settings_html = _Run._settings_html
    _ask = _Run._ask
    _ollama = _Run._ollama
    _backend = _Run._backend
    _chat_id = _Run._chat_id

    def _answer(self, *, config=None, grounding=None, modules=None, hit=None, reply=REPLY, mode="full"):
        """One real streamed answer through the loaded index's own retrieve. Returns the answers.csv rows."""
        host = self._ollama(base._ndjson([reply[i:i + 9] for i in range(0, len(reply), 9)]))
        hit = dict(HIT, **(hit or {}))
        self.retrieved = []

        def retrieve(query, k=5, tldr_lane=True):
            self.retrieved.append((query, k, tldr_lane))
            return [dict(hit)]

        index = MagicMock(HybridRetriever=lambda index_dir: MagicMock(retrieve=retrieve))
        log = Path(tempfile.mkdtemp()) / "answers.csv"
        self.streamlit.cache_resource.clear()
        self.at.session_state["mode"] = mode
        if config is not None:
            self.at.session_state["config_mode"] = config
        if grounding is not None:
            self.at.session_state["grounding_enabled"] = grounding
        with patch("app.streaming.stream_answer_question", base.REAL_STREAM), patch("app.streaming.OLLAMA_HOST", host), \
                patch("app.engine.LOG_CSV", log), patch("app.engine.index_ok", return_value=True), \
                patch.dict(sys.modules, {"rag.query_hybrid": index, **(modules or {})}):
            self.at.run()
            self._ask("extract a .tar.gz file", "landing")
        self.assertFalse(self.at.exception)
        if not log.exists():
            return []
        with log.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _record(self):
        return self.store.load_chat(self._chat_id()).messages[-1].get("record")

    def _spy(self):
        spy = _ImportSpy()
        self.enterContext(patch.object(sys, "meta_path", [spy, *sys.meta_path]))
        self.enterContext(patch.dict(sys.modules))
        sys.modules.pop("app.grounding", None)
        return spy

    # ---------- the evaluated config never touches the Enhanced backend ----------

    def test_frozen_answers_never_import_or_call_grounding(self):
        spy = self._spy()
        # Grounding switched on and the module one import away: the evaluated config still never reaches it.
        rows = self._answer(grounding=True)
        self.assertNotIn('class="issue', self._html())
        self._settings_html()
        self.assertEqual(spy.seen, [])
        self.assertNotIn("app.grounding", sys.modules)
        self.assertEqual(self.at.session_state["config_mode"], enhanced.FROZEN)
        self.assertEqual([r["config"] for r in rows], ["base+RAG+CoT"])
        self.assertEqual(self._record()["config_mode"], "frozen")
        self.assertNotIn("grounding", self._record())
        self.assertIn(">evaluated config</span>", self._html())

    def test_frozen_with_grounding_installed_still_does_not_call_it(self):
        grounding = _grounding()
        self._answer(grounding=True, modules={"app.grounding": grounding})
        self.assertEqual(grounding.calls, [])

    # ---------- Enhanced: the same answer ----------

    def test_enhanced_answers_exactly_like_the_evaluated_config(self):
        frozen_rows = self._answer()
        (frozen_body,) = base._NdjsonOllama.bodies
        frozen_retrieved = self.retrieved
        self.at = self.AppTest.from_file(base.APP, default_timeout=20)
        rows = self._answer(config=enhanced.ENHANCED, modules={"app.grounding": _grounding()})
        (body,) = base._NdjsonOllama.bodies
        self.assertEqual(body, frozen_body)
        self.assertEqual((body["think"], body["options"]["temperature"]), (False, 0.2))
        self.assertEqual(self.retrieved, frozen_retrieved)
        self.assertEqual(self.retrieved[0][1], 5)
        self.assertEqual([r["config"] for r in rows], [r["config"] for r in frozen_rows])
        # Grounding is off, so nothing Enhanced ran on this answer.
        self.assertEqual(self._record()["config_mode"], "frozen")
        self.assertIn(">evaluated config</span>", self._html())

    def test_enhanced_without_grounding_installed_says_so_and_answers_the_same(self):
        rows = self._answer(config=enhanced.ENHANCED, grounding=True, modules={"app.grounding": None})
        self.assertEqual([r["config"] for r in rows], ["base+RAG+CoT"])
        self.assertEqual(self._record()["config_mode"], "frozen")
        self.assertNotIn("ac-gr", self._html())

    def test_base_mode_ignores_enhanced(self):
        grounding = _grounding()
        self._answer(config=enhanced.ENHANCED, grounding=True, modules={"app.grounding": grounding}, mode="base")
        self.assertEqual(grounding.calls, [])
        self.assertIn(">evaluated config</span>", self._html())

    # ---------- the grounding score ----------

    def test_grounding_scores_after_the_answer_is_shown_and_stored(self):
        stored = []

        def on_call():
            (chat,) = self.store.list_chats()
            stored.append(self.store.load_chat(chat.chat_id).messages[-1]["record"]["answer"])

        grounding = _grounding(0.82, "consistent", on_call=on_call)
        rows = self._answer(config=enhanced.ENHANCED, grounding=True, modules={"app.grounding": grounding})
        self.assertEqual(stored, [REPLY])
        (call,) = grounding.calls
        self.assertEqual(call["excerpts"], ["tar\ntar"])
        html = self._html()
        self.assertIn("grounding <b>0.82</b> · supported", html)
        self.assertNotIn("checking…", html)
        self.assertIn(">evaluated config + Enhanced checks</span>", html)
        self.assertEqual([r["config"] for r in rows], ["base+RAG+CoT+grounding"])
        record = self._record()
        self.assertEqual(record["config_mode"], "enhanced")
        self.assertEqual(record["grounding"], {"score": 0.82, "label": "consistent", "model": enhanced.GROUNDING_MODEL})

    def test_grounding_off_in_enhanced_is_never_called(self):
        grounding = _grounding()
        self._answer(config=enhanced.ENHANCED, modules={"app.grounding": grounding})
        self.assertEqual(grounding.calls, [])
        self.assertNotIn("ac-gr", self._html())

    def test_a_grounding_failure_keeps_the_answer_and_shows_a_compact_card(self):
        self._backend()
        grounding = _grounding(error=BACKEND.GroundingModelError("not cached"))
        self._answer(config=enhanced.ENHANCED, grounding=True,
                     modules={"app.grounding": grounding, "app.backend": BACKEND})
        html = self._html()
        self.assertIn('data-kind="grounding_model_unavailable"', html)
        self.assertIn("The grounding model isn&#x27;t on this machine", html)
        self.assertIn("<code>bash install.sh --enhanced</code>", html)
        self.assertIn('class="issue compact', html)
        record = self._record()
        self.assertEqual(record["answer"], REPLY)
        self.assertEqual(record["grounding"]["issue"]["kind"], "grounding_model_unavailable")

    def test_an_untyped_grounding_failure_says_only_the_score_is_missing(self):
        grounding = _grounding(error=OSError("disk"))
        self._answer(config=enhanced.ENHANCED, grounding=True, modules={"app.grounding": grounding})
        html = self._html()
        self.assertIn('data-kind="grounding_failed"', html)
        self.assertIn("only its grounding score is missing", html)
        self.assertNotIn("disk", html)

    def test_grounding_is_skipped_for_an_abstained_answer(self):
        grounding = _grounding()
        self._answer(config=enhanced.ENHANCED, grounding=True, modules={"app.grounding": grounding},
                     hit={"dense": 0.1})
        self.assertEqual(grounding.calls, [])
        self.assertEqual(base._NdjsonOllama.bodies, [])

    # ---------- the command check ----------

    def test_commands_missing_from_the_excerpts_are_listed(self):
        self._answer(reply="Install it:\n```bash\nsudo apt-fast install p7zip\n```\nthen `tar -xjf a.tar.bz2`.",
                     hit={"text": TAR_PAGE})
        html = self._html()
        self.assertIn('class="ac-cmd warn"', html)
        self.assertIn("<code>apt-fast</code><code>tar -j</code>", html)
        self.assertEqual(self._record()["command_check"],
                         {"commands": ["apt-fast", "tar"], "missing": ["apt-fast", "tar -j"], "flags_checked": 1})

    def test_commands_in_the_excerpts_say_so_quietly(self):
        self._answer(hit={"text": TAR_PAGE})
        self.assertIn(">1 command · all in the excerpts</span>", self._html())
        self.assertEqual(self._record()["command_check"]["missing"], [])

    def test_the_command_check_runs_in_enhanced_too(self):
        self._answer(config=enhanced.ENHANCED, grounding=True, hit={"text": TAR_PAGE},
                     modules={"app.grounding": _grounding()})
        self.assertIn("all in the excerpts", self._html())

    def test_no_command_check_without_excerpts(self):
        self._answer(mode="base")
        self.assertNotIn("ans-checks", self._html())

    def test_a_reloaded_chat_shows_its_badges_again(self):
        self._answer(config=enhanced.ENHANCED, grounding=True, hit={"text": TAR_PAGE},
                     modules={"app.grounding": _grounding()})
        cid = self._chat_id()
        with patch.dict(sys.modules, {"app.grounding": _grounding()}):
            self.at.button(key="new-chat").click().run()
            self.assertNotIn("ans-checks", self._html())
            self.at.button(key=f"hist-open-{cid}").click().run()
        self.assertFalse(self.at.exception)
        html = self._html()
        self.assertIn("evaluated config + Enhanced checks", html)
        self.assertIn("all in the excerpts", html)
        self.assertIn("grounding <b>0.82</b>", html)

    # ---------- Settings ----------

    def test_settings_default_to_the_evaluated_config(self):
        self.at.run()
        html = self._settings_html()
        self.assertIn("Answer config", html)
        (radio,) = [r for r in self.at.radio if r.key == "config-pick"]
        self.assertEqual(radio.value, enhanced.FROZEN)
        self.assertEqual(radio.options, ["Evaluated config (frozen)", "Enhanced"])
        self.assertIn("Exactly what the thesis eval graded", html)
        self.assertEqual([t for t in self.at.toggle if t.key == "grounding-pick"], [])

    def test_picking_enhanced_offers_grounding_switched_off(self):
        self.enterContext(patch.dict(sys.modules, {"app.grounding": _grounding()}))
        self.at.run()
        self._settings_html()
        self.at.radio(key="config-pick").set_value(enhanced.ENHANCED).run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.session_state["config_mode"], enhanced.ENHANCED)
        html = self._settings_html()
        self.assertIn("exactly as in the evaluated config", html)
        self.assertIn("not part of reported eval", html)
        self.assertEqual(self.at.radio(key="config-pick").value, enhanced.ENHANCED)
        toggle = self.at.toggle(key="grounding-pick")
        self.assertEqual((toggle.label, toggle.value), ("Grounding score · experimental", False))
        self.assertIn("Off by default; whether it stays is undecided.", toggle.help)
        self.assertNotIn("Grounding model", html)
        toggle.set_value(True).run()
        self.assertTrue(self.at.session_state["grounding_enabled"])
        html = self._settings_html()
        self.assertTrue(self.at.toggle(key="grounding-pick").value)
        self.assertIn("Grounding model", html)
        self.at.radio(key="config-pick").set_value(enhanced.FROZEN).run()
        self.assertEqual(self.at.session_state["config_mode"], enhanced.FROZEN)
        self.assertNotIn("Grounding model", self._settings_html())

    def test_settings_say_when_grounding_is_not_in_this_build(self):
        self.enterContext(patch.dict(sys.modules, {"app.grounding": None}))
        self.at.session_state["config_mode"] = enhanced.ENHANCED
        self.at.session_state["grounding_enabled"] = True
        self.at.run()
        html = self._settings_html()
        self.assertIn('class="dlg-note warn"', html)
        self.assertIn("app/grounding.py can&#x27;t be imported", html)
        self.assertEqual([t for t in self.at.toggle if t.key == "grounding-pick"], [])
        self.assertNotIn("Grounding model", html)


@unittest.skipUnless(find_spec("app.grounding"), "app/grounding.py is PR #28")
class RealBackendTests(unittest.TestCase):
    """ground() against the real app.grounding, with its model faked."""

    def test_ground_through_the_real_score(self):
        grounding = enhanced.grounding_backend()
        with patch.object(grounding, "_load_scorer", return_value=MagicMock(predict=lambda pairs: [0.27])):
            got = enhanced.ground("Use tar.", ["tar\npage"])
        self.assertEqual(got, {"score": 0.27, "label": "hallucinated", "model": grounding.GROUNDING_MODEL})
        with self.assertRaises(ValueError):
            enhanced.ground("", ["tar"])

    def test_a_failing_model_is_typed(self):
        grounding = enhanced.grounding_backend()
        broken = MagicMock(predict=MagicMock(side_effect=OSError("no model")))
        with patch.object(grounding, "_load_scorer", return_value=broken), \
                self.assertRaises(grounding.GroundingModelError):
            enhanced.ground("Use tar.", ["tar\npage"])


if __name__ == "__main__":
    unittest.main()
