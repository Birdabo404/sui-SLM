#!/usr/bin/env python3
"""Ollama NDJSON streaming integration tests against a local fake server."""
import csv
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from app.chat_memory import ChatContext, Turn
from app.engine import OllamaInfo
from app.streaming import (
    CancellableChat,
    ChatCancelled,
    OllamaStreamError,
    retrieval_chart,
    stream_answer_question,
)


def _line(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def _terminal():
    return {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "eval_count": 20,
        "eval_duration": 2_000_000_000,
        "prompt_eval_count": 40,
        "prompt_eval_duration": 500_000_000,
        "load_duration": 100_000_000,
    }


class _StreamingOllama(BaseHTTPRequestHandler):
    fragments = []
    status = 200
    bodies = []
    first_fragment_sent = threading.Event()
    release = threading.Event()
    wait_after_first = False

    def do_POST(self):
        size = int(self.headers.get("Content-Length") or 0)
        type(self).bodies.append(json.loads(self.rfile.read(size)))
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            for index, fragment in enumerate(type(self).fragments):
                self.wfile.write(fragment)
                self.wfile.flush()
                if index == 0:
                    type(self).first_fragment_sent.set()
                    if type(self).wait_after_first:
                        type(self).release.wait(3)
        except OSError:
            pass

    def log_message(self, *args):
        pass


class _Retriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, k=5, tldr_lane=True):
        self.calls.append((query, k, tldr_lane))
        return self.hits[:k]


class StreamingTests(unittest.TestCase):
    def setUp(self):
        _StreamingOllama.fragments = []
        _StreamingOllama.status = 200
        _StreamingOllama.bodies = []
        _StreamingOllama.first_fragment_sent.clear()
        _StreamingOllama.release.clear()
        _StreamingOllama.wait_after_first = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingOllama)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        _StreamingOllama.release.set()
        self.server.shutdown()
        self.server.server_close()

    def _stream(self, log, **kwargs):
        return stream_answer_question(
            "extract a tar file",
            "base",
            "qwen3.5:2b",
            chat=CancellableChat(self.host),
            log_path=log,
            git_sha="abc1234",
            model_info=OllamaInfo(version="0.12.3", model_digest="sha256:abc"),
            **kwargs,
        )

    def test_incremental_text_progress_and_terminal_stats(self):
        first = _line({"message": {"role": "assistant", "content": "Use "}, "done": False})
        _StreamingOllama.fragments = [
            b"\n" + first[:17],
            first[17:],
            _line({"message": {"role": "assistant", "content": "`tar`."}, "done": False}),
            _line(_terminal()),
        ]
        progress = []
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            stream = self._stream(
                log,
                temperature=0,
                seed=42,
                keep_alive="30m",
                on_progress=progress.append,
            )
            with self.assertRaises(RuntimeError):
                _ = stream.result
            self.assertIs(iter(stream), stream)
            chunks = list(stream)
            with open(log, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))

        self.assertEqual(chunks, ["Use ", "`tar`."])
        self.assertEqual(stream.result["text"], "Use `tar`.")
        self.assertEqual(stream.result["eval_count"], 20)
        self.assertEqual(stream.result["eval_duration"], 2_000_000_000)
        self.assertEqual(stream.result["prompt_eval_count"], 40)
        self.assertEqual(stream.result["prompt_eval_duration"], 500_000_000)
        self.assertEqual(stream.result["load_duration"], 100_000_000)
        self.assertEqual(stream.result["tok_s"], 10.0)
        self.assertEqual(row["tok_s"], "10.0000")
        self.assertEqual((progress[-1].chunk_count, progress[-1].character_count), (2, 10))
        self.assertGreater(progress[-1].chunks_per_s, 0)
        self.assertGreater(progress[-1].characters_per_s, 0)
        self.assertFalse(hasattr(progress[-1], "token_count"))
        body = _StreamingOllama.bodies[0]
        self.assertTrue(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["seed"], 42)
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(list(stream), [])

    def test_retrieval_chart_keeps_raw_score_semantics_and_context(self):
        _StreamingOllama.fragments = [
            _line({"message": {"role": "assistant", "content": "Run `ls`."}, "done": False}),
            _line(_terminal()),
        ]
        hits = [{
            "title": "List files",
            "path": "tldr/ls.md",
            "source": "tldr",
            "text": "Use ls.",
            "dense": 0.61,
            "rrf": 1 / 60,
            "bm25": 4.2,
        }]
        retriever = _Retriever(hits)
        context = ChatContext(
            retrieval_query="previous question current question",
            turns=(Turn("previous question", "previous answer"),),
            retrieval_turns=1,
        )
        charts = []
        with tempfile.TemporaryDirectory() as td:
            stream = stream_answer_question(
                "current question",
                "rag",
                "qwen3.5:2b",
                retriever=retriever,
                chat=CancellableChat(self.host),
                log_path=Path(td) / "answers.csv",
                git_sha="abc1234",
                context=context,
                on_retrieval=charts.append,
            )
            self.assertEqual(list(stream), ["Run `ls`."])

        self.assertEqual(retriever.calls, [("previous question current question", 5, True)])
        self.assertEqual(stream.result["context_turns"], 1)
        chart = charts[0]
        self.assertFalse(chart.weak)
        self.assertEqual(chart.dense_threshold, 0.30)
        self.assertEqual(chart.rows[0].dense_cosine, 0.61)
        self.assertAlmostEqual(chart.rows[0].rrf_rank_score, 1 / 60)
        self.assertEqual(chart.rows[0].bm25_raw, 4.2)
        self.assertIn("not confidence", " ".join(chart.notes))
        self.assertFalse(hasattr(chart.rows[0], "confidence"))
        sent = _StreamingOllama.bodies[0]["messages"]
        self.assertEqual([message["content"] for message in sent[1:3]], ["previous question", "previous answer"])

    def test_fragmented_utf8_and_terminal_without_newline(self):
        event = json.dumps(
            {"message": {"role": "assistant", "content": "café"}, "done": False},
            ensure_ascii=False,
        ).encode("utf-8") + b"\n"
        split = event.index("é".encode("utf-8")) + 1
        _StreamingOllama.fragments = [
            event[:split],
            event[split:],
            _line(_terminal()).rstrip(b"\n"),
        ]
        with tempfile.TemporaryDirectory() as td:
            stream = self._stream(Path(td) / "answers.csv")
            self.assertEqual(list(stream), ["café"])
        self.assertEqual(stream.result["text"], "café")

    def test_cancel_closes_stream_and_writes_no_log(self):
        _StreamingOllama.fragments = [
            _line({"message": {"role": "assistant", "content": "partial"}, "done": False}),
            _line(_terminal()),
        ]
        _StreamingOllama.wait_after_first = True
        box = {}
        chunk_seen = threading.Event()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            stream = self._stream(log, on_progress=lambda progress: chunk_seen.set())

            def consume():
                try:
                    box["chunks"] = list(stream)
                except BaseException as e:
                    box["error"] = e

            worker = threading.Thread(target=consume)
            worker.start()
            self.assertTrue(_StreamingOllama.first_fragment_sent.wait(2))
            self.assertTrue(chunk_seen.wait(2))
            stream.cancel()
            _StreamingOllama.release.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(box.get("error"), ChatCancelled)
            self.assertFalse(log.exists())
            self.assertIsNone(stream.chat._conn)

    def test_early_close_cleans_up_and_writes_no_log(self):
        _StreamingOllama.fragments = [
            _line({"message": {"role": "assistant", "content": "partial"}, "done": False}),
            _line(_terminal()),
        ]
        _StreamingOllama.wait_after_first = True
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            stream = self._stream(log)
            self.assertEqual(next(stream), "partial")
            stream.close()
            _StreamingOllama.release.set()
            self.assertFalse(log.exists())
            self.assertIsNone(stream.chat._conn)

    def test_cancel_during_retrieval_suppresses_callbacks_and_http(self):
        hits = [{
            "title": "doc",
            "path": "doc.md",
            "text": "text",
            "dense": 0.8,
            "rrf": 1 / 60,
        }]
        chat = CancellableChat(self.host)
        charts = []

        class CancellingRetriever:
            def retrieve(self, query, k=5, tldr_lane=True):
                chat.cancel()
                return hits

        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            stream = stream_answer_question(
                "q",
                "rag",
                "m",
                retriever=CancellingRetriever(),
                chat=chat,
                log_path=log,
                on_retrieval=charts.append,
            )
            with self.assertRaises(ChatCancelled):
                list(stream)
            self.assertFalse(log.exists())
        self.assertEqual(charts, [])
        self.assertEqual(_StreamingOllama.bodies, [])

    def test_cancel_from_progress_callback_emits_no_text_and_writes_no_log(self):
        _StreamingOllama.fragments = [
            _line({"message": {"role": "assistant", "content": "partial"}, "done": False}),
            _line(_terminal()),
        ]
        holder = []
        progress = []

        def stop(update):
            progress.append(update)
            holder[0].cancel()

        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            stream = self._stream(log, on_progress=stop)
            holder.append(stream)
            with self.assertRaises(ChatCancelled):
                next(stream)
            self.assertFalse(log.exists())
        self.assertEqual(len(progress), 1)

    def test_protocol_and_http_errors_write_no_log_and_close(self):
        cases = (
            ([b"not json\n"], 200, OllamaStreamError, "invalid NDJSON"),
            ([_line({"message": {"content": "partial"}, "done": False})], 200, OllamaStreamError, "terminal"),
            ([_line({"error": "model missing"})], 200, OllamaStreamError, "model missing"),
            ([_line([])], 200, OllamaStreamError, "JSON object"),
            ([_line({"message": {"content": 3}, "done": False})], 200, OllamaStreamError, "must be text"),
            ([_line({"message": {"content": ""}, "done": 0})], 200, OllamaStreamError, "must be boolean"),
            ([b"server exploded"], 500, requests.HTTPError, "HTTP 500"),
        )
        for index, (fragments, status, error_type, message) in enumerate(cases):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as td:
                _StreamingOllama.fragments = fragments
                _StreamingOllama.status = status
                chat = CancellableChat(self.host)
                log = Path(td) / f"answers-{index}.csv"
                stream = stream_answer_question(
                    "q",
                    "base",
                    "m",
                    chat=chat,
                    log_path=log,
                    git_sha="abc1234",
                )
                with self.assertRaisesRegex(error_type, message):
                    list(stream)
                self.assertFalse(log.exists())
                self.assertIsNone(chat._conn)

    def test_cancelled_weak_retrieval_yields_nothing_and_writes_no_log(self):
        hits = [{
            "title": "unrelated",
            "path": "x.md",
            "source": "tldr",
            "text": "nothing",
            "dense": 0.10,
            "rrf": 1 / 60,
        }]
        chat = CancellableChat(self.host)
        chat.cancel()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            stream = stream_answer_question(
                "q",
                "full",
                "m",
                retriever=_Retriever(hits),
                chat=chat,
                log_path=log,
                git_sha="abc1234",
            )
            with self.assertRaises(ChatCancelled):
                list(stream)
            self.assertFalse(log.exists())
        self.assertEqual(_StreamingOllama.bodies, [])

    def test_chart_helper_does_not_normalize_incomparable_scores(self):
        chart = retrieval_chart(
            [{"dense": -0.2, "rrf": 0.016, "bm25": 12.0, "path": "doc.md"}],
            threshold=0.30,
            weak=True,
        )
        self.assertTrue(chart.weak)
        self.assertEqual(
            (chart.rows[0].dense_cosine, chart.rows[0].rrf_rank_score, chart.rows[0].bm25_raw),
            (-0.2, 0.016, 12.0),
        )


if __name__ == "__main__":
    unittest.main()
