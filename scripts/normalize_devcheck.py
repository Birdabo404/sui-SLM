#!/usr/bin/env python3
"""Compare DEV-only dense retrieval with query normalization off and on."""
import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.query_normalization import normalize_query
from rag.chroma_client import persistent_client
from shared.frozen import load_frozen

DEV_FILE = ROOT / "eval" / "dev.jsonl"
DEFAULT_INDEX = Path(os.environ.get("RAG_INDEX", ROOT / "rag" / "rag-index-v2"))
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PARAPHRASES = (
    ("tuning-tar", "Could you please tell me how to extract a tar.gz file on my Ubuntu laptop?"),
    ("tuning-ssh", "Would you show me how to connect to another lab PC over ssh, please?"),
    ("tuning-apt", "I need to know how to install a package from the terminal on my Ubuntu machine, thanks."),
    ("tuning-chmod", "Can you help me make a script executable on my computer?"),
    ("tuning-chown", "Is it possible to change the owner of a folder on my Linux Mint laptop?"),
    ("tuning-df", "Could you please tell me how to check how much disk space is left?"),
    ("tuning-du", "I want to know how to find which folder is using the most space, thanks."),
    ("tuning-kill", "Would you help me figure out how to stop a frozen program on my machine?"),
    ("tuning-top", "Please show me how to see which processes use the most CPU?"),
    ("tuning-scp", "Can you tell me how to copy a file to another computer on the network, please?"),
    ("tuning-ip", "I need to know how to show my IP address on my Fedora PC."),
    ("tuning-grep", "Could you help me search for text inside files on my computer?"),
    ("tuning-find", "Would you please show me how to find a file by name?"),
    ("tuning-systemctl", "Is it possible to restart a service on my Debian machine, thanks?"),
    ("tuning-git", "Hello, can you please tell me how to clone a git repository on my laptop?"),
)


@dataclass
class DenseIndex:
    collection: object
    model: SentenceTransformer
    description: str
    temporary: tempfile.TemporaryDirectory | None = None

    def top_score(self, query):
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=1,
            include=["distances"],
        )
        distances = result.get("distances") or [[]]
        return None if not distances[0] else 1.0 - float(distances[0][0])


def _load_dev():
    rows = []
    with DEV_FILE.open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append(json.loads(stripped))
    return rows


def _real_index(path):
    return (path / "chunks.jsonl").is_file() and (path / "chroma").is_dir()


def _open_real_index(path):
    collection = persistent_client(path / "chroma").get_collection("linux_lab")
    model_name = collection.metadata.get("embed_model", DEFAULT_MODEL)
    return DenseIndex(
        collection=collection,
        model=SentenceTransformer(model_name, device="cpu"),
        description=f"real index `{path}` using `{model_name}`",
    )


def _build_stand_in(rows):
    temporary = tempfile.TemporaryDirectory(prefix="normalize-devcheck-")
    client = chromadb.PersistentClient(
        path=temporary.name,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection("normalize_dev_stand_in", metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer(DEFAULT_MODEL, device="cpu")
    answerable = [row for row in rows if row.get("answerable") is True]
    documents = [
        f"Linux command documentation for {row['source']}. Usage: {row['gold_answer']}."
        for row in answerable
    ]
    collection.add(
        ids=[row["id"] for row in answerable],
        documents=documents,
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
    )
    return DenseIndex(
        collection=collection,
        model=model,
        temporary=temporary,
        description=(
            "temporary stand-in MiniLM index built only from DEV gold command/source summaries; "
            "the real `rag-index-v2` was unavailable, so these scores are diagnostic and not main-result numbers"
        ),
    )


def _decision(score, threshold):
    return "abstain" if score is None or score < threshold else "answer"


def _measure(index, identifier, group, question, answerable, threshold):
    normalized = normalize_query(question)
    off_score = index.top_score(question)
    on_score = index.top_score(normalized)
    off = _decision(off_score, threshold)
    on = _decision(on_score, threshold)
    return {
        "id": identifier,
        "group": group,
        "question": question,
        "normalized": normalized,
        "answerable": answerable,
        "off_score": off_score,
        "off": off,
        "on_score": on_score,
        "on": on,
    }


def _escape(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def _score(value):
    return "—" if value is None else f"{value:.3f}"


def _summary(rows):
    return {
        "rows": len(rows),
        "flips": sum(row["off"] == "abstain" and row["on"] == "answer" for row in rows),
        "regressions": sum(row["off"] == "answer" and row["on"] == "abstain" for row in rows),
        "should_stay": sum(
            row["answerable"] is False and row["on"] == "abstain"
            for row in rows
        ),
        "should_total": sum(row["answerable"] is False for row in rows),
    }


def _table(rows):
    lines = [
        "| ID | Input | Normalized retrieval query | Off top-1 | Off | On top-1 | On |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {_escape(row['id'])} | {_escape(row['question'])} | "
            f"{_escape(row['normalized'])} | {_score(row['off_score'])} | {row['off']} | "
            f"{_score(row['on_score'])} | {row['on']} |"
        )
    return "\n".join(lines)


def _report(index, dev_rows, paraphrase_rows, threshold):
    dev = _summary(dev_rows)
    paraphrases = _summary(paraphrase_rows)
    summary = (
        "| Set | Rows | Abstain → answer | Answer → abstain | Should-abstain stayed abstained |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| DEV | {dev['rows']} | {dev['flips']} | {dev['regressions']} | "
        f"{dev['should_stay']}/{dev['should_total']} |\n"
        f"| Filler paraphrases | {paraphrases['rows']} | {paraphrases['flips']} | "
        f"{paraphrases['regressions']} | — |"
    )
    return (
        "# Query normalization DEV check\n\n"
        f"> Index: {index.description}.\n\n"
        f"> Decision rule: answer when top-1 MiniLM cosine is at least {threshold:.2f}; "
        "no model is called.\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## DEV questions\n\n{_table(dev_rows)}\n\n"
        f"## Filler-laden DEV paraphrases\n\n{_table(paraphrase_rows)}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    rows = _load_dev()
    threshold = float(load_frozen()["threshold"])
    index = _open_real_index(args.index) if _real_index(args.index) else _build_stand_in(rows)
    by_id = {row["id"]: row for row in rows}
    dev_results = [
        _measure(index, row["id"], "dev", row["question"], row["answerable"], threshold)
        for row in rows
    ]
    paraphrase_results = [
        _measure(
            index,
            f"paraphrase-{number:02d}",
            "paraphrase",
            question,
            by_id[source_id]["answerable"],
            threshold,
        )
        for number, (source_id, question) in enumerate(PARAPHRASES, 1)
    ]
    print(_report(index, dev_results, paraphrase_results, threshold))


if __name__ == "__main__":
    main()
