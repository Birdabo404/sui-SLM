"""Enhanced-only hybrid retrieval primitives.

The frozen app path must not import or call this module. The UI-owned toggle
selects this API only for Enhanced mode; its default keeps today's evaluated
retriever unchanged.
"""
from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Callable, Mapping, Sequence

from app.backend import AppRetrievalError
from rag.query_hybrid import rrf_fuse


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
METHOD_LABEL = "dense+keyword+rerank"
DENSE_LIMIT = 20
KEYWORD_LIMIT = 20
KEEP = 5
RRF_K = 60

Hit = Mapping[str, object]
Search = Callable[[str, int], Sequence[Hit]]


class RerankerModelError(AppRetrievalError):
    """The Enhanced-mode cross-encoder is unavailable or failed."""

    code = "reranker_model_unavailable"
    message = "The Enhanced retrieval reranker is not available in the local cache."
    action = "Use Evaluated config, or cache the Enhanced models while online."


@dataclass(frozen=True)
class HybridResult:
    """Enhanced retrieval output plus the unchanged dense abstention signal."""

    hits: tuple[dict[str, object], ...]
    dense_gate_similarity: float | None
    method: str = METHOD_LABEL


def _hit_id(hit: Hit) -> str:
    value = hit.get("id") or hit.get("path")
    if value is None:
        raise ValueError("hybrid candidates require an id or path")
    return str(value)


def _candidate_text(hit: Hit) -> str:
    return str(hit.get("text") or hit.get("document") or "")


def merge_rrf(
    dense_hits: Sequence[Hit],
    keyword_hits: Sequence[Hit],
    *,
    rrf_k: int = RRF_K,
) -> tuple[dict[str, object], ...]:
    """Merge dense and BM25 rankings with the existing app RRF semantics.

    ``rag.query_hybrid.rrf_fuse`` uses zero-based ranks, so the first item in a
    lane contributes ``1 / (60 + 0)``. Original ``dense`` values are copied
    without normalization or replacement for the later 0.30 abstention gate.
    """

    dense_by_id = {_hit_id(hit): hit for hit in dense_hits}
    keyword_by_id = {_hit_id(hit): hit for hit in keyword_hits}
    dense_ids = list(dense_by_id)
    keyword_ids = list(keyword_by_id)
    scores = rrf_fuse((dense_ids, keyword_ids), k=rrf_k)
    merged = []
    for hit_id in sorted(scores, key=lambda item: (-scores[item], item)):
        item = dict(keyword_by_id.get(hit_id, {}))
        item.update(dense_by_id.get(hit_id, {}))
        methods = []
        if hit_id in dense_by_id:
            methods.append("dense")
        if hit_id in keyword_by_id:
            methods.append("keyword")
        item["rrf"] = float(scores[hit_id])
        item["retrieval_methods"] = tuple(methods)
        merged.append(item)
    return tuple(merged)


@lru_cache(maxsize=2)
def _load_reranker(model_name: str, local_files_only: bool):
    """Load the optional CPU reranker once, only after Enhanced mode asks."""

    try:
        module = importlib.import_module("sentence_transformers")
        return module.CrossEncoder(
            model_name,
            device="cpu",
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise RerankerModelError("cross-encoder could not be loaded") from error


def retrieve_enhanced(
    question: str,
    *,
    dense_search: Search,
    keyword_search: Search,
    reranker=None,
    dense_limit: int = DENSE_LIMIT,
    keyword_limit: int = KEYWORD_LIMIT,
    keep: int = KEEP,
    rrf_k: int = RRF_K,
    model_name: str = RERANKER_MODEL,
    local_files_only: bool = True,
) -> HybridResult:
    """Retrieve and rerank Enhanced-mode candidates without changing refusal.

    ``dense_search`` and ``keyword_search`` are adapters over the existing
    Chroma and BM25 stores. Each is called as ``search(question, limit)``.
    Both adapters must attach the original Chroma ``dense`` similarity to every
    candidate, including keyword-only candidates. Callers must continue to
    apply the frozen 0.30 gate to those values, never to ``rrf`` or ``rerank``.
    """

    dense_hits = tuple(dense_search(question, dense_limit))[:dense_limit]
    keyword_hits = tuple(keyword_search(question, keyword_limit))[:keyword_limit]
    candidates = merge_rrf(dense_hits, keyword_hits, rrf_k=rrf_k)
    if not candidates:
        return HybridResult((), None)
    if any(hit.get("dense") is None for hit in candidates):
        raise ValueError("every Enhanced candidate requires its original dense similarity")

    model = reranker or _load_reranker(model_name, local_files_only)
    pairs = [(question, _candidate_text(hit)) for hit in candidates]
    try:
        scores = tuple(float(value) for value in model.predict(pairs, show_progress_bar=False))
    except Exception as error:
        raise RerankerModelError("cross-encoder reranking failed") from error
    if len(scores) != len(candidates):
        raise RerankerModelError("cross-encoder returned the wrong number of scores")

    ranked = []
    for hit, rerank_score in zip(candidates, scores):
        item = dict(hit)
        item["rerank"] = rerank_score
        ranked.append(item)
    ranked.sort(key=lambda hit: float(hit["rerank"]), reverse=True)
    kept = tuple(ranked[:keep])
    dense_scores = [
        float(hit["dense"])
        for hit in kept
        if hit.get("dense") is not None
    ]
    gate_similarity = max(dense_scores) if dense_scores else None
    return HybridResult(kept, gate_similarity)
