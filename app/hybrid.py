"""Enhanced-only hybrid retrieval primitives.

The frozen app path must not import or call this module. The UI-owned toggle
selects this API only for Enhanced mode; its default keeps today's evaluated
retriever unchanged.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import importlib
from typing import Callable, Mapping, Sequence

from app.backend import AppRetrievalError, RetrievalQueryError
from app.engine import WEAK_DENSE, is_weak
from rag.query_hybrid import HybridRetriever, dense_scores, rrf_fuse, tokenize


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
METHOD_LABEL = "dense+keyword+rerank"
DENSE_LIMIT = 20
KEYWORD_LIMIT = 20
KEEP = 5
RRF_K = 60
RERANKER_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

Hit = Mapping[str, object]
Search = Callable[[str, int], Sequence[Hit]]


class RerankerModelError(AppRetrievalError):
    """The Enhanced-mode cross-encoder is unavailable or failed."""

    code = "reranker_model_unavailable"
    message = "The Enhanced retrieval reranker is not available in the local cache."
    action = "Use Evaluated config, or run `bash install.sh --enhanced` while online."


@dataclass(frozen=True)
class HybridResult:
    """Enhanced retrieval output plus the unchanged dense abstention signal."""

    hits: tuple[dict[str, object], ...]
    dense_gate_similarity: float | None
    gate_hits: tuple[dict[str, object], ...] = ()
    weak: bool = False
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


def cache_reranker_model(
    *,
    model_name: str = RERANKER_MODEL,
    revision: str = RERANKER_REVISION,
    local_files_only: bool = True,
) -> str:
    """Cache the pinned CPU reranker, or verify that its cache is complete."""

    try:
        hub = importlib.import_module("huggingface_hub")
        return str(
            hub.snapshot_download(
                repo_id=model_name,
                revision=revision,
                local_files_only=local_files_only,
                allow_patterns=RERANKER_FILES,
            )
        )
    except Exception as error:
        raise RerankerModelError("cross-encoder assets are not cached") from error


@lru_cache(maxsize=2)
def _load_reranker(model_name: str, revision: str, local_files_only: bool):
    """Load the optional CPU reranker once, only after Enhanced mode asks."""

    try:
        module = importlib.import_module("sentence_transformers")
        model_path = cache_reranker_model(
            model_name=model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        return module.CrossEncoder(
            model_path,
            device="cpu",
            local_files_only=True,
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
    model_revision: str = RERANKER_REVISION,
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

    model = reranker or _load_reranker(model_name, model_revision, local_files_only)
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
    gate_dense_scores = [
        float(hit["dense"])
        for hit in kept
        if hit.get("dense") is not None
    ]
    gate_similarity = max(gate_dense_scores) if gate_dense_scores else None
    return HybridResult(kept, gate_similarity)


def _first_row(value) -> list:
    if value is None or len(value) == 0 or value[0] is None:
        return []
    return list(value[0])


def _max_dense(hits: Sequence[Hit]) -> float | None:
    values = [float(hit["dense"]) for hit in hits if hit.get("dense") is not None]
    return max(values) if values else None


def retrieve_hybrid(
    question: str,
    retriever: HybridRetriever,
    *,
    reranker=None,
    dense_limit: int = DENSE_LIMIT,
    keyword_limit: int = KEYWORD_LIMIT,
    keep: int = KEEP,
    rrf_k: int = RRF_K,
    model_name: str = RERANKER_MODEL,
    model_revision: str = RERANKER_REVISION,
    local_files_only: bool = True,
) -> HybridResult:
    """Run the real Enhanced pipeline while preserving frozen abstention.

    The same ``HybridRetriever.retrieve(question, k=5, tldr_lane=True)`` call
    used by the evaluated path supplies ``gate_hits`` and ``weak``. Enhanced
    ranking changes only the five excerpts sent to generation; it cannot
    change the evaluated refusal decision.
    """

    try:
        gate_hits = tuple(retriever.retrieve(question, k=KEEP, tldr_lane=True))
        weak = is_weak(gate_hits, threshold=WEAK_DENSE)
        source = getattr(retriever, "delegate", retriever)
        query_embedding = source.model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()
        where = (
            {"distro": {"$ne": source.exclude_distro}}
            if source.exclude_distro
            else None
        )
        dense_result = source.col.query(
            query_embeddings=query_embedding,
            n_results=dense_limit,
            where=where,
        )
        dense_ids = [
            str(hit_id)
            for hit_id in _first_row(dense_result.get("ids"))[:dense_limit]
        ]

        bm25_values = source.bm25.get_scores(tokenize(question))
        keyword_indexes = sorted(
            range(len(source.rows)),
            key=lambda index: -bm25_values[index],
        )[:keyword_limit]
        keyword_ids = [str(source.rows[index]["id"]) for index in keyword_indexes]
        candidate_ids = list(dict.fromkeys([*dense_ids, *keyword_ids]))
        similarities = dense_scores(
            source.col,
            query_embedding,
            candidate_ids,
            dense_result,
        )
        rows_by_id = {str(row["id"]): row for row in source.rows}

        def candidates(ids):
            return tuple(
                {
                    **dict(rows_by_id[hit_id]),
                    "dense": similarities.get(hit_id),
                }
                for hit_id in ids
                if hit_id in rows_by_id
            )

        dense_hits = candidates(dense_ids)
        keyword_hits = candidates(keyword_ids)
        result = retrieve_enhanced(
            question,
            dense_search=lambda query, limit: dense_hits[:limit],
            keyword_search=lambda query, limit: keyword_hits[:limit],
            reranker=reranker,
            dense_limit=dense_limit,
            keyword_limit=keyword_limit,
            keep=keep,
            rrf_k=rrf_k,
            model_name=model_name,
            model_revision=model_revision,
            local_files_only=local_files_only,
        )
        return replace(
            result,
            dense_gate_similarity=_max_dense(gate_hits),
            gate_hits=tuple(dict(hit) for hit in gate_hits),
            weak=weak,
        )
    except AppRetrievalError:
        raise
    except Exception as error:
        raise RetrievalQueryError("Enhanced retrieval failed") from error
