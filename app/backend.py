"""App-only backend boundaries and user-safe failure types."""
from dataclasses import dataclass

import requests

from rag.query_hybrid import HybridRetriever


class AppRetrievalError(RuntimeError):
    """A local retrieval failure that the app can report without internals."""


class IndexLoadError(AppRetrievalError):
    """The on-disk index is incomplete, corrupt, or incompatible."""


class EmbeddingModelError(AppRetrievalError):
    """The index's embedding model is unavailable from the offline cache."""


class RetrievalQueryError(AppRetrievalError):
    """A loaded index failed while serving a query."""


class OllamaUnavailable(requests.ConnectionError):
    """Ollama could not be reached before a response began."""


class OllamaTimedOut(requests.Timeout):
    """Ollama exceeded the app request timeout."""


class OllamaDisconnected(requests.ConnectionError):
    """Ollama disconnected after a response had begun."""


class OllamaModelMissing(requests.RequestException):
    """Ollama reported that the requested generation model is absent."""


class OllamaResponseError(requests.RequestException):
    """Ollama returned an unusable response."""


@dataclass(frozen=True)
class SafeRetriever:
    """Wrap a retriever so query failures do not leak implementation errors."""

    delegate: object

    def retrieve(self, question, k=5, tldr_lane=True):
        try:
            return self.delegate.retrieve(question, k=k, tldr_lane=tldr_lane)
        except AppRetrievalError:
            raise
        except Exception as error:
            raise RetrievalQueryError("local retrieval failed") from error


def _looks_like_missing_embedding(error):
    text = f"{type(error).__module__} {type(error).__name__} {error}".lower()
    hints = (
        "sentence_transform",
        "huggingface",
        "local_files_only",
        "offline mode",
        "cached files",
        "config.json",
    )
    return any(hint in text for hint in hints)


def load_app_retriever(index_dir, factory=HybridRetriever):
    """Load the local retriever and classify failures for the app boundary."""
    try:
        return SafeRetriever(factory(index_dir))
    except AppRetrievalError:
        raise
    except Exception as error:
        if _looks_like_missing_embedding(error):
            raise EmbeddingModelError("embedding model is not cached") from error
        raise IndexLoadError("local index could not be loaded") from error
