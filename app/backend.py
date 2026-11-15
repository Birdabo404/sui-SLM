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


def backend_error_text(
    error,
    *,
    host="",
    index_dir="",
    model="",
    model_command="",
):
    """Return one plain app message without exposing exception internals."""
    if isinstance(error, EmbeddingModelError):
        return (
            "The embedding model is not available in the local cache. "
            "Run `bash install.sh` once while online, then try again."
        )
    if isinstance(error, IndexLoadError):
        return (
            f"The local search index at {index_dir} could not be opened. "
            "It may be incomplete or incompatible. Rebuild the index, then try again."
        )
    if isinstance(error, AppRetrievalError):
        return (
            "The local search index failed while searching. "
            "Rebuild the index, then try again."
        )
    if isinstance(error, OllamaModelMissing):
        if model_command.startswith("ollama create"):
            return (
                f"{model} isn't installed in Ollama on this machine. "
                "The exported GGUF and Modelfile are not included in this repository. "
                f"Copy them to this machine, then run `{model_command}` with their actual path."
            )
        if model_command:
            return (
                f"{model} isn't installed in Ollama on this machine. "
                f"Install it with `{model_command}`, then ask again."
            )
        return (
            f"The Ollama model {model!r} is not installed. "
            "Install or create it in Ollama, then try again."
        )
    if isinstance(error, OllamaTimedOut):
        return (
            "Ollama took too long to answer. The request was stopped. "
            "Try again, or restart Ollama if it keeps happening."
        )
    if isinstance(error, OllamaDisconnected):
        return (
            "Ollama disconnected before the answer finished. "
            "The partial answer was discarded; try again."
        )
    if isinstance(error, (OllamaUnavailable, requests.ConnectionError)):
        return (
            f"Ollama is not running or could not be reached at {host}. "
            "Start it with `ollama serve`, then try again."
        )
    return (
        "Ollama could not finish this answer. "
        "Try again, or restart Ollama if the problem continues."
    )


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
