"""App-only backend boundaries and user-safe failure types."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import requests

from rag.query_hybrid import HybridRetriever


class UserFacingBackendError(Exception):
    """Base class for typed backend failures with stable UI-safe copy."""

    code = "backend_error"
    message = "The local backend could not complete this request."
    action = "Try again."

    def __init__(self, detail=None):
        self.detail = "" if detail is None else str(detail)
        super().__init__(self.user_message)

    @property
    def user_message(self):
        return f"{self.message} {self.action}"


class AppRetrievalError(UserFacingBackendError, RuntimeError):
    """A local retrieval failure that the app can report without internals."""

    code = "retrieval_failed"
    message = "The local search index failed while searching."
    action = "Rebuild the index, then try again."


class IndexMissingError(AppRetrievalError):
    """The configured on-disk index does not exist."""

    code = "index_missing"
    message = "The local search index is missing."
    action = "Build the index, then try again."


class IndexLoadError(AppRetrievalError):
    """The on-disk index is incomplete, corrupt, or incompatible."""

    code = "index_corrupt"
    message = (
        "The local search index could not be opened because it is incomplete, "
        "corrupt, or incompatible."
    )
    action = "Rebuild the index, then try again."


class EmbeddingModelError(AppRetrievalError):
    """The index's embedding model is unavailable from the offline cache."""

    code = "embedding_model_missing"
    message = "The embedding model is not available in the local cache."
    action = "Run `bash install.sh` once while online, then try again."


class RetrievalQueryError(AppRetrievalError):
    """A loaded index failed while serving a query."""

    code = "index_query_failed"


class OllamaUnavailable(UserFacingBackendError, requests.ConnectionError):
    """Ollama could not be reached before a response began."""

    code = "ollama_unavailable"
    message = "Ollama is not running or could not be reached."
    action = "Start it with `ollama serve`, then try again."


class OllamaTimedOut(UserFacingBackendError, requests.Timeout):
    """Ollama exceeded the app request timeout."""

    code = "ollama_timeout"
    message = "Ollama took too long to answer, so the request was stopped."
    action = "Try again, or restart Ollama if it keeps happening."


class OllamaDisconnected(UserFacingBackendError, requests.ConnectionError):
    """Ollama disconnected after a response had begun."""

    code = "ollama_disconnected"
    message = "Ollama disconnected before the answer finished."
    action = "Discard the partial answer, then try again."


class OllamaModelMissing(UserFacingBackendError, requests.RequestException):
    """Ollama reported that the requested generation model is absent."""

    code = "ollama_model_missing"
    message = "The selected Ollama model is not installed."
    action = "Install or create the model in Ollama, then try again."


class OllamaResponseError(UserFacingBackendError, requests.RequestException):
    """Ollama returned an unusable response."""

    code = "ollama_response_error"
    message = "Ollama returned a response the app could not use."
    action = "Try again, or restart Ollama if the problem continues."


class AnswerStatus(str, Enum):
    """Non-error terminal status returned by the app streaming path."""

    COMPLETE = "complete"
    ABSTAIN_EMPTY_RETRIEVAL = "abstain_empty_retrieval"
    ABSTAIN_WEAK_RETRIEVAL = "abstain_weak_retrieval"

    @property
    def message(self):
        if self is AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL:
            return "No documents were retrieved for this question."
        if self is AnswerStatus.ABSTAIN_WEAK_RETRIEVAL:
            return "The retrieved documents were too weak to support an answer."
        return "The answer completed."

    @property
    def action(self):
        if self is AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL:
            return "Ask a more specific question or rebuild the index if this keeps happening."
        if self is AnswerStatus.ABSTAIN_WEAK_RETRIEVAL:
            return "Rephrase the question or ask the lab technician."
        return "No action is needed."

    @property
    def user_message(self):
        return f"{self.message} {self.action}"


def answer_status(hits, weak):
    """Classify a completed app answer without changing abstain policy."""
    if not hits:
        return AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL
    if weak:
        return AnswerStatus.ABSTAIN_WEAK_RETRIEVAL
    return AnswerStatus.COMPLETE


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


def load_app_retriever(index_dir, factory=HybridRetriever, is_dir=None):
    """Load the local retriever and classify failures for the app boundary."""
    index_path = Path(index_dir)
    path_is_dir = is_dir or Path.is_dir
    if not path_is_dir(index_path):
        raise IndexMissingError("configured index directory does not exist")
    try:
        return SafeRetriever(factory(index_path))
    except AppRetrievalError:
        raise
    except Exception as error:
        if _looks_like_missing_embedding(error):
            raise EmbeddingModelError("embedding model is not cached") from error
        raise IndexLoadError("local index could not be loaded") from error
