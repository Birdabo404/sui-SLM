"""A stand-in for app/backend.py with the error contract in docs/errors.md (PR #20): the same class tree, codes,
messages and AnswerStatus values, so the UI can be tested with the backend present on a branch that predates it."""
from enum import Enum
from pathlib import Path
from types import ModuleType

import requests


class UserFacingBackendError(Exception):
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
    code = "retrieval_failed"
    message = "The local search index failed while searching."
    action = "Rebuild the index, then try again."


class IndexMissingError(AppRetrievalError):
    code = "index_missing"
    message = "The local search index is missing."
    action = "Build the index, then try again."


class IndexLoadError(AppRetrievalError):
    code = "index_corrupt"
    message = "The local search index could not be opened because it is incomplete, corrupt, or incompatible."
    action = "Rebuild the index, then try again."


class EmbeddingModelError(AppRetrievalError):
    code = "embedding_model_missing"
    message = "The embedding model is not available in the local cache."
    action = "Run `bash install.sh` once while online, then try again."


class RetrievalQueryError(AppRetrievalError):
    code = "index_query_failed"


class GroundingModelError(UserFacingBackendError):  # app/grounding.py, PR #28
    code = "grounding_model_unavailable"
    message = "The optional grounding check is not available in the local cache."
    action = "Turn off grounding, or run `bash install.sh --enhanced` while online."


class OllamaUnavailable(UserFacingBackendError, requests.ConnectionError):
    code = "ollama_unavailable"
    message = "Ollama is not running or could not be reached."
    action = "Start it with `ollama serve`, then try again."


class OllamaTimedOut(UserFacingBackendError, requests.Timeout):
    code = "ollama_timeout"
    message = "Ollama took too long to answer, so the request was stopped."
    action = "Try again, or restart Ollama if it keeps happening."


class OllamaDisconnected(UserFacingBackendError, requests.ConnectionError):
    code = "ollama_disconnected"
    message = "Ollama disconnected before the answer finished."
    action = "Discard the partial answer, then try again."


class OllamaModelMissing(UserFacingBackendError, requests.RequestException):
    code = "ollama_model_missing"
    message = "The selected Ollama model is not installed."
    action = "Install or create the model in Ollama, then try again."


class OllamaResponseError(UserFacingBackendError, requests.RequestException):
    code = "ollama_response_error"
    message = "Ollama returned a response the app could not use."
    action = "Try again, or restart Ollama if the problem continues."


class PlainStreamError(requests.RequestException):
    """app.streaming.OllamaStreamError as it is before PR #20: an untyped RequestException carrying Ollama's own
    text. PR #20 makes it an OllamaResponseError, which hides that text."""


class AnswerStatus(str, Enum):
    COMPLETE = "complete"
    ABSTAIN_EMPTY_RETRIEVAL = "abstain_empty_retrieval"
    ABSTAIN_WEAK_RETRIEVAL = "abstain_weak_retrieval"

    @property
    def user_message(self):  # message + action, as in the backend
        return {
            "complete": "The answer completed. No action is needed.",
            "abstain_empty_retrieval": "No documents were retrieved for this question. Ask a more specific question "
                                       "or rebuild the index if this keeps happening.",
            "abstain_weak_retrieval": "The retrieved documents were too weak to support an answer. Rephrase the "
                                      "question or ask the lab technician.",
        }[self.value]


class SafeRetriever:
    def __init__(self, delegate):
        self.delegate = delegate

    def retrieve(self, question, k=5, tldr_lane=True):
        try:
            return self.delegate.retrieve(question, k=k, tldr_lane=tldr_lane)
        except AppRetrievalError:
            raise
        except Exception as error:
            raise RetrievalQueryError("local retrieval failed") from error


def load_app_retriever(index_dir, factory=None, is_dir=None):
    if not (is_dir or Path.is_dir)(Path(index_dir)):
        raise IndexMissingError("configured index directory does not exist")
    try:
        return SafeRetriever(factory(Path(index_dir)))
    except AppRetrievalError:
        raise
    except Exception as error:
        text = f"{type(error).__module__} {type(error).__name__} {error}".lower()
        hints = ("sentence_transform", "huggingface", "local_files_only", "offline mode", "cached files", "config.json")
        if any(hint in text for hint in hints):
            raise EmbeddingModelError("embedding model is not cached") from error
        raise IndexLoadError("local index could not be loaded") from error


def module():
    fake = ModuleType("app.backend")
    for name, value in globals().items():
        if not name.startswith("_") and name not in ("module", "PlainStreamError"):
            setattr(fake, name, value)
    return fake
