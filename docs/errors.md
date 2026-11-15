# App backend error contract

The app backend exposes typed failures and terminal answer statuses so the UI
does not need to inspect exception text. This contract applies only to the app
path. It does not change evaluation behavior or settings.

## UI integration

Catch `UserFacingBackendError` around retriever loading and answer iteration.
Show its `user_message` exactly as written below. Do not show `detail`, the
exception cause, or a traceback. If iteration fails after text has streamed,
discard the partial answer before showing the error.

Read `RetrievalChart.status` when retrieval finishes, or
`StreamingAnswer.result["status"]` after successful iteration, to distinguish a
normal answer from an abstention. Statuses are `AnswerStatus` values, not
exceptions. Existing abstention text and the threshold decision remain
unchanged.

## Failures

| Situation | Type and code | User message |
| --- | --- | --- |
| Ollama is stopped or unreachable | `OllamaUnavailable` / `ollama_unavailable` | Ollama is not running or could not be reached. Start it with `ollama serve`, then try again. |
| The selected Ollama model is missing | `OllamaModelMissing` / `ollama_model_missing` | The selected Ollama model is not installed. Install or create the model in Ollama, then try again. |
| The embedding model is absent from the offline cache | `EmbeddingModelError` / `embedding_model_missing` | The embedding model is not available in the local cache. Run `bash install.sh` once while online, then try again. |
| The configured index directory is missing | `IndexMissingError` / `index_missing` | The local search index is missing. Build the index, then try again. |
| The index is incomplete, corrupt, or incompatible | `IndexLoadError` / `index_corrupt` | The local search index could not be opened because it is incomplete, corrupt, or incompatible. Rebuild the index, then try again. |
| A loaded index fails during search | `RetrievalQueryError` / `index_query_failed` | The local search index failed while searching. Rebuild the index, then try again. |
| Ollama exceeds the request timeout | `OllamaTimedOut` / `ollama_timeout` | Ollama took too long to answer, so the request was stopped. Try again, or restart Ollama if it keeps happening. |
| Ollama disconnects after generation starts | `OllamaDisconnected` / `ollama_disconnected` | Ollama disconnected before the answer finished. Discard the partial answer, then try again. |
| Ollama returns an unusable response | `OllamaResponseError` / `ollama_response_error` | Ollama returned a response the app could not use. Try again, or restart Ollama if the problem continues. |

## Non-error terminal statuses

| Situation | Status | User message |
| --- | --- | --- |
| Generation completes normally | `AnswerStatus.COMPLETE` / `complete` | The answer completed. No action is needed. |
| Retrieval returns no documents and the app abstains | `AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL` / `abstain_empty_retrieval` | No documents were retrieved for this question. Ask a more specific question or rebuild the index if this keeps happening. |
| Retrieval is below the existing confidence threshold and the app abstains | `AnswerStatus.ABSTAIN_WEAK_RETRIEVAL` / `abstain_weak_retrieval` | The retrieved documents were too weak to support an answer. Rephrase the question or ask the lab technician. |
