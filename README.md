# sui-SLM

Offline Linux and computer-lab help assistant for students. A small local model answers from a local index of Ubuntu and Linux docs. Target machine: ThinkPad T14 Gen 2 (i5 11th gen, Ubuntu). Measured runs boot with `mem=8G`.

Thesis: Enhancing Small Language Models for Low-End Devices using Retrieval Augmented Generation, Knowledge Distillation, and Chain-of-Thought Prompting.

## Quickstart

On the T14, with Ubuntu and Python 3.12, 3.13 or 3.14 (`install.sh` checks and exits otherwise):

1. Install Ollama from https://ollama.com/download/linux. `install.sh` does not install it.
2. From this repo, with the network on:

   ```
   bash install.sh
   ```

   This creates `.venv`, installs the Python packages, caches the MiniLM embedding model, pulls `qwen3.5:2b`, and installs the `sui-slm` launcher in `~/.local/bin` with a desktop entry. Ollama has to be running for the pull; if it isn't, the script says so and stops. It is safe to run again.
3. Fetch the corpus and build the index (needs the network once; the build takes several minutes on CPU):

   ```
   bash corpus/fetch_corpus.sh
   bash corpus/fetch_command_manpages.sh
   bash corpus/fetch_gnome_user_docs.sh
   . .venv/bin/activate
   python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index
   python rag/trim/make_trimmed_index.py --src rag/rag-index --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index-v2
   ```

   The man-page and GNOME scripts use the newest `corpus/snapshot-*` folder; set `SNAP=` to pick another. chromadb is pinned at 1.0.8, which cannot read an index built with chromadb 0.6.3, so rebuild rather than copy an old `chroma/` folder forward.
4. Start the app:

   ```
   sui-slm
   ```

5. Open Settings (bottom of the left rail). Its offline check should read "Ready to answer offline": Ollama reachable, the model installed, the index present with its chunk count, and MiniLM cached. From here on the network can be off.

## Launcher

`sui-slm` (or `python -m app.launcher` from `.venv`) does the following:

- sets offline defaults (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, no telemetry) without overriding values already set;
- uses `OLLAMA_HOST` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL` (default `qwen3.5:2b`) and `RAG_INDEX` (default `rag/rag-index-v2`);
- starts `ollama serve` if Ollama isn't answering on a local address, and exits with "Ollama is not available. Install it or start `ollama serve`." if it still isn't;
- starts Streamlit on 127.0.0.1:8501 without file watching (`--port` picks another port) and opens it in the browser; if the app is already running, it just opens the browser.

`sui-slm --dry-run` prints the Streamlit command and environment and starts nothing. Without the launcher: `ollama serve`, then `.venv/bin/python -m streamlit run streamlit_app.py`.

## Modes

| Mode | What it does |
|---|---|
| BASE | The plain model, no documents. |
| +CoT | A step-by-step prompt, no documents. A `Reasoning:` section it writes is folded under "Show reasoning". |
| +RAG | Retrieves the top 5 chunks (MiniLM + BM25 hybrid, with a tldr lane). If no chunk reaches cosine 0.30, it says it doesn't know and never calls the model. Otherwise it answers from the chunks and cites them as `[n]`. |
| +RAG+CoT | +RAG with the step-by-step prompt. The default. |
| Compare | Runs any two model and mode pairs one after the other and shows them side by side. |

Every request sends `think: false` and `temperature` 0.2. Models: `qwen3.5:2b` (default base, `OLLAMA_MODEL`), `thesis-base-q4km`, and the KD student `thesis-student` (or `OLLAMA_DISTILLED_MODEL`), which runs +RAG+CoT only. The two thesis models are built with `ollama create` from the KD export (`kd/README.md`); the app never downloads, builds or substitutes a model.

## Troubleshooting

Every failure shows as a card in the chat: a title, what happened, and the next step where there is one. The full error goes to the terminal running the app, never to the page. When `app/backend.py` is present, the middle line is its typed error's `user_message`, word for word (codes in `docs/errors.md`); without it, the app classifies plain `requests` and socket errors itself into the same cards under the same codes. The same check runs before you ask: on the landing page when something would block the next question, and in Settings.

| Card (code) | Cause | Next step |
|---|---|---|
| Ollama isn't running (`ollama_unavailable`) | Nothing answers at `OLLAMA_HOST`. | `ollama serve`, or start the app with `sui-slm`, which starts it. |
| *model* isn't installed (`ollama_model_missing`) | Ollama runs but doesn't have the picked model. Nothing is asked, and no other model stands in. | The command on the card: `ollama pull qwen3.5:2b`, or `ollama create thesis-student -f Modelfile` / `ollama create thesis-base-q4km -f Modelfile-base` from the KD export. |
| The lab docs index isn't there (`index_missing`) | `RAG_INDEX` has no `chunks.jsonl` or no `chroma/`. BASE and +CoT still work. | Build it (Quickstart step 3), or set `RAG_INDEX` to an index you have. |
| The lab docs index couldn't be opened (`index_corrupt`) | The folder exists but chromadb can't load it, often an index built with chromadb 0.6.3. | Rebuild it with `rag/build_index.py`; don't copy an old `chroma/` folder. |
| The embedding model isn't on this machine (`embedding_model_missing`) | MiniLM is not in the local Hugging Face cache and the app runs offline. | `bash install.sh` once with the network on. |
| Looking up the lab docs failed (`index_query_failed`) | The index loaded, but a search raised an error. | Ask again; if it repeats, restart the app and read the terminal. |
| No fix on this one | Not an error: nothing was retrieved (`abstain_empty_retrieval`) or no chunk reached cosine 0.30 (`abstain_weak_retrieval`), so the app abstained without calling the model. A line under the abstain text says which. | Rephrase with the command or tool name, or ask the lab technician. |
| answer cut off at the length limit | Not an error: a chip in the answer footer. Ollama ended with `done_reason=length`, at the token cap or the 4096-token context, so the text may stop mid-sentence. | Ask a narrower question, or use a mode without CoT. |
| Ollama took too long (`ollama_timeout`) | No data from Ollama within the client's read timeout (300 s). Usually a model still loading or a busy CPU. | Ask again; if it repeats, restart Ollama. |
| The answer stopped mid-stream (`ollama_disconnected`) | Ollama closed the stream before its final record. The partial answer is discarded, and the card says how many characters were. | `ollama ps` to check it is still running with the model loaded, then ask again. |
| Ollama returned an error (`ollama_response_error`) | Any other error reply from Ollama. Without `app/backend.py` the card shows its first line. | Ask again; if it repeats, restart Ollama. |
| Something in the app failed (`unexpected`) | An unexpected exception; the card names its type only. | Ask again; the traceback is in the terminal. |

The launcher itself prints "Ollama is not available. Install it or start `ollama serve`." when it can't start Ollama: install it, or start it by hand and run `sui-slm` again.

## Known limitations

- 0.30 and k=5 in `shared/frozen.json` are provisional defaults, not tuned values (see Eval), and `eval/test.jsonl` is a format placeholder, not the thesis test set.
- The threshold gates on retrieval similarity only. A question that matches a document closely but isn't answered by it still goes to the model, and a 2B model can then be wrong. Check commands before running them. The app shows commands and never runs them.
- A 2B model does not always follow the answer format (numbered steps, one command per box, `Source: [n]`). The app renders what it writes, with the narrow fixes described under How the app works.
- English only.
- Any click while an answer streams (Stop, a tab, History, Settings) reruns Streamlit and discards the partial answer. Failed and stopped answers are not kept either.
- The live rate in the rail counts characters, since Ollama's stream carries no per-chunk token counts. tok/s comes from Ollama's own counts once an answer ends.
- The offline check looks for MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) in the Hugging Face cache. An index built with `--model` set to another embedding model isn't covered by that row.
- The read timeout (300 s) is fixed in the client and can't be changed from the UI.
- A long answer can stop at the length limit (an app-only token cap where the generation guard is in place, or the 4096-token context). The footer says so, but the rest of the answer is not fetched.
- The old sidebar UI (`POLARIS_ENABLED=0`) still shows plain Streamlit error boxes.

## How the app works

The default UI is Polaris (dark, compact, with a Light mode toggle in the rail's footer). The left rail holds the run controls, a live monitor and Settings; its chevron hides it, and it stays hidden across reruns and tabs until reopened. It fits without scrolling on screens 600 px tall and up; below that it scrolls rather than hide a control. Chats open as tabs along the top, and History lists every saved chat. One answer and a compare share the same chat column (880 px at most), and the two compare sides stack when it is narrower than 620 px. Modes are the thesis configs BASE / +CoT / +RAG / +RAG+CoT; BASE is the plain model with no documents, and +CoT is a step-by-step prompt (every request sends `think: false`). The model picker offers the default base (`OLLAMA_MODEL`, still the default), `thesis-base-q4km` and the KD student (`thesis-student`, or `OLLAMA_DISTILLED_MODEL`), which runs +RAG+CoT only. It checks Ollama's `/api/tags`: a model that isn't installed is marked as such with its `ollama pull` or `ollama create` command (`kd/README.md`), and asking it shows that command in a card. Nothing is downloaded, built or swapped for another model. Settings (a dialog) opens with the offline start check, then shows the model, index, frozen threshold/k, keep-alive, context, git, Ollama and warm-up, and exports CSV/JSON from the session records, not from `logs/answers.csv`. Compare runs any two model and mode pairs one after the other; when a distilled model is configured, it defaults to base+RAG+CoT versus distilled+RAG+CoT. `POLARIS_ENABLED=0` restores the old settings sidebar. Fonts are local OFL files under `app/static/` — no CDN.

Answers stream into the page as Ollama sends them, one `/api/chat` request per answer and none for an abstention; the card that streams is the card that stays, each block in its own slot. Until the answer's first text, a small ASCII cube (45 × 44 px) tumbles beside a plain label: 48 frames drawn in Python by turning the cube about two axes at once, projecting it in perspective and keeping the nearest face per character cell (a depth buffer), shaded by a fixed light, then played by one CSS animation with no JS. It stays through retrieval, the wait and any reasoning being written under "Show reasoning", goes the moment answer text starts or the answer stops, fails or abstains, and holds its first frame under reduced motion. The request sets `num_ctx` 4096, `temperature` 0.2 and `keep_alive`, and no `num_predict` or `stop`, so the model stops on its own; if Ollama ends a reply at a length limit (`done_reason: length`) the answer says it was cut off. Ollama's stream carries no per-chunk token counts, so the rail monitor's live trace counts characters in the app and says so; it never shows a live tok/s. Every finished answer, including each compare side and every answer reopened from history, shows its response time and Ollama's own tok/s (`eval_count / eval_duration`) with the raw counts beside it, or says tok/s is unavailable or not generated (an abstention never calls the model). The character trace is kept for the current session only; history and exports keep the same fields as before. The monitor's Scores view charts each retrieved chunk's raw scores: dense cosine (the number the 0.30 threshold gates), the RRF rank score and the raw BM25 score. RRF and BM25 are not confidences. BASE retrieves nothing and shows no scores. Stop discards a partial answer; so does switching tabs or opening a chat from History mid-answer, since any click reruns Streamlit. The eval scripts do not use streaming.

In Polaris, the app adds its own format rules to the thesis system prompt (`app/answer_format.py`). For how-to questions it asks for a one-line intro, numbered steps with each command in its own fenced bash block, and `Source: [n]` in +RAG and +RAG+CoT only; for explanations it asks for plain sentences. A 2B model does not always follow this, and the app renders whatever it writes, with three narrow exceptions that never change command bytes. Inside a numbered step that has no code block of its own, an inline `` `span` `` that is plainly a shell command also gets a copy box under the step, and the sentence keeps the span. A finished answer that is only prose (no numbered steps, no code blocks, like the student's `Answer:`) gets one box per distinct such command, under the paragraph it first appears in. A span counts only when it parses as shell words, has an argument, and starts (after an optional `sudo`) with a name from a fixed list of lab commands. Names that are also English words (`groups`, `find`) also need a shell-like argument such as a flag, path, number or `<placeholder>`. A bash block in which every line is one complete command starting with a known name is shown as one box per line, so a pipe or `&&` inside a line stays in that line's box. A block with a line ending in `\`, `|`, `&&` or `||`, a heredoc, a `$` prompt, a comment, a shell keyword or indentation stays one box. Commands written after the last step stay with that step only if the answer has one step, the earlier steps have their own boxes, or there is one block right under the step or after its colon. Otherwise they are shown flush with a note that the model did not say which step each belongs to, and no step is given one. `shared/prompt.py` and `engine.build_messages`, which eval and KD use, are unchanged, and eval runs without these rules. Each command renders as a dark code box with a copy button, and step numbers stay correct with code boxes between steps. After a click the button shows a check and "Copied" for 1.5 s, then its icon again, on every box (compare sides and reopened chats too) and the same in Light mode, since the boxes are dark in both. That is the app's only script (`app/copy-feedback.js`): Streamlit's button gives no sign it copied, so a hidden, zero-height `components.html` iframe adds the script to the page once. A `[n]` that matches no retrieved excerpt is removed and named under the answer; BASE and +CoT say they had no documents. tldr-style `{{username}}` is shown as `<username>`; other braces (Jinja, Go templates, `${{ }}`, JSON) are left alone. Only what the model marks as reasoning is folded under "Show reasoning", closed by default, and thinking stays off: a `Reasoning:` (or `Thinking:`) section up to the `Answer:` line, a `<think>` block, or the KD student's `Steps:` block when an `Answer:` line follows it. The student's steps are the teacher's reasoning (`kd/teacher_generate.py` asks for "short reasoning steps that show how the excerpt leads to the answer"), so the card shows its `Answer:` sentences and `Source:`; a `Steps:` reply that never writes `Answer:` is shown as the answer once it ends. Everything else is the answer and streams as it arrives, in every mode. A finished answer is never empty: reasoning with nothing after it, an unclosed `<think>`, or a reasoning section holding commands when the part after `Answer:` has none are shown as the answer. ChatML markers a model writes as text (`<|im_start|>assistant`, `<|im_end|>`, `<|endoftext|>`) are removed, never cut at, so no words are dropped. A follow-up such as "explain step 2" means step 2 of the previous answer as shown, never a line folded under "Show reasoning" (an answer with no numbered steps is said to have none): the app quotes that step to the model and adds its words to the retrieval query. If the commands were left unassigned, the model sees them listed apart from the steps and is told not to guess. A follow-up asked in compare mode gives each side its own previous step. A single answer after a compare gets both previous steps labelled by side, mode and model, and is asked to answer each separately; a side that was stopped or failed is named as such. The old sidebar UI (`POLARIS_ENABLED=0`) keeps the thesis prompt as is.

The app and bundled lab corpus are English-only. Questions in other languages are outside this thesis build's scope.

Pick the model and the mode separately in the sidebar when Polaris is off.

- Model: `base` (`qwen3.5:2b`) or `distilled` (Ollama name, empty until you register the GGUF).
- Mode: base, CoT, RAG, RAG+CoT.

That covers the five thesis configs: base, base+CoT, base+RAG, base+RAG+CoT, distilled+RAG+CoT.

RAG uses hybrid MiniLM+BM25 with the tldr lane. If none of the top-k fused chunks has MiniLM cosine of at least the threshold in `shared/frozen.json`, the app says it does not know. 0.30 and k=5 there are provisional defaults, not a tuned result. After `--split dev` on the real index, replace those values and commit the sweep CSV. Each answer logs model, mode, latency, app RSS, Ollama RSS, total RSS, system used memory, threshold, and k to `logs/answers.csv`. If `logs/answers.csv` already exists with a different header, it is renamed to `answers.<timestamp>.csv` first.

The app does not call the internet. Chats are saved locally in `~/.local/share/sui-slm` (or `$XDG_DATA_HOME/sui-slm`).

## Eval

`--split` is required: `dev` or `test`. JSONL fields are `id`, `question`, `answerable`, `gold_answer`, `source`. Optional `set` labels the row as `tuning`, `unanswerable`, or `held-out`. Temperature is 0 and the seed is 42.

- `eval/dev.jsonl`: the 24 ranking-tuning questions (`set=tuning`) plus unanswerable questions (`set=unanswerable`). Threshold and top-k are swept here only.
- `eval/test.jsonl`: placeholder example of the JSONL format, not the thesis test set. The real set is 150-200 held-out questions written by the student. Do not report results from this placeholder.
- `shared/frozen.json`: MiniLM cosine threshold and top-k. Provisional defaults (0.30 and 5) until replaced after `--split dev` on the real index. Every eval CSV row records `split`, `threshold`, and `k`.

```
python eval/run_eval.py --split dev --out logs/eval-dev.csv
python eval/run_eval.py --split test --no-sweep --questions eval/your-heldout.jsonl --out logs/eval-test.csv --distilled your-student:latest
```

`--split test` without `--no-sweep` exits with an error. `--split test` also rejects `--threshold`, `-k`, and `--frozen`. The distilled config is skipped until `--distilled` or `OLLAMA_DISTILLED_MODEL` is set.

## Folder layout

```
app/          Streamlit chat UI
corpus/       Fetch scripts for the doc snapshot
rag/          Index builder, hybrid query, spot checks
kd/           Teacher Q&A kit for distillation
eval/         Ablation eval runner
shared/       System prompt and provisional threshold/k (app + kd/build_sft.py)
```

## Corpus licenses

- tldr-pages: CC BY 4.0
- man-pages: mixed
- Ubuntu and GNOME help: CC BY-SA 3.0
- Ubuntu Server docs: license unverified
