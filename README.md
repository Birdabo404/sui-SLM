# sui-SLM

Offline Linux and computer-lab help assistant for students. A small local model answers from a local index of Ubuntu and Linux docs. Target machine: ThinkPad T14 Gen 2 (i5 11th gen, Ubuntu). Measured runs boot with `mem=8G`.

Thesis: Enhancing Small Language Models for Low-End Devices using Retrieval Augmented Generation, Knowledge Distillation, and Chain-of-Thought Prompting.

## Setup

Needs Python 3.12, 3.13, or 3.14. `install.sh` prints the version first and exits if it is outside that range.

On Ubuntu, from this repo:

```
bash install.sh
```

That creates `.venv`, installs Python packages, checks for Ollama, and pulls `qwen3.5:2b`. It then prints the commands to fetch the corpus and build the index. `install.sh` is safe to run again.

Ollama is not installed by the script. Install it first from https://ollama.com/download/linux if `ollama` is missing.

Fetch scripts: after `bash corpus/fetch_corpus.sh`, the man-page and GNOME scripts pick the newest `corpus/snapshot-*` folder. Set `SNAP=` to override.

chromadb is pinned at 1.0.8. That version cannot read an index built with chromadb 0.6.3. After this install, rebuild `rag/rag-index` (and `rag/rag-index-v2`) with `rag/build_index.py`. Do not copy an old `chroma/` folder forward.

## Run

```
sui-slm
```

`install.sh` installs `sui-slm` in `~/.local/bin` and a desktop launcher. The command starts local Ollama when needed, uses `rag/rag-index-v2` unless `RAG_INDEX` is set, starts Streamlit without file watching, and opens the local app in the browser. It sets offline defaults without overriding explicit environment values.

The default UI is Polaris (dark, compact, with a Light mode toggle in the rail's footer). The left rail holds the run controls, a live monitor and Settings; its chevron hides it, and it stays hidden across reruns and tabs until reopened. It fits without scrolling on screens 600 px tall and up; below that it scrolls rather than hide a control. Chats open as tabs along the top, and History lists every saved chat. One answer and a compare share the same chat column (880 px at most), and the two compare sides stack when it is narrower than 620 px. Modes are the thesis configs BASE / +CoT / +RAG / +RAG+CoT; BASE is the plain model with no documents, and +CoT is a step-by-step prompt (every request sends `think: false`). The model picker offers the default base (`OLLAMA_MODEL`, still the default), `thesis-base-q4km` and the KD student (`thesis-student`, or `OLLAMA_DISTILLED_MODEL`), which runs +RAG+CoT only. It checks Ollama's `/api/tags`: a model that isn't installed is marked as such with the `ollama create` command from `kd/README.md`, and asking it shows that error. Nothing is downloaded, built or swapped for another model. Settings (a dialog) shows the model, index, frozen threshold/k, keep-alive, context, git, Ollama and warm-up, and exports CSV/JSON from the session records, not from `logs/answers.csv`. Compare runs any two model and mode pairs one after the other; when a distilled model is configured, it defaults to base+RAG+CoT versus distilled+RAG+CoT. `POLARIS_ENABLED=0` restores the old settings sidebar. Fonts are local OFL files under `app/static/` — no CDN.

Answers stream into the page as Ollama sends them, one `/api/chat` request per answer and none for an abstention; the card that streams is the card that stays, each block in its own slot. Until the answer's first text, a small ASCII cube (45 × 44 px) tumbles beside a plain label: 48 frames drawn in Python by turning the cube about two axes at once, projecting it in perspective and keeping the nearest face per character cell (a depth buffer), shaded by a fixed light, then played by one CSS animation with no JS. It stays through retrieval, the wait and any reasoning being written under "Show reasoning", goes the moment answer text starts or the answer stops, fails or abstains, and holds its first frame under reduced motion. The request sets `num_ctx` 4096, `temperature` 0.2 and `keep_alive`, and no `num_predict` or `stop`, so the model stops on its own; if Ollama ends a reply at a length limit (`done_reason: length`) the answer says it was cut off. Ollama's stream carries no per-chunk token counts, so the rail monitor's live trace counts characters in the app and says so; it never shows a live tok/s. Every finished answer, including each compare side and every answer reopened from history, shows its response time and Ollama's own tok/s (`eval_count / eval_duration`) with the raw counts beside it, or says tok/s is unavailable or not generated (an abstention never calls the model). The character trace is kept for the current session only; history and exports keep the same fields as before. The monitor's Scores view charts each retrieved chunk's raw scores: dense cosine (the number the 0.30 threshold gates), the RRF rank score and the raw BM25 score. RRF and BM25 are not confidences. BASE retrieves nothing and shows no scores. Stop discards a partial answer; so does switching tabs or opening a chat from History mid-answer, since any click reruns Streamlit. The eval scripts do not use streaming.

In Polaris, the app adds its own format rules to the thesis system prompt (`app/answer_format.py`). For how-to questions it asks for a one-line intro, numbered steps with each command in its own fenced bash block, and `Source: [n]` in +RAG and +RAG+CoT only; for explanations it asks for plain sentences. A 2B model does not always follow this, and the app renders whatever it writes, with three narrow exceptions that never change command bytes. Inside a numbered step that has no code block of its own, an inline `` `span` `` that is plainly a shell command also gets a copy box under the step, and the sentence keeps the span. A span counts only when it parses as shell words, has an argument, and starts (after an optional `sudo`) with a name from a fixed list of lab commands. Names that are also English words (`groups`, `find`) also need a shell-like argument such as a flag, path, number or `<placeholder>`. A bash block in which every line is one complete command starting with a known name is shown as one box per line, so a pipe or `&&` inside a line stays in that line's box. A block with a line ending in `\`, `|`, `&&` or `||`, a heredoc, a `$` prompt, a comment, a shell keyword or indentation stays one box. Commands written after the last step stay with that step only if the answer has one step, the earlier steps have their own boxes, or there is one block right under the step or after its colon. Otherwise they are shown flush with a note that the model did not say which step each belongs to, and no step is given one. `shared/prompt.py` and `engine.build_messages`, which eval and KD use, are unchanged, and eval runs without these rules. Each command renders as a dark code box with a copy button, and step numbers stay correct with code boxes between steps. A `[n]` that matches no retrieved excerpt is removed and named under the answer; BASE and +CoT say they had no documents. tldr-style `{{username}}` is shown as `<username>`; other braces (Jinja, Go templates, `${{ }}`, JSON) are left alone. Reasoning that +CoT or +RAG+CoT writes (a `Reasoning:` section before `Answer:`) is folded under "Show reasoning", closed by default, and thinking stays off. While those modes stream, everything before the `Answer:` line goes under "Show reasoning", numbered or not, so a draft never shows as the answer; the answer starts streaming at that line, and a reply that never writes `Answer:` appears when it ends. A BASE or +RAG reply that opens with a `Reasoning:` or `Steps:` label is read the same way. A finished reply without `Answer:` hides only the block a `Reasoning:` label opens, so text without that label is never hidden. A follow-up such as "explain step 2" means step 2 of the previous answer as shown: the app quotes that step to the model and adds its words to the retrieval query. If the commands were left unassigned, the model sees them listed apart from the steps and is told not to guess. A follow-up asked in compare mode gives each side its own previous step. A single answer after a compare gets both previous steps labelled by side, mode and model, and is asked to answer each separately; a side that was stopped or failed is named as such. The old sidebar UI (`POLARIS_ENABLED=0`) keeps the thesis prompt as is.

The app and bundled lab corpus are English-only. Questions in other languages are outside this thesis build's scope.

Pick the model and the mode separately in the sidebar when Polaris is off.

- Model: `base` (`qwen3.5:2b`) or `distilled` (Ollama name, empty until you register the GGUF).
- Mode: base, CoT, RAG, RAG+CoT.

That covers the five thesis configs: base, base+CoT, base+RAG, base+RAG+CoT, distilled+RAG+CoT.

RAG uses hybrid MiniLM+BM25 with the tldr lane. If none of the top-k fused chunks has MiniLM cosine of at least the threshold in `shared/frozen.json`, the app says it does not know. 0.30 and k=5 there are provisional defaults, not a tuned result. After `--split dev` on the real index, replace those values and commit the sweep CSV. Each answer logs model, mode, latency, app RSS, Ollama RSS, total RSS, system used memory, threshold, and k to `logs/answers.csv`. If `logs/answers.csv` already exists with a different header, it is renamed to `answers.<timestamp>.csv` first.

The app does not call the internet. The embedding model must already be in the Hugging Face cache (install.sh downloads it).

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
