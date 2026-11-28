# sui-SLM

Offline Linux and computer-lab help assistant for students. A small local model answers from a local index of Ubuntu and Linux docs.

Thesis: Enhancing Small Language Models for Low-End Devices Using Retrieval-Augmented Generation, Knowledge Distillation, and Chain-of-Thought Prompting.

Authors: Clyde Heinrich Lontoc and John Bert P. Cruz, Caraga State University.

The live app is Polaris (`app/polaris.py`): compare mode, chat history, a reasoning panel, command code blocks with copy buttons, numbered sources, response time and tok/s, `num_predict=512` on app requests, and a rule-based identity response. Eval does not use those app-only pieces. The app and the bundled lab corpus are English-only.

## Measured machine

Target laptop: ThinkPad T14 Gen 2 (i5 11th gen, Ubuntu).

Official speed runs booted the laptop with `mem=10752M`. After that, `free -h` shows about 8.0 GB usable RAM plus 4 GB swap. `mem=8G` only gave about 5.5 GB usable, so it was not used.

For the 4B teacher runs only, the Ollama prompt cache was turned off with `LLAMA_ARG_CACHE_RAM=0`.

## Quick start for lab staff

Needs Python 3.12, 3.13, or 3.14. `install.sh` prints the version first and exits if it is outside that range.

1. Install Ollama from https://ollama.com/download/linux. `install.sh` does not install it.
2. Start Ollama, then from this repo:

   ```
   ollama serve
   bash install.sh
   ```

   That creates `.venv`, installs Python packages, caches MiniLM, pulls `qwen3.5:2b`, and installs the `sui-slm` launcher in `~/.local/bin`. It is safe to run again. If `ollama` is missing or not running, the script says so and stops.
3. Fetch the corpus (needs the network once). The man-page and GNOME scripts use the newest `corpus/snapshot-*` folder; set `SNAP=` to pick another:

   ```
   bash corpus/fetch_corpus.sh
   bash corpus/fetch_command_manpages.sh
   bash corpus/fetch_gnome_user_docs.sh
   ```
4. Build the eval index on the laptop into `rag/lab-index`. Default sources skip Arch Wiki and Ubuntu Server docs (Server docs have no license). Do not copy an old `rag-index` or `rag-index-v2` forward. chromadb 1.0.8 cannot read a 0.6.3 `chroma/` folder.

   ```
   . .venv/bin/activate
   python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/lab-index
   ```
5. Register the thesis GGUFs from the KD export (`kd/README.md`). Thesis configs 1-4 use `thesis-base-q4km`, not stock `qwen3.5:2b`:

   ```
   ollama create thesis-student -f Modelfile
   ollama create thesis-base-q4km -f Modelfile-base
   ```
6. Start the app:

   ```
   sui-slm
   ```

   `sui-slm` sets `RAG_INDEX` to `rag/rag-index-v2` if it is unset (the tree the live demo used). After a fresh `lab-index` build, point it at that folder:

   ```
   RAG_INDEX=rag/lab-index sui-slm
   ```

   Without the launcher: `ollama serve`, then `.venv/bin/python -m streamlit run streamlit_app.py`. `sui-slm --dry-run` prints the Streamlit command and environment and starts nothing.

Open the system check (bottom of the left rail, or Settings, System). The last line should end "ready to answer offline". After that the network can be off.

## Launcher

`sui-slm` (or `python -m app.launcher` from `.venv`):

- sets offline defaults (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, no telemetry) without overriding values already set
- uses `OLLAMA_HOST` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL` (default `qwen3.5:2b`) and `RAG_INDEX` (default `rag/rag-index-v2`)
- starts `ollama serve` if Ollama is not answering on a local address, and exits with "Ollama is not available. Install it or start `ollama serve`." if it still is not
- starts Streamlit on 127.0.0.1:8501 without file watching (`--port` picks another port) and opens it in the browser; if the app is already running, it just opens the browser

`NORMALIZE_QUESTION_BEFORE_SEARCH=1` turns on the optional retrieval-query normalizer (`0` turns it off). It is off by default, makes no model call, and never changes the question sent to Ollama.

## Modes

| Mode | What it does |
|---|---|
| BASE | The plain model, no documents. |
| +CoT | A step-by-step prompt, no documents. A `Reasoning:` section it writes is shown under "Reasoning (chain of thought)" above the answer. |
| +RAG | Retrieves the top 5 chunks (MiniLM + BM25 hybrid, with a tldr lane). If no chunk reaches cosine 0.30, it says it does not know and never calls the model. Otherwise it answers from the chunks and cites them as `[n]`. |
| +RAG+CoT | +RAG with the step-by-step prompt. The default. |
| Compare | Runs any two model and mode pairs one after the other and shows them side by side. |

Every app request sends `think: false` and `temperature` 0.2. Eval uses temperature 0 and seed 42. Models: `qwen3.5:2b` (default base, `OLLAMA_MODEL`), `thesis-base-q4km`, and the KD student `thesis-student` (or `OLLAMA_DISTILLED_MODEL`), which runs +RAG+CoT only. The app never downloads, builds, or substitutes a model.

That covers the five thesis configs: base, base+CoT, base+RAG, base+RAG+CoT, distilled+RAG+CoT.

## How the app works

Themes, Settings, History, and the system check live in the left rail. Chats open as tabs. History keeps saved chats on disk under `~/.local/share/sui-slm` (or `$XDG_DATA_HOME/sui-slm`). `POLARIS_ENABLED=0` restores the old sidebar UI.

Answers stream from Ollama. Until the first answer text, a small ASCII cube plays next to a label. App requests set `num_ctx` 4096, `temperature` 0.2, `keep_alive`, and `num_predict` 512 (`APP_NUM_PREDICT` in `app/streaming.py`, including both compare sides). Eval (`eval/run_eval.py` through `app.engine`) does not send `num_predict`. If Ollama ends a reply at a length limit, the card says it was cut off.

A finished answer shows response time and Ollama's tok/s (`eval_count / eval_duration`). The live rate in the rail counts characters, because the stream has no per-chunk token counts.

Cited `[n]` numbers are named from the retrieved excerpts. Command lines render as code boxes with a copy button (`app/copy-feedback.js`). Chain of thought opens by default under "Reasoning (chain of thought)" and can be collapsed.

A rule-based identity response answers questions about the assistant itself (`app/identity.py`) before retrieval, with no call to Ollama. The card is labelled "Fixed response · not model-generated". Eval never uses it.

`shared/prompt.py` and `engine.build_messages`, which eval and KD use, are unchanged. The Polaris UI adds format rules on top for display (`app/answer_format.py`).

The app does not call the internet. The embedding model must already be in the Hugging Face cache (`install.sh` downloads it). Each answer logs model, mode, latency, RSS, threshold, and k to `logs/answers.csv`. If that file already exists with a different header, it is renamed to `answers.<timestamp>.csv` first.

## Troubleshooting

Failures show as a card in the chat: a title, what happened, and the next step. The full error goes to the terminal, not the page.

| Card (code) | Cause | Next step |
|---|---|---|
| Ollama isn't running (`ollama_unavailable`) | Nothing answers at `OLLAMA_HOST`. | `ollama serve`, or start the app with `sui-slm`. |
| *model* isn't installed (`ollama_model_missing`) | Ollama runs but does not have the picked model. | `ollama pull qwen3.5:2b`, or `ollama create thesis-student -f Modelfile` / `ollama create thesis-base-q4km -f Modelfile-base` from the KD export. |
| The lab docs index isn't there (`index_missing`) | `RAG_INDEX` has no `chunks.jsonl` or no `chroma/`. BASE and +CoT still work. | Build `rag/lab-index` (Quick start step 4), or set `RAG_INDEX` to an index you have. |
| The lab docs index couldn't be opened (`index_corrupt`) | Often an index built with chromadb 0.6.3. | Rebuild with `rag/build_index.py`. Do not copy an old `chroma/` folder. |
| The embedding model isn't on this machine (`embedding_model_missing`) | MiniLM is not in the local Hugging Face cache. | `bash install.sh` once with the network on. |
| No fix on this one | Nothing was retrieved, or no chunk reached cosine 0.30, so the app abstained. | Rephrase with the command or tool name, or ask the lab technician. |

`index_ok` exits if `chunks.jsonl` still has `source=ubuntu-server-docs` unless `INCLUDE_UBUNTU_SERVER=1`. The launcher sets that to 1 so the older `rag-index-v2` demo tree can load; a new `lab-index` build does not include Server docs.

## Eval

`--split` is required: `dev` or `test`. JSONL fields are `id`, `question`, `answerable`, `gold_answer`, `source`. Optional `set` labels the row as `tuning`, `unanswerable`, or `held-out`. Temperature is 0 and the seed is 42. Every request sends `think: false`.

- `eval/dev.jsonl`: the 24 ranking-tuning questions (`set=tuning`) plus unanswerable questions (`set=unanswerable`). Threshold and top-k are swept here only.
- `eval/test.jsonl`: placeholder example of the JSONL format, not the thesis test set. The held-out test set used in the thesis is 30 questions. Do not report results from this placeholder.
- `shared/frozen.json`: MiniLM cosine threshold 0.30 and top-k 5. These are the frozen values used in the evaluation.

```
python eval/run_eval.py --split dev --out logs/eval-dev.csv
python eval/run_eval.py --split test --no-sweep --questions eval/your-heldout.jsonl --out logs/eval-test.csv --distilled your-student:latest
```

`--split test` without `--no-sweep` exits with an error. `--split test` also rejects `--threshold`, `-k`, and `--frozen`. The distilled config is skipped until `--distilled` or `OLLAMA_DISTILLED_MODEL` is set.

## Folder layout

```
app/          Streamlit chat UI (Polaris in app/polaris.py)
corpus/       Fetch scripts for the doc snapshot
rag/          Index builder, hybrid query, spot checks
kd/           Teacher Q&A kit for distillation (Colab notebook: kd/colab_distill.ipynb)
eval/         Ablation eval runner
shared/       System prompt and frozen threshold/k (app + kd/build_sft.py)
docs/         Chapter 3 methodology outline
```

## Corpus licenses

- tldr-pages: CC BY 4.0
- man-pages: mixed
- Ubuntu desktop help: CC BY-SA 3.0 (ubuntu-docs `.deb` copyright, `Files: ubuntu-help/*/*.page*`; `ubuntu-docs_26.04.1/copyright`). `help.ubuntu.com/legal.html` has said 4.0; we cite the package copyright.
- GNOME help: CC BY-SA 3.0
- Ubuntu Server docs: not in the default corpus (no LICENSE file). `INCLUDE_UBUNTU_SERVER=1` to fetch anyway.
