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
ollama serve
. .venv/bin/activate
streamlit run app/app.py
```

The default UI is Polaris (dark, compact). Modes are BASE / COT / RAG / FULL. The sidebar is read-only (model, index, frozen threshold/k, git, Ollama, warm-up). Export CSV/JSON from the session records, not from `logs/answers.csv`. Compare modes runs BASE then FULL one after the other. `POLARIS_ENABLED=0` restores the old settings sidebar. Fonts are local OFL files under `app/static/` — no CDN.

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
