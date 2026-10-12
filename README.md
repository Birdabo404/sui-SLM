# sui-SLM

Offline Linux and computer-lab help assistant for students. A small local model answers from a local index of Ubuntu and Linux docs. Target machine: ThinkPad T14 Gen 2 (i5 11th gen, Ubuntu). Measured runs boot with `mem=8G`.

Thesis: Enhancing Small Language Models for Low-End Devices using Retrieval Augmented Generation, Knowledge Distillation, and Chain-of-Thought Prompting.

## Setup

On Ubuntu, from this repo:

```
bash install.sh
```

That creates `.venv`, installs Python packages, checks for Ollama, and pulls `qwen3.5:2b`. It then prints the commands to fetch the corpus and build the index. `install.sh` is safe to run again.

Ollama is not installed by the script. Install it first from https://ollama.com/download/linux if `ollama` is missing.

Fetch scripts: after `bash corpus/fetch_corpus.sh`, the man-page and GNOME scripts pick the newest `corpus/snapshot-*` folder. Set `SNAP=` to override.

## Run

```
ollama serve
. .venv/bin/activate
streamlit run app/app.py
```

Pick the model and the mode separately in the sidebar.

- Model: `base` (`qwen3.5:2b`) or `distilled` (Ollama name, empty until you register the GGUF).
- Mode: base, CoT, RAG, RAG+CoT.

That covers the five thesis configs: base, base+CoT, base+RAG, base+RAG+CoT, distilled+RAG+CoT.

RAG uses hybrid MiniLM+BM25 with the tldr lane. If none of the top-k fused chunks has MiniLM cosine of at least 0.30, the app says it does not know. Each answer logs model, mode, latency, app RSS, Ollama RSS, total RSS, and system used memory to `logs/answers.csv`. If `logs/answers.csv` already exists with a different header, it is renamed to `answers.<timestamp>.csv` first.

Eval also prints an abstain-threshold sweep (precision/recall on answerable vs unanswerable questions). Pass `--no-sweep` to skip it.

The app does not call the internet. The embedding model must already be in the Hugging Face cache (install.sh downloads it).

## Eval

JSONL questions need `id`, `question`, `answerable`, `gold_answer`, `source`. Temperature is 0 and the seed is 42.

```
python eval/run_eval.py --questions eval/questions.example.jsonl --out logs/eval.csv
python eval/run_eval.py --questions eval/questions.jsonl --distilled your-student:latest
```

One CSV row per question per config, with the answer, cited sources, latency, RAM, and git commit. The distilled config is skipped until `--distilled` or `OLLAMA_DISTILLED_MODEL` is set.

## Folder layout

```
app/          Streamlit chat UI
corpus/       Fetch scripts for the doc snapshot
rag/          Index builder, hybrid query, spot checks
kd/           Teacher Q&A kit for distillation
eval/         Ablation eval runner
shared/       System prompt used by the app and kd/build_sft.py
```

## Corpus licenses

- tldr-pages: CC BY 4.0
- man-pages: mixed
- Ubuntu and GNOME help: CC BY-SA 3.0
- Ubuntu Server docs: license unverified
