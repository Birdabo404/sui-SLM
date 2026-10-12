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

## Run

```
ollama serve
. .venv/bin/activate
streamlit run app/app.py
```

Default model is `qwen3.5:2b` at `http://127.0.0.1:11434`. Change the name in the sidebar (or set `OLLAMA_MODEL`) when you swap in a distilled GGUF registered with Ollama.

Sidebar ablation modes: base, CoT only, RAG only, full. Full uses hybrid MiniLM+BM25 retrieval with the tldr lane. Weak hits return "I don't know". Each answer logs latency and peak RSS to `logs/answers.csv`.

The app does not call the internet. The embedding model must already be in the Hugging Face cache (install.sh downloads it).

## Folder layout

```
app/          Streamlit chat UI
corpus/       Fetch scripts for the doc snapshot
rag/          Index builder, hybrid query, spot checks
kd/           Teacher Q&A kit for distillation
shared/       System prompt used by the app and kd/build_sft.py
```

## Corpus licenses

- tldr-pages: CC BY 4.0
- man-pages: mixed
- Ubuntu and GNOME help: CC BY-SA 3.0
- Ubuntu Server docs: license unverified
