# Chapter 3 Methodology (outline)

Matches what the repo actually builds. This is a section outline, not chapter prose. Lines marked TODO are facts the student must supply (counts, dates, measurements, instrument items). Do not invent them.

## 3.1 System design

- Offline Linux / computer-lab help assistant for first- and second-year IT students.
- Target machine: ThinkPad T14 Gen 2 (i5 11th gen, Ubuntu). Measured runs boot with `mem=8G`.
- Local chat UI: Streamlit (`app/app.py`). Local model: Ollama on `127.0.0.1:11434`. Base model name: `qwen3.5:2b`.
- No outbound calls at answer time. MiniLM must already be in the Hugging Face cache (`install.sh` downloads it).
- Shared system prompt in `shared/prompt.py` is used by the chat app and by `kd/build_sft.py`. Do not fork the text.
- Answer log: `logs/answers.csv` (model, mode, latency, app RSS, Ollama RSS, total RSS, system used memory, threshold, k). Stale headers are rotated, not overwritten.
- Python: 3.12–3.14. CPU torch 2.9.1 from the pytorch CPU index. chromadb 1.0.8 (rebuild the index after that pin).
- TODO: lab Ubuntu version on the T14.
- TODO: photo or diagram of the laptop + Ollama + Streamlit + index layout.

## 3.2 Corpus and licenses

- Snapshot folder: `corpus/snapshot-*`. Fetch scripts pick the newest folder unless `SNAP=` is set.
- Sources in the default v2 index (`rag/rag-index-v2`):
  - tldr-pages: CC BY 4.0
  - command-manpages: per package (see `licenses/command-manpages`)
  - man-pages: mixed (see `licenses/man-pages-LICENSES`); sections 1/5/7/8 only (2/3 are C APIs)
  - Ubuntu desktop help: CC BY-SA 3.0
  - GNOME user docs: CC BY-SA 3.0
  - Ubuntu Server docs: license unverified (repo has no LICENSE; `docs/conf.py` licence field empty). Treat as unverified until Canonical confirms. Pin by commit hash, not tarball sha256.
- Arch Wiki is on disk for old indexes only. Dropped from default `build_index.py --sources` and from v2 (pacman/AUR answers are wrong on the Ubuntu lab image).
- English only. Alias stubs and link-list landing pages are filtered.
- TODO: snapshot date actually used on the T14 (`corpus/snapshot-YYYY-MM-DD`).
- TODO: chunk counts per source after the T14 rebuild (v2 on the box was 21,814 chunks: command-manpages 14,799; man-pages 2,782; ubuntu-server-docs 2,632; tldr 932; gnome-user-docs 640; ubuntu-docs 29).
- TODO: Ubuntu Server docs commit actually fetched (`USD_COMMIT` or the default in `fetch_corpus.sh`).

## 3.3 RAG pipeline

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, cosine, CPU.
- Store: chromadb PersistentClient, collection `linux_lab`, telemetry off (`Settings(anonymized_telemetry=False)` and `ANONYMIZED_TELEMETRY=False`).
- Hybrid retrieve (`rag/query_hybrid.py`): MiniLM top-50 + BM25 top-50, fused with RRF (`k=60`). Optional third lane: MiniLM top-20 over `source=tldr` only. The app uses the tldr lane.
- Weak retrieval: abstain if none of the top-k fused hits has MiniLM cosine >= threshold. Not RRF. RRF is a rank score (any rank-0 hit is 1/60).
- Threshold and k live in `shared/frozen.json`. Committed values 0.30 and 5 are provisional defaults, not a tuned result.
- Tune only on `--split dev`. Write the chosen pair into `shared/frozen.json` and commit the sweep CSV. `--split test` refuses `--threshold`, `-k`, `--frozen`, and the sweep.
- chromadb 1.0.8 cannot read a 0.6.3 `chroma/` folder. Rebuild after install.
- TODO: freeze threshold and k after the real `--split dev` sweep on the T14 index; replace the provisional 0.30 / 5.
- TODO: attach that sweep CSV (filename + git commit).

## 3.4 CoT prompting

- Four prompt modes in `shared/prompt.py` / `app/engine.py`:
  - `base`: short answer, no excerpts, no step request
  - `cot`: "Think step by step" / write 2–5 steps, no excerpts
  - `rag`: excerpts only, cite the excerpt, no step list
  - `full` (RAG+CoT): excerpts + step-by-step; same SYSTEM text as SFT
- Temperature 0 and seed 42 on eval. Chat UI uses the same `ollama_chat` helper (`think` is off).
- CoT is prompting only. There is no separate process-supervision loss.
- TODO: one worked example (question, excerpts, steps, answer) from a real `full` turn, with the cited chunk.

## 3.5 Knowledge distillation (response / sequence-level, not soft-target)

- This is response-level (sequence-level) KD: a larger teacher writes grounded Q&A, then the 2B is SFT'd on those chat rows. There is no logit / temperature / KL distillation.
- Teacher default: `qwen3.5:9b` through Ollama. The existing scripts (`kd/teacher_generate.py`, `kd/judge_rows.py`) only speak Ollama's HTTP API. Colab notebook: `kd/colab_distill.ipynb` on a free T4. `USE_4B_TEACHER = True` falls back to `qwen3.5:4b` if 9B will not load.
- Why Ollama, not transformers 4-bit, for the teacher: the KD scripts already talk to Ollama; `qwen3.5:9b` is already a ~6.6–7.6 GB Q4 and fits a T4 16 GB card. A transformers 4-bit teacher would be a second stack.
- Data steps (resume-safe; Drive flush each row):
  1. `make_seed_chunks.py` (seed 42): one chunk per page, mix tldr=400, command-manpages=250, man-pages=100, gnome-user-docs=200, ubuntu-docs=29, ubuntu-server-docs=150. No Arch.
  2. `teacher_generate.py`: one student-style question + 2–5 steps + answer, facts only from that chunk.
  3. `judge_rows.py`: second teacher pass, yes/no on support and whether it answers the question.
  4. `build_sft.py --holdout`: drop bad JSON, skip, judge=no, invented backticked commands/flags. Answerable rows = source chunk + 2 distractors. Abstain rows (~25%) = same question, no source chunk, target = `ABSTAIN`. Train/val split by seed chunk.
- Student: LoRA / QLoRA SFT of `Qwen/Qwen3.5-2B` (notebook uses Unsloth; `USE_QLORA` is on). Unsloth documents large 4-bit error on Qwen3.5 and prefers bf16 LoRA (~5 GB VRAM for 2B, still T4-sized). Record which flag you used.
- Merge adapters, export GGUF `Q4_K_M`, `ollama create` on the laptop. Distilled config is skipped until `--distilled` / `OLLAMA_DISTILLED_MODEL` is set.
- TODO: teacher model actually used (9B or 4B) and Colab GPU type.
- TODO: seed count, rows generated, rows dropped (bad JSON / skip / judge / invented flag / holdout-near). The invented-flag count is a Ch4 number.
- TODO: SFT epochs, max seq, LoRA rank, train/val sizes, train loss curve.
- TODO: GGUF filename, `ollama create` name, file size on the T14.
- TODO: hand-check of ~50 kept teacher rows (pass/fail).

## 3.6 Five configs (KD isolated by 4 vs 5)

| id | model | mode |
|---|---|---|
| base | `qwen3.5:2b` | base |
| base+CoT | `qwen3.5:2b` | cot |
| base+RAG | `qwen3.5:2b` | rag |
| base+RAG+CoT | `qwen3.5:2b` | full |
| distilled+RAG+CoT | student GGUF | full |

- Defined in `app.engine.THESIS_CONFIGS`. Sidebar picks model and mode separately.
- Isolation: configs 1–4 hold the model fixed (`qwen3.5:2b`) and change prompt / retrieval. Config 5 holds RAG+CoT fixed and changes only the weights. The KD claim is 4 vs 5, not 1 vs 5.
- Do not report 5 as "RAG+CoT+KD beats base" without the 4 vs 5 pair.
- TODO: Ollama name of the registered student GGUF.

## 3.7 Dev / test split and leakage control

- `--split` is required: `dev` or `test`. JSONL fields: `id`, `question`, `answerable`, `gold_answer`, `source`. Optional `set`: `tuning` | `unanswerable` | `held-out`.
- Dev (`eval/dev.jsonl`): 24 ranking-tuning questions (`set=tuning`) plus labeled unanswerable rows. Threshold / k sweep runs here only. These 24 were also used for earlier spot checks, so they are not a clean test set.
- Test: 150–200 hand-written held-out questions (`eval/your-heldout.jsonl`). `eval/test.jsonl` is a format placeholder (`placeholder: true`). Do not report results from it.
- `build_sft.py --holdout` drops training questions with MiniLM cosine >= 0.80 vs any held-out question. Write the held-out file before SFT.
- `--split test` refuses the sweep and refuses threshold / k / frozen overrides. Use the committed `shared/frozen.json`.
- Seed 42 and temperature 0 on every eval row. Every row records `split`, `threshold`, `k`.
- TODO: write the real held-out file (count, how many answerable / unanswerable, who wrote it, date). Keep it out of git if the repo is public (`.gitignore` lists `eval/your-heldout.jsonl`).
- TODO: confirm no question string overlap between `eval/dev.jsonl` and the real held-out file.
- TODO: confirm `dropped_too_close_to_holdout` from `build_sft.py`.

## 3.8 Metrics

Score the five configs on the held-out test file (`--split test --no-sweep`). Do not retune on test.

- Wrong-command rate: gold command / flag is missing, or a command / flag that is not in the gold (or not in the cited chunk) is present. TODO: exact coding rules and double-coder check.
- Unsupported answers: a command, flag, path, or fact not in the retrieved excerpts (RAG configs) or not in the gold source (non-RAG). TODO: coding rules. `build_sft.py` `invented()` is the training-time version of this check, not the eval metric.
- Abstention accuracy, split:
  - answerable items: model should answer; abstain is a miss (false abstain)
  - unanswerable items: model should abstain; a command is a miss (false answer)
  - App abstain text is `shared.prompt.ABSTAIN`, also fired when retrieval is weak (`is_weak`: max MiniLM cosine < threshold).
  - Dev sweep already prints abstain precision / recall vs gold `answerable`. Test uses the frozen threshold / k only.
- Latency: `latency_s` in the eval CSV (chat turn wall time).
- Peak RAM under an emulated `mem=8G` cap: boot the T14 with `mem=8G`. `RamSampler` records peak app RSS, Ollama RSS, total (app+Ollama), and system used memory, sampled every 250 ms. Report peak `total_mb` and peak `system_used_mb` per config. TODO: confirm the kernel `mem=` line actually used.
- TODO: who scores wrong-command / unsupported (one coder or two), and a small agreement sample.

## 3.9 Statistics

- Pairwise McNemar on binary outcomes (correct vs not; or abstain-correct vs not) for planned pairs. The KD pair is base+RAG+CoT vs distilled+RAG+CoT. Prompt / RAG pairs stay on the 2B.
- Effect size for each McNemar pair. TODO: pick one and stick to it (Cohen's g or odds ratio). Report n and the 2x2 table.
- Holm correction across the planned family. Write the family down before looking at test. Suggested family: CoT vs base; RAG vs base; RAG+CoT vs RAG; RAG+CoT vs CoT; distilled+RAG+CoT vs base+RAG+CoT. TODO: freeze that list in this chapter, not in Ch4.
- Do not run the threshold sweep on test and then test it.
- TODO: alpha (0.05 unless the proposal already fixed another value).

## 3.10 ISO 25010 usability

- Characteristic in scope: usability (ISO/IEC 25010), not the whole quality model.
- Subcharacteristics to score: appropriateness recognizability, learnability, operability, user error protection, user interface aesthetics, accessibility.
- Instrument: short task-based session on the T14 (same five configs or the RAG+CoT UI only — pick one and say so). Map each item to a subcharacteristic.
- TODO: item list (or SUS items plus the ISO mapping table).
- TODO: n, who the users are (lab students, not the author), consent wording.
- TODO: tasks (e.g. extract tar.gz, ssh, apt install) and whether the index and model are already loaded.

## 3.11 Limitations

- 8G emulation is a kernel cap, not a full lab-image replay. Other processes on the T14 still move the RAM numbers.
- 0.30 / k=5 stay provisional until the real `--split dev` sweep is committed.
- `eval/test.jsonl` is not a result. Ch4 numbers come from `eval/your-heldout.jsonl` only after it exists.
- KD is sequence-level. The 2B never sees 9B logits.
- Abstain SFT rows use random distractors, not retrieved-but-wrong pages. Easy negatives.
- Teacher may invent flags; the filter drops those rows and the leftover errors are still possible. Judge is the same model family.
- Ubuntu Server docs licence is unverified. Arch Wiki was removed because it answers for the wrong distro.
- Colab T4 can disconnect; teacher/judge resume from the JSONL on Drive. A 4B teacher is a weaker teacher, not the same run.
- Unsloth advises against 4-bit QLoRA on Qwen3.5. If `USE_QLORA=True`, say so in Ch4.
- MiniLM holdout filter (0.80 cosine) can miss paraphrases and can over-drop near-duplicates that are still fair train items.
- Spot-check 22/24 (hybrid + tldr lane on v2) was tuned on the same 24 questions as `eval/dev.jsonl`. Do not call that a test result.
