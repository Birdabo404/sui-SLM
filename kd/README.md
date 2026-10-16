# Teacher Q&A kit for the KD step

Turns the RAG corpus into chat-format training rows for fp16 LoRA of Qwen3.5-2B.
Nothing here has been run with the real 9B teacher yet.

## Files
- seed_chunks.jsonl     picked by make_seed_chunks.py (seed 42). Default mix (no Server docs):
                        tldr 460, command-manpages 295, man-pages 115, gnome-user-docs 230, ubuntu-docs 29.
                        ubuntu-docs often yields about 16 pages, so the file is usually about 1,116 rows.
                        Use the printed count, not the 1,129 mix total. One chunk per page. No Arch Wiki.
                        `--holdout` drops chunks whose path matches a held-out `gold_source` / `source`
                        (including nested gzipped man pages such as command-manpages/coreutils/ls.1.gz).
- teacher_generate.py   9B writes 1 student-style question + 2-5 reasoning steps + answer per chunk,
                        using only that chunk. Resumable. Std-lib only, talks to Ollama.
- judge_rows.py         optional 2nd 9B pass that grades each row yes/no (supported, actually answers it).
- build_sft.py          drops rows with invented commands/flags (checked against the source chunk) and
                        rows the judge said no to, then builds:
                          answerable rows: source chunk + 2 distractor chunks as context -> steps + answer + source
                          abstain rows (25%): same question, context WITHOUT the answer -> "I don't know, ask the lab technician"
                        Output sft_train.jsonl / sft_val.jsonl as {"messages": [...], "text": ChatML}.
                        The notebook drops the messages column before SFT so TRL uses the rendered text.
                        `--holdout` accepts JSONL or a question-per-line file. It drops rows whose command
                        stem matches a held-out gold_source (tldr, man, and command-manpages for that
                        command), then drops questions with MiniLM cosine >= 0.80 vs any held-out
                        question. Both drop counts are printed. Each row also has a ChatML `text` field
                        with Qwen3.5's empty think block.
                        SYSTEM prompt is imported from shared/prompt.py (same text the chat app uses).

## Run on Colab (T4)

Use `kd/colab_distill.ipynb`. It mounts Drive, clones this repo, stops if `your-heldout.jsonl` is missing, runs `make_seed_chunks.py` → `teacher_generate.py` → `judge_rows.py` → `build_sft.py --holdout`, then fp16 LoRA of Qwen3.5-2B, and exports two Q4_K_M GGUFs in the same Colab session (same llama.cpp): the LoRA student (`work/gguf`) and unmodified `unsloth/Qwen3.5-2B` (`work/gguf-base`). Both Modelfiles come from `modelfile_text()` (same TEMPLATE, SYSTEM, and stop). The last cell prints both laptop commands:

    ollama create thesis-student -f Modelfile
    ollama create thesis-base-q4km -f Modelfile-base

Thesis configs 1–4 must use `thesis-base-q4km`, not stock `ollama pull qwen3.5:2b`. That tag is Q8_0 and a different template; using it would confound the 4 vs 5 KD pair with quant and template. Eval:

    python eval/run_eval.py --split test --no-sweep --questions eval/your-heldout.jsonl --out logs/eval-test.csv --base-model thesis-base-q4km --distilled thesis-student

Teacher is Ollama (`qwen3.5:9b`), not transformers 4-bit, because those two scripts only speak Ollama HTTP and the Q4 9B fits a T4. Set `USE_4B_TEACHER = True` for `qwen3.5:4b`. `USE_QLORA` defaults to False (fp16 LoRA; the T4 has no bf16). Writes under Drive `thesis-kd/`; re-run to resume (JSONL append, `resume_from_checkpoint`, reload `work/lora`).

From this repo, after `rag/lab-index` exists:
    python kd/make_seed_chunks.py --chunks rag/lab-index/chunks.jsonl --out kd/seed_chunks.jsonl --holdout eval/your-heldout.jsonl
    python kd/teacher_generate.py --seeds kd/seed_chunks.jsonl --out kd/teacher_raw.jsonl --model qwen3.5:9b

## Things to decide / watch
- Write the held-out test questions BEFORE make_seed_chunks.py and build_sft.py so --holdout can keep them out of training.
- Hand-check ~50 kept rows. The flag filter only catches invented options; the judge is a model too.
  The drop counts build_sft.py prints (how often the 9B invents flags) are worth reporting in Ch4.
- Risky commands (wipefs, dd, mkfs, rm -r) are in the corpus. The prompt asks the teacher to open with a
  warning and send the student to the lab technician; check that it does.
- SYSTEM prompt in shared/prompt.py must stay identical for the offline chat app and SFT rows.
- Thesis configs 1–4 use `thesis-base-q4km` from the Colab export. Do not score them against stock `qwen3.5:2b` (Q8_0).
- Abstain rows use random distractors, so they are easy. Harder ones (retrieved-but-wrong pages) are later work.
