# Teacher Q&A kit for the KD step (prepared 2026-10-12 on the box)

Turns the trimmed RAG corpus into chat-format training rows for LoRA-tuning Qwen3.5-2B.
Nothing here has been run with the real 9B teacher yet. The plumbing was smoke-tested end to end on the
box with a tiny Qwen3.5-0.8B (outputs in smoke-test-0.8b/, quality is NOT representative of the 9B).

## Files
- seed_chunks.jsonl     1,116 doc chunks picked by make_seed_chunks.py (seed 42) from rag-index-v2 (2026-10-12):
                        tldr 400, Ubuntu command man pages 250, man-pages 100, GNOME help 200, Ubuntu docs 16
                        (only 16 pages qualify), Ubuntu Server docs 150. No Arch Wiki (dropped from the corpus).
                        One chunk per page, no alias stubs or link-list landing pages.
- teacher_generate.py   9B writes 1 student-style question + 2-5 reasoning steps + answer per chunk,
                        using only that chunk. Resumable. Std-lib only, talks to Ollama.
- judge_rows.py         optional 2nd 9B pass that grades each row yes/no (supported, actually answers it).
- build_sft.py          drops rows with invented commands/flags (checked against the source chunk) and
                        rows the judge said no to, then builds:
                          answerable rows: source chunk + 2 distractor chunks as context -> steps + answer + source
                          abstain rows (25%): same question, context WITHOUT the answer -> "I don't know, ask the lab technician"
                        Output sft_train.jsonl / sft_val.jsonl as {"messages": [...]}, ready for TRL SFTTrainer.
                        --holdout test_questions.txt drops training questions too close to your test set.
                        SYSTEM prompt is imported from shared/prompt.py (same text the chat app uses).

## Run on Colab (T4 is enough for qwen3.5:9b at 4-bit)
    from google.colab import drive; drive.mount('/content/drive')
    !mkdir -p /content/drive/MyDrive/thesis-kd   # upload this folder's files here
    %cd /content/drive/MyDrive/thesis-kd
    !apt-get -qq install -y zstd && curl -fsSL https://ollama.com/install.sh | sh
    !nohup ollama serve > ollama.log 2>&1 &
    !sleep 5 && ollama pull qwen3.5:9b
    !python teacher_generate.py --model qwen3.5:9b --limit 20     # look at teacher_raw.jsonl first
    !python teacher_generate.py --model qwen3.5:9b                # the rest; rerun after a disconnect
    !python judge_rows.py --model qwen3.5:9b
    !python build_sft.py --raw teacher_judged.jsonl --holdout test_questions.txt
Writing to Drive means a Colab disconnect loses at most one row.

From this repo, after the v2 index exists:
    python kd/make_seed_chunks.py --chunks rag/rag-index-v2/chunks.jsonl --out kd/seed_chunks.jsonl
    python kd/teacher_generate.py --seeds kd/seed_chunks.jsonl --out kd/teacher_raw.jsonl --model qwen3.5:9b

## Things to decide / watch
- Write the held-out test questions BEFORE build_sft.py so --holdout can keep them out of training.
- Hand-check ~50 kept rows. The flag filter only catches invented options; the judge is a model too.
  The drop counts build_sft.py prints (how often the 9B invents flags) are worth reporting in Ch4.
- Risky commands (wipefs, dd, mkfs, rm -r) are in the corpus. The prompt asks the teacher to open with a
  warning and send the student to the lab technician; check that it does.
- SYSTEM prompt in shared/prompt.py must stay identical for the offline chat app and SFT rows.
- Abstain rows use random distractors, so they are easy. Harder ones (retrieved-but-wrong pages) are a v2.
