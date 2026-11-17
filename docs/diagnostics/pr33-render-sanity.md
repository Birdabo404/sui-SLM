# Revised thesis-student diagnosis

## Real-student measurement

The following result is **Engineering Lead's measurement**, not a capture
produced by this branch:

- Model: the owner's real `thesis-student` GGUF and deployed Modelfile,
  hash-matched before the run.
- Path: Ollama `/api/chat` through the app request path with the approved
  `think:false`, `num_ctx:4096`, `num_predict:512`, and
  `repeat_penalty:1.1` settings.
- Questions: 22.
- Result: 22/22 ended naturally with `done_reason:stop`; answer lengths were
  101–175 evaluated tokens; 0/22 reached the 512-token cap.
- Modelfile: explicit ChatML with an empty think block and only
  `<|im_end|>` as a stop token.

This rules out the app prediction cap, repetition guard, and deployed template
as the cause of the short visible answer. No Modelfile rebuild is indicated.
`scripts/capture_raw_chat.py` remains available as optional confirmation on
the owner's machine.

## Revised interpretation

`kd/teacher_generate.py` trains `Steps:` as two to five short reasoning steps
that show how an excerpt leads to the answer. It separately trains `Answer:`
as the concise one-to-four-sentence response shown to the student. Therefore:

- `Steps:` followed by `Answer:` belongs under collapsed **Show reasoning**.
- The concise `Answer:` is the complete trained answer, not a truncated stream.
- Inline commands in a bare answer need copy boxes even when the model does not
  emit a fenced code block.
- A finished `Steps:` response with no `Answer:` still needs the non-empty
  fallback rather than disappearing.

PR #33 implements that interpretation and also keeps inline code byte-for-byte
while cleaning citations.

## Supporting proxy captures

Before the real model was available, this branch captured 20 DEV questions
through the same app request body for stock `qwen3.5:2b`, a matched Q4_K_M
base, and two template-only proxies. All four variants produced Markdown,
ended naturally on all 20 questions, and produced zero one-sentence raw
streams. Those proxy results remain useful for isolating transport and
template behavior, but the real-student measurement above supersedes them for
the actual model's length and output shape.
