# Ollama model-variant configuration comparison

These values come from `ollama show -v` and `ollama show --modelfile` on the
machine that produced the raw stream captures. The fine-tuned
`thesis-student` GGUF was unavailable, so the two student rows use the matched
public Q4_K_M base weights to isolate template behavior.

| Role | Captured local tag | Weights / quantization | Template and parser | Model-level parameters |
|---|---|---|---|---|
| Original app default | `qwen3.5:2b` | Ollama Qwen3.5 2B, Q8_0, model ID `0689d44085e0` | `TEMPLATE {{ .Prompt }}`, plus `RENDERER qwen3.5` and `PARSER qwen3.5`; renderer/parser template source | `temperature 1`, `top_p 0.95`, `top_k 20`, `presence_penalty 1.5`; no explicit `stop` or `num_ctx` |
| Matched compressed base | `thesis-base-q4km-repro` | Unsloth Qwen3.5-2B Q4_K_M, model ID `226880fd5895` | Explicit ChatML over `.System` and `.Messages`, assistant empty-think prefix; no Ollama renderer/parser | `temperature 0`, `num_ctx 4096`, only stop `<\|im_end\|>` |
| Intended student template proxy | `thesis-student-template-proxy` | Same Q4_K_M base weights for this proxy, model ID `226880fd5895`; a real student would have fine-tuned weights | Exactly the same explicit ChatML and empty-think prefix as the matched base | Exactly the same parameters as the matched base |
| Stale student-template proxy | `thesis-student-legacy-proxy` | Same Q4_K_M base weights for this proxy, model ID `594c1a7e6794` | Plain `TEMPLATE {{ .Prompt }}` with **no** Qwen3.5 renderer/parser | `temperature 0`, `num_ctx 4096`; no explicit stop |

## What the app overrides

All four capture roles receive the same +RAG+CoT message list and the same
request options:

- `think: false`
- `temperature: 0.2`
- `num_ctx: 4096`
- `num_predict: 512`
- `repeat_penalty: 1.1`
- `stream: true`

The request temperature and context override the corresponding Modelfile
values. The request does not set `stop`, `top_p`, `top_k`, or
`presence_penalty`, so model-level values for those fields remain active.

The student is selected with `model_key="distilled"` and is locked to
+RAG+CoT, but the capture uses +RAG+CoT for every model. Before response
parsing, the request path and raw stream transport are otherwise the same.
The compatibility fix in this PR acts only after raw student content reaches
the app.

## Important distinction

The stock model's visible `TEMPLATE {{ .Prompt }}` is not equivalent to the
stale exported student template: stock Ollama also supplies its Qwen3.5
renderer and parser. The stale export has neither. The corrected matched base
and intended student use explicit ChatML instead, including the empty
`<think></think>` block and `<|im_end|>` stop.

In the 20-question raw capture, all four available variants emitted
multi-sentence Markdown and ended with `done_reason: stop`. The Q4_K_M base and
both template proxies use identical weights, so that controlled comparison did
not reproduce the symptom by changing template/stop configuration alone. The
real student's fine-tuned weights and deployed ThinkPad Modelfile remain the
only model-side variables not captured here.
