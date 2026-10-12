#!/usr/bin/env python3
"""Step 2 of the KD data: the teacher (Qwen3.5-9B through Ollama) writes one lab-style question per
seed chunk plus a step-by-step answer that uses ONLY that chunk. Run it on Colab (T4 is enough for
qwen3.5:9b at 4-bit) or overnight on the laptop with everything else closed.
Resumable: rows already in --out are skipped, so a Colab disconnect only costs the current chunk.
  python teacher_generate.py --seeds seed_chunks.jsonl --out teacher_raw.jsonl --model qwen3.5:9b
Standard library only (talks to Ollama's HTTP API on localhost:11434).
"""
import argparse, json, os, sys, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="seed_chunks.jsonl")
ap.add_argument("--out", default="teacher_raw.jsonl")
ap.add_argument("--model", default="qwen3.5:9b")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--limit", type=int, default=0, help="stop after N new rows (0 = all), handy for a test run")
ap.add_argument("--num-ctx", type=int, default=4096)
a = ap.parse_args()

PROMPT = """You are helping build training data for an offline assistant in a university Linux computer lab (Ubuntu PCs).
Below is ONE excerpt from the official documentation ({source}, "{title}").

<excerpt>
{text}
</excerpt>

Write ONE question that a first- or second-year IT student in the lab could realistically ask, in plain
everyday words (not copied from the excerpt), and that this excerpt fully answers. Then answer it.

Rules:
- Use ONLY facts, commands and options that appear in the excerpt. Do not add any flag, path or command that is not in it.
- "steps": 2 to 5 short reasoning steps that show how the excerpt leads to the answer.
- "answer": the final answer for the student, 1 to 4 sentences. Put commands in backticks, with real example values instead of {{{{placeholders}}}}.
- If the command can delete data, wipe disks, or change system settings, start the answer with a one-line warning
  and mention it needs the lab technician's permission.
- If the excerpt is too thin to support a useful lab question, return {{"skip": true}}.

Reply with JSON only, in this shape:
{{"question": "...", "steps": ["...", "..."], "answer": "..."}}"""

def ask(prompt):
    body = {"model": a.model, "stream": False, "format": "json", "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.3, "num_ctx": a.num_ctx}}
    req = urllib.request.Request(f"{a.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]

done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["seed_id"] for l in open(a.out, encoding="utf-8") if l.strip()}
seeds = [json.loads(l) for l in open(a.seeds, encoding="utf-8")]
todo = [s for s in seeds if s["id"] not in done]
print(f"{len(done)} done, {len(todo)} to go, model {a.model}", flush=True)
n = 0; t0 = time.time()
with open(a.out, "a", encoding="utf-8") as f:
    for s in todo:
        rec = {"seed_id": s["id"], "source": s["source"], "title": s["title"], "license": s["license"], "teacher": a.model}
        try:
            raw = ask(PROMPT.format(source=s["source"], title=s["title"], text=s["text"]))
            rec["raw"] = raw
            try: rec.update(json.loads(raw))
            except json.JSONDecodeError: rec["error"] = "bad_json"
        except Exception as e:
            rec["error"] = f"request_failed: {e}"
            print(rec["error"], file=sys.stderr, flush=True); time.sleep(5)
            continue  # not written, so it is retried on the next run
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
        if n % 10 == 0: print(f"{n} rows, {(time.time()-t0)/n:.1f} s/row", flush=True)
        if a.limit and n >= a.limit: break
print(f"wrote {n} new rows -> {a.out}")
