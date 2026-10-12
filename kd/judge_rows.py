#!/usr/bin/env python3
"""Optional step 2b: a second, stricter teacher pass that grades each generated row. The flag check in
build_sft.py only catches invented commands/options; this catches answers that are vague, wrong for the
question, or not actually supported by the excerpt. Adds "judge": "yes"/"no" and a reason to each row.
  python judge_rows.py --raw teacher_raw.jsonl --seeds seed_chunks.jsonl --out teacher_judged.jsonl
Then: python build_sft.py --raw teacher_judged.jsonl ...   (rows judged "no" are dropped)
Resumable like teacher_generate.py.
"""
import argparse, json, os, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="teacher_raw.jsonl"); ap.add_argument("--seeds", default="seed_chunks.jsonl")
ap.add_argument("--out", default="teacher_judged.jsonl"); ap.add_argument("--model", default="qwen3.5:9b")
ap.add_argument("--host", default="http://localhost:11434"); ap.add_argument("--num-ctx", type=int, default=4096)
a = ap.parse_args()
PROMPT = """You are a strict reviewer of training data for a Linux lab assistant. Grade the answer below.

<excerpt title="{title}">
{text}
</excerpt>

Question: {question}
Answer: {answer}

Reply "yes" ONLY if ALL are true:
1. Every fact, command and option in the answer is stated in the excerpt.
2. The answer actually answers the question (not vague, not a restatement of the question).
3. A student following it would get the right result.
Otherwise reply "no".
JSON only: {{"verdict": "yes" or "no", "reason": "one short sentence"}}"""
seeds = {json.loads(l)["id"]: json.loads(l) for l in open(a.seeds, encoding="utf-8")}
done = {json.loads(l)["seed_id"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
n = 0
with open(a.out, "a", encoding="utf-8") as f:
    for l in open(a.raw, encoding="utf-8"):
        r = json.loads(l)
        if r["seed_id"] in done: continue
        if r.get("answer") and not r.get("error"):
            s = seeds[r["seed_id"]]
            body = {"model": a.model, "stream": False, "format": "json", "think": False,
                    "messages": [{"role": "user", "content": PROMPT.format(title=s["title"], text=s["text"], question=r["question"], answer=r["answer"])}],
                    "options": {"temperature": 0, "num_ctx": a.num_ctx}}
            req = urllib.request.Request(f"{a.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                v = json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"])
                r["judge"] = "yes" if str(v.get("verdict", "")).lower().startswith("y") else "no"; r["judge_reason"] = v.get("reason", "")
            except Exception as e:
                print("judge failed, will retry next run:", e); continue
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1
print(f"judged {n} rows -> {a.out}")
