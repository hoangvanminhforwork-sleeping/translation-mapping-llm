"""Tiny fact-insertion pilot: lexical retrieval + local LLM generation.

Model: qwen2.5:1.5b via Ollama (fits fully in the 4 GB RTX 3050).
This is a pilot, not the 200-sentence study.

Arms:
  A. lexical retrieval + direct insertion prompt
  B. char n-gram Euclidean retrieval + task-prompt framing

Outputs: results/pilot_results.json, results/pilot_summary.json
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:1.5b"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [t for t in norm(text).split() if len(t) > 1]


def lexical_scores(sentence: str, facts: list[dict]) -> list[float]:
    st = set(tokens(sentence))
    out = []
    for f in facts:
        ft = set(tokens(f["fact_vi"] + " " + f["dish"]))
        out.append(len(st & ft) / max(1, len(ft)))
    return out


def char_ngram(text: str, n: int = 3) -> Counter:
    s = " " + norm(text) + " "
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


def tf_vector(text: str, n: int = 3) -> dict[str, float]:
    c = char_ngram(text, n)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def euclidean(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


def embedding_scores(sentence: str, facts: list[dict]) -> list[float]:
    sv = tf_vector(sentence)
    return [euclidean(sv, tf_vector(f["fact_vi"] + " " + f["dish"])) for f in facts]


def ollama(prompt: str, *, temperature: float = 0.1, num_predict: int = 120) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "top_p": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                obj = json.loads(r.read().decode("utf-8"))
            obj["wall_seconds"] = time.perf_counter() - t0
            return obj
        except Exception:
            if attempt == 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def prompt_direct(sentence: str, fact: str) -> str:
    return f"""Bạn là biên tập viên tiếng Việt. Viết lại câu sau thành MỘT câu tiếng Việt tự nhiên, chèn thông tin được cung cấp mà không thêm dữ kiện mới.
Câu gốc: {sentence}
Thông tin cần chèn: {fact}
Chỉ trả về câu đã viết lại, không giải thích."""


def prompt_task(sentence: str, fact: str) -> str:
    return f"""Nhiệm vụ: giúp người nước ngoài hiểu món ăn Việt Nam.
Hãy nói lại câu sau thành MỘT câu tiếng Việt tự nhiên, lồng thông tin được cung cấp vào câu.
Câu gốc: {sentence}
Thông tin: {fact}
Chỉ trả về một câu tiếng Việt, không giải thích."""


def keyword_coverage(text: str, keywords: list[str]) -> float:
    t = norm(text)
    return sum(1 for k in keywords if norm(k) in t) / max(1, len(keywords))


def judge(output: str, sentence: str, fact: str, keywords: list[str]) -> dict:
    prompt = f"""You are a strict Vietnamese evaluator. Return ONLY JSON.
Judge the rewritten sentence against the source and the fact.

Source: {sentence}
Fact to insert: {fact}
Rewritten: {output}
Keywords that should appear (Vietnamese): {", ".join(keywords)}

Return JSON with keys:
- faithful: 1 if the rewritten sentence contains the fact and no new factual claim, else 0
- fluent: integer 1-5 (5 = natural Vietnamese, 1 = broken)
- one_sentence: 1 if exactly one sentence, else 0
- note: short reason (<= 20 words)
"""
    obj = ollama(prompt, temperature=0.0, num_predict=120)
    raw = (obj.get("response") or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"faithful": 0, "fluent": 1, "one_sentence": 0, "note": "unparseable judge output", "raw": raw[:300]}
    data["judge_wall_seconds"] = obj.get("wall_seconds")
    return data


def main() -> None:
    facts = load_jsonl(DATA / "fact_bank.jsonl")
    sentences = load_jsonl(DATA / "pilot_sentences.jsonl")
    fact_by_id = {f["id"]: f for f in facts}

    # Retrieval accuracy (selection only).
    retrieval = []
    for s in sentences:
        lex = lexical_scores(s["sentence_vi"], facts)
        emb = embedding_scores(s["sentence_vi"], facts)
        lex_id = facts[max(range(len(facts)), key=lambda i: lex[i])]["id"]
        emb_id = facts[min(range(len(facts)), key=lambda i: emb[i])]["id"]
        retrieval.append({
            "id": s["id"],
            "expected": s["expected_fact_id"],
            "lexical_top1": lex_id,
            "embedding_top1": emb_id,
            "lexical_ok": lex_id == s["expected_fact_id"],
            "embedding_ok": emb_id == s["expected_fact_id"],
        })

    rows = []
    for s in sentences:
        fact = fact_by_id[s["expected_fact_id"]]
        lex = lexical_scores(s["sentence_vi"], facts)
        emb = embedding_scores(s["sentence_vi"], facts)
        lex_fact = facts[max(range(len(facts)), key=lambda i: lex[i])]
        emb_fact = facts[min(range(len(facts)), key=lambda i: emb[i])]

        arms = [
            ("A_lexical_direct", lex_fact, prompt_direct),
            ("B_embedding_task", emb_fact, prompt_task),
        ]
        for arm, used_fact, prompt_fn in arms:
            t0 = time.perf_counter()
            obj = ollama(prompt_fn(s["sentence_vi"], used_fact["fact_vi"]))
            gen_wall = time.perf_counter() - t0
            output = (obj.get("response") or "").strip()
            output = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", output).strip()
            cov = keyword_coverage(output, fact["keywords"])
            j = judge(output, s["sentence_vi"], fact["fact_vi"], fact["keywords"])
            rows.append({
                "id": s["id"],
                "dish": s["dish"],
                "arm": arm,
                "source": s["sentence_vi"],
                "expected_fact": fact["fact_vi"],
                "used_fact_id": used_fact["id"],
                "retrieval_ok": used_fact["id"] == fact["id"],
                "output": output,
                "keyword_coverage": round(cov, 3),
                "fact_pass_80": cov >= 0.8,
                "gen_wall_seconds": round(gen_wall, 3),
                "eval_count": obj.get("eval_count"),
                "eval_duration_ns": obj.get("eval_duration"),
                "judge": j,
            })

    # Summary.
    def mean(xs):
        return round(statistics.mean(xs), 3) if xs else None

    summary = {
        "model": MODEL,
        "sentences": len(sentences),
        "facts": len(facts),
        "retrieval": {
            "lexical_top1_acc": round(sum(r["lexical_ok"] for r in retrieval) / len(retrieval), 3),
            "embedding_top1_acc": round(sum(r["embedding_ok"] for r in retrieval) / len(retrieval), 3),
            "lexical_errors": [r["id"] for r in retrieval if not r["lexical_ok"]],
            "embedding_errors": [r["id"] for r in retrieval if not r["embedding_ok"]],
        },
        "generation": {},
        "note": "Self-judged by the same model; treat faithful/fluent as a screening signal, not a human gold label.",
    }
    for arm in ["A_lexical_direct", "B_embedding_task"]:
        sub = [r for r in rows if r["arm"] == arm]
        summary["generation"][arm] = {
            "n": len(sub),
            "fact_pass_80_rate": round(sum(r["fact_pass_80"] for r in sub) / len(sub), 3),
            "mean_keyword_coverage": mean([r["keyword_coverage"] for r in sub]),
            "judge_faithful_rate": round(sum(1 for r in sub if r["judge"].get("faithful") == 1) / len(sub), 3),
            "judge_fluent_mean": mean([float(r["judge"].get("fluent", 0)) for r in sub]),
            "one_sentence_rate": round(sum(1 for r in sub if r["judge"].get("one_sentence") == 1) / len(sub), 3),
            "mean_gen_seconds": mean([r["gen_wall_seconds"] for r in sub]),
            "mean_tokens_per_second": mean([
                (r["eval_count"] or 0) / ((r["eval_duration_ns"] or 1) / 1e9) for r in sub
            ]),
        }

    (OUT / "pilot_results.json").write_text(
        json.dumps({"summary": summary, "retrieval": retrieval, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
