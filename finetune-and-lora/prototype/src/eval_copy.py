"""Evaluate an adapter on the fixed 2,000-sentence copy test, and compare to the baseline.

The sentences are read from data/eval_sources.jsonl, which was copied from the
baseline run, so the comparison is paired: same sentence, same prompt, same model
family, before vs after the adapter.
"""

from __future__ import annotations

import csv
import json
import time
import unicodedata

import torch

from . import config as C
from . import train


def normalize(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def load_eval(lang: str) -> list[dict]:
    rows = [json.loads(l) for l in (C.DATA / "eval_sources.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["lang"] == lang]
    rows.sort(key=lambda r: r["id"])
    return rows


def generate_all(model_key: str, force: bool = False) -> str:
    spec = C.MODELS[model_key]
    lang = spec["lang"]
    out = C.RESULTS / "raw" / f"ft_{model_key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if force and out.exists():
        out.unlink()

    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = [r for r in load_eval(lang) if r["id"] not in done]
    if not rows:
        print(f"{model_key}: eval generations already present")
        return str(out)

    tok, model = train.attach_adapter(model_key)
    dev = train.device()
    bs = spec["batch"]
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for start in range(0, len(rows), bs):
            chunk = rows[start : start + bs]
            prompts = [spec["prefix"] + r["src"] for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            longest = int(enc["input_ids"].shape[1]) + 8
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=min(C.MAX_NEW_TOKENS, longest),
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            hyps = tok.batch_decode(gen, skip_special_tokens=True)
            for r, hyp in zip(chunk, hyps):
                fh.write(json.dumps({"id": r["id"], "src": r["src"], "hyp": hyp}, ensure_ascii=False) + "\n")
            fh.flush()
            if (start // bs) % 20 == 0:
                rate = (start + len(chunk)) / max(1e-9, time.time() - t0)
                print(f"  {start + len(chunk):5d}/{len(rows)}  {rate:5.1f} sent/s", flush=True)
    print(f"{model_key}: generated {len(rows)} in {time.time() - t0:.0f}s")
    return str(out)


def score(model_key: str) -> dict:
    """Metrics for one model, computed the same way as the baseline."""
    import sacrebleu

    spec = C.MODELS[model_key]
    lang = spec["lang"]
    rows = [json.loads(l) for l in (C.RESULTS / "raw" / f"ft_{model_key}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    hyps = [r["hyp"] for r in rows]
    refs = [r["src"] for r in rows]

    exact = sum(normalize(h) == normalize(s) for h, s in zip(hyps, refs)) / len(rows)
    tokz = "char" if lang == "vi" else "13a"
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokz, force=True).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    rec, prec = [], []
    for h, s in zip(hyps, refs):
        ht, st = normalize(h).split(), normalize(s).split()
        rec.append(sum(1 for t in st if t in ht) / max(1, len(st)))
        prec.append(sum(1 for t in ht if t in st) / max(1, len(ht)))
    return {
        "model": model_key,
        "lang": lang,
        "n": len(rows),
        "exact_match_norm": round(exact, 4),
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
        "tok_recall": round(sum(rec) / len(rec), 4),
        "tok_precision": round(sum(prec) / len(prec), 4),
        "mean_hyp_words": round(sum(len(h.split()) for h in hyps) / len(hyps), 2),
        "mean_ref_words": round(sum(len(s.split()) for s in refs) / len(refs), 2),
    }


def compare(force: bool = False) -> list[dict]:
    """before/after table, joined by (lang, id) against the fixed eval set."""
    import csv as _csv

    base = {}
    for line in (C.DATA / "eval_sources.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            base[f"{r['lang']}:{r['id']}"] = r

    out = []
    for model_key in C.ORDER:
        raw = C.RESULTS / "raw" / f"ft_{model_key}.jsonl"
        if not raw.exists():
            continue
        spec = C.MODELS[model_key]
        lang = spec["lang"]
        rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
        pairs = [(base[f"{lang}:{r['id']}"], r) for r in rows if f"{lang}:{r['id']}" in base]
        before = [normalize(b["src"]) == normalize(b["baseline_hyp"]) for b, _ in pairs]
        after = [normalize(b["src"]) == normalize(r["hyp"]) for b, r in pairs]
        flipped = sum(1 for b, a in zip(before, after) if a and not b)
        broke = sum(1 for b, a in zip(before, after) if b and not a)
        s = score(model_key)
        out.append(
            {
                "model": model_key,
                "n": s["n"],
                "exact_before": round(sum(before) / max(1, len(before)), 4),
                "exact_after": s["exact_match_norm"],
                "flipped_to_exact": flipped,
                "broke_previously_exact": broke,
                "bleu_after": s["bleu"],
                "chrf_after": s["chrf"],
                "mean_hyp_words_after": s["mean_hyp_words"],
                "mean_ref_words": s["mean_ref_words"],
            }
        )

    path = C.RESULTS / "before_after.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['model']:<14} exact {r['exact_before']:.1%} -> {r['exact_after']:.1%} "
              f"(+{r['flipped_to_exact']} fixed, -{r['broke_previously_exact']} broken)  BLEU {r['bleu_after']}")
    return out