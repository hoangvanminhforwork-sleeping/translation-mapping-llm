"""Instruction-following test: can the model copy the sentence it is given?

Prompt (from thought01.md):  Vietnamese -> "Noi: {s_v}"   English -> "Say: {s_e}"
The model is asked for one thing: repeat the sentence exactly. Anything else
(paraphrase, truncation, refusal, degenerate repetition) is a copy failure.

Resumable: raw generations are appended to one JSONL per model; a re-run skips
ids already present. Metrics are computed from the JSONL, never from memory.
"""

from __future__ import annotations

import json
import random
import time
import unicodedata

import duckdb
import torch

from . import config as C
from . import models as M

RAW = C.RESULTS / "raw"


# ---------------------------------------------------------------- data ------
def _source_is_vi(prompt: str) -> bool:
    """Which side of the pair is Vietnamese, from the prompt template.

    The MTET test split has SIX templates, not four:
        What is the English translation for        -> source is vi
        Translate the following sentence into English -> source is vi
        Dich cau sau sang tieng Anh                -> source is vi
        What is the Vietnamese translation for      -> source is en
        Translate the following sentence into Vietnamese -> source is en
        Dich cau sau sang tieng Viet                -> source is en
    Matching on "English translation" alone silently swaps 433/2536 rows.
    """
    p = prompt.lower()
    if "into vietnamese" in p or "vietnamese translation" in p or "sang tiếng việt" in p:
        return False
    if "into english" in p or "english translation" in p or "sang tiếng anh" in p:
        return True
    raise ValueError(f"unknown prompt template: {prompt[:80]!r}")


def sample_sentences(n: int = C.N_COPY_SENTENCES, seed: int = C.SEED) -> list[dict]:
    """Deterministic sample of the MTET test split, with a per-sentence language tag.

    Direction comes from the prompt template, then is cross-checked by Vietnamese
    diacritics (a row whose assigned Vietnamese side has none, while the other side
    has some, is swapped and counted). The count is printed, never hidden.
    """
    con = duckdb.connect()
    rows = con.sql(
        f"""SELECT prompt, translation.source AS src, translation.target AS tgt
            FROM read_parquet('{C.MTET_TEST.as_posix()}')"""
    ).fetchall()
    con.close()

    pairs = []
    swaps = ambiguous = 0
    for prompt, src, tgt in rows:
        vi_is_src = _source_is_vi(prompt)
        vi, en = (src, tgt) if vi_is_src else (tgt, src)
        if not _has_diacritics(vi) and _has_diacritics(en):
            vi, en = en, vi
            swaps += 1
        elif not _has_diacritics(vi) and not _has_diacritics(en):
            ambiguous += 1
        pairs.append({"vi": vi, "en": en, "dir": "vi2en" if vi_is_src else "en2vi"})

    print(f"test split: {len(pairs)} pairs | diacritic swaps {swaps} | no-diacritics-both-sides {ambiguous}")

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(pairs)), min(n, len(pairs))))
    out = []
    for i, j in enumerate(idx):
        out.append({"id": i, **{k: pairs[j][k] for k in ("vi", "en", "dir")}})
    return out


def _has_diacritics(s: str) -> bool:
    return any(ch in "ăâđêôơưĂÂĐÊÔƠƯ" or unicodedata.combining(ch) for ch in s)


# ------------------------------------------------------------ generation ----
def _generate(tok, model, prompts: list[str], max_new_tokens: int) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(M.device())
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.batch_decode(out, skip_special_tokens=True)


def run_copy(model_key: str, n: int = C.N_COPY_SENTENCES, force: bool = False) -> str:
    spec = C.MODELS[model_key]
    lang = spec["lang"]
    prefix = spec["copy_prefix"]
    raw_path = RAW / f"copy_{model_key}.jsonl"
    RAW.mkdir(parents=True, exist_ok=True)
    if force and raw_path.exists():
        raw_path.unlink()

    done: set[int] = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except json.JSONDecodeError:  # a kill mid-write can leave one partial line
                    continue

    rows = sample_sentences(n)
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print(f"{model_key}: nothing to do ({len(done)} rows already on disk)")
        return str(raw_path)
    if len(done):
        print(f"{model_key}: resuming, {len(done)} done / {len(todo)} to go")

    tok, model, info = M.load(model_key)
    t0 = time.time()
    bs = spec["batch"]
    with raw_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), bs):
            chunk = todo[start : start + bs]
            srcs = [r[lang] for r in chunk]
            prompts = [prefix + s for s in srcs]
            longest = max(len(tok(p)["input_ids"]) for p in prompts) + 8
            max_new = min(C.MAX_NEW_TOKENS, longest)
            hyps = _generate(tok, model, prompts, max_new)
            for r, src, hyp in zip(chunk, srcs, hyps):
                fh.write(json.dumps({"id": r["id"], "src": src, "hyp": hyp}, ensure_ascii=False) + "\n")
            fh.flush()
            if (start // bs) % 20 == 0:
                rate = (start + len(chunk)) / max(1e-9, time.time() - t0)
                print(f"  {start + len(chunk):5d}/{len(todo)}  {rate:5.1f} sent/s  vram={M.peak_vram_mb():.0f} MB")

    M.free(model)
    print(f"{model_key}: wrote {raw_path.name} in {time.time() - t0:.1f}s ({info['device']})")
    return str(raw_path)


# --------------------------------------------------------------- metrics ----
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(s.split())


def metrics(model_key: str) -> dict:
    raw_path = RAW / f"copy_{model_key}.jsonl"
    spec = C.MODELS[model_key]
    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])

    hyps = [r["hyp"] for r in rows]
    refs = [r["src"] for r in rows]

    exact = sum(h.strip() == s.strip() for h, s in zip(hyps, refs)) / len(rows)
    exact_norm = sum(normalize(h) == normalize(s) for h, s in zip(hyps, refs)) / len(rows)
    empty = sum(not h.strip() for h in hyps) / len(rows)

    tokz = "char" if spec["lang"] == "vi" else "13a"
    import sacrebleu

    corpus_bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokz).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    hyp_len = sum(len(h.split()) for h in hyps) / len(hyps)
    ref_len = sum(len(s.split()) for s in refs) / len(refs)

    # token-level recall/precision against the source, character 3-gram free
    def toks(x):
        return normalize(x).split()

    rec, prec = [], []
    for h, s in zip(hyps, refs):
        ht, st = toks(h), toks(s)
        if not st:
            continue
        overlap = sum(1 for t in st if t in ht)
        rec.append(overlap / len(st))
        prec.append(overlap / max(1, len(ht)))

    return {
        "model": model_key,
        "lang": spec["lang"],
        "size_class": spec["size_class"],
        "n": len(rows),
        "exact_match": round(exact, 4),
        "exact_match_norm": round(exact_norm, 4),
        "empty_rate": round(empty, 4),
        "bleu": round(corpus_bleu, 2),
        "chrf": round(chrf, 2),
        "tok_recall": round(sum(rec) / len(rec), 4),
        "tok_precision": round(sum(prec) / len(prec), 4),
        "mean_hyp_words": round(hyp_len, 2),
        "mean_ref_words": round(ref_len, 2),
    }


def write_summary(model_keys: list[str] | None = None) -> "object":
    import csv

    keys = model_keys or [k for k in C.MODEL_ORDER if (RAW / f"copy_{k}.jsonl").exists()]
    out = [metrics(k) for k in keys]
    path = C.RESULTS / "copy_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"summary -> {path.relative_to(C.ROOT)}")
    return out


def write_rows(model_key: str) -> str:
    import csv

    rows = [json.loads(l) for l in (RAW / f"copy_{model_key}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    path = C.RESULTS / f"copy_{model_key}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "exact", "src", "hyp"])
        w.writeheader()
        for r in rows:
            w.writerow({"id": r["id"], "exact": int(normalize(r["hyp"]) == normalize(r["src"])), **r})
    return str(path)
