"""Perplexity (pseudo-perplexity over the sentence itself) and zero-shot BLEU.

Perplexity question from the mission: "Is it high in the dish name?"
Answering it needs a within-sentence control, otherwise "high" may just mean
"nouns are rare". So every token is tagged dish / non-dish, and the comparison
is paired per sentence, with a bootstrap CI.

Pseudo-perplexity: labels = the sentence itself, decoder is causal, so token t is
scored given tokens < t. First token has no context and is excluded.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import config as C
from . import data as D
from . import models as M


# ------------------------------------------------------- pseudo-perplexity --
def _token_nll(tok, model, text: str):
    """Per-label NLL for one sentence.

    T5 shifts internally: `decoder_input_ids = [pad, l0, ..., l_{L-2}]`, so
    `logits[j]` predicts `labels[j]`. Applying an extra shift (as a decoder-only
    model would need) asks the model to predict the token *after next* and inflates
    every NLL by a full token of missing context.

    `logits[0]` predicts the first token from a pad token only, i.e. with no
    decoder-side context. The encoder still sees the whole sentence, which is why
    this quantity is a *copy loss*, not a corpus language-model perplexity.
    """
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=256)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(M.device())
    mask = enc["attention_mask"].to(M.device())
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, labels=ids).logits[0].float()

    logp = logits.log_softmax(-1)
    tgt = ids[0]
    true_logit = logits.gather(1, tgt.unsqueeze(1))
    nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    rank = (logits > true_logit).sum(1) + 1
    return tok.convert_ids_to_tokens(ids[0].tolist()), offsets, nll.cpu().numpy(), rank.cpu().numpy()


def perplexity(model_key: str, force: bool = False) -> str:
    """Token-level NLL for the 100 own-language sentences, tagged dish/non-dish."""
    spec = C.MODELS[model_key]
    lang = spec["lang"]
    out_path = C.RESULTS / f"ppl_tokens_{model_key}.csv"
    if out_path.exists() and not force:
        print(f"{model_key}: {out_path.name} exists, skipping (use force=True)")
        return str(out_path)

    rows = D.build(force=False)
    tok, model, _ = M.load(model_key)
    t0 = time.time()
    records = []
    for r in rows:
        text = r[lang]
        dish = r["dish_vi"] if lang == "vi" else r["dish_en"]
        start, end = D.find_span(text, dish)
        if start < 0:
            raise RuntimeError(f"id {r['id']}: dish {dish!r} not found in {lang} sentence")
        pieces, offsets, nll, rank = _token_nll(tok, model, text)
        for j in range(len(offsets)):
            s, e = offsets[j]
            if e <= s:  # special token
                continue
            in_dish = (s < end) and (e > start)
            records.append(
                {
                    "model": model_key,
                    "lang": lang,
                    "id": r["id"],
                    "pos": j,
                    "token": pieces[j],
                    "span": text[s:e],
                    "in_dish": int(in_dish),
                    "nll": float(nll[j]),
                    "rank": int(rank[j]),
                    "has_context": int(j > 0),
                }
            )
    M.free(model)

    import csv

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"{model_key}: {len(records)} tokens in {time.time() - t0:.1f}s -> {out_path.name}")
    return str(out_path)


def explain(tok, model, text: str, dish: str) -> list[dict]:
    """Per-token table for one sentence: piece, span, dish flag, NLL, rank.

    Used by the notebook (to show one sentence) and by the smoke test (to verify
    that the dish-token spans concatenate back to the dish name).
    """
    start, end = D.find_span(text, dish)
    if start < 0:
        raise RuntimeError(f"dish {dish!r} not found in {text!r}")
    pieces, offsets, nll, rank = _token_nll(tok, model, text)
    rows = []
    for j in range(len(offsets)):
        s, e = offsets[j]
        if e <= s:
            continue
        rows.append(
            {
                "pos": j,
                "piece": pieces[j].replace("\u2581", "\u2423"),
                "span": text[s:e],
                "dish": bool((s < end) and (e > start)),
                "nll": round(float(nll[j]), 3),
                "rank": int(rank[j]),
                "ctx": j > 0,
            }
        )
    return rows


def ppl_summary(force: bool = False) -> list[dict]:
    """Aggregate per model: dish vs non-dish NLL, rank, and a paired bootstrap CI."""
    import csv

    out = []
    rng = np.random.default_rng(C.SEED)
    for model_key in [k for k in C.MODEL_ORDER if (C.RESULTS / f"ppl_tokens_{k}.csv").exists()]:
        rows = list(csv.DictReader((C.RESULTS / f"ppl_tokens_{model_key}.csv").open(encoding="utf-8")))
        # token 0 is scored from a pad token only: no decoder context, excluded from stats
        rows = [r for r in rows if r.get("has_context", "1") == "1"]
        per_sentence = {}
        for r in rows:
            d = per_sentence.setdefault(int(r["id"]), {"dish": [], "other": [], "rdish": [], "rother": []})
            key = "dish" if r["in_dish"] == "1" else "other"
            d[key].append(float(r["nll"]))
            d["rdish" if key == "dish" else "rother"].append(float(r["rank"]))

        nll_all = np.array([float(r["nll"]) for r in rows])
        deltas = []
        for d in per_sentence.values():
            if d["dish"] and d["other"]:
                deltas.append(np.mean(d["dish"]) - np.mean(d["other"]))
        deltas = np.array(deltas)
        boot = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(2000)])

        dish_nll = [float(r["nll"]) for r in rows if r["in_dish"] == "1"]
        other_nll = [float(r["nll"]) for r in rows if r["in_dish"] == "0"]
        dish_rank = [float(r["rank"]) for r in rows if r["in_dish"] == "1"]
        other_rank = [float(r["rank"]) for r in rows if r["in_dish"] == "0"]
        out.append(
            {
                "model": model_key,
                "lang": C.MODELS[model_key]["lang"],
                "tokens": len(rows),
                "ppl_all": round(float(np.exp(nll_all.mean())), 2),
                "nll_all": round(float(nll_all.mean()), 3),
                "dish_tokens": len(dish_nll),
                "nll_dish": round(float(np.mean(dish_nll)), 3),
                "ppl_dish": round(float(np.exp(np.mean(dish_nll))), 2),
                "rank_dish": round(float(np.mean(dish_rank)), 1),
                "nll_other": round(float(np.mean(other_nll)), 3),
                "ppl_other": round(float(np.exp(np.mean(other_nll))), 2),
                "rank_other": round(float(np.mean(other_rank)), 1),
                "delta_nll": round(float(deltas.mean()), 3),
                "delta_ci_lo": round(float(np.quantile(boot, 0.025)), 3),
                "delta_ci_hi": round(float(np.quantile(boot, 0.975)), 3),
                "dish_harder_sentences": int((deltas > 0).sum()),
                "sentences": len(deltas),
            }
        )

    path = C.RESULTS / "ppl_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(
            f"{r['model']:<14} ppl {r['ppl_all']:8.2f} | dish {r['ppl_dish']:8.2f} (rank {r['rank_dish']:6.1f}) | "
            f"other {r['ppl_other']:8.2f} (rank {r['rank_other']:6.1f}) | delta {r['delta_nll']:+.3f} "
            f"[{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]"
        )
    return out


# ----------------------------------------------------- zero-shot translate ---
def run_translate(model_key: str, direction: str, force: bool = False) -> str:
    """direction: en2vi (vit5) or vi2en (flan-t5). Cross-model, base<->base, large<->large."""
    spec = C.MODELS[model_key]
    tgt_lang = "vi" if direction == "en2vi" else "en"
    if spec["lang"] != tgt_lang:
        raise ValueError(f"{model_key} cannot produce {tgt_lang} target for {direction}")
    src_lang = "en" if direction == "en2vi" else "vi"

    raw_path = C.RESULTS / "raw" / f"tr_{model_key}_{direction}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if force and raw_path.exists():
        raw_path.unlink()

    done = set()
    if raw_path.exists():
        done = {json.loads(l)["id"] for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    rows = [r for r in D.build(force=False) if r["id"] not in done]
    if not rows:
        print(f"{model_key} {direction}: nothing to do")
        return str(raw_path)

    tok, model, _ = M.load(model_key)
    t0 = time.time()
    bs = spec["batch"]
    with raw_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(rows), bs):
            chunk = rows[start : start + bs]
            prompts = [C.TRANSLATE_PROMPT[direction].format(r[src_lang]) for r in chunk]
            hyps = _generate(tok, model, prompts, 96)
            for r, hyp in zip(chunk, hyps):
                fh.write(json.dumps({"id": r["id"], "hyp": hyp}, ensure_ascii=False) + "\n")
            fh.flush()
    M.free(model)
    print(f"{model_key} {direction}: {len(rows)} rows in {time.time() - t0:.1f}s")
    return str(raw_path)


def _generate(tok, model, prompts, max_new_tokens):
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


def translate_summary(force: bool = False) -> list[dict]:
    import csv
    import unicodedata

    import sacrebleu

    pairs = {r["id"]: r for r in D.build(force=False)}
    out = []
    for raw_path in sorted((C.RESULTS / "raw").glob("tr_*.jsonl")):
        model_key, direction = raw_path.stem[3:].rsplit("_", 1)
        tgt_lang = "vi" if direction == "en2vi" else "en"
        tokz = "char" if tgt_lang == "vi" else "13a"
        rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows.sort(key=lambda r: r["id"])

        hyps = [r["hyp"] for r in rows]
        refs = [(pairs[r["id"]]["vi"] if tgt_lang == "vi" else pairs[r["id"]]["en"]) for r in rows]
        dish_key = "dish_vi" if tgt_lang == "vi" else "dish_en"

        def norm(s):
            return unicodedata.normalize("NFC", s).lower()

        kept = sum(norm(pairs[r["id"]][dish_key]) in norm(h) for r, h in zip(rows, hyps)) / len(rows)
        out.append(
            {
                "model": model_key,
                "direction": direction,
                "n": len(rows),
                "bleu": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokz).score, 2),
                "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
                "dish_name_kept": round(kept, 4),
                "empty_rate": round(sum(not h.strip() for h in hyps) / len(rows), 4),
                "len_ratio": round(np.mean([len(h.split()) / max(1, len(s.split())) for h, s in zip(hyps, refs)]), 3),
            }
        )

    path = C.RESULTS / "bleu_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['model']:<14} {r['direction']}  BLEU {r['bleu']:6.2f}  chrF {r['chrf']:6.2f}  dish_kept {r['dish_name_kept']:.2f}")
    return out
