"""Decoder control: repeat the vi->en comparison with the *plain* flan-t5-base decoder.

Why: the LoRA adapter in `../finetune-and-lora` was trained to copy its input, so with it the decoder
cannot translate and BLEU cannot attribute loss to the map (`REPORT.md` §4). Plain flan-t5-base keeps its
instruction-tuning, including its translation behaviour, so this run is the first one where BLEU can say
something about the representation.

Same three things are measured as in `bleu.py`, on the same 1,000 test sentences:
    mapped_base          vit5 encodes vi -> ridge map into *base* flan space -> plain flan decoder
    flan_encoder_base    plain flan encoder reads the Vietnamese -> same decoder  (the real ceiling)
    zeroshot_vit5        reused from bleu.py (it does not involve flan at all)

Idempotent: skips any artefact that already exists unless --force.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import bleu as B
from . import config as C
from . import encode as E
from . import mapping as M


def _plain_flan():
    tok = AutoTokenizer.from_pretrained(C.MODELS["flan-t5-base"]["hf_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        C.MODELS["flan-t5-base"]["hf_id"], dtype=E.DT, low_cpu_mem_usage=True
    ).to("cuda").eval()
    return tok, model


def _fit(x, y, split, tag: str, force: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    out = C.DATA / f"map_{tag}.npz"
    tr, va = split == "train", split == "val"
    best = None
    sweep = []
    for lam in C.RIDGE_LAMBDAS:
        w, b = M.fit_ridge(x[tr], y[tr], lam)
        m = M._metrics(x[va] @ w + b, y[va], y[tr])
        sweep.append({"lambda": lam, **m})
        score = m["cosine"] - 0.5 * m["mse"]
        if best is None or score > best[0]:
            best = (score, lam, w, b, m)
    _, lam, w, b, val_m = best
    test_m = M._metrics(x[split == "test"] @ w + b, y[split == "test"], y[tr])
    ident = M._metrics(x[split == "test"], y[split == "test"], y[tr])
    if force or not out.exists():
        np.savez(out, w=w.astype(np.float32), b=b.astype(np.float32), lambda_=lam)
    summary = {"lambda": lam, "val": val_m, "test": test_m, "control_identity_test": ident,
               "sweep": sweep}
    (C.RESULTS / f"control_base_mapping_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("lambda", "val", "test", "control_identity_test")},
                     indent=2))
    return w, b, summary


def run(force: bool = False) -> dict:
    import sacrebleu

    t = pq.read_table(C.PAIRS, columns=["vi", "en", "split"])
    vi_all, en_all = t["vi"].to_pylist(), t["en"].to_pylist()
    split = np.array(t["split"].to_pylist())
    te = split == "test"
    vi_te = [a for a, s in zip(vi_all, split) if s == "test"]
    en_te = [b for b, s in zip(en_all, split) if s == "test"]
    print(f"test sentences: {len(vi_te)}")

    # 1. encode the English side into the *base* flan space (the adapter-free model).
    E.encode_texts("flan-t5-base", en_all, False, "flan_en_base", force)
    E.encode_texts("flan-t5-base", vi_te, False, "flan_vi_test_base", force)
    x_vi = np.load(C.DATA / "enc_vit5_vi.npy")            # source, unchanged (vit5 is never adapt-tuned here)
    x_en_base = np.load(C.DATA / "enc_flan_en_base.npy")  # target, base flan

    # 2. fit into the base space.
    w, b, _ = _fit(x_vi, x_en_base, split, "vi_en_base", force)

    # 3. generate with the plain decoder.
    tok, model = _plain_flan()
    mapped_vec = x_vi[te] @ w + b
    gens = {}
    if force or not (C.RESULTS / "gen_base_mapped.jsonl").exists():
        gens["mapped_base"] = B._gen_from_vectors(model, tok, mapped_vec)
        print(f"mapped_base: {len(gens['mapped_base'])}", flush=True)
    if force or not (C.RESULTS / "gen_base_flan_encoder.jsonl").exists():
        gens["flan_encoder_base"] = B._gen_from_text_encoder(model, tok, vi_te)
        print(f"flan_encoder_base: {len(gens['flan_encoder_base'])}", flush=True)
    for name, hyps in gens.items():
        with (C.RESULTS / f"gen_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for s, h in zip(vi_te, hyps):
                fh.write(json.dumps({"vi": s, "hyp": h}, ensure_ascii=False) + "\n")

    # 4. score, plus the structural diagnostics that make a BLEU table readable.
    scores = {}
    for name in ("mapped_base", "flan_encoder_base", "zeroshot_vit5"):
        path = C.RESULTS / f"gen_{name}.jsonl"
        if not path.exists():
            continue
        hyps = [json.loads(l)["hyp"] for l in path.read_text(encoding="utf-8").splitlines()]
        if len(hyps) != len(en_te):
            print(f"skip {name}: {len(hyps)} vs {len(en_te)}")
            continue
        scores[name] = {
            "bleu": round(sacrebleu.corpus_bleu(hyps, [en_te]).score, 3),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [en_te]).score, 3),
            "empty": sum(1 for h in hyps if not h.strip()),
            "mean_hyp_tokens": round(float(np.mean([len(h.split()) for h in hyps])), 2),
            "degenerate": round(float(np.mean([
                (1.0 - len(set(h.lower().split())) / max(1, len(h.split()))) > 0.5 for h in hyps])), 3),
            "n": len(hyps),
        }
    out = {"n": len(vi_te), "decoder": "google/flan-t5-base (no adapter)", "scores": scores}
    (C.RESULTS / "bleu_vi_en_base_decoder.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.force)
