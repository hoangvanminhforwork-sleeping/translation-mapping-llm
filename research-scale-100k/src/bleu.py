"""vi -> en translation from *mapped representations*, scored with corpus BLEU.

Three runs on the same 1,000 test sentences, same decoder, same decode settings:

    zeroshot_vit5   vit5-base alone, prompted  ("Dịch sang tiếng Anh: ...")
    mapped          vit5 encodes vi -> f -> flan-t5-base+LoRA decoder generates
    flan_encoder    flan-t5-base+LoRA encodes the *Vietnamese* text itself -> same decoder generates

`mapped` is the research claim. `flan_encoder` is the control that separates two failure modes:
if it is also bad, the decoder cannot translate from a single pooled vector at all; if it is good
and `mapped` is bad, the mapping is what destroys the information.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from . import config as C

DT = torch.bfloat16
PROMPT_VI_EN = "Dịch sang tiếng Anh: "


def _flan():
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(C.MODELS["flan-t5-base"]["hf_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        C.MODELS["flan-t5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True
    )
    adapter = C.ADAPTERS / "flan-t5-base"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"missing flan adapter at {adapter}; train it first")
    model = PeftModel.from_pretrained(model, str(adapter)).to("cuda").eval()
    return tok, model


def _vit5():
    tok = AutoTokenizer.from_pretrained(C.MODELS["vit5-base"]["hf_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        C.MODELS["vit5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True
    ).to("cuda").eval()
    return tok, model


@torch.no_grad()
def _gen_from_vectors(model, tok, vectors: np.ndarray) -> list[str]:
    """Feed mapped vectors straight into the decoder's cross-attention."""
    out = []
    for s in range(0, len(vectors), C.GEN_BATCH):
        v = torch.tensor(vectors[s : s + C.GEN_BATCH], dtype=torch.float32).to("cuda")
        v = v.unsqueeze(1).to(DT)                     # (B, 1, 768): one "token" per sentence
        enc = BaseModelOutput(last_hidden_state=v)
        attn = torch.ones((v.shape[0], 1), dtype=torch.long, device="cuda")
        ids = model.generate(encoder_outputs=enc, attention_mask=attn,
                             max_new_tokens=C.MAX_NEW_TOKENS, num_beams=1, do_sample=False)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


@torch.no_grad()
def _gen_from_text_encoder(model, tok, texts: list[str]) -> list[str]:
    out = []
    for s in range(0, len(texts), C.GEN_BATCH):
        enc = tok(texts[s : s + C.GEN_BATCH], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to("cuda")
        ids = model.generate(**enc, max_new_tokens=C.MAX_NEW_TOKENS, num_beams=1, do_sample=False)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


@torch.no_grad()
def _gen_vit5(vit5, vtok, texts: list[str]) -> list[str]:
    out = []
    for s in range(0, len(texts), C.GEN_BATCH):
        enc = vtok([PROMPT_VI_EN + t for t in texts[s : s + C.GEN_BATCH]], padding=True,
                   truncation=True, max_length=80, return_tensors="pt").to("cuda")
        ids = vit5.generate(**enc, max_new_tokens=C.MAX_NEW_TOKENS, num_beams=1, do_sample=False)
        out += vtok.batch_decode(ids, skip_special_tokens=True)
    return out


def run(limit: int | None = None, force: bool = False) -> dict:
    import sacrebleu

    t = pq.read_table(C.PAIRS, columns=["vi", "en", "split"])
    vi = [a for a, s in zip(t["vi"].to_pylist(), t["split"].to_pylist()) if s == "test"]
    en = [b for b, s in zip(t["en"].to_pylist(), t["split"].to_pylist()) if s == "test"]
    if limit:
        vi, en = vi[:limit], en[:limit]
    print(f"test sentences: {len(vi)}")

    split = np.array(t["split"].to_pylist())
    te = split == "test"
    x_vi = np.load(C.DATA / "enc_vit5_vi.npy")[te][: len(vi)]
    m = np.load(C.DATA / "map_vi_en.npz")
    mapped_vec = x_vi @ m["w"] + m["b"]

    runs: dict[str, list[str]] = {}
    ftok, flan = _flan()

    if force or not (C.RESULTS / "gen_mapped.jsonl").exists():
        t0 = time.time()
        runs["mapped"] = _gen_from_vectors(flan, ftok, mapped_vec)
        print(f"mapped: {len(runs['mapped'])} in {time.time()-t0:.0f}s")
    if force or not (C.RESULTS / "gen_flan_encoder.jsonl").exists():
        t0 = time.time()
        runs["flan_encoder"] = _gen_from_text_encoder(flan, ftok, vi)
        print(f"flan_encoder: {time.time()-t0:.0f}s")
    del flan
    torch.cuda.empty_cache()
    if force or not (C.RESULTS / "gen_zeroshot_vit5.jsonl").exists():
        vt, vit5 = _vit5()
        t0 = time.time()
        runs["zeroshot_vit5"] = _gen_vit5(vit5, vt, vi)
        print(f"zeroshot_vit5: {time.time()-t0:.0f}s")

    scores = {}
    for name in ("mapped", "flan_encoder", "zeroshot_vit5"):
        path = C.RESULTS / f"gen_{name}.jsonl"
        if name in runs:                      # fresh generations this call: always overwrite
            with path.open("w", encoding="utf-8") as fh:
                for s, h in zip(vi, runs[name]):
                    fh.write(json.dumps({"vi": s, "hyp": h}, ensure_ascii=False) + "\n")
        if not path.exists():
            continue
        hyps = [json.loads(l)["hyp"] for l in path.read_text(encoding="utf-8").splitlines()]
        if len(hyps) != len(en):
            print(f"skip {name}: {len(hyps)} hypotheses vs {len(en)} references")
            continue
        bleu = sacrebleu.corpus_bleu(hyps, [en])
        scores[name] = {"bleu": round(bleu.score, 3),
                        "chrf": round(sacrebleu.corpus_chrf(hyps, [en]).score, 3),
                        "empty": sum(1 for h in hyps if not h.strip()),
                        "mean_hyp_tokens": round(float(np.mean([len(h.split()) for h in hyps])), 2),
                        "n": len(hyps)}
    ref_words = float(np.mean([len(s.split()) for s in en]))
    out = {"n": len(vi), "mean_ref_tokens": round(ref_words, 2),
           "note": "sacrebleu corpus_bleu default tokenizer 13a, lowercase=False",
           "scores": scores}
    (C.RESULTS / "bleu_vi_en.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.limit, a.force)
