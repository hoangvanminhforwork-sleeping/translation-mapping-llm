"""Encode the 100k pairs into sentence vectors with both models.

Two matrices are the whole input to the mapping:
    enc_vit5_vi.npy      (n, 768)  vit5-base encoder, masked-mean pooled, Vietnamese text
    enc_flan_en.npy      (n, 768)  flan-t5-base + LoRA adapter, encoder, masked-mean pooled, English

f: vit5_vi -> flan_en is the vi->en direction; g reads the same two matrices backwards.
enc_flan_vi_test.npy is a control: flan's own encoder on the Vietnamese *test* sentences, used in
`bleu.py` to separate "the mapping lost information" from "the decoder cannot translate".

The encoder module is taken from `base_model.model` when a PEFT wrapper is present, because
`PeftModelForSeq2SeqLM` does not expose `.encoder` directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

DT = torch.bfloat16
BATCH = 96
MAX_LEN = 64


def load_encoder(model_key: str, with_adapter: bool):
    """Return (tokenizer, encoder_module, d_model) ready for forward passes on cuda."""
    spec = C.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_id"], dtype=DT, low_cpu_mem_usage=True)
    if with_adapter:
        from peft import PeftModel

        adapter = C.ADAPTERS / model_key
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"missing adapter at {adapter}; train it first")
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.to("cuda").eval()
    return tok, model.encoder, int(model.config.d_model)


@torch.no_grad()
def encode_texts(model_key: str, texts: list[str], with_adapter: bool, tag: str,
                 force: bool = False) -> Path:
    out = C.DATA / f"enc_{tag}.npy"
    if out.exists() and not force:
        print(f"{tag}: exists, reusing {out.name}")
        return out
    tok, encoder, d_model = load_encoder(model_key, with_adapter)
    # Length bucketing: padding waste dominates when sentences are 20-300 chars.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vec = np.zeros((len(texts), d_model), dtype=np.float32)
    t0, done = time.time(), 0
    for s in range(0, len(order), BATCH):
        idx = order[s : s + BATCH]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to("cuda")
        h = encoder(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"]).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec[idx] = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        done += len(idx)
        if done % (BATCH * 25) < BATCH or done == len(order):
            el = max(time.time() - t0, 1e-9)
            print(f"  {tag} {done}/{len(order)}  {done/el:6.1f} seq/s  "
                  f"eta {(len(order)-done)/(done/el)/60:5.1f} min", flush=True)
    np.save(out, vec)
    json.dump({"tag": tag, "model": model_key, "adapter": with_adapter, "rows": len(texts),
               "dim": d_model, "batch": BATCH, "max_len": MAX_LEN, "pooling": "masked_mean",
               "seconds": round(time.time() - t0, 1), "file": out.name},
              (C.DATA / f"enc_{tag}.json").open("w"), indent=2)
    print(f"{tag}: saved {out.name} {vec.shape} in {time.time()-t0:.0f}s")
    del encoder, tok
    torch.cuda.empty_cache()
    return out


def build(force: bool = False, limit: int | None = None) -> None:
    t = pq.read_table(C.PAIRS)
    vi, en, split = t["vi"].to_pylist(), t["en"].to_pylist(), t["split"].to_pylist()
    test_vi = [v for v, s in zip(vi, split) if s == "test"]
    if limit:
        vi, en, test_vi = vi[:limit], en[:limit], test_vi[:min(limit, len(test_vi))]
        print(f"LIMIT ACTIVE: {limit} rows (smoke test, do not report these)")
    print(f"pairs: {len(vi)} ({len(test_vi)} test)")
    encode_texts("vit5-base", vi, False, "vit5_vi", force)
    encode_texts("flan-t5-base", en, True, "flan_en", force)
    encode_texts("flan-t5-base", test_vi, True, "flan_vi_test", force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.force, a.limit)
