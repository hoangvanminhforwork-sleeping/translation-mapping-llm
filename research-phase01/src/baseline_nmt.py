"""Calibration baseline: a dedicated vi->en NMT model on the same 2,000 test pairs.

Why: a BLEU of 6.28 from a base-size decoder trained for one epoch is only interpretable next to what a
*purpose-built* translation model achieves on the same sentences. `Helsinki-NLP/opus-mt-vi-en` is 77M
parameters, downloads in seconds and needs no adapter, so it is a fair "what a real NMT model does here"
reference — not a competitor to the method (different task budget, different training data).

Writes results/gen_baseline.jsonl in the same schema, so src.score treats it as another channel.
"""
from __future__ import annotations

import json
import sys
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C
from .fit_tok import iter_slices

sys.stdout.reconfigure(encoding="utf-8")
MODEL = "Helsinki-NLP/opus-mt-vi-en"


def main(n: int = 2000, batch: int = 32):
    out_path = C.RESULTS / "gen_baseline.jsonl"
    if out_path.exists():
        print(f"{out_path.name} exists; delete it to rerun")
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.to("cuda").eval()

    rows = None
    for _i, vi, en in iter_slices("test", n):
        rows = list(zip(vi, en))
        break
    print(f"{len(rows)} pairs")
    t0 = time.time()
    hyps = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            enc = tok([r[0] for r in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=96).to("cuda")
            gen = model.generate(**enc, max_new_tokens=96, num_beams=1, do_sample=False)
            hyps.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (s // batch) % 10 == 0:
                print(f"  {min(s + batch, len(rows))}/{len(rows)}  "
                      f"{min(s + batch, len(rows)) / (time.time() - t0):.1f} sent/s", flush=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, ((vi, en), hyp) in enumerate(zip(rows, hyps)):
            f.write(json.dumps({"id": i, "vi": vi, "ref": en, "hyp": hyp}, ensure_ascii=False) + "\n")
    print(f"wrote {out_path.name} ({len(hyps)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
