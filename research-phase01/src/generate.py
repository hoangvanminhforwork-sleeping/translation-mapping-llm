"""Generate English from the mapped / native / identity channels of the same 2,000 test pairs.

    mapped    vit5(vi) -> W -> flan decoder          (the proposal)
    native    flan encoder(vi) -> flan decoder       (same decoder, its own encoder reading Vietnamese)
    identity  vit5(vi) -> flan decoder, no map       (control: is W doing the work?)

All three use one adapter, greedy decoding, max 96 new tokens, and the same 2,000 pairs, so the only
difference between rows is the channel.

Usage
    python -m src.generate --n 2000 [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from . import config as C
from .fit_tok import iter_slices
from .train_decoder import Channels, DT

sys.stdout.reconfigure(encoding="utf-8")


def load_model():
    flan = AutoModelForSeq2SeqLM.from_pretrained(
        C.MODELS["flan-t5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True)
    if not (C.DECODER_ADAPTER / "adapter_config.json").exists():
        raise SystemExit(f"missing adapter at {C.DECODER_ADAPTER}; run src.train_decoder first")
    model = PeftModel.from_pretrained(flan, str(C.DECODER_ADAPTER)).to("cuda").eval()
    return model


@torch.no_grad()
def run(model, ch: Channels, pairs: list[tuple[str, str]], cond: str, batch: int, max_new: int):
    out = []
    t0 = time.time()
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        vi = [p[0] for p in chunk]
        if cond == "native":
            h, am = ch.native(vi)
        else:
            h, am = ch.mapped(vi, use_map=(cond == "mapped"))
        gen = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=am,
                             max_new_tokens=max_new, num_beams=1, do_sample=False)
        hyp = ch.en_tok.batch_decode(gen, skip_special_tokens=True)
        out.extend(hyp)
        if (s // batch) % 10 == 0:
            done = s + len(chunk)
            print(f"  {cond} {done}/{len(pairs)}  {done / (time.time() - t0):.1f} sent/s", flush=True)
    return out


def main(n: int, batch: int, max_new: int, force: bool):
    chunk = None
    for _i, vi, en in iter_slices("test", n):
        chunk = (vi, en)
        break
    vi, en = chunk
    pairs = list(zip(vi, en))
    print(f"test pairs: {len(pairs)}")

    todo = [c for c in ("mapped", "native", "identity")
            if force or not (C.RESULTS / f"gen_{c}.jsonl").exists()]
    if not todo:
        print("all generation files exist; use --force to redo")
        return
    model = load_model()
    ch = Channels(model)
    for cond in todo:
        hyps = run(model, ch, pairs, cond, batch, max_new)
        path = C.RESULTS / f"gen_{cond}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, (v, e, hyp) in enumerate(zip(vi, en, hyps)):
                f.write(json.dumps({"id": i, "vi": v, "ref": e, "hyp": hyp}, ensure_ascii=False) + "\n")
        print(f"{cond}: wrote {path.name} ({len(hyps)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.n, a.batch, a.max_new, a.force)
