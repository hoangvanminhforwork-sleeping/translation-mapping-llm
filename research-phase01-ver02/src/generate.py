"""Generate the target language from every channel of the same test pairs.

    mapped    source encoder -> ridge W -> target decoder          (the proposal)
    lgbm      source encoder -> 512 LightGBM boosters -> decoder   (nonlinear alternative)
    native    target model's own encoder reading the source text   (same decoder, no map)
    identity  source encoder -> decoder, no map                    (control: is W doing the work?)
    copy      source encoder -> ridge W -> *copy* adapter          (pure vector readout)

All channels use one adapter (except `copy`), greedy decoding, and the same pairs, so the only
difference between rows is the channel. Per sentence we also record the mean NLL of the reference
under that channel: it is the decoder-side quantity the mapping is supposed to minimise, and it is
what the neural-mining section correlates with representation error.

Usage
    python -m src.generate --direction vi_en --n 5000 [--channels mapped,native] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from . import config as C
from .embed import load_chosen
from .train_decoder import DT, Channels, direction_sides

sys.stdout.reconfigure(encoding="utf-8")
ALL = ("mapped", "lgbm", "native", "identity", "copy")


def load_model(direction: str, kind: str):
    _, tgt = direction_sides(direction)
    base = T5ForConditionalGeneration.from_pretrained(str(C.MODELS[tgt]["path"]), dtype=DT,
                                                      low_cpu_mem_usage=True)
    key = f"copy_{tgt}" if kind == "copy" else f"trans_{tgt}"
    p = C.ADAPTERS[key]
    if not (p / "adapter_config.json").exists():
        raise SystemExit(f"missing adapter {p}; train it first")
    return PeftModel.from_pretrained(base, str(p)).to("cuda").eval()


def test_pairs(direction: str, n: int):
    src, tgt = direction_sides(direction)
    d = load_chosen(("vi", "en", "split"))
    te = [i for i, s in enumerate(d["split"]) if s == "test"]
    return [(d[src][i], d[tgt][i]) for i in te[:n]]


@torch.no_grad()
def nll(model, ch, h, am, refs):
    """Mean NLL of the reference under the given channel (teacher forcing)."""
    lab = ch.labels(refs)
    out = model(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=am, labels=lab)
    # per-sentence NLL: recompute without reduction
    logits = out.logits.float()
    lp = torch.log_softmax(logits, -1)
    tgt = lab.clone()
    tgt[tgt == -100] = 0
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = (lab != -100).float()
    return (-(tok_lp * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()


@torch.no_grad()
def run(model, ch, pairs, cond, batch, max_new, lgbm=None):
    src_txt = [p[0] for p in pairs]
    refs = [p[1] for p in pairs]
    hyps, nlls = [], []
    t0 = time.time()
    for s in range(0, len(pairs), batch):
        bsrc = src_txt[s:s + batch]
        bref = refs[s:s + batch]
        if cond == "native":
            h, am = ch.native(bsrc)
        elif cond == "mapped":
            h, am = ch.mapped(bsrc, use_map=True)
        elif cond == "identity":
            h, am = ch.mapped(bsrc, use_map=False)
        elif cond == "lgbm":
            # `mapped` hands the mapper a flat float32 numpy array and re-shapes the result
            h, am = ch.mapped(bsrc, mapper=lgbm)
        else:  # copy
            h, am = ch.mapped(bsrc, use_map=True)
        gen = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=am,
                             max_new_tokens=max_new, num_beams=1, do_sample=False)
        hyps += ch.tok[ch.tgt].batch_decode(gen, skip_special_tokens=True)
        nlls += list(nll(model, ch, h, am, bref))
        if (s // batch) % 10 == 0:
            done = s + len(bsrc)
            print(f"  {cond} {done}/{len(pairs)}  {done / (time.time() - t0):.1f} sent/s", flush=True)
    return hyps, nlls


def main(direction: str, n: int, batch: int, max_new: int, conds: list[str], force: bool):
    pairs = test_pairs(direction, n)
    print(f"{direction}: {len(pairs):,} test pairs", flush=True)
    lgbm = None
    if "lgbm" in conds:
        from .fit_lgbm import LgbmMap
        lgbm = LgbmMap(direction)
    for cond in conds:
        out = C.RESULTS / f"gen_{direction}_{cond}.jsonl"
        if out.exists() and not force:
            print(f"{cond}: exists, skip")
            continue
        model = load_model(direction, "copy" if cond == "copy" else "trans")
        ch = Channels(model, direction)
        hyps, nlls = run(model, ch, pairs, cond, batch, max_new, lgbm)
        with out.open("w", encoding="utf-8") as f:
            for i, ((s_, r), h, nl) in enumerate(zip(pairs, hyps, nlls)):
                f.write(json.dumps({"id": i, "src": s_, "ref": r, "hyp": h, "nll": float(nl)},
                                   ensure_ascii=False) + "\n")
        print(f"{cond}: wrote {out.name} ({len(hyps)} rows)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["vi_en", "en_vi"], required=True)
    ap.add_argument("--n", type=int, default=C.N_GEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--channels", default=",".join(ALL))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.direction, a.n, a.batch, a.max_new, [c for c in a.channels.split(",") if c], a.force)
