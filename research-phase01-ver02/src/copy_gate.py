"""The copy gate: can each model reproduce a sentence from its own encoder states?

Mission §2: "check whether the model can copy exactly the input... Ignore 'Nói' and 'Say', just
{s_e} => {s_e}". So the task is identity through the whole model, no prompt:

    encoder(s) -> decoder -> s

Why it is a gate and not a nicety: the proposal injects *predicted target vectors* into the
decoder's cross-attention. If the decoder cannot reproduce a sentence from vectors that are
already exactly right (its own encoder's output), no mapping can be blamed for the failure, and
no BLEU number downstream means anything. Phase 01 measured 0 % / 2.65 % zero-shot on the
220M models; here we measure the 60M pair and fine-tune a decoder-only LoRA if needed.

Usage
    python -m src.copy_gate eval  --side en|vi [--adapter]
    python -m src.copy_gate train --side en|vi [--pairs 50000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from . import config as C
from .embed import load_chosen
from .tokutil import load_tok

sys.stdout.reconfigure(encoding="utf-8")
DT = torch.bfloat16
N_EVAL = 2000


def decoder_targets(n_layers: int = 6) -> list[str]:
    names = []
    for i in range(n_layers):
        b = f"decoder.block.{i}"
        names += [f"{b}.layer.0.SelfAttention.{p}" for p in ("q", "k", "v", "o")]
        names += [f"{b}.layer.1.EncDecAttention.{p}" for p in ("q", "k", "v", "o")]
        names += [f"{b}.layer.2.DenseReluDense.{p}" for p in ("wi_0", "wi_1", "wo")]
    return names


def _rows(side: str, n: int, split: str = "lora"):
    """Sentences for the copy task. `lora` = the dirty shard (train-00000), never the map corpus.

    The lora split is reservoir-sampled while streaming: holding all 500k rows of one shard costs
    ~150 MB and a full scan, and we only need 50k of them.
    """
    col = "vi" if side == "vi" else "en"
    if split == "test":
        d = load_chosen((col, "split"))
        idx = [i for i, s in enumerate(d["split"]) if s == "test"]
        rng = np.random.default_rng(42)
        pick = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
        return [d[col][idx[j]] for j in pick]

    rng = np.random.default_rng(42)
    out: list[str] = []
    seen = 0
    pf = pq.ParquetFile(C.PAIRS)
    for rb in pf.iter_batches(batch_size=16384, columns=["file", col]):
        d = rb.to_pydict()
        for i, f in enumerate(d["file"]):
            if f != C.LORA_FILE:
                continue
            v = d[col][i]
            if len(out) < n:
                out.append(v)
            else:
                j = int(rng.integers(0, seen + 1))
                if j < n:
                    out[j] = v
            seen += 1
    print(f"copy rows: reservoir {len(out):,} of {seen:,} in {C.LORA_FILE}", flush=True)
    return out


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def encoder_of(model):
    """The frozen encoder, whether or not the model is wrapped in a PEFT adapter."""
    if isinstance(model, PeftModel):
        return model.base_model.model.encoder
    return model.encoder


def load_model(side: str, adapter: bool):
    m = T5ForConditionalGeneration.from_pretrained(str(C.MODELS[side]["path"]), dtype=DT).to("cuda").eval()
    p = C.ADAPTERS[f"copy_{side}"]
    if adapter and (p / "adapter_config.json").exists():
        m = PeftModel.from_pretrained(m, str(p)).to("cuda").eval()
    return m


@torch.no_grad()
def evaluate(side: str, adapter: bool, n: int = N_EVAL, batch: int = 64, tag: str = ""):
    tok = load_tok(side)
    model = load_model(side, adapter)
    sents = _rows(side, n)
    exact = []
    hyps = []
    t0 = time.time()
    for s in range(0, len(sents), batch):
        chunk = sents[s:s + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        am = enc["attention_mask"].cuda()
        h = encoder_of(model)(input_ids=ids, attention_mask=am).last_hidden_state
        gen = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h),
                             attention_mask=am, max_new_tokens=C.MAX_LEN, num_beams=1, do_sample=False)
        out = tok.batch_decode(gen, skip_special_tokens=True)
        hyps += out
        exact += [int(_norm(o) == _norm(r)) for o, r in zip(out, chunk)]
        if (s // batch) % 8 == 0:
            print(f"  {s + len(chunk)}/{len(sents)}  {len(sents) / (time.time() - t0):.1f} sent/s", flush=True)
    res = {"side": side, "adapter": adapter, "tag": tag, "n": len(sents),
           "exact_match": float(np.mean(exact)), "empty_rate": float(np.mean([len(h.split()) == 0 for h in hyps])),
           "mean_hyp_words": float(np.mean([len(h.split()) for h in hyps])),
           "mean_ref_words": float(np.mean([len(r.split()) for r in sents])),
           "sent_per_s": round(len(sents) / (time.time() - t0), 1)}
    p = C.RESULTS / f"copy_gate_{side}{'_lora' if adapter else ''}.json"
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (C.RESULTS / f"copy_gate_{side}{'_lora' if adapter else ''}.jsonl").open("w", encoding="utf-8") as f:
        for r, h in zip(sents, hyps):
            f.write(json.dumps({"ref": r, "hyp": h}, ensure_ascii=False) + "\n")
    print(f"{side} adapter={adapter}: exact {res['exact_match']:.3f}  empty {res['empty_rate']:.3f}  "
          f"words {res['mean_hyp_words']:.1f}/{res['mean_ref_words']:.1f}  -> {p.name}")
    return res


def train(side: str, pairs: int, batch: int, lr: float):
    out = C.ADAPTERS[f"copy_{side}"]
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tok(side)
    model = T5ForConditionalGeneration.from_pretrained(str(C.MODELS[side]["path"]), dtype=DT).to("cuda")
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=decoder_targets(int(model.config.num_layers)), task_type="SEQ_2_SEQ_LM")
    if (out / "adapter_config.json").exists():
        print(f"resuming {out}")
        model = PeftModel.from_pretrained(model, str(out), is_trainable=True)
    else:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sents = _rows(side, pairs)
    print(f"copy training: {len(sents):,} {side} sentences, batch {batch}, lr {lr}", flush=True)
    log = C.RESULTS / f"copy_train_{side}.csv"
    log.write_text("step,loss,sent_per_s\n", encoding="utf-8")
    rng = np.random.default_rng(1234)
    order = rng.permutation(len(sents))
    t0, step, buf = time.time(), 0, []
    for s in range(0, len(order), batch):
        chunk = [sents[i] for i in order[s:s + batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        am = enc["attention_mask"].cuda()
        lab = ids.clone()
        lab[am == 0] = -100
        out_ = model(input_ids=ids, attention_mask=am, labels=lab)
        out_.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        buf.append((step, float(out_.loss.item()), (s + len(chunk)) / (time.time() - t0)))
        if step % 50 == 0:
            mean_loss = float(np.mean([b[1] for b in buf])) if buf else float(out_.loss.item())
            with log.open("a", encoding="utf-8") as f:
                for r in buf:
                    f.write(f"{r[0]},{r[1]:.5f},{r[2]:.1f}\n")
            buf = []
            print(f"  step {step}  loss {mean_loss:.4f}  "
                  f"{(s + len(chunk)) / (time.time() - t0):.1f} sent/s  "
                  f"eta {(len(order) - s - len(chunk)) / ((s + len(chunk)) / (time.time() - t0)) / 60:.1f} min", flush=True)
            if step % 200 == 0:
                model.save_pretrained(out)
        if step * batch >= len(order):
            break
    if buf:
        with log.open("a", encoding="utf-8") as f:
            for r in buf:
                f.write(f"{r[0]},{r[1]:.5f},{r[2]:.1f}\n")
    model.save_pretrained(out)
    (out / "done.json").write_text(json.dumps(
        {"side": side, "pairs": len(sents), "steps": step, "wall_min": round((time.time() - t0) / 60, 1),
         "batch": batch, "lr": lr, "lora_r": 16, "targets": "decoder only", "prompt": "none (identity)"},
        indent=2), encoding="utf-8")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "train"])
    ap.add_argument("--side", choices=["en", "vi"], required=True)
    ap.add_argument("--adapter", action="store_true")
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--pairs", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    if a.mode == "eval":
        evaluate(a.side, a.adapter, a.n)
    else:
        train(a.side, a.pairs, a.batch, a.lr)
