"""Train a decoder that can read *either* channel, so the BLEU gap between channels is attributable
to the map and not to the decoder.

Two conditions, same pair, same target (the English reference), same decoder weights:
    native  flan-t5-base's own frozen encoder reads the Vietnamese text
    mapped  vit5-base's frozen encoder reads the Vietnamese; W maps every token into flan space and
            the decoder's cross-attention reads that sequence instead

The encoder never trains (LoRA sits on decoder modules only, by exact module name). That is
deliberate: if the encoder moved, the native states would drift out of the space W was fitted into
and the two conditions would stop being comparable.

Without this stage the BLEU comparison is meaningless. research-scale-100k measured 0.141 / 3.115 /
0.240 on three runs whose decoder had been taught to *copy* its input, so none of those numbers was
a translation measurement (see that folder's REPORT.md §3.3).

Usage
    python -m src.train_decoder --pairs 100000 --batch 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from . import config as C
from .fit_tok import iter_slices
from .tokutil import PairTokenizer

sys.stdout.reconfigure(encoding="utf-8")
DT = torch.bfloat16


def decoder_targets(n_layers: int = 12) -> list[str]:
    """Every decoder-side projection, by exact module suffix (PEFT matches on endswith)."""
    names = []
    for i in range(n_layers):
        b = f"decoder.block.{i}"
        base = [f"{b}.layer.0.SelfAttention.{p}" for p in ("q", "k", "v", "o")]
        cross = [f"{b}.layer.1.EncDecAttention.{p}" for p in ("q", "k", "v", "o")]
        ffn = [f"{b}.layer.2.DenseReluDense.{p}" for p in ("wi_0", "wi_1", "wo")]
        names += base + cross + ffn
    return names


class Channels:
    """Builds (encoder_outputs, attention_mask) for each condition."""

    def __init__(self, peft_model):
        self.tok = PairTokenizer(C.MAX_LEN)
        self.en_tok = AutoTokenizer.from_pretrained(C.MODELS["flan-t5-base"]["hf_id"], use_fast=True)
        self.t5 = peft_model.base_model.model  # frozen encoder lives here
        self.vit5 = AutoModelForSeq2SeqLM.from_pretrained(
            C.MODELS["vit5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True).to("cuda").eval()
        self.vit5.requires_grad_(False)
        z = np.load(C.MAP_VI_EN)
        self.W = torch.from_numpy(z["W"].astype(np.float32)).cuda()
        self.b = torch.from_numpy(z["b"].astype(np.float32)).cuda()

    @torch.no_grad()
    def en_ids(self, en: list[str]):
        enc = self.en_tok(en, padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    @torch.no_grad()
    def labels(self, en: list[str]):
        ids, am = self.en_ids(en)
        lab = ids.clone()
        lab[am == 0] = -100
        return lab

    @torch.no_grad()
    def native(self, vi: list[str]):
        ids, am = self.en_ids(vi)
        h = self.t5.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        return h, am

    @torch.no_grad()
    def mapped(self, vi: list[str], use_map: bool = True):
        b = self.tok.encode_vi(vi)
        L = max(8, ((int(b["vi_mask"].sum(1).max()) + 7) // 8) * 8)
        ids = torch.from_numpy(b["vi_ids"][:, :L]).cuda()
        am = torch.from_numpy(b["vi_mask"][:, :L]).cuda()
        off = torch.from_numpy(b["vi_off"][:, :L]).cuda()
        h = self.vit5.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        keep = ((off[..., 1] - off[..., 0]) > 0) & (am > 0)  # drop specials (span 0,0)
        if use_map:
            h = (h.float() @ self.W + self.b).to(DT)
        return h, keep.long()


def train(pairs: int, batch: int, log_every: int, lr: float, max_steps: int | None):
    torch.cuda.empty_cache()
    flan = AutoModelForSeq2SeqLM.from_pretrained(
        C.MODELS["flan-t5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True).to("cuda")
    n_dec = int(flan.config.num_decoder_layers)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=decoder_targets(n_dec), task_type="SEQ_2_SEQ_LM")
    if (C.DECODER_ADAPTER / "adapter_config.json").exists():
        from peft import PeftModel

        print(f"resuming adapter from {C.DECODER_ADAPTER}")
        model = PeftModel.from_pretrained(flan, str(C.DECODER_ADAPTER), is_trainable=True)
    else:
        model = get_peft_model(flan, cfg)
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    ch = Channels(model)
    C.DECODER_ADAPTER.mkdir(parents=True, exist_ok=True)
    log_path = C.RESULTS / "decoder_train.csv"
    log_path.write_text("step,pairs,cond,loss,sec\n", encoding="utf-8")

    print(f"vram after load: {torch.cuda.memory_allocated() / 2**20:.0f} MB", flush=True)
    chunk = None
    for _idx, vi, en in iter_slices("train", pairs):
        chunk = (vi, en)
        break
    vi, en = chunk
    print(f"training on {len(vi):,} pairs, batch {batch}, lr {lr}", flush=True)

    t0 = time.time()
    seen = step = 0
    buf = []
    rng = np.random.default_rng(1234)
    order = rng.permutation(len(vi))
    stop = False
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        bvi = [vi[i] for i in idx]
        ben = [en[i] for i in idx]
        lab_all = ch.labels(ben)
        half = max(1, len(idx) // 2)
        for cond, sel in (("native", slice(0, half)), ("mapped", slice(half, len(idx)))):
            if sel.stop - sel.start < 1:
                continue
            bv, lab = bvi[sel], lab_all[sel]
            h, am = ch.native(bv) if cond == "native" else ch.mapped(bv)
            out = model(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=am, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += sel.stop - sel.start
            buf.append((step, seen, cond, float(out.loss.item()), time.time() - t0))
            if len(buf) >= log_every:
                with log_path.open("a", encoding="utf-8") as f:
                    for r in buf:
                        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.1f}\n")
                nat = [r[3] for r in buf if r[2] == "native"]
                mpp = [r[3] for r in buf if r[2] == "mapped"]
                rate = seen / (time.time() - t0)
                print(f"  step {step} pairs {seen}/{pairs}  loss native {np.mean(nat):.4f}  mapped {np.mean(mpp):.4f}  "
                      f"{rate:.1f} pairs/s  eta {(pairs - seen) / max(rate, 1e-9) / 60:.1f} min", flush=True)
                buf = []
            if step % 500 == 0:
                model.save_pretrained(C.DECODER_ADAPTER)
                (C.DECODER_ADAPTER / "state.json").write_text(
                    json.dumps({"step": step, "pairs_seen": seen}), encoding="utf-8")
            if max_steps and step >= max_steps:
                stop = True
                break
        if stop:
            break

    if buf:
        with log_path.open("a", encoding="utf-8") as f:
            for r in buf:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.1f}\n")
    model.save_pretrained(C.DECODER_ADAPTER)
    meta = {"steps": step, "pairs_seen": seen, "wall_min": round((time.time() - t0) / 60, 1),
            "batch": batch, "lr": lr, "max_pairs": pairs,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "decoder only"},
            "channels": ["native", "mapped"], "encoder_frozen": True}
    (C.DECODER_ADAPTER / "done.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done: {step} steps, {seen} pairs, {(time.time() - t0) / 60:.1f} min -> {C.DECODER_ADAPTER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    train(a.pairs, a.batch, a.log_every, a.lr, a.max_steps)
