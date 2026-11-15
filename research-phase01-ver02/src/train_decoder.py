"""Translation decoder: one LoRA adapter per direction, reading *either* channel.

Two conditions, same pair, same target, same decoder weights:
    native  the target model's own frozen encoder reads the source text
    mapped  the source model's frozen encoder reads it; W maps every token into target space and
            the decoder's cross-attention reads that sequence instead

The encoder never trains (LoRA sits on decoder modules by exact name). That is deliberate: if the
encoder moved, the native states would drift out of the space W was fitted into and the two
conditions would stop being comparable.

Why this adapter exists at all: the copy adapter (§2) proves the decoder can render a vector, but
it is trained with T_out = T_in, which invites a length-copy shortcut. This adapter is trained on
real translation (T_out != T_in), so its BLEU is a translation measurement.

Usage
    python -m src.train_decoder --direction vi_en --pairs 80000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

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


def decoder_targets(n_layers: int = 6) -> list[str]:
    names = []
    for i in range(n_layers):
        b = f"decoder.block.{i}"
        names += [f"{b}.layer.0.SelfAttention.{p}" for p in ("q", "k", "v", "o")]
        names += [f"{b}.layer.1.EncDecAttention.{p}" for p in ("q", "k", "v", "o")]
        names += [f"{b}.layer.2.DenseReluDense.{p}" for p in ("wi_0", "wi_1", "wo")]
    return names


def direction_sides(direction: str):
    src, tgt = direction.split("_")
    return src, tgt


class Channels:
    """Builds (encoder_outputs, attention_mask) for each condition."""

    def __init__(self, peft_model, direction: str):
        self.dir = direction
        self.src, self.tgt = direction_sides(direction)
        self.tok = {s: load_tok(s) for s in (self.src, self.tgt)}
        self.model = peft_model
        self.decoder_model = peft_model.base_model.model if isinstance(peft_model, PeftModel) else peft_model
        self.src_model = T5ForConditionalGeneration.from_pretrained(
            str(C.MODELS[self.src]["path"]), dtype=DT, low_cpu_mem_usage=True).to("cuda").eval()
        self.src_model.requires_grad_(False)
        z = np.load(C.MAP[direction])
        self.W = torch.from_numpy(z["W"].astype(np.float32)).cuda()
        self.b = torch.from_numpy(z["b"].astype(np.float32)).cuda()

    @torch.no_grad()
    def _enc(self, tok, texts):
        e = tok(texts, padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
        return e["input_ids"].cuda(), e["attention_mask"].cuda()

    @torch.no_grad()
    def labels(self, texts):
        ids, am = self._enc(self.tok[self.tgt], texts)
        lab = ids.clone()
        lab[am == 0] = -100
        return lab

    @torch.no_grad()
    def native(self, texts):
        # the *target* model's own tokenizer: these ids go into the target model's encoder.
        # Using the source tokenizer here would index past the target embedding matrix
        # (ViT5 has 36 096 rows, t5-small has 32 128).
        ids, am = self._enc(self.tok[self.tgt], texts)
        h = self.decoder_model.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        return h, am

    @torch.no_grad()
    def mapped(self, texts, use_map: bool = True, mapper=None):
        # tokenise with offsets so specials (span 0,0) can be dropped
        b = self.tok[self.src].backend_tokenizer.encode_batch(texts, add_special_tokens=True)
        n = len(texts)
        L = min(C.MAX_LEN, max(len(e.ids) for e in b))
        ids = torch.zeros((n, L), dtype=torch.long)
        am = torch.zeros((n, L), dtype=torch.long)
        keep = torch.zeros((n, L), dtype=torch.bool)
        vmax = C.MODELS[self.src]["vocab"]
        for r, e in enumerate(b):
            m = min(len(e.ids), L)
            a = np.asarray(e.ids[:m], dtype=np.int64)
            a[a >= vmax] = 0
            ids[r, :m] = torch.from_numpy(a)
            am[r, :m] = 1
            o = np.asarray(e.offsets[:m], dtype=np.int64)
            keep[r, :m] = torch.from_numpy((o[:, 1] - o[:, 0]) > 0)
        ids, am, keep = ids.cuda(), am.cuda(), keep.cuda()
        h = self.src_model.encoder(input_ids=ids, attention_mask=am).last_hidden_state  # (B,T,D)
        if use_map:
            if mapper is not None:
                B, T, Dd = h.shape
                flat = h.reshape(-1, Dd).float().cpu().numpy()
                h = torch.from_numpy(mapper(flat)).to(h.device).reshape(B, T, Dd)
            else:
                h = h.float() @ self.W + self.b
        # keep the batch shape: the decoder's cross-attention needs (B,T,D), and specials are
        # removed by the attention mask rather than by flattening
        return h.to(DT), keep.long()


def train(direction: str, pairs: int, batch: int, log_every: int, lr: float):
    src, tgt = direction_sides(direction)
    out = C.ADAPTERS[f"trans_{tgt}"]
    out.mkdir(parents=True, exist_ok=True)
    base = T5ForConditionalGeneration.from_pretrained(str(C.MODELS[tgt]["path"]), dtype=DT,
                                                      low_cpu_mem_usage=True).to("cuda")
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=decoder_targets(int(base.config.num_layers)), task_type="SEQ_2_SEQ_LM")
    if (out / "adapter_config.json").exists():
        print(f"resuming adapter from {out}")
        model = PeftModel.from_pretrained(base, str(out), is_trainable=True)
    else:
        model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    ch = Channels(model, direction)

    chosen = json.loads((C.DATA / "chosen_shard.json").read_text(encoding="utf-8"))["file"]
    d = load_chosen(("vi", "en", "split"))
    idx = [i for i, s in enumerate(d["split"]) if s == "train"]
    rng = np.random.default_rng(1234)
    pick = rng.choice(len(idx), size=min(pairs, len(idx)), replace=False)
    sents = [(d[src][idx[j]], d[tgt][idx[j]]) for j in pick]
    print(f"{direction}: {len(sents):,} train pairs from {chosen}, batch {batch}, lr {lr}", flush=True)

    log_path = C.RESULTS / f"decoder_train_{direction}.csv"
    log_path.write_text("step,pairs,cond,loss,sec\n", encoding="utf-8")
    t0, step, seen, buf = time.time(), 0, 0, []
    order = rng.permutation(len(sents))
    for s in range(0, len(order), batch):
        chunk = [sents[i] for i in order[s:s + batch]]
        bsrc = [c[0] for c in chunk]
        btgt = [c[1] for c in chunk]
        lab_all = ch.labels(btgt)
        half = max(1, len(chunk) // 2)
        for cond, sel in (("native", slice(0, half)), ("mapped", slice(half, len(chunk)))):
            if sel.stop - sel.start < 1:
                continue
            bv, lab = bsrc[sel], lab_all[sel]
            h, am = ch.native(bv) if cond == "native" else ch.mapped(bv)
            o = model(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=am, labels=lab)
            o.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += sel.stop - sel.start
            buf.append((step, seen, cond, float(o.loss.item()), time.time() - t0))
            if len(buf) >= log_every:
                with log_path.open("a", encoding="utf-8") as f:
                    for r in buf:
                        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.1f}\n")
                nat = [r[3] for r in buf if r[2] == "native"]
                mpp = [r[3] for r in buf if r[2] == "mapped"]
                rate = seen / (time.time() - t0)
                print(f"  step {step} pairs {seen}/{len(sents)}  loss native {np.mean(nat):.4f}  "
                      f"mapped {np.mean(mpp):.4f}  {rate:.1f} pairs/s  "
                      f"eta {(len(sents) - seen) / max(rate, 1e-9) / 60:.1f} min", flush=True)
                buf = []
            if step % 400 == 0:
                model.save_pretrained(out)
        if seen >= len(sents):
            break
    if buf:
        with log_path.open("a", encoding="utf-8") as f:
            for r in buf:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.1f}\n")
    model.save_pretrained(out)
    (out / "done.json").write_text(json.dumps(
        {"direction": direction, "steps": step, "pairs_seen": seen, "batch": batch, "lr": lr,
         "wall_min": round((time.time() - t0) / 60, 1), "lora_r": 16,
         "targets": "decoder only", "channels": ["native", "mapped"], "encoder_frozen": True,
         "shard": chosen}, indent=2), encoding="utf-8")
    print(f"done: {step} steps, {seen} pairs, {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["vi_en", "en_vi"], required=True)
    ap.add_argument("--pairs", type=int, default=80_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()
    train(a.direction, a.pairs, a.batch, a.log_every, a.lr)
