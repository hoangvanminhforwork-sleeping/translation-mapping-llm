"""LABELLED CONTROL — not part of the novel claim.

Question it answers: how much of the gap between "frozen 77M model" and "a model that has seen
MTet" is ordinary domain adaptation? Without this number nobody can say whether the reference-free
reranking arms captured something real or merely a fraction of what 4 M LoRA parameters would buy.

Protocol: LoRA (r=16, q/k/v/o, both stacks) on A = opus-mt-vi-en, 150 k train sentences, 1 epoch,
fp16, batch 24. Nothing is tuned; it is a single pre-registered configuration.

Usage
    python -m src.lora_control --build-data          # writes data/lora_train.parquet (150k)
    python -m src.lora_control --train               # writes models/lora_vi_en
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
N_TRAIN = 150_000
OUT_ADAPTER = C.MODELS / "lora_vi_en"
DATA_FILE = C.DATA / "lora_train.parquet"


def build_data() -> None:
    if DATA_FILE.exists():
        print(f"{DATA_FILE.name} exists")
        return
    pf = pq.ParquetFile(C.SPLIT)
    keep_vi, keep_en = [], []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["vi", "en", "split"])
        d = t.to_pylist()
        for r in d:
            if r["split"] == "train":
                keep_vi.append(r["vi"])
                keep_en.append(r["en"])
        del t, d
        if len(keep_vi) >= N_TRAIN:
            break
    tbl = pa.table({"vi": keep_vi[:N_TRAIN], "en": keep_en[:N_TRAIN]})
    pq.write_table(tbl, DATA_FILE, compression="zstd")
    print(f"wrote {DATA_FILE.name} ({len(keep_vi[:N_TRAIN])} rows)")


class Pairs(Dataset):
    def __init__(self, tok, path):
        t = pq.read_table(path)
        self.vi = t.column("vi").to_pylist()
        self.en = t.column("en").to_pylist()
        self.tok = tok

    def __len__(self):
        return len(self.vi)

    def __getitem__(self, i):
        return self.vi[i], self.en[i]


def collate(tok):
    def fn(batch):
        src = tok([b[0] for b in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=96)
        tgt = tok(text_target=[b[1] for b in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=96)
        labels = tgt["input_ids"].masked_fill(tgt["attention_mask"] == 0, -100)
        return src["input_ids"], src["attention_mask"], labels
    return fn


def train(epochs: float = 1.0, batch: int = 24, lr: float = 3e-4, max_steps: int = 0) -> None:
    from peft import LoraConfig, get_peft_model

    if (OUT_ADAPTER / "adapter_config.json").exists():
        print(f"{OUT_ADAPTER} exists")
        return
    tok = AutoTokenizer.from_pretrained(C.A_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(C.A_ID, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model = model.to("cuda").train()

    ds = Pairs(tok, DATA_FILE)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                    collate_fn=collate(tok), drop_last=True)
    steps = int(len(dl) * epochs)
    if max_steps:
        steps = min(steps, max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    step = 0
    while step < steps:
        for ids, am, labels in dl:
            ids, am, labels = ids.to("cuda"), am.to("cuda"), labels.to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=am, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  step {step}/{steps}  loss {loss.item():.4f}  "
                      f"{step * batch / el:.1f} pairs/s  eta {(steps - step) * batch / (step * batch / el) / 60:.0f} min",
                      flush=True)
            if step >= steps:
                break
    OUT_ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_ADAPTER))
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (C.RESULTS / "lora_control.json").write_text(json.dumps(
        {"adapter": str(OUT_ADAPTER), "trainable_params": int(n), "pairs": int(N_TRAIN),
         "epochs": epochs, "batch": batch, "lr": lr, "steps": steps,
         "minutes": (time.time() - t0) / 60}, indent=2), encoding="utf-8")
    print(f"saved {OUT_ADAPTER} ({n} trainable params) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-data", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    if a.build_data:
        build_data()
    if a.train:
        train(a.epochs, a.batch, a.lr, a.max_steps)
