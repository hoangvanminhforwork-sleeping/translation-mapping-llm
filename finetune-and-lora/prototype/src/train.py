"""LoRA fine-tuning on the copy task. Resumable, one file per run.

Task:      "Nói: {s}" -> {s}   (vi)     "Say: {s}" -> {s}   (en)
Objective: the model must reproduce the sentence exactly. Nothing else changes.

Why LoRA before full fine-tuning: it is ~1% of the parameters, ~1 MB on disk, and
if the copy circuit can be reached at all, LoRA reaches it. A negative LoRA result
is therefore a strong signal that the task needs more than an adapter.
"""

from __future__ import annotations

import csv
import json
import math
import time

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base(model_key: str, with_lora: bool = True):
    spec = C.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], use_fast=True)
    dtype = DT[C.DTYPE]
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_id"], dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_id"], torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device())
    if with_lora:
        cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=C.LORA["r"],
            lora_alpha=C.LORA["alpha"],
            lora_dropout=C.LORA["dropout"],
            target_modules=C.LORA["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, cfg)
    return tok, model


def attach_adapter(model_key: str):
    """Base model + trained adapter, for evaluation."""
    tok, model = load_base(model_key, with_lora=False)
    path = C.ADAPTERS / model_key
    if not (path / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {path}; train first")
    model = PeftModel.from_pretrained(model, str(path))
    model.eval()
    return tok, model


def _pretokenize(tok, rows, lang: str, prefix: str) -> list[tuple[list[int], list[int]]]:
    out = []
    for r in rows:
        text = r[lang]
        src = tok(prefix + text, max_length=C.MAX_LEN, truncation=True)["input_ids"]
        tgt = tok(text, max_length=C.MAX_LEN, truncation=True)["input_ids"]
        out.append((src, tgt))
    return out


def _collate(batch, pad_id: int):
    max_in = max(len(a) for a, _ in batch)
    max_out = max(len(b) for _, b in batch)
    input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels = torch.full((len(batch), max_out), -100, dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        input_ids[i, : len(a)] = torch.tensor(a, dtype=torch.long)
        attn[i, : len(a)] = 1
        labels[i, : len(b)] = torch.tensor(b, dtype=torch.long)
    return input_ids.to(device()), attn.to(device()), labels.to(device())


def train(model_key: str, force: bool = False) -> dict:
    spec = C.MODELS[model_key]
    lang = spec["lang"]
    log_path = C.RESULTS / f"train_{model_key}.csv"
    adapter_dir = C.ADAPTERS / model_key
    progress_path = adapter_dir / "progress.json"
    done_path = adapter_dir / "done.json"

    if done_path.exists() and not force:
        print(f"{model_key}: already trained ({json.loads(done_path.read_text())})")
        return json.loads(done_path.read_text())
    if force:
        for p in (log_path, progress_path, done_path):
            if p.exists():
                p.unlink()

    rows = [json.loads(l) for l in (C.DATA / f"train_copy_{C.N_TRAIN}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tok, model = load_base(model_key)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{model_key}: trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)", flush=True)

    examples = _pretokenize(tok, rows, lang, spec["prefix"])
    order = list(range(len(examples)))  # already seed-shuffled when the file was built
    bs = spec["batch"]
    steps_per_epoch = math.ceil(len(examples) / bs)
    total_steps = steps_per_epoch

    start_step = 0
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        start_step = prog["step"]
        # load_adapter, not PeftModel.from_pretrained: model is already a PeftModel here
        model.load_adapter(str(adapter_dir), adapter_name="default", is_trainable=True)
        print(f"{model_key}: resuming from step {start_step}/{total_steps}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < C.WARMUP:
            return C.LR * (step + 1) / C.WARMUP
        prog = (step - C.WARMUP) / max(1, total_steps - C.WARMUP)
        return C.LR * 0.5 * (1 + math.cos(math.pi * prog))

    write_header = not log_path.exists() or start_step == 0
    t0 = time.time()
    losses = []
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["step", "loss", "lr", "elapsed_s", "sent_per_s", "vram_mb"])
        for step in range(start_step, total_steps):
            batch_idx = order[step * bs : (step + 1) * bs]
            if not batch_idx:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            input_ids, attn, labels = _collate([examples[i] for i in batch_idx], tok.pad_token_id)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, C.GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach()))
            if step % C.LOG_EVERY == 0 or step == total_steps - 1:
                elapsed = time.time() - t0
                done = step - start_step + 1
                w.writerow([step, round(float(loss.detach()), 4), f"{lr_at(step):.2e}",
                            round(elapsed, 1), round(done * bs / max(1e-9, elapsed), 2),
                            round(torch.cuda.max_memory_allocated() / 1024**2, 0)])
                fh.flush()
                print(f"  step {step:5d}/{total_steps}  loss {float(loss.detach()):.3f}  "
                      f"{done*bs/max(1e-9, elapsed):5.1f} sent/s  vram {torch.cuda.max_memory_allocated()/1024**2:.0f} MB", flush=True)

            if (step + 1) % C.SAVE_EVERY == 0 or step == total_steps - 1:
                model.save_pretrained(str(adapter_dir))
                progress_path.write_text(json.dumps({"step": step + 1}))

    duration = time.time() - t0
    summary = {
        "model": model_key,
        "train_rows": len(rows),
        "steps": total_steps - start_step,
        "seconds": round(duration, 1),
        "sent_per_s": round((total_steps - start_step) * bs / max(1e-9, duration), 2),
        "final_loss_100": round(sum(losses[-100:]) / len(losses[-100:]), 4),
        "trainable_params": trainable,
        "total_params": total,
        "adapter_mb": round(sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()) / 1024**2, 2),
    }
    done_path.write_text(json.dumps(summary, indent=2))
    print(f"{model_key}: done in {duration/60:.1f} min | last-100 loss {summary['final_loss_100']} | adapter {summary['adapter_mb']} MB")
    return summary