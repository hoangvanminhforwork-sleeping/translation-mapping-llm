"""Model loading with an explicit device/dtype report.

res-33: a run that "used the GPU" is a false claim unless the device string is
printed. Every load() call returns that string, and callers print it.
"""

from __future__ import annotations

import gc
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def load(model_key: str, dtype: str | None = None):
    """Load tokenizer + model, print device/dtype/params, return (tok, model, info)."""
    spec = C.MODELS[model_key]
    dtype_name = dtype or C.DTYPE
    torch_dtype = DTYPES[dtype_name]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], use_fast=True)
    if not tok.is_fast:
        raise RuntimeError(f"{model_key}: fast tokenizer unavailable, offset mapping would be unreliable")

    kwargs = dict(low_cpu_mem_usage=True)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_id"], dtype=torch_dtype, **kwargs)
    except TypeError:  # transformers < 4.56
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["hf_id"], torch_dtype=torch_dtype, **kwargs)
    model.to(device()).eval()

    cfg = model.config
    info = {
        "model_key": model_key,
        "hf_id": spec["hf_id"],
        "d_model": cfg.d_model,
        "layers_enc": cfg.num_layers,
        "layers_dec": cfg.num_decoder_layers,
        "heads": cfg.num_heads,
        "vocab": cfg.vocab_size,
        "params": sum(p.numel() for p in model.parameters()),
        "dtype": dtype_name,
        "device": device(),
        "load_s": round(time.time() - t0, 2),
        "vram_mb": round(peak_vram_mb(), 1),
        "pad_id": tok.pad_token_id,
        "eos_id": tok.eos_token_id,
    }
    print(
        f"loaded {model_key:<14} {info['params']/1e6:7.1f}M params  d_model={info['d_model']} "
        f"enc/dec={info['layers_enc']}/{info['layers_dec']}  device={info['device']} dtype={info['dtype']} "
        f"vram={info['vram_mb']:.0f} MB  {info['load_s']}s"
    )
    return tok, model, info


def free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
