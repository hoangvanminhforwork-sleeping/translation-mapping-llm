"""Frozen encoders -> three sentence representations each.

We deliberately do the pooling ourselves (mean / CLS / last non-pad token)
instead of calling sentence-transformers, so that both languages are treated
identically and 'representation' is an explicit experimental variable.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import config as C


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(candidates, expected_dim: int = C.DIM):
    """Return (name, tokenizer, model).  First candidate that loads at 768 wins."""
    errors = []
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, cache_dir=str(C.HF_HOME))
            model = AutoModel.from_pretrained(name, cache_dir=str(C.HF_HOME))
            dim = int(model.config.hidden_size)
            if dim != expected_dim:
                raise ValueError(f"hidden_size={dim}, expected {expected_dim}")
            model.to(device()).eval()
            return name, tok, model
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no candidate encoder could be loaded:\n  " + "\n  ".join(errors))


@torch.no_grad()
def encode(texts, tok, model, batch: int = C.ENCODE_BATCH,
           max_tokens: int = C.MAX_TOKENS) -> dict:
    """dict pooling -> float32 array [n, 768]."""
    dev = device()
    amp = torch.float16 if dev == "cuda" else torch.float32
    model.to(dev)
    if dev == "cuda":
        model.half()

    out = {p: [] for p in C.POOLINGS}
    for start in range(0, len(texts), batch):
        chunk = list(texts[start:start + batch])
        enc = tok(chunk, padding=True, truncation=True, max_length=max_tokens,
                  return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state                     # [B,T,D]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B,T,1]

        out["mean"].append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
        out["cls"].append(hidden[:, 0].float().cpu().numpy())
        lengths = enc["attention_mask"].sum(1).clamp(min=1) - 1
        out["last"].append(hidden[torch.arange(hidden.size(0)), lengths].float().cpu().numpy())

    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def free(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_report() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    return (f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
