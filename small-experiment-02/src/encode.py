"""Frozen encoders -> cached float32 vectors.

Inherited from small-experiment/src/encode.py (same models, same pooling, same
max_tokens).  Difference: results are cached as .npy instead of pushed to MySQL,
because MySQL lives in Docker and costs 1-2 GB RAM we do not have.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import config as C


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(name: str, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(C.HF_HOME))
    model = AutoModel.from_pretrained(name, cache_dir=str(C.HF_HOME))
    model.to(device()).eval()
    model.requires_grad_(False)
    if dtype == torch.float16 and device() == "cuda":
        model.half()
    return tok, model


def vector_path(lang: str, pooling: str):
    return C.ARTIFACTS / f"vec_{lang}_{pooling}.npy"


@torch.no_grad()
def encode(texts, tok, model, pooling: str, batch: int = C.ENCODE_BATCH) -> np.ndarray:
    """[n, 768] float32 for one pooling."""
    dev = device()
    out = []
    for start in range(0, len(texts), batch):
        chunk = list(texts[start:start + batch])
        enc = tok(chunk, padding=True, truncation=True, max_length=C.MAX_TOKENS,
                  return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)

        if pooling == "mean":
            v = (hidden * mask).sum(1) / mask.sum(1)
        elif pooling == "cls":
            v = hidden[:, 0]
        else:                                    # last non-pad token ([SEP] / </s>)
            lengths = enc["attention_mask"].sum(1).clamp(min=1) - 1
            v = hidden[torch.arange(hidden.size(0), device=dev), lengths]
        out.append(v.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def cached(texts, lang: str, pooling: str, tok, model, force: bool = False) -> np.ndarray:
    path = vector_path(lang, pooling)
    if path.exists() and not force:
        vec = np.load(path)
        assert vec.shape == (len(texts), C.DIM), (vec.shape, len(texts))
        return vec
    vec = encode(texts, tok, model, pooling)
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    return vec


def free(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
