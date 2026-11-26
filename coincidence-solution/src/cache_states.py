"""Phase 0 — cache every hidden state the night needs, once.

Why cache: the decoders are frozen and the source encoder is frozen, so `E_vi`, `E_nat`, `E_ref`
are constants. Caching turns "fit a map on 20 k sentences" from an encoder-bound job into a
decoder-only job (minutes), and makes every later script restartable without re-encoding.

Layout (fixed width MAX_LEN=64 so the arrays are memmaps, never fully in RAM):

    agent-work/cache/{set}/E_vi.npy    (N,64,512) fp16   source encoder, vi token positions
    agent-work/cache/{set}/m_vi.npy    (N,64)     int8   `keep` mask (specials dropped)
    agent-work/cache/{set}/E_nat.npy   (N,64,512) fp16   t5-small encoder reading the VI source
    agent-work/cache/{set}/m_nat.npy   (N,64)     int8
    agent-work/cache/{set}/E_ref.npy   (N,64,512) fp16   t5-small encoder reading the EN reference
    agent-work/cache/{set}/m_ref.npy   (N,64)     int8
    agent-work/cache/{set}/lab.npy     (N,64)     int32  decoder labels (-100 = pad)
    agent-work/cache/{set}/texts.json                    vi/en strings + row ids

Idempotent: an existing file is never rewritten; delete a file to force it.

Usage
    python -m src.cache_states                 # test 1k + val 1k + fit 20k
    python -m src.cache_states --sets test     # one set
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from . import cc

sys.stdout.reconfigure(encoding="utf-8")

SETS = {"test": (cc.N_TEST, 0), "test2": (1_000, 1_000), "val": (cc.N_VAL, 0),
        "fit": (cc.N_TRAIN_FIT, 0)}


def paths(set_name: str):
    d = cc.CACHE / set_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_fixed(name: str, arr: np.ndarray, dtype, width: int, set_name: str, batch: int = 128):
    """Write `arr` (N, L, ...) into a fixed-width memmap, in chunks. Returns the memmap."""
    d = paths(set_name)
    p = d / f"{name}.npy"
    n = arr.shape[0]
    shape = (n, width) + arr.shape[2:]
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)
    for s in range(0, n, batch):
        chunk = arr[s:s + batch]
        mm[s:s + batch, :chunk.shape[1]] = chunk
    mm.flush()
    del mm
    return p


def have(set_name: str) -> bool:
    d = paths(set_name)
    return all((d / f"{k}.npy").exists() for k in
               ("E_vi", "m_vi", "E_nat", "m_nat", "E_ref", "m_ref", "lab")) and \
        (d / "texts.json").exists()


def build(set_name: str, vi_enc, en_enc, batch: int):
    if have(set_name):
        print(f"{set_name}: cached, skip")
        return
    t0 = time.time()
    n, offset = SETS[set_name]
    split = "test" if set_name.startswith("test") else set_name
    pairs = cc.load_fit_pairs(n) if set_name == "fit" else cc.load_pairs(split, n, offset)
    vi, en = pairs["vi"], pairs["en"]
    print(f"{set_name}: {len(vi):,} pairs", flush=True)

    E_vi, m_vi = cc.encode_vi_mapped(vi_enc, vi, batch)
    print(f"  E_vi  {E_vi.shape}  {time.time() - t0:.0f}s", flush=True)
    E_nat, m_nat = cc.encode("en", en_enc, vi, batch)          # target tokenizer, VI text
    print(f"  E_nat {E_nat.shape}  {time.time() - t0:.0f}s", flush=True)
    E_ref, m_ref = cc.encode("en", en_enc, en, batch)          # target tokenizer, EN reference
    print(f"  E_ref {E_ref.shape}  {time.time() - t0:.0f}s", flush=True)

    lab = np.full((len(en), cc.MAX_LEN), -100, dtype=np.int32)
    for s in range(0, len(en), batch):
        L = cc.labels(en[s:s + batch])
        lab[s:s + L.shape[0], :L.shape[1]] = L.cpu().numpy()

    write_fixed("E_vi", E_vi, np.float16, cc.MAX_LEN, set_name)
    write_fixed("m_vi", m_vi, np.int8, cc.MAX_LEN, set_name)
    write_fixed("E_nat", E_nat, np.float16, cc.MAX_LEN, set_name)
    write_fixed("m_nat", m_nat, np.int8, cc.MAX_LEN, set_name)
    write_fixed("E_ref", E_ref, np.float16, cc.MAX_LEN, set_name)
    write_fixed("m_ref", m_ref, np.int8, cc.MAX_LEN, set_name)
    write_fixed("lab", lab, np.int32, cc.MAX_LEN, set_name)

    (paths(set_name) / "texts.json").write_text(json.dumps(
        {"vi": vi, "en": en, "row": pairs["row"], "n": len(vi)}, ensure_ascii=False),
        encoding="utf-8")
    print(f"{set_name}: done in {time.time() - t0:.0f}s", flush=True)


def main(sets: list[str], batch: int):
    vi_enc = cc.load_encoder("vi", dtype=cc.DT)
    en_enc = cc.load_encoder("en", dtype=cc.DT)
    for s in sets:
        build(s, vi_enc, en_enc, batch)
    print("cache: complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test,val,fit")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    main([s for s in a.sets.split(",") if s], a.batch)
