"""Per-token ridge fit with streaming sufficient statistics.

Why streaming: 3.46M pairs x 2 sides x ~25 tokens x 768 dims is ~130 GB of float32 activation.
Nobody can hold that, and the GPU has 4 GB. But ridge only needs three second-moment matrices

    A = X^T X,  B = X^T Y,  C = Y^T Y    (+ sums, for the intercept)

and they are additive over shards. Each shard of 200k pairs is encoded, folded into (A, B, C) and
thrown away; `fit` sums whatever shards exist. A run that dies at shard 12 keeps shards 0-11, and a
later run can add shards and get a better map without recomputing anything.

Modes
    python -m src.fit_tok shards [--shards 0,1,2]   # encode + accumulate (GPU)
    python -m src.fit_tok fit                       # solve W, sweep lambda on val, save npz
    python -m src.fit_tok smoke [--n 2000]          # throughput + alignment stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSeq2SeqLM

from . import config as C
from .tokenbatch import AlignedBatcher
from .tokutil import PairTokenizer

sys.stdout.reconfigure(encoding="utf-8")
DT = torch.bfloat16
D = 768
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


# ---------------------------------------------------------------- corpus access
def iter_slices(split: str, chunk_pairs: int):
    """Yield (index, vi, en) chunks of `chunk_pairs` rows of one split, in one pass over the parquet."""
    pf = pq.ParquetFile(C.PAIRS)
    buf_vi: list[str] = []
    buf_en: list[str] = []
    idx = 0
    for rb in pf.iter_batches(batch_size=8192, columns=["vi", "en", "split"]):
        sp = rb.column("split").to_pylist()
        vi = rb.column("vi").to_pylist()
        en = rb.column("en").to_pylist()
        for i, s in enumerate(sp):
            if s != split:
                continue
            buf_vi.append(vi[i])
            buf_en.append(en[i])
            if len(buf_vi) >= chunk_pairs:
                yield idx, buf_vi, buf_en
                idx += 1
                buf_vi, buf_en = [], []
    if buf_vi:
        yield idx, buf_vi, buf_en


def count_split(split: str) -> int:
    t = pq.read_table(C.PAIRS, columns=["split"])
    return int(np.sum(np.asarray(t.column("split").to_pylist()) == split))


# ---------------------------------------------------------------- models
class Encoders:
    def __init__(self):
        self.tok = PairTokenizer(C.MAX_LEN)
        self.vit5 = AutoModelForSeq2SeqLM.from_pretrained(
            C.MODELS["vit5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True).to("cuda").eval()
        self.flan = AutoModelForSeq2SeqLM.from_pretrained(
            C.MODELS["flan-t5-base"]["hf_id"], dtype=DT, low_cpu_mem_usage=True).to("cuda").eval()
        self.vit5.requires_grad_(False)
        self.flan.requires_grad_(False)


# ---------------------------------------------------------------- accumulation
class Moments:
    """A, B, C live on the GPU in float64; only 3 x 4.7 MB cross the bus per shard."""

    def __init__(self, device: str = "cuda"):
        z = lambda: torch.zeros((D, D), dtype=torch.float64, device=device)  # noqa: E731
        self.A, self.B, self.C = z(), z(), z()
        self.sx = torch.zeros(D, dtype=torch.float64, device=device)
        self.sy = torch.zeros(D, dtype=torch.float64, device=device)
        self.n = 0
        self.pairs = 0

    def add(self, X: torch.Tensor, Y: torch.Tensor):
        self.A += (X.T @ X).double()
        self.B += (X.T @ Y).double()
        self.C += (Y.T @ Y).double()
        self.sx += X.sum(0).double()
        self.sy += Y.sum(0).double()
        self.n += X.shape[0]

    def save(self, path, meta: dict):
        np.savez_compressed(path, A=self.A.cpu().numpy(), B=self.B.cpu().numpy(),
                            C=self.C.cpu().numpy(), sx=self.sx.cpu().numpy(),
                            sy=self.sy.cpu().numpy(), n=np.array(self.n),
                            pairs=np.array(self.pairs), meta=json.dumps(meta))

    @classmethod
    def load(cls, path) -> "Moments":
        z = np.load(path, allow_pickle=False)
        m = cls(device="cpu")
        m.A = torch.from_numpy(z["A"])
        m.B = torch.from_numpy(z["B"])
        m.C = torch.from_numpy(z["C"])
        m.sx = torch.from_numpy(z["sx"])
        m.sy = torch.from_numpy(z["sy"])
        m.n = int(z["n"])
        if "pairs" in z:
            m.pairs = int(z["pairs"])
        elif "meta" in z:  # shards written before `pairs` was a separate array
            try:
                m.pairs = int(json.loads(str(z["meta"]))["pairs"])
            except Exception:  # noqa: BLE001
                m.pairs = 0
        return m

    def __iadd__(self, o: "Moments"):
        self.A += o.A.to(self.A.device)
        self.B += o.B.to(self.B.device)
        self.C += o.C.to(self.C.device)
        self.sx += o.sx.to(self.sx.device)
        self.sy += o.sy.to(self.sy.device)
        self.n += o.n
        self.pairs += o.pairs
        return self


def accumulate(enc: Encoders, vi: list[str], en: list[str], batch: int, label: str) -> Moments:
    m = Moments()
    t0 = time.time()
    total = len(vi)
    done = 0
    for s, n, X, Y, _lens in AlignedBatcher(enc, batch).batches(vi, en):
        m.add(X, Y)
        m.pairs += n
        done = s + n
        if done % (batch * 10) < batch or done == total:
            el = max(time.time() - t0, 1e-9)
            print(f"  {label} {done}/{total}  {done / el:6.1f} seq/s  tok/pair {m.n / max(m.pairs, 1):5.1f}  "
                  f"eta {(total - done) / (done / el) / 60:5.1f} min", flush=True)
    return m


# ---------------------------------------------------------------- modes
def mode_shards(which: list[int], batch: int):
    n_train = count_split("train")
    n_shards = (n_train + C.SHARD - 1) // C.SHARD
    C.STATS.mkdir(parents=True, exist_ok=True)
    todo = set(which) if which else set(range(n_shards))
    print(f"train {n_train:,} pairs -> {n_shards} shards of {C.SHARD:,}; target {sorted(todo)}")
    enc = Encoders()
    for idx, vi, en in iter_slices("train", C.SHARD):
        out = C.STATS / f"train-{idx:04d}.npz"
        if idx not in todo or out.exists():
            print(f"shard {idx}: skip ({len(vi):,} pairs)", flush=True)
            continue
        t_shard = time.time()
        m = accumulate(enc, vi, en, batch, label=f"shard{idx}")
        m.save(out, {"split": "train", "shard": idx, "pairs": m.pairs})
        print(f"shard {idx}: {m.pairs:,} pairs, {m.n:,} token pairs, "
              f"{(time.time() - t_shard) / 60:.1f} min -> {out.name}", flush=True)
        del m
        torch.cuda.empty_cache()

    val = C.STATS / "val.npz"
    if not val.exists():
        for _idx, vi, en in iter_slices("val", C.N_VAL):
            print(f"val: {len(vi)} pairs")
            m = accumulate(enc, vi, en, batch, label="val")
            m.save(val, {"split": "val", "pairs": m.pairs})
            print(f"val: {m.n:,} token pairs -> {val.name}")


def _solve(m: Moments, lam: float):
    n = float(m.n)
    mx, my = m.sx.cpu() / n, m.sy.cpu() / n
    Ac = (m.A.cpu() - n * torch.outer(mx, mx)).numpy()
    Bc = (m.B.cpu() - n * torch.outer(mx, my)).numpy()
    W = np.linalg.solve(Ac + lam * np.eye(D), Bc)
    b = (my - mx @ torch.from_numpy(W)).numpy()
    return W, b


def swapped(m: Moments) -> Moments:
    """The same statistics read in the other direction: X'=Y, Y'=X.

    A ridge fit only needs (XᵀX, XᵀY, YᵀY, Σx, Σy), so the en->vi map costs nothing extra: swap them.
    """
    s = Moments(device="cpu")
    s.A, s.B, s.C = m.C, m.B.T.clone(), m.A
    s.sx, s.sy, s.n = m.sy, m.sx, m.n
    s.pairs = m.pairs
    return s


def _mse(m: Moments, W, b):
    """MSE of (W, b) on the moments of m, without touching activations."""
    n = float(m.n)
    mx, my = m.sx.cpu() / n, m.sy.cpu() / n
    Wt = torch.from_numpy(W)
    bt = torch.from_numpy(b)
    Ac = m.A.cpu() - n * torch.outer(mx, mx)
    Bc = m.B.cpu() - n * torch.outer(mx, my)
    Cc = m.C.cpu() - n * torch.outer(my, my)
    tr = torch.trace(Wt.T @ Ac @ Wt) - 2 * torch.trace(Wt.T @ Bc) + torch.trace(Cc)
    res = bt + mx @ Wt - my
    return float((tr / n).item() + (res @ res).item())


def mode_fit():
    shards = sorted(C.STATS.glob("train-*.npz"))
    if not shards:
        raise SystemExit("no train shards; run `shards` first")
    tot = Moments(device="cpu")
    for p in shards:
        tot += Moments.load(p)
    val = Moments.load(C.STATS / "val.npz")
    print(f"train: {len(shards)} shards, {tot.pairs:,} pairs, {tot.n:,} token pairs")
    print(f"val:   {val.pairs:,} pairs, {val.n:,} token pairs")

    rows_all = []
    for direction, (tr, va, out_path) in {
        "vi_en": (tot, val, C.MAP_VI_EN),
        "en_vi": (swapped(tot), swapped(val), C.DATA / "map_en_vi.npz"),
    }.items():
        rows, best = [], None
        for lam in LAMBDAS:
            W, b = _solve(tr, lam)
            mse_tr, mse_va = _mse(tr, W, b), _mse(va, W, b)
            rows.append((lam, mse_tr, mse_va))
            print(f"  {direction} lambda {lam:>8g}  mse_train {mse_tr:.6f}  mse_val {mse_va:.6f}")
            if best is None or mse_va < best[1]:
                best = (lam, mse_va, W, b)
        lam, mse_va, W, b = best
        base = rows[0][2]
        gain = (base - mse_va) / base
        if gain < 1e-5:
            # the sweep is flat to within rounding: prefer the *least* regularised solution rather than
            # reporting which lambda happened to win by 1e-6 of float noise
            lam = LAMBDAS[0]
            W, b = _solve(tr, lam)
            mse_va = _mse(va, W, b)
            print(f"{direction}: lambda sweep flat (best gain {gain:.2e} < 1e-5) -> lambda=0, "
                  f"i.e. no regularisation is needed at this sample size")
        else:
            print(f"{direction}: chosen lambda={lam} (val mse {mse_va:.6f}, gain {gain:.2e} over lambda=0)")
        np.savez_compressed(out_path, W=W.astype(np.float32), b=b.astype(np.float32), lam=lam,
                            n_pairs=tr.pairs, n_tokens=tr.n, shards=len(shards), direction=direction)
        rows_all += [(direction,) + r for r in rows]
        print(f"{direction}: saved {out_path.name}")

    (C.RESULTS / "ridge_sweep.csv").write_text(
        "direction,lambda,mse_train,mse_val\n"
        + "\n".join(f"{d},{a},{b_:.8f},{c:.8f}" for d, a, b_, c in rows_all) + "\n", encoding="utf-8")


def mode_smoke(n: int = 2000, batch: int = 256):
    enc = Encoders()
    rows = None
    for _idx, vi, en in iter_slices("train", n):
        rows = (vi, en)
        break
    t0 = time.time()
    m = accumulate(enc, rows[0], rows[1], batch, label="smoke")
    el = time.time() - t0
    print(f"smoke: {len(rows[0])} pairs in {el:.1f}s = {len(rows[0]) / el:.1f} seq/s, "
          f"{m.n / max(m.pairs, 1):.1f} tok/pair")
    print(f"projected for 3.45M pairs: {3456741 / (len(rows[0]) / el) / 3600:.2f} h")
    print(f"vram {torch.cuda.memory_allocated() / 2**20:.0f} MB allocated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["shards", "fit", "smoke"])
    ap.add_argument("--shards", default="")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    which = [int(x) for x in a.shards.split(",") if x.strip()] if a.shards else []
    if a.mode == "shards":
        mode_shards(which, a.batch)
    elif a.mode == "fit":
        mode_fit()
    else:
        mode_smoke(a.n, a.batch)
