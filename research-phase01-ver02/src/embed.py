"""Moment accumulation + sentence embeddings in one streaming pass over one shard.

Why streaming: 2.7M pairs x 2 sides x ~25 tokens x 512 dims is ~280 GB of float32 activation.
Ridge only needs three second-moment matrices (A = X^T X, B = X^T Y, C = Y^T Y, plus sums for
the intercept), and they are additive over shards. Each shard is encoded, folded into
(A, B, C), and thrown away; `fit` sums whatever shards exist, so a run that dies keeps its work.

Modes
    python -m src.embed smoke  [--n 4000]        # throughput + alignment stats
    python -m src.embed moments --file <shard>   # per-token moments for one shard, resumable
    python -m src.embed sentence                 # (N,512) fp16 source embeddings for the coreset
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import T5ForConditionalGeneration

from . import config as C
from .tokenbatch import AlignedBatcher, SideBatcher
from .tokutil import PairTokenizer

sys.stdout.reconfigure(encoding="utf-8")
DT = torch.bfloat16
D = 512
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


class Encoders:
    """Both frozen encoders on the GPU at once (2 x 60M bf16 = ~240 MB)."""

    def __init__(self):
        self.tok = PairTokenizer(C.MAX_LEN)
        self.model = {}
        for side in ("vi", "en"):
            m = T5ForConditionalGeneration.from_pretrained(
                str(C.MODELS[side]["path"]), dtype=DT, low_cpu_mem_usage=True).to("cuda").eval()
            m.requires_grad_(False)
            self.model[side] = m


# ---------------------------------------------------------------- corpus access
def iter_rows(file: str | None = None, chunk: int = 200_000, limit: int | None = None):
    """Yield (vi, en) chunks, optionally restricted to one shard, in `id` order."""
    pf = pq.ParquetFile(C.PAIRS)
    cols = ["vi", "en", "file"] if file else ["vi", "en"]
    buf_vi, buf_en, got = [], [], 0
    for rb in pf.iter_batches(batch_size=8192, columns=cols):
        d = rb.to_pydict()
        for i, v in enumerate(d["vi"]):
            if file and d["file"][i] != file:
                continue
            buf_vi.append(v)
            buf_en.append(d["en"][i])
            if len(buf_vi) >= chunk:
                yield buf_vi, buf_en
                got += len(buf_vi)
                buf_vi, buf_en = [], []
                if limit and got >= limit:
                    return
    if buf_vi:
        yield buf_vi, buf_en


def split_of(ids: np.ndarray, n: int) -> np.ndarray:
    """80/10/10 on the *position inside the shard*, so the split is a pure function of the data."""
    pos = ids / max(n, 1)
    return np.where(pos < C.SPLIT["train"], "train",
                    np.where(pos < C.SPLIT["train"] + C.SPLIT["val"], "val", "test"))


# ---------------------------------------------------------------- accumulation
class Moments:
    def __init__(self, device: str = "cuda"):
        z = lambda: torch.zeros((D, D), dtype=torch.float64, device=device)  # noqa: E731
        self.A, self.B, self.C = z(), z(), z()
        self.sx = torch.zeros(D, dtype=torch.float64, device=device)
        self.sy = torch.zeros(D, dtype=torch.float64, device=device)
        self.n = 0
        self.pairs = 0

    def add(self, X, Y):
        self.A += (X.T @ X).double()
        self.B += (X.T @ Y).double()
        self.C += (Y.T @ Y).double()
        self.sx += X.sum(0).double()
        self.sy += Y.sum(0).double()
        self.n += X.shape[0]

    def save(self, path, meta: dict):
        np.savez_compressed(path, A=self.A.cpu().numpy(), B=self.B.cpu().numpy(), C=self.C.cpu().numpy(),
                            sx=self.sx.cpu().numpy(), sy=self.sy.cpu().numpy(),
                            n=np.array(self.n), pairs=np.array(self.pairs), meta=json.dumps(meta))

    @classmethod
    def load(cls, path) -> "Moments":
        z = np.load(path, allow_pickle=False)
        m = cls(device="cpu")
        m.A, m.B, m.C = (torch.from_numpy(z[k]) for k in ("A", "B", "C"))
        m.sx, m.sy = (torch.from_numpy(z[k]) for k in ("sx", "sy"))
        m.n, m.pairs = int(z["n"]), int(z["pairs"])
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


def accumulate(enc, vi, en, batch, label, log_every=10):
    m = Moments()
    t0 = time.time()
    total, done = len(vi), 0
    for s, n, X, Y, _lens in AlignedBatcher(enc, batch).batches(vi, en):
        m.add(X, Y)
        m.pairs += n
        done = s + n
        if (s // batch) % log_every == 0 or done == total:
            el = max(time.time() - t0, 1e-9)
            print(f"  {label} {done}/{total}  {done / el:6.1f} seq/s  tok/pair {m.n / max(m.pairs, 1):5.1f}  "
                  f"eta {(total - done) / (done / el) / 60:5.1f} min", flush=True)
    return m


# ---------------------------------------------------------------- modes
def mode_moments(file: str, batch: int):
    # counts come from the manifest: reading the 6.67M-row `file` column just to count costs ~500 MB
    man = json.loads(C.PAIRS_MANIFEST.read_text(encoding="utf-8"))
    n_total = int(man["per_file"][file])
    print(f"shard {file}: {n_total:,} pairs")
    splits = {k: C.DATA / "stats" / f"{file.split('-')[1]}_{k}.npz" for k in ("train", "val")}
    for k, p in splits.items():
        p.parent.mkdir(parents=True, exist_ok=True)
    if all(p.exists() for p in splits.values()):
        print("both moment files exist; use --force to redo")
        return
    enc = Encoders()
    # one pass: id order is md5 order, so the first 80 % is train, the next 10 % val
    buf_tr = ([], [])
    buf_va = ([], [])
    seen = 0
    for vi, en in iter_rows(file=file, chunk=50_000):
        ids = np.arange(seen, seen + len(vi))
        sp = split_of(ids, n_total)
        for j, s in enumerate(sp):
            if s == "train":
                buf_tr[0].append(vi[j]); buf_tr[1].append(en[j])
            elif s == "val":
                buf_va[0].append(vi[j]); buf_va[1].append(en[j])
        seen += len(vi)
    print(f"collected train {len(buf_tr[0]):,} / val {len(buf_va[0]):,}", flush=True)

    if not splits["train"].exists():
        t0 = time.time()
        m = accumulate(enc, buf_tr[0], buf_tr[1], batch, "train")
        m.save(splits["train"], {"file": file, "split": "train", "pairs": m.pairs})
        print(f"train moments: {m.n:,} token pairs, {(time.time() - t0) / 60:.1f} min")
    if not splits["val"].exists():
        m = accumulate(enc, buf_va[0], buf_va[1], batch, "val")
        m.save(splits["val"], {"file": file, "split": "val", "pairs": m.pairs})
        print(f"val moments: {m.n:,} token pairs")
    (C.DATA / "stats" / "chosen.json").write_text(
        json.dumps({"file": file, "n_pairs": n_total, "n_train": len(buf_tr[0]), "n_val": len(buf_va[0])},
                   indent=2), encoding="utf-8")


def mode_sentence(batch: int, limit: int | None):
    """Mean-pooled vi-side embeddings for every pair, fp16, in id order (coreset input)."""
    if C.SENT_EMB.exists() and C.SENT_EMB_META.exists():
        print(f"{C.SENT_EMB.name} exists — reusing")
        return
    enc = Encoders()
    n_total = pq.ParquetFile(C.PAIRS).metadata.num_rows
    if limit:
        n_total = min(n_total, limit)
    out = np.lib.format.open_memmap(C.SENT_EMB, mode="w+", dtype=np.float16, shape=(n_total, D))
    files, ids = [], []
    t0, done = time.time(), 0
    for vi, _en in iter_rows(chunk=100_000, limit=limit):
        for s, n, pooled in SideBatcher(enc, "vi", batch).batches(vi):
            out[done + s: done + s + n] = pooled.astype(np.float16)
        done += len(vi)
        if (done // 100_000) % 5 == 0:
            print(f"  {done:,}/{n_total:,}  {done / (time.time() - t0):6.1f} sent/s  "
                  f"eta {(n_total - done) / (done / max(time.time() - t0, 1e-9)) / 60:5.1f} min", flush=True)
    out.flush()
    import pyarrow as pa
    t = pq.read_table(C.PAIRS, columns=["id", "file"])
    if limit:
        t = t.slice(0, limit)
    pq.write_table(t, C.SENT_EMB_META, compression="zstd")
    print(f"wrote {C.SENT_EMB.name} {out.shape} in {(time.time() - t0) / 60:.1f} min")


def mode_sentence_sample(per_file: int, batch: int):
    """Stratified coreset sample: the first `per_file` rows of each shard (id order = md5 order,
    i.e. a content-independent random sample), vi-side mean-pooled, fp16.

    Streams the parquet: only 20k strings are ever held in RAM (an earlier version loaded all
    6.67M `vi` values and needed 2.4 GB).
    """
    out_npy = C.DATA / "sent_sample_fp16.npy"
    out_meta = C.DATA / "sent_sample_meta.parquet"
    if out_npy.exists() and out_meta.exists():
        print(f"{out_npy.name} exists — reusing")
        return
    import pyarrow as pa
    enc = Encoders()
    pf = pq.ParquetFile(C.PAIRS)
    n_rows = pf.metadata.num_rows
    out = np.zeros((min(n_rows, per_file * 14), D), dtype=np.float16)
    counts: dict[str, int] = {}
    keep_row, keep_file, buf = [], [], []
    g = 0
    t0, done = time.time(), 0

    def flush():
        nonlocal done
        if not buf:
            return
        for _s, n, pooled in SideBatcher(enc, "vi", batch).batches(buf):
            out[done:done + n] = pooled.astype(np.float16)
            done += n
        buf.clear()

    for rb in pf.iter_batches(batch_size=8192, columns=["file", "vi"]):
        d = rb.to_pydict()
        for f, v in zip(d["file"], d["vi"]):
            if counts.get(f, 0) < per_file:
                counts[f] = counts.get(f, 0) + 1
                keep_row.append(g)
                keep_file.append(f)
                buf.append(v)
                if len(buf) >= 20_000:
                    flush()
                    print(f"  {done:,}  {done / (time.time() - t0):6.1f} sent/s", flush=True)
            g += 1
    flush()
    out = out[:done]
    np.save(out_npy, out)
    pq.write_table(pa.table({"row": np.asarray(keep_row), "file": keep_file}), out_meta, compression="zstd")
    print(f"wrote {out_npy.name} {out.shape} in {(time.time() - t0) / 60:.1f} min")


def mode_subset(file: str):
    """Materialise the chosen shard as data/chosen.parquet with its 80/10/10 split.

    Every downstream stage (moments, LightGBM, adapters, generation, geometry) then reads 30 MB
    instead of scanning 370 MB and holding 6.67M strings in RAM.
    """
    out = C.DATA / "chosen.parquet"
    if out.exists():
        print(f"{out.name} exists — reusing")
        return
    import pyarrow as pa
    man = json.loads(C.PAIRS_MANIFEST.read_text(encoding="utf-8"))
    n = int(man["per_file"][file])
    pf = pq.ParquetFile(C.PAIRS)
    pos, vi, en, sp = [], [], [], []
    seen = 0
    for rb in pf.iter_batches(batch_size=16384, columns=["file", "vi", "en"]):
        d = rb.to_pydict()
        for i, f in enumerate(d["file"]):
            if f != file:
                continue
            r = seen / n
            pos.append(seen)
            vi.append(d["vi"][i])
            en.append(d["en"][i])
            sp.append("train" if r < C.SPLIT["train"] else
                      ("val" if r < C.SPLIT["train"] + C.SPLIT["val"] else "test"))
            seen += 1
    t = pa.table({"pos": np.asarray(pos, dtype=np.int32), "split": sp, "vi": vi, "en": en})
    pq.write_table(t, out, compression="zstd")
    print(f"wrote {out.name}: {seen:,} rows "
          f"({sp.count('train'):,} train / {sp.count('val'):,} val / {sp.count('test'):,} test)")


def load_chosen(columns=("vi", "en", "split")):
    """Read the materialised chosen shard (see `mode_subset`)."""
    p = C.DATA / "chosen.parquet"
    if not p.exists():
        raise SystemExit("data/chosen.parquet missing; run `python -m src.embed subset --file <shard>`")
    return pq.read_table(p, columns=list(columns)).to_pydict()


def mode_smoke(n: int, batch: int):
    enc = Encoders()
    vi, en = next(iter(iter_rows(chunk=n, limit=n)))
    t0 = time.time()
    m = accumulate(enc, vi, en, batch, "smoke", log_every=2)
    el = time.time() - t0
    print(f"smoke: {len(vi)} pairs in {el:.1f}s = {len(vi) / el:.1f} seq/s, {m.n / m.pairs:.1f} tok/pair")
    print(f"vram {torch.cuda.memory_allocated() / 2**20:.0f} MB, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    t1 = time.time()
    tot = 0
    for _s, nn, _p in SideBatcher(enc, "vi", batch).batches(vi):
        tot += nn
    print(f"sentence mode: {tot / (time.time() - t1):.1f} sent/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["moments", "sentence", "sentence_sample", "subset", "smoke"])
    ap.add_argument("--file", default="")
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--per-file", type=int, default=50_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.mode == "moments":
        if not a.file:
            raise SystemExit("--file <shard> required")
        mode_moments(a.file, a.batch)
    elif a.mode == "sentence":
        mode_sentence(a.batch, a.limit)
    elif a.mode == "sentence_sample":
        mode_sentence_sample(a.per_file, a.batch)
    elif a.mode == "subset":
        if not a.file:
            raise SystemExit("--file <shard> required")
        mode_subset(a.file)
    else:
        mode_smoke(a.n, a.batch)
