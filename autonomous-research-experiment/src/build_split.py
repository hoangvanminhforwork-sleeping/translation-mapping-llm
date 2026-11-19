"""Rebuild the 80/10/10 split with a bigger test/val (10k each) from the frozen 3.46M pairs.

Rule (deterministic, no RNG): order rows by md5(vi); rank 0..9999 = test, 10000..19999 = val,
rest = train. The historical split used rank 0..1999 as test, so `core` (rank < 2000) is a subset
of the new test -> the published 22.27 BLEU stays quotable on the same sentences.

Two streaming passes (7.7 GB RAM): pass 1 keeps only (md5, id); pass 2 rewrites row group by row
group. Idempotent: exits if data/split.parquet already exists (use --force).
"""
from __future__ import annotations

import hashlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")


def main(force: bool = False) -> None:
    if C.SPLIT.exists() and not force:
        n = pq.ParquetFile(C.SPLIT).metadata.num_rows
        print(f"{C.SPLIT.name} exists ({n} rows); use --force to rebuild")
        return

    src = pq.ParquetFile(C.SOURCE_PAIRS)
    keys: list[tuple[str, int]] = []
    for rg in range(src.num_row_groups):
        t = src.read_row_group(rg, columns=["vi"])
        for s in t.column("vi").to_pylist():
            keys.append((hashlib.md5(s.encode("utf-8")).hexdigest(), len(keys)))
        del t
    print(f"keyed {len(keys)} rows")

    keys.sort()
    split_of = {}
    core_of = {}
    for rank, (k, i) in enumerate(keys):
        if rank < C.N_TEST:
            split_of[i] = "test"
            core_of[i] = rank < C.N_TEST_CORE
        elif rank < C.N_TEST + C.N_VAL:
            split_of[i] = "val"
        else:
            split_of[i] = "train"
    del keys

    writer = None
    n = 0
    for rg in range(src.num_row_groups):
        t = src.read_row_group(rg)
        ids = t.column("id").to_pylist()
        part = pa.table({
            "id": t.column("id"),
            "vi": t.column("vi"),
            "en": t.column("en"),
            "split": pa.array([split_of[i] for i in ids], pa.string()),
            "core": pa.array([core_of.get(i, False) for i in ids], pa.bool_()),
        })
        writer = writer or pq.ParquetWriter(C.SPLIT, part.schema, compression="zstd")
        writer.write_table(part)
        n += len(ids)
        del t, part
        print(f"  rowgroup {rg + 1}/{src.num_row_groups}  {n} rows", flush=True)
    writer.close()

    out = pq.read_table(C.SPLIT, columns=["split"]).group_by("split").aggregate([("split", "count")])
    print("split counts:", out.to_pylist())
    print(f"wrote {C.SPLIT} ({C.SPLIT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
