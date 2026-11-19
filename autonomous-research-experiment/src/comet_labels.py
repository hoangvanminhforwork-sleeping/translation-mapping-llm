"""Reference-based teacher signal, computed ONCE per candidate pool.

Writes one row per candidate: (id, rank, comet). Used for three things and nothing else:
  - the *oracle* arm (headroom),
  - the training target of the tiny learned ranker,
  - the ground truth against which every reference-free utility is correlated.

COMET here is a measurement instrument, not something the reference-free arms can see.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out = C.RESULTS / a.out
    if out.exists() and not a.force:
        print(f"{out.name} exists; --force to redo")
        return

    with open(C.RESULTS / a.pool, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    payload, index = [], []
    triples = {}
    for r in rows:
        for j, c in enumerate(r["cands"]):
            key = (r["vi"], c["hyp"], r["ref"])
            if key not in triples:
                triples[key] = len(payload)
                payload.append({"src": r["vi"], "mt": c["hyp"], "ref": r["ref"]})
            index.append(triples[key])
    print(f"{len(rows)} rows, {len(index)} candidate slots, {len(payload)} unique triples")

    from comet import download_model, load_from_checkpoint

    ckpt = download_model(C.COMET_ID)
    model = load_from_checkpoint(ckpt)
    res = model.predict(payload, batch_size=a.batch, gpus=1 if torch.cuda.is_available() else 0,
                        progress_bar=True)
    scores = np.asarray(res.scores, dtype=np.float32)
    print(f"system mean over unique triples {scores.mean():.4f}")

    tbl = pa.table({"id": pa.array([r["id"] for r in rows for _ in r["cands"]], pa.int64()),
                    "rank": pa.array([j for r in rows for j in range(len(r["cands"]))], pa.int32()),
                    "comet": pa.array(scores[index], pa.float32())})
    pq.write_table(tbl, out, compression="zstd")
    print(f"wrote {out.name} ({len(index)} rows)")


if __name__ == "__main__":
    main()
