"""Purpose-built reference-free QE as a utility (the baseline any hand-built signal must beat).

`Unbabel/wmt22-cometkiwi-da` predicts quality from (src, mt) alone — exactly what our round-trip
signals try to do by hand. Scoring it over the *same* pool makes the comparison fair.

This model is a metric, like the COMET scorer already used throughout; it is not a third translation
model, and it never sees a reference.

Usage
    python -m src.kiwi --pool pool_test_beam8.jsonl --subset core --out sel_A11_kiwi.jsonl
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
KIWI_ID = "Unbabel/wmt22-cometkiwi-da"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_test_beam8.jsonl")
    ap.add_argument("--subset", default="core", choices=["core", "all"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="A11_kiwi")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    outp = C.RESULTS / f"kiwi_{a.pool.replace('pool_', '').replace('.jsonl', '')}_{a.subset}.parquet"
    rows = [json.loads(line) for line in open(C.RESULTS / a.pool, encoding="utf-8")]
    if a.subset == "core":
        rows = [r for r in rows if r.get("core")]
    print(f"{len(rows)} rows")

    if outp.exists() and not a.force:
        scores = pq.read_table(outp).to_pylist()
    else:
        payload, index = [], []
        seen = {}
        for r in rows:
            for j, c in enumerate(r["cands"]):
                key = (r["vi"], c["hyp"])
                if key not in seen:
                    seen[key] = len(payload)
                    payload.append({"src": r["vi"], "mt": c["hyp"]})
                index.append(seen[key])
        print(f"{len(index)} slots, {len(payload)} unique pairs")
        from comet import download_model, load_from_checkpoint

        model = load_from_checkpoint(download_model(KIWI_ID))
        res = model.predict(payload, batch_size=a.batch, gpus=1 if torch.cuda.is_available() else 0)
        sc = np.asarray(res.scores, dtype=np.float32)
        scores = [{"id": r["id"], "rank": j, "kiwi": float(sc[index[k]])}
                  for k, (r, j) in enumerate([(r, j) for r in rows for j in range(len(r["cands"]))])]
        pq.write_table(pa.Table.from_pylist(scores), outp, compression="zstd")
        print(f"wrote {outp.name}")

    by_id: dict[int, list[dict]] = {}
    for s in scores:
        by_id.setdefault(s["id"], []).append(s)
    for v in by_id.values():
        v.sort(key=lambda x: x["rank"])

    path = C.RESULTS / f"sel_{a.name}.jsonl"
    changed = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            ss = by_id[r["id"]]
            pick = max(ss, key=lambda x: x["kiwi"])
            j = pick["rank"]
            changed += j != 0
            c = r["cands"][j]
            f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                                "hyp": c["hyp"], "rank": j, "src": "kiwi"}, ensure_ascii=False) + "\n")
    print(f"wrote sel_{a.name}.jsonl ({len(rows)} rows, changed {changed})")


if __name__ == "__main__":
    main()
