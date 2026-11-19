"""Selection by an arbitrary candidate-level feature column (used for the idf ablation)."""
from __future__ import annotations

import argparse
import json
import sys

import pyarrow.parquet as pq

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--feat", required=True)
ap.add_argument("--col", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()

rows = [json.loads(line) for line in open(C.RESULTS / a.pool, encoding="utf-8")]
feats = pq.read_table(C.RESULTS / a.feat).to_pylist()
by: dict[int, list[dict]] = {}
for f in feats:
    by.setdefault(f["id"], []).append(f)

path = C.RESULTS / f"sel_{a.name}.jsonl"
changed = 0
with path.open("w", encoding="utf-8") as fh:
    for r in rows:
        cs = by[r["id"]]
        pick = max(cs, key=lambda c: c[a.col])
        j = pick["rank"]
        changed += j != 0
        c = r["cands"][j]
        fh.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                             "hyp": c["hyp"], "rank": j, "src": a.col}, ensure_ascii=False) + "\n")
print(f"wrote sel_{a.name}.jsonl ({len(rows)} rows, changed {changed})")
