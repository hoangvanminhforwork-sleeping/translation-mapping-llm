"""Flatten a pool to its rank-0 candidate (used for the LoRA control pools, which have no features)."""
from __future__ import annotations

import argparse
import json
import sys

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()

rows = [json.loads(line) for line in open(C.RESULTS / a.pool, encoding="utf-8")]
with open(C.RESULTS / f"sel_{a.name}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        c = r["cands"][0]
        f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                            "hyp": c["hyp"], "rank": 0, "src": c["src"]}, ensure_ascii=False) + "\n")
print(f"wrote sel_{a.name}.jsonl ({len(rows)} rows)")
