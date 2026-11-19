"""chrF-MBR selection on any pool, without needing a feature table.

`arms.py` needs features because most of its rules do; MBR is the one rule that needs only the
candidates, and after the LoRA control produced its own pool, MBR on *that* pool is the cheapest test
of whether adaptation and selection are complementary.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sacrebleu.metrics import CHRF

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
CHRF_M = CHRF()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(C.RESULTS / a.pool, encoding="utf-8")]
    picks = []
    for k, r in enumerate(rows):
        hyps = [c["hyp"] for c in r["cands"]]
        n = len(hyps)
        util = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    util[i] += CHRF_M.sentence_score(hyps[i], [hyps[j]]).score
        picks.append(int(util.argmax()))
        if k % 2000 == 0:
            print(f"  {k}/{len(rows)}", flush=True)

    path = C.RESULTS / f"sel_{a.name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r, j in zip(rows, picks):
            c = r["cands"][j]
            f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                                "hyp": c["hyp"], "rank": j, "src": c["src"]}, ensure_ascii=False) + "\n")
    print(f"wrote sel_{a.name}.jsonl ({len(rows)} rows, changed {sum(j != 0 for j in picks)} times)")


if __name__ == "__main__":
    main()
