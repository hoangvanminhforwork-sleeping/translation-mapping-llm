"""Five disagreements, in full text, with both utilities and the teacher's verdict.

A table can say "latent agreement has Spearman 0.31". Only examples can show *what kind of mistake*
that number is made of - which is the difference between a metric and an insight.

Picks the rows where the string-pick and the latent-pick disagree AND the COMET difference between
them is largest, so the reader sees the stakes, not the average.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")

K = 5


def main() -> None:
    pool = [json.loads(line) for line in open(C.RESULTS / "pool_test_beam8.jsonl", encoding="utf-8")]
    feat = pq.read_table(C.RESULTS / "feat_test_beam8.parquet").to_pylist()
    com = pq.read_table(C.RESULTS / "comet_test_beam8.parquet").to_pylist()

    by_id: dict[int, list[dict]] = {}
    for f in feat:
        by_id.setdefault(f["id"], []).append(f)
    for v in by_id.values():
        v.sort(key=lambda x: x["rank"])
    cc = {(r["id"], r["rank"]): r["comet"] for r in com}

    cases = []
    for r in pool:
        fs = by_id[r["id"]]
        i_str = int(np.argmax([f["chrF_bt"] for f in fs]))
        i_lat = int(np.argmax([f["lat_match_f1_c"] for f in fs]))
        if i_str == i_lat:
            continue
        gap = cc[(r["id"], i_lat)] - cc[(r["id"], i_str)]
        cases.append((abs(gap), gap, r, fs, i_str, i_lat))
    cases.sort(key=lambda x: x[0], reverse=True)

    lines = ["# Five disagreements between string and latent round-trip\n"]
    for _, gap, r, fs, i_s, i_l in cases[:K]:
        lines.append(f"## COMET(latent pick) − COMET(string pick) = {gap:+.3f}\n")
        lines.append(f"**Source (vi):** {r['vi']}\n")
        lines.append(f"**Reference (en):** {r['ref']}\n")
        lines.append(f"**string pick** (chrF_bt {fs[i_s]['chrF_bt']:.1f}, "
                     f"lat_f1_c {fs[i_s]['lat_match_f1_c']:.3f}, COMET {cc[(r['id'], i_s)]:.3f}): "
                     f"{fs[i_s]['hyp']}\n")
        lines.append(f"**latent pick** (chrF_bt {fs[i_l]['chrF_bt']:.1f}, "
                     f"lat_f1_c {fs[i_l]['lat_match_f1_c']:.3f}, COMET {cc[(r['id'], i_l)]:.3f}): "
                     f"{fs[i_l]['hyp']}\n")
        lines.append(f"**back-translation of the latent pick:** {fs[i_l]['bt']}\n")
    out = "\n".join(lines)
    (C.RESULTS / "EXAMPLES.md").write_text(out, encoding="utf-8")
    print(f"wrote EXAMPLES.md ({len(cases)} disagreeing rows, showing {min(K, len(cases))})")


if __name__ == "__main__":
    main()
