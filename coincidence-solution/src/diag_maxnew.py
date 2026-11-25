"""Diagnostic — is the failed recolor replication a `max_new_tokens` artefact?

E9 (`GEN_VARIANTS.md`) generated with `--max-new 48` in bf16 and reported ridge 2.73 ->
ridge+recolor(0.1) 3.17. This package generates 64 tokens in fp32 and measures 2.50 -> 2.53.
Before calling that a failed replication, remove the decoding confound: same maps, same adapter,
same 1 000 sentences, max_new in {48, 64}.

Usage
    python -m src.diag_maxnew
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from . import cc

sys.stdout.reconfigure(encoding="utf-8")


def main():
    model = cc.load_decoder(dtype=cc.DT_A)
    t = cc.CACHE / "test"
    txt = json.loads((t / "texts.json").read_text(encoding="utf-8"))
    refs = txt["en"]
    n = len(refs)
    E = torch.from_numpy(np.asarray(np.load(t / "E_vi.npy", mmap_mode="r")[:n]).astype(np.float32)).cuda()
    am = torch.from_numpy(np.asarray(np.load(t / "m_vi.npy", mmap_mode="r")[:n]).astype(np.float32)).cuda()
    rows = {}
    for arm in ("ridge_sub", "ridge_recolor10"):
        z = np.load(cc.WORK / "maps" / f"{arm}.npz")
        H = E @ torch.from_numpy(z["W"]).cuda() + torch.from_numpy(z["b"]).cuda()
        for mn in (48, 64):
            t0 = time.time()
            hyps = cc.generate(model, H, am, max_new=mn)
            rows[f"{arm}_max{mn}"] = {
                "bleu": round(float(cc.bleu(hyps, refs)), 3),
                "chrf": round(float(cc.chrf(hyps, refs)), 3),
                "len_mean": round(float(np.mean([len(h.split()) for h in hyps])), 1),
                "sec": round(time.time() - t0, 1)}
            print(f"{arm:16s} max_new {mn}  bleu {rows[f'{arm}_max{mn}']['bleu']:.2f}  "
                  f"chrf {rows[f'{arm}_max{mn}']['chrf']:.2f}  "
                  f"len {rows[f'{arm}_max{mn}']['len_mean']:.1f}", flush=True)
    d = (rows["ridge_recolor10_max48"]["bleu"] - rows["ridge_sub_max48"]["bleu"])
    d64 = (rows["ridge_recolor10_max64"]["bleu"] - rows["ridge_sub_max64"]["bleu"])
    rows["delta_max48"] = round(d, 3)
    rows["delta_max64"] = round(d64, 3)
    print(f"recolor delta: max48 {d:+.2f}   max64 {d64:+.2f}", flush=True)
    (cc.RESULTS / "diag_maxnew.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
