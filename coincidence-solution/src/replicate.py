"""Replication check — does the headline hold on a *disjoint* 1 000-sentence test sample?

The main table uses test rows 0-999 of the md5-ordered test split. This evaluates the same fitted
maps on rows 1000-1999, which were never used for anything in this session. If the gap survives
there, it is not a split artefact.

Usage
    python -m src.replicate
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from . import cc

sys.stdout.reconfigure(encoding="utf-8")
ARMS = ("ce_ridge_e16", "ce_ridge_e6", "ce_ridge", "distill", "ridge_full",
        "ridge_recolor10", "ce_scratch")


def main():
    model = cc.load_decoder(dtype=cc.DT_A)
    t = cc.CACHE / "test2"
    txt = json.loads((t / "texts.json").read_text(encoding="utf-8"))
    refs = txt["en"]
    E = np.asarray(np.load(t / "E_vi.npy", mmap_mode="r")).astype(np.float32)
    m = np.asarray(np.load(t / "m_vi.npy", mmap_mode="r")).astype(np.float32)
    lab = torch.from_numpy(np.asarray(np.load(t / "lab.npy", mmap_mode="r")).astype(np.int64)).cuda()
    H_E = torch.from_numpy(E).cuda()
    am = torch.from_numpy(m).cuda()
    out_p = cc.RESULTS / "replicate.json"
    out = (json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists()
           else {"split": "test rows 1000-1999", "n": len(refs), "arms": {}})
    for arm in ARMS:
        p = cc.WORK / "maps" / f"{arm}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        H = H_E @ torch.from_numpy(d["W"]).cuda() + torch.from_numpy(d["b"]).cuda()
        t0 = time.time()
        hyps = cc.generate(model, H, am)
        ce = cc.ce_per_sentence(model, H, am, lab=lab)
        out["arms"][arm] = {"bleu": round(float(cc.bleu(hyps, refs)), 3),
                            "chrf": round(float(cc.chrf(hyps, refs)), 3),
                            "ce": round(float(ce.mean()), 4),
                            "degen": round(float(cc.degen_rate(hyps)), 3)}
        (cc.WORK / "hyps_test2").mkdir(parents=True, exist_ok=True)
        (cc.WORK / "hyps_test2" / f"{arm}.json").write_text(json.dumps(hyps, ensure_ascii=False),
                                                            encoding="utf-8")
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"{arm:16s} bleu {out['arms'][arm]['bleu']:.2f}  chrf {out['arms'][arm]['chrf']:.2f} "
              f"ce {out['arms'][arm]['ce']:.3f}  degen {out['arms'][arm]['degen']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    print("wrote results/replicate.json")


if __name__ == "__main__":
    main()
