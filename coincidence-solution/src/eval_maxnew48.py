"""Re-score the headline arms at E9's decoding setting (`max_new=48`) so the table is directly
comparable with `GEN_VARIANTS.md` (ridge 2.73 / recolor 3.17) and with E7/E8's published numbers.

The main table in this package uses `max_new=64`; both settings are reported because greedy length
matters by ~0.2-0.35 BLEU here (see results/diag_maxnew.json).

Usage
    python -m src.eval_maxnew48
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from . import cc

sys.stdout.reconfigure(encoding="utf-8")
ARMS = ("distill", "ridge_full", "ridge_sub", "ridge_recolor10", "ridge_scale122", "ce_ridge",
        "ce_ridge_e6", "ce_scratch")
MAX_NEW = 48


def main():
    model = cc.load_decoder(dtype=cc.DT_A)
    t = cc.CACHE / "test"
    txt = json.loads((t / "texts.json").read_text(encoding="utf-8"))
    refs = txt["en"]
    E = np.asarray(np.load(t / "E_vi.npy", mmap_mode="r")).astype(np.float32)
    m = np.asarray(np.load(t / "m_vi.npy", mmap_mode="r")).astype(np.float32)
    lab = torch.from_numpy(np.asarray(np.load(t / "lab.npy", mmap_mode="r")).astype(np.int64)).cuda()
    H_E = torch.from_numpy(E).cuda()
    am = torch.from_numpy(m).cuda()
    out_p = cc.RESULTS / "maxnew48.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"max_new": MAX_NEW, "n": len(refs), "arms": {}}
    for arm in ARMS:
        p = cc.WORK / "maps" / f"{arm}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        H = H_E @ torch.from_numpy(d["W"]).cuda() + torch.from_numpy(d["b"]).cuda()
        t0 = time.time()
        hyps = cc.generate(model, H, am, max_new=MAX_NEW)
        ce = cc.ce_per_sentence(model, H, am, lab=lab)
        out["arms"][arm] = {"bleu": round(float(cc.bleu(hyps, refs)), 3),
                            "chrf": round(float(cc.chrf(hyps, refs)), 3),
                            "ce": round(float(ce.mean()), 4),
                            "degen": round(float(cc.degen_rate(hyps)), 3)}
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (cc.WORK / "hyps48").mkdir(parents=True, exist_ok=True)
        (cc.WORK / "hyps48" / f"{arm}.json").write_text(json.dumps(hyps, ensure_ascii=False),
                                                        encoding="utf-8")
        print(f"{arm:16s} bleu {out['arms'][arm]['bleu']:.2f}  chrf {out['arms'][arm]['chrf']:.2f} "
              f"ce {out['arms'][arm]['ce']:.3f}  degen {out['arms'][arm]['degen']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    print("wrote results/maxnew48.json")


if __name__ == "__main__":
    main()
