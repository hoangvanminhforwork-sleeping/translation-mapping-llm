"""Calibration — before any new claim, reproduce E9's published channels with this code.

If `ridge` here does not match `GEN_VARIANTS.md` (BLEU 2.73 / chrF 19.73 / CE 3.228) and `native`
does not match (BLEU 4.20 / chrF 14.82 / CE 3.413) on the same 1 000 test sentences, the
reimplementation is wrong and every number downstream is uninterpretable.

Also answers: is `ref` (target encoder reading the English reference) a usable ceiling?

Usage
    python -m src.validate [--n 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from . import cc

sys.stdout.reconfigure(encoding="utf-8")


def main(n: int):
    model = cc.load_decoder(dtype=cc.DT_A)
    d = cc.CACHE / "test"
    txt = json.loads((d / "texts.json").read_text(encoding="utf-8"))
    n = min(n, len(txt["vi"]))
    refs = txt["en"][:n]
    W, b = cc.load_map()
    rows = {}
    t0 = time.time()
    for name, ek, mk in (("ridge", "E_vi", "m_vi"), ("native", "E_nat", "m_nat"),
                         ("ref", "E_ref", "m_ref")):
        E = torch.from_numpy(np.asarray(np.load(d / f"{ek}.npy", mmap_mode="r")[:n])
                             .astype(np.float32)).cuda()
        am = torch.from_numpy(np.asarray(np.load(d / f"{mk}.npy", mmap_mode="r")[:n])
                              .astype(np.float32)).cuda()
        H = E @ W + b if name == "ridge" else E
        hyps = cc.generate(model, H, am)
        ce = cc.ce_per_sentence(model, H, am, cc.labels(refs))
        rows[name] = {"bleu": round(float(cc.bleu(hyps, refs)), 3),
                      "chrf": round(float(cc.chrf(hyps, refs)), 3),
                      "ce": round(float(ce.mean()), 4),
                      "degen": round(float(cc.degen_rate(hyps)), 3),
                      "norm": round(float((H * am.unsqueeze(-1)).pow(2).sum((1, 2)).sqrt().mean()), 2)}
        print(f"{name:7s} bleu {rows[name]['bleu']:.2f}  chrf {rows[name]['chrf']:.2f}  "
              f"ce {rows[name]['ce']:.3f}  degen {rows[name]['degen']:.3f}  "
              f"|H| {rows[name]['norm']:.1f}  ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    (cc.RESULTS / "calibration.json").write_text(json.dumps(
        {"n": n, "channels": rows,
         "expected_e9": {"ridge": {"bleu": 2.73, "chrf": 19.73, "ce": 3.228},
                         "native": {"bleu": 4.20, "chrf": 14.82, "ce": 3.413}}},
        indent=2), encoding="utf-8")
    print("wrote results/calibration.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    main(ap.parse_args().n)
