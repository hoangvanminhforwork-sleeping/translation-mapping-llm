"""The naive channel: inject the source encoder's states into the target decoder with no bridge.

This is the literal form of "two frozen monolingual models, no translation model" — the same
decoder and the same rank-16 LoRA as every other arm, the same 1 000 held-out sentences, the only
difference from `ridge`/`ce_*` being that no map is applied.

Feeds `paper.md` figure 2(a) and the Introduction.
Writes `results/identity.json`. Idempotent: skips if the file exists (`--force` to redo).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from . import cc


def main() -> None:
    out = cc.RESULTS / "identity.json"
    if out.exists() and "--force" not in sys.argv:
        print("exists:", out, "(pass --force to redo)")
        return

    d = cc.load_pairs("test", n=cc.N_TEST)
    print("n =", len(d["vi"]), flush=True)

    vi = cc.load_encoder("vi")
    h, keep = cc.encode_vi_mapped(vi, d["vi"])
    h = torch.from_numpy(h).cuda().to(cc.DT)
    am = torch.from_numpy(keep).cuda()
    del vi
    torch.cuda.empty_cache()

    dec = cc.load_decoder()
    lab = cc.labels(d["en"])

    ce = cc.ce_per_sentence(dec, h, am, lab=lab)
    hyps = cc.generate(dec, h, am, max_new=64)

    res = {
        "channel": "identity (no bridge)",
        "n": len(d["vi"]),
        "bleu": round(cc.bleu(hyps, d["en"]), 3),
        "chrf": round(cc.chrf(hyps, d["en"]), 3),
        "ce": round(float(ce.mean()), 4),
        "degen": round(cc.degen_rate(hyps), 3),
        "empty": round(sum(not x.strip() for x in hyps) / len(hyps), 3),
        "len_words": round(float(np.mean([len(x.split()) for x in hyps])), 2),
        "ref_words": round(float(np.mean([len(x.split()) for x in d["en"]])), 2),
        "examples": [{"vi": d["vi"][i], "ref": d["en"][i], "hyp": hyps[i]} for i in range(6)],
    }
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
