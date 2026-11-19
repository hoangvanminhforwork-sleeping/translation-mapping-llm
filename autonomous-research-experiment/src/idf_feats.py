"""Does IDF weighting fix the latent signal's known failure mode?

The qualitative analysis (results/EXAMPLES.md) showed latent agreement being fooled by *format*
evidence: function words, digits, `%s` templates. Those tokens are frequent, so a token-frequency
(idf) weight on the nearest-neighbour agreement is the natural fix. If latent agreement is a real
quality signal, idf weighting should raise its correlation; if it is shape-matching, idf will hurt,
because the shape is exactly what frequent tokens carry.

Cached back-translations (`results/bt_*.json`) are reused, so this costs one encoder pass.

Usage
    python -m src.idf_feats --pool pool_test_beam8.jsonl --out feat_idf_test.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")


@torch.no_grad()
def encode_ids(texts, tok, model, batch: int):
    """(ids, states) per sentence, states L2-normalised, on CPU."""
    enc_model = model.get_encoder()
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to("cuda")
        h = enc_model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].bool()
        ids = enc["input_ids"]
        for bi in range(h.shape[0]):
            out.append((ids[bi][mask[bi]].cpu().numpy(),
                        F.normalize(h[bi][mask[bi]], dim=-1).cpu()))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_test_beam8.jsonl")
    ap.add_argument("--out", default="feat_idf_test.parquet")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(C.RESULTS / a.pool, encoding="utf-8")]
    bt = json.loads((C.RESULTS / f"bt_{a.pool.replace('.jsonl', '')}.json").read_text(encoding="utf-8"))
    print(f"{len(rows)} rows, {len(bt)} cached back-translations")

    tok = AutoTokenizer.from_pretrained(C.A_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(C.A_ID, dtype=torch.float16,
                                                 low_cpu_mem_usage=True).to("cuda").eval()

    # document frequency over the source side of this pool (a cheap, local idf)
    df = Counter()
    n_doc = 0
    for r in rows:
        for t in set(tok(r["vi"])["input_ids"]):
            df[t] += 1
        n_doc += 1
    idf = {t: math.log((n_doc + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    print(f"idf over {n_doc} documents, {len(idf)} token types")

    recs = []
    t0 = time.time()
    for s in range(0, len(rows), a.chunk):
        chunk = rows[s:s + a.chunk]
        srcs = encode_ids([r["vi"] for r in chunk], tok, model, a.batch)
        u = sorted({c["hyp"] for r in chunk for c in r["cands"]})
        bts = dict(zip(u, encode_ids([bt[x] for x in u], tok, model, min(a.batch, 64))))
        for r, (sid, sh) in zip(chunk, srcs):
            for j, c in enumerate(r["cands"]):
                bid, bh = bts[c["hyp"]]
                sim = sh @ bh.T
                w_src = np.array([idf.get(int(t), 1.0) for t in sid], dtype=np.float32)
                w_bt = np.array([idf.get(int(t), 1.0) for t in bid], dtype=np.float32)
                pre = sim.max(dim=1).values.numpy()
                rec = sim.max(dim=0).values.numpy()
                wp = float((pre * w_src).sum() / max(w_src.sum(), 1e-6))
                wr = float((rec * w_bt).sum() / max(w_bt.sum(), 1e-6))
                recs.append({"id": r["id"], "rank": j, "idf_pre": wp, "idf_rec": wr,
                             "idf_f1": 2 * wp * wr / max(wp + wr, 1e-6)})
        print(f"  chunk {s + len(chunk)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    pq.write_table(pa.Table.from_pylist(recs), C.RESULTS / a.out, compression="zstd")
    print(f"wrote {a.out} ({len(recs)} rows)")


if __name__ == "__main__":
    main()
