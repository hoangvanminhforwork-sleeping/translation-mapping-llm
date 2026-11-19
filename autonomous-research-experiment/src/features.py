"""Reference-free features for every candidate in the pool.

Two families, computed side by side so the comparison is paired by construction:

  string round-trip   back-translate the English candidate with B (opus-mt-en-vi), then compare the
                      Vietnamese back-translation to the source with chrF / char-3gram Jaccard.
  latent round-trip   embed BOTH the source and the back-translation with A's frozen encoder
                      (same model, same Vietnamese tokenizer -> one shared space) and measure
                      agreement: pooled cosine and token-level nearest-neighbour precision/recall/F1.
                      Every latent feature is computed TWICE: on raw states and on *centred* states
                      (mean subtracted, as the project's earlier phases learned that an uncentred
                      cosine is dominated by the shared mean direction).

Nothing here reads `ref`: these features are reference-free by construction.

Memory strategy: back-translations are computed once per unique string; encoder states are never all
kept - the pool is processed in chunks and states are freed after the features are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
CHRF_M = CHRF()
torch.manual_seed(C.SEED)


def load_pool(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def backtranslate(texts: list[str], batch: int, max_new: int) -> list[str]:
    tok = AutoTokenizer.from_pretrained(C.B_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(C.B_ID, dtype=torch.float16,
                                                 low_cpu_mem_usage=True).to("cuda").eval()
    out = []
    t0 = time.time()
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if (s // batch) % 40 == 0:
            done = min(s + batch, len(texts))
            print(f"  bt {done}/{len(texts)}  {done / (time.time() - t0):.1f} sent/s", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def encode(texts: list[str], tok, model, batch: int) -> list[torch.Tensor]:
    """Per-token encoder states, fp32 on CPU, NOT normalised (centring happens later)."""
    enc_model = model.get_encoder()
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to("cuda")
        h = enc_model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].bool()
        for bi in range(h.shape[0]):
            out.append(h[bi][mask[bi]].cpu())
    return out


def feats(hs: torch.Tensor, hb: torch.Tensor, mu: torch.Tensor, suffix: str) -> dict:
    """Latent agreement between source states and back-translation states."""
    if suffix == "_c":
        hs, hb = hs - mu, hb - mu
    hs, hb = F.normalize(hs, dim=-1), F.normalize(hb, dim=-1)
    sim = hs @ hb.T
    pre = sim.max(dim=1).values
    rec = sim.max(dim=0).values
    return {
        f"lat_cos_mean{suffix}": float((hs.mean(0) @ hb.mean(0)).item()),
        f"lat_match_pre{suffix}": float(pre.mean()),
        f"lat_match_rec{suffix}": float(rec.mean()),
        f"lat_match_f1{suffix}": float(2 * pre.mean() * rec.mean() / (pre.mean() + rec.mean() + 1e-8)),
        f"lat_match_std{suffix}": float(pre.std()),
        f"lat_len_ratio{suffix}": float(hs.shape[0] / max(hb.shape[0], 1)),
    }


def backtranslate_cached(uniq: list[str], cache_path, batch: int, max_new: int) -> dict:
    """Back-translate with a JSON checkpoint every 10 k strings: a crash costs 10 k, not 80 k."""
    bt = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [u for u in uniq if u not in bt]
    print(f"back-translations: cached {len(bt)}, todo {len(todo)}")
    for i in range(0, len(todo), 10_000):
        chunk = todo[i:i + 10_000]
        bt.update(zip(chunk, backtranslate(chunk, batch, max_new)))
        cache_path.write_text(json.dumps(bt), encoding="utf-8")
        print(f"  checkpoint {len(bt)}/{len(uniq)}", flush=True)
    return bt


def char3(a: str, b: str) -> float:
    A = {a[i:i + 3] for i in range(max(len(a) - 2, 1))}
    B = {b[i:i + 3] for i in range(max(len(b) - 2, 1))}
    return len(A & B) / max(len(A | B), 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--mu-sample", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out = C.RESULTS / a.out
    if out.exists() and not a.force:
        print(f"{out.name} exists; --force to redo")
        return

    rows = load_pool(C.RESULTS / a.pool)
    cands = [c for r in rows for c in r["cands"]]
    uniq = sorted({c["hyp"] for c in cands})
    print(f"{len(rows)} rows, {len(cands)} candidates, {len(uniq)} unique")

    bt_path = C.RESULTS / f"bt_{Path(a.pool).stem}.json"
    bt = backtranslate_cached(uniq, bt_path, a.batch, C.MAX_NEW)

    tok = AutoTokenizer.from_pretrained(C.A_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(C.A_ID, dtype=torch.float16,
                                                 low_cpu_mem_usage=True).to("cuda").eval()

    # corpus mean direction, estimated on a sample of sources (used for the centred variants)
    sample = [r["vi"] for r in rows[:a.mu_sample]]
    st = encode(sample, tok, model, a.batch)
    mu = torch.cat([s.mean(0, keepdim=True) for s in st], 0).mean(0)
    del st
    print(f"mu sample={len(sample)}  ||mu||={mu.norm():.2f}")

    recs = []
    t0 = time.time()
    for s in range(0, len(rows), a.chunk):
        chunk = rows[s:s + a.chunk]
        hs_list = encode([r["vi"] for r in chunk], tok, model, a.batch)
        u = sorted({c["hyp"] for r in chunk for c in r["cands"]})
        hb_map = dict(zip(u, encode([bt[x] for x in u], tok, model, min(a.batch, 64))))
        for r, hs in zip(chunk, hs_list):
            srclen = len(r["vi"].split())
            for j, c in enumerate(r["cands"]):
                h = c["hyp"]
                hb = hb_map[h]
                tgtlen = len(h.split())
                rec = {"id": r["id"], "rank": j, "src": c["src"], "hyp": h,
                       "hyp_len": tgtlen, "src_len": srclen, "len_ratio": tgtlen / max(srclen, 1),
                       "bt": bt[h], "chrF_bt": float(CHRF_M.sentence_score(bt[h], [r["vi"]]).score),
                       "char3_bt": char3(bt[h], r["vi"]),
                       **feats(hs, hb, mu, ""), **feats(hs, hb, mu, "_c")}
                recs.append(rec)
        print(f"  chunk {s + len(chunk)}/{len(rows)}  {len(recs)} candidate rows  "
              f"{time.time() - t0:.0f}s", flush=True)
        del hs_list, hb_map

    pq.write_table(pa.Table.from_pylist(recs), out.with_suffix(".parquet"), compression="zstd")
    print(f"wrote {out.with_suffix('.parquet').name} ({len(recs)} candidate rows)")


if __name__ == "__main__":
    main()
