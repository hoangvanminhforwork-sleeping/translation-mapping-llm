"""Generate candidate pools with the frozen translator (A = opus-mt-vi-en).

Modes
    greedy   num_beams=1                       -> the historical baseline (22.27 BLEU)
    beam     num_beams=K, K returned           -> n-best pool
    sample   K samples at T, top_p             -> diverse pool
    mixed    beam-K + sample-K                 -> union (dedup by string) = the pool used for reranking

Every row stores its full pool, so reranking experiments never re-generate.
Writes {out}.tmp then renames -> a killed run can never look finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(C.SEED)


def load_split_rows(split: str, n: int, scan_limit: int = 0) -> list[dict]:
    """Rows in md5 order (the split was built by ranking md5), so 'first n' == 'smallest n md5'.

    scan_limit: stop reading row groups once this many `split` rows are buffered (train has 3.4 M
    rows and reading all of them costs ~1.5 GB RAM). The buffer is still md5-sorted, so the result
    is deterministic.
    """
    pf = pq.ParquetFile(C.SPLIT)
    out = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["id", "vi", "en", "core", "split"])
        out.extend(r for r in t.to_pylist() if r["split"] == split)
        del t
        if scan_limit and len(out) >= scan_limit:
            break
    for r in out:
        r.pop("split")
    out.sort(key=lambda r: hashlib.md5(r["vi"].encode("utf-8")).hexdigest())  # same rule as build_split
    return out[:n]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--mode", default="greedy", choices=["greedy", "beam", "sample", "mixed"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=C.MAX_NEW)
    ap.add_argument("--out", default="")
    ap.add_argument("--scan-limit", type=int, default=0)
    ap.add_argument("--lp", type=float, default=1.0, help="beam length penalty")
    ap.add_argument("--model", default=C.A_ID)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out = C.RESULTS / (a.out or f"pool_{a.split}_{a.mode}{a.k}.jsonl")
    if out.exists() and not a.force:
        print(f"{out.name} exists; --force to redo")
        return

    scan = a.scan_limit or ((6 * a.n) if a.split == "train" else 0)
    rows = load_split_rows(a.split, a.n, scan)
    print(f"{len(rows)} rows from split={a.split} (md5 order)")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, dtype=torch.float16,
                                                 low_cpu_mem_usage=True)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter, torch_dtype=torch.float16)
    model = model.to("cuda").eval()

    # which hypotheses to build
    if a.mode == "greedy":
        plans = [dict(num_beams=1, num_return_sequences=1, tag="greedy")]
    elif a.mode == "beam":
        plans = [dict(num_beams=a.k, num_return_sequences=a.k, length_penalty=a.lp, tag=f"beam{a.k}")]
    elif a.mode == "sample":
        plans = [dict(do_sample=True, temperature=a.temp, top_p=0.95, num_return_sequences=a.k,
                      tag=f"sample{a.k}")]
    else:
        plans = [dict(num_beams=a.k, num_return_sequences=a.k, length_penalty=a.lp, tag=f"beam{a.k}"),
                 dict(do_sample=True, temperature=a.temp, top_p=0.95, num_return_sequences=a.k,
                      tag=f"sample{a.k}")]

    t0 = time.time()
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for s in range(0, len(rows), a.batch):
            chunk = rows[s:s + a.batch]
            enc = tok([r["vi"] for r in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to("cuda")
            pool: list[list[dict]] = [[] for _ in chunk]
            for plan in plans:
                tag = plan.pop("tag")
                gen = model.generate(**enc, max_new_tokens=a.max_new, **plan,
                                     return_dict_in_generate=True)
                seqs = gen.sequences
                # sequences_scores = summed log-prob of the generated tokens; normalise by length.
                # (compute_transition_scores materialises batch x len x vocab and OOMs on 4 GB.)
                raw = getattr(gen, "sequences_scores", None)
                if raw is None:
                    lp_norm = [float("nan")] * seqs.shape[0]
                else:
                    raw = raw.float().cpu()
                    gen_len = (seqs != tok.pad_token_id).sum(1).clamp(min=1).float().cpu()
                    lp_norm = (raw / gen_len).tolist()
                nr = plan["num_return_sequences"]
                texts = tok.batch_decode(seqs, skip_special_tokens=True)
                for bi, r in enumerate(chunk):
                    for j in range(nr):
                        k = bi * nr + j
                        pool[bi].append({"src": tag, "hyp": texts[k], "lp": float(lp_norm[k])})
                plan["tag"] = tag
            for r, cands in zip(chunk, pool):
                f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["en"],
                                    "core": r["core"], "cands": cands}, ensure_ascii=False) + "\n")
            done = min(s + a.batch, len(rows))
            if (s // a.batch) % 5 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(rows)}  {done / el:.2f} sent/s  "
                      f"eta {(len(rows) - done) / max(done / el, 1e-9) / 60:.1f} min", flush=True)
    tmp.replace(out)
    print(f"wrote {out.name} ({len(rows)} rows) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
