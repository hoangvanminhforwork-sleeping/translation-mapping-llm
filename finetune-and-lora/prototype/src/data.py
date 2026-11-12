"""Build the 50k copy-training set and the 2k fixed evaluation set.

Two rules that matter for the comparison to be honest:
1. The eval sentences are the SAME 2,000 the baseline failed on (read from the
   baseline CSVs), so "before" and "after" are paired row by row.
2. Training text comes from the train shard, the eval text from the test split.
   No sentence appears in both.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import unicodedata

import duckdb

from . import config as C


def _source_is_vi(prompt: str) -> bool:
    p = prompt.lower()
    if "into vietnamese" in p or "vietnamese translation" in p or "sang tiếng việt" in p:
        return False
    if "into english" in p or "english translation" in p or "sang tiếng anh" in p:
        return True
    raise ValueError(f"unknown template: {prompt[:80]!r}")


def _has_diacritics(s: str) -> bool:
    return any(ch in "ăâđêôơưĂÂĐÊÔƠƯ" or unicodedata.combining(ch) for ch in s)


def build_train(n: int = C.N_TRAIN, force: bool = False) -> dict:
    out = C.DATA / f"train_copy_{n}.jsonl"
    if out.exists() and not force:
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"train set cached: {len(rows)} rows -> {out.name}")
    else:
        con = duckdb.connect()
        pairs = con.sql(
            f"""SELECT prompt, translation.source AS src, translation.target AS tgt
                FROM read_parquet('{C.TRAIN_PARQUET.as_posix()}')"""
        ).fetchall()
        con.close()

        clean = []
        for prompt, src, tgt in pairs:
            vi, en = (src, tgt) if _source_is_vi(prompt) else (tgt, src)
            if not _has_diacritics(vi) and _has_diacritics(en):
                vi, en = en, vi
            if not vi or not en:
                continue
            clean.append({"vi": vi, "en": en})

        rng = random.Random(C.SEED)
        picked = rng.sample(clean, n)
        rows = [{"id": i, **p} for i, p in enumerate(picked)]
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"train set built: {len(rows)} pairs -> {out.name}")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "train_rows": len(rows),
        "seed": C.SEED,
        "source": str(C.TRAIN_PARQUET),
        "sha256": sha,
    }
    (C.DATA / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"sha256 {sha[:16]}...  pairs: {len(rows)}")
    return manifest


def build_eval(force: bool = False) -> dict:
    """The exact 2,000 baseline sentences, copied into this project for provenance."""
    out = C.DATA / "eval_sources.jsonl"
    if out.exists() and not force:
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"eval set cached: {len(rows)} rows")
        return {"eval_rows": len(rows)}

    rows = []
    shas = {}
    for lang, path in C.EVAL_RAW.items():
        if not path.exists():
            raise SystemExit(f"baseline generations missing: {path}\nrun research-prototype copy stage first")
        shas[lang] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"id": int(r["id"]), "lang": lang, "src": r["src"], "baseline_hyp": r["hyp"]})

    rows.sort(key=lambda r: (r["lang"], r["id"]))
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {"eval_rows": len(rows), "per_lang": len(rows) // 2, "source_sha256": shas}
    (C.DATA / "eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"eval set built: {len(rows)} rows (same sentences as the baseline)")
    return manifest