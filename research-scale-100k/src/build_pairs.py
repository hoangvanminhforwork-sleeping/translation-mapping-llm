"""M3 data: 100k EN<->VI parallel sentence pairs, deterministic sample of the 14 shards.

Idempotent: if data/pairs_100k.parquet exists and its manifest hash matches, nothing is written.

The corpus stores each row as translation.{source,target} with the direction varying per row,
so the Vietnamese side is detected by counting Vietnamese-only characters (diacritics + đ).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CORPUS = PROJECT / "data" / "data" / "train-*.parquet"
OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = OUT / "pairs_100k.parquet"
MANIFEST = OUT / "pairs_manifest.json"

N_TOTAL = 100_000
SEED = 42

VI_CHARS = set("ăâđêôơưĂÂĐÊÔƠƯ")
COMBINING = re.compile(r"[\u0300-\u036f]")


def vi_score(s: str) -> int:
    """Count Vietnamese-specific signals: đ/ă/â/ê/ô/ơ/ư and any combining tone mark."""
    if not s:
        return 0
    n = sum(1 for ch in s if ch in VI_CHARS)
    return n + len(COMBINING.findall(unicodedata.normalize("NFD", s)))


def build(force: bool = False) -> dict:
    if PAIRS.exists() and MANIFEST.exists() and not force:
        print(f"{PAIRS.name} exists, reusing (see {MANIFEST.name})")
        return json.loads(MANIFEST.read_text())

    con = duckdb.connect()
    # One row per pair, deduped, length-filtered, then a deterministic sample.
    rows = con.sql(
        f"""
        WITH raw AS (
            SELECT translation.source AS a, translation.target AS b
            FROM read_parquet('{CORPUS.as_posix()}')
            WHERE translation.source IS NOT NULL AND translation.target IS NOT NULL
        ), clean AS (
            SELECT a, b FROM raw
            WHERE length(a) BETWEEN 20 AND 300 AND length(b) BETWEEN 20 AND 300
            QUALIFY row_number() OVER (PARTITION BY a ORDER BY b) = 1
        )
        SELECT a, b FROM clean USING SAMPLE {N_TOTAL} ROWS (reservoir, {SEED})
        """
    ).fetchall()

    pairs = []
    skipped = 0
    for a, b in rows:
        if vi_score(a) > vi_score(b):
            vi, en = a, b
        elif vi_score(b) > vi_score(a):
            vi, en = b, a
        else:
            skipped += 1
            continue
        pairs.append((vi.strip(), en.strip()))

    n = len(pairs)
    # 98 / 1 / 1 split, fixed by position so the split is reproducible without a RNG.
    idx = list(range(n))
    test = {i for i in idx[:1000]}
    val = {i for i in idx[1000:2000]}
    split = ["test" if i in test else "val" if i in val else "train" for i in idx]

    tbl = pa.table(
        {
            "id": pa.array(list(range(n)), pa.int32()),
            "vi": pa.array([p[0] for p in pairs], pa.string()),
            "en": pa.array([p[1] for p in pairs], pa.string()),
            "split": pa.array(split, pa.string()),
        }
    )
    pq.write_table(tbl, PAIRS, compression="zstd")

    h = hashlib.sha256(PAIRS.read_bytes()).hexdigest()
    meta = {
        "file": PAIRS.name,
        "rows": n,
        "skipped_equal_vi_score": skipped,
        "n_total_requested": N_TOTAL,
        "seed": SEED,
        "split": {"train": n - 2000, "val": 1000, "test": 1000},
        "len_chars": {"min": 20, "max": 300},
        "corpus": str(CORPUS.relative_to(PROJECT)),
        "sha256": h,
        "built_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    MANIFEST.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    build()
