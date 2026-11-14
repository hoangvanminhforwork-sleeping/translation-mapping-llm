"""Pair corpus: 8.27M EN-VI pairs, deduped, minus every row a previous agent touched.

Deterministic without an RNG: rows are ordered by md5(vietnamese), so the test/val split is a
function of the text only, and re-running the builder cannot move a sentence between splits.

Idempotent: if data/pairs.parquet exists with the recorded sha256, nothing is written.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import duckdb

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")

MANIFEST = C.DATA / "pairs_manifest.json"

def _exclusion_sql(con: duckdb.DuckDBPyConnection) -> str:
    """One table of every text (vi or en) used by an earlier experiment."""
    parts = []
    p = C.EXCLUDE["scaled_100k"]
    if p.exists():
        parts.append(f"SELECT vi AS t FROM read_parquet('{p.as_posix()}')")
        parts.append(f"SELECT en AS t FROM read_parquet('{p.as_posix()}')")
    p = C.EXCLUDE["copy_lora_train_50k"]
    if p.exists():
        parts.append(f"SELECT vi AS t FROM read_json_auto('{p.as_posix()}')")
        parts.append(f"SELECT en AS t FROM read_json_auto('{p.as_posix()}')")
    p = C.EXCLUDE["copy_lora_eval_4k"]
    if p.exists():
        parts.append(f"SELECT src AS t FROM read_json_auto('{p.as_posix()}')")
    con.execute("CREATE OR REPLACE TEMP TABLE excluded AS " + " UNION ".join(parts))
    n = con.sql("SELECT count(*) FROM excluded").fetchone()[0]
    print(f"excluded texts from prior experiments: {n:,}")
    return "excluded"


def build(force: bool = False) -> dict:
    if C.PAIRS.exists() and MANIFEST.exists() and not force:
        meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
        h = hashlib.sha256(C.PAIRS.read_bytes()).hexdigest()
        if h == meta.get("sha256"):
            print(f"{C.PAIRS.name} present and hash matches ({meta['rows']:,} rows) — reusing")
            return meta
        print("hash mismatch, rebuilding")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    _exclusion_sql(con)

    # Vietnamese side = the one with more non-ASCII characters (diacritics). Ties are dropped:
    # they are the rows where neither side is recognisably Vietnamese.
    q = f"""
    WITH raw AS (
        SELECT translation.source AS a, translation.target AS b
        FROM read_parquet('{C.CORPUS.as_posix()}')
        WHERE translation.source IS NOT NULL AND translation.target IS NOT NULL
    ), clean AS (
        SELECT a, b,
               length(regexp_replace(a, '[\\x00-\\x7f]', '', 'g')) AS na,
               length(regexp_replace(b, '[\\x00-\\x7f]', '', 'g')) AS nb
        FROM raw
        WHERE length(a) BETWEEN 20 AND 300 AND length(b) BETWEEN 20 AND 300
    ), sided AS (
        SELECT CASE WHEN na > nb THEN a ELSE b END AS vi,
               CASE WHEN na > nb THEN b ELSE a END AS en
        FROM clean
        WHERE na <> nb
    ), dedup AS (
        SELECT vi, en, row_number() OVER (PARTITION BY vi ORDER BY en) AS rn FROM sided
    ), kept AS (
        SELECT vi, en FROM dedup
        WHERE rn = 1
          AND vi NOT IN (SELECT t FROM excluded)
          AND en NOT IN (SELECT t FROM excluded)
    )
    SELECT vi, en, row_number() OVER (ORDER BY md5(vi)) AS pos FROM kept
    """
    print("building pairs (md5-ordered split, no RNG)…")
    con.execute(f"CREATE OR REPLACE TABLE pinned AS {q}")
    total = con.sql("SELECT count(*) FROM pinned").fetchone()[0]
    print(f"pairs after dedup + exclusions: {total:,}")

    con.execute(
        f"""
        COPY (
            SELECT (pos - 1)::INTEGER AS id, vi, en,
                   CASE WHEN pos <= {C.N_TEST} THEN 'test'
                        WHEN pos <= {C.N_TEST + C.N_VAL} THEN 'val'
                        ELSE 'train' END AS split
            FROM pinned
            ORDER BY pos
        ) TO '{C.PAIRS.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """
    )

    counts = dict(con.sql("SELECT split, count(*) FROM read_parquet('" + C.PAIRS.as_posix() + "') GROUP BY 1").fetchall())
    h = hashlib.sha256(C.PAIRS.read_bytes()).hexdigest()
    meta = {
        "file": C.PAIRS.name,
        "rows": total,
        "split": {k: int(v) for k, v in counts.items()},
        "split_rule": f"ORDER BY md5(vi); first {C.N_TEST} = test, next {C.N_VAL} = val, rest = train",
        "vi_side_rule": "higher count of non-ASCII chars; ties dropped",
        "len_chars": [20, 300],
        "corpus": str(C.CORPUS.relative_to(C.PROJECT)),
        "excluded_from": {k: str(v.relative_to(C.PROJECT)) for k, v in C.EXCLUDE.items()},
        "sha256": h,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    build(force="--force" in sys.argv)
