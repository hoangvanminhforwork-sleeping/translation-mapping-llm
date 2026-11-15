"""Pair corpus with a per-shard tag: 14 train shards -> one parquet, deduped per (file, vi).

Why per-file dedup (not global): the mission asks which shard is the most representative, so each
shard must keep its own content. Global dedup would hand every duplicate to whichever file DuckDB
read first and bias the comparison.

Deterministic without an RNG: `id` is `row_number() OVER (ORDER BY md5(vi), file)`, so re-running
cannot move a sentence between splits.

Idempotent: if data/pairs.parquet exists and its sha256 matches the manifest, nothing is written.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import duckdb

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")


def _exclusion_sql(con) -> str:
    parts = []
    p = C.EXCLUDE["scaled_100k"]
    if p.exists():
        parts += [f"SELECT vi AS t FROM read_parquet('{p.as_posix()}')",
                  f"SELECT en AS t FROM read_parquet('{p.as_posix()}')"]
    p = C.EXCLUDE["copy_lora_train_50k"]
    if p.exists():
        parts += [f"SELECT vi AS t FROM read_json_auto('{p.as_posix()}')",
                  f"SELECT en AS t FROM read_json_auto('{p.as_posix()}')"]
    p = C.EXCLUDE["copy_lora_eval_4k"]
    if p.exists():
        parts.append(f"SELECT src AS t FROM read_json_auto('{p.as_posix()}')")
    con.execute("CREATE OR REPLACE TEMP TABLE excluded AS " + " UNION ".join(parts))
    print(f"excluded texts from prior experiments: {con.sql('SELECT count(*) FROM excluded').fetchone()[0]:,}")
    return "excluded"


def build(force: bool = False) -> dict:
    if C.PAIRS.exists() and C.PAIRS_MANIFEST.exists() and not force:
        meta = json.loads(C.PAIRS_MANIFEST.read_text(encoding="utf-8"))
        if hashlib.sha256(C.PAIRS.read_bytes()).hexdigest() == meta.get("sha256"):
            print(f"{C.PAIRS.name} present and hash matches ({meta['rows']:,} rows) — reusing")
            return meta
        print("hash mismatch, rebuilding")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{(C.WORK / 'duckdb_tmp').as_posix()}'")
    _exclusion_sql(con)

    # vi side = more non-ASCII chars (diacritics); ties dropped (neither side looks Vietnamese).
    q = f"""
    WITH raw AS (
        SELECT regexp_extract(filename, 'train-[0-9]+-of-[0-9]+') AS file,
               translation.source AS a, translation.target AS b
        FROM read_parquet('{C.CORPUS.as_posix()}', filename=true)
        WHERE translation.source IS NOT NULL AND translation.target IS NOT NULL
    ), clean AS (
        SELECT file, a, b,
               length(regexp_replace(a, '[\\x00-\\x7f]', '', 'g')) AS na,
               length(regexp_replace(b, '[\\x00-\\x7f]', '', 'g')) AS nb
        FROM raw
        WHERE length(a) BETWEEN 20 AND 300 AND length(b) BETWEEN 20 AND 300
    ), sided AS (
        SELECT file, CASE WHEN na > nb THEN a ELSE b END AS vi,
                      CASE WHEN na > nb THEN b ELSE a END AS en
        FROM clean WHERE na <> nb
    ), dedup AS (
        SELECT file, vi, en, row_number() OVER (PARTITION BY file, vi ORDER BY en) AS rn FROM sided
    ), kept AS (
        SELECT file, vi, en FROM dedup
        WHERE rn = 1
          AND vi NOT IN (SELECT t FROM excluded)
          AND en NOT IN (SELECT t FROM excluded)
    )
    SELECT row_number() OVER (ORDER BY md5(vi), file) - 1 AS id, file, vi, en FROM kept
    """
    print("building pairs (per-file dedup, md5-ordered ids)…")
    con.execute(f"CREATE OR REPLACE TABLE pinned AS {q}")
    total = con.sql("SELECT count(*) FROM pinned").fetchone()[0]
    print(f"pairs: {total:,}")

    con.execute(f"""
        COPY (SELECT id::INTEGER AS id, file, vi, en FROM pinned ORDER BY id)
        TO '{C.PAIRS.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    per_file = con.sql(
        f"SELECT file, count(*) n FROM read_parquet('{C.PAIRS.as_posix()}') GROUP BY 1 ORDER BY 1").fetchall()
    h = hashlib.sha256(C.PAIRS.read_bytes()).hexdigest()
    meta = {
        "file": C.PAIRS.name,
        "rows": total,
        "per_file": {f: int(n) for f, n in per_file},
        "dedup": "per (file, vi)",
        "id_rule": "row_number() OVER (ORDER BY md5(vi), file)",
        "vi_side_rule": "higher count of non-ASCII chars; ties dropped",
        "len_chars": [20, 300],
        "corpus": str(C.CORPUS.relative_to(C.PROJECT)),
        "excluded_from": {k: str(v.relative_to(C.PROJECT)) for k, v in C.EXCLUDE.items()},
        "lora_file": C.LORA_FILE,
        "sha256": h,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    C.PAIRS_MANIFEST.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_file"}, indent=2))
    return meta


if __name__ == "__main__":
    build(force="--force" in sys.argv)
