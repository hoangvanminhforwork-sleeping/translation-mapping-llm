"""Diagnose where the 8.33M -> 3.46M drop happens in build_pairs."""
import sys

import duckdb

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
con = duckdb.connect()
con.execute("PRAGMA threads=8")
src = C.CORPUS.as_posix()
def n(sql):
    return con.sql(sql).fetchone()[0]

print("rows total          ", f"{n(f'''SELECT count(*) FROM read_parquet('{src}')'''):,}")
print("translation non-null", f"{n(f'''SELECT count(*) FROM read_parquet('{src}') WHERE translation.source IS NOT NULL AND translation.target IS NOT NULL'''):,}")
print("len 20-300          ", f"{n(f'''SELECT count(*) FROM read_parquet('{src}') WHERE translation.source IS NOT NULL AND length(translation.source) BETWEEN 20 AND 300 AND length(translation.target) BETWEEN 20 AND 300'''):,}")
print("len 10-400          ", f"{n(f'''SELECT count(*) FROM read_parquet('{src}') WHERE translation.source IS NOT NULL AND length(translation.source) BETWEEN 10 AND 400 AND length(translation.target) BETWEEN 10 AND 400'''):,}")
print("distinct vi (20-300)", f"{n(f'''SELECT count(DISTINCT CASE WHEN length(regexp_replace(translation.source, '[\\x00-\\x7f]', '', 'g')) >= length(regexp_replace(translation.target, '[\\x00-\\x7f]', '', 'g')) THEN translation.source ELSE translation.target END) FROM read_parquet('{src}') WHERE length(translation.source) BETWEEN 20 AND 300 AND length(translation.target) BETWEEN 20 AND 300'''):,}")
print("equal non-ascii cnt ", f"{n(f'''SELECT count(*) FROM read_parquet('{src}') WHERE length(translation.source) BETWEEN 20 AND 300 AND length(translation.target) BETWEEN 20 AND 300 AND length(regexp_replace(translation.source, '[\\x00-\\x7f]', '', 'g')) = length(regexp_replace(translation.target, '[\\x00-\\x7f]', '', 'g'))'''):,}")
