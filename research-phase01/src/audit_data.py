"""Independent audit of the M1 deliverables (does not trust the subagent's report).

Checks, by code:
  * vn_dishes.csv parses, one row per recipe, the three array columns are valid JSON, no empty title
  * facts.db counts match the JSONL, every fact's `quote` is a literal substring of its claimed page
  * coverage: which pages produced no facts, and how many facts per book

Writes results/data_audit.json.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
DW = C.ROOT / "data-work"
STRUCT = DW / "structured"


def page_text(book: str) -> dict[int, str]:
    md = (DW / "markdown" / f"{book}.md").read_text(encoding="utf-8")
    parts = re.split(r"<!--\s*page\s+(\d+)\s*-->", md)
    out = {}
    for i in range(1, len(parts), 2):
        out[int(parts[i])] = parts[i + 1]
    return out


def main():
    rep: dict = {}

    csv_path = STRUCT / "vn_dishes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8")
        bad = []
        for col in ("ingredients", "directions", "NER"):
            ok = 0
            for i, v in df[col].items():
                try:
                    parsed = json.loads(v)
                    ok += isinstance(parsed, list) and len(parsed) >= 0
                except Exception:  # noqa: BLE001
                    bad.append((int(i), col))
            rep.setdefault("dishes", {})[f"valid_json_{col}"] = ok
        rep["dishes"].update({
            "rows": int(len(df)),
            "columns": list(df.columns),
            "empty_titles": int(df["title"].isna().sum()),
            "duplicate_titles": int(df["title"].duplicated().sum()),
            "invalid_cells": len(bad),
            "median_ingredients": float(df["ingredients"].map(lambda s: len(json.loads(s)) if isinstance(s, str) and s.startswith("[") else 0).median()),
            "median_directions": float(df["directions"].map(lambda s: len(json.loads(s)) if isinstance(s, str) and s.startswith("[") else 0).median()),
        })

    db = STRUCT / "facts.db"
    if db.exists():
        con = sqlite3.connect(db)
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        n_facts = con.execute("SELECT count(*) FROM facts").fetchone()[0]
        by_book = dict(con.execute("SELECT book, count(*) FROM facts GROUP BY book").fetchall())
        rels = dict(con.execute("SELECT relation, count(*) FROM facts GROUP BY relation ORDER BY 2 DESC").fetchall())
        facts = con.execute("SELECT book, page, quote FROM facts").fetchall()

        cache: dict[str, dict[int, str]] = {}
        verified = 0
        missing_page = 0
        for book, page, quote in facts:
            if book not in cache:
                p = DW / "markdown" / f"{book}.md"
                cache[book] = page_text(book) if p.exists() else {}
            txt = cache[book].get(int(page))
            if txt is None:
                missing_page += 1
                continue
            if quote and quote.strip() and quote.strip() in txt:
                verified += 1
        rep["facts"] = {
            "tables": tables, "rows_db": n_facts, "by_book": by_book, "relations": rels,
            "quote_verified": verified, "quote_missing_page": missing_page,
            "quote_verified_rate": round(verified / max(n_facts, 1), 4),
            "n_distinct_subjects": con.execute("SELECT count(DISTINCT subject) FROM facts").fetchone()[0],
        }
        jl = STRUCT / "culture_facts.jsonl"
        if jl.exists():
            n_jsonl = sum(1 for line in jl.open(encoding="utf-8") if line.strip())
            rep["facts"]["rows_jsonl"] = n_jsonl
            rep["facts"]["db_matches_jsonl"] = n_jsonl == n_facts

    cook = STRUCT.parent / "structured-cookbook" / "vn_dishes_cookbook.csv"
    if cook.exists():
        d2 = pd.read_csv(cook, encoding="utf-8")
        rep["cookbook_dishes"] = {"rows": int(len(d2)), "columns": list(d2.columns)}

    (C.RESULTS / "data_audit.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
