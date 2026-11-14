"""Stage 3: build structured/facts.db (SQLite) from the CSV + JSONL.

Tables
------
facts(id TEXT PK, lang, subject, relation, object, category, book, page INTEGER, quote)
dishes(id TEXT PK, title, ingredients, directions, link, source, NER)   -- arrays as JSON text
provenance(book TEXT, pages INTEGER, ocr_model TEXT, cost_usd REAL)

Idempotent by construction: DROP TABLE IF EXISTS + CREATE + bulk insert, so running it
twice yields the same file. Costs are attributed per book from the LLM cache tags
(`dish-*`, `facts-<book>-<page>`), so provenance.cost_usd sums to the real spend of the
generating run minus the one-off smoke test.
"""
from __future__ import annotations

import json, pathlib, sqlite3, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

DB_PATH = common.OUT_DIR / "facts.db"
CSV_PATH = common.OUT_DIR / "vn_dishes.csv"
JSONL_PATH = common.OUT_DIR / "culture_facts.jsonl"

DDL = """
DROP TABLE IF EXISTS facts;
DROP TABLE IF EXISTS dishes;
DROP TABLE IF EXISTS provenance;
CREATE TABLE facts (
    id        TEXT PRIMARY KEY,
    lang      TEXT NOT NULL,
    subject   TEXT NOT NULL,
    relation  TEXT NOT NULL,
    object    TEXT NOT NULL,
    category  TEXT NOT NULL,
    book      TEXT NOT NULL,
    page      INTEGER NOT NULL,
    quote     TEXT NOT NULL
);
CREATE TABLE dishes (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    ingredients TEXT NOT NULL,
    directions  TEXT NOT NULL,
    link        TEXT NOT NULL,
    source      TEXT NOT NULL,
    NER         TEXT NOT NULL
);
CREATE TABLE provenance (
    book      TEXT PRIMARY KEY,
    pages     INTEGER NOT NULL,
    ocr_model TEXT NOT NULL,
    cost_usd  REAL NOT NULL
);
CREATE INDEX idx_facts_book_page ON facts(book, page);
CREATE INDEX idx_facts_rel ON facts(relation);
"""


def cost_by_book() -> dict[str, float]:
    out = {b: 0.0 for b in common.BOOKS}
    for p in common.CACHE_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = rec.get("tag", "")
        cost = float((rec.get("usage") or {}).get("cost") or 0.0)
        if tag.startswith("facts-"):
            book = max((b for b in common.BOOKS if tag.startswith(f"facts-{b}-")),
                       key=len, default=None)
        elif tag.startswith("dish-"):
            book = "vn_mon_an_truyen_thong"
        else:
            book = None
        if book:
            out[book] = round(out[book] + cost, 6)
    return out


def main():
    import csv

    rows_dish = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    for r in rows_dish:
        for c in ("ingredients", "directions", "NER"):
            json.loads(r[c])          # fail loudly if a row is not valid JSON
    facts = [json.loads(line) for line in open(JSONL_PATH, encoding="utf-8") if line.strip()]

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    con.executemany(
        "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
        [(f["id"], f["lang"], f["subject"], f["relation"], f["object"],
          f["category"], f["book"], int(f["page"]), f["quote"]) for f in facts])
    con.executemany(
        "INSERT OR REPLACE INTO dishes VALUES (?,?,?,?,?,?,?)",
        [(r["id"], r["title"], r["ingredients"], r["directions"], r["link"],
          r["source"], r["NER"]) for r in rows_dish])

    cb = cost_by_book()
    prov = []
    for book, meta in common.BOOKS.items():
        pages = len(common.load_pages(book))
        prov.append((book, pages,
                     "unknown (pre-existing OCR pass shipped with the inputs; "
                     "not re-run by this agent)", cb.get(book, 0.0)))
    con.executemany("INSERT OR REPLACE INTO provenance VALUES (?,?,?,?)", prov)
    con.commit()

    n_f = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    n_d = con.execute("SELECT COUNT(*) FROM dishes").fetchone()[0]
    n_p = con.execute("SELECT COUNT(*) FROM provenance").fetchone()[0]
    uniq = con.execute("SELECT COUNT(DISTINCT id) FROM facts").fetchone()[0]
    con.close()
    print(f"facts.db: facts={n_f} (unique id {uniq}), dishes={n_d}, provenance={n_p}")
    print("cost attributed per book:", json.dumps(cb, ensure_ascii=False))


if __name__ == "__main__":
    main()
