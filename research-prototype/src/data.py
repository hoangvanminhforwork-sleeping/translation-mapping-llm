"""Build and validate the 100-pair culinary test set.

The JSONL is the authored source of truth; the CSV is a readable, SQL-friendly
derivative. Both are committed. Validation is by code (exe-41), not by eye:
  * exactly 100 rows, ids 1..100
  * every dish name literally present in its own sentence (both languages)
  * no duplicate dish, no duplicate sentence
"""

from __future__ import annotations

import csv
import json

from . import config as C


def load_pairs() -> list[dict]:
    rows = [json.loads(line) for line in C.CULINARY_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: r["id"])
    return rows


def find_span(text: str, needle: str) -> tuple[int, int]:
    """Case-insensitive character span of `needle` in `text`.

    Vietnamese lower() preserves length, so indexes stay valid; the slice is
    verified instead of assumed. English sentences start with a capital, which is
    why a case-sensitive find() is the wrong default here.
    """
    start = text.lower().find(needle.lower())
    if start < 0:
        return (-1, -1)
    if text[start : start + len(needle)].lower() != needle.lower():
        return (-1, -1)
    return (start, start + len(needle))


def validate(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    if len(rows) != 100:
        problems.append(f"expected 100 rows, got {len(rows)}")
    if [r["id"] for r in rows] != list(range(1, 101)):
        problems.append("ids are not exactly 1..100")
    for r in rows:
        if find_span(r["vi"], r["dish_vi"])[0] < 0:
            problems.append(f"id {r['id']}: dish_vi {r['dish_vi']!r} not in vi")
        if find_span(r["en"], r["dish_en"])[0] < 0:
            problems.append(f"id {r['id']}: dish_en {r['dish_en']!r} not in en")
        if not (3 <= len(r["vi"].split()) <= 40) or not (3 <= len(r["en"].split()) <= 40):
            problems.append(f"id {r['id']}: sentence length out of range")
    for key in ("dish_vi", "vi", "en"):
        vals = [r[key] for r in rows]
        if len(set(vals)) != len(vals):
            dup = [v for v in set(vals) if vals.count(v) > 1]
            problems.append(f"duplicate values in {key}: {dup[:3]}")
    return problems


def write_csv(rows: list[dict]) -> None:
    with C.CULINARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "dish_vi", "dish_en", "vi", "en"])
        w.writeheader()
        w.writerows(rows)


def build(force: bool = False) -> list[dict]:
    rows = load_pairs()
    problems = validate(rows)
    if problems:
        raise SystemExit("culinary set invalid:\n  " + "\n  ".join(problems))
    if force or not C.CULINARY_CSV.exists():
        write_csv(rows)
    print(f"OK  {len(rows)} pairs | csv -> {C.CULINARY_CSV.relative_to(C.ROOT)}")
    print(f"    vi chars {sum(len(r['vi']) for r in rows)} | en chars {sum(len(r['en']) for r in rows)}")
    return rows


if __name__ == "__main__":
    build(force=True)
