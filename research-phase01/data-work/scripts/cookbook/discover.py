"""Reproduce the three headline numbers + validation of the CSV.

  python discover.py
prints (ASCII-safe): TOC entries, body recipe blocks, extracted rows, plus
quality stats used in REPORT.md.  Zero API calls.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from segment import build, classify, SRC  # noqa: E402

OUT = Path(__file__).resolve().parents[4] / "research-phase01" / "data-work" / "structured-cookbook"
TOC_END = 400          # front matter region, verified: last TOC line is < 300
TOC_ENTRY = re.compile(r"^(?P<t>.*?\S)[\s.]{1,8}(?P<p>\d{1,4})\s*$")


def main() -> int:
    lines = SRC.read_text(encoding="utf-8").splitlines()

    toc = []
    for i, l in enumerate(lines[:TOC_END]):
        s = l.strip()
        m = TOC_ENTRY.match(s)
        if m and len(m.group("t")) >= 3 and not re.search(r"\d\.\d", s):
            toc.append((i, int(m.group("p")), s))
    pages = [p for _, p, _ in toc]

    ing, dr = classify(lines)
    blocks = build(lines)
    usable = [b for b in blocks if b["n_lines"] >= 3 and b["title"]]

    rows = list(csv.DictReader((OUT / "vn_dishes_cookbook.csv").open(encoding="utf-8")))
    bad = 0
    tot_i = tot_d = 0
    empty_t = 0
    gib = 0
    for r in rows:
        try:
            i_, d_, n_ = (json.loads(r["ingredients"]), json.loads(r["directions"]), json.loads(r["NER"]))
        except json.JSONDecodeError:
            bad += 1
            continue
        if not i_ or not d_:
            bad += 1
        tot_i += len(i_)
        tot_d += len(d_)
        if not r["title"].strip():
            empty_t += 1
        if re.search(r"[\^~`@|\\]|&lt;|&gt;", r["title"]):
            gib += 1

    res = {
        "toc_entries": len(toc),
        "toc_max_page": max(pages),
        "toc_page_span": [min(pages), max(pages)],
        "toc_last_line": toc[-1][0],
        "ingredient_headers": len(ing),
        "directions_headers": len(dr),
        "blocks_total": len(blocks),
        "blocks_usable": len(usable),
        "blocks_with_ing_and_dir": sum(1 for b in blocks if b["ing"] is not None and b["dir"] is not None),
        "blocks_ing_only": sum(1 for b in blocks if b["ing"] is not None and b["dir"] is None),
        "blocks_headerless": sum(1 for b in blocks if b["ing"] is None),
        "first_recipe_line": blocks[0]["start"] + 1,
        "last_recipe_line": blocks[-1]["end"] + 1,
        "rows": len(rows),
        "rows_bad_json_or_empty": bad,
        "rows_empty_title": empty_t,
        "rows_garbled_title": gib,
        "ingredients_total": tot_i,
        "directions_total": tot_d,
        "avg_ingredients": round(tot_i / len(rows), 2),
        "avg_directions": round(tot_d / len(rows), 2),
    }
    fails = json.loads((OUT / "failures.json").read_text(encoding="utf-8"))
    res["failures"] = len(fails["failures"])
    res["skipped"] = len(fails["skipped"])
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
