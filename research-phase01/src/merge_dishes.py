"""Merge the two Vietnamese recipe extractions into one RecipeNLG-schema file.

Two independent sources, kept distinguishable in `source` and `li:
  structured/vn_dishes.csv           167 recipes, OCR'd PDF  "mon_an_vn_truyen_thong"
  structured-cookbook/…cookbook.csv 1235 recipes, text .md   "ky_thuat_nau_an"  (57 % titles rebuilt)

The merge does not deduplicate across books: the same dish can legitimately appear twice with different
ingredients. It only reports title collisions so a reader knows the overlap.

Output: data-work/recipes_vn_all.csv + data-work/recipes_vn_all_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata

import pandas as pd

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
COLS = ["id", "title", "ingredients", "directions", "link", "source", "NER"]


def norm(t: str) -> str:
    t = unicodedata.normalize("NFD", str(t).lower())
    t = re.sub(r"[\u0300-\u036f]", "", t)
    return re.sub(r"[^a-z0-9 ]", "", t).strip()


def main():
    parts = []
    a = C.ROOT / "data-work" / "structured" / "vn_dishes.csv"
    if a.exists():
        d = pd.read_csv(a, encoding="utf-8")
        d["origin"] = "pdf_ocr"
        parts.append(d[COLS + ["origin"]])
    b = C.ROOT / "data-work" / "structured-cookbook" / "vn_dishes_cookbook.csv"
    if b.exists():
        d = pd.read_csv(b, encoding="utf-8")
        d["origin"] = "cookbook_md"
        parts.append(d[COLS + ["origin"]])
    if not parts:
        raise SystemExit("no dish files found")

    df = pd.concat(parts, ignore_index=True)
    df["id"] = [f"vn-{i:05d}" for i in range(1, len(df) + 1)]
    out = C.ROOT / "data-work" / "recipes_vn_all.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    keys = df["title"].map(norm)
    dupes = int(keys.duplicated().sum())
    meta = {
        "file": out.name,
        "rows": int(len(df)),
        "by_origin": df["origin"].value_counts().to_dict(),
        "columns": list(df.columns),
        "duplicate_titles_after_normalisation": dupes,
        "duplicate_examples": sorted(set(df.loc[keys.duplicated(keep=False), "title"]))[:20],
        "schema": "RecipeNLG: id,title,ingredients,directions,link,source,NER (arrays are JSON strings)",
        "caveat": "cookbook_md titles are 57% reconstructed from ingredients/steps where OCR damaged the "
                  "title line; pdf_ocr rows keep the printed title",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (C.ROOT / "data-work" / "recipes_vn_all_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "duplicate_examples"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
