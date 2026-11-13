"""Diagnose the three generation runs: are they translations, copies, or degenerate loops?

A BLEU number alone cannot tell you this, and in this pipeline the difference is the whole result.
"""

import json
import re
import unicodedata

import pyarrow.parquet as pq

from src import config as C

VI = set("ăâđêôơưĂÂĐÊÔƠƯ")
TONE = re.compile(r"[\u0300-\u036f]")


def vi_ratio(s: str) -> float:
    """Share of words carrying Vietnamese diacritics/tones."""
    words = s.split()
    if not words:
        return 0.0
    hit = sum(1 for w in words if any(c in VI for c in w) or TONE.search(unicodedata.normalize("NFD", w)))
    return hit / len(words)


def degeneracy(s: str) -> float:
    words = s.lower().split()
    return 1.0 - (len(set(words)) / len(words)) if words else 1.0


t = pq.read_table(C.PAIRS, columns=["vi", "en", "split"])
vi = [a for a, s in zip(t["vi"].to_pylist(), t["split"].to_pylist()) if s == "test"]
en = [b for b, s in zip(t["en"].to_pylist(), t["split"].to_pylist()) if s == "test"]

print(f"{'run':<16}{'n':>6}{'VI_word_ratio':>15}{'copy_of_input':>15}{'degenerate':>12}{'empty':>7}")
for name in ("mapped", "flan_encoder", "zeroshot_vit5"):
    rows = [json.loads(l) for l in (C.RESULTS / f"gen_{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    hyps = [r["hyp"] for r in rows]
    n = len(hyps)
    vi_r = sum(vi_ratio(h) for h in hyps) / n
    copy = sum(h.strip().lower() == s.strip().lower() for h, s in zip(hyps, vi)) / n
    deg = sum(degeneracy(h) > 0.5 for h in hyps) / n
    empty = sum(1 for h in hyps if not h.strip()) / n
    print(f"{name:<16}{n:>6}{vi_r:>15.3f}{copy:>15.3f}{deg:>12.3f}{empty:>7.3f}")

print("\nreference VI_word_ratio (sanity, English refs):",
      round(sum(vi_ratio(x) for x in en) / len(en), 3))
print("input  VI_word_ratio (the Vietnamese sources):",
      round(sum(vi_ratio(x) for x in vi) / len(vi), 3))
