"""Probe: extract TOC entries and body headings from vn_mon_an_truyen_thong.md."""
import re, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
MD = ROOT / "research-phase01" / "data-work" / "markdown" / "vn_mon_an_truyen_thong.md"
OUT = ROOT / "agent-work" / "structurer"

text = MD.read_text(encoding="utf-8")
parts = re.split(r"<!-- page (\d+) -->", text)
pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}

# --- TOC pages 1..7
toc = []
for pg in range(1, 8):
    for line in pages[pg].splitlines():
        m = re.match(r"^(.+?)\.{3,}\s*(\d{1,3})\s*$", line.strip())
        if m:
            toc.append({"toc_title": m.group(1).strip(), "page": int(m.group(2))})

# --- body headings
heads = []
for pg in sorted(pages):
    if pg <= 7:
        continue
    for line in pages[pg].splitlines():
        s = line.strip()
        if s.startswith("#"):
            heads.append({"page": pg, "level": len(s) - len(s.lstrip("#")), "title": s.lstrip("# ").strip()})

(OUT / "toc.json").write_text(json.dumps({"toc": toc, "headings": heads}, ensure_ascii=False, indent=1), encoding="utf-8")
print("toc entries:", len(toc))
print("heading count:", len(heads))
print("toc pages set:", sorted({t["page"] for t in toc})[:10], "...")
print("head pages set:", sorted({h["page"] for h in heads})[:10], "...")
print("match:", len(set(t["page"] for t in toc) & set(h["page"] for h in heads)))
