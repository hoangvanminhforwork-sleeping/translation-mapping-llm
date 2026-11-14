"""Extract recipes from the OCR'd cookbook into RecipeNLG schema rows.

Usage
  python extract.py                 # full run, resumable, idempotent
  python extract.py --force         # ignore cache, re-call the API
  python extract.py --limit 20      # smoke test on the first 20 blocks
  python extract.py --report        # rebuild REPORT.md from cache only (no API)

Every response is cached at structured-cookbook/cache/<sha1>.json, so a rerun with
all caches present performs zero API calls and writes no byte that differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from segment import build, SRC  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "research-phase01" / "data-work" / "structured-cookbook"
CACHE = OUTDIR / "cache"
ENV_PATH = ROOT / "research-phase01" / ".env"
CSV_PATH = OUTDIR / "vn_dishes_cookbook.csv"
COST_PATH = OUTDIR / "cost.json"
FAIL_PATH = OUTDIR / "failures.json"
MANIFEST = OUTDIR / "manifest.json"

MODEL = "qwen/qwen3.7-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PRICE_IN = 0.03    # USD / 1M tokens
PRICE_OUT = 0.13
BUDGET = 0.50
BUDGET_STOP = 0.45
PROMPT_VER = "v1"
MAX_CHARS = 14000  # keep one request bounded; only 1 block on the whole book exceeds this

SOURCE = "ky_thuat_nau_an"
LINK_BASE = "research-phase01/data/Vietnamese/Kỹ thuật nấu ăn toàn tập.md#L"

PROMPT = """Bạn là bộ trích xuất dữ liệu công thức nấu ăn từ sách scan tiếng Việt (OCR nhiều lỗi).
Đoạn dưới đây là MỘT mục của sách. Hãy trả về DUY NHẤT một JSON object, không markdown, không giải thích:

{"is_recipe": true, "title": "...", "ingredients": ["..."], "directions": ["..."], "NER": ["..."]}

Quy tắc:
1. is_recipe = false nếu đoạn văn KHÔNG phải công thức nấu ăn (lý thuyết, mục lục, danh sách tên món, trang trí). Khi đó các mảng để rỗng.
2. title: tên món ăn, lấy từ dòng tiêu đề ở đầu đoạn. Nếu dòng tiêu đề bị OCR thành chuỗi vô nghĩa, hãy viết lại tên món ngắn gọn dựa trên nguyên liệu và cách làm có trong đoạn; nếu không đủ căn cứ thì để "".
3. ingredients: MỖI dòng nguyên liệu in trong sách là MỘT phần tử, giữ nguyên tiếng Việt, gồm cả định lượng nếu có. Không gộp nhiều dòng, không tách một dòng, không thêm nguyên liệu không có.
4. directions: MỖI bước in trong sách là MỘT phần tử, đúng thứ tự. Giữ nguyên văn (chỉ sửa lỗi chính tả OCR hiển nhiên). Không thêm bước không có.
5. NER: 2-4 thực thể món ăn hoặc nguyên liệu chính XUẤT HIỆN TRONG TITLE.
6. KHÔNG dịch sang ngôn ngữ khác. KHÔNG bịa.
7. Bỏ các dòng tiêu đề như "Nguyên liệu", "Cách làm", "Thực hiện" và số trang.

VĂN BẢN:
"""

REPAIR = (
    "Lần trả lời trước không phải JSON hợp lệ theo schema. Hãy trả lại CHỈ một JSON object dạng:\n"
    '{{"is_recipe": true, "title": "...", "ingredients": ["..."], "directions": ["..."], "NER": ["..."]}}\n'
    "Không markdown, không giải thích. Nếu đoạn văn không phải công thức thì is_recipe=false và các mảng rỗng.\n\n"
    "VĂN BẢN GỐC:\n{src}\n\nTRẢ LỜI CŨ (hỏng):\n{bad}\n"
)

# used only when the model said "not a recipe" but the block structurally is one
# (ingredient header + >=3 ingredient lines + >=2 direction lines)
STRONG = (
    "Đoạn văn dưới đây CHẮC CHẮN là một công thức nấu ăn: nó có mục nguyên liệu và mục cách làm. "
    "Dòng tiêu đề có thể bị OCR hỏng — hãy đặt tên món ngắn gọn dựa trên nguyên liệu và cách làm.\n"
    "Trả về DUY NHẤT JSON: "
    '{{"is_recipe": true, "title": "...", "ingredients": ["..."], "directions": ["..."], "NER": ["..."]}}\n'
    "Mỗi dòng nguyên liệu = 1 phần tử; mỗi bước = 1 phần tử; không dịch; không bịa.\n\nVĂN BẢN:\n{src}\n"
)


def load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() == "DataWorker":
                return v.strip().strip('"').strip("'").strip()
    raise RuntimeError("DataWorker not found in research-phase01/.env")


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def call_api(key: str, prompt: str, max_tokens: int = 2600) -> dict:
    import requests

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    }
    r = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"},
                      json=payload, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    ch = d["choices"][0]
    text = (ch["message"].get("content") or "").strip()
    u = d.get("usage") or {}
    pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    return {"text": text, "prompt_tokens": pt, "completion_tokens": ct,
            "usd": (pt * PRICE_IN + ct * PRICE_OUT) / 1e6,
            "finish_reason": ch.get("finish_reason"), "model": d.get("model")}


def call_retry(key: str, prompt: str, tries: int = 5, **kw) -> dict:
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            return call_api(key, prompt, **kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if attempt == tries or not any(s in msg for s in
                                           ("429", "HTTP 5", "timed out", "Timeout",
                                            "Connection", "Remote end closed", "502", "503")):
                raise
            time.sleep(delay + random.uniform(0, 1.2))
            delay = min(delay * 1.8, 30)


def parse_payload(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for k in ("ingredients", "directions", "NER"):
        v = obj.get(k, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return None
        obj[k] = [x.strip() for x in v if x and x.strip()]
    if not isinstance(obj.get("title", ""), str):
        return None
    obj["title"] = (obj.get("title") or "").strip()
    obj["is_recipe"] = bool(obj.get("is_recipe", True))
    return obj


def phase(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    blocks = build(lines)
    blocks = [b for b in blocks if b["n_lines"] >= 3 and b["title"]]
    if args.limit:
        blocks = blocks[:args.limit]
    phase(f"blocks: {len(blocks)}")

    jobs = []
    for b in blocks:
        text = b["text"][:MAX_CHARS]
        prompt = PROMPT + text
        jobs.append({"block": b, "prompt": prompt, "key": sha(f"{PROMPT_VER}|{MODEL}|{text}")})

    def is_struct_recipe(j: dict) -> bool:
        b = j["block"]
        if b["ing"] is not None and b["n_ing_lines"] >= 3 and b["n_dir_lines"] >= 2:
            return True
        # header-less dessert/drink recipes: bullets + real quantities, short section
        t = b["text"].splitlines()
        bullets = sum(1 for x in t if x[:1] in "-•■*+♦")
        quant = re.search(r"\d+\s*(?:g|kg|gr|gam|lít|ml|cl|l|muỗng|xúp|chén|bát|"
                          r"quả|trứng|củ|tép|con|trái)\b", b["text"])
        return bullets >= 4 and bool(quant) and b["n_lines"] <= 120

    # --- cache pass -------------------------------------------------------
    todo = []
    for j in jobs:
        cp = CACHE / f"{j['key']}.json"
        if cp.exists() and not args.force:
            try:
                rec = json.loads(cp.read_text(encoding="utf-8"))
                j["cached"] = rec
                if (not rec.get("strong_fix") and rec.get("parse")
                        and not rec["parse"].get("is_recipe") and is_struct_recipe(j)):
                    j["strong_only"] = True
                    todo.append(j)
                continue
            except json.JSONDecodeError:
                pass
        todo.append(j)
    phase(f"cache hits: {len(jobs) - len(todo)}  to call: {len(todo)} "
          f"(strong_only: {sum(1 for j in todo if j.get('strong_only'))})")

    cost = {"model": MODEL, "price_in_usd_per_m": PRICE_IN, "price_out_usd_per_m": PRICE_OUT,
            "cumulative_usd": 0.0, "limit_usd": BUDGET, "calls": 0, "cached": len(jobs) - len(todo)}
    if COST_PATH.exists() and not args.force:
        try:
            old = json.loads(COST_PATH.read_text(encoding="utf-8"))
            cost["cumulative_usd"] = float(old.get("cumulative_usd", 0.0))
            cost["calls"] = int(old.get("calls", 0))
        except Exception:  # noqa: BLE001
            pass

    lock = threading.Lock()
    key = None
    if todo:
        key = load_key()

    stop = threading.Event()

    def work(j: dict) -> tuple[dict, dict | None, Exception | None]:
        if stop.is_set():
            return j, None, RuntimeError("budget stop")
        try:
            if j.get("strong_only"):
                b = j["block"]
                res = call_retry(key, STRONG.format(src=b["text"][:MAX_CHARS]))
                obj = parse_payload(res["text"])
                rec = j["cached"]
                u0 = rec.get("usage") or {}
                merged = {"text": u0.get("text", ""),
                          "prompt_tokens": u0.get("prompt_tokens", 0) + res["prompt_tokens"],
                          "completion_tokens": u0.get("completion_tokens", 0) + res["completion_tokens"],
                          "usd": u0.get("usd", 0.0) + res["usd"],
                          "model": res.get("model"), "finish_reason": res.get("finish_reason")}
                if obj is not None and obj.get("is_recipe"):
                    rec["parse"] = obj
                rec["usage"] = merged
                rec["strong_fix"] = True
                with lock:
                    (CACHE / f"{j['key']}.json").write_text(
                        json.dumps(rec, ensure_ascii=False, indent=0), encoding="utf-8")
                    cost["cumulative_usd"] += res["usd"]
                    cost["calls"] += 1
                    if cost["cumulative_usd"] > BUDGET_STOP:
                        stop.set()
                return j, (rec, res), None
            res = call_retry(key, j["prompt"])
            obj = parse_payload(res["text"])
            repaired = False
            if obj is None:  # one repair attempt
                rp = REPAIR.format(src=j["block"]["text"][:MAX_CHARS],
                                   bad=res["text"][:1500])
                res2 = call_retry(key, rp)
                obj = parse_payload(res2["text"])
                res = {"text": res["text"], "prompt_tokens": res["prompt_tokens"] + res2["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"] + res2["completion_tokens"],
                       "usd": res["usd"] + res2["usd"], "model": res2.get("model"),
                       "finish_reason": res2.get("finish_reason")}
                repaired = True
            # structural override: the block has an ingredient header, >=3 ingredient
            # lines and >=2 direction lines, so a "not a recipe" verdict is a model slip
            b = j["block"]
            if (obj is not None and not obj.get("is_recipe")
                    and b["ing"] is not None and b["n_ing_lines"] >= 3 and b["n_dir_lines"] >= 2):
                res3 = call_retry(key, STRONG.format(src=b["text"][:MAX_CHARS]))
                obj3 = parse_payload(res3["text"])
                res = {"text": res["text"], "prompt_tokens": res["prompt_tokens"] + res3["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"] + res3["completion_tokens"],
                       "usd": res["usd"] + res3["usd"], "model": res3.get("model"),
                       "finish_reason": res3.get("finish_reason")}
                if obj3 is not None and obj3.get("is_recipe"):
                    obj = obj3
                    repaired = True
            rec = {"block_start": j["block"]["start"], "raw_title": j["block"]["title"],
                   "parse": obj, "repaired": repaired, "usage": res,
                   "strong_fix": bool(is_struct_recipe(j)), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with lock:
                (CACHE / f"{j['key']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=0),
                                                        encoding="utf-8")
                cost["cumulative_usd"] += res["usd"]
                cost["calls"] += 1
                if cost["cumulative_usd"] > BUDGET_STOP:
                    stop.set()
            return j, (rec, res), None
        except Exception as e:  # noqa: BLE001
            return j, None, e

    t0 = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, j) for j in todo]
            done = 0
            for f in as_completed(futs):
                j, rec, err = f.result()
                done += 1
                if err:
                    with lock:
                        cost.setdefault("errors", []).append({"start": j["block"]["start"], "err": str(err)[:200]})
                if done % 25 == 0 or done == len(todo):
                    phase(f"{done}/{len(todo)} calls  ${cost['cumulative_usd']:.4f}  "
                          f"{time.time() - t0:.0f}s")

    cost["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    COST_PATH.write_text(json.dumps(cost, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- assemble ---------------------------------------------------------
    rows, failures, skipped = [], [], []
    for j in jobs:
        cp = CACHE / f"{j['key']}.json"
        if not cp.exists():
            failures.append({"start": j["block"]["start"], "raw_title": j["block"]["title"],
                             "reason": "no_cache_entry (API error or budget stop)"})
            continue
        rec = json.loads(cp.read_text(encoding="utf-8"))
        obj = rec.get("parse")
        if not obj:
            failures.append({"start": j["block"]["start"], "raw_title": j["block"]["title"],
                             "reason": "invalid_json_after_repair", "raw": (rec.get("usage") or {}).get("text", "")[:400]})
            continue
        if not obj.get("is_recipe"):
            skipped.append({"start": j["block"]["start"], "raw_title": j["block"]["title"],
                            "reason": "model_is_recipe=false"})
            continue
        if not obj["ingredients"] or not obj["directions"]:
            skipped.append({"start": j["block"]["start"], "raw_title": j["block"]["title"],
                            "reason": f"empty_part ing={len(obj['ingredients'])} dir={len(obj['directions'])}"})
            continue
        rows.append({
            "start": j["block"]["start"],
            "title": obj["title"],
            "ingredients": obj["ingredients"],
            "directions": obj["directions"],
            "NER": obj["NER"],
        })

    rows.sort(key=lambda r: r["start"])
    import csv
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "title", "ingredients", "directions", "link", "source", "NER"])
        for n, r in enumerate(rows, 1):
            w.writerow([f"cookbook-{n:04d}", r["title"],
                        json.dumps(r["ingredients"], ensure_ascii=False),
                        json.dumps(r["directions"], ensure_ascii=False),
                        f"{LINK_BASE}{r['start'] + 1}", SOURCE,
                        json.dumps(r["NER"], ensure_ascii=False)])

    FAIL_PATH.write_text(json.dumps({"failures": failures, "skipped": skipped},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    MANIFEST.write_text(json.dumps({"prompt_ver": PROMPT_VER, "model": MODEL,
                                    "blocks": len(blocks), "rows": len(rows),
                                    "source_lines": len(lines),
                                    "blocks_digest": sha(json.dumps([b["start"] for b in blocks]))},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    phase(f"rows: {len(rows)}  failures: {len(failures)}  skipped: {len(skipped)}  "
          f"cost: ${cost['cumulative_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
