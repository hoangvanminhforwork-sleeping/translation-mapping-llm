"""Shared helpers for the structuring pipeline (Phases: dishes, facts, db, report).

Conventions
-----------
* All paths are derived from the repo root (4 levels up from this file).
* API key is read from research-phase01/.env (var DataWorker) and never printed.
* Every LLM response is cached in structured/cache/<sha1>.json  -> reruns cost $0.
* Spend is accumulated in structured/cost.json by summing cached usage too,
  so the recorded total is the *actual* spend of the run that produced the data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import pathlib

_LOCK = threading.Lock()

ROOT = pathlib.Path(__file__).resolve().parents[4]
MD_DIR = ROOT / "research-phase01" / "data-work" / "markdown"
OUT_DIR = ROOT / "research-phase01" / "data-work" / "structured"
CACHE_DIR = OUT_DIR / "cache"
SCRATCH = ROOT / "agent-work" / "structurer"
COST_PATH = OUT_DIR / "cost.json"

MODEL = "qwen/qwen3.7-flash"
PRICE_IN = 0.03 / 1_000_000
PRICE_OUT = 0.13 / 1_000_000
HARD_CAP = 0.85
SOFT_CAP = 0.70

BOOKS = {
    "vn_mon_an_truyen_thong": {"ocr_model": "unknown (pre-existing OCR pass)", "kind": "recipe-book"},
    "vn_van_hoa_am_thuc": {"ocr_model": "unknown (pre-existing OCR pass)", "kind": "academic"},
    "us_american_cuisine": {"ocr_model": "unknown (pre-existing OCR pass)", "kind": "bookey-summary"},
    "us_food_younger_land": {"ocr_model": "unknown (pre-existing OCR pass)", "kind": "bookey-summary"},
    "us_sioux_chef": {"ocr_model": "unknown (pre-existing OCR pass)", "kind": "publisher-blurb"},
}


def load_pages(slug: str) -> dict[int, str]:
    """Return {page_number: text} for a markdown OCR file with <!-- page N --> markers."""
    text = (MD_DIR / f"{slug}.md").read_text(encoding="utf-8")
    parts = re.split(r"<!-- page (\d+) -->", text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def api_key() -> str:
    env = (ROOT / "research-phase01" / ".env").read_text(encoding="utf-8")
    for line in env.splitlines():
        line = line.strip()
        if line.startswith("DataWorker="):
            v = line.split("=", 1)[1].strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            return v
    raise SystemExit("DataWorker key not found in research-phase01/.env")


_LAST_GOOD: dict | None = None


def _blank_state() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
            "cost_usd_estimated": 0.0, "by_stage": {}, "history": []}


def _cost_state() -> dict:
    """Read cost.json; never raise on a torn/partial read (another thread may be writing)."""
    global _LAST_GOOD
    for _ in range(5):
        if not COST_PATH.exists():
            return _LAST_GOOD if _LAST_GOOD is not None else _blank_state()
        try:
            _LAST_GOOD = json.loads(COST_PATH.read_text(encoding="utf-8"))
            return _LAST_GOOD
        except Exception:
            time.sleep(0.01)
    return _LAST_GOOD if _LAST_GOOD is not None else _blank_state()


def _save_cost(state: dict) -> None:
    global _LAST_GOOD
    _LAST_GOOD = state
    try:
        tmp = COST_PATH.with_suffix(f".{os.getpid()}_{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        for _ in range(20):
            try:
                os.replace(tmp, COST_PATH)
                break
            except OSError:
                time.sleep(0.05)
    except Exception:
        pass


def _add_cost(stage: str, pt: int, ct: int, cost: float, cached: bool) -> None:
    with _LOCK:
        state = _cost_state()
        est = pt * PRICE_IN + ct * PRICE_OUT
        state["calls"] += 0 if cached else 1
        state["prompt_tokens"] += pt
        state["completion_tokens"] += ct
        state["cost_usd"] = round(state["cost_usd"] + (0.0 if cached else cost), 6)
        state["cost_usd_estimated"] = round(state["cost_usd_estimated"] + est, 6)
        st = state["by_stage"].setdefault(stage, {"calls": 0, "prompt_tokens": 0,
                                                 "completion_tokens": 0, "cost_usd": 0.0})
        st["calls"] += 0 if cached else 1
        st["prompt_tokens"] += pt
        st["completion_tokens"] += ct
        st["cost_usd"] = round(st["cost_usd"] + (0.0 if cached else cost), 6)
        _save_cost(state)


def rebuild_cost_from_cache() -> dict:
    """Rebuild cost.json from the cache: one cache file == one billed call.

    This is the authoritative spend figure (the runtime counter double-counts tokens on
    cache hits, which is fine for the live budget guard but wrong for reporting).
    """
    state = _blank_state()
    for p in CACHE_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("error"):
            stage = "failed_calls"
        else:
            tag = rec.get("tag", "")
            stage = "dishes" if tag.startswith("dish-") else \
                    "facts" if tag.startswith("facts-") else \
                    "smoketest" if tag.startswith("smoke") else "other"
        u = rec.get("usage") or {}
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        c = float(u.get("cost") or 0.0)
        state["calls"] += 1
        state["prompt_tokens"] += pt
        state["completion_tokens"] += ct
        state["cost_usd"] = round(state["cost_usd"] + c, 6)
        state["cost_usd_estimated"] = round(state["cost_usd_estimated"] + pt * PRICE_IN + ct * PRICE_OUT, 6)
        st = state["by_stage"].setdefault(stage, {"calls": 0, "prompt_tokens": 0,
                                                  "completion_tokens": 0, "cost_usd": 0.0})
        st["calls"] += 1
        st["prompt_tokens"] += pt
        st["completion_tokens"] += ct
        st["cost_usd"] = round(st["cost_usd"] + c, 6)
    _save_cost(state)
    return state


def chat(stage: str, system: str, user: str, max_tokens: int = 2000,
         temperature: float = 0.0, force: bool = False, tag: str = "") -> tuple[str, dict]:
    """Cached chat completion. Returns (content, usage)."""
    import requests

    key = hashlib.sha1(
        json.dumps([MODEL, system, user, max_tokens, temperature], ensure_ascii=False).encode()
    ).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath = CACHE_DIR / f"{key}.json"
    state = _cost_state()

    if cpath.exists() and not force:
        try:
            rec = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            rec = None                      # partially written / corrupt -> treat as a miss
        if rec is not None:
            if rec.get("error"):
                raise RuntimeError(rec["error"])   # replay a cached API failure
            _add_cost(stage, rec["usage"].get("prompt_tokens", 0),
                      rec["usage"].get("completion_tokens", 0),
                      rec["usage"].get("cost", 0.0), cached=True)
            return rec["content"], {**rec["usage"], "cached": True}

    if state["cost_usd"] >= HARD_CAP:
        raise SystemExit(f"STOP: budget cap reached (${state['cost_usd']:.4f})")
    if os.environ.get("DS_DEBUG_CACHE"):
        print(f"[MISS] tag={tag} key={key[:12]} len={len(user)} "
              f"head={user[:80]!r}", file=sys.stderr)
    if state["cost_usd"] >= SOFT_CAP:
        print(f"[warn] soft cap reached (${state['cost_usd']:.4f}); continuing carefully", file=sys.stderr)

    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning": {"enabled": False},
    }
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    last = None
    for attempt in range(6):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=300)
            if r.status_code in (429, 502, 503):
                time.sleep(2 ** attempt)
                last = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            cost = usage.get("cost")
            if cost is None:
                cost = pt * PRICE_IN + ct * PRICE_OUT
            rec = {"tag": tag, "content": content,
                   "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cost": cost}}
            # atomic publish: concurrent readers never see a partial file. A local I/O
            # hiccup must never turn into a second API call, so failures here are swallowed.
            try:
                tmp = cpath.with_suffix(f".{os.getpid()}_{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                for _ in range(20):
                    try:
                        os.replace(tmp, cpath)
                        break
                    except OSError:
                        time.sleep(0.05)
            except Exception:
                pass
            _add_cost(stage, pt, ct, cost, cached=False)
            return content, {**rec["usage"], "cached": False}
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            time.sleep(2 ** attempt)
    # publish the failure too, so a rerun replays it (deterministic, $0);
    # never clobber a good reply that another thread may have just written
    try:
        prior = None
        if cpath.exists():
            try:
                prior = json.loads(cpath.read_text(encoding="utf-8"))
            except Exception:
                prior = None
        if not (prior and prior.get("content")):
            rec = {"tag": tag, "content": "", "error": f"LLM call failed after retries: {last}",
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}}
            tmp = cpath.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cpath)
            raise RuntimeError(rec["error"])
        return prior["content"], {**prior["usage"], "cached": True}
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError(f"LLM call failed after retries: {last}")


def parse_json_block(text: str):
    """Extract the first JSON object/array from a model reply (tolerates ``` fences)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # prefer the outermost object; only fall back to an array if there is no '{'
    pairs = [("{", "}")] if "{" in t else [("[", "]")]
    for opener, closer in pairs:
        i = t.find(opener)
        j = t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    raise ValueError("no JSON found")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
