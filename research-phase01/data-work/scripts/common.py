"""Shared helpers for the PDF -> page images -> vision-OCR -> markdown pipeline.

No secrets are ever printed. The API key is read from research-phase01/.env (var DataWorker)
and stripped of surrounding double quotes.
"""
from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_WORK = PROJECT_ROOT / "research-phase01" / "data-work"
SCRATCH = PROJECT_ROOT / "agent-work" / "dataworker"
PAGES = DATA_WORK / "pages"
OCR = DATA_WORK / "ocr"
MARKDOWN = DATA_WORK / "markdown"
LOG_PATH = DATA_WORK / "ocr_log.jsonl"
COST_PATH = DATA_WORK / "cost.json"
STOP_PATH = SCRATCH / "STOPPED.md"
ENV_PATH = PROJECT_ROOT / "research-phase01" / ".env"

MODEL = "qwen/qwen3.7-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PRICE_IN_PER_M = 0.03
PRICE_OUT_PER_M = 0.13
COST_LIMIT_USD = 0.50
DPI = 180

BOOKS = {
    "vn_mon_an_truyen_thong": r"data\Vietnamese\mon_an_vn_truyen_thong_992_4159.pdf",
    "vn_van_hoa_am_thuc": r"data\Vietnamese\van-hoa-am-thuc-viet-nam-nhin-tu-ly-luan-va-thuc-tien-1772676558.pdf",
    "us_american_cuisine": r"data\America\American Cuisine PDF.pdf",
    "us_sioux_chef": r"data\America\PDF The Sioux Chef's Indigenous Kitchen pdf.pdf",
    "us_food_younger_land": r"data\America\The Food of a Younger Land PDF.pdf",
}


def book_path(slug: str) -> Path:
    return PROJECT_ROOT / BOOKS[slug]


def ensure_dirs() -> None:
    for d in (PAGES, OCR, MARKDOWN, SCRATCH, DATA_WORK / "scripts"):
        d.mkdir(parents=True, exist_ok=True)


def load_key() -> str:
    """Read DataWorker from .env without ever echoing it."""
    for line in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "DataWorker":
            return v.strip().strip('"').strip("'").strip()
    raise RuntimeError("DataWorker key not found in research-phase01/.env")


def page_png(slug: str, n: int) -> Path:
    return PAGES / slug / f"page-{n:04d}.png"


def page_md(slug: str, n: int) -> Path:
    return OCR / slug / f"page-{n:04d}.md"


def render_page(doc, slug: str, n: int, dpi: int = DPI, force: bool = False) -> Path:
    """n is 1-based. Skips if the PNG already exists with non-zero size."""
    out = page_png(slug, n)
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    page = doc[n - 1]
    pix = page.get_pixmap(dpi=dpi, colorspace="rgb", alpha=False)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(pix.tobytes("png"))
    os.replace(tmp, out)
    return out


PROMPT_VERSION = "v1"
PROMPT = f"""[transcribe-{PROMPT_VERSION}] You are a document transcription engine. Transcribe the attached page image EXACTLY.
Rules:
1. Output the page's text verbatim in its ORIGINAL language. Never translate.
2. Preserve every diacritic and accent character exactly (Vietnamese: bánh chưng, phở, ẩm thực; do not drop or mangle marks).
3. Keep the natural reading order (top-to-bottom, left-to-right, columns read as columns).
4. A heading that is visibly a heading becomes a markdown heading: use "## " for it. Sub-heading may use "### ".
5. Render tables as markdown pipe tables and lists as markdown bullet/numbered lists.
6. Do NOT summarise, do NOT add commentary, do NOT describe the image, do NOT wrap the answer in code fences.
7. Output ONLY the transcription. No preamble, no closing remarks.
8. If the page has no meaningful text (blank, plain cover, or a pure illustration/photo), output exactly one token: [BLANK] for empty pages, or [IMAGE-ONLY] for image-only pages.
"""


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# Fallback prompt used only to redo a page whose first transcription looked short (v2).
PROMPT2_VERSION = "v2"
PROMPT2 = f"""[transcribe-{PROMPT2_VERSION}] Transcribe every character visible on this page image.

Rules:
1. Copy the text verbatim in its ORIGINAL language. Never translate, never summarise.
2. Preserve all diacritics exactly (Vietnamese: bánh chưng, phở, đ, ẩm thực).
3. Read in natural order: header, body, footnotes, page number, footer.
4. Do not skip lines. Even a single short sentence on an otherwise empty page must be output.
5. Headings -> "## "; tables -> markdown pipe tables; lists -> markdown lists.
6. Output ONLY the transcription, no commentary, no code fences.
7. Only if there is literally no text: output [BLANK] (empty page) or [IMAGE-ONLY] (picture only).
"""


def call_api(key: str, png: Path, prompt: str = PROMPT, max_tokens: int = 8192,
             temperature: float = 0.0, reasoning: bool = False) -> dict:
    """Single request. Returns {'text','prompt_tokens','completion_tokens','usd'}.

    reasoning=False is deliberate: thinking tokens are billed as output but never appear in
    the transcription, and they eat the max_tokens budget (measured on page 1: 2777 reasoning
    tokens -> 3127 completion tokens vs 330 with reasoning off, same page, same text).
    """
    import requests

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _b64_png(png)},
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": reasoning},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    text = (msg.get("content") or "").strip()
    finish = choice.get("finish_reason")
    usage = data.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    usd = (pt * PRICE_IN_PER_M + ct * PRICE_OUT_PER_M) / 1_000_000
    return {"text": text, "prompt_tokens": pt, "completion_tokens": ct, "usd": usd,
            "finish_reason": finish, "truncated": finish == "length", "raw_usage": usage}


def call_with_retry(key: str, png: Path, tries: int = 4, **kw) -> dict:
    delay = 2.0
    last = None
    for attempt in range(1, tries + 1):
        try:
            return call_api(key, png, **kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            last = e
            retryable = any(s in msg for s in ("HTTP 429", "HTTP 5", "timed out", "Timeout",
                                               "Connection", "Remote end closed"))
            if attempt == tries or not retryable:
                raise
            time.sleep(delay + random.uniform(0, 1.0))
            delay *= 2
    raise last  # pragma: no cover


def append_log(rec: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def total_cost() -> float:
    """Sum of usd over the log file (single source of truth)."""
    if not LOG_PATH.exists():
        return 0.0
    total = 0.0
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line).get("usd") or 0.0)
            except Exception:  # noqa: BLE001
                continue
    return total


def write_cost_json(extra: dict | None = None) -> dict:
    tot = total_cost()
    payload = {
        "model": MODEL,
        "price_input_usd_per_m": PRICE_IN_PER_M,
        "price_output_usd_per_m": PRICE_OUT_PER_M,
        "cumulative_usd": round(tot, 6),
        "limit_usd": COST_LIMIT_USD,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_stop(reason: str) -> None:
    STOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STOP_PATH.write_text(
        "# PIPELINE STOPPED (cost guard)\n\n"
        f"- reason: {reason}\n"
        f"- cumulative_usd: {total_cost():.6f}\n"
        f"- limit_usd: {COST_LIMIT_USD}\n"
        f"- timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
        encoding="utf-8",
    )
