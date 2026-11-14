"""Probe: what does `usage` look like, and can we turn reasoning off to cut output tokens?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import requests  # noqa: E402

key = common.load_key()
png = common.page_png("vn_mon_an_truyen_thong", 1)
import base64

b64 = base64.b64encode(png.read_bytes()).decode("ascii")


def go(label: str, extra: dict):
    payload = {
        "model": common.MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": common.PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]}],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    payload.update(extra)
    r = requests.post(common.ENDPOINT,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json=payload, timeout=300)
    print(f"--- {label}: HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return
    d = r.json()
    ch = d["choices"][0]
    print("finish_reason:", ch.get("finish_reason"))
    print("usage:", json.dumps(d.get("usage"), ensure_ascii=False))
    txt = ch["message"].get("content") or ""
    print("chars:", len(txt), "| head:", txt[:90].replace("\n", " / "))
    u = d.get("usage") or {}
    pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    usd = (pt * 0.03 + ct * 0.13) / 1e6
    print(f"usd={usd:.6f}")


go("default", {})
go("reasoning-off", {"reasoning": {"enabled": False}})
