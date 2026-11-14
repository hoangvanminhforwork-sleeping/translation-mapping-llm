"""Smoke test: 1 tiny call, checks the model reply and the usage/cost fields."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

content, usage = common.chat(
    stage="smoketest",
    system="You answer with strict JSON only.",
    user='Return {"ok": true, "model": "<your model id>"} and nothing else.',
    max_tokens=100,
    tag="smoke",
)
print("USAGE:", usage)
print("REPLY:", content[:300].encode("unicode_escape").decode())
