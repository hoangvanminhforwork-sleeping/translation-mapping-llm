"""Download the four checkpoints into the project-local HF cache.

Only the files actually needed are fetched: the repositories also ship TF, Flax
and duplicate-safetensors weights, which would double the byte cost for nothing.
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

from . import config as C

# Weights: vit5 has only pytorch_model.bin; flan-t5 has both formats, prefer safetensors.
ALLOW = {
    "vit5-base": ["*.json", "spiece.model", "pytorch_model.bin"],
    "vit5-large": ["*.json", "spiece.model", "pytorch_model.bin"],
    "flan-t5-base": ["*.json", "spiece.model", "model.safetensors"],
    "flan-t5-large": ["*.json", "spiece.model", "model.safetensors"],
}


def fetch(model_key: str) -> str:
    spec = C.MODELS[model_key]
    t0 = time.time()
    path = snapshot_download(
        repo_id=spec["hf_id"],
        allow_patterns=ALLOW[model_key],
        max_workers=4,
    )
    print(f"{model_key:<14} {spec['hf_id']:<22} -> {path}  ({time.time() - t0:.0f}s)", flush=True)
    return path


def main(keys: list[str]) -> None:
    for k in keys:
        fetch(k)
    print("all requested checkpoints present", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or C.MODEL_ORDER)
