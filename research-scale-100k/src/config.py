"""Paths, models, and the numbers that define this experiment. One place."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # research-scale-100k/
PROJECT = ROOT.parent
RESEARCH = PROJECT / "research-prototype"         # shared checkpoint cache + baseline prototype
FINETUNE = PROJECT / "finetune-and-lora"          # LoRA adapters

SHARED_HF = RESEARCH / "models" / "hf"
HF_HOME = Path(os.environ.get("HF_HOME", str(SHARED_HF)))
os.environ["HF_HOME"] = str(HF_HOME)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
for _d in (DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

PAIRS = DATA / "pairs_100k.parquet"
ADAPTERS = FINETUNE / "adapters"

MODELS = {
    "vit5-base": {"hf_id": "VietAI/vit5-base", "lang": "vi", "batch": 32},
    "flan-t5-base": {"hf_id": "google/flan-t5-base", "lang": "en", "batch": 32},
}

SEED = 42
RIDGE_LAMBDAS = [0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
N_BLEU = 1000          # the whole test split
GEN_BATCH = 64
MAX_NEW_TOKENS = 96
