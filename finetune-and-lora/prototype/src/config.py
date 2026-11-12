"""Paths, models, LoRA hyper-parameters. One place; nothing hard-coded elsewhere."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # prototype/
PROJECT = ROOT.parent  # finetune-and-lora/
RESEARCH = PROJECT.parent / "research-prototype"  # sibling baseline prototype

# The checkpoints were already downloaded once. Re-downloading 7.6 GB to prove a
# point is waste, so the shared cache is the default; override with HF_HOME.
SHARED_HF = RESEARCH / "models" / "hf"
HF_HOME = Path(os.environ.get("HF_HOME", str(SHARED_HF)))
os.environ["HF_HOME"] = str(HF_HOME)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
ADAPTERS = PROJECT / "adapters"
AGENT_WORK = ROOT / "agent-work"
for _d in (DATA, RESULTS, ADAPTERS, AGENT_WORK):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = RESEARCH.parent / "data" / "data" / "train-00000-of-00014.parquet"
# Baseline generations, reused so "before" and "after" see identical sentences.
# The raw JSONL is the primary evidence; the per-model CSV is a derived view.
EVAL_RAW = {
    "vi": RESEARCH / "results" / "raw" / "copy_vit5-base.jsonl",
    "en": RESEARCH / "results" / "raw" / "copy_flan-t5-base.jsonl",
}

MODELS = {
    "vit5-base": {"hf_id": "VietAI/vit5-base", "lang": "vi", "prefix": "Nói: ", "batch": 32},
    "flan-t5-base": {"hf_id": "google/flan-t5-base", "lang": "en", "prefix": "Say: ", "batch": 32},
    "vit5-large": {"hf_id": "VietAI/vit5-large", "lang": "vi", "prefix": "Nói: ", "batch": 4},
    "flan-t5-large": {"hf_id": "google/flan-t5-large", "lang": "en", "prefix": "Say: ", "batch": 4},
}
ORDER = ["vit5-base", "flan-t5-base", "vit5-large", "flan-t5-large"]

N_TRAIN = 50_000
SEED = 42

# LoRA on q,v only: the copy circuit lives in attention, and q,v are 2/3 of the
# cheap options. r=16 is small enough that the adapter is ~1 MB on a T5-base.
LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q", "v"]}
LR = 1e-3
WARMUP = 100
GRAD_CLIP = 1.0
MAX_LEN = 64
MAX_NEW_TOKENS = 128
SAVE_EVERY = 500
LOG_EVERY = 50
DTYPE = "bfloat16"