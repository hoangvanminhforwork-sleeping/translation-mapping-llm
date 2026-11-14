"""Single source of truth for paths, models and hyper-parameters.

Everything a later agent needs to reproduce a number lives here, not in the scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # research-phase01
PROJECT = ROOT.parent                               # repo root
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Shared HF cache (7.6 GB, already holds vit5-base / flan-t5-base / vit5-large / flan-t5-large).
os.environ.setdefault("HF_HOME", str(PROJECT / "research-prototype" / "models" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CORPUS = PROJECT / "data" / "data" / "train-*.parquet"
PAIRS = DATA / "pairs.parquet"
STATS = DATA / "stats"

MODELS = {
    "vit5-base": {"hf_id": "VietAI/vit5-base", "d_model": 768},
    "flan-t5-base": {"hf_id": "google/flan-t5-base", "d_model": 768},
}

# Prior artefacts whose rows must never appear in train/val/test again (mission §2: no leakage).
EXCLUDE = {
    "scaled_100k": PROJECT / "research-scale-100k" / "data" / "pairs_100k.parquet",
    "copy_lora_train_50k": PROJECT / "finetune-and-lora" / "prototype" / "data" / "train_copy_50000.jsonl",
    "copy_lora_eval_4k": PROJECT / "finetune-and-lora" / "prototype" / "data" / "eval_sources.jsonl",
}

MAX_LEN = 64        # subword tokens, both sides
BATCH = 96          # pairs per forward during encoding (both encoders at once does not fit: see learn note)
N_TEST = 2000
N_VAL = 2000
SHARD = 200_000     # pairs per statistics shard (resume unit)

MAP_VI_EN = DATA / "map_vi_en.npz"
DECODER_ADAPTER = ROOT / "adapters" / "flan-decoder-vi-en"
