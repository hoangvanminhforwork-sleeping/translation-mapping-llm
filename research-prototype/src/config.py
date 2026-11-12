"""Single source of truth for paths, models and experiment constants.

Every other module imports from here. No hard-coded absolute paths elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------
# src/config.py -> research-prototype/
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

# Hugging Face cache MUST stay inside the project (never C:).
HF_HOME = ROOT / "models" / "hf"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA = ROOT / "data"
RESULTS = ROOT / "results"
AGENT_WORK = ROOT / "agent-work"
for _d in (HF_HOME, DATA, RESULTS, AGENT_WORK):
    _d.mkdir(parents=True, exist_ok=True)

CULINARY_JSONL = DATA / "culinary_100_pairs.jsonl"
CULINARY_CSV = DATA / "culinary_100_pairs.csv"

# MTET test split: the file the mission names. 2,536 clean en<->vi pairs.
MTET_TEST = PROJECT / "data" / "data" / "test-00000-of-00001.parquet"

# --- models ----------------------------------------------------------------
# Pairing rule from the mission: base pairs with base, large with large.
# `prompt` is the copy-instruction prefix from thought01.md (Say:/Nói:).
MODELS = {
    "vit5-base": {
        "hf_id": "VietAI/vit5-base",
        "lang": "vi",
        "size_class": "base",
        "copy_prefix": "Nói: ",
        "batch": 16,
    },
    "vit5-large": {
        "hf_id": "VietAI/vit5-large",
        "lang": "vi",
        "size_class": "large",
        "copy_prefix": "Nói: ",
        "batch": 8,
    },
    "flan-t5-base": {
        "hf_id": "google/flan-t5-base",
        "lang": "en",
        "size_class": "base",
        "copy_prefix": "Say: ",
        "batch": 16,
    },
    "flan-t5-large": {
        "hf_id": "google/flan-t5-large",
        "lang": "en",
        "size_class": "large",
        "copy_prefix": "Say: ",
        "batch": 8,
    },
}

# Order matters: base first (mission), and cheap before expensive.
MODEL_ORDER = ["vit5-base", "flan-t5-base", "vit5-large", "flan-t5-large"]

PAIRS = {
    "base": ("vit5-base", "flan-t5-base"),
    "large": ("vit5-large", "flan-t5-large"),
}

# Zero-shot translation prompts. English target -> flan-t5, Vietnamese target -> vit5.
TRANSLATE_PROMPT = {
    "en2vi": "translate English to Vietnamese: {}",
    "vi2en": "translate Vietnamese to English: {}",
}

DTYPE = "bfloat16"      # fp16 is numerically unstable for T5-family checkpoints
N_COPY_SENTENCES = 2000
SEED = 42
MAX_NEW_TOKENS = 128
