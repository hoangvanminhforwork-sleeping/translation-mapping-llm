"""Single source of truth for tonight's experiment (autonomous-research-experiment).

Models (2, both frozen):
    A = Helsinki-NLP/opus-mt-vi-en   77M   -> the translator / candidate generator
    B = Helsinki-NLP/opus-mt-en-vi   77M   -> back-translator for round-trip signals

Question: can a *reference-free* quality signal rank a candidate pool as well as
reference-based COMET, and does the LATENT (representation) version beat the
string version?

Nothing here fine-tunes A or B except the explicitly labelled LoRA control (A5).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
for d in (DATA, RESULTS, LOGS, MODELS):
    d.mkdir(exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SOURCE_PAIRS = PROJECT / "research-phase01" / "data" / "pairs.parquet"
SPLIT = DATA / "split.parquet"

A_ID = "Helsinki-NLP/opus-mt-vi-en"
B_ID = "Helsinki-NLP/opus-mt-en-vi"
COMET_ID = "Unbabel/wmt22-comet-da"

N_TEST = 10_000       # new test size (superset of the historical 2,000)
N_VAL = 10_000
N_TEST_CORE = 2_000   # historical split: first 2,000 by md5(vi) -> comparable to 22.27
MAX_NEW = 96
SEED = 20261119
