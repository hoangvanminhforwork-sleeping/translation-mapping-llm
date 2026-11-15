"""Single source of truth: paths, models, split rule, hyper-parameters.

Everything a later agent needs to reproduce a number lives here, not in the scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # research-phase01-ver02
PROJECT = ROOT.parent                               # repo root
DATA = ROOT / "data"
RESULTS = ROOT / "results"
WORK = ROOT / "agent-work"
for d in (DATA, RESULTS, WORK):
    d.mkdir(parents=True, exist_ok=True)

# Shared HF cache. The two new checkpoints were pulled with `local_dir` because this
# huggingface_hub build cannot create symlinks without Developer Mode (WinError 1314).
os.environ.setdefault("HF_HOME", str(PROJECT / "research-prototype" / "models" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
LOCAL_MODELS = PROJECT / "research-prototype" / "models" / "hf" / "local"

CORPUS = PROJECT / "data" / "data" / "train-*.parquet"
PAIRS = DATA / "pairs.parquet"          # every row of the 14 shards, per-file deduped, tagged with `file`
PAIRS_MANIFEST = DATA / "pairs_manifest.json"
STATS = DATA / "stats"

MODELS = {
    "en": {"hf_id": "google-t5/t5-small", "path": LOCAL_MODELS / "t5-small", "d_model": 512, "vocab": 32128},
    "vi": {"hf_id": "nbdaaa/t5-small-vietnamese", "path": LOCAL_MODELS / "t5-small-vietnamese",
           "d_model": 512, "vocab": 36096},
}

# The shard that an earlier agent already sampled for copy-LoRA (50k rows, seed 42).
# It is therefore "dirty": used for our own LoRA, never for the map's train/val/test.
# NOTE: no ".parquet" suffix — this is the value of the `file` column in pairs.parquet.
LORA_FILE = "train-00000-of-00014"

# Rows a previous experiment touched must not reappear in the map corpus (mission §2).
# NOTE: research-phase01's own pairs.parquet is deliberately *not* excluded. Its 100k decoder rows
# were trained under a different architecture (vit5/flan-base); what matters for validity is that
# *our* test split is never seen by *our* training, and that the copy-LoRA shard is dropped.
EXCLUDE = {
    "scaled_100k": PROJECT / "research-scale-100k" / "data" / "pairs_100k.parquet",
    "copy_lora_train_50k": PROJECT / "finetune-and-lora" / "prototype" / "data" / "train_copy_50000.jsonl",
    "copy_lora_eval_4k": PROJECT / "finetune-and-lora" / "prototype" / "data" / "eval_sources.jsonl",
}

MAX_LEN = 64          # subword tokens, both sides
BATCH = 256           # pairs per forward (two 60M encoders fit together: measured)
SHARD = 200_000       # pairs per moment shard (resume unit)

# 80/10/10, assigned by md5(vi) order inside the chosen shard: no RNG, re-runs cannot move a row.
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

# Generation evaluation sizes.
N_GEN = 3_000        # sentences generated per channel (BLEU/chrF)
N_COMET = 1_000       # subset scored with COMET (it runs at ~7 sent/s)

MAP = {"vi_en": DATA / "map_vi_en.npz", "en_vi": DATA / "map_en_vi.npz"}
MAP_LGBM = {"vi_en": DATA / "lgbm_vi_en.npz", "en_vi": DATA / "lgbm_en_vi.npz"}
SENT_EMB = DATA / "sent_vi_fp16.npy"     # (N, 512) fp16, coreset input
SENT_EMB_META = DATA / "sent_vi_meta.parquet"

ADAPTERS = {
    "copy_en": ROOT / "adapters" / "copy-t5-small",
    "copy_vi": ROOT / "adapters" / "copy-t5-small-vi",
    "trans_en": ROOT / "adapters" / "trans-t5-small-vi-en",
    "trans_vi": ROOT / "adapters" / "trans-t5-small-en-vi",
}

# Coreset: K = number of MTet domains (Law, Religion, News, Medical) = number of dataset fields.
K_MAIN = 4
K_SENS = 32
