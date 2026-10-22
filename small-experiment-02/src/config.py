"""Single source of truth for experiment 02 (Vector2 -> Token2 readout).

Models, pairs and split are INHERITED from small-experiment/. Nothing here
re-downloads or re-samples.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]           # small-experiment-02/
PROJECT = ROOT.parent                                # repo root
EXP1 = PROJECT / "small-experiment"                  # experiment 01 (source of truth)
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
MAPS = ROOT / "maps"
AGENT_WORK = ROOT / "agent-work"

# models live with experiment 01; do not duplicate 936 MB
HF_HOME = EXP1 / "models" / "hf"
os.environ.setdefault("HF_HOME", str(HF_HOME))

# NLTK data must not land on C:
os.environ.setdefault("NLTK_DATA", str(AGENT_WORK / "nltk_data"))

PAIRS_SRC = EXP1 / "artifacts" / "sample_pairs.parquet"
PAIRS = ARTIFACTS / "pairs.parquet"

for _d in (ARTIFACTS, RESULTS, MAPS, AGENT_WORK):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
SEED = 42
N_TRAIN = 10_000
N_TEST = 2_000

# ---------------------------------------------------------------- encoders (frozen, inherited)
EN_MODEL = "BAAI/bge-base-en-v1.5"
VI_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DIM = 768
POOLINGS = ("mean", "cls", "last")
MAX_TOKENS = 128
ENCODE_BATCH = 64

# ---------------------------------------------------------------- maps (refit here, never persisted in exp 01)
RIDGE_ALPHA = 100.0                                  # retrieval-optimal, summary.md section 3
MLP_HIDDEN = 2048
MLP_LR = 3e-4
MLP_EPOCHS = 60
MLP_PATIENCE = 8
MLP_BATCH = 256
MLP_WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.1

# ---------------------------------------------------------------- inversion readout
INV_STEPS = 150
INV_LR = 0.05                                        # Adam step on soft token embeddings
INV_PROJECT_EVERY = 5                                # nearest-vocabulary snap interval
INV_BLEND = 0.5                                      # unused with straight-through (kept for clarity)
INV_TOKEN_BUDGET = 6000                              # max tokens in one forward (all arms)
INV_AMP = True                                       # fp16 autocast for the inversion forward

ARMS = ("roundtrip", "identity", "ridge", "ann", "shuffled")
DIRECTIONS = (("en2vi", "en", "vi"), ("vi2en", "vi", "en"))

# ---------------------------------------------------------------- evaluation
N_EVAL = 2_000                                       # test items inverted per arm
TGT_LANG = {"en2vi": "vi", "vi2en": "en"}
