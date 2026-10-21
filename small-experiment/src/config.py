"""Single source of truth for the prototype.  Every experiment value lives here."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # small-experiment/
PROJECT = ROOT.parent                                # repo root
DATA_DIR = PROJECT / "data"
PARQUET_DIR = DATA_DIR / "data"
RECIPE_CSV = DATA_DIR / "RecipeNLG.csv"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
AGENT_WORK = ROOT / "agent-work"
MODELS = ROOT / "models"

# ---------------------------------------------------------------- sampling
SEED = 42
SAMPLE_SHARD = "train-00008-of-00014.parquet"        # fixed after the seed draw; see ASSUMPTIONS.md
N_TRAIN = 10_000
N_TEST = 2_000
MAX_WORDS = 64                                      # keeps the embedder inside 128 tokens
SAMPLE_PARQUET = ARTIFACTS / "sample_pairs.parquet"

# ---------------------------------------------------------------- encoders
# 768-d on both sides so the mapping is a square problem (missions.md #2).
# Candidates are tried in order; whichever loads is recorded in manifest.json.
EN_ENCODERS = ("BAAI/bge-base-en-v1.5",)             # English-native, 768, sentence-tuned
VI_ENCODERS = (
    "bkai-foundation-models/vietnamese-bi-encoder",  # PhoBERT-base + contrastive tuning
    "dangvantuan/vietnamese-embedding",              # PhoBERT-base, alternative tuning
    "vinai/phobert-base-v2",                         # plain Vietnamese MLM (fallback, no fastBPE)
)
DIM = 768
POOLINGS = ("mean", "cls", "last")
MAX_TOKENS = 128
ENCODE_BATCH = 64
HF_HOME = MODELS / "hf"

# ---------------------------------------------------------------- storage
DB_HOST = "127.0.0.1"
DB_PORT = 3307
DB_USER = "root"
DB_PASSWORD = "csi_proto"
DB_NAME = "csi_proto"

# ---------------------------------------------------------------- mapping
HIDDEN = 1024
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0)
VAL_FRACTION = 0.1                                  # carved out of train only
TOP_K = (1, 5, 10)

# ---------------------------------------------------------------- probing
N_POS = 10_000
N_NEG = 10_000
CV_FOLDS = 5
