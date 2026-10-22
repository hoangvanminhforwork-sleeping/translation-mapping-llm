"""Pairs: inherited from experiment 01, copied once and hashed."""
import hashlib
import shutil

import pandas as pd
import pyarrow.parquet as pq

from . import config as C


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pairs(force: bool = False) -> pd.DataFrame:
    """12000 rows: pair_id, split, vi, en.  Same draw as experiment 01."""
    if force or not C.PAIRS.exists():
        C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.PAIRS_SRC, C.PAIRS)
    df = pq.read_table(C.PAIRS).to_pandas()
    assert len(df) == C.N_TRAIN + C.N_TEST, len(df)
    assert (df.split == "test").sum() == C.N_TEST
    return df


def split(df: pd.DataFrame):
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    return train, test


def shuffled_index(n: int, seed: int = C.SEED):
    """Derangement-ish control: every item gets another item's vector."""
    import numpy as np
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(n)
    clash = perm == np.arange(n)
    perm[clash] = (perm[clash] + 1) % n
    return perm
