"""Sampling.  The output artifact is the *only* input of later stages, so the
draw happens once and is then frozen on disk (missions.md reproducibility rule).

Vietnamese is detected by the presence of a Vietnamese-only diacritic.  The
shard is bidirectional, so this filter is what turns it into VI -> EN pairs.
"""
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C
from . import manifest

VI_DIACRITICS = ("[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộ"
                 "ờớởỡợùúủũụừứửữựỳýỷỹỵ]")
PAIRS_ARTIFACT = C.SAMPLE_PARQUET
TITLES_ARTIFACT = C.ARTIFACTS / "probe_titles.parquet"


def _rng() -> np.random.Generator:
    return np.random.default_rng(C.SEED)


def _sql_path(p) -> str:
    return str(p).replace("\\", "/")


def _pairs_sql() -> str:
    return f"""
    SELECT DISTINCT translation.source AS vi, translation.target AS en
    FROM read_parquet('{_sql_path(C.PARQUET_DIR / C.SAMPLE_SHARD)}')
    WHERE regexp_matches(lower(translation.source), '{VI_DIACRITICS}')
      AND NOT regexp_matches(lower(translation.target), '{VI_DIACRITICS}')
      AND length(trim(translation.source)) > 0
      AND length(trim(translation.target)) > 0
      AND len(string_split(trim(translation.source), ' ')) <= {C.MAX_WORDS}
      AND len(string_split(trim(translation.target), ' ')) <= {C.MAX_WORDS}
    ORDER BY vi, en
    """


def sample_pairs(force: bool = False) -> pd.DataFrame:
    """Materialise N_TRAIN + N_TEST VI/EN pairs.  Re-running reuses the artifact."""
    if PAIRS_ARTIFACT.exists() and not force:
        return pq.read_table(PAIRS_ARTIFACT).to_pandas()

    shard = C.PARQUET_DIR / C.SAMPLE_SHARD
    if not shard.exists():
        raise FileNotFoundError(shard)

    con = duckdb.connect()
    try:
        candidates = con.sql(_pairs_sql()).df()
    finally:
        con.close()

    need = C.N_TRAIN + C.N_TEST
    if len(candidates) < need:
        raise RuntimeError(f"only {len(candidates)} candidates, need {need}")

    rng = _rng()
    picked = np.sort(rng.choice(len(candidates), size=need, replace=False))
    df = candidates.iloc[picked].reset_index(drop=True)
    df["pair_id"] = np.arange(need, dtype=np.int32)
    df["split"] = np.where(np.arange(need) < C.N_TRAIN, "train", "test")

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df[["pair_id", "split", "vi", "en"]],
                                        preserve_index=False), PAIRS_ARTIFACT)

    manifest.record(
        "sample_pairs",
        seed=C.SEED, shard=C.SAMPLE_SHARD,
        shard_sha256=manifest.sha256(shard),
        sql=_pairs_sql().strip(),
        n_candidates=int(len(candidates)), n_sampled=int(len(df)),
        n_train=int((df.split == "train").sum()), n_test=int((df.split == "test").sum()),
        artifact=str(PAIRS_ARTIFACT.name),
        artifact_sha256=manifest.sha256(PAIRS_ARTIFACT),
    )
    return df


def probe_titles(force: bool = False) -> pd.DataFrame:
    """N_POS distinct RecipeNLG dish titles (positives of the food probe)."""
    if TITLES_ARTIFACT.exists() and not force:
        return pq.read_table(TITLES_ARTIFACT).to_pandas()

    con = duckdb.connect()
    try:
        titles = con.sql(f"""
            SELECT DISTINCT trim(title) AS title
            FROM read_csv('{_sql_path(C.RECIPE_CSV)}', header = true)
            WHERE title IS NOT NULL AND trim(title) <> ''
            ORDER BY title
        """).df()
    finally:
        con.close()

    rng = _rng()
    pick = np.sort(rng.choice(len(titles), size=min(C.N_POS, len(titles)), replace=False))
    df = titles.iloc[pick].reset_index(drop=True)

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), TITLES_ARTIFACT)

    manifest.record(
        "probe_titles", seed=C.SEED, n_candidates=int(len(titles)),
        n_sampled=int(len(df)), artifact=str(TITLES_ARTIFACT.name),
        artifact_sha256=manifest.sha256(TITLES_ARTIFACT),
    )
    return df


def probe_negatives(pairs: pd.DataFrame | None = None) -> list[str]:
    """Non-food English sentences: the English side of the sampled corpus.

    Assumption: the parallel corpus is general-domain, so its English side is a
    reasonable 'not a dish name' pool.  Length is matched roughly by construction
    (both come from the same <=64-word filter).
    """
    pairs = sample_pairs() if pairs is None else pairs
    rng = _rng()
    n = min(C.N_NEG, len(pairs))
    pick = np.sort(rng.choice(len(pairs), size=n, replace=False))
    return pairs["en"].iloc[pick].tolist()
