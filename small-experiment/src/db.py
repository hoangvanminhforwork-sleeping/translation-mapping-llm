"""MySQL 9.x storage: relational rows + native VECTOR(768) columns.

Why MySQL 9 and not pgvector: pgvector is a PostgreSQL extension.  MySQL 9 has a
native VECTOR type and DISTANCE(), so one container gives us both the metadata
(SQL joins/filters) and the embeddings.  At 12k x 768 there is no need for an
approximate index; a full scan is exact and fast.
"""
import json
import time

import numpy as np
import pymysql

from . import config as C


def connect(with_db: bool = True):
    return pymysql.connect(
        host=C.DB_HOST, port=C.DB_PORT, user=C.DB_USER, password=C.DB_PASSWORD,
        database=C.DB_NAME if with_db else None, charset="utf8mb4",
        autocommit=False, connect_timeout=5,
        read_timeout=1800, write_timeout=1800,
    )


def wait_ready(timeout: int = 240) -> str:
    """Block until the container accepts connections; returns its version."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            con = connect(with_db=False)
            with con.cursor() as cur:
                cur.execute("SELECT VERSION()")
                version = cur.fetchone()[0]
            con.close()
            return version
        except Exception as exc:          # container still booting
            last = exc
            time.sleep(3)
    raise RuntimeError(f"MySQL not reachable after {timeout}s: {last}")


def ensure_schema() -> str:
    """Idempotent DDL; safe to run any number of times."""
    version = wait_ready()
    con = connect(with_db=False)
    try:
        with con.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {C.DB_NAME} "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
            cur.execute(f"USE {C.DB_NAME}")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pairs (
                    pair_id   INT UNSIGNED NOT NULL PRIMARY KEY,
                    split     ENUM('train','test') NOT NULL,
                    vi        TEXT NOT NULL,
                    en        TEXT NOT NULL,
                    KEY idx_split (split)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    pair_id  INT UNSIGNED NOT NULL,
                    lang     CHAR(2) NOT NULL,
                    model    VARCHAR(160) NOT NULL,
                    pooling  VARCHAR(8) NOT NULL,
                    dim      SMALLINT UNSIGNED NOT NULL,
                    vec      VECTOR({C.DIM}) NOT NULL,
                    PRIMARY KEY (pair_id, lang, pooling),
                    KEY idx_lookup (lang, pooling),
                    CONSTRAINT fk_emb_pair FOREIGN KEY (pair_id)
                        REFERENCES pairs (pair_id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        con.commit()
    finally:
        con.close()
    return version


def _vec_literal(v: np.ndarray) -> str:
    """MySQL wants a JSON-like array string; 6 decimals ~ 1e-6 absolute error."""
    return "[" + ",".join(f"{x:.6f}" for x in np.asarray(v, dtype=np.float32)) + "]"


def upsert_pairs(df, chunk: int = 500) -> int:
    sql = ("INSERT INTO pairs (pair_id, split, vi, en) VALUES (%s,%s,%s,%s) AS new "
           "ON DUPLICATE KEY UPDATE split=new.split, vi=new.vi, en=new.en")
    con = connect()
    n = 0
    try:
        with con.cursor() as cur:
            for i in range(0, len(df), chunk):
                part = df.iloc[i:i + chunk]
                cur.executemany(sql, [(int(r.pair_id), r.split, r.vi, r.en)
                                      for r in part.itertuples(index=False)])
                con.commit()
                n += len(part)
    finally:
        con.close()
    return n


def upsert_embeddings(rows, chunk: int = 200) -> int:
    """rows = iterable of (pair_id, lang, model, pooling, np.ndarray)."""
    sql = ("INSERT INTO embeddings (pair_id, lang, model, pooling, dim, vec) "
           "VALUES (%s,%s,%s,%s,%s,STRING_TO_VECTOR(%s)) AS new "
           "ON DUPLICATE KEY UPDATE model=new.model, dim=new.dim, vec=new.vec")
    con = connect()
    n = 0
    buf = []
    try:
        with con.cursor() as cur:
            for pair_id, lang, model_name, pooling, vec in rows:
                buf.append((int(pair_id), lang, model_name, pooling, C.DIM, _vec_literal(vec)))
                if len(buf) >= chunk:
                    cur.executemany(sql, buf)
                    con.commit()
                    n += len(buf)
                    buf.clear()
            if buf:
                cur.executemany(sql, buf)
                con.commit()
                n += len(buf)
    finally:
        con.close()
    return n


def counts() -> dict:
    con = connect()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pairs")
            pairs = cur.fetchone()[0]
            cur.execute("SELECT lang, pooling, COUNT(*) FROM embeddings GROUP BY 1,2 ORDER BY 1,2")
            groups = {(l, p): c for l, p, c in cur.fetchall()}
    finally:
        con.close()
    return {"pairs": pairs, "embeddings": groups}


def fetch_vectors(lang: str, pooling: str, split: str | None = None):
    """Return (pair_ids, matrix) ascending by pair_id -> stable alignment."""
    sql = ("SELECT e.pair_id, VECTOR_TO_STRING(e.vec) "
           "FROM embeddings e JOIN pairs p ON p.pair_id = e.pair_id "
           "WHERE e.lang = %s AND e.pooling = %s")
    args = [lang, pooling]
    if split:
        sql += " AND p.split = %s"
        args.append(split)
    sql += " ORDER BY e.pair_id"

    con = connect()
    try:
        with con.cursor() as cur:
            cur.execute(sql, args)
            data = cur.fetchall()
    finally:
        con.close()

    ids = np.array([r[0] for r in data], dtype=np.int64)
    mat = np.empty((len(data), C.DIM), dtype=np.float32)
    for i, (_, text) in enumerate(data):
        mat[i] = np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float32)
    return ids, mat


# NOTE: MySQL 9.4 *Community* ships the VECTOR type and STRING_TO_VECTOR /
# VECTOR_TO_STRING / VECTOR_DIM, but NOT DISTANCE() / VECTOR_DISTANCE() - those
# are HeatWave-only (verified on 9.4.0: error 1305).  So MySQL is the store and
# the search runs client-side over the fetched vectors (exact cosine, no index).
# See agent-work/ASSUMPTIONS.md S8.


