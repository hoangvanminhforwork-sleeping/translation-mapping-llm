"""Run metadata.  Everything that makes a result traceable lives here.

The point: a result file is only meaningful together with the exact data hash,
the model ids, the package versions and the git commit that produced it.
"""
import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata as _md

from . import config as C

KEY_PKGS = ("torch", "transformers", "numpy", "pandas", "pyarrow", "duckdb",
            "scikit-learn", "pymysql", "cryptography")
MANIFEST = C.AGENT_WORK / "manifest.json"


def sha256(path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=C.PROJECT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _version(pkg: str):
    try:
        return _md.version(pkg)
    except Exception:
        return None


def env() -> dict:
    return {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "packages": {p: _version(p) for p in KEY_PKGS},
    }


def record(stage: str, **payload) -> dict:
    """Merge one stage's metadata into agent-work/manifest.json (idempotent)."""
    C.AGENT_WORK.mkdir(parents=True, exist_ok=True)
    doc = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else {}
    doc.setdefault("environment", env())
    doc[stage] = {**env(), **payload}
    MANIFEST.write_text(json.dumps(doc, indent=2, ensure_ascii=False), "utf-8")
    return doc[stage]
