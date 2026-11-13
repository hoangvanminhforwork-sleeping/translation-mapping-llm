"""Execute every notebook in this folder in place, in dependency order.

    python tools/execute_notebooks.py          # all
    python tools/execute_notebooks.py mapping  # one, by substring

Each notebook gets its own kernel: loading vit5-base and flan-t5-base together does not fit in
4 GB of VRAM, so a notebook that needs both reads the other's output files instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORDER = [
    "01_mapping_100k.ipynb",
]


def run(rel: str) -> bool:
    nb = ROOT / rel
    if not nb.exists():
        print(f"MISSING {rel}")
        return False
    cmd = [
        sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.kernel_name=research-scale-100k",
        "--ExecutePreprocessor.timeout=1800",
        str(nb),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok = proc.returncode == 0
    print(f"{'OK  ' if ok else 'FAIL'} {rel}")
    if not ok:
        for line in (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]:
            print("      " + line[:160])
    return ok


def main(argv: list[str]) -> int:
    targets = [r for r in ORDER if not argv or any(a in r for a in argv)]
    results = [run(r) for r in targets]
    print(f"\n{sum(results)}/{len(results)} notebooks executed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
