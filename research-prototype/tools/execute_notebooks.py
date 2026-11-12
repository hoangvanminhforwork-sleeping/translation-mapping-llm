"""Execute every notebook in place, in dependency order.

Each notebook runs in its own kernel, so a model loaded by one notebook is never
shared with the next (4 GB of VRAM would not survive that).

    python tools/execute_notebooks.py            # all
    python tools/execute_notebooks.py compare    # one, by substring
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORDER = [
    "model-exploratory/00_compare_models.ipynb",
    "model-exploratory/01_vit5-base.ipynb",
    "model-exploratory/02_flan-t5-base.ipynb",
    "model-exploratory/03_vit5-large.ipynb",
    "model-exploratory/04_flan-t5-large.ipynb",
    "model-instruction-following/01_instruction_following.ipynb",
    "test-token-generation/01_testset_and_perplexity.ipynb",
    "test-token-generation/02_bleu_zero_shot.ipynb",
]


def run(rel: str) -> bool:
    nb = ROOT / rel
    if not nb.exists():
        print(f"MISSING {rel}")
        return False
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=research-prototype",
        "--ExecutePreprocessor.timeout=900",
        str(nb),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = proc.returncode == 0
    print(f"{'OK  ' if ok else 'FAIL'} {rel}")
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            print("      " + line[:160])
    return ok


def main(argv: list[str]) -> int:
    targets = [r for r in ORDER if not argv or any(a in r for a in argv)]
    results = [run(r) for r in targets]
    print(f"\n{sum(results)}/{len(results)} notebooks executed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
