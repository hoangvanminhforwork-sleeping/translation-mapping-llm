"""Execute 01_results.ipynb in place, so the committed notebook carries real outputs.

Uses nbclient with the shared venv's interpreter through a kernelspec registered once as
`research-phase01`. Running it twice is safe: it re-executes and rewrites the same outputs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
NB = ROOT / "01_results.ipynb"
KERNEL = "research-phase01"


def ensure_kernel():
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        if KERNEL in KernelSpecManager().find_kernel_specs():
            return
    except Exception:  # noqa: BLE001
        pass
    subprocess.run([str(PY), "-m", "ipykernel", "install", "--user", "--name", KERNEL,
                    "--display-name", KERNEL], check=True)


def main():
    ensure_kernel()
    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name=KERNEL, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(nb, NB)
    n_ok = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
    print(f"executed {n_ok} code cells, wrote {NB.name}")


if __name__ == "__main__":
    main()
