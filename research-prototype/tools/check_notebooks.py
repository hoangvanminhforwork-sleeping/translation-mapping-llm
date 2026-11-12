"""Static check: every code cell in every notebook must compile.

Cheap insurance. A stray brace in an f-string in the notebook builder would
otherwise only surface at the end of the run, when the GPU work is already spent.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
bad = 0
for nb_path in sorted(ROOT.glob("**/*.ipynb")):
    if ".ipynb_checkpoints" in str(nb_path):
        continue
    nb = nbformat.read(nb_path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            bad += 1
            print(f"SYNTAX {nb_path.relative_to(ROOT)} cell {i}: {exc.msg} (line {exc.lineno})")
            print("      " + (cell.source.splitlines()[exc.lineno - 1] if exc.lineno else ""))
    print(f"checked {nb_path.relative_to(ROOT)}: {sum(1 for c in nb.cells if c.cell_type == 'code')} code cells")

print("syntax errors:", bad)
sys.exit(1 if bad else 0)
