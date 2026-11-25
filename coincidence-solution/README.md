# coincidence-solution (E11)

Testing the "coincidence" idea from `mental-model.md`: instead of making the mapped vector *close* to
the target vector, find a memory that makes the decoder *behave*. Session 2026-11-26, $0 API.

**Result:** the same 263 200-parameter map, fitted on the frozen decoder's cross-entropy instead of
least squares, goes from **2.47 to 7.04 BLEU** (2.67 → 7.59 at E9's decoding setting) with
degeneration down from 16.7 % to 9.0 %. Data, scale and covariance controls are in the report.

| file | read it for |
|---|---|
| `REPORT.md` | findings, controls, claim boundaries, reproduction commands |
| `LEARNING_NOTE.md` | decisions, trade-offs, and the traps (read §4 before touching arm names) |
| `HANDOFF.md` | exact state, invalid artefacts, what to do next |
| `NOTEBOOK.ipynb` | every table and figure, re-derived from `results/*.json` in seconds |
| `src/` | 10 modules; `cc.py` → `coincidence.py` → `fit_behaviour.py` → `analyse_basis.py` |
| `results/` | every json/csv the report cites |
| `agent-work/maps/` | fitted `(W, b)` per arm — the deliverable |

Quick start (the venv is shared with `research-scale-100k`):

```powershell
$py = "..\research-scale-100k\.venv\Scripts\python.exe"
& $py -m src.cache_states   # 150 s, rebuilds agent-work/cache (gitignored, 4.1 GB)
& $py -m src.validate       # must print ridge CE 3.228 / native CE 3.413 — stop if it does not
```
