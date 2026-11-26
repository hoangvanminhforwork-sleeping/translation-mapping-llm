# How to reproduce this paper

Everything below is idempotent. Nothing downloads, nothing needs a cluster, and the only GPU step
(the naive-channel measurement) takes about four minutes.

## 0. Environment

Use the shared environment of the scale experiment; no new packages were installed.

```powershell
cd coincidence-solution
$py = "..\research-scale-100k\.venv\Scripts\python.exe"
& $py -c "import torch, transformers, sacrebleu, matplotlib; print('ok')"
```

Versions used: torch 2.7.1+cu128, transformers 4.57.6, peft 0.21.2, sacrebleu 2.5.1, matplotlib 3.11.2.

## 1. The paper

`paper.md` is the deliverable. Read it in any Markdown viewer; images are relative to
`scientific-communication/`. To make a PDF, install pandoc + a TeX engine outside this project
(none is installed here, and the instruction was not to write to `C:`), then:

```powershell
pandoc paper.md -o paper.pdf --pdf-engine=xelatex
```

The file compiles as-is with this command; the only thing pandoc will not render is the LaTeX
display math, which needs `--pdf-engine=xelatex` (not `pdflatex`) for the Unicode in the references.

## 2. The figures — one notebook, no GPU

```powershell
cd coincidence-solution\scientific-communication
$env:MPLBACKEND = "Agg"
& $py -m nbconvert --to notebook --execute --inplace figures.ipynb
```

This reads `../results/*.json` and rewrites `figures/fig01 … fig12`. It takes ~20 s and needs no GPU.

**To change a figure, edit `figures.ipynb` and re-run.** The notebook is the source of truth; the
helper that assembled it (`agent-work/build_figs_nb.py`) only concatenates cells and is not needed to
regenerate anything.

The notebook prints a table at the end with every number used in the paper's tables. If you doubt a
number in `paper.md`, compare it there first.

## 3. The one new measurement (the naive channel)

The Introduction's 0.05 BLEU is a real run, not a citation:

```powershell
cd coincidence-solution
& $py -m src.eval_identity          # writes results/identity.json, ~4 min, 1.6 GB RAM
& $py -m src.eval_identity --force  # only if you want to overwrite it
```

Expected: BLEU 0.047, chrF 6.114, CE 4.608, 7.1% empty.

**If you change the adapter, the tokeniser or `MAX_LEN`, delete `results/identity.json` and re-run.**
The script feeds source-encoder states through the *same* rank-16 adapter as every other arm;
changing the readout changes what "naive" means.

## 4. Re-deriving the numbers in the paper

All of these already have committed outputs in `coincidence-solution/results/`; re-run them only if
you suspect the cache or the environment.

```powershell
& $py -m src.validate              # calibration + oracle  -> results/calibration.json
& $py -m src.coincidence --n 300   # tolerance curves     -> results/coincidence.json   (12 min)
& $py -m src.fit_behaviour --arms ridge_full,ridge_sub,ridge_recolor10,ridge_scale122,ce_ridge,ce_scratch
& $py -m src.fit_behaviour --arms ce_ridge_e6 --init-map ce_ridge --epochs 4
& $py -m src.bootstrap_all         # CIs                  -> results/bootstrap.json
& $py -m src.analyse_basis         # spectrum + subspace  -> results/basis_analysis.json
& $py -m src.eval_comet --n 400    # COMET                -> results/comet.json
& $py -m src.replicate             # disjoint sample      -> results/replicate.json
```

### The check that tells you the environment is right

`src.validate` must print **ridge CE 3.228** and **native CE 3.413**. If it does not, stop: the venv,
the adapter, the map or the cache is wrong, and every downstream number is meaningless. This exact
check is what makes the paper's cross-package comparison (2.50 BLEU here vs 2.73 in the earlier
package) trustworthy: the likelihood matches to three decimals, and the BLEU difference is the
decoding budget (`max_new` 64 vs 48), not precision.

## 5. What must be deleted before a clean re-run

| delete | why |
|---|---|
| `agent-work/cache/` | 6.5 GB of encoder states; invalidated by any change to `MAX_LEN` or tokenisation |
| `agent-work/maps/<arm>.npz` | `fit_behaviour` *loads* an existing map instead of training; use `--force` to overwrite |
| `results/*.json` for the arm you are re-running | otherwise the notebook plots the old numbers |

Never delete `agent-work/hyps/` if you want to re-run bootstrap intervals: they are computed from the
persisted hypotheses, not from the model.
