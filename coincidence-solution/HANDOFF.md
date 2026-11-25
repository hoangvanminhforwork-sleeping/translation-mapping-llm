# Handoff — coincidence-solution — 2026-11-26 06:30 (UTC+7)

State: **complete for the session's clock.** All scripts run end-to-end, all numbers in `REPORT.md`
are traceable to a file in `results/`, the working tree is committed. One arm (`distill`) is
specified and unrun; one run (`ce_ridge_e12`) was killed by a server restart at epoch 2 of 6.

## The one-paragraph state

Fitting the 263 200-parameter bridge on the frozen decoder's **cross-entropy** instead of
least-squares gives **2.47 → 7.04 BLEU** (max_new 64) / **2.67 → 7.59** (max_new 48) on the same
1 000 held-out sentences, with CE 3.232 → 2.761 and degeneration 16.7 % → 9.0 %. Controls separate
the causes: the data is identical (`ridge_sub` = 2.47), pure scale correction buys +0.42, E9's
recolour buys +0.61, and CE from an identity initialisation fails (2.25) — so the MSE solution is a
good *initialiser* and a bad *objective*. The mechanism is measured: the decoder's behavioural
tolerance is ~5× smaller than the least-squares bridge's error, and that error is isotropic with
respect to the decoder's sensitive subspace (1.3–1.4× chance), so no null-space argument rescues it.

## Numbers to trust, and where they come from

| claim | value | file |
|---|---|---|
| calibration vs E9 (CE exact, BLEU within decoding budget) | ridge 2.50 / CE 3.228 vs E9 2.73 / 3.228 | `results/calibration.json` |
| oracle ceiling (decoder reading the English reference) | 74.22 BLEU / CE 0.302 | `results/calibration.json` |
| level set: 5 % random displacement | 78 % outputs identical, BLEU −0.6 | `results/coincidence.json` |
| level set: map's own error direction, ε=1 | 0 % identical, BLEU 2.4, CE 3.32 | `results/coincidence.json` |
| anisotropy at equal displacement | CE ×2.1 (grad) vs ×1.002 (random) | `results/coincidence.json` |
| map's error in the read subspace | 1.4× chance (top-64) → isotropic | `results/coincidence.json` |
| CE-fitted map, 2 / 6 epochs | 5.33 / 7.04 BLEU (max64) | `results/behaviour_maps.json` |
| bootstrap vs published map | +2.83 [+2.28, +3.49] / +4.55 (see note) | `results/bootstrap.json` |
| E9 recolour replication | 2.67 → 3.43 at max_new 48 | `results/maxnew48.json` |

> **Note on `bootstrap.json`:** it is written per invocation and the harness killed the 9-arm run
> after one arm, so the file may contain a subset. `ce_ridge`'s CI (+2.83 [+2.28, +3.49], 500
> resamples) is confirmed by two independent runs. Regenerate any missing row with
> `python -m src.bootstrap_all <arm> [<arm> ...]` (≈75 s per arm, CPU only).

## What is NOT done (in priority order)

1. **`distill` arm** — KL(student ‖ oracle) added to CE, teacher = the decoder reading `E_ref`.
   Fully implemented in `src/fit_behaviour.py`; the cached teacher states for the fit split are
   already on disk (`agent-work/cache/fit/{E_ref,m_ref}.npy`). Run:
   `python -m src.fit_behaviour --arms distill --init-map ce_ridge_e6 --epochs 4`
   Hypothesis: soft targets beat hard labels because the oracle's distribution carries *which*
   alternatives are acceptable.
2. **Convergence** — `val CE` was still falling at epoch 6 (3.013 → 2.929 → 2.889 → 2.856 → 2.838 →
   2.825). A cosine schedule over 20 epochs is the cheapest open number.
3. **COMET** on the seven arms (installed, ~7 sent/s → 2.5 min/arm). BLEU and CE agree, but COMET
   closes the "fluent but unfaithful" question that the Oct-31 pilot raised.
4. **en→vi** — the `trans-t5-small-en-vi` adapter exists; `cc.py` is direction-agnostic except for
   `MAP`/`ADAPTER` constants.
5. **Two adapters instead of one mixed adapter** (open since E7/E8). This is no longer cosmetic: the
   CE objective is fitted *against* the adapter, so a mixed adapter confounds the channel difference.
6. **Test-time refinement** — gradient steps on `δ` at inference to measure how much of the remaining
   CE gap is reachable at all (uses the reference; diagnostic only).

## Invalid / do-not-quote artefacts

| artefact | problem |
|---|---|
| `results/coincidence.json` → `oracles.nat.ridge_error.cos_sent` = −2.13 | inner product and norms were taken under different masks, so it is not a cosine. Curves are unaffected; the field is flagged in `REPORT.md` §7. |
| `results/coincidence.json` → `oracles.ref.ridge_error` | absent by design: positions are not comparable across tokenisations (see `LEARNING_NOTE.md` §2). |
| `results/train_ce_ridge_e12.csv` | **killed run** (server restart at epoch 2 of 6). It is not an arm; do not read it as a result. The 6-epoch arm is `ce_ridge_e6`. |
| `agent-work/maps/ridge_recolor01.npz` | deleted. It was t=0.01 and its name was misread as t=0.1 (see `LEARNING_NOTE.md` §4). Use `ridge_recolor10` (t=0.10). |
| any BLEU compared across phases | `max_new` alone is worth 0.2–0.35 BLEU here. E9/E7/E8 used 48; this package's main table uses 64 and also reports 48. |

## Environment as measured

- Shared venv: `../research-scale-100k/.venv` (torch 2.7.1+cu128, transformers 4.57.6, peft 0.21.2,
  sklearn 1.9.1, sacrebleu 2.5.1). No new packages were installed; **$0.00 spent**.
- RTX 3050 Laptop 4 GB: peak ≈ 1.6 GB VRAM (fp32 decoder + one batch). RAM is the binding limit:
  2.25 GB free after closing Edge/Zalo/ChatGPT; everything is memmapped for that reason.
- Wall clock: cache 150 s · calibration 168 s · coincidence 720 s · CE arms ≈ 110 s/epoch +
  ≈ 90 s eval per arm · `analyse_basis` 289 s · `eval_maxnew48` 367 s.

## Layout

```
coincidence-solution/
  REPORT.md            findings, controls, claim boundaries, reproduction
  LEARNING_NOTE.md     decisions, trade-offs, traps (read §4 before touching arm names)
  HANDOFF.md           this file
  NOTEBOOK.ipynb       the same pipeline as cells, results read from results/*.json
  src/                 10 modules, documented in LEARNING_NOTE §9
  results/             every json/csv the report cites
  agent-work/cache/    6.5 GB, gitignored, rebuildable in 150 s
  agent-work/maps/     (W,b) per arm, committed (1 MB each) — the deliverable
  agent-work/hyps/     greedy hypotheses per arm, committed (n=1000)
  agent-work/logs/     raw stdout of every run
```

## Reproducibility checklist for the next agent

1. `git status` — the tree should be clean at commit `AGENT: ...` (see `git log -1`).
2. `python -m src.validate` — must print ridge CE 3.228 and native CE 3.413. If it does not, the
   venv, the adapter or the cache is wrong; stop and fix that before interpreting anything.
3. `python -m src.fit_behaviour --arms ce_ridge` with the existing `agent-work/maps/ce_ridge.npz`
   present will **load** the map rather than retrain (idempotent by design). Pass `--force` only if
   you mean to overwrite a published arm.
