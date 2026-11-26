# Handoff — scientific-communication — 2026-11-26 15:45 (UTC+7)

## State

**Complete and self-consistent.** `paper.md` is written, all twelve figures are generated from
committed result files, the one new measurement (`results/identity.json`) is reproducible, and the
numbers in the paper are traceable to JSON.

| file | role | regenerate with |
|---|---|---|
| `paper.md` | the deliverable | edit by hand |
| `figures/fig01…fig12.png` | figures | `nbconvert --execute figures.ipynb` |
| `figures.ipynb` | figure source (measured + conceptual) | edit cells, re-run |
| `REPORT.md` | mission report: decisions, corrections, word-count plan | — |
| `LEARNING_NOTE.md` | the teaching document (why the argument works) | — |
| `REPRODUCE.md` | step-by-step reproduction | — |
| `agent-work/lit-digest.md` (also copied to `lit-digest.md`) | 26 verified paper summaries built from local PDFs | — |

Outside this folder: `coincidence-solution/src/eval_identity.py` and
`coincidence-solution/results/identity.json` were **added by this session**.

## The paper in one paragraph (LaTeX-precise)

Two frozen 60M T5 models (`t5-small-vietnamese` → `t5-small`), $d=512$, disjoint tokenisers. A
position-wise affine bridge $f(h)=Wh+b$ with $|\theta|=262{,}656$. Baseline: ridge on
$\mathcal{L}_{\text{emb}}=\|f(H_{vi})-\mathrm{Align}(H_{en})\|^2$, 85M token pairs. Intervention:
identical parameter count, minimise $\mathcal{L}_{\text{beh}}=-\log p(s_{en}\mid f(H_{vi}))$ with the
decoder frozen, from the ridge warm start, 20k pairs, AdamW lr $10^{-4}$, batch 32, 2/6/16 epochs.
Result on 1,000 held-out sentences, greedy, `max_new=64`: BLEU 2.47 → 7.90; chrF 19.77 → 29.36; CE
3.232 → 2.738; degeneration 16.7% → 6.7%; COMET 0.480 → 0.583; paired bootstrap +5.40 [4.50, 6.27].
Ceiling: the same decoder reading $H_{en}(s_{en})$ gives 74.22 BLEU / CE 0.302 — an empirical
behavioural ceiling of the readout.

Mechanism: the decoder's tolerance is a thin, anisotropic band ($\varepsilon=0.05$ random changes 22%
of outputs; the ridge memory sits at $\varepsilon=1$ along its own error direction, 0% identical). The
least-squares residual is isotropic in the decoder's sensitive subspace (0.8–1.4× chance for
$k=1,8,64,256$), so no null-space argument rescues it. Distillation from the oracle adds 0.0001 nats.

## Numbers, with their file

| claim | value | file |
|---|---|---|
| naive channel (new this session) | BLEU 0.047 / chrF 6.114 / CE 4.608 / 7.1% empty | `results/identity.json` |
| calibration vs parent package | ridge CE 3.228, native CE 3.413 (exact) | `results/calibration.json` |
| oracle | 74.217 BLEU / 84.70 chrF / 0.3017 CE | `results/calibration.json` |
| nine arms, max64 and max48 | see Table 4 | `results/behaviour_maps.json`, `results/maxnew48.json` |
| CIs | +5.398 [4.495, 6.269] for `ce_ridge_e16` | `results/bootstrap.json` |
| tolerance curves | random vs ridge-direction vs grad | `results/coincidence.json` |
| subspace enrichment | 0.777×, 1.236×, 1.420×, 1.335× | `results/coincidence.json` |
| pooled-Frobenius enrichment (the 3–5× footnote) | 5.1×, 3.4×, 2.4× for $k$=8,64,256 | `results/basis_analysis.json` |
| COMET | 0.4363 → 0.5830, monotone in BLEU | `results/comet.json` |
| replication | 2.68 → 8.079, gap +5.40 | `results/replicate.json` |

## Do not quote

- `results/coincidence.json → oracles.nat.ridge_error.cos_sent` (−2.13) is not a cosine; different
  masks. The paper does not use it.
- `results/train_ce_ridge_e12.csv` is the killed 12-epoch run; no BLEU exists for it.
- Any cross-package BLEU comparison without naming `max_new`. The paper always reports both 64 and 48.
- `agent-work/maps/ridge_recolor01.npz` was deleted (it was t=0.01, misread as 0.1). Use
  `ridge_recolor10`.

## Known deviations from the brief

1. **Length.** Body ≈ 6,100 words, captions + tables ≈ 1,000, references excluded. Target was
   4,000–5,000. `REPORT.md` §5 has a ranked cut list totalling ≈ 580 words and a recommendation.
2. **Conceptual figures leak past the Introduction.** Figures 4 and 5 sit in Problem definition and
   Methodology. Both are formalisation aids with measured numbers only; the rule in `mission.md` said
   Introduction and Literature review. If enforced literally, move them.
3. **Title.** The user reserved the title. Three alternates are in `REPORT.md` §2.

## If you continue this work

- **Do not edit `paper.md` with PowerShell `Set-Content`.** It re-encoded the file to ANSI once and
  destroyed every em dash, `×` and `≥`. Use the editor tool, or write with `encoding='utf-8'` in
  Python and verify `len([c for c in text if ord(c) > 127])` is unchanged.
- Re-running `src.fit_behaviour` **loads** an existing `agent-work/maps/<arm>.npz` instead of
  training. `--force` overwrites a published arm.
- `src.validate` printing CE 3.228/3.413 is the environment check; anything else means stop.
- The figure notebook needs no GPU and no cache. It is the cheapest way to check a number.
