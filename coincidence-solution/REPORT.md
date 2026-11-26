# Coincidence solution — fit the bridge on the decoder's behaviour, not on vector error

**Session:** 2026-11-26, 00:40 → 10:00 (UTC+7) · **Budget spent:** $0.00 of $3.00 (local GPU only)
**Code:** `coincidence-solution/src/` · **Results:** `coincidence-solution/results/`
**Question from `mental-model.md`:** does there exist `H_e' ≠ H_e` with `d(H_e') ≈ d(H_e)`, and can we
find it from another encoder's states?

---

## 0. TL;DR — five numbers

| # | finding | number | file |
|---|---|---|---|
| 1 | Fitting the **same** 263 200-parameter map on the decoder's cross-entropy instead of MSE | **2.47 -> 7.90 BLEU** at 16 epochs (2.67 -> **8.46** at max_new 48) - CE 3.232 -> 2.738 - degenerate 16.7 % -> 6.7 % - COMET 0.4797 -> 0.5830 | `behaviour_maps.json`, `maxnew48.json`, `comet.json` |
| 2 | The gain is **not** scale correction: pure isotropic rescaling to the same ‖H‖ | +0.42 BLEU (17 % of the 2-epoch gain) | `behaviour_maps.json` |
| 3 | It is **not** extra data: ridge refitted on the *same* 20 k pairs | 2.47 vs 2.50, ΔBLEU −0.03 [−0.25, +0.17] | `bootstrap.json` |
| 4 | The decoder's level set is **thin and anisotropic**: a 5 % random displacement of the oracle memory changes 22 % of outputs; the steepest direction at the same norm raises CE 2.1× and halves BLEU | anisotropy ≥ 100× | `coincidence.json` |
| 5 | E9's "recolour = +16 % BLEU for free" **replicates in magnitude** (+28 % at its own decoding setting) but is not significant on 1 000 sentences — and is 5× smaller than behaviour fitting | 2.67 → 3.43 BLEU | `maxnew48.json` |
| 6 | An independent learned metric agrees: **COMET is monotone in BLEU across all seven arms** | 0.4797 → **0.5641** | `comet.json` |

**One sentence:** a linear bridge between two frozen monolingual models should be fitted on the
quantity it is judged by — the target decoder's likelihood — and when it is, the same 0.26 M
parameters produce **2.8× the BLEU** of the least-squares bridge that the project has used since E1.

---

## 1. The question, made measurable

`mental-model.md` proposes replacing "make the mapped vector close to the target vector" with
"find any memory that makes the decoder behave correctly". Taken literally the claim is trivially
true (a many-to-one map always has level sets), so three measurable versions were fixed before any
code was written:

| id | question | instrument |
|---|---|---|
| **Q1** | How large is the decoder's level set around a good memory, in units of memory displacement? | perturb `H_ref` by `ε·δ`, measure output identity / chrF / BLEU / CE |
| **Q2** | Is the level set isotropic — is the map's actual error direction worse than random? | same sweep along `H_ridge − H_ref` and along `∂CE/∂H` |
| **Q3** | Where does the map's error sit relative to the level set, and does it concentrate in the directions the decoder reads? | singular subspace of the per-sentence gradient `∂CE/∂H` |

`d(H_e') ≈ d(H_e)` becomes **exact-match rate of greedy outputs** plus graded chrF, which is
well defined across memories of different length — unlike any Euclidean distance, since a memory at
vi token positions and one at en token positions have no position correspondence.

## 2. Setup and calibration (nothing downstream is interpretable without this)

Everything reuses E8/E9's assets so numbers are comparable: `chosen.parquet` (479 394 rows,
80/10/10 by `md5(vi)`), `map_vi_en.npz` (263 200 params), the `trans-t5-small-vi-en` LoRA adapter
(decoder-only, 1.18 M trainable params), both 60 M checkpoints. New: `coincidence-solution/src/cc.py`
re-implements the channels (the two `src` packages cannot be imported together — see
`research-phase01-1.2/HANDOFF.md`).

Reimplementation check, n = 1 000 test sentences, fp32, `max_new=64`
(`results/calibration.json`):

| channel | BLEU | chrF | CE | E9 published | verdict |
|---|---|---|---|---|---|
| ridge | 2.50 | 19.77 | **3.228** | 2.73 / 19.73 / **3.228** | CE exact, BLEU −0.23 (fp32 vs bf16, `max_new`) |
| native | 3.75 | 14.72 | **3.413** | 4.20 / 14.82 / **3.413** | CE exact |
| **ref (new)** | **74.22** | **84.70** | **0.302** | — | the oracle is real |

The third row is the session's enabling measurement: **the decoder reading the target model's
encoder states of the English reference produces 74 BLEU.** So there exists a memory the adapter can
read perfectly, and the headroom from the mapped channel is CE 3.228 → 0.302. Without this row
"the decoder is the bottleneck" and "the map is the bottleneck" cannot be separated.

Re-scored at E9's own decoding setting (`max_new=48`), the reproduction is exact:
ridge **2.71** vs published 2.73 (`results/maxnew48.json`).

## 3. Q1/Q2 — the level set is thin, and the decoder is wildly anisotropic

n = 300, oracle memory `H_ref`, base output 74.3 BLEU / CE 0.290 / ‖H‖ = 17.4
(`results/coincidence.json`). `ε` is displacement as a fraction of that memory's own Frobenius norm.

**Random direction** (isotropic null hypothesis):

| ε | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
|---|---|---|---|---|---|
| identical outputs | 0.78 | 0.64 | 0.47 | 0.21 | 0.03 |
| chrF vs base | 97.0 | 94.5 | 91.3 | 82.6 | 41.1 |
| BLEU | 73.7 | 72.2 | 72.1 | 66.2 | 23.3 |

**Along the map's own error direction** (`ε = 1` *is* the ridge memory):

| ε | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 |
|---|---|---|---|---|---|
| identical outputs | 0.38 | 0.11 | 0.02 | 0.00 | 0.00 |
| BLEU | 70.4 | 51.0 | 17.8 | **2.4** | 0.5 |
| CE | 0.33 | 0.62 | 1.74 | **3.32** | 4.92 |

Two readings:

1. **The level set is thin.** A 5 % displacement already changes 22 % of outputs. Exact behavioural
   coincidence is not a broad basin; it is a narrow shell around the correct memory.
2. **The map's error is ~5× the tolerance radius.** BLEU stays within 3 % of the oracle out to about
   ε ≈ 0.2 in a random direction, while the ridge memory sits at ε = 1.0 (it reproduces the ridge
   channel exactly: CE 3.32 vs 3.228, BLEU 2.4 vs 2.50 — the small gap is the attention mask, see
   §7). A least-squares map is not "slightly off"; it is five times outside the region where the
   decoder still behaves.

**Anisotropy** (oracle = `native`, whose memory shares the vi tokenisation with the map, so
positions correspond):

| direction at ε = 0.05 | identical | BLEU | CE |
|---|---|---|---|
| random | 0.41 | 4.07 | 3.402 |
| **`∂CE/∂H`** | **0.00** | **1.42** | **7.164** |

At the same displacement magnitude a random perturbation is invisible (BLEU +0.03, CE +0.006) while
the steepest direction multiplies CE by 2.1 and halves BLEU. **The decoder's sensitivity spans at
least two orders of magnitude in the same ball.**

## 4. Q3 — the map's error is *isotropic*; there is no null-space free lunch

I expected the map's error to concentrate in the few directions the decoder reads (which would
explain the failure). It does not:

| subspace of `∂CE/∂H` at `H_nat` (dim 32 768) | top-1 | top-8 | top-64 | top-256 |
|---|---|---|---|---|
| share of ‖H_ridge − H_nat‖² | 0.0024 % | 0.030 % | 0.277 % | 1.04 % |
| chance level | 0.0031 % | 0.024 % | 0.195 % | 0.781 % |
| enrichment | 0.8× | 1.2× | 1.4× | 1.3× |

So the residual of a least-squares bridge is **direction-blind**: it spreads error evenly, and
therefore necessarily spends ~1/512 of its energy on the single direction the decoder cares most
about. Combined with §3, this is the mechanism the project has been measuring indirectly since E9:
*minimising average vector error does not minimise behavioural error, because the two are related by
a badly conditioned map.* A random (isotropic) error is not a benign error here.

> The claim "the map's error is in the null space, so it should be harmless" is **falsified**.
> There is a large null space, but the least-squares solution does not seek it.

## 5. The intervention — fit the same map on behaviour

`ce_ridge`: W initialised from ridge, then 2 epochs of AdamW (lr 1e-4, batch 32) on the **decoder's
mean per-token CE** of the reference, teacher-forced, decoder frozen, 20 000 train pairs, cached
encoder states. Only `W` (512²) and `b` (512) move — **263 200 parameters, unchanged**.

All rows n = 1 000 held-out test sentences; BLEU/chrF/CE/degen at `max_new=64` (and 48 in brackets).

| arm | what it is | BLEU | chrF | CE | degen | ‖H‖ | zdist | val CE |
|---|---|---|---|---|---|---|---|---|
| `ridge_full` | published E9 map, 85 M token pairs | 2.50 [2.71] | 19.77 | 3.228 | 0.155 | 7.6 | 6.2 | 3.303 |
| `ridge_sub` | same objective, **same 20 k pairs** | 2.47 [2.67] | 19.68 | 3.232 | 0.167 | 7.6 | 6.2 | 3.307 |
| `ridge_recolor10` | E9's best free win, t = 0.10 | 3.08 [3.43] | 20.70 | 3.262 | 0.144 | 8.8 | 8.6 | 3.344 |
| `ridge_scale122` | pure isotropic ×1.22 (matches CE arm's ‖H‖) | 2.89 [3.21] | 20.88 | 3.248 | 0.190 | 9.3 | 7.9 | 3.325 |
| **`ce_ridge`** | **CE objective, 2 epochs** | **5.33 [5.72]** | **25.00** | **2.857** | **0.118** | 9.3 | 10.6 | 2.929 |
| **`ce_ridge_e6`** | **CE objective, 6 epochs** | **7.04 [7.59]** | **27.24** | **2.761** | **0.090** | 11.3 | 14.3 | 2.825 |
| `distill` | CE + KL to the oracle, 2 more epochs | 7.25 [7.77] | 27.57 | 2.741 | 0.104 | 11.9 | 15.4 | 2.809 |
| **`ce_ridge_e16`** | **CE objective, 16 epochs** | **7.90 [8.46]** | **29.36** | **2.738** | **0.067** | 13.8 | 18.8 | 2.797 |
| `ce_scratch` | CE objective from `W = I`, 2 epochs | 2.25 [2.43] | 17.28 | 3.314 | 0.265 | 24.4 | 33.7 | 3.364 |

Paired bootstrap over sentences, 200 resamples, vs `ridge_full` (`results/bootstrap.json`):

| arm | ΔBLEU [95 % CI] | p(Δ>0) |
|---|---|---|
| `ridge_sub` | −0.03 [−0.22, +0.17] | 0.41 |
| `ridge_scale122` | +0.39 [−0.27, +1.19] | 0.82 |
| `ridge_recolor10` | +0.59 [−0.08, +1.42] | 0.93 |
| `ce_scratch` | −0.26 [−0.77, +0.33] | 0.21 |
| **`ce_ridge`** | **+2.83 [+2.29, +3.47]** | **1.00** |
| **`ce_ridge_e6`** | **+4.55 [+3.78, +5.26]** | **1.00** |
| **`distill`** | **+4.76 [+3.92, +5.57]** | **1.00** |
| **`ce_ridge_e16`** | **+5.40 [+4.50, +6.27]** | **1.00** |

The two cheap fixes sit at p ≈ 0.8–0.9 on 1 000 sentences: positive, consistent with E9's direction,
**not individually significant at this sample size** (E9 itself asked for ≥ 3 000 pairs before
tabulating the recolour gain). The CE arms are outside that ambiguity by a factor of five.

For scale: E8's best mapped channel was 2.84 BLEU and E7's (250 M + 220 M models, 768-d, 3.46 M
pairs) was 6.28. `ce_ridge_e6` reaches 7.04 with **two 60 M models and 20 k pairs** — and E7/E8's
absolute numbers are not comparable across phases, so the defensible statement is the within-setup
one: **2.47 → 7.04, a 2.85× increase from changing only the objective.**

### 5.1 Controls — what the gain is *not*

| control | result | share of the 2-epoch gain (+2.86) |
|---|---|---|
| more data | `ridge_sub` = `ridge_full` (−0.03, CI [−0.22, +0.17]) | **0 %** — the objective, not the corpus |
| isotropic scale | ×1.22 → +0.39 BLEU | 14 % |
| covariance matching | E9 recolour t=0.1 → +0.59 | 21 % |
| better initialisation alone | `ce_scratch` = 2.25 (−0.26 [−0.77, +0.33]) | — |

The last row is the interesting negative: **CE from identity does not beat ridge and lands in a
worse basin** (val CE 3.364 vs ridge's 3.307, ‖H‖ 24.4 — it blows the memory scale past the genuine
22.9). The BLEU difference alone is not significant; the val CE and the scale are the tell. So the
two objectives are not interchangeable: the least-squares solution is a poor *answer* but an
excellent *initialisation*. In this setup MSE is doing warm-starting, not modelling.

### 5.2 COMET agrees — it is not a fluency artefact

n = 400 test sentences, `Unbabel/wmt22-comet-da` (`results/comet.json`). BLEU can reward fluent but
unfaithful output — the Oct-31 pilot's failure mode — so an independent learned metric was run on
every arm:

| arm | BLEU (max64) | **COMET** |
|---|---|---|
| `ce_scratch` | 2.25 | 0.4363 |
| `ridge_sub` | 2.47 | 0.4751 |
| `ridge_full` | 2.50 | 0.4797 |
| `ridge_scale122` | 2.89 | 0.4898 |
| `ridge_recolor10` | 3.08 | 0.4966 |
| `ce_ridge` | 5.33 | 0.5331 |
| **`ce_ridge_e6`** | **7.04** | **0.5641** |
| **`distill`** | **7.25** | **0.5678** |
| **`ce_ridge_e16`** | **7.90** | **0.5830** |

**COMET is monotone in BLEU across all seven arms** — every ranking survives a metric that is
neither n-gram nor likelihood based. For scale: E7's mapped channel (250 M + 220 M models, 768-d,
3.46 M pairs) reached COMET 0.600 and `opus-mt-vi-en` 0.752, so the 6-epoch map at 0.564 is closing
on a much bigger pipeline with 60 M models.

### 5.3 Distillation from the oracle memory adds nothing measurable
`distill` = CE + KL(student ‖ teacher) where the teacher is the decoder reading `H_ref` ("the
answer"), warm-started from `ce_ridge_e6` for 2 more epochs. It reaches the best raw numbers of the
night (7.25 BLEU, COMET 0.5678, CE 2.741) — but the KL term is not what did it:

| run | val CE after 2 extra epochs |
|---|---|
| `ce_ridge_e12` (CE only, killed before eval — log survives) | **2.8089** |
| `distill` (CE + KL) | **2.8090** |

Identical to four decimals. So the +0.21 BLEU over `ce_ridge_e6` is the two extra epochs, **not** the
soft targets. This is the mission's most literal question — "can we transport the oracle's
behaviour into the map?" — answered negatively at this capacity: the student's memory cannot
represent a distribution whose CE is 0.30, and forcing it to try adds nothing to the hard label.
The `ce_ridge_e12` arm was killed by a server restart before generation, so its BLEU is unavailable;
only its val-CE trajectory (2.8190 → 2.8089) survives.

### 5.4 Convergence: CE plateaus, BLEU keeps rising

`val CE` per epoch, 16 epochs total (`ce_ridge_e16`, warm-started from `ce_ridge_e6`):

```
2.819  2.809  2.806  2.800  2.807  2.802  2.794  2.799  2.795  2.797
```

It plateaus at **≈2.795 by epoch 4** and then oscillates by ±0.005 — the CE landscape is flat there.
BLEU over the same span: **7.04 (6 epochs) → 7.90 (16 epochs)**, chrF 27.24 → 29.36, degeneration
9.0 % → 6.7 %, COMET 0.5641 → 0.5830. So inside the CE plateau there is still a third of a BLEU of
movement, and CE and BLEU are **not** monotone at the margin — a small second-order version of the
project's own "representation metric ≠ usability" theme.

Practical reading: 2 epochs of CE already captures most of the effect (+2.83 of +5.40); 16 epochs
buys the rest for 20 minutes of GPU. A cosine schedule would probably reach the same place faster,
and the method's ceiling is still unmeasured — CE has converged, BLEU has not stopped moving.

### 5.5 What the improvement looks like (verified by eye, not only by BLEU)

From `NOTEBOOK.ipynb` §5, same test rows:

| | |
|---|---|
| REF | In 1999, Metzen wrote a short story set in the StarCraft universe with fellow Blizzard employee Sam Moore. |
| ridge | In **1894**, the company was re-released by the late John S. Kennedy, … |
| CE (6 ep) | In **1999**, Vicki wrote a short story about the Battle of Rome with the former Fabiano assistant of Granada. |
| REF | Read the instructions on the form and make sure to submit all of the required documents and information. |
| ridge | Use the form to read and read the text and the text. |
| CE (6 ep) | See the instructions for the unit and get the information you need to complete and the information you need to verify. |

The CE map copies the year and the sentence frame; the ridge map invents a date and loops. The
honest counter-example is in the same sample: one CE output still collapses into
`the sex of the sex of the …` (degeneration is 9.0 %, not 0), and on a short sentence
(*"You are important to me."*) the CE map is worse than ridge. The improvement is real but partial,
which is what 7 BLEU on a 22-BLEU problem should look like.

(The first sample row is also a reminder that MTet contains misaligned pairs — the reference itself
carries a stray `â€` and an unrelated sentence. BLEU is depressed for every channel; the *gaps*
survive, the absolutes are bounded. See `CURRENT-PROGRESS.md` §8.8.)

## 6. Replication of E9's recolor result (with one false alarm to record)

E9 (`GEN_VARIANTS.md`) reported ridge 2.73 → ridge+recolor(t=0.1) 3.17. This package initially
measured +0.03 and I nearly recorded a failed replication. The cause was mine: the arm-name
convention is integer percent, so `ridge_recolor01` is **t = 0.01**, not 0.1. With `ridge_recolor10`:

| decoding | ridge | recolor t=0.1 | Δ |
|---|---|---|---|
| `max_new=48` (E9's setting) | 2.67 | **3.43** | **+0.76 (+28 %)** |
| `max_new=64` (this package) | 2.47 | 3.08 | +0.59 (+24 %) |

E9's claim **holds in sign and magnitude**; the transform also matches a literal implementation to
1e-7 (`results/diag_recolor.json`), and its z-distances reproduce E9's units (ridge 6.18 vs 6.31;
genuine 22.86 vs 22.77; full recolour 50.7 vs 61.6). What the bootstrap adds is that on 1 000
sentences the gain is **not individually significant** (+0.59 [−0.08, +1.42], p = 0.93) — exactly
the caveat E9 wrote into its own handoff. It is a real but small effect; behaviour fitting is ~5×
larger.

## 7. Claim boundaries (what this does *not* say)

1. **vi→en only**, one model pair (`t5-small-vietnamese` → `t5-small`), one adapter trained on 80 k
   pairs. Nothing here was tested on ViT5/Flan (E7) or on en→vi.
2. **The CE objective needs parallel data at fit time.** It is supervised MT, not unsupervised
   alignment; the "no new data" claim means *no new pairs beyond the ones already on disk*.
3. **Not competitive MT.** opus-mt-vi-en is 22.3 BLEU on this test set (E10), and the oracle ceiling
   is 74.2. 7.04 is a 2.85× improvement on a weak baseline, not a translation system.
4. **Behavioural coincidence with the oracle is still ≈ 0.** `ce_ridge_e6` reproduces the oracle's
   exact sentence 0.1 % of the time (chrF-to-oracle 26.2 vs ridge's 19.1). The level set of *oracle*
   behaviour is not reached; the CE arms improve within the achievable set.
5. **The mask caveat.** In the perturbation sweep the oracle's attention mask is held fixed, so
   `ε = 1` is "the ridge memory read with the oracle's mask", not the ridge channel. It reproduces
   the ridge channel to 0.1 BLEU / 0.09 CE, which is why the sweep is trusted, but the residual is
   there.
6. **`cos_sent` in `coincidence.json` is invalid** (norms and inner products were taken under
   different masks, giving values outside [−1, 1]). Do not quote it; the curve is the evidence.
7. n = 1 000 for generation (BLEU resolution ≈ ±0.3; the effects reported are 4.5+), n = 300 for the
   perturbation curves, n = 400 for COMET, one seed each. No human evaluation.

## 8. What this means for the paper

The project's methodological thesis (§8.6 of `CURRENT-PROGRESS.md`: *representation similarity ≠
usability*) now has a **constructive** half:

> Not only do representation metrics fail to rank bridges, they are the wrong *loss*. Fitting the
> same bridge on the decoder's cross-entropy — a 0.26 M-parameter change of objective, no new data,
> no new parameters — multiplies BLEU by 2.85 and reduces degeneration by half. The mechanism is
> measured: the decoder's behavioural tolerance is ~5× smaller than a least-squares bridge's error,
> and that error is isotropic, so no null-space argument rescues it.

This is also a clean answer to the neural-mining question ("why does linear keep winning?"): the
linear class was never the problem; the *estimator* was. `ce_ridge` is still linear, still 263 k
parameters, and now three times better.

## 9. Next steps, ranked by value per hour

1. **Train to convergence** with a schedule (cosine, 20 epochs). val CE is still falling; the
   ceiling of this method is unknown and it is the cheapest open number.
2. **Distillation arm** (KL to the decoder's distribution over the oracle memory) — implemented in
   `src/fit_behaviour.py` (`--arms distill --init-map ce_ridge_e6`). If soft targets beat hard labels,
   this is the literal form of the mission's "transport the oracle's behaviour" idea.
3. **Two adapters instead of one mixed adapter** (open since E7/E8) — the CE objective now depends
   on the adapter, so this is no longer cosmetic.
4. **en→vi** with the same recipe: the `trans-t5-small-en-vi` adapter exists.
5. **Test-time refinement**: gradient steps on `δ` at inference to measure how much of the remaining
   CE gap is reachable at all (uses the reference — a diagnostic, not a method).

## 10. Reproduce

```powershell
cd coincidence-solution
$py = "..\research-scale-100k\.venv\Scripts\python.exe"

& $py -m src.cache_states          # 150 s, 6.5 GB in agent-work/cache (skips what exists)
& $py -m src.validate              # calibration vs E9      -> results/calibration.json
& $py -m src.coincidence --n 300   # Q1/Q2/Q3               -> results/coincidence.json   (12 min)
& $py -m src.fit_behaviour --arms ridge_full,ridge_sub,ridge_recolor10,ridge_scale122,ce_ridge,ce_scratch
& $py -m src.fit_behaviour --arms ce_ridge_e6 --init-map ce_ridge --epochs 4
& $py -m src.analyse_basis         # oracle overlap + subspace -> results/basis_analysis.json
& $py -m src.eval_maxnew48         # comparable-to-E9 table    -> results/maxnew48.json
& $py -m src.bootstrap_all         # CIs                      -> results/bootstrap.json
& $py -m src.diag_recolor          # fold == literal check     -> results/diag_recolor.json
& $py -m src.diag_maxnew           # decoding confound         -> results/diag_maxnew.json
& $py -m src.eval_comet --n 400   # independent metric        -> results/comet.json
```

Every script is idempotent: caches and `.npz` maps are skipped if present, `--force` rebuilds.
Total GPU time for the whole night ≈ 75 min on an RTX 3050 4 GB.

## 11. Artefacts

| path | what |
|---|---|
| `results/calibration.json` | the three channels vs E9's published numbers |
| `results/coincidence.json` | perturbation curves + anisotropy + subspace enrichment |
| `results/behaviour_maps.json` | every arm's metrics, training history, bootstrap |
| `results/maxnew48.json` | the same arms at E9's decoding setting |
| `results/basis_analysis.json` | oracle overlap, gradient spectrum, per-arm subspace energy |
| `results/bootstrap.json` | paired CIs vs `ridge_full` |
| `results/comet.json` | COMET per arm (n=400) |
| `results/diag_recolor.json`, `results/diag_maxnew.json` | the two validity checks |
| `results/train_*.csv` | per-25-step training logs |
| `agent-work/maps/*.npz` | fitted (W, b) per arm — **committed**, 1 MB each |
| `agent-work/hyps/*.json` | generated hypotheses per arm (n=1000) |
| `agent-work/cache/` | cached encoder states, 6.5 GB — **gitignored**, rebuildable in 150 s |
