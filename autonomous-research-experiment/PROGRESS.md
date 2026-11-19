# PROGRESS — autonomous research experiment (2026-11-19 → 20)

**Deadline:** 07:00 (VN). Hard stop of GPU work 05:30, documentation done 06:30.
**Objective agreed with the user:** maximise *insight* toward the highest BLEU; scientific novelty is a
constraint, not an optional extra. Money: ≤ $3 (API), only `coding`.

## Decisions locked with the user (their words in quotes)

| # | decision | value |
|---|---|---|
| 1 | fine-tune the 77M as a *claim* | **forbidden** — "no novelty or new knowledge" |
| 2 | base model | frozen `Helsinki-NLP/opus-mt-vi-en` (77M) |
| 3 | second model | frozen `Helsinki-NLP/opus-mt-en-vi` (reverse) — 2 models total |
| 4 | training | tiny module ≤ 5 M params allowed; LoRA only as a *labelled control* |
| 5 | data | MTet only tonight (3.46 M pairs); culinary = qualitative only |
| 6 | metric bank | MTet test, **10 k** sentences (was 2 k), COMET on a sub-sample |
| 7 | output | this folder; notebook + report + handoff + one `AGENT:` commit |
| 8 | top insight wanted | **"Is latent agreement a usable quality signal?"** |

## Data / splits

`data/split.parquet` — 3,456,741 pairs, rebuilt from `research-phase01/data/pairs.parquet` by
**rank of md5(vi)** (no RNG): test 10,000 (ranks 0–9,999), val 10,000 (10,000–19,999),
train 3,436,741. The historical 2 k test = ranks 0–1,999 (`core=True`), so the published
**22.27 BLEU** for `opus-mt-vi-en` remains quotable on exactly those sentences.

## Run log (append-only, newest last)

| t | what | result |
|---|---|---|
| 22:20 | read mission + CURRENT-PROGRESS + hardware | 2 decisions blocked, asked user |
| 22:40 | closed Edge/Zalo/ChatGPT/ChatGPT-Classic | +1.3 GB RAM (3.7 GB in use, 2.2 GB free) |
| 22:45 | downloaded `opus-mt-en-vi` to project-local HF cache (D:) | 16 s, avoids C: |
| 22:47 | built `data/split.parquet` | test 10k / val 10k / train 3.44M |
| 22:50 | smoke: greedy generation | 26–34 sent/s, batch 32 fp16, 4 GB VRAM fits |
| 22:53 | **OOM** in `compute_transition_scores` (batch×len×vocab in fp32) | dropped it; use `sequences_scores` (not populated for Marian) → `lp=nan` |
| 22:54 | **self-inflicted stall**: a PowerShell chain kept running after step 1 failed, colliding with a smoke test on the same GPU | killed stray pythons; rule reinforced: one chain, explicit `logs/*.done` sentinel |
| 22:56 | relaunched chain 1: greedy 10 k → beam-8 10 k → beam-8 5 k (train) | running |
| 22:59 | **second self-inflicted stall**: the earlier PowerShell chain had *not* died (only its python child did), so it kept spawning the remaining steps and competed with chain 1 for the GPU (greedy dropped 34 → 12 sent/s) | killed the whole process subtree by PID; lesson: kills must target the shell, not the child, and a chain must own a `.done` sentinel |
| 22:59 | pool sanity (200-row beam-8 smoke): 7.97/8 unique candidates, mean pairwise chrF 80.9 | enough diversity for reranking to have a ceiling, not much |
| 23:00 | spotted a **misaligned MTet pair** in the smoke sample (reference about a questionnaire, source about cognitive development) | re-confirms the phase-01 warning; quantify the share in the report, do not quote absolute BLEU as translation quality |
| 23:05 | greedy pool done (10 k) | 4.1 MB |
| 23:25 | beam-8 pool done (10 k) | 19.5 min; **7.96/8 unique strings per row, mean pairwise chrF 78.1** (pool is redundant → reranking headroom is bounded) |
| 23:37 | beam-8 pool done (5 k train, for the tiny ranker) | 10.7 min |
| 23:39 | **reproduction check**: greedy on the historical 2 k core → BLEU **22.29** vs the published **22.27** | the pipeline is calibrated; beam-8 top-1 makes it **23.32 (+1.03)** |
| 23:38 | queue 2 started: features (79 568 unique back-translations at ~30 sent/s) → COMET labels | running |
| 00:05 | **crash**: `MarianMTModel` has no `.encoder` (it is `.model.encoder` / `get_encoder()`); the 24 min of back-translation was lost because it lived only in RAM | fixed; back-translation now checkpoints to `results/bt_*.json` every 10 k strings — a crash costs 10 k, not 80 k |
| 00:06 | queue 2 restarted | bt at 55 sent/s after warm-up |
| 00:07 | **budget check** (mission asks for it): OpenRouter `coding` key — used **$0.16 of $3.00**, $2.84 left | no spend on GPU work; this is reasoning only |
| 00:37 | features (test) done: 80 000 candidate rows, 8.2 MB | encode ≈ 23 rows/s after back-translation |
| 00:44 | all reference-free selections written | |
| 00:47 | **first arm table (10 k test, no COMET yet)** | see below |
| 00:48 | **third process-tree incident**: the 23:38 queue-2 shell had survived its crashed child and was running `comet_test` in parallel with the 00:06 queue-2 (same trap as 22:59) | killed the wrapper, let the orphan finish; noted again in LEARNING_NOTE §4 |
| 01:00 | `--boot 1000` on 10 k sentences costs ~5 min *per arm* (sacrebleu re-tokenises every resample) | killed, restarted with `--boot 200`; the 1 000-resample version is a 40-minute luxury, not a necessity |

### First arm table — 10 000 test sentences, frozen `opus-mt-vi-en`

| arm | BLEU | chrF | reading |
|---|---|---|---|
| A0 greedy (historical baseline) | 21.84 | 44.93 | calibrates against 22.27 on the 2 k core |
| A1 beam-8 top-1 | 22.85 | 45.89 | **+1.01 BLEU for free** |
| A3 length-ratio closest to 1 | 22.55 | 46.39 | naive length prior: worse BLEU, better chrF |
| A5 chrF-MBR (consensus) | **23.04** | **46.59** | best training-free arm so far |
| A6 string round-trip | 22.29 | 46.02 | loses 0.56 BLEU vs beam |
| A7 latent round-trip (uncentred) | 22.31 | 45.98 | **+0.02 BLEU vs string = no advantage** |
| A7c latent round-trip (centred) | 22.32 | 46.00 | centring does not rescue it |

Both round-trip families buy chrF (46.0 vs 45.89) and pay BLEU (22.3 vs 22.85): they reward
*coverage/recall* over *precision*. That is the first substantive finding of the night.

### 01:40 — the central result (test pool, 10 k sentences)

Candidate-level Spearman against COMET (80 000 candidates, 10 k sentences):

| utility | Spearman ρ | top-1 accuracy |
|---|---|---|
| string round-trip (chrF of back-translation vs source) | +0.394 | 0.356 |
| **latent round-trip** (token-level NN agreement, same encoder) | **+0.478** | **0.370** |
| latent, centred | +0.480 | 0.369 |
| learned on string only (2 433 params) | +0.496 ± 0.001 | 0.353 |
| learned on string + latent (3 137 params) | **+0.557** ± 0.001 | 0.355 |

**H1 is confirmed at the scoring level**: Δρ(latent − string) = **+0.083 [+0.075, +0.092]** (300
bootstrap resamples over *sentences*), and adding latent features to the tiny ranker buys another
+0.06 ρ over seed noise (sd 0.001).

**But it does not survive the usability test.** Selections from the same pool:

| arm | BLEU | Δ vs beam [95 % CI] |
|---|---|---|
| beam-8 top-1 | 22.85 | — |
| chrF-MBR | **23.04** | **+0.19 [+0.07, +0.32]** |
| string round-trip | 22.29 | −0.56 [−0.71, −0.40] |
| latent round-trip | 22.31 | −0.54 [−0.70, −0.38] |
| learned (string) | 22.09 | −0.76 [−0.92, −0.59] |
| learned (string+latent) | 22.15 | −0.70 [−0.88, −0.53] |
| **oracle (true COMET)** | **26.23** | **+3.38 [+3.22, +3.56]** |

The frozen 77 M model's own 8-best pool **already contains a 26.23-BLEU answer**; the best
reference-free rule converts 5.6 % of that headroom. Latent agreement is a better *scorer* and not a
better *selector*: +0.083 ρ buys +0.02 BLEU (inside noise). Even a ranker trained on COMET labels
with references loses 0.70 BLEU to plain beam search.

| 01:57 | LoRA control trained: 1 179 648 params (r=16 on q/k/v/o, 6+6 layers), 2 500 steps = 60 k pairs, **7.5 min** at 134 pairs/s | loss ended ~2.3; beam-8 generation with the adapter started |
| 02:04 | resources: GPU 52 °C, 28 %, RAM 1.85 GB free, CPU 50 % | healthy; one job at a time |
| 02:04 | **budget**: $0.249 of $3.00 used | local GPU work costs nothing |
| 02:33 | LoRA control generated (beam-8 + greedy, 10 k) | **BLEU 26.50 / 25.66**, COMET 0.7839 / 0.7742 — beats the frozen model's *oracle* (26.23) on BLEU |
| 02:42 | **length-penalty hypothesis tested and rejected**: lp 1.2 → +0.11 [−0.09, +0.34], lp 1.5 → +0.06 [−0.20, +0.33] on the 2 k core | under-length is real but is not caused by lp; the mechanism story in REPORT §7 was too simple |
| 02:50 | **complementarity hypothesis tested and rejected**: chrF-MBR on the *adapted* pool | 25.81 BLEU (−0.69 vs 26.50), COMET 0.7828 — MBR's +0.19 did not transfer |
| 02:52 | final tables regenerated; report, learning note, handoff written | commit follows |
| 03:00 | **diverse-pool test** (2 k core, beam-8 + 8 samples = 16 candidates, 14.94 distinct/row): ceiling **26.40** vs 26.54 for beam-8; MBR 23.39 (vs 23.34); round-trip rules **−1.4 BLEU** | the ceiling is set by candidate *quality*, not count; more diversity actively hurts adequacy-style signals |
| 03:05 | COMET-Kiwi arm attempted | **not run**: `Unbabel/wmt22-cometkiwi-da` is a gated repo, `GatedRepoError 401` (no HF credentials here). Logged as a gap, not quietly dropped |
| 03:20 | notebook executed end-to-end (`NOTEBOOK_executed.ipynb`); 14 quoted numbers re-checked against the JSONs (0 mismatches) | |
| 03:32 | **replication on a second pool** (the adapted model's, 2 k / 16 k candidates): string ρ +0.286, latent ρ +0.362, **Δ +0.076** (frozen pool: +0.083); top-1 0.374 → 0.392 | the gap is a property of the signal, not of one pool |
| 03:35 | selection replicated too: on the adapted pool, latent 25.89 / MBR 25.85 / string 25.71, all below its own top-1 26.66 | same verdict; latent loses least |

## Final answer to the user's question, in one line

Latent agreement is a significantly **better scorer** than string agreement (Δρ +0.083 [+0.075,
+0.092]) and an equally poor **selector** (BLEU +0.02, both below plain beam); the frozen model's own
8-best pool hides 3.38 BLEU that no reference-free rule could extract, and 7.5 minutes of 1.2 M-param
LoRA bought 3.65 — more than the entire selection headroom.
| 22:55 | pre-registered the hypotheses (`HYPOTHESES.md`) before any test number existed | — |

## Deviations from the pre-registration

| t | deviation | reason |
|---|---|---|
| 22:53 | `lp` (teacher-forced model confidence) is dropped from the feature set | `compute_transition_scores` OOMs on 4 GB; recomputing teacher-forced log-probs for 80 k candidate pairs costs ~20 min GPU, and MBR-chrF already covers the "consensus confidence" idea. Logged, not hidden. |

## Standing risks

- RAM is the binding constraint (2.2 GB free); no browser, one GPU job at a time.
- COMET (`Unbabel/wmt22-comet-da`, XLM-R large) must share the 4 GB card → batch 16–32.
- `pyarrow` filters on a single file fail in this version → always filter in Python.
- transformers 4.57.6: `dtype=` not `torch_dtype=`; greedy `generate` has no `beam_indices`.
