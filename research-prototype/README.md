# research-prototype

Baseline prototype for the "copy first, then map embeddings" methodology: can
ViT5 / Flan-T5 **repeat** a sentence when told to, and what does their
perplexity say about dish names?

Scope: **no fine-tuning, no layer mapping** here. Those are
`small-experiment` (mapping + probing) and `small-experiment-02` (vector->token
readout). This folder answers only the gate question: *is the copy prompt usable
at all, and where does the model's knowledge sit?*

## Layout

```
research-prototype/
  src/                 reusable code (notebooks are thin wrappers)
    config.py          paths, model registry, prompt formats, seeds
    data.py            validates + writes the 100-pair culinary set
    models.py          loading with an explicit device/dtype report
    copy_eval.py       2k copy test (resumable) + metrics
    score.py           pseudo-perplexity (dish vs rest) + zero-shot BLEU
    run_all.py         one CLI entry point
  data/                culinary_100_pairs.jsonl (authored) + .csv (derived)
  results/             metrics CSVs + raw generations (committed = evidence)
  model-exploratory/   one notebook per model
  model-instruction-following/
  test-token-generation/
  models/hf/           HF cache (gitignored, D: only)
```

## Reproduce from zero

```powershell
cd research-prototype
.\setup.ps1                      # creates .venv (py 3.12), installs pinned reqs, prints device
python -m src.run_all data       # validate + write the 100 pairs
python -m src.run_all copy       # 2,000 sentences x 4 models (resumable)
python -m src.run_all ppl        # pseudo-perplexity, dish vs non-dish
python -m src.run_all translate  # zero-shot BLEU, base<->base and large<->large
```

Re-running any stage skips finished work. `--force` recomputes. To start clean,
delete `results\` (derived) — never `data\` (authored).

## Reading the numbers

* `exact_match` — the model returned the sentence verbatim. The copy gate.
* `bleu` on the copy task — 100 means perfect copying. Any gap is the cost of
  the prompt itself, before any mapping is trained.
* `ppl_dish` vs `ppl_other` — within-sentence comparison, paired bootstrap CI.
  Not comparable across models in absolute terms; the *delta* is the signal.
* `rank_dish` — mean rank of the true dish-name token in the model's own
  distribution. A rank of 3 means "the model would have produced it in its top 3".

Model pairing follows the mission: **base with base, large with large**.
