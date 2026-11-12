# Learning note — ViT5 / Flan-T5 baseline prototype

## 1. What we were actually trying to do

The paper's idea is to translate between Vietnamese and English **without training a translator**: run a
sentence through one model, take a vector from one of its layers, push that vector through a small learned
map into the other model's vector space, and let the second model continue generating.

Before any of that can work, there is a cheaper question that decides everything downstream:

> Can a small pretrained T5-family model **repeat a sentence it is given**?

If it cannot repeat, you cannot ask it to "read the vector and speak", because the reading-out step is
exactly a repetition under a different input channel. So this prototype measures three things: the copy
gate, how the models cut dish names into tokens, and whether the models find dish names surprising
(perplexity / copy-loss). Everything else — the map, the probing, the layer choice — is deliberately out
of scope here.

## 2. How to reproduce it from zero

> **Delete first, or you will silently reuse stale output.**
> `Remove-Item -Recurse -Force research-prototype\results` deletes derived results only.
> Never delete `research-prototype\data\` — that is authored source, not derived.

```powershell
cd D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\research-prototype

# 1. environment (idempotent; -Reset rebuilds the venv)
.\setup.ps1                 # py 3.12 venv + pinned requirements + kernelspec; prints device

# 2. check you are inside the venv, and that the GPU is real
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.prefix); print(torch.cuda.is_available())"

# 3. models (~7.6 GB, allow_patterns skips the TF/Flax duplicates)
$env:HF_HOME = "$PWD\models\hf"
.\.venv\Scripts\python.exe -m src.fetch_models

# 4. the four stages (each is resumable; re-running skips finished work)
.\.venv\Scripts\python.exe -m src.run_all data
.\.venv\Scripts\python.exe -m src.run_all copy
.\.venv\Scripts\python.exe -m src.run_all ppl
.\.venv\Scripts\python.exe -m src.run_all translate

# 5. notebooks (they read results/*.csv; the live cells re-run small samples)
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace ^
  --ExecutePreprocessor.kernel_name=research-prototype --ExecutePreprocessor.timeout=900 ^
  model-exploratory\00_compare_models.ipynb
#    ... repeat for each notebook, or loop over the three folders
```

Verification, not vibes:

```powershell
Get-ChildItem results\*.csv | Select-Object Name, Length
Get-Content results\copy_summary.csv   # exact_match / bleu / chrf per model
```

To force a recomputation: add `--force` to a stage, e.g. `python -m src.run_all copy --force`.
`--models vit5-base` limits a stage to one model.

## 3. Decisions and trade-offs

| Decision | Alternative | Why this one | What it costs |
|---|---|---|---|
| 100 hand-authored **parallel** pairs (vi+en, same dish) | 200 independent sentences (literal mission text) | gives BLEU a reference **and** gives perplexity a within-sentence control | the references are one annotator's taste; BLEU is bounded by them |
| Zero-shot translation keeps the dish name transliterated (`pho bo`) | translate it descriptively (`beef noodle soup`) | matches the paper's cultural-preservation stance | BLEU rewards keeping the name; a descriptive reference would score differently |
| Metric = **copy-loss** `P(token | token<t, full sentence in encoder)` | corpus language-model perplexity | a T5 encoder sees the whole sentence, so true LM perplexity is not available; copy-loss is the model's own estimate of "how hard is this token to reproduce", which is exactly the copy gate | absolute values are **not** comparable with literature PPL; only compare within a model |
| `bfloat16` | `float16` | fp16 is numerically unstable for T5-family heads | slightly slower on some GPUs |
| greedy decoding (`num_beams=1`) | beam search | 8,000 generations inside a 3 h box | BLEU is pessimistic; exact-match/copy measurement is unaffected |
| `max_new_tokens = input + 8`, capped 128 | fixed 512 | a legitimate copy needs only its own length; a runaway loop is truncated instead of eating the budget | a model that needs >8 extra tokens to finish is scored as truncated — that is visible in `mean_hyp_words` |
| copy test on a fixed seed-42 sample of 2,000 of 2,536 test pairs | all 2,536 | mission asked for 2k; determinism lets a re-run resume by id | 536 pairs unusable in the headline number |
| DuckDB over the parquet | MySQL/Postgres in Docker | Docker engine was not running; the DB would cost 1–2 GB RAM on a 7.7 GB machine, and the parquet holds the same 2,536 rows the mission names | no server-side SQL; no shared DB for other tools |
| `src/*.py` + thin notebooks | everything inside notebooks | a 20-minute generation cannot live in a cell that gets lost on compaction (doc-50) | one more layer to read |
| base↔base, large↔large pairing | any mix | mission instruction; also `d_model` must match (768↔768, 1024↔1024) for a future map | fewer combinations explored |

## 4. What went wrong, and the lesson

**4.1 Vietnamese text destroyed by a PowerShell pipeline**
- *Symptom:* `dish_vi 'b\ufffdnh x\ufffdo' not in vi` for 90 of 100 rows.
- *Cause:* `Get-Content part1.jsonl, part2.jsonl | Set-Content merged.jsonl` round-trips text through the
  console codepage (cp1252), not UTF-8. The source files were fine; the merge was not.
- *Fix:* merge with Python (`read_text(encoding='utf-8')` → `write_text(..., encoding='utf-8')`).
- *Rule:* **never merge or transform UTF-8 content through a PowerShell text pipeline**; PowerShell is for
  process control, Python is for text.

**4.2 BOM that only `json.loads` notices**
- *Symptom:* `Unexpected UTF-8 BOM (decode using utf-8-sig)`.
- *Cause:* PowerShell 5.1 `Set-Content -Encoding UTF8` writes a BOM; Python's `json.loads` rejects it.
- *Rule:* when a file is written by PowerShell and read by a parser, either strip the BOM
  (`encoding='utf-8-sig'`) or write the file from Python.

**4.3 17% of sentences silently swapped languages**
- *Symptom:* ViT5 received English, Flan-T5 received Vietnamese.
- *Cause:* the MTET test split has **six** prompt templates; my direction check matched only four and
  missed `Translate the following sentence into English:` (433/2536 rows).
- *Fix:* match all six, then cross-check with diacritics and *count* any swap.
- *Rule:* **a smoke test must print the data, not just the timing.** This bug would have looked like a
  scientific result ("Flan-T5 copies worse than ViT5") for the rest of the mission.

**4.4 Every perplexity number was inflated by one token of context**
- *Symptom:* NLL ≈ 17 nats for ` has` in an easy sentence; the dish name looked harder than it was.
- *Cause:* T5 shifts internally (`decoder_input_ids = [pad, l0 … l_{L-2}]`), so `logits[j]` already predicts
  `labels[j]`. I applied a second shift, asking the model for the token *after* next.
- *Fix:* drop the manual shift; verify by eyeballing one sentence and checking that the dish-token spans
  concatenate back to the dish name.
- *Rule:* when you compute a loss by hand, **align it against the library's own shift convention** and
  print one example you can check with your eyes.

**4.5 1.6 MB/s, then 34 MB/s**
- *Symptom:* `pip` downloaded nothing for 5 minutes; the 3.27 GB torch wheel implied ~35 min.
- *Cause:* Wi-Fi throughput, not a broken process (the adapter showed 1.8 MB/s; `hf_transfer` crashed on
  this network).
- *Fix:* avoid re-downloading what already exists locally; use `allow_patterns` on the HF repos; later the
  laptop was moved and the link became 34 MB/s.
- *Rule:* **measure the link before planning around it, and re-measure after any change.** A bandwidth
  number is a resource like RAM.

**4.6 Console scripts vanish when you move `site-packages`**
- *Symptom:* `Jupyter command jupyter-nbconvert not found`.
- *Cause:* the venv was seeded by moving `site-packages` from the previous experiment's venv (to save a
  3.3 GB download at 1.6 MB/s). The `.dist-info` came along, but the `Scripts\*.exe` shims did not.
- *Fix:* `python -m ipykernel install --prefix .venv --name research-prototype`, then always call
  `python -m nbconvert` / `python -m pip`.
- *Rule:* a venv is `pyvenv.cfg` + `Scripts/` + `Lib/site-packages/`; moving only `site-packages` gives you
  imports but not entry points. Prefer `pip install` when bandwidth allows.

## 5. If you want to go deeper

- **Span corruption / sentinels.** T5 is pretrained by corrupting spans and generating `<extra_id_k>`
  tokens. That is why ViT5 emits `*`, `Z` and `_` when asked to copy: you are prompting a denoiser with a
  task it was never trained for. Read the T5 paper's §3.1 and look at the tokenizer's `<extra_id_*>` ids.
- **Teacher forcing and the shift convention.** The whole 4.4 bug is this: next-token prediction means
  `logits[t]` scores `labels[t]`, which only holds if the decoder input is shifted right by one. Read the
  `shift_tokens_right` function in `transformers`; it is 10 lines and prevents a class of silent errors.
- **Pseudo-perplexity vs corpus perplexity.** "Perplexity" of one sentence under a seq2seq model is not a
  standard quantity. Whatever definition you choose, state the conditioning (here: the full sentence in the
  encoder) or the number is meaningless to a reviewer.
- **BLEU tokenisation.** English BLEU uses `13a`; Vietnamese has no standard tokeniser, so `--tokenize char`
  is the common choice. Changing the tokeniser can move a BLEU by several points — never compare BLEU
  across papers without checking it.
- **Anisotropy in embedding spaces.** From `small-experiment-02`: raw cosine similarity between two
  languages is dominated by a shared mean direction (`||mu|| = 10.8` vs mean norm `15.5`). Centering changes
  a "0.80" cosine into "0.52". Any future map must report whether the mean was removed.
