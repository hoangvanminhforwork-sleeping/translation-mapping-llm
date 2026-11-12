# Learning note — LoRA copy fine-tuning

## 1. What we were actually trying to do

The mapping idea needs a *read-out*: hand the target model a vector and let it speak. Before that, the same
model must be able to do the trivial version of the job — read a sentence and repeat it. `research-prototype`
showed it cannot (ViT5-base: 0 % exact copies; Flan-T5-base: 2.65 %). This project teaches the copy task
with a LoRA adapter, then re-runs the *same* 2,000 sentences to see whether the gate opens.

## 2. How to reproduce it from zero

> **Delete first:** `Remove-Item -Recurse -Force adapters\vit5-base` (or pass `--force`) if you want a fresh
> adapter. Deleting `data\train_copy_50000.jsonl` is safe — it is rebuilt in ~20 s. Never delete
> `research-prototype\results\raw\*`; the eval set is copied from those files.

```powershell
cd D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\finetune-and-lora\prototype
.\setup.ps1                                     # creates ..\.venv, installs peft etc., prints device
..\.venv\Scripts\python.exe -m src.run_all data
..\.venv\Scripts\python.exe -m src.run_all train --models vit5-base
..\.venv\Scripts\python.exe -m src.run_all eval  --models vit5-base
..\.venv\Scripts\python.exe -m src.run_all compare
```

Verification:

```powershell
Get-Content results\before_after.csv
Get-Content results\raw\ft_vit5-base.jsonl -TotalCount 3
```

Resume: adapters are written every 500 steps with `progress.json`; the next run continues from there.
`--force` starts over.

## 3. Decisions and trade-offs

| Decision | Alternative | Why | Cost |
|---|---|---|---|
| LoRA `r=16` on `q`,`v` | full fine-tune | 0.78 % trainable (1.77 M / 227.7 M), ~1 MB adapter, and if copying is reachable it is reachable here | if LoRA fails we cannot tell whether the task is unreachable or the adapter too small |
| 50k examples, **1 epoch** | 3 epochs | time box | under-training is possible; the loss curve in `results/train_*.csv` shows whether it had converged |
| Train on the **same 50k pairs** for both languages | separate corpora | the two models then see identical content, so a difference is a model difference | less data diversity per language |
| Eval on the **exact baseline sentences** | fresh sample | paired comparison: same prompt, same sentence, only the adapter differs | the eval set is "easy" in the sense that we already know the failure modes |
| greedy decode at eval | beam | identical decoding to the baseline | pessimistic; a beam would look better |
| `max_len=64` + dynamic padding | fixed 128 | most sentences fit; padding costs time linearly | sentences >64 tokens are truncated (visible in the data) |
| adapter checkpoints every 500 steps | only at the end | a kill at minute 11 of 12 must not cost the whole run | a few MB of temporary files |

## 4. What went wrong, and the lesson

- **The GPU was busy with the previous mission.** Fine-tuning had to wait for the baseline, or the baseline
  had to be stopped. Because every stage is resumable, stopping was cheap — but only because the raw
  generations are appended line-by-line with a flush. *Lesson:* resumability is what makes a priority change
  affordable mid-session.
- **The eval set depended on a CSV that did not exist yet** (the baseline job writes per-model CSVs only at
  the end). Fixed by reading the **raw JSONL**, which is the primary evidence and exists as soon as the first
  batch lands. *Lesson:* point downstream work at the most primitive artifact, not at a derived view.
- **`torch` would have been re-downloaded (3.3 GB).** The venv was seeded by copying `site-packages` from the
  working environment, because moving it would have broken the process using it. *Lesson:* copy, do not move,
  when the source environment is in use; and record the deviation in the report.
- **Clock discipline.** A 40-line markdown file took ~50 minutes of wall-clock in the first mission. Writing
  prose while a job runs is free; writing prose while the GPU idles is not. *Lesson:* launch the long job
  first, then write.

## 5. If you want to go deeper

- **LoRA mechanics.** `W + (alpha/r)·BA`: the adapter is a low-rank update, and `alpha/r` is the effective
  learning-rate multiplier. Read the LoRA paper §4 for why `r` small suffices for style/format tasks.
- **Why copying is easy but not free.** Copying is a routing problem (attend to the source token, emit it).
  T5 already has the attention path; the adapter only has to make the decoder *prefer* it over the
  pretrained denoising prior — that is why a tiny adapter can flip the behaviour.
- **Exact match vs BLEU.** Exact match is brutal and appropriate here: the task is copying, so a paraphrase
  is a failure. BLEU is reported only to see *how far* a failure is from the reference.
- **Prompt-format sensitivity.** `Nói:`/`Say:` are arbitrary strings; the model has never seen them. A
  different prefix would give a different baseline. This is the same reason FLAN's instruction tuning used
  thousands of templates.
- **Catastrophic forgetting is not measured here.** The adapter may damage translation ability. A proper
  follow-up evaluates the adapter on the MTET test split as well as on the copy task.
