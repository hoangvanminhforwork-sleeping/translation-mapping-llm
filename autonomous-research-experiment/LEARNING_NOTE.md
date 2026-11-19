# Learning note — what this night taught, and how to judge it yourself

*Written for a human reader (student) who wants the judgement, not just the numbers. Filled in
progressively; the last section is written after the results existed.*

## 1. The decision that shaped everything

The mission asked for the highest BLEU. The evidence in `CURRENT-PROGRESS.md` says the project's own
architecture plateaus at **3–6 BLEU** while a frozen off-the-shelf 77 M model already gets **22.27**.
So "maximise BLEU" and "keep the map architecture" were incompatible within 8 hours. The user then
ruled that *fine-tuning the 77 M is not research* — it is engineering with a known outcome.

That left exactly one honest design space: **keep the model frozen and make the *choice* of output
better**. Selection/reranking is where a frozen model can still be improved, and selection is a
research question, not a tuning knob — as long as the selection rule is principled and the gain is
measured against a pool everyone shares.

> **Transferable rule:** when compute forbids training, the scientific object becomes the *decision
> rule*, not the model. Design the experiment so every arm shares one pool; then every difference is
> attributable to the rule.

## 2. Why the question is "latent vs string", not "does reranking help"

Reranking with MBR or with a QE model is known to work (published since 2020). Reproducing it earns
nothing. The open question — and the one this project has been circling for six phases without asking
it directly — is whether *representation-space* agreement carries information that *surface-string*
agreement does not.

The project's earlier phases found that representation similarity **failed** to rank mappers, and that
"similarity ≠ usability" (CURRENT-PROGRESS §8.6). Tonight asks the same question one level down: does
it fail to rank *hypotheses* too? Either answer is informative, which is why the experiment is
pre-registered (`HYPOTHESES.md`) rather than written up afterwards.

## 3. Design choices worth defending (and their costs)

| choice | why | cost paid |
|---|---|---|
| 10 k test instead of 2 k | ±0.5 BLEU differences are invisible at n=2 000 | 5× generation/comet time; solved by one shared pool |
| keep the historical 2 k as `core` | the 22.27 number must stay comparable | tiny extra bookkeeping |
| one pool, many selectors | paired comparison by construction; no re-generation per arm | pool must be good enough for all arms (beam-8) |
| latent comparison inside **A's encoder** | source and back-translation are both Vietnamese, so one frozen encoder gives one shared space — no cross-model alignment problem | the signal cannot see English-side quality directly |
| centred latent features | the project already learned that an uncentred cosine is dominated by the mean direction | one extra feature family to report |
| tiny MLP ranker (≤ 5 M) | tests whether the signals *combine*, not whether a big model can memorise | 3 seeds needed to separate signal from init noise |
| LoRA **control** | without it, "selection captured 60 % of headroom" has no meaning | ~1 h GPU, explicitly not claimed as novel |

## 4. Traps that cost time tonight (all mine)

1. `compute_transition_scores` on a 4 GB card: it materialises `batch × len × vocab` in fp32. One
   batch of 32 × 96 × 65 k = ~800 MB → OOM. Fix: use `sequences_scores`, or teacher-force later.
2. `generate()` returns `beam_indices` only for beam search; `sequences_scores` only when requested,
   and **absent** (not `None`) for greedy Marian. Always `getattr(x, "y", None)`.
3. A PowerShell chain is not a process tree you can kill by killing `python.exe`: the shell survives
   and starts the next step, which then competes for the GPU. Symptom: throughput halves, not an
   error. Always kill the *shell* PID and use a `.done` sentinel file.
4. Console encoding: printing Vietnamese from a `python -c` on Windows crashes with cp1252 unless
   `PYTHONIOENCODING=utf-8` — it looks like a data bug, it is a terminal bug.
5. `pyarrow` `filters=` on a single parquet file (not a dataset) raises a schema error in this
   version; filtering row groups in Python is both simpler and memory-safe.

## 5. How to read the results table without fooling yourself

- Compare arms **within** the same pool. A BLEU that crosses pools (greedy vs beam vs LoRA) is a
  different experiment, not a better one.
- "Headroom captured" is the number that decides usefulness. A utility with +0.3 Spearman that picks
  the same candidate as the top-1 in 70 % of rows converts almost nothing.
- Watch `degenerate_rate`: a reranker that rewards length can win BLEU by emitting loops.
- MTet contains misaligned pairs. If the best-of-8 COMET is low for a row, the row is suspect
  regardless of which arm won it; the report states the share.

## 6. What I would tell the next agent (and the reviewer)

*(written after the results existed)*

1. **The instrument mattered more than the idea.** Reproducing 22.29/0.7522 against the published
   22.27/0.752 turned an interesting night into a comparable one. Calibrate first; a result that
   cannot be anchored to a past number is a rumour.
2. **Report the pool, not the arm.** A reranker can only be as good as the candidates it sees. Say
   7.96/8 unique, mean pairwise chrF 78.1, oracle 26.23 BLEU — then every "selector failed" claim has
   a ceiling attached to it.
3. **Correlation and usability are different objects.** +0.083 Spearman looked like a win and bought
   +0.02 BLEU. Report top-1 accuracy and headroom-captured next to ρ, always; the project's whole
   methodological thesis is that the first does not imply the second.
4. **Test the metric agreement, not just the metric.** MBR wins BLEU/chrF and loses COMET; the latent
   arm wins COMET and loses BLEU. Two honest metrics, two different "best systems". Any claim about
   which signal is good must name the yardstick.
5. **Free wins are hiding in defaults.** +1 BLEU sat in `num_beams=1` for every experiment this
   project ever ran. Before inventing an arm, read the decode call.
6. **Process hygiene is a research variable.** Three times tonight a background chain survived its
   child and competed for the GPU; the symptom is *halved throughput, not an error*. Timestamped logs
   caught it; "it feels slower" would not have.
7. **A tiny supervised ranker can be worse than no ranker** (ρ 0.557, −0.70 BLEU). Predicting a
   metric well is not the same problem as choosing an output; regression compresses toward the mean
   and the decision boundary is coarse. Anyone proposing "predict COMET, then pick argmax" must
   report the selection outcome, not the correlation.
8. **Check the cheap weight lever before building a selection framework.** 1.2 M LoRA parameters
   (1.5 % of the model) and 7.5 minutes bought +3.65 BLEU — more than the whole reference-based
   oracle headroom (+3.38) and 19× the best reference-free rule (+0.19). If a paper's contribution is
   a *selector*, it has to be compared against "just adapt the model", or a reviewer will do it for
   you. This is the most uncomfortable and most useful number of the night.

### What is genuinely new here (state it narrowly)

Not "latent agreement beats string agreement" (it does, as a scorer) and not "reranking helps" (it
did not, except MBR). The defensible sentence is:

> For two frozen 77 M models on a 10 k vi→en test set, representation-space round-trip agreement is a
> significantly better *predictor* of COMET than string-space round-trip agreement (Δρ +0.083,
> CI [+0.075, +0.092]) while being an equally poor *selector* (BLEU 22.31 vs 22.29, both below plain
> beam search); the frozen model's own 8-best pool contains 3.38 BLEU more than any reference-free
> rule can extract.

That sentence is checkable from committed files, does not overclaim, and is the kind of thing a
reviewer can reject on evidence rather than on taste — which is the best a night's work can offer.
