# Pre-registration — written before any test-set number existed

**Question (the one the user ranked first): is *latent* agreement a usable reference-free quality
signal for machine translation — and does it beat the *string* version of the same idea?**

Everything below was fixed at ~22:55, before the 10 k test pool finished generating. Deviations are
logged at the bottom, not silently fixed.

## Design in one paragraph

One frozen translator `A = opus-mt-vi-en` (77 M) produces an 8-best pool per test sentence. One frozen
back-translator `B = opus-mt-en-vi` (77 M) turns every candidate back into Vietnamese. Two utilities
are then computed on exactly the same candidates:

- **string round-trip**: chrF and char-3-gram Jaccard between the back-translation and the source;
- **latent round-trip**: both Vietnamese strings are embedded with **A's encoder** (same model, same
  Vietnamese tokenizer → *one shared space*, unlike the cross-model spaces the project measured
  before), and agreement is measured as pooled cosine plus token-level nearest-neighbour
  precision / recall / F1.

Reranking = pick the candidate with the highest utility. No reference is used by any arm.

## Hypotheses (decided now, scored later)

| id | hypothesis | operational test | what would make it false |
|---|---|---|---|
| **H1** *(primary)* | latent round-trip ranks candidates better than string round-trip | Spearman(utility, COMET) and top-1 accuracy on the 10 k test pool; paired bootstrap over sentences for BLEU/chrF/COMET of the *selections* | ΔSpearman ≤ 0 **and** ΔBLEU CI straddles 0 |
| **H2** | reference-free **consensus** (chrF-MBR) is a stronger utility than either round-trip signal | same three diagnostics | chrF-MBR loses on ≥ 2 of the 3 |
| **H3** | adding latent features to a tiny learned ranker helps beyond seed noise | ΔSpearman(string+latent vs string) > sd over 3 seeds | Δ ≤ sd |
| **H4** *(control)* | 4 M LoRA parameters buy more than any training-free selection | BLEU of `A5_lora` vs best training-free arm | any training-free arm ≥ LoRA |

## Metrics and inference

- BLEU (`sacrebleu`, `13a`), chrF, COMET (`Unbabel/wmt22-comet-da`), all corpus-level on the full
  10 k test; **paired** bootstrap (1 000 resamples, same sentence indices for every arm).
- 2 k `core` subset reported separately so the historical `22.27 BLEU` for `opus-mt-vi-en` stays
  comparable.
- "Headroom captured" = (BLEU(arm) − BLEU(beam top-1)) / (BLEU(oracle) − BLEU(beam top-1)), where the
  oracle picks the best candidate by true COMET. This is the number that says whether a utility is
  *usable*, not merely correlated.

## Known threats to validity (stated before looking)

1. **Correlated candidates.** Beam-8 hypotheses share prefixes; a reranker's ceiling is small if the
   pool is redundant. Report the unique-string rate per row so the reader can judge.
2. **COMET as teacher and as judge.** The oracle, the learned ranker's labels and the final metric are
   the same model. Any "latent ≈ COMET" claim is therefore an upper bound on usefulness, not proof of
   human agreement. (Cheap partial check: report chrF too; it is a different instrument.)
3. **Domain.** MTet is law/news/medical, and earlier phases found ~1 in 12 misaligned pairs, so every
   absolute number is depressed; only *differences between arms* are interpretable.
4. **Sentence-level only.** Span-level food/culture phenomena are out of scope tonight.
5. **The back-translation is itself a model output.** A good hypothesis can back-translate badly
   (domain shift vi→en→vi). This is a property of the signal, not a bug in the code — but it caps
   H1's ceiling and is worth measuring: report the correlation of each signal with hypothesis length.

## Deviation log

(filled in after the runs — any change to the plan above must appear here with a timestamp and reason)
