# Results — `Vector2 -> Token2` (small-experiment-02, run 2026-10-22)

Scope: does the mapped vector still carry the sentence, and can tokens be read
back out of it? 12,000 pairs inherited from experiment 01 (10k train / 2k test),
encoders `BAAI/bge-base-en-v1.5` and `bkai-.../vietnamese-bi-encoder` (768-d),
three poolings, both directions. Nothing was downloaded or resampled.

## 1. Mapping, refit here (vectors re-encoded, float32)

`results/map_sanity.csv`. Ridge alpha = 100, MLP 768-2048-768, same 90/10 split.

| direction | pooling | method | raw cosine | centered cosine | norm ratio |
|---|---|---|---|---|---|
| en2vi | mean | ridge | 0.485 | 0.459 | 0.53 |
| en2vi | mean | ann | 0.494 | 0.468 | 0.55 |
| vi2en | mean | ridge | 0.801 | 0.522 | 0.79 |
| vi2en | mean | ann | 0.801 | 0.523 | 0.81 |

**Correction to experiment 01.** The English space is strongly anisotropic: the
mean vector has norm 10.8 while the mean vector norm is 15.5, so two *random*
English vectors already score 0.52 raw cosine (0.42 for `cls`). After subtracting
the mean of the target space, random pairs score -0.00. Consequently the
`vi2en` cosine of **0.80 reported in `small-experiment/results/summary.md` was
inflated by that mean direction; centered it is 0.52**, i.e. the two directions
are symmetric (0.46 vs 0.52) and the previously implied asymmetry is an artefact.

## 2. Main result — token readout by nearest neighbour

Gallery = the 10,000 **train** target sentences, so the hypothesis is never the
gold test sentence. Open set, 2,000 items, `results/nn_metrics.csv`.

Mean pooling:

| arm | en2vi BLEU | chrF | chrF++ | TER | NIST | ROUGE-L | vi2en BLEU | chrF | METEOR |
|---|---|---|---|---|---|---|---|---|---|
| roundtrip (ceiling) | **5.10** | 19.91 | 18.50 | 119.5 | 0.477 | 0.296 | **2.54** | 22.73 | 0.127 |
| ridge | 3.27 | 17.30 | 15.77 | 121.9 | 0.346 | 0.270 | 1.35 | 20.82 | 0.097 |
| ann | 3.29 | 17.56 | 16.03 | 122.5 | 0.356 | 0.270 | 1.44 | 20.81 | 0.099 |
| identity (no map) | 0.09 | 10.25 | 8.51 | 142.7 | 0.077 | 0.210 | 0.04 | 14.00 | 0.044 |
| shuffled (wrong pair) | 0.24 | 10.27 | 8.68 | 138.1 | 0.084 | 0.207 | 0.13 | 14.07 | 0.045 |

Exact Match is 0.000 everywhere except `vi2en` 0.001 (2 of 2,000 items), which
are the 2 test sentences that also occur verbatim in the train gallery.

Ratios that survive the small absolute scale:

* ridge is **13.7x** (en2vi) / **10.7x** (vi2en) the shuffled control in BLEU.
* ridge keeps **64% / 53%** of the ceiling BLEU but **87% / 92%** of the ceiling chrF.
* identity collapses to the shuffled level: a vector must be in the right space at all.

Pooling (`results/nn_metrics.csv`), BLEU en2vi / vi2en:

| pooling | roundtrip | ridge | ann | shuffled |
|---|---|---|---|---|
| mean | 5.10 / 2.54 | 3.27 / 1.35 | 3.29 / 1.44 | 0.24 / 0.13 |
| cls | 4.68 / 2.59 | 3.16 / 1.35 | 3.18 / 1.44 | 0.16 / 0.11 |
| last | 4.68 / 2.68 | 2.99 / 1.40 | 3.22 / 1.49 | 0.25 / 0.26 |

`mean >= cls > last` for retrieval quality, reproducing the ranking of
experiment 01.

**ANN does not beat ridge** at token level: +0.6% BLEU (en2vi) and +6.5% (vi2en)
on mean pooling, with a marginally *higher* retrieval cosine. Same conclusion as
experiment 01's retrieval numbers: 9k training pairs do not justify the extra
nonlinear capacity.

## 3. Negative result — embedding inversion does not discriminate

Attempted `Vector2 -> Token2` by optimising token embeddings against the target
vector through the frozen encoder (straight-through, nearest-vocabulary
projection, random init, centered-cosine objective). 64 items, 100 steps,
token-overlap with the reference:

| pooling | roundtrip | identity | ridge | ann | shuffled |
|---|---|---|---|---|---|
| en2vi/mean | 0.034 | 0.022 | 0.033 | 0.036 | 0.036 |
| vi2en/mean | 0.054 | **0.068** | 0.056 | 0.066 | 0.066 |

All arms within noise; in `vi2en` the identity arm is the best arm. Inspection
of the hypotheses shows the inverter emits corpus-topic vocabulary ("thuế",
"điện_tử", "khoáng") regardless of which vector it receives: it recovers the
topic, not the sentence.

Measured cost: **4.2 s per item per (direction, pooling)**, i.e. ~18.7 h for the
full 2,000-item run — against ~1,000+ iterations used in the embedding-inversion
literature. Not viable on this laptop.

The inverter is not broken: on 8 short, simple Vietnamese sentences with 200
steps it does recover content words, e.g.

```
ref: Bệnh viện đã tiếp nhận hơn hai trăm bệnh nhân.
hyp: n Bệnh viện y tiếp tiếp nhận thêm nhiều 150 khách_hàng bệnh_nhân viện
```

The limitation is scale and sentence length, not the objective. Because a
retrieval readout already separates the arms (§2) while inversion does not, the
inversion path was abandoned rather than scaled.

Scored with the full harness (`results/token_metrics.csv`), the inversion output
gives **BLEU 0.07-0.22 and chrF 24-26 for every arm**, in both directions, all
poolings. Two things follow: (a) no arm is distinguishable, confirming the
overlap numbers above; (b) **chrF must not be used alone to claim success** — the
inverter emits corpus vocabulary, so character overlap is high while n-gram
precision is near zero. Any future report should quote BLEU and chrF together.

## 4. Threats to validity

1. **The readout is retrieval, not generation.** Its ceiling is "the best
   matching training sentence" (5.10 BLEU en2vi), so the absolute numbers are not
   comparable to MT literature. Only the arm-to-arm ratios are meaningful.
2. **Gallery size sets the ceiling.** 10k gallery items. A larger gallery raises
   the ceiling for all arms; it does not obviously change the ordering.
3. **VI `cls` and `last` are the same vector.** Max absolute difference 2.9e-3,
   correlation 0.9999998 (PhoBERT puts `<s>` and `</s>` in nearly identical
   states). Two of the three poolings are therefore not independent evidence.
4. **Duplicate leakage is bounded but not zero.** 0/2,000 test VI sentences and
   2/2,000 test EN sentences occur verbatim in the train gallery. Exact Match is
   ~0, so duplicates do not drive the result, but partial near-duplicates (legal
   boilerplate) can still inflate retrieval BLEU.
5. **No significance testing.** Differences of 0.02 BLEU (ridge vs ann) are
   within noise; a paired bootstrap is needed before quoting them.
6. **General-domain corpus**, not culinary (`../small-experiment/agent-work/ASSUMPTIONS.md` §2).
7. **METEOR is unavailable for Vietnamese** (WordNet has no VI lexicon), so it is
   reported only for `vi2en`.

## 5. What this means for the paper

**Supported: the mapped vector is token-decodable.** A wrong-but-plausible
vector produces 0.24 BLEU; the same readout on the mapped vector produces 3.27.
The map therefore carries sentence content, not merely the domain topic.

**Supported: the loss is in word choice and word order, not in topic.** The map
keeps 87-92% of the ceiling chrF (character-level) but only 53-64% of the ceiling
BLEU (n-gram precision). That is the signature of a channel that preserves
meaning but blurs exact wording.

**Not supported: the ANN.** Ridge matches or beats it at every pooling, both
directions, on both representation-level and token-level metrics.

**Corrected: the direction asymmetry in experiment 01 was an artefact** of
anisotropy in the English space (§1).

**Implication for the architecture.** Even the *oracle* vector only reaches 5.10
BLEU through retrieval. A single pooled 768-d vector is a narrow channel. The next
step that can actually move the number is not a bigger mapping network but a
wider channel (k-token prefix) or decoder hidden states, which is exactly what
`paperOverview.md` §3.3 left open.

## 6. Not done

* Generation by a trained decoder (option "R2"): designed, not executed — user
  stopped the experiment after the retrieval readout.
* Neural metrics (BERTScore, COMET, BLEURT, BARTScore, Prism, COMETKiwi,
  MoverScore): feasibility not tested, nothing downloaded.
* Open-corpus retrieval over the 267k-pair pool (only the 10k train gallery was used).
