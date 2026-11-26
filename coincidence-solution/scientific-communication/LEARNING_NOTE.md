# Learning note — how to argue that an objective is wrong

This is the teaching document. `paper.md` is the product; this file is the reasoning behind it, the
adversarial questions I asked myself, and the traps that a reader of the result could fall into. Read
it if you want to defend the paper in a room, not only reproduce it.

---

## 1. The one-sentence idea, and why it is easy to state and hard to believe

> A bridge between two frozen models should be fitted on what the receiving model *does*, not on how
> close its input is to a target vector.

Everyone nods at this. Almost nobody acts on it, because the substitute is convenient: a linear map
fitted on squared error has a closed form, so it is fast, deterministic, and does not need a decoder
in the loop. The argument of the paper is that this convenience stopped being a necessity the moment
the decoder was frozen — and it stayed in the literature as a habit.

**The falsifiable prediction.** If the objective did not matter, two maps with the same function
class, the same parameter count and the same data would be interchangeable. They are not: 2.47 vs
7.90 BLEU. That is the entire empirical weight of the paper, and it is one table.

## 2. Why the "Lipschitz bound" defence is vacuous, in three lines

The standard justification for MSE is: a perturbation $\varepsilon$ changes the logit by at most
$\|W_{\text{lm}}\|\,\|\varepsilon\|$, so a small error is safe. Two things are wrong with using it as
a defence.

1. The bound is **over all directions**. It controls the worst case, not your case. If your error is
   concentrated in the one direction the decoder weights 100× more, the bound is loose by that factor
   and tells you nothing.
2. It does not run **backwards**. Small behaviour change does not imply small vector error, and that
   is the direction you need to claim "fitting the vector is enough".

This is why the paper spends a section measuring *where* the error sits rather than arguing about its
size. A bound is not evidence.

## 3. The experiment that decides it: `ce_scratch`

If MSE is simply "the wrong objective", then optimising the right objective from scratch should work.
It does not: `ce_scratch` (cross-entropy from the identity map) is *worse* than ridge on BLEU, worse
on validation CE, and overshoots the memory norm.

The lesson is not "CE is bad". It is that **two different jobs are being done**:

- least squares finds the right *basin* (where the decoder's tolerance region lives);
- the decoder's likelihood finds the right *point inside the basin*.

This is the most transferable insight in the package. When someone says "representation fitting is
useless", the correct reply is "it is a good initialiser". When someone says "just train on the task",
the reply is "from where?". The composition is what works, and it costs one closed-form solve plus two
epochs.

**How I would test the claim further:** start CE from a *random* map with the right norm (not
identity). If it also fails, the basin claim is about direction, not scale; if it succeeds, ridge's
value is only its norm, and §5.5's scale control already says that is 14% of the story.

## 4. The null-space story, and why killing it was worth a section

The intuitive rescue for MSE is: "the error lives in directions the decoder ignores, so it is
harmless." It is a good hypothesis — it explains why a 37% relative vector error can coexist with
readable text. It is also false here, and the measurement is cheap if you already have gradients:

- take the per-sentence gradient $\partial\mathrm{CE}/\partial H$ at a good memory,
- take its top-$k$ singular directions,
- project the least-squares residual onto them,
- compare with chance ($k/32768$).

Chance-normalised: 0.8×, 1.2×, 1.4×, 1.3× for $k=1,8,64,256$. There is no concentration. The residual
is spread like an isotropic error would be, which is exactly what minimising an unweighted norm
produces when the target is anisotropic.

**Why this matters rhetorically:** it converts "our method is better" into "the mechanism everyone
assumed cannot be the mechanism". A paper with a falsified mechanism is harder to dismiss than a paper
with a better number.

**The honest caveat** (in §5.4): weighting sentences by their error magnitude instead of equally
raises the concentration to 3–5× at small $k$. That is a real effect and I report it. It still leaves
98% of the error outside the top-256 directions, so the conclusion survives — but a reviewer who
recomputes will find the 3–5× number, and it is better that it is already in the text.

## 5. Reading a perturbation curve correctly

Three rules, each learned the hard way in this package.

1. **Displacement must be relative.** $\varepsilon$ is a fraction of the memory's own Frobenius norm.
   Absolute norms are not comparable across models, sentences or tokenisations.
2. **Direction matters more than size.** At $\varepsilon=0.05$: random → CE 3.402 (+0.006);
   steepest → CE 7.164 (×2.1). Same ball. If you report one curve, you will conclude the wrong thing
   about tolerance.
3. **You need two oracles, and they answer different questions.** The reference memory
   ($H_2$ on the English reference) is a *behavioural* ceiling; the native memory ($H_2$ on the
   Vietnamese source) is *position-aligned* with the map. Perturbation curves need no correspondence,
   so they run on the first; any per-position projection needs correspondence, so it runs on the
   second. Mixing them produces a meaningless "distance to the oracle" — a reviewer will catch it.

## 6. What "the decoder's tolerance" really is

A decoder does not induce a metric. It induces a **partition** of memory space: memories that decode
to the same string. Around a good memory, the cell of that partition is a thin, curved slab. Two
consequences follow, and both are in the paper:

- **Thin**: a 5% random displacement already changes 22% of outputs. There is no wide basin to hide in.
- **Tilted**: the slab is much thinner along the directions the decoder reads. Error of a fixed
  Euclidean size is invisible along the slab and fatal across it.

The linguistic reading: the decoder is a *threshold device* with a preferred coordinate system. A
distance in the ambient space is a poor proxy for "how far from the answer", because the space in which
the answer is "close" is not the space in which the error is measured.

**Analogy that worked for me.** Imagine tuning a radio: the frequency knob has a narrow band where the
station is clear, and the volume knob does not. A loss that treats both knobs equally will spend half
its budget on volume. A decoder is the same: token identity is frequency, magnitude is volume.

## 7. The controls, and the order of disbelief

When a number improves by 3×, the prior should be "something trivial happened". I ordered the controls
by how plausible the trivial explanation is:

| trivial explanation | test | result |
|---|---|---|
| "you used more data" | same 20k pairs | −0.03 BLEU |
| "you fixed the shrinkage" | multiply by 1.22 | +0.39 |
| "you matched the covariance" | recolour 10% | +0.59 |
| "you just found a better starting point" | CE from identity | worse |
| "it is BLEU noise" | paired bootstrap | +5.40 [4.50, 6.27] |
| "it is a split artefact" | disjoint 1,000 | same gap |
| "it is a BLEU artefact" | COMET | same ranking |

The order matters: if you test the cheapest explanation last, you will have already written the paper
in your head.

## 8. Figures: what makes a conceptual figure honest

You asked for figures that look professional and communicate intuition. The rule I used:

> A conceptual figure may invent **geometry** but never **numbers**.

Figure 4 invents a 2-D band; every number printed on it (0.2 width, 2.5 BLEU, 74 BLEU) is measured.
Figure 3 invents the positions of papers on two axes, and says so in the caption ("positions are the
authors' reading, not a measurement"). Figure 1 invents the concept graph entirely, and is labelled
"conceptual model".

Two practical rules for making them look like a paper rather than a slide deck:

- **Claims as panel titles.** "(a) the effect is 5× outside the ambiguity" beats "(a) BLEU
  differences". The title should be the sentence you want the reader to remember.
- **One accent colour per meaning, used consistently** (slate = representation-fitted, blue =
  behaviour-fitted, amber = cheap repairs, red dashed = ceiling). The reader learns the legend once
  and then reads seven figures without effort.

## 9. Numbers a reader will (rightly) attack

- **"+3.2×"** is 2.47 → 7.90. If someone prefers to compare with the published map (2.50 → 7.90) it
  is 3.16×. Either is fine; say which.
- **"11% of the ceiling"** is 7.90/74.22. It is a fraction of *BLEU*, not of translation quality.
  BLEU is not linear in quality, so do not say "we solved 11% of translation".
- **"5× outside the tolerance"** is $\varepsilon=1$ vs the $\varepsilon\approx0.2$ radius where BLEU
  is still within 3% of the oracle. It is a statement about a random-direction radius, not a
  Euclidean distance to the truth.
- **The 74.2 oracle is not a bound on the method.** It is the best memory we can construct. A
  different memory could in principle decode better; nothing in the measurement excludes it.

## 10. What I would do next, and why in this order

1. **Train to convergence with a cosine schedule.** The cheapest open number; CE is flat but BLEU was
   still rising at 16 epochs, so the method's ceiling is unmeasured.
2. **Two adapters instead of one mixed adapter.** The CE objective is fitted *against* the adapter, so
   a mixed adapter confounds channel differences. This is no longer cosmetic.
3. **en→vi.** Same recipe, adapter exists. One afternoon.
4. **A bridge that can reorder.** A position-wise linear map cannot move content across positions;
   that is a structural bound on the whole approach, and it is the most interesting theoretical
   question the data raises.

If you only have one day: run (1). If you have a week: (3) then (2). If you want a paper rather than a
result: (4).
