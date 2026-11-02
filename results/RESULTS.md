# Results — Phase 0

**What was run.** One falsification matrix over a frozen `Qwen/Qwen2.5-0.5B-Instruct`, 200 synthetic
three-agent Theory-of-Mind items, six layers, six mechanisms, five values of α.

**Hardware and software.** 2 vCPU / 3 GB RAM, CPU only, no CUDA, no MPS. torch 2.15.0+cpu,
transformers 5.22.1, numpy, scipy not installed (see §9). Seed 0 throughout. Wall clock for the
whole measurement: **≈ 2 h 10 min**, of which the base pass was 19 min and each activation-steering
arm was 9.5 min.

**The command.**

```bash
python src/run_phase0.py --n-items 200 --layers 4,8,12,16,20,24 \
    --act-alphas 1.0 --score-alphas 0.25,0.5,1.0,2.0,4.0 --out-dir results
python src/run_controls.py --layer 4 --alpha 1.0 --n-items 200 --out-dir results/controls
python tools/analyze_heldout.py
```

Every number below comes from the JSON in `results/`. Nothing is estimated, rounded up, or carried
over from a different run.

---

## The headline

> **The partner belief is strongly linearly decodable from the model's own activations, and no
> intervention we tested turned that into a usable change in the model's answers.**
>
> - The belief is readable: out-of-fold 3-way ranking accuracy **0.720** at layer 16
>   (chance 0.333, base model 0.370; p = 2.5 × 10⁻¹¹ vs base, p = 8.5 × 10⁻¹¹ vs a shuffled-label
>   control).
> - The belief is **not** usable: the only mechanism that changes the forward pass — activation
>   steering — helps at exactly one layer, and at that layer a **matched random direction helps
>   about as much** (0.480 vs 0.430, paired p = 0.189, not significant).
> - The mechanism that *does* beat the oracle (0.910 vs the oracle's 0.510) does so by
>   **bypassing the model**, not by changing it: its accuracy converges to the probe's own
>   read-out accuracy. Its headline number in the matrix summary (0.910) is an **in-sample**
>   artefact; out-of-fold it is 0.695.
> - The base model is **at chance**: 0.370 against a 0.333 chance level, p = 0.29.

The repository's question was whether a training-free decoder-level intervention can recover what
merely *telling* the model the belief recovers. **On this model and this task, it cannot — and the
matrix shows exactly where the attempt fails.** The failing step is not the probe (it works) and
not the control (it is clean). It is that the signal is present in the activations at a layer where
the model does not act on it, and pushing the model at the layer where it *does* act is not
belief-specific.

---

## 1. The precondition: does the base model have any partner-belief signal at all?

Before asking whether steering helps, the matrix checks whether there is anything to steer. With
three options and a uniform random guess at 1/3:

| arm | correct | accuracy | 95 % CI (bootstrap) | binomial vs chance |
|---|---|---|---|---|
| base (`none`) | 74 / 200 | **0.370** | [0.305, 0.440] | p = 0.294 |
| **`prompt_oracle` (ORACLE)** | 102 / 200 | **0.510** | [0.445, 0.585] | p = 2.6 × 10⁻⁷ |

The base model is **not significantly above chance**. This is the single most important fact in the
document, and it was not known in advance. It bounds everything that follows: an intervention
cannot be shown to *help the model use a belief the model does not have*, and the 0.510 oracle is
the realistic ceiling for this model on this task, not 1.0.

The error-mode breakdown shows the same story from the other side. The base model answers with the
**true** location (a perspective-blind answer) on 28 % of items and with the **third agent's**
belief (a wrong-attribution answer) on 35 %. It is not making one systematic mistake; it is
guessing, and guessing slightly more often in the direction of the truth.

The ORACLE arm states the protagonist's belief in the prompt. It reaches 0.510 — so the model *can*
use a stated belief, but only to the tune of 0.510. Every decoder-level arm is measured against
that bound, not against 1.0.

---

## 2. The read curve — is the belief linearly encoded?

The probe is a closed-form shrinkage Fisher-LDA direction fitted on the hidden state of each
candidate's last token, with the label *"is this the location the protagonist believes?"*.

**Pair-wise held-out accuracy is the wrong statistic for this question.** Two of an item's three
candidates are negatives, so a probe that always answered "not the belief-consistent candidate"
would score 2/3. The quantity a belief-margin intervention actually relies on is the **3-way
ranking** accuracy: given the three candidates, does the margin rank the belief-consistent one
first? Chance is 1/3.

All numbers below are **out-of-fold**: 5-fold cross-validation over *items* (not over pairs), so
every item is scored by a probe that never saw it, and the bootstrap resamples over all 200 items.

| layer | 3-way ranking acc (out-of-fold) | 95 % CI | shuffled-label control | base model | p vs base | p vs shuffled |
|---|---|---|---|---|---|---|
| L4 | 0.420 | [0.355, 0.490] | 0.430 | 0.370 | 0.331 | 0.835 |
| L8 | 0.545 | [0.475, 0.615] | 0.345 | 0.370 | 0.0011 | 8.8 × 10⁻⁵ |
| L12 | 0.530 | [0.465, 0.600] | 0.385 | 0.370 | 0.0021 | 0.0059 |
| **L16** | **0.720** | **[0.655, 0.780]** | 0.375 | 0.370 | **2.49 × 10⁻¹¹** | 8.53 × 10⁻¹¹ |
| L20 | 0.695 | [0.630, 0.755] | 0.360 | 0.370 | 6.85 × 10⁻¹⁰ | 1.12 × 10⁻⁹ |
| L24 | 0.595 | [0.530, 0.660] | 0.355 | 0.370 | 7.55 × 10⁻⁶ | 3.13 × 10⁻⁶ |

**The partner belief is linearly encoded, and it becomes readable at layer 8 and peaks at layer 16.**
At layer 16 the probe ranks the correct candidate first on 72 % of items it has never seen — nearly
twice the chance level and three times the base model, and far above the shuffled-label control.

The layer-4 row is the one to remember: **the probe has no signal there at all** (0.420, p = 0.33 vs
base, 0.430 for the shuffled control). And layer 4 is exactly where the one forward-pass effect
appears (§4). The two curves are anti-correlated at the point where it matters.

A single 80/20 split, kept as an independent second estimate, agrees: 3-way held-out accuracy
0.800 at L16, 0.700 at L20, and 0.575/0.550/0.575 at L4/L8/L12. The pair-wise held-out accuracies
in that split (0.575–0.708) sit at or below their trivial baseline of 0.667, which is the
signature of a probe that *ranks* well without separating the classes at a threshold.

---

## 3. The score-level arms — and why their headline numbers are not evidence

The `margin` mechanism adds α × (standardised probe margin) to each candidate's log-probability
and takes the argmax. The `margin_shuffled` control does the same with a probe fitted on
permuted labels. The `logit` mechanism adds α × the candidate's logits projected onto the
LM-head image of the direction.

**Out-of-fold, the `margin` arm beats its shuffled control everywhere from layer 8 upward:**

| layer | α | margin (out-of-fold) | margin_shuffled | difference | p (paired) |
|---|---|---|---|---|---|
| L8 | 4 | 0.545 | 0.355 | +0.190 | 4.2 × 10⁻⁵ |
| L12 | 4 | 0.590 | 0.410 | +0.180 | 1.8 × 10⁻⁴ |
| L16 | 4 | **0.720** | 0.330 | +0.390 | 2.0 × 10⁻¹⁴ |
| L20 | 4 | 0.695 | 0.370 | +0.325 | 1.6 × 10⁻¹¹ |
| L24 | 4 | 0.625 | 0.405 | +0.220 | 1.2 × 10⁻⁶ |
| L4 | any | 0.390–0.425 | 0.375–0.400 | ≤ +0.040 | ≥ 0.23 |

**But look at where the numbers land.** At α = 4 the `margin` arm's out-of-fold accuracy is
**0.720 at L16, 0.695 at L20, 0.625 at L24** — and the probe's own out-of-fold 3-way accuracy is
**0.720, 0.695, 0.595** at the same layers. The arm converges onto the probe.

That is not a coincidence, and it is the central methodological finding of this phase:

> **A score-level belief-margin intervention does not make the model use the belief. It makes the
> probe pick the answer, and then reports the probe's accuracy as if it were the model's.**

As α grows, the α × margin term dominates the candidate log-probabilities and the model's own
likelihood is discarded. The arm asymptotes to "choose the candidate the probe likes", which is the
read-out curve of §2 — not an intervention curve.

**The in-sample numbers are larger and they are an artefact.** The matrix summary reports
`margin-L20-a4 = 0.910` and `margin-L24-a4 = 0.870`, because the probe is fitted on 80 % of the
pairs and then scores *every* pair, including the ones it was trained on. Those numbers are
leakage, and they are the reason this document does not headline them. The out-of-fold numbers are
0.695 and 0.625 — a difference of 0.215 and 0.245 respectively.

Note also that `margin-L20-a4` (0.910) **beats the prompt oracle** (0.510). That is not a
superior method; it is the same bypass seen from the other side. The oracle is an upper bound on
*prompt-level* injection, and the margin arm does not inject anything.

**The `logit` arm** is flat in α (0.490 at L24 for every α from 0.25 to 4). The bias vector it
adds is large enough to saturate the argmax at the smallest α in the grid, so the α sweep carries
no information for that mechanism. This is a defect in the α grid, not a finding, and it is
recorded as such.

---

## 4. The forward-pass arms — the one real effect, and the control that deflates it

Activation steering adds α · scale · û to the residual stream at layer *L*, where û is the fitted
belief direction and `scale` is the mean hidden-state norm at that layer, so α means the same thing
at every depth.

| layer | accuracy | 95 % CI | world rate | other rate | p vs base |
|---|---|---|---|---|---|
| **L4** | **0.480** | [0.415, 0.550] | 0.205 | 0.315 | **0.00387** |
| L8 | 0.345 | [0.275, 0.415] | 0.315 | 0.340 | 0.446 |
| L12 | 0.405 | [0.340, 0.475] | 0.220 | 0.375 | 0.297 |
| L16 | 0.400 | [0.335, 0.470] | 0.265 | 0.335 | 0.330 |
| L20 | 0.370 | [0.305, 0.440] | 0.240 | 0.390 | 1.000 |
| L24 | — | — | — | — | *no act arm: see below* |

Layer 4 is the only layer with a significant effect: **+11 points over the base model**, and it
shifts the error modes in the intended direction (world errors 0.280 → 0.205).

**There is no activation arm at layer 24.** The steering hook adds to the *input* of a decoder
layer, so a model with 24 decoder layers can be steered at layers 0–23; layer 24 is
`hidden_states[-1]`, the final output, and there is no decoder layer after it whose input could be
perturbed. The first version of the runner built an arm for every requested layer and died with
`IndexError: layer 24 out of range for 24 layers` **after the whole run had been paid for**. Layer
24 is still probed and still carries the margin and logit arms. The summary records
`act_layers_skipped: [24]`.

### The matched controls

The activation arm is the only mechanism that changes the forward pass, and an activation arm that
"helps" is indistinguishable from generic confidence steering unless something of the same
magnitude, in the same layer, is shown *not* to help. Three arms, matched in layer, direction norm
and α, all produced by the same code path on the same day:

| arm | direction | accuracy | 95 % CI | p vs base |
|---|---|---|---|---|
| `act_real` | the fitted belief direction | **0.480** | [0.415, 0.550] | 0.00387 |
| `act_random` | a uniformly random unit direction | **0.430** | [0.365, 0.500] | 0.0396 |
| `act_reversed` | the negation of the belief direction | 0.380 | [0.315, 0.450] | 0.732 |
| base | — | 0.370 | [0.305, 0.440] | — |

Paired by item:

| comparison | difference | 95 % CI (bootstrap of the paired difference) | Wilcoxon p |
|---|---|---|---|
| `act_real` vs `act_random` | +0.050 | [−0.025, +0.120] | **0.189** |
| `act_real` vs `act_reversed` | +0.100 | [+0.030, +0.170] | **0.0065** |
| `act_random` vs `act_reversed` | +0.050 | [+0.000, +0.105] | 0.059 |

### The random control, repeated until it is a distribution

**One random draw is not a distribution.** The comparison above is a single draw, and the Phase-0
text said so. The sweep re-runs the control with several independent random directions per layer,
each with a recorded seed and its cosine to the real direction.

| layer | belief direction | random draws | mean | sd | min | max | draws beating the belief direction | sign-test p |
|---|---|---|---|---|---|---|---|---|
| **L4** | **0.480** | 6 | 0.400 | 0.042 | 0.340 | 0.450 | **0 / 6** | **0.031** |
| L8 | 0.345 | 2 | 0.328 | 0.018 | 0.315 | 0.340 | 0 / 2 | 0.50 |
| L12 | 0.405 | 2 | 0.355 | 0.042 | 0.325 | 0.385 | 0 / 2 | 0.50 |
| **L16** | 0.400 | 4 | **0.411** | 0.042 | 0.355 | 0.455 | **3 / 4** | 0.63 |
| **L20** | 0.370 | 2 | **0.397** | 0.039 | 0.370 | 0.425 | **1 / 2** | 0.50 |

The sign test is the aggregate statistic: under the null that the belief direction is exchangeable
with the random draws, each paired sign is a fair coin, so the exact two-sided p-value for *n*
positive differences out of *n* is 2^(1−n). It uses every draw and cherry-picks nothing — the
individual paired tests share the belief arm and the base predictions and are therefore correlated,
so quoting the smallest of them would be a multiple-comparison selection. The per-draw tests are
reported alongside it:

| layer | random draw | accuracy | belief − random | 95 % CI | Wilcoxon p |
|---|---|---|---|---|---|
| L4 | seed 977 | 0.430 | +0.050 | [−0.025, +0.120] | 0.189 |
| L4 | seed 978 | 0.450 | +0.030 | [−0.045, +0.105] | 0.439 |
| L4 | seed 979 | 0.420 | +0.060 | [−0.010, +0.130] | 0.103 |
| L4 | seed 980 | 0.400 | +0.080 | [+0.010, +0.155] | 0.033 |
| L4 | seed 981 | 0.340 | +0.140 | [+0.070, +0.210] | 0.00018 |
| L4 | seed 982 | 0.360 | +0.120 | [+0.050, +0.190] | 0.0011 |
| L16 | seed 977 | 0.455 | −0.055 | [−0.115, +0.000] | 0.063 |
| L16 | seed 978 | 0.410 | −0.010 | [−0.075, +0.055] | 0.758 |
| L16 | seed 979 | 0.355 | +0.045 | [−0.025, +0.120] | 0.241 |
| L16 | seed 980 | 0.425 | −0.025 | [−0.110, +0.060] | 0.553 |
| L20 | seed 977 | 0.370 | +0.000 | [−0.060, +0.055] | 1.000 |
| L20 | seed 978 | 0.425 | −0.055 | [−0.135, +0.020] | 0.166 |

**This changes the Phase-0 conclusion, in the direction of a stronger effect.**

- **At layer 4 the belief direction beats 6 of 6 matched random directions** (sign-test p = 0.031),
  and no random draw reaches its accuracy. The Phase-0 statement "directional but not shown to be
  belief-specific" is upgraded to **"directional, and distinguishable from a matched random
  direction of the same size at the same layer"**.
- **The single draw that Phase 0 used was one of the more favourable random directions.** Its
  p-value of 0.189 was not an unlucky draw; it was a *generous* one, and one draw understated the
  effect. A different single draw would have given p = 0.00018.
- **At layers 16 and 20 the random directions do better than the belief direction** (mean 0.411 vs
  0.400 at L16; 0.397 vs 0.370 at L20), and 3 of 4 random draws beat the belief direction at L16.
  At the layer where the belief is most readable, pushing along the belief direction is *worse* than
  pushing along an arbitrary one.

**This is the result the phase exists to produce, and it is still a negative one.** The effect is
directional (p = 0.0065 against the sign flip) and, at layer 4, it is distinguishable from a
matched random direction. But layer 4 is the layer where the belief **cannot be read at all**
(probe accuracy 0.420, p = 0.33). A direction that does not carry a readable belief still steers
better than a random one — so what the belief direction is doing at layer 4 is *not* obviously
about the belief, and the one place the belief is demonstrably present (layer 16) is the place
where the belief direction steers no better than noise. See §5.

A **re-run of the real arm reproduced the matrix result exactly** (0.480, identical predictions to
the matrix's `act-L4-a1`), and the sweep's draw 0 at layer 4 reproduced the original single-draw
control exactly (0.430, identical predictions), so these are deterministic, not flukes of a run.

---

## 5. The dissociation

Putting §2 and §4 side by side is the point of the repository:

| layer | read (out-of-fold 3-way) | bite (activation steering) |
|---|---|---|
| L4 | 0.420 (no signal) | **0.480** (best effect) |
| L8 | 0.545 | 0.345 |
| L12 | 0.530 | 0.405 |
| L16 | **0.720** (best read) | 0.400 |
| L20 | 0.695 | 0.370 |
| L24 | 0.595 | — (not steerable) |

**The best-reading layer is not the best-biting layer, and the relationship is inverted.** The
belief is clearest at layer 16, where steering does nothing. The only layer where steering does
something is layer 4, where the belief cannot be read at all.

Two readings of that inversion are consistent with these data, and this phase cannot separate them:

1. **The signal is encoded but not used.** The model holds the partner's belief at layers 16–20
   and does not act on it; steering layer 4 changes something else that happens to help.
2. **The layer-4 effect is not about the belief at all.** The probe has no signal at layer 4, so
   the direction being pushed is not the belief direction in any meaningful sense.

### The inversion, as a statistic

"The best-reading layer is not the best-biting layer" was, until now, a claim about two tables.
`tools/read_bite_dissociation.py` turns it into a rank correlation over the five steerable layers,
with an **exact permutation** p-value over all 5! = 120 orderings of the bite vector:

| statistic | value | exact p |
|---|---|---|
| Spearman's ρ (read vs. bite) | **−0.600** | 0.350 |
| Kendall's τ-b (read vs. bite) | −0.400 | 0.483 |

**The correlation is negative and it is not significant**, and both halves of that sentence matter.
With five layers the asymptotic p-value is meaningless, so the exact permutation test is used; and
with five layers even a *perfect* negative correlation would reach only p = 2/120 = 0.017. The data
are therefore **suggestive, not established**: the layers where the belief reads best are the layers
where steering it does least, but five layers cannot demonstrate that.

### The sharper form: does the belief direction beat a random one *more* where the belief reads better?

A random direction is the baseline for "perturbing the residual stream here does something". The
**advantage** of the belief direction over the mean matched random direction isolates the part of
the steering effect that could be about the belief at all — and it is the better question.

| layer | read (out-of-fold 3-way) | belief direction | random mean | **advantage** | random draws beating the belief direction |
|---|---|---|---|---|---|
| **L4** | 0.420 | 0.480 | 0.400 (6 draws) | **+0.080** | **0 / 6** |
| L8 | 0.545 | 0.345 | 0.328 (2 draws) | +0.017 | 0 / 2 |
| L12 | 0.530 | 0.405 | 0.355 (2 draws) | +0.050 | 0 / 2 |
| **L16** | **0.720** | 0.400 | **0.411** (4 draws) | **−0.011** | **3 / 4** |
| **L20** | 0.695 | 0.370 | **0.397** (2 draws) | **−0.027** | **1 / 2** |

| statistic | value | exact p |
|---|---|---|
| Spearman's ρ (read accuracy vs. advantage) | **−0.900** | 0.083 |
| Kendall's τ-b (read accuracy vs. advantage) | −0.800 | 0.083 |

The correlation is **strongly negative**: the belief direction's steering advantage is *largest
where the belief is least readable*, and it is **negative** at the two layers where the belief
reads best. At layer 16 the belief direction is, on average, *worse* than a random direction of
the same size.

The exact p-value is 0.083, which with five layers is just short of conventional significance —
the same limit that applies to every five-layer statistic in this document. The pattern is
consistent across the three readable layers (L8, L12 positive; L16, L20 negative) and the
ordering is monotone in read accuracy at the top end, but the honest description is
**strongly suggestive, not established**.

What this does establish, without any statistical framing at all, is the *qualitative* pattern:
**the one layer where the belief direction demonstrably beats random steering (L4, 6/6) is the
layer where the belief cannot be read at all.** That is the read-vs-bite dissociation with the
controls attached.
## 6. The belief-accuracy frontier — what the intervention actually adds

Phase 0 reported, in prose, that the score-level margin arms "re-rank with the probe rather than
change the model". That claim can be measured exactly, and cheaply: the score-level mechanism acts
entirely on **cached** quantities (the base log-probabilities and the probe's margin), so the whole
frontier is computable from `results/base_cache.npz` with **no model loaded**.

`tools/belief_accuracy_frontier.py` simulates a probe of a given quality by corrupting its
**training labels** (each item's belief-consistent candidate is relabelled as an incorrect one with
probability *noise*), then runs the Phase-0 protocol unchanged: item-level 5-fold cross-validation,
out-of-fold quantities, 3-way ranking accuracy, and a matched shuffled-label control for every arm.
The x-axis is the **achieved** out-of-fold read accuracy of the degraded probe, not the nominal noise
rate — the achieved number is what a deployed estimator would actually deliver.

### The frontier at the best-reading layer

Layer 16, α = 4 (fixed before the sweep, not chosen after):

| label noise | achieved read acc | margin arm | shuffled control | arm − base | p vs control | p vs base |
|---|---|---|---|---|---|---|
| 0 | **0.720** | **0.720** | 0.330 | +0.350 | 2.03e-14 | 2.21e-14 |
| 0.15 | 0.655 | 0.635 | 0.320 | +0.265 | 1.59e-10 | 1.54e-09 |
| 0.30 | 0.535 | 0.575 | 0.400 | +0.205 | 4.97e-04 | 6.78e-06 |
| 0.45 | 0.490 | 0.515 | 0.340 | +0.145 | 1.75e-04 | 3.22e-04 |
| 0.60 | 0.350 | 0.405 | 0.365 | +0.035 | 0.365 | 0.392 |
| 0.75 | 0.280 | 0.290 | 0.310 | −0.080 | 0.655 | 0.0455 |

**The break-even belief accuracy is ≈ 0.34** — barely above the 0.333 chance level. Below that the
intervention is net-negative; above it, it is net-positive *relative to the base model*. The
least-squares crossing of the fitted arm-accuracy line with the base model is at a read accuracy of
**0.337**. The bracket is [0.280, 0.350]: the achieved read accuracy is not monotone in the noise
rate (one corruption draw per level), so the break-even is a bracket, not a measured threshold.

### The arm's accuracy is the probe's accuracy

| label noise | achieved read acc | margin arm | arm − probe | p (paired) | arm picks what the probe picks |
|---|---|---|---|---|---|
| 0 | 0.720 | 0.720 | +0.000 | 1 | **0.830** |
| 0.15 | 0.655 | 0.635 | −0.020 | 0.45 | 0.835 |
| 0.30 | 0.535 | 0.575 | +0.040 | 0.144 | 0.805 |
| 0.45 | 0.490 | 0.515 | +0.025 | 0.384 | 0.780 |
| 0.60 | 0.350 | 0.405 | +0.055 | 0.063 | 0.745 |
| 0.75 | 0.280 | 0.290 | +0.010 | 0.695 | 0.760 |

Mean contribution of the intervention over the probe that produced its margin: **+0.018**, and **no
single level is significant**.

The last column is the mechanism. At α = 4 the arm's prediction *is* the probe's prediction on
**83 % of items**: the standardised margin, multiplied by 4, overwhelms the difference between the
base model's candidate log-probabilities. The α grid shows the slide:

| α | arm accuracy | arm − probe | arm picks what the probe picks |
|---|---|---|---|
| 0.25 | 0.405 | −0.315 | 0.425 |
| 0.5 | 0.450 | −0.270 | 0.475 |
| 1 | 0.530 | −0.190 | 0.565 |
| 2 | 0.605 | −0.115 | 0.675 |
| 4 | 0.720 | +0.000 | 0.830 |

As α grows, the arm moves monotonically from the base model's ranking (0.370) to the probe's
ranking (0.720) and **never passes the probe**. The score-level belief-margin intervention is a
*dial between "the model decides" and "the probe decides"*, and it never produces an answer that
neither of them would have given on its own.

### Does it ever beat its own probe?

Across all 180 configurations (every layer × every label-noise level × every α), the arm beats the
probe that produced its margin in **91** of them, and the beat is nominally significant in 15.
**None survives a Bonferroni correction** over the 180 comparisons; in the other direction, **13 are
significantly *worse* than the probe**.

Where the beats come from is the interesting part: **86 of the 91 are at layers where the probe is
weak** (achieved read accuracy < 0.5). There the base model's ranking still carries information, so
base + probe can beat probe alone. The largest such beat is +0.095 at L4, where the probe reads
0.420 and the base model reads 0.370 — an arm that beats the probe but gains little over the base
model it was supposed to improve on. At a strong-probe layer the arm slides from base to probe and
does not pass it.

*(The paired Wilcoxon test is two-sided, so a small p-value on a negative difference means the arm
is significantly **worse** than the probe. Only the sign carries the claim.)*

---

## 7. Cost

| quantity | measured |
|---|---|
| base pass (200 items × 6 layers × 3 candidates, base + oracle) | 1169 s |
| one activation-steering arm (200 items × 3 candidates) | 566 s |
| one control arm | 543–629 s |
| the whole scoring-arm sweep (90 arms over cached quantities) | < 60 s |
| Fisher-LDA fit, 600 pairs × 896 dims | 0.18 s |
| model load | 17 s |
| total wall clock for everything reported here | ≈ 2 h 10 min on 2 vCPU |

The scoring arms are free because they act on cached quantities. That is the single design decision
that made a 6-layer × 5-α sweep affordable on a laptop, and it is why the matrix can be extended
rather than sampled.

---

## 8. What was NOT run

This section exists because a null result that is not reported is a hidden result, and an
unreported omission is worse.

| not run | why | what it would change |
|---|---|---|
| **A second model.** Everything is `Qwen/Qwen2.5-0.5B-Instruct`. | 0.5B was chosen so the whole matrix fits in 3 GB and on a laptop. | The base model is at chance; a stronger model might have a partner-belief signal to steer, which would change every conclusion. **This is the most important omission.** |
| **Activation steering at layer 24.** | Not possible: the hook perturbs a decoder layer's input and layer 24 is the final output. | Leaves the deepest layer out of the bite curve. |
| **Free-form generation, and with it the MAUVE / ΔPPL utility floor.** | Every arm scores three fixed candidate continuations; nothing generates text. | The utility floor is unmeasured, so we cannot say whether the steered model is still a fluent model. Task `C7` stays open. |
| **The `Hi-ToM` real dataset** (`src/bmd/real_data.py`, task `D9`). | The synthetic generator gives exact ground truth for the two error modes; the real dataset does not label them. | No claim here transfers to real ToM data. |
| **A second-order belief subspace** (what the partner believes about *me*). | Planned for Phase 2. | — |
| **An estimated belief in place of the ORACLE.** | The ORACLE is the whole point of Phase 0: it is the upper bound, not a deployable method. | The method cannot be deployed with what was measured here. |
| **Other probe families** (`logreg`, `mlp`). | `lda` was pre-registered for Phase 0. | Would test whether the read curve is an artefact of the LDA estimator. |
| **α values below 0.25 for the `logit` mechanism.** | The `logit` bias saturates the argmax even at α = 0.25. | The α grid was wrong for that mechanism; it carries no information as run. |
| **Any seed other than 0.** | One seed. | The bootstrap CI quantifies item sampling, not run-to-run variation. The control arms did give one data point on run-to-run variation: the re-run of the real arm reproduced exactly. |

---

## 9. Threats to validity

1. **One model, one seed, one scale.** Every conclusion is about a 0.5B model on 200 synthetic
   items. The base model is at chance, which is a property of *this* model on *this* task.
2. **The synthetic task is easier than a real ToM task in one specific way**: the three candidate
   locations are single nouns with identical token shape, so no length bias confounds the
   likelihood comparison. That is a deliberate control, and it is also a simplification.
3. **The probe is fitted on candidates, not on the prompt.** It reads "which of these three
   continuations matches the belief" — a different question from "what does the model believe".
   The 0.720 at layer 16 is a *discrimination* result, not a readout of a belief state.
4. **The p-values are exact.** SciPy was installed for the final statistics pass, so every
   p-value in this document comes from `scipy.stats.wilcoxon` with `zero_method="wilcox"`. The
   SciPy-free fallback in `bmd.evaluate.paired_test` is retained and is asserted by the CI suite
   to agree with the exact test — it previously did **not**, because it gave tied absolute
   differences distinct ranks instead of their average rank, which made it anti-conservative.
   Every statistic here was recomputed with `tools/recompute_all_stats.py` from the stored
   per-item predictions, so the numbers can be re-derived independently.
5. **The random-direction control now has a distribution, but few draws.** Six draws at layer 4,
   two at L8 and L12 and L20, four at L16. Six draws can separate "the real arm is the best of
   seven" (p = 1/7 = 0.143 under exchangeability) from "the real arm beats six of six by paired
   sign" (p = 0.031), but it cannot do much more. The direction of the effect is consistent across
   layers; its magnitude is not well resolved.
6. **Five layers cannot establish a rank correlation.** Every correlation in this document is
   computed over five layers, where even a perfect negative correlation reaches only p = 2/120.
   The dissociation is reported as suggestive at p = 0.083, and the read-vs-bite correlation as
   suggestive at p = 0.350.

---

## 10. Reproducing this

```bash
python src/run_phase0.py --n-items 200 --layers 4,8,12,16,20,24 \
    --act-alphas 1.0 --score-alphas 0.25,0.5,1.0,2.0,4.0 --out-dir results
python src/run_controls.py --layer 4 --alpha 1.0 --n-items 200 --out-dir results/controls
bash tools/run_random_control_sweep.sh    # the repeated random-direction controls
python tools/analyze_heldout.py
python tools/read_bite_dissociation.py     # the read-vs-bite rank correlation
python tools/belief_accuracy_frontier.py  # the belief-accuracy frontier (no model loaded)
python tools/recompute_stats.py            # re-derives every statistic from the stored predictions
python tools/make_phase0_figures.py
python tools/check_integrity.py
```

The base pass is cached in `results/base_cache.npz`, keyed by model, item count, seed and layers,
and each arm writes its own JSON as it completes, so a killed run resumes instead of starting over.
The arm JSONs store the **per-item predictions**, so every statistic above can be recomputed
without re-running the model.

**Files.**

| path | what |
|---|---|
| `results/phase0_summary.json` | the matrix summary: config, per-layer probe fits, every arm |
| `results/ablation/*.json` | one file per arm |
| `results/controls/*.json` | the matched activation arms, with per-item predictions |
| `results/controls/act_random_s*.json` | the repeated random-direction draws, one file per draw |
| `results/heldout/heldout_analysis.json` | the leakage-controlled cross-validation |
| `results/dissociation/read_bite_dissociation.json` | the read-vs-bite rank correlation |
| `results/frontier/belief_accuracy_frontier.json` | the belief-accuracy × steering-gain frontier |
| `results/tables.md` | the tables above, generated from the JSON |
| `results/base_cache.npz` | the cached activations (not committed — 14 MB) |

---

## 11. What this phase licenses us to say

**Supported by these data:**

- A partner belief is linearly decodable from a frozen LLM's activations, peaking at layer 16 with
  0.720 out-of-fold 3-way ranking accuracy against a 0.333 chance level, and is *not* decodable at
  layer 4.
- Score-level belief-margin interventions outperform a shuffled-label control at every layer from 8
  upward, and their accuracy converges to the probe's own read-out accuracy.
- **The score-level intervention is a dial between "the model decides" and "the probe decides", and
  it never produces an answer neither would have given.** At L16 and α = 4 the arm's prediction is
  the probe's prediction on 83 % of items and its accuracy (0.720) is exactly the probe's (0.720).
  Across 180 configurations the arm beats its own probe in 91, but none survives a Bonferroni
  correction, while 13 are significantly worse. The break-even belief accuracy is ≈ 0.34.
- Activation steering at layer 4 improves the model by 11 points over base, beats its sign-flipped
  control (p = 0.0065), and beats **6 of 6** matched random directions of the same size at the
  same layer (sign-test p = 0.031; no random draw reaches its accuracy).
- **The belief direction's steering advantage over random directions is negative at the two layers
  where the belief reads best** (−0.011 at L16, −0.027 at L20) and positive at the layer where it
  cannot be read at all (+0.080 at L4). Spearman ρ(read accuracy, advantage) = **−0.900**, exact
  permutation p = 0.083 — strongly suggestive, and short of significance with five layers.
- `Qwen/Qwen2.5-0.5B-Instruct` has no measurable partner-belief signal on this task at baseline
  (0.370 vs 0.333 chance, p = 0.29), and stating the belief in the prompt lifts it to 0.510.

**Not supported, and not claimed:**

- That any decoder-level intervention makes the model use a partner belief.
- That the layer-16 signal is usable by the model.
- That any of this transfers to a larger model, a real ToM dataset, or free-form generation.
- That the in-sample margin numbers (up to 0.910) represent an intervention. They do not.
- That the read/bite inversion is established. The read-vs-bite rank correlation is −0.600 with an
  exact permutation p-value of 0.350; the sharper advantage form is −0.900 with p = 0.083. Both are
  **suggestive, not demonstrated**, and both are limited by having only five layers.
- That the layer-4 steering effect is *about the partner belief*. It is distinguishable from a
  random direction, but it happens at the one layer where the belief cannot be read, and at the
  layers where the belief can be read the belief direction steers no better than a random one.
