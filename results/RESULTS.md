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

**This is the result the phase exists to produce.**

- The effect is **directional**: pushing along the belief direction beats pushing against it
  (p = 0.0065). It is not noise.
- The effect is **not shown to be belief-specific**: a random direction of the same magnitude at
  the same layer reaches 0.430, and the gap to the real direction is **not significant**
  (p = 0.189, CI [−0.025, +0.120]). The data are compatible with the belief direction being up to
  12 points better than a random one, or 2.5 points worse.
- A **re-run of the real arm reproduced the matrix result exactly** (0.480, identical to the
  matrix's `act-L4-a1`), so the effect is deterministic, not a fluke of the run.

The honest statement is therefore: *at layer 4, perturbing the residual stream in the direction of
the fitted belief subspace improves the model's answers by 11 points, but this improvement is not
distinguishable from the improvement produced by perturbing it in an arbitrary direction of the
same size.* One random draw is not a distribution, so the correct next step is to repeat the random
control over several seeds; the current data cannot settle it.

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
   the direction being pushed is not the belief direction in any meaningful sense — which is
   consistent with the random control performing nearly as well.

The matched control is what makes reading 2 the more parsimonious one, and it is the reason the
control exists.

---

## 6. Cost

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

## 7. What was NOT run

This section exists because a null result that is not reported is a hidden result, and an
unreported omission is worse.

| not run | why | what it would change |
|---|---|---|
| **A second model.** Everything is `Qwen/Qwen2.5-0.5B-Instruct`. | 0.5B was chosen so the whole matrix fits in 3 GB and on a laptop. | The base model is at chance; a stronger model might have a partner-belief signal to steer, which would change every conclusion. **This is the most important omission.** |
| **Repeated random-direction controls.** One random draw was run at layer 4. | 9.5 min per draw on this hardware. | The real-vs-random comparison (p = 0.189) is underpowered. Five to ten draws would turn it into a proper test. |
| **Activation steering at layer 24.** | Not possible: the hook perturbs a decoder layer's input and layer 24 is the final output. | Leaves the deepest layer out of the bite curve. |
| **Free-form generation, and with it the MAUVE / ΔPPL utility floor.** | Every arm scores three fixed candidate continuations; nothing generates text. | The utility floor is unmeasured, so we cannot say whether the steered model is still a fluent model. Task `C7` stays open. |
| **The `Hi-ToM` real dataset** (`src/bmd/real_data.py`, task `D9`). | The synthetic generator gives exact ground truth for the two error modes; the real dataset does not label them. | No claim here transfers to real ToM data. |
| **A second-order belief subspace** (what the partner believes about *me*). | Planned for Phase 2. | — |
| **An estimated belief in place of the ORACLE.** | The ORACLE is the whole point of Phase 0: it is the upper bound, not a deployable method. | The method cannot be deployed with what was measured here. |
| **Other probe families** (`logreg`, `mlp`). | `lda` was pre-registered for Phase 0. | Would test whether the read curve is an artefact of the LDA estimator. |
| **α values below 0.25 for the `logit` mechanism.** | The `logit` bias saturates the argmax even at α = 0.25. | The α grid was wrong for that mechanism; it carries no information as run. |
| **Any seed other than 0.** | One seed. | The bootstrap CI quantifies item sampling, not run-to-run variation. The control arms did give one data point on run-to-run variation: the re-run of the real arm reproduced exactly. |

---

## 8. Threats to validity

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
5. **The random-direction control is a single draw.** A single draw of a random variable is not
   its distribution, and the real-vs-random comparison is the one this phase cannot settle.

---

## 9. Reproducing this

```bash
python src/run_phase0.py --n-items 200 --layers 4,8,12,16,20,24 \
    --act-alphas 1.0 --score-alphas 0.25,0.5,1.0,2.0,4.0 --out-dir results
python src/run_controls.py --layer 4 --alpha 1.0 --n-items 200 --out-dir results/controls
python tools/analyze_heldout.py
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
| `results/controls/*.json` | the three matched activation arms, with per-item predictions |
| `results/heldout/heldout_analysis.json` | the leakage-controlled cross-validation |
| `results/tables.md` | the tables above, generated from the JSON |
| `results/base_cache.npz` | the cached activations (not committed — 14 MB) |

---

## 10. What this phase licenses us to say

**Supported by these data:**

- A partner belief is linearly decodable from a frozen LLM's activations, peaking at layer 16 with
  0.720 out-of-fold 3-way ranking accuracy against a 0.333 chance level, and is *not* decodable at
  layer 4.
- Score-level belief-margin interventions outperform a shuffled-label control at every layer from 8
  upward, and their accuracy converges to the probe's own read-out accuracy.
- Activation steering at layer 4 improves the model by 11 points over base, and beats its
  sign-flipped control, but does **not** significantly beat a matched random direction.
- `Qwen/Qwen2.5-0.5B-Instruct` has no measurable partner-belief signal on this task at baseline
  (0.370 vs 0.333 chance, p = 0.29), and stating the belief in the prompt lifts it to 0.510.

**Not supported, and not claimed:**

- That any decoder-level intervention makes the model use a partner belief.
- That the layer-16 signal is usable by the model.
- That any of this transfers to a larger model, a real ToM dataset, or free-form generation.
- That the in-sample margin numbers (up to 0.910) represent an intervention. They do not.
