# Roadmap

Each phase names its **principal failure mode** — the thing most likely to make the phase fail.
A phase whose failure mode materialises is reported, not worked around.

---

## Phase 0 — Falsification matrix *(complete)*

**Question.** Does a training-free, decoder-level intervention on a partner belief change the
model's answers, and can it match an oracle prompt that states the belief outright?

**Deliverables.** `src/bmd/*`, the matrix runner, the results document.

**Principal failure mode.** *The belief is not linearly decodable.* Belief-consistency is a
**relational** property — "does this location match the one the protagonist believes?" — and a
linear readout of a joint (story, option) representation may be unable to express a comparison.
If the held-out probe accuracy is at chance, the repository reports that and stops: there is
nothing to steer with.

**Second failure mode.** *The intervention changes the text but not the decision.* Steering a
0.5B model on a scored multiple-choice task may move token probabilities without flipping the
chosen option, producing a null that means nothing about the mechanism.

---

## Phase 0b — Making the negative result an actual claim *(complete)*

**Question.** Are the two Phase-0 statements that rested on a single random draw and on prose
actually true?

**Deliverables.** The repeated random-direction sweep, the read-vs-bite rank correlation with an
exact permutation p-value, the belief-accuracy × steering-gain frontier, and the frontier figure.

**Principal failure mode — and it did not materialise.** *The repeated random control would
contradict the single-draw result.* It did not: at layer 4 the real arm beat **6 of 6** matched
random directions (sign-test p = 0.031, 0 draws reached its accuracy), where the single draw had
given p = 0.189. **The single draw happened to be one of the more favourable random directions**,
and the one-draw conclusion was underpowered in the direction of *understating* the effect.

**What this changed.** The Phase-0 statement "directional but not shown to be belief-specific" is
now "directional and distinguishable from a matched random direction of the same size at the same
layer". That is an upgrade — and it does **not** rescue the phase's central negative result, because
layer 4 is the layer where the belief probe has *no* signal (0.420). A direction that cannot read
the belief but steers better than a random direction is still not a belief.

---

## Phase 1 — From oracle to estimator

**Question.** Can a **frozen, training-free** estimator recover a partner's belief well enough for
the decoder to exploit?

**Deliverables.** A belief estimator over fixed slots (`beliefs`, `goals`, `knowledge`,
`intentions`), a second-order slot set (`what the partner believes about me`), and the
belief-accuracy sweep.

**Principal failure mode.** *The estimator's errors dominate.* A decoder that amplifies a wrong
belief is worse than no decoder, and the Phase-0 result (measured against an oracle) says nothing
about how much belief error the method tolerates. This is why the confidence gate (Phase 2) is
not optional.

**What Phase 0b added.** The frontier measured that break-even directly for the score-level
mechanism: the intervention is net-positive only while the probe's **achieved** read accuracy stays
above ≈ 0.34, barely above the 0.333 chance level. But the frontier also showed why the mechanism
is uninteresting in this form: at large α the arm's prediction **is** the probe's prediction, so
the arm's accuracy is the probe's accuracy and the intervention adds +0.018 on average (not
significant at any level). The estimator therefore has to be good *and* the intervention has to be
applied somewhere it adds to the model's computation rather than replacing it.

---

## Phase 2 — Calibration gate and the break-even point

**Question.** How accurate must a belief be before steering on it is net-positive?

**Deliverables.** Deliberate belief-noise injection at 0–40 %, the outcome curve against belief
accuracy, the break-even point, and a training-free confidence gate on `alpha`.

**Status.** The belief-noise injection and the break-even point are **done** for the score-level
mechanism (`tools/belief_accuracy_frontier.py`): the break-even is ≈ 0.34 achieved read accuracy,
bracket [0.280, 0.350], with the least-squares crossing at 0.337. What is **not** done is the same
curve for the **forward-pass** mechanism, which is where the question actually bites, because that
mechanism costs a forward pass per configuration.

**Principal failure mode.** *The frontier is flat.* If belief accuracy does not interact with
steering gain, the gate is decorative and the honest result is "belief accuracy does not
moderate this intervention". That is still a result.

**Second failure mode.** *The gate is uncalibrated.* The estimator's own confidence may be
uncorrelated with its accuracy, in which case scaling `alpha` by it does nothing except
reintroduce noise. Calibration must be **measured**, not assumed.

---

## Phase 2b — A second model *(the next experiment, and it needs a real machine)*

**Question.** Is any of this a fact about the mechanism, or about a 0.5B model that is **at chance**
on the task (0.370, p = 0.29)?

**Deliverables.** The whole Phase-0 matrix, the controls, the sweep and the frontier, re-run on
`Qwen/Qwen2.5-1.5B-Instruct` in **bfloat16** (3.1 GB instead of 6.2 GB), via
`tools/run_second_model_macos.sh`. The 0.5B results are not touched; everything lands in
`results/qwen1_5b/`.

**Principal failure mode.** *The 1.5B model is still at chance.* A base model that cannot do the
task may not encode a partner belief worth steering, and "we could not steer the belief" would
then be a statement about the model rather than about the mechanism. This is the **most important
omission** in the repository and the reason the second model comes before any further mechanism
work.

**Second failure mode.** *Two models is not a model family.* If the conclusion holds for both, it
is still one architecture family. `HuggingFaceTB/SmolLM2-360M-Instruct` and `gpt2` are the
pre-registered follow-ups.

---

## Phase 3 — The prompt-oracle comparison, properly

**Question.** How does a decoder-level intervention compare with **simply telling** the model the
belief — in prompt tokens, in retrieved context, and in a tool result?

**Deliverables.** A three-way comparison (decoder intervention vs. prompt injection vs. retrieval
of the belief as context), at matched token cost, on the same items.

**Principal failure mode.** *The comparison is confounded by token cost.* A prompt statement
costs tokens; a margin costs none at inference. The comparison must be reported at **matched
budget**, or it is meaningless.

---

## Phase 4 — Free-form generation and the utility floor

**Question.** Does the method survive when the model must produce free text rather than choose
among three fixed candidates?

**Deliverables.** Per-token margin decoding with the mandatory top-k restriction, ΔPPL and MAUVE
guards, and generation-level human spot-checks.

**Principal failure mode.** *Per-token margin is too expensive.* The feasibility pilot measured
**1298 ms per token** for full-vocabulary margin computation against **0.17 ms** for top-64 — a
64-token reply would carry **83 seconds** of overhead against **11 milliseconds**. The top-k
restriction is not an optimisation, it is the difference between viable and impossible on a
laptop.

---

## Phase 5 — Generalisation

**Question.** Does any of this survive a real benchmark and a different model?

**Deliverables.** `Hi-ToM/Hi-ToM_Dataset` and BigToM loaders, a second model family
(`HuggingFaceTB/SmolLM2-360M-Instruct`, `gpt2`), and a **partner-swap portability matrix** —
does a method's gain transfer across partner models, or is it tuned to one?

**Principal failure mode.** *The synthetic conclusions do not transfer.* The synthetic task has
exact labels and a known error taxonomy; a published benchmark has neither. If the mechanism
depends on the synthetic task's regularity, the Phase-0 result is about the task, not the method.

---

## Explicitly not planned

- **A new steering algorithm for steering's sake.** The literature already has a crowded field
  here (SWAI, CLS, LQR steering, Steering Vector Fields). This repository tests one mechanism
  against an oracle for a specific, under-served target.
- **Training anything.** No gradient steps on the language model, at any phase.
- **A partner model at inference.** The whole point is that the decoder needs no auxiliary model.
