# Roadmap

Each phase names its **principal failure mode** — the thing most likely to make the phase fail.
A phase whose failure mode materialises is reported, not worked around.

---

## Phase 0 — Falsification matrix *(in progress)*

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

---

## Phase 2 — Calibration gate and the break-even point

**Question.** How accurate must a belief be before steering on it is net-positive?

**Deliverables.** Deliberate belief-noise injection at 0–40 %, the outcome curve against belief
accuracy, the break-even point, and a training-free confidence gate on `alpha`.

**Principal failure mode.** *The frontier is flat.* If belief accuracy does not interact with
steering gain, the gate is decorative and the honest result is "belief accuracy does not
moderate this intervention". That is still a result.

**Second failure mode.** *The gate is uncalibrated.* The estimator's own confidence may be
uncorrelated with its accuracy, in which case scaling `alpha` by it does nothing except
reintroduce noise. Calibration must be **measured**, not assumed.

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
