# Belief-Margin Decoding

**Can a training-free, decoder-level intervention on a partner belief recover what simply
*telling* the model the belief recovers — and if it cannot, is the reason that the belief is not
linearly encoded, or that it is encoded but not used?**

This repository answers that question by measurement. It is deliberately built so that **a null
result is a complete, publishable outcome**.

---

## The claim, stated with its caveats

**Claim.** In a frozen small LLM, the belief a fictional protagonist holds is *linearly decodable*
from the model's internal representations, and a **closed-form, gradient-free margin** added to the
logits can steer the model's answers toward that belief.

**Caveat 1 — the belief is given, not inferred.** The Phase-0 experiment supplies the belief as
ground truth. The *estimator* that would infer a belief in deployment is Phase-1 work and is
**not implemented here**.

**Caveat 2 — the task is synthetic.** The benchmark is a three-agent scene generated with exact
ground truth, not a published dataset. It exists to make the mechanism testable at laptop scale
with unambiguous labels. A loader for `Hi-ToM/Hi-ToM_Dataset` is planned (see `PROJECT.md`, D9).

**Caveat 3 — the scale is small.** `Qwen/Qwen2.5-0.5B-Instruct` on CPU. Nothing here is a claim
about frontier models.

**Caveat 4 — the dissociation is not new.** That a linearly decodable feature can be causally inert
is established (see `METHODS.md` §0 for the prior art). This repository's contribution is the
*narrow* question above, applied to a partner belief, at the logit level, against a prompt oracle.

---

## What the repository contains

| | |
|---|---|
| **A falsification matrix** | `src/bmd/matrix.py` — 4 mechanisms x 6 layers x 5 strengths, plus two controls |
| **A synthetic ToM benchmark** | `src/bmd/data.py` — three agents, three locations, exact ground truth, two distinguishable error modes |
| **A closed-form belief subspace** | `src/bmd/subspace.py` — shrinkage Fisher LDA, one linear solve, milliseconds |
| **The interventions** | `src/bmd/intervene.py` — activation steering, margin decoding, logit bias |
| **Metrics and statistics** | `src/bmd/evaluate.py` — three-way error taxonomy, bootstrap CIs, Wilcoxon signed-rank |
| **A zero-background explainer** | [`EXTENDED_INTRODUCTION.md`](EXTENDED_INTRODUCTION.md) |
| **The methods log** | [`METHODS.md`](METHODS.md) — Done / Intended / Undecided, with selection rules fixed in advance |
| **The pre-registered plan** | [`PROJECT.md`](PROJECT.md) — atomic tasks and the publication gate |
| **The measured results** | [`../results/RESULTS.md`](../results/RESULTS.md) |

---

## How to read the results in one minute

1. **Two curves.** The *read* curve (probe accuracy per layer) says where the belief is decodable.
   The *bite* curve (intervention effect per layer) says where it can be used. They can disagree.
2. **The oracle arm.** `prompt_oracle` states the belief in the prompt using ground truth. It is
   the upper bound that every decoder-level arm is measured against.
3. **The shuffled control.** `margin_shuffled` repeats the intervention with a subspace fitted on
   permuted labels. An arm that does not beat its shuffled control is **no effect**.
4. **Null is a result.** The pre-registered thresholds in `PROJECT.md` §9 are fixed before the run
   and are not adjusted afterwards.

---

## Reproducing

```bash
pip install -r requirements-dev.txt
pip install -e .

# the full Phase-0 matrix (CPU-only, no GPU required)
python src/run_phase0.py --n-items 200 --layers 4,8,12,16,20,24 \
    --act-alphas 1.0 --score-alphas 0.25,0.5,1.0,2.0,4.0 --out-dir results

# integrity checks that can fail
python tools/check_integrity.py
```

The run is **resumable**: the base pass is cached to `results/base_cache.npz` and each arm writes
its own JSON, so a killed run continues instead of restarting.

**Requirements:** CPU or Apple Silicon (MPS). No CUDA. The reference machine for the recorded run
was a 2-vCPU / 3 GB Linux container; an M-series Mac is materially faster.

---

## Repository status

Private until the pre-registered publication gate in [`PROJECT.md`](PROJECT.md) §9 is satisfied:
at least 50 % of the code tasks done, at least 50 % of the results tasks done, at least one fully
measured arm reported, and CI green on `main`.
