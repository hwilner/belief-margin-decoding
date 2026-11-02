# Read-vs-bite dissociation

Every number below is re-derived from per-item predictions stored on disk; no model was run to produce this file.

Base model accuracy: **0.370**. Layers: L4, L8, L12, L16, L20.

## Per layer

| layer | read (5-fold item CV, 3-way) | read shuffled control | bite (act alpha=1) | bite - base | bite - read | p(bite vs base) | random draws |
|---|---|---|---|---|---|---|---|
| L4 | 0.420 | 0.430 | 0.480 | +0.110 | +0.060 | 0.004 | 6 |
| L8 | 0.545 | 0.345 | 0.345 | -0.025 | -0.200 | 0.446 | 2 |
| L12 | 0.530 | 0.385 | 0.405 | +0.035 | -0.125 | 0.297 | 2 |
| L16 | 0.720 | 0.375 | 0.400 | +0.030 | -0.320 | 0.330 | 4 |
| L20 | 0.695 | 0.360 | 0.370 | +0.000 | -0.325 | 1.000 | 2 |

## Rank correlation between the two curves

- Spearman rho = **-0.600**, exact permutation p = **0.350** (120 permutations, 5! orderings).
- Kendall tau-b = **-0.400**, exact permutation p = **0.483**.
- Best read: **L16** (0.720). Best bite: **L4** (0.480).

A **negative** correlation means the layers where the partner belief is most readable are the layers where steering it does least. That is the dissociation the Phase-0 matrix reported in prose.

## Real direction vs each random-direction draw (paired, by item)

| layer | real | random | real - random | p (paired Wilcoxon) |
|---|---|---|---|---|
| L4 | 0.480 (act_real) | 0.430 (act_random_s977_L4) | +0.050 | 0.189 |
| L4 | 0.480 (act_real) | 0.450 (act_random_s978_L4) | +0.030 | 0.439 |
| L4 | 0.480 (act_real) | 0.420 (act_random_s979_L4) | +0.060 | 0.102 |
| L4 | 0.480 (act_real) | 0.400 (act_random_s980_L4) | +0.080 | 0.033 |
| L4 | 0.480 (act_real) | 0.340 (act_random_s981_L4) | +0.140 | 1.83e-04 |
| L4 | 0.480 (act_real) | 0.360 (act_random_s982_L4) | +0.120 | 0.001 |
| L8 | 0.345 (act-L8-a1) | 0.315 (act_random_s977_L8) | +0.030 | 0.273 |
| L8 | 0.345 (act-L8-a1) | 0.340 (act_random_s978_L8) | +0.005 | 0.884 |
| L12 | 0.405 (act-L12-a1) | 0.385 (act_random_s977_L12) | +0.020 | 0.564 |
| L12 | 0.405 (act-L12-a1) | 0.325 (act_random_s978_L12) | +0.080 | 0.070 |
| L16 | 0.400 (act-L16-a1) | 0.455 (act_random_s977_L16) | -0.055 | 0.063 |
| L16 | 0.400 (act-L16-a1) | 0.410 (act_random_s978_L16) | -0.010 | 0.758 |
| L16 | 0.400 (act-L16-a1) | 0.355 (act_random_s979_L16) | +0.045 | 0.241 |
| L16 | 0.400 (act-L16-a1) | 0.425 (act_random_s980_L16) | -0.025 | 0.553 |
| L20 | 0.370 (act-L20-a1) | 0.370 (act_random_s977_L20) | +0.000 | 1.000 |
| L20 | 0.370 (act-L20-a1) | 0.425 (act_random_s978_L20) | -0.055 | 0.166 |

Real direction is ahead in **11** of 16 pairwise comparisons; the random direction is ahead in 4.

## The sharper dissociation: does the belief direction beat a random one *more* where the belief reads better?

A random direction is the baseline for "perturbing the residual stream here does something". The advantage of the belief direction over the mean matched random direction isolates the part of the steering effect that could be about the belief at all.

| layer | read (out-of-fold 3-way) | belief direction | random mean | advantage | random draws that beat the belief direction |
|---|---|---|---|---|---|
| L4 | 0.420 | 0.480 | 0.400 (6 draws) | +0.080 | 0/6 |
| L8 | 0.545 | 0.345 | 0.328 (2 draws) | +0.017 | 0/2 |
| L12 | 0.530 | 0.405 | 0.355 (2 draws) | +0.050 | 0/2 |
| L16 | 0.720 | 0.400 | 0.411 (4 draws) | -0.011 | 3/4 |
| L20 | 0.695 | 0.370 | 0.397 (2 draws) | -0.027 | 1/2 |

Spearman ρ(read accuracy, advantage) = **-0.900**, exact permutation p = **0.083** (120 permutations).
Kendall τ-b = **-0.800**, exact permutation p = **0.083**.

A **negative** correlation means the belief direction's steering advantage is *largest where the belief is least readable*. That is the dissociation in its sharpest form: the steering effect is not proportional to how much of the belief the layer carries.

