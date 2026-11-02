# Read-vs-bite dissociation

Every number below is re-derived from per-item predictions stored on disk; no model was run to produce this file.

Base model accuracy: **0.370**. Layers: L4, L8, L12, L16, L20.

## Per layer

| layer | read (5-fold item CV, 3-way) | read shuffled control | bite (act alpha=1) | bite - base | bite - read | p(bite vs base) | random draws |
|---|---|---|---|---|---|---|---|
| L4 | 0.420 | 0.430 | 0.480 | +0.110 | +0.060 | 0.004 | 1 |
| L8 | 0.545 | 0.345 | 0.345 | -0.025 | -0.200 | 0.446 | 0 |
| L12 | 0.530 | 0.385 | 0.405 | +0.035 | -0.125 | 0.297 | 0 |
| L16 | 0.720 | 0.375 | 0.400 | +0.030 | -0.320 | 0.330 | 0 |
| L20 | 0.695 | 0.360 | 0.370 | +0.000 | -0.325 | 1.000 | 0 |

## Rank correlation between the two curves

- Spearman rho = **-0.600**, exact permutation p = **0.350** (120 permutations, 5! orderings).
- Kendall tau-b = **-0.400**, exact permutation p = **0.483**.
- Best read: **L16** (0.720). Best bite: **L4** (0.480).

A **negative** correlation means the layers where the partner belief is most readable are the layers where steering it does least. That is the dissociation the Phase-0 matrix reported in prose.

## Real direction vs each random-direction draw (paired, by item)

| layer | real | random | real - random | p (paired Wilcoxon) |
|---|---|---|---|---|
| L4 | 0.480 (act_real) | 0.430 (act_random) | +0.050 | 0.189 |

Real direction is ahead in **1** of 1 pairwise comparisons; the random direction is ahead in 0.

