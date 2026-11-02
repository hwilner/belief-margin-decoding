# Belief-accuracy x steering-gain frontier

Computed from the cached base pass: **no model was loaded**. The probe's quality is simulated by corrupting its training labels, and every quantity is out-of-fold under item-level 5-fold cross-validation.

Base model accuracy: **0.370**.

## The frontier at the best-reading layer

Layer **16** (the layer with the best Phase-0 belief readout), alpha = 4 (fixed before the sweep).

| label noise | achieved read acc (3-way) | 95% CI | margin arm | shuffled control | arm - base | p vs control | p vs base |
|---|---|---|---|---|---|---|---|
| 0 | 0.720 | [0.655, 0.780] | 0.720 | 0.330 | +0.350 | 2.03e-14 | 2.21e-14 |
| 0.15 | 0.655 | [0.585, 0.725] | 0.635 | 0.320 | +0.265 | 1.59e-10 | 1.54e-09 |
| 0.3 | 0.535 | [0.465, 0.605] | 0.575 | 0.400 | +0.205 | 0.000497 | 6.78e-06 |
| 0.45 | 0.490 | [0.425, 0.555] | 0.515 | 0.340 | +0.145 | 0.000175 | 0.000322 |
| 0.6 | 0.350 | [0.285, 0.415] | 0.405 | 0.365 | +0.035 | 0.365 | 0.392 |
| 0.75 | 0.280 | [0.220, 0.340] | 0.290 | 0.310 | -0.080 | 0.655 | 0.0455 |

**Break-even:** the intervention still beats the base model down to an achieved read accuracy of **0.350** (5 of 6 noise levels), and fails from 0.280 upwards. The least-squares crossing of the fitted arm-accuracy line with the base model is at a read accuracy of **0.337**.

### What the intervention adds over the probe that produced it

The arm's accuracy is the probe's accuracy plus whatever the intervention contributes. This column is that contribution.

| label noise | achieved read acc | margin arm | arm - probe | p (paired) | arm picks what the probe picks |
|---|---|---|---|---|---|
| 0 | 0.720 | 0.720 | +0.000 | 1 | 0.830 |
| 0.15 | 0.655 | 0.635 | -0.020 | 0.45 | 0.835 |
| 0.3 | 0.535 | 0.575 | +0.040 | 0.144 | 0.805 |
| 0.45 | 0.490 | 0.515 | +0.025 | 0.384 | 0.780 |
| 0.6 | 0.350 | 0.405 | +0.055 | 0.063 | 0.745 |
| 0.75 | 0.280 | 0.290 | +0.010 | 0.695 | 0.760 |

Mean contribution over the noise levels: **+0.018** (min -0.020, max +0.055).

### The alpha grid, at the cleanest probe

Layer L16, label noise 0, probe read accuracy 0.720, base model 0.370.

| alpha | arm accuracy | arm - probe | arm picks what the probe picks | p vs base |
|---|---|---|---|---|
| 0.25 | 0.405 | -0.315 | 0.425 | 0.00815 |
| 0.5 | 0.450 | -0.270 | 0.475 | 6.33e-05 |
| 1 | 0.530 | -0.190 | 0.565 | 4.07e-08 |
| 2 | 0.605 | -0.115 | 0.675 | 4.81e-10 |
| 4 | 0.720 | +0.000 | 0.830 | 2.21e-14 |

The agreement column is the mechanism. At large alpha the margin term dominates the base log-probabilities, the arm's prediction *is* the probe's prediction, and the arm's accuracy is the probe's accuracy. The intervention is then not changing the model's answer; it is substituting the probe's answer for it.

### Does the intervention ever beat its own probe?

Across all 180 configurations (every layer x every label-noise level x every alpha), the arm beats the probe that produced its margin in **91** of them, and the beat is nominally significant (paired p < 0.05) in **15**.

- **Where the probe is strong** (achieved read accuracy >= 0.5): 5 of the beats.
- **Where the probe is weak** (achieved read accuracy < 0.5): 86 of the beats.
- Surviving a Bonferroni correction over all 180 comparisons (p < 2.78e-04): **0** beat the probe; in the other direction, **13** are significantly **worse** than the probe.

The paired Wilcoxon test is two-sided, so a small p-value on a negative difference means the arm is significantly *worse* than the probe. Only the sign carries the claim.

At a strong-probe layer the arm slides from the base model's ranking to the probe's and does not pass it: the margin term replaces the model's contribution instead of adding to it. The largest beat at a strong-probe layer is +0.060.

Where the probe is weak, the base model's ranking still carries information, and base + probe can beat probe alone. The largest such beat is +0.095 (L4, noise 0, margin-a1), where the probe reads 0.420 and the base model reads 0.370 -- an arm that beats the probe but gains little over the base model it was supposed to improve on.

## Every layer

| layer | noise | achieved read acc | margin arm (alpha=4) | shuffled control | arm - base |
|---|---|---|---|---|---|
| L4 | 0 | 0.420 | 0.420 | 0.390 | +0.050 |
| L4 | 0.15 | 0.420 | 0.400 | 0.410 | +0.030 |
| L4 | 0.3 | 0.365 | 0.370 | 0.365 | +0.000 |
| L4 | 0.45 | 0.340 | 0.340 | 0.315 | -0.030 |
| L4 | 0.6 | 0.350 | 0.385 | 0.395 | +0.015 |
| L4 | 0.75 | 0.255 | 0.315 | 0.390 | -0.055 |
| L8 | 0 | 0.545 | 0.545 | 0.355 | +0.175 |
| L8 | 0.15 | 0.470 | 0.490 | 0.350 | +0.120 |
| L8 | 0.3 | 0.420 | 0.440 | 0.360 | +0.070 |
| L8 | 0.45 | 0.385 | 0.390 | 0.360 | +0.020 |
| L8 | 0.6 | 0.335 | 0.365 | 0.405 | -0.005 |
| L8 | 0.75 | 0.285 | 0.330 | 0.375 | -0.040 |
| L12 | 0 | 0.530 | 0.590 | 0.410 | +0.220 |
| L12 | 0.15 | 0.490 | 0.490 | 0.385 | +0.120 |
| L12 | 0.3 | 0.430 | 0.440 | 0.405 | +0.070 |
| L12 | 0.45 | 0.410 | 0.410 | 0.355 | +0.040 |
| L12 | 0.6 | 0.280 | 0.345 | 0.420 | -0.025 |
| L12 | 0.75 | 0.265 | 0.330 | 0.325 | -0.040 |
| L16 | 0 | 0.720 | 0.720 | 0.330 | +0.350 |
| L16 | 0.15 | 0.655 | 0.635 | 0.320 | +0.265 |
| L16 | 0.3 | 0.535 | 0.575 | 0.400 | +0.205 |
| L16 | 0.45 | 0.490 | 0.515 | 0.340 | +0.145 |
| L16 | 0.6 | 0.350 | 0.405 | 0.365 | +0.035 |
| L16 | 0.75 | 0.280 | 0.290 | 0.310 | -0.080 |
| L20 | 0 | 0.695 | 0.695 | 0.370 | +0.325 |
| L20 | 0.15 | 0.615 | 0.605 | 0.370 | +0.235 |
| L20 | 0.3 | 0.490 | 0.495 | 0.390 | +0.125 |
| L20 | 0.45 | 0.470 | 0.495 | 0.335 | +0.125 |
| L20 | 0.6 | 0.345 | 0.355 | 0.345 | -0.015 |
| L20 | 0.75 | 0.275 | 0.310 | 0.325 | -0.060 |
| L24 | 0 | 0.595 | 0.625 | 0.405 | +0.255 |
| L24 | 0.15 | 0.535 | 0.550 | 0.390 | +0.180 |
| L24 | 0.3 | 0.485 | 0.525 | 0.455 | +0.155 |
| L24 | 0.45 | 0.455 | 0.465 | 0.300 | +0.095 |
| L24 | 0.6 | 0.360 | 0.380 | 0.370 | +0.010 |
| L24 | 0.75 | 0.290 | 0.315 | 0.370 | -0.055 |

