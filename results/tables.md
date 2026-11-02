### Baselines

| arm | accuracy | 95% CI (bootstrap) | world rate | other rate | p vs base (Wilcoxon) |
|---|---|---|---|---|---|
| base (`none`) | 0.370 | [0.305, 0.440] | 0.280 | 0.350 | 1 |
| **`prompt_oracle` (ORACLE)** | 0.510 | [0.445, 0.585] | 0.200 | 0.290 | 5.57e-06 |

### The read curve — belief probe, per layer

| layer | probe train acc | probe held-out acc | shuffled control held-out acc | ‖w‖ |
|---|---|---|---|---|
| L4 | 0.863 | **0.592** | 0.467 | 5.784e+02 |
| L8 | 0.940 | **0.633** | 0.542 | 4.965e+02 |
| L12 | 0.963 | **0.692** | 0.475 | 4.158e+02 |
| L16 | 0.988 | **0.733** | 0.558 | 3.405e+02 |
| L20 | 0.990 | **0.683** | 0.525 | 1.278e+02 |
| L24 | 0.969 | **0.675** | 0.483 | 6.672e+00 |

### Scoring arms at alpha = 1.0 — margin, its control, and logit

| layer | margin | margin_shuffled (control) | logit | margin − control |
|---|---|---|---|---|
| L4 | 0.555 | 0.340 | 0.270 | +0.215 |
| L8 | 0.545 | 0.380 | 0.250 | +0.165 |
| L12 | 0.580 | 0.360 | 0.250 | +0.220 |
| L16 | 0.610 | 0.380 | 0.250 | +0.230 |
| L20 | 0.600 | 0.395 | 0.420 | +0.205 |
| L24 | 0.560 | 0.390 | 0.490 | +0.170 |

### Alpha sweep (margin mechanism)

| layer | a=0.25 | a=0.5 | a=1 | a=2 | a=4 |
|---|---|---|---|---|---|
| L4 | 0.405 | 0.440 | 0.555 | 0.675 | 0.750 |
| L8 | 0.420 | 0.455 | 0.545 | 0.680 | 0.830 |
| L12 | 0.420 | 0.475 | 0.580 | 0.745 | 0.880 |
| L16 | 0.420 | 0.500 | 0.610 | 0.795 | 0.905 |
| L20 | 0.415 | 0.480 | 0.600 | 0.760 | 0.910 |
| L24 | 0.410 | 0.465 | 0.560 | 0.765 | 0.870 |

### Activation-steering arms (alpha = 1.0)

| layer | accuracy | 95% CI | world rate | other rate | p vs base |
|---|---|---|---|---|---|
| L4 | 0.480 | [0.415, 0.550] | 0.205 | 0.315 | 0.00175 |
| L8 | 0.345 | [0.275, 0.415] | 0.315 | 0.340 | 0.446 |
| L12 | 0.405 | [0.340, 0.475] | 0.220 | 0.375 | 0.297 |
| L16 | 0.400 | [0.335, 0.470] | 0.265 | 0.335 | 0.33 |
| L20 | 0.370 | [0.305, 0.440] | 0.240 | 0.390 | 1 |

### Every arm

| arm | mechanism | layer | alpha | accuracy | world | other | p vs base |
|---|---|---|---|---|---|---|---|
| `margin-L20-a4` | margin | 20 | 4 | 0.910 | 0.005 | 0.085 | 2.69e-25 |
| `margin-L16-a4` | margin | 16 | 4 | 0.905 | 0.015 | 0.080 | 1.2e-24 |
| `margin-L12-a4` | margin | 12 | 4 | 0.880 | 0.005 | 0.115 | 9.71e-23 |
| `margin-L24-a4` | margin | 24 | 4 | 0.870 | 0.020 | 0.110 | 4.1e-23 |
| `margin-L8-a4` | margin | 8 | 4 | 0.830 | 0.015 | 0.155 | 3.58e-20 |
| `margin-L16-a2` | margin | 16 | 2 | 0.795 | 0.055 | 0.150 | 2.98e-20 |
| `margin-L24-a2` | margin | 24 | 2 | 0.765 | 0.055 | 0.180 | 6.21e-19 |
| `margin-L20-a2` | margin | 20 | 2 | 0.760 | 0.060 | 0.180 | 1.03e-18 |
| `margin-L4-a4` | margin | 4 | 4 | 0.750 | 0.110 | 0.140 | 4.55e-15 |
| `margin-L12-a2` | margin | 12 | 2 | 0.745 | 0.055 | 0.200 | 1.26e-17 |
| `margin-L8-a2` | margin | 8 | 2 | 0.680 | 0.075 | 0.245 | 5.54e-14 |
| `margin-L4-a2` | margin | 4 | 2 | 0.675 | 0.150 | 0.175 | 2.08e-13 |
| `margin-L16-a1` | margin | 16 | 1 | 0.610 | 0.120 | 0.270 | 4.26e-12 |
| `margin-L20-a1` | margin | 20 | 1 | 0.600 | 0.125 | 0.275 | 1.18e-11 |
| `margin-L12-a1` | margin | 12 | 1 | 0.580 | 0.100 | 0.320 | 9.13e-11 |
| `margin-L24-a1` | margin | 24 | 1 | 0.560 | 0.135 | 0.305 | 7.07e-10 |
| `margin-L4-a1` | margin | 4 | 1 | 0.555 | 0.185 | 0.260 | 7.54e-09 |
| `margin-L8-a1` | margin | 8 | 1 | 0.545 | 0.145 | 0.310 | 4.6e-08 |
| `prompt_oracle` | prompt_oracle | — | 0 | 0.510 | 0.200 | 0.290 | 5.57e-06 |
| `margin-L16-a0.5` | margin | 16 | 0.5 | 0.500 | 0.185 | 0.315 | 3.41e-07 |
| `logit-L24-a0.25` | logit | 24 | 0.25 | 0.490 | 0.280 | 0.230 | 0.0246 |
| `logit-L24-a0.5` | logit | 24 | 0.5 | 0.490 | 0.280 | 0.230 | 0.0246 |
| `logit-L24-a1` | logit | 24 | 1 | 0.490 | 0.280 | 0.230 | 0.0246 |
| `logit-L24-a2` | logit | 24 | 2 | 0.490 | 0.280 | 0.230 | 0.0246 |
| `logit-L24-a4` | logit | 24 | 4 | 0.490 | 0.280 | 0.230 | 0.0246 |
| `act-L4-a1` | act | 4 | 1 | 0.480 | 0.205 | 0.315 | 0.00175 |
| `margin-L20-a0.5` | margin | 20 | 0.5 | 0.480 | 0.185 | 0.335 | 2.73e-06 |
| `margin-L12-a0.5` | margin | 12 | 0.5 | 0.475 | 0.175 | 0.350 | 4.59e-06 |
| `margin-L24-a0.5` | margin | 24 | 0.5 | 0.465 | 0.180 | 0.355 | 1.31e-05 |
| `margin-L8-a0.5` | margin | 8 | 0.5 | 0.455 | 0.215 | 0.330 | 3.74e-05 |
| `margin-L4-a0.5` | margin | 4 | 0.5 | 0.440 | 0.245 | 0.315 | 0.000465 |
| `margin-L8-a0.25` | margin | 8 | 0.25 | 0.420 | 0.245 | 0.335 | 0.00157 |
| `margin-L12-a0.25` | margin | 12 | 0.25 | 0.420 | 0.225 | 0.355 | 0.00157 |
| `margin-L16-a0.25` | margin | 16 | 0.25 | 0.420 | 0.240 | 0.340 | 0.00157 |
| `logit-L20-a0.25` | logit | 20 | 0.25 | 0.420 | 0.190 | 0.390 | 0.297 |
| `logit-L20-a0.5` | logit | 20 | 0.5 | 0.420 | 0.190 | 0.390 | 0.297 |
| `logit-L20-a1` | logit | 20 | 1 | 0.420 | 0.190 | 0.390 | 0.297 |
| `logit-L20-a2` | logit | 20 | 2 | 0.420 | 0.190 | 0.390 | 0.297 |
| `logit-L20-a4` | logit | 20 | 4 | 0.420 | 0.190 | 0.390 | 0.297 |
| `margin-L20-a0.25` | margin | 20 | 0.25 | 0.415 | 0.250 | 0.335 | 0.0027 |
| `margin-L24-a0.25` | margin | 24 | 0.25 | 0.410 | 0.235 | 0.355 | 0.00468 |
| `act-L12-a1` | act | 12 | 1 | 0.405 | 0.220 | 0.375 | 0.297 |
| `margin-L4-a0.25` | margin | 4 | 0.25 | 0.405 | 0.265 | 0.330 | 0.0196 |
| `act-L16-a1` | act | 16 | 1 | 0.400 | 0.265 | 0.335 | 0.33 |
| `margin_shuffled-L20-a1` | margin_shuffled | 20 | 1 | 0.395 | 0.265 | 0.340 | 0.353 |
| `margin_shuffled-L24-a2` | margin_shuffled | 24 | 2 | 0.395 | 0.245 | 0.360 | 0.384 |
| `margin_shuffled-L24-a4` | margin_shuffled | 24 | 4 | 0.395 | 0.245 | 0.360 | 0.475 |
| `margin_shuffled-L24-a0.5` | margin_shuffled | 24 | 0.5 | 0.390 | 0.250 | 0.360 | 0.102 |
| `margin_shuffled-L24-a1` | margin_shuffled | 24 | 1 | 0.390 | 0.255 | 0.355 | 0.317 |
| `margin_shuffled-L8-a0.25` | margin_shuffled | 8 | 0.25 | 0.380 | 0.280 | 0.340 | 0.48 |
| `margin_shuffled-L8-a1` | margin_shuffled | 8 | 1 | 0.380 | 0.295 | 0.325 | 0.67 |
| `margin_shuffled-L8-a4` | margin_shuffled | 8 | 4 | 0.380 | 0.315 | 0.305 | 0.803 |
| `margin_shuffled-L16-a0.5` | margin_shuffled | 16 | 0.5 | 0.380 | 0.265 | 0.355 | 0.527 |
| `margin_shuffled-L16-a1` | margin_shuffled | 16 | 1 | 0.380 | 0.270 | 0.350 | 0.617 |
| `margin_shuffled-L16-a2` | margin_shuffled | 16 | 2 | 0.380 | 0.280 | 0.340 | 0.732 |
| `margin_shuffled-L20-a2` | margin_shuffled | 20 | 2 | 0.380 | 0.290 | 0.330 | 0.796 |
| `margin_shuffled-L24-a0.25` | margin_shuffled | 24 | 0.25 | 0.380 | 0.265 | 0.355 | 0.317 |
| `margin_shuffled-L4-a0.25` | margin_shuffled | 4 | 0.25 | 0.375 | 0.290 | 0.335 | 0.705 |
| `margin_shuffled-L8-a0.5` | margin_shuffled | 8 | 0.5 | 0.375 | 0.285 | 0.340 | 0.739 |
| `margin_shuffled-L20-a0.5` | margin_shuffled | 20 | 0.5 | 0.375 | 0.280 | 0.345 | 0.782 |
| `margin_shuffled-L20-a4` | margin_shuffled | 20 | 4 | 0.375 | 0.330 | 0.295 | 0.908 |
| `none` | none | — | 0 | 0.370 | 0.280 | 0.350 | 1 |
| `act-L20-a1` | act | 20 | 1 | 0.370 | 0.240 | 0.390 | 1 |
| `margin_shuffled-L12-a0.25` | margin_shuffled | 12 | 0.25 | 0.370 | 0.285 | 0.345 | 1 |
| `margin_shuffled-L16-a0.25` | margin_shuffled | 16 | 0.25 | 0.370 | 0.275 | 0.355 | 1 |
| `margin_shuffled-L20-a0.25` | margin_shuffled | 20 | 0.25 | 0.370 | 0.280 | 0.350 | 1 |
| `margin_shuffled-L8-a2` | margin_shuffled | 8 | 2 | 0.365 | 0.330 | 0.305 | 0.866 |
| `margin_shuffled-L12-a2` | margin_shuffled | 12 | 2 | 0.365 | 0.295 | 0.340 | 0.876 |
| `margin_shuffled-L16-a4` | margin_shuffled | 16 | 4 | 0.365 | 0.300 | 0.335 | 0.889 |
| `margin_shuffled-L4-a0.5` | margin_shuffled | 4 | 0.5 | 0.360 | 0.290 | 0.350 | 0.564 |
| `margin_shuffled-L4-a4` | margin_shuffled | 4 | 4 | 0.360 | 0.330 | 0.310 | 0.827 |
| `margin_shuffled-L12-a0.5` | margin_shuffled | 12 | 0.5 | 0.360 | 0.290 | 0.350 | 0.414 |
| `margin_shuffled-L12-a1` | margin_shuffled | 12 | 1 | 0.360 | 0.285 | 0.355 | 0.637 |
| `margin_shuffled-L12-a4` | margin_shuffled | 12 | 4 | 0.360 | 0.315 | 0.325 | 0.785 |
| `act-L8-a1` | act | 8 | 1 | 0.345 | 0.315 | 0.340 | 0.446 |
| `margin_shuffled-L4-a1` | margin_shuffled | 4 | 1 | 0.340 | 0.300 | 0.360 | 0.221 |
| `margin_shuffled-L4-a2` | margin_shuffled | 4 | 2 | 0.335 | 0.330 | 0.335 | 0.354 |
| `logit-L4-a0.25` | logit | 4 | 0.25 | 0.270 | 0.420 | 0.310 | 0.033 |
| `logit-L4-a0.5` | logit | 4 | 0.5 | 0.270 | 0.420 | 0.310 | 0.033 |
| `logit-L4-a1` | logit | 4 | 1 | 0.270 | 0.420 | 0.310 | 0.033 |
| `logit-L4-a2` | logit | 4 | 2 | 0.270 | 0.420 | 0.310 | 0.033 |
| `logit-L4-a4` | logit | 4 | 4 | 0.270 | 0.420 | 0.310 | 0.033 |
| `logit-L8-a0.25` | logit | 8 | 0.25 | 0.250 | 0.505 | 0.245 | 0.00804 |
| `logit-L8-a0.5` | logit | 8 | 0.5 | 0.250 | 0.505 | 0.245 | 0.00804 |
| `logit-L8-a1` | logit | 8 | 1 | 0.250 | 0.505 | 0.245 | 0.00804 |
| `logit-L8-a2` | logit | 8 | 2 | 0.250 | 0.505 | 0.245 | 0.00804 |
| `logit-L8-a4` | logit | 8 | 4 | 0.250 | 0.505 | 0.245 | 0.00804 |
| `logit-L12-a0.25` | logit | 12 | 0.25 | 0.250 | 0.405 | 0.345 | 0.0175 |
| `logit-L12-a0.5` | logit | 12 | 0.5 | 0.250 | 0.405 | 0.345 | 0.0175 |
| `logit-L12-a1` | logit | 12 | 1 | 0.250 | 0.405 | 0.345 | 0.0175 |
| `logit-L12-a2` | logit | 12 | 2 | 0.250 | 0.405 | 0.345 | 0.0175 |
| `logit-L12-a4` | logit | 12 | 4 | 0.250 | 0.405 | 0.345 | 0.0175 |
| `logit-L16-a0.25` | logit | 16 | 0.25 | 0.250 | 0.440 | 0.310 | 0.0153 |
| `logit-L16-a0.5` | logit | 16 | 0.5 | 0.250 | 0.440 | 0.310 | 0.0153 |
| `logit-L16-a1` | logit | 16 | 1 | 0.250 | 0.440 | 0.310 | 0.0153 |
| `logit-L16-a2` | logit | 16 | 2 | 0.250 | 0.440 | 0.310 | 0.0153 |
| `logit-L16-a4` | logit | 16 | 4 | 0.250 | 0.440 | 0.310 | 0.0153 |
