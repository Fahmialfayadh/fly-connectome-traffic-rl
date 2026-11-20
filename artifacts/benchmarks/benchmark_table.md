# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fixed Timer | -17.711 +/- 1.420 | 8.14 | 22.33 +/- 0.97 | 209.8 +/- 16.8 | 37.0 | 0.00 | 0.003 |
| normal | PPO-MLP | -10.582 +/- 0.785 | 4.97 | 13.79 +/- 0.67 | 211.0 +/- 15.4 | 70.7 | 0.00 | 0.273 |
| normal | PPO-GRU | -12.285 +/- 0.668 | 5.76 | 15.82 +/- 0.47 | 212.3 +/- 14.0 | 74.1 | 0.00 | 0.224 |
| normal | Fly Connectome PPO | -11.522 +/- 0.588 | 5.32 | 14.60 +/- 0.48 | 214.0 +/- 12.7 | 67.5 | 0.00 | 0.387 |
| asymmetric | Fixed Timer | -28.849 +/- 2.864 | 14.71 | 25.73 +/- 1.67 | 329.5 +/- 18.0 | 37.0 | 1.70 | 0.003 |
| asymmetric | PPO-MLP | -17.497 +/- 2.664 | 8.73 | 14.92 +/- 1.49 | 340.8 +/- 22.2 | 71.9 | 0.20 | 0.102 |
| asymmetric | PPO-GRU | -19.795 +/- 2.626 | 9.17 | 15.91 +/- 1.66 | 331.5 +/- 19.2 | 68.7 | 1.50 | 0.119 |
| asymmetric | Fly Connectome PPO | -16.432 +/- 1.565 | 8.35 | 14.60 +/- 1.23 | 333.9 +/- 22.3 | 62.2 | 0.00 | 0.369 |
| rush_hour | Fixed Timer | -29.093 +/- 2.731 | 14.90 | 25.21 +/- 1.52 | 327.9 +/- 15.5 | 37.0 | 1.60 | 0.003 |
| rush_hour | PPO-MLP | -18.440 +/- 2.028 | 9.29 | 16.02 +/- 1.58 | 337.1 +/- 21.2 | 71.5 | 0.50 | 0.093 |
| rush_hour | PPO-GRU | -20.243 +/- 2.523 | 9.86 | 17.20 +/- 1.33 | 329.9 +/- 18.6 | 70.6 | 0.80 | 0.126 |
| rush_hour | Fly Connectome PPO | -18.720 +/- 2.523 | 9.36 | 16.20 +/- 1.16 | 335.3 +/- 26.3 | 66.6 | 0.30 | 0.414 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -18.720 | 16.20 | 335.3 | 66.6 | 0.30 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -18.032 | 15.65 | 333.1 | 64.7 | 0.60 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -18.363 | 16.25 | 330.5 | 67.2 | 0.10 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -18.968 | 16.52 | 336.0 | 67.5 | 0.40 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -18.993 | 16.34 | 342.0 | 67.0 | 0.50 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -18.284 | 15.94 | 332.5 | 66.9 | 0.20 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -18.287 | 15.88 | 331.6 | 66.8 | 0.50 |
