# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fixed Timer | -17.908 +/- 1.317 | 8.22 | 22.47 +/- 0.79 | 209.9 +/- 15.5 | 37.0 | 0.10 | 0.002 |
| normal | PPO-MLP | -10.582 +/- 0.785 | 4.97 | 13.79 +/- 0.67 | 211.0 +/- 15.4 | 70.7 | 0.00 | 0.097 |
| normal | PPO-GRU | -12.341 +/- 0.899 | 5.78 | 15.83 +/- 0.73 | 213.3 +/- 16.7 | 74.2 | 0.00 | 0.184 |
| normal | Fly Connectome PPO | -11.427 +/- 0.588 | 5.27 | 14.48 +/- 0.59 | 213.6 +/- 12.2 | 67.6 | 0.00 | 0.542 |
| asymmetric | Fixed Timer | -30.564 +/- 3.966 | 15.56 | 27.09 +/- 2.47 | 329.3 +/- 14.8 | 37.0 | 1.70 | 0.005 |
| asymmetric | PPO-MLP | -16.787 +/- 2.009 | 8.35 | 14.65 +/- 1.14 | 332.4 +/- 18.7 | 71.9 | 0.10 | 0.167 |
| asymmetric | PPO-GRU | -21.074 +/- 6.518 | 9.67 | 16.78 +/- 4.55 | 331.4 +/- 16.9 | 68.0 | 2.00 | 0.221 |
| asymmetric | Fly Connectome PPO | -16.585 +/- 1.727 | 8.44 | 14.70 +/- 1.39 | 335.4 +/- 23.7 | 61.9 | 0.10 | 0.599 |
| rush_hour | Fixed Timer | -30.049 +/- 4.126 | 15.39 | 26.26 +/- 2.51 | 323.5 +/- 18.8 | 37.0 | 1.30 | 0.005 |
| rush_hour | PPO-MLP | -18.594 +/- 2.753 | 9.40 | 16.11 +/- 1.92 | 338.7 +/- 22.7 | 71.4 | 0.20 | 0.121 |
| rush_hour | PPO-GRU | -20.442 +/- 2.823 | 9.87 | 17.31 +/- 1.67 | 327.3 +/- 14.3 | 71.2 | 0.70 | 0.135 |
| rush_hour | Fly Connectome PPO | -18.483 +/- 1.835 | 9.19 | 16.26 +/- 0.83 | 328.5 +/- 20.3 | 66.8 | 0.10 | 0.608 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -18.483 | 16.26 | 328.5 | 66.8 | 0.10 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -18.282 | 15.88 | 333.8 | 64.4 | 0.30 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -18.333 | 16.03 | 334.8 | 67.2 | 0.30 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -18.822 | 16.54 | 331.8 | 67.9 | 0.30 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -18.915 | 16.38 | 339.1 | 66.8 | 0.30 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -18.753 | 16.47 | 331.7 | 67.4 | 0.20 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -18.021 | 15.85 | 330.4 | 66.5 | 0.20 |
