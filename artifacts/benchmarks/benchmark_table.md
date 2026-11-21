# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Shuffled Connectome PPO | -11.101 +/- 0.708 | 5.28 | 14.42 +/- 0.68 | 214.4 +/- 14.0 | 72.6 | 0.00 | 0.175 |
| normal | Fly Connectome PPO (Fly__1) | -10.160 +/- 1.044 | 4.77 | 13.11 +/- 0.88 | 212.1 +/- 16.9 | 72.1 | 0.00 | 0.163 |
| normal | PPO-MLP (MLP__1) | -10.318 +/- 0.975 | 4.78 | 13.18 +/- 0.72 | 212.0 +/- 17.0 | 69.6 | 0.00 | 0.062 |
| normal | Fixed Timer | -17.908 +/- 1.317 | 8.22 | 22.47 +/- 0.79 | 209.9 +/- 15.5 | 37.0 | 0.10 | 0.001 |
| asymmetric | Shuffled Connectome PPO | -16.613 +/- 1.721 | 8.39 | 14.58 +/- 0.79 | 334.0 +/- 22.5 | 67.5 | 0.00 | 0.309 |
| asymmetric | Fly Connectome PPO (Fly__1) | -17.015 +/- 1.854 | 8.14 | 14.21 +/- 0.91 | 334.3 +/- 21.7 | 68.2 | 0.40 | 0.364 |
| asymmetric | PPO-MLP (MLP__1) | -16.185 +/- 2.312 | 8.02 | 13.82 +/- 1.35 | 337.1 +/- 26.8 | 67.7 | 0.10 | 0.098 |
| asymmetric | Fixed Timer | -30.564 +/- 3.966 | 15.56 | 27.09 +/- 2.47 | 329.3 +/- 14.8 | 37.0 | 1.70 | 0.003 |
| rush_hour | Shuffled Connectome PPO | -18.296 +/- 1.424 | 9.29 | 15.84 +/- 0.82 | 341.9 +/- 20.3 | 68.7 | 0.20 | 0.305 |
| rush_hour | Fly Connectome PPO (Fly__1) | -17.486 +/- 1.798 | 8.73 | 15.06 +/- 0.76 | 336.4 +/- 24.3 | 70.5 | 0.40 | 0.293 |
| rush_hour | PPO-MLP (MLP__1) | -17.751 +/- 2.211 | 8.84 | 15.08 +/- 1.11 | 339.6 +/- 24.2 | 69.8 | 0.10 | 0.086 |
| rush_hour | Fixed Timer | -30.049 +/- 4.126 | 15.39 | 26.26 +/- 2.51 | 323.5 +/- 18.8 | 37.0 | 1.30 | 0.002 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -17.486 | 15.06 | 336.4 | 70.5 | 0.40 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -17.754 | 15.13 | 337.5 | 68.5 | 0.20 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -17.931 | 15.57 | 334.5 | 69.5 | 0.20 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -17.133 | 14.90 | 328.4 | 70.9 | 0.30 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -17.184 | 14.97 | 333.2 | 70.5 | 0.20 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -18.078 | 15.27 | 342.0 | 69.5 | 0.40 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -17.306 | 14.91 | 338.2 | 69.7 | 0.20 |
