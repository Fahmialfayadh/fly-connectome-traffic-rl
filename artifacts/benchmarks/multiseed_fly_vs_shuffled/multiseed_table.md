# Multi-Seed Fly vs Shuffled Evaluation Summary

## 1. Aggregated Performance Across Training / Rewiring Seeds

| Scenario | Architecture | Seeds (N) | Mean Reward (Across Seeds) | Mean Wait (s) | Mean Throughput | Mean Collisions (Across Seeds) |
|---|---|---:|---:|---:|---:|---:|
| normal | Fly Connectome PPO | 3 | -35.553 +/- 0.233 | 25.87 +/- 0.31 | 352.7 | 0.00 +/- 0.00 |
| normal | Shuffled Connectome PPO | 3 | -35.559 +/- 0.135 | 26.15 +/- 0.26 | 352.6 | 0.00 +/- 0.00 |
| asymmetric | Fly Connectome PPO | 3 | -48.880 +/- 0.224 | 23.11 +/- 0.41 | 572.8 | 0.00 +/- 0.00 |
| asymmetric | Shuffled Connectome PPO | 3 | -48.100 +/- 0.204 | 23.68 +/- 0.17 | 571.5 | 0.00 +/- 0.00 |
| rush_hour | Fly Connectome PPO | 3 | -78.960 +/- 0.456 | 27.59 +/- 0.31 | 881.4 | 0.01 +/- 0.01 |
| rush_hour | Shuffled Connectome PPO | 3 | -79.342 +/- 0.242 | 28.03 +/- 0.15 | 880.8 | 0.00 +/- 0.00 |
| stress_test | Fly Connectome PPO | 3 | -404.670 +/- 3.812 | 88.56 +/- 0.18 | 1460.1 | 7.05 +/- 1.94 |
| stress_test | Shuffled Connectome PPO | 3 | -401.897 +/- 6.130 | 87.60 +/- 1.28 | 1459.1 | 8.51 +/- 3.25 |

## 2. Paired `stress_test` Collision Isolation by Seed (50 Traffic Seeds Each)

| Training / Rewiring Seed | Fly Collisions (Mean +/- Std, Median) | Shuffled Collisions (Mean +/- Std, Median) | Fly < Shuffled (Win / Tie / Loss) | Fly Reward | Shuffled Reward |
|---:|---:|---:|---:|---:|---:|
| 42 | 4.34 +/- 2.77 (med 4.0) | 12.88 +/- 5.23 (med 13.0) | 46/50 (92%) [46W / 1T / 3L] | -399.310 | -402.899 |
| 101 | 8.02 +/- 3.79 (med 8.0) | 5.10 +/- 2.59 (med 5.0) | 8/50 (16%) [8W / 3T / 39L] | -407.856 | -393.939 |
| 202 | 8.80 +/- 4.23 (med 8.0) | 7.56 +/- 3.20 (med 7.0) | 21/50 (42%) [21W / 1T / 28L] | -406.843 | -408.854 |

## 3. Per-Controller Scenario Breakdown

# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fly Connectome PPO (seed 42) | -35.875 +/- 1.600 | 9.50 | 26.31 +/- 1.05 | 352.5 +/- 17.1 | 54.4 | 0.00 | 0.294 |
| normal | Shuffled Connectome PPO (seed 42) | -35.748 +/- 2.186 | 9.42 | 26.03 +/- 1.25 | 353.0 +/- 19.8 | 55.0 | 0.00 | 0.293 |
| normal | Fly Connectome PPO (seed 101) | -35.455 +/- 2.065 | 9.24 | 25.64 +/- 0.97 | 351.0 +/- 19.5 | 54.1 | 0.00 | 0.300 |
| normal | Shuffled Connectome PPO (seed 101) | -35.447 +/- 2.151 | 9.57 | 26.51 +/- 1.11 | 352.2 +/- 20.1 | 55.0 | 0.00 | 0.289 |
| normal | Fly Connectome PPO (seed 202) | -35.330 +/- 1.750 | 9.32 | 25.66 +/- 1.05 | 354.6 +/- 18.6 | 54.6 | 0.00 | 0.303 |
| normal | Shuffled Connectome PPO (seed 202) | -35.482 +/- 2.015 | 9.37 | 25.92 +/- 1.14 | 352.5 +/- 17.9 | 55.0 | 0.00 | 0.295 |
| normal | Fixed Timer | -41.354 +/- 2.308 | 11.25 | 31.25 +/- 1.28 | 349.2 +/- 20.0 | 45.0 | 0.00 | 0.002 |
| asymmetric | Fly Connectome PPO (seed 42) | -48.896 +/- 2.647 | 13.84 | 23.67 +/- 0.89 | 572.4 +/- 26.9 | 55.0 | 0.00 | 0.349 |
| asymmetric | Shuffled Connectome PPO (seed 42) | -47.840 +/- 2.606 | 13.89 | 23.76 +/- 0.87 | 572.2 +/- 25.9 | 55.0 | 0.00 | 0.350 |
| asymmetric | Fly Connectome PPO (seed 101) | -48.599 +/- 2.806 | 13.26 | 22.70 +/- 0.92 | 570.7 +/- 23.1 | 54.2 | 0.00 | 0.336 |
| asymmetric | Shuffled Connectome PPO (seed 101) | -48.124 +/- 2.959 | 13.91 | 23.83 +/- 0.86 | 570.6 +/- 26.5 | 54.9 | 0.00 | 0.522 |
| asymmetric | Fly Connectome PPO (seed 202) | -49.147 +/- 2.855 | 13.49 | 22.96 +/- 0.94 | 575.3 +/- 25.4 | 54.7 | 0.00 | 0.398 |
| asymmetric | Shuffled Connectome PPO (seed 202) | -48.337 +/- 2.714 | 13.72 | 23.44 +/- 0.92 | 571.8 +/- 27.3 | 55.0 | 0.00 | 0.372 |
| asymmetric | Fixed Timer | -61.765 +/- 3.094 | 19.23 | 33.04 +/- 1.00 | 570.2 +/- 25.0 | 45.0 | 0.00 | 0.002 |
| rush_hour | Fly Connectome PPO (seed 42) | -79.584 +/- 3.666 | 25.64 | 28.00 +/- 0.99 | 883.9 +/- 25.8 | 54.6 | 0.00 | 0.289 |
| rush_hour | Shuffled Connectome PPO (seed 42) | -79.682 +/- 3.475 | 25.65 | 28.08 +/- 0.92 | 882.2 +/- 28.8 | 54.3 | 0.00 | 0.279 |
| rush_hour | Fly Connectome PPO (seed 101) | -78.785 +/- 3.772 | 24.91 | 27.27 +/- 0.99 | 882.1 +/- 29.9 | 54.7 | 0.02 | 0.293 |
| rush_hour | Shuffled Connectome PPO (seed 101) | -79.198 +/- 3.366 | 25.64 | 28.17 +/- 0.84 | 877.9 +/- 30.0 | 54.2 | 0.00 | 0.283 |
| rush_hour | Fly Connectome PPO (seed 202) | -78.510 +/- 3.562 | 24.99 | 27.49 +/- 0.98 | 878.1 +/- 25.3 | 54.7 | 0.00 | 0.294 |
| rush_hour | Shuffled Connectome PPO (seed 202) | -79.145 +/- 4.354 | 25.43 | 27.82 +/- 1.20 | 882.2 +/- 26.0 | 54.7 | 0.00 | 0.275 |
| rush_hour | Fixed Timer | -104.975 +/- 6.900 | 35.55 | 38.86 +/- 1.95 | 856.2 +/- 24.6 | 45.0 | 0.00 | 0.002 |
| stress_test | Fly Connectome PPO (seed 42) | -399.310 +/- 22.310 | 145.34 | 88.43 +/- 4.80 | 1452.1 +/- 14.5 | 55.0 | 4.34 | 0.295 |
| stress_test | Shuffled Connectome PPO (seed 42) | -402.899 +/- 21.260 | 143.22 | 86.22 +/- 3.76 | 1465.1 +/- 12.8 | 55.0 | 12.88 | 0.285 |
| stress_test | Fly Connectome PPO (seed 101) | -407.856 +/- 19.531 | 147.31 | 88.81 +/- 3.72 | 1465.4 +/- 11.9 | 55.0 | 8.02 | 0.275 |
| stress_test | Shuffled Connectome PPO (seed 101) | -393.939 +/- 22.540 | 142.87 | 87.27 +/- 5.06 | 1451.5 +/- 11.2 | 54.8 | 5.10 | 0.360 |
| stress_test | Fly Connectome PPO (seed 202) | -406.843 +/- 21.247 | 146.72 | 88.44 +/- 4.19 | 1462.6 +/- 13.1 | 55.0 | 8.80 | 0.419 |
| stress_test | Shuffled Connectome PPO (seed 202) | -408.854 +/- 20.700 | 147.84 | 89.30 +/- 4.29 | 1460.5 +/- 12.6 | 55.0 | 7.56 | 0.414 |
| stress_test | Fixed Timer | -365.494 +/- 14.610 | 126.59 | 76.51 +/- 2.58 | 1487.1 +/- 31.6 | 45.0 | 19.62 | 0.002 |
