# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fly Connectome PPO (fly_ppofrom_ipynb) | -36.047 +/- 1.646 | 9.63 | 26.58 +/- 0.93 | 353.6 +/- 21.0 | 54.9 | 0.00 | 0.318 |
| normal | Shuffled Connectome PPO (fly_shuffled_ppofrom_ipynb) | -37.284 +/- 2.201 | 9.74 | 27.38 +/- 1.04 | 348.2 +/- 23.3 | 55.0 | 0.00 | 0.537 |
| normal | PPO-MLP (mlp_ppofrom_ipynb) | -37.825 +/- 2.123 | 9.88 | 27.06 +/- 0.90 | 356.2 +/- 23.5 | 51.5 | 0.00 | 0.240 |
| normal | PPO-GRU (gru_ppofrom_ipynb) | -338.292 +/- 35.812 | 82.98 | 261.71 +/- 32.09 | 136.2 +/- 15.3 | 4.1 | 39.80 | 0.312 |
| normal | Fixed Timer | -40.626 +/- 2.134 | 11.01 | 30.69 +/- 1.72 | 348.8 +/- 22.4 | 45.0 | 0.00 | 0.003 |
| asymmetric | Fly Connectome PPO (fly_ppofrom_ipynb) | -47.527 +/- 3.282 | 13.85 | 23.81 +/- 0.97 | 566.9 +/- 32.9 | 55.0 | 0.00 | 0.426 |
| asymmetric | Shuffled Connectome PPO (fly_shuffled_ppofrom_ipynb) | -50.742 +/- 2.917 | 14.83 | 25.82 +/- 0.72 | 557.3 +/- 31.5 | 54.5 | 0.00 | 0.419 |
| asymmetric | PPO-MLP (mlp_ppofrom_ipynb) | -54.384 +/- 3.823 | 14.49 | 24.84 +/- 0.70 | 567.5 +/- 22.0 | 46.3 | 0.10 | 0.120 |
| asymmetric | PPO-GRU (gru_ppofrom_ipynb) | -334.140 +/- 12.386 | 75.13 | 140.02 +/- 6.92 | 338.7 +/- 18.4 | 0.0 | 98.70 | 0.169 |
| asymmetric | Fixed Timer | -61.537 +/- 3.308 | 19.12 | 32.92 +/- 0.77 | 569.5 +/- 27.7 | 45.0 | 0.00 | 0.004 |
| rush_hour | Fly Connectome PPO (fly_ppofrom_ipynb) | -83.159 +/- 2.995 | 27.31 | 29.67 +/- 0.75 | 885.7 +/- 23.2 | 55.0 | 0.00 | 0.499 |
| rush_hour | Shuffled Connectome PPO (fly_shuffled_ppofrom_ipynb) | -83.560 +/- 4.734 | 27.09 | 29.34 +/- 0.73 | 890.0 +/- 36.4 | 53.8 | 0.00 | 0.441 |
| rush_hour | PPO-MLP (mlp_ppofrom_ipynb) | -81.916 +/- 3.228 | 25.91 | 28.25 +/- 1.12 | 886.1 +/- 19.8 | 53.9 | 0.00 | 0.129 |
| rush_hour | PPO-GRU (gru_ppofrom_ipynb) | -455.784 +/- 14.077 | 123.11 | 199.52 +/- 10.24 | 334.1 +/- 19.2 | 2.4 | 104.00 | 0.204 |
| rush_hour | Fixed Timer | -104.742 +/- 8.067 | 35.43 | 38.61 +/- 2.18 | 855.9 +/- 25.3 | 45.0 | 0.00 | 0.004 |
| stress_test | Fly Connectome PPO (fly_ppofrom_ipynb) | -386.369 +/- 14.918 | 137.59 | 85.73 +/- 3.00 | 1424.8 +/- 10.3 | 55.0 | 11.60 | 0.353 |
| stress_test | Shuffled Connectome PPO (fly_shuffled_ppofrom_ipynb) | -385.007 +/- 13.307 | 139.67 | 85.43 +/- 2.87 | 1451.8 +/- 11.8 | 54.3 | 5.20 | 0.446 |
| stress_test | PPO-MLP (mlp_ppofrom_ipynb) | -489.276 +/- 21.849 | 148.69 | 119.05 +/- 7.82 | 988.9 +/- 66.5 | 32.4 | 95.30 | 0.113 |
| stress_test | PPO-GRU (gru_ppofrom_ipynb) | -599.206 +/- 7.321 | 167.09 | 199.35 +/- 4.82 | 498.8 +/- 13.7 | 0.0 | 160.50 | 0.136 |
| stress_test | Fixed Timer | -365.067 +/- 17.968 | 126.19 | 76.47 +/- 3.09 | 1482.8 +/- 31.6 | 45.0 | 20.90 | 0.002 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -83.159 | 29.67 | 885.7 | 55.0 | 0.00 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -82.836 | 29.44 | 883.6 | 55.0 | 0.00 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -83.992 | 29.74 | 893.2 | 55.0 | 0.00 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -83.170 | 29.69 | 885.1 | 55.0 | 0.00 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -82.766 | 29.50 | 884.3 | 55.0 | 0.00 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -83.276 | 29.72 | 878.8 | 55.0 | 0.00 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -83.479 | 29.77 | 881.7 | 55.0 | 0.00 |
