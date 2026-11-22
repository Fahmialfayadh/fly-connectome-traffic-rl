# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fly Connectome PPO (fly_ppo_new_env) | -35.432 +/- 1.534 | 9.32 | 25.83 +/- 1.23 | 351.8 +/- 22.7 | 54.8 | 0.00 | 0.177 |
| normal | Fly Connectome PPO (fly_ppo_withstress) | -35.199 +/- 1.824 | 9.27 | 25.75 +/- 1.12 | 352.5 +/- 22.6 | 55.0 | 0.00 | 0.199 |
| normal | Fixed Timer | -40.626 +/- 2.134 | 11.01 | 30.69 +/- 1.72 | 348.8 +/- 22.4 | 45.0 | 0.00 | 0.004 |
| asymmetric | Fly Connectome PPO (fly_ppo_new_env) | -48.835 +/- 2.570 | 13.95 | 23.68 +/- 0.92 | 574.9 +/- 29.8 | 55.0 | 0.00 | 0.199 |
| asymmetric | Fly Connectome PPO (fly_ppo_withstress) | -47.862 +/- 3.112 | 13.51 | 23.14 +/- 0.89 | 571.4 +/- 31.6 | 55.0 | 0.00 | 0.454 |
| asymmetric | Fixed Timer | -61.537 +/- 3.308 | 19.12 | 32.92 +/- 0.77 | 569.5 +/- 27.7 | 45.0 | 0.00 | 0.004 |
| rush_hour | Fly Connectome PPO (fly_ppo_new_env) | -77.955 +/- 3.780 | 25.00 | 27.40 +/- 0.85 | 881.6 +/- 28.7 | 54.7 | 0.00 | 0.585 |
| rush_hour | Fly Connectome PPO (fly_ppo_withstress) | -79.080 +/- 2.880 | 25.30 | 27.74 +/- 0.67 | 882.2 +/- 34.6 | 54.9 | 0.00 | 0.416 |
| rush_hour | Fixed Timer | -104.742 +/- 8.067 | 35.43 | 38.61 +/- 2.18 | 855.9 +/- 25.3 | 45.0 | 0.00 | 0.003 |
| stress_test | Fly Connectome PPO (fly_ppo_new_env) | -395.780 +/- 22.556 | 143.67 | 87.39 +/- 4.77 | 1451.7 +/- 11.0 | 55.0 | 5.70 | 0.399 |
| stress_test | Fly Connectome PPO (fly_ppo_withstress) | -386.363 +/- 21.059 | 132.14 | 80.47 +/- 3.84 | 1455.0 +/- 16.3 | 55.0 | 26.50 | 0.489 |
| stress_test | Fixed Timer | -365.067 +/- 17.968 | 126.19 | 76.47 +/- 3.09 | 1482.8 +/- 31.6 | 45.0 | 20.90 | 0.007 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -77.955 | 27.40 | 881.6 | 54.7 | 0.00 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -78.566 | 27.81 | 878.4 | 54.3 | 0.00 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -78.450 | 27.73 | 880.5 | 54.6 | 0.00 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -79.248 | 27.96 | 880.8 | 54.8 | 0.00 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -77.205 | 27.39 | 876.9 | 54.8 | 0.00 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -79.101 | 27.65 | 887.3 | 54.6 | 0.00 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -79.659 | 28.22 | 878.9 | 54.6 | 0.00 |
