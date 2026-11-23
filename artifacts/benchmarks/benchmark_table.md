# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fly Connectome PPO (fly_ppo_new_env) | -35.446 +/- 1.833 | 9.32 | 25.82 +/- 1.12 | 352.4 +/- 18.4 | 54.7 | 0.00 | 0.265 |
| normal | Shuffled Connectome PPO (fly_shuffled_ppo_new_env) | -35.630 +/- 1.671 | 9.37 | 25.99 +/- 1.05 | 353.2 +/- 18.0 | 55.0 | 0.00 | 0.289 |
| normal | PPO-GRU (gru_ppo_new_env) | -34.923 +/- 1.922 | 9.32 | 26.07 +/- 1.43 | 348.0 +/- 18.0 | 53.5 | 0.00 | 0.176 |
| normal | PPO-MLP (mlp_ppo_new_env) | -34.605 +/- 1.347 | 9.21 | 25.75 +/- 0.97 | 349.1 +/- 21.0 | 52.9 | 0.00 | 0.128 |
| normal | Fixed Timer | -40.626 +/- 2.134 | 11.01 | 30.69 +/- 1.72 | 348.8 +/- 22.4 | 45.0 | 0.00 | 0.002 |
| asymmetric | Fly Connectome PPO (fly_ppo_new_env) | -48.370 +/- 2.332 | 13.75 | 23.57 +/- 0.92 | 571.5 +/- 35.8 | 55.0 | 0.00 | 0.282 |
| asymmetric | Shuffled Connectome PPO (fly_shuffled_ppo_new_env) | -47.624 +/- 2.724 | 13.88 | 23.83 +/- 0.89 | 570.1 +/- 32.7 | 55.0 | 0.00 | 0.264 |
| asymmetric | PPO-GRU (gru_ppo_new_env) | -47.893 +/- 1.775 | 13.34 | 22.63 +/- 0.73 | 577.1 +/- 24.7 | 53.6 | 0.00 | 0.132 |
| asymmetric | PPO-MLP (mlp_ppo_new_env) | -47.616 +/- 2.887 | 13.47 | 23.27 +/- 0.78 | 565.5 +/- 30.6 | 53.5 | 0.00 | 0.089 |
| asymmetric | Fixed Timer | -61.537 +/- 3.308 | 19.12 | 32.92 +/- 0.77 | 569.5 +/- 27.7 | 45.0 | 0.00 | 0.004 |
| rush_hour | Fly Connectome PPO (fly_ppo_new_env) | -79.574 +/- 2.329 | 25.65 | 27.96 +/- 0.87 | 886.7 +/- 24.7 | 54.8 | 0.00 | 0.393 |
| rush_hour | Shuffled Connectome PPO (fly_shuffled_ppo_new_env) | -79.431 +/- 3.493 | 25.60 | 28.05 +/- 0.96 | 879.7 +/- 27.7 | 54.3 | 0.00 | 0.262 |
| rush_hour | PPO-GRU (gru_ppo_new_env) | -80.937 +/- 4.166 | 26.01 | 28.09 +/- 0.90 | 894.5 +/- 35.6 | 53.9 | 0.00 | 0.098 |
| rush_hour | PPO-MLP (mlp_ppo_new_env) | -76.871 +/- 1.919 | 24.71 | 27.18 +/- 0.79 | 879.3 +/- 26.8 | 54.0 | 0.00 | 0.079 |
| rush_hour | Fixed Timer | -104.742 +/- 8.067 | 35.43 | 38.61 +/- 2.18 | 855.9 +/- 25.3 | 45.0 | 0.00 | 0.005 |
| stress_test | Fly Connectome PPO (fly_ppo_new_env) | -399.038 +/- 17.601 | 144.89 | 88.38 +/- 4.17 | 1448.8 +/- 13.6 | 54.9 | 5.00 | 0.266 |
| stress_test | Shuffled Connectome PPO (fly_shuffled_ppo_new_env) | -395.579 +/- 18.526 | 140.52 | 84.97 +/- 3.40 | 1460.1 +/- 13.1 | 54.9 | 14.00 | 0.264 |
| stress_test | PPO-GRU (gru_ppo_new_env) | -361.209 +/- 22.370 | 129.12 | 78.82 +/- 5.67 | 1468.7 +/- 21.9 | 52.6 | 8.60 | 0.099 |
| stress_test | PPO-MLP (mlp_ppo_new_env) | -381.887 +/- 22.457 | 131.83 | 78.69 +/- 3.92 | 1487.1 +/- 14.5 | 53.8 | 25.20 | 0.085 |
| stress_test | Fixed Timer | -365.067 +/- 17.968 | 126.19 | 76.47 +/- 3.09 | 1482.8 +/- 31.6 | 45.0 | 20.90 | 0.002 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -79.574 | 27.96 | 886.7 | 54.8 | 0.00 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -78.480 | 27.44 | 885.2 | 54.4 | 0.00 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -78.988 | 27.93 | 881.6 | 54.7 | 0.00 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -79.107 | 28.01 | 879.1 | 54.7 | 0.00 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -80.407 | 28.18 | 886.2 | 54.6 | 0.00 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -78.734 | 27.55 | 887.3 | 54.8 | 0.00 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -79.440 | 27.94 | 885.2 | 54.6 | 0.00 |
