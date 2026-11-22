# Benchmark Evaluation Results

## Scenario Comparison

| Scenario | Agent | Mean Reward | Mean Queue | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions | Inference (ms/step) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | Fly Connectome PPO (Fly__1) | -10.150 +/- 1.041 | 4.77 | 13.08 +/- 0.84 | 212.5 +/- 16.5 | 72.2 | 0.00 | 0.390 |
| normal | Fly Connectome PPO (fly_ppo_withstress) | -11.048 +/- 0.953 | 5.06 | 13.82 +/- 0.70 | 214.8 +/- 19.5 | 66.9 | 0.00 | 0.568 |
| asymmetric | Fly Connectome PPO (Fly__1) | -16.751 +/- 1.823 | 7.91 | 13.93 +/- 0.82 | 333.2 +/- 20.9 | 68.3 | 0.00 | 0.514 |
| asymmetric | Fly Connectome PPO (fly_ppo_withstress) | -15.926 +/- 1.568 | 7.81 | 13.54 +/- 0.68 | 338.2 +/- 22.5 | 67.9 | 0.00 | 0.427 |
| rush_hour | Fly Connectome PPO (Fly__1) | -17.751 +/- 2.692 | 8.87 | 15.18 +/- 1.39 | 339.1 +/- 23.6 | 69.9 | 0.00 | 0.500 |
| rush_hour | Fly Connectome PPO (fly_ppo_withstress) | -17.514 +/- 2.556 | 8.57 | 14.94 +/- 1.54 | 332.5 +/- 15.5 | 68.2 | 0.00 | 0.802 |
| stress_test | Fly Connectome PPO (Fly__1) | -313.567 +/- 6.742 | 182.61 | 133.45 +/- 3.26 | 551.8 +/- 11.7 | 68.1 | 46.90 | 0.869 |
| stress_test | Fly Connectome PPO (fly_ppo_withstress) | -301.109 +/- 5.376 | 172.15 | 120.59 +/- 2.10 | 594.7 +/- 10.3 | 69.4 | 45.00 | 0.432 |

## Pharmacology and Lesioning Results (rush_hour)

| Condition | g_exc | g_inh | Active Neurons | Mean Reward | Mean Wait (s) | Mean Throughput | Mean Switches | Mean Collisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (g_exc=1.0, g_inh=1.0) | 1.00 | 1.00 | 2500 | -17.751 | 15.18 | 339.1 | 69.9 | 0.00 |
| High Excitation (g_exc=1.8, g_inh=1.0) | 1.80 | 1.00 | 2500 | -18.257 | 15.54 | 337.1 | 68.4 | 0.00 |
| Low Inhibition (g_exc=1.0, g_inh=0.2) | 1.00 | 0.20 | 2500 | -18.221 | 15.46 | 340.7 | 69.4 | 0.00 |
| High Inhibition (g_exc=1.0, g_inh=2.0) | 1.00 | 2.00 | 2500 | -18.305 | 15.80 | 332.5 | 70.6 | 0.00 |
| CX Lesion 25% | 1.00 | 1.00 | 2050 | -18.668 | 16.00 | 333.6 | 70.5 | 0.00 |
| MB Lesion 25% | 1.00 | 1.00 | 2325 | -17.890 | 15.17 | 339.5 | 69.8 | 0.00 |
| Random Lesion 25% | 1.00 | 1.00 | 1875 | -18.175 | 15.39 | 341.9 | 69.6 | 0.00 |
