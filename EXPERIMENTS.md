# Experiment Tracking

This document tracks all experiments and their results. Update after each training run.

## Experiment Log

### Experiment 1: DDPG Baseline
- **Date:**
- **Algorithm:** DDPG
- **Config:** `configs/ddpg_config.yaml`
- **Hyperparameters:**
  - Learning rate (actor): 1e-4
  - Learning rate (critic): 1e-3
  - Batch size: 128
  - Buffer size: 500k
  - Noise: OU (θ=0.15, σ=0.2)
- **Training:**
  - Episodes:
  - Total steps:
  - Training time:
- **Results:**
  - Average reward (last 100 ep):
  - Best reward:
  - Success rate:
- **Observations:**
  -
- **Model saved:** `models/exp1_ddpg_baseline.pth`

---

### Experiment 2: SAC Baseline
- **Date:**
- **Algorithm:** SAC
- **Config:** `configs/sac_config.yaml`
- **Hyperparameters:**
  - Learning rate: 3e-4
  - Batch size: 256
  - Buffer size: 1M
  - Auto alpha: True
- **Training:**
  - Episodes:
  - Total steps:
  - Training time:
- **Results:**
  - Average reward (last 100 ep):
  - Best reward:
  - Success rate:
- **Observations:**
  -
- **Model saved:** `models/exp2_sac_baseline.pth`

---

### Experiment 3: DQN Baseline
- **Date:**
- **Algorithm:** DQN
- **Config:** `configs/dqn_config.yaml`
- **Hyperparameters:**
  - Learning rate: 1e-4
  - Batch size: 64
  - Epsilon decay: 100k steps
  - Frame stack: 4
- **Training:**
  - Episodes:
  - Total steps:
  - Training time:
- **Results:**
  - Average reward (last 100 ep):
  - Best reward:
  - Success rate:
- **Observations:**
  -
- **Model saved:** `models/exp3_dqn_baseline.pth`

---

## Hyperparameter Tuning Experiments

### Experiment 4: DDPG - Learning Rate Sweep
- **Date:**
- **Variations:**
  - LR_actor: [5e-5, 1e-4, 5e-4]
  - LR_critic: [5e-4, 1e-3, 5e-3]
- **Best configuration:**
  - LR_actor:
  - LR_critic:
  - Avg reward:
- **Notes:**
  -

---

### Experiment 5: Network Architecture Comparison
- **Date:**
- **Variations:**
  - Architecture A: 3 conv layers, 512 FC
  - Architecture B: 4 conv layers, 1024 FC
  - Architecture C: 3 conv layers, 256 FC
- **Best configuration:**
  - Architecture:
  - Avg reward:
- **Notes:**
  -

---

## Algorithm Comparison Summary

| Algorithm | Episodes | Avg Reward | Best Reward | Success Rate | Training Time | Notes |
|-----------|----------|------------|-------------|--------------|---------------|-------|
| DQN       |          |            |             |              |               |       |
| DDPG      |          |            |             |              |               |       |
| SAC       |          |            |             |              |               |       |

## Key Findings

### What Worked
1.
2.
3.

### What Didn't Work
1.
2.
3.

### Surprising Results
1.
2.

## Best Configuration

**Algorithm:**
**Hyperparameters:**
-
-
-

**Performance:**
- Average reward:
- Success rate:
- Track completion:

## Future Work / Ideas to Try
- [ ]
- [ ]
- [ ]

## Visualizations

- Training curves: `results/plots/training_curves.png`
- Algorithm comparison: `results/plots/algorithm_comparison.png`
- Best agent video: `results/videos/best_agent.mp4`
