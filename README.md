# Car Racing Reinforcement Learning Hackathon

## Project Overview
This project implements and trains deep reinforcement learning agents for the OpenAI Gym "Car Racing" environment using various RL algorithms.

**Environment:** [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)

## Environment Details
- **Observation Space:** RGB image (96x96x3)
- **Action Space:** Continuous [steering, gas, brake]
  - Steering: [-1, 1] (left to right)
  - Gas: [0, 1]
  - Brake: [0, 1]
- **Reward:** -0.1 per frame + 1000/N for each track tile visited (N = total tiles)
- **Episode Termination:** Off-track or completing the track

## Project Structure
```
car_racing_hackathon/
├── src/
│   ├── agents/          # RL agent implementations
│   ├── networks/        # Neural network architectures
│   └── utils/           # Helper functions and utilities
├── models/              # Saved model checkpoints
├── results/
│   ├── plots/           # Performance plots and visualizations
│   └── logs/            # Training logs
├── notebooks/           # Jupyter notebooks for analysis
├── configs/             # Configuration files
├── README.md
├── PLAN.md             # Detailed implementation plan
└── requirements.txt    # Python dependencies
```

## Algorithms Implemented
1. **DQN (Deep Q-Network)** - Discrete action space version
2. **DDPG (Deep Deterministic Policy Gradient)** - Actor-Critic for continuous control
3. **SAC (Soft Actor-Critic)** - State-of-the-art continuous control

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train an agent
python src/train.py --algorithm ddpg --config configs/ddpg_config.yaml

# Evaluate an agent
python src/evaluate.py --model models/ddpg_best.pth

# Visualize results
python src/visualize_results.py --results results/logs/
```

## Team Members
- [Add team member names]

## Results Summary
[To be filled during hackathon]

## References
- Gymnasium CarRacing: https://gymnasium.farama.org/environments/box2d/car_racing/
- DQN Paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- DDPG Paper: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- SAC Paper: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
