# Quick Start Guide

## Installation

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Test the environment:**
```bash
python -c "import gymnasium as gym; env = gym.make('CarRacing-v3', render_mode='rgb_array'); print('Environment loaded successfully!')"
```

## Project Structure Overview

```
car_racing_hackathon/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── ddpg_agent.py   # DDPG algorithm
│   │   ├── sac_agent.py    # SAC algorithm
│   │   └── dqn_agent.py    # DQN algorithm
│   ├── networks/            # Neural networks
│   │   ├── actor.py        # Actor networks
│   │   ├── critic.py       # Critic networks
│   │   └── q_network.py    # Q-networks for DQN
│   └── utils/               # Utilities
│       ├── replay_buffer.py # Experience replay
│       ├── preprocessing.py # Image preprocessing
│       └── logger.py        # Training logger
├── configs/                 # YAML config files
├── models/                  # Saved models
└── results/                 # Training results
```

## Running Your First Training

1. **Start with DDPG (recommended for continuous control):**
```bash
python src/train.py --algorithm ddpg --episodes 1000
```

2. **Monitor training with tensorboard:**
```bash
tensorboard --logdir results/logs/
```

3. **Evaluate trained model:**
```bash
python src/evaluate.py --model models/ddpg_best.pth --episodes 10 --render
```

## Key Commands

### Training
```bash
# Train DDPG
python src/train.py --algorithm ddpg --config configs/ddpg_config.yaml

# Train SAC
python src/train.py --algorithm sac --config configs/sac_config.yaml

# Train DQN
python src/train.py --algorithm dqn --config configs/dqn_config.yaml

# Resume training from checkpoint
python src/train.py --algorithm ddpg --resume models/ddpg_checkpoint.pth
```

### Evaluation
```bash
# Evaluate and render
python src/evaluate.py --model models/best_model.pth --render

# Evaluate multiple episodes
python src/evaluate.py --model models/best_model.pth --episodes 50

# Record video
python src/evaluate.py --model models/best_model.pth --record --output videos/
```

### Analysis
```bash
# Generate training plots
python src/plot_results.py --log_dir results/logs/

# Compare algorithms
python src/compare_algorithms.py --models models/ddpg_best.pth models/sac_best.pth models/dqn_best.pth
```

## Development Workflow

### 1. Implement Base Algorithm
- Start with one algorithm (DDPG recommended)
- Test on simple training run (100 episodes)
- Verify learning is happening (reward increasing)

### 2. Hyperparameter Search
- Modify config files
- Run multiple experiments with different hyperparameters
- Track results in tensorboard

### 3. Compare Algorithms
- Implement 2-3 different algorithms
- Train each with similar computational budget
- Create comparison plots

### 4. Optimize Best Algorithm
- Focus on most promising algorithm
- Fine-tune hyperparameters
- Train for longer duration

## Debugging Tips

### Agent not learning?
- Check learning rate (try 1e-4, 5e-4, 1e-3)
- Verify reward signal is received
- Check if actions are being taken (not stuck at zeros)
- Reduce batch size if memory issues
- Add more exploration noise

### Training unstable?
- Lower learning rate
- Add gradient clipping
- Increase target network update rate (τ)
- Use layer normalization
- Check for NaN values in losses

### Poor performance after training?
- Train for more episodes (1000+)
- Check if overfitting to specific track seeds
- Verify preprocessing is correct
- Try different network architectures
- Adjust reward shaping

## Performance Benchmarks

| Algorithm | Episodes | Avg Reward | Training Time (GPU) |
|-----------|----------|------------|---------------------|
| Random    | -        | ~-50       | -                   |
| DQN       | 1000     | 400-600    | 6-8 hours           |
| DDPG      | 1000     | 600-800    | 8-10 hours          |
| SAC       | 1000     | 700-900    | 10-12 hours         |

*Note: These are approximate estimates. Actual results depend on hyperparameters.*

## Common Issues

### Issue: "Box2D not installed"
```bash
pip install box2d-py
# or
pip install gymnasium[box2d]
```

### Issue: CUDA out of memory
- Reduce batch size
- Reduce replay buffer size
- Use gradient accumulation

### Issue: Environment rendering slow
- Use `render_mode='rgb_array'` for training (no display)
- Only use `render_mode='human'` for evaluation

## Next Steps

1. Read `PLAN.md` for detailed implementation strategy
2. Review Car Racing environment documentation
3. Start with baseline DDPG implementation
4. Set up experiment tracking (tensorboard)
5. Run first training session
6. Analyze results and iterate

## Resources

- **Environment Docs:** https://gymnasium.farama.org/environments/box2d/car_racing/
- **DDPG Paper:** https://arxiv.org/abs/1509.02971
- **SAC Paper:** https://arxiv.org/abs/1801.01290
- **DQN Paper:** https://www.nature.com/articles/nature14236

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review the documentation
3. Search for similar issues online
4. Ask team members
5. Check Gymnasium GitHub issues
