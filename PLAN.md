# Implementation Plan - Car Racing RL Hackathon

## 1. Algorithms to Implement

### Algorithm 1: DDPG (Deep Deterministic Policy Gradient)
**Why:** Best suited for continuous action spaces like Car Racing
- Actor-Critic architecture
- Off-policy learning with replay buffer
- Deterministic policy with exploration noise

**Network Architecture:**
- **Actor Network:**
  - Input: 96x96x3 RGB image
  - Conv2D(32, 8x8, stride=4) → ReLU
  - Conv2D(64, 4x4, stride=2) → ReLU
  - Conv2D(64, 3x3, stride=1) → ReLU
  - Flatten → FC(512) → ReLU
  - Output: FC(3) with Tanh activation [steering, gas, brake]

- **Critic Network:**
  - Input: State (image) + Action (3D vector)
  - State path: Same conv layers as actor
  - Action path: FC(64) → ReLU
  - Concatenate → FC(256) → ReLU → FC(1) (Q-value)

**Key Hyperparameters:**
- Learning rate (actor): 1e-4, 5e-4, 1e-3
- Learning rate (critic): 1e-3, 5e-3, 1e-2
- Discount factor (γ): 0.99, 0.95
- Soft update rate (τ): 0.001, 0.005
- Replay buffer size: 100k, 500k
- Batch size: 64, 128, 256
- Exploration noise (Ornstein-Uhlenbeck): σ=0.2, θ=0.15

---

### Algorithm 2: SAC (Soft Actor-Critic)
**Why:** State-of-the-art for continuous control, more stable than DDPG
- Maximum entropy framework
- Automatic temperature tuning
- Double Q-networks for stability

**Network Architecture:**
- **Actor Network (Stochastic):**
  - Same convolutional backbone
  - Flatten → FC(512) → ReLU
  - Mean: FC(3)
  - Log_std: FC(3)
  - Output: Gaussian policy with reparameterization trick

- **Twin Critic Networks (Q1, Q2):**
  - Input: State + Action
  - Same architecture as DDPG critic
  - Take minimum Q-value for policy update (reduce overestimation)

**Key Hyperparameters:**
- Learning rate: 3e-4, 1e-3
- Discount factor (γ): 0.99
- Soft update rate (τ): 0.005
- Replay buffer size: 1M
- Batch size: 256
- Target entropy: -dim(action_space) = -3
- Initial temperature (α): 0.2 (auto-tuned)

---

### Algorithm 3: DQN with Discretized Actions
**Why:** Simpler to implement, good baseline for comparison
- Discretize continuous action space
- Experience replay and target network
- Frame stacking for temporal information

**Action Discretization:**
- Steering: {-1, -0.5, 0, 0.5, 1} (5 values)
- Gas: {0, 0.5, 1} (3 values)
- Brake: {0, 1} (2 values)
- Total actions: 5 × 3 × 2 = 30 discrete actions (or simplify to 9 common actions)

**Network Architecture:**
- Input: 4 stacked frames (96x96x12)
- Conv2D(32, 8x8, stride=4) → ReLU
- Conv2D(64, 4x4, stride=2) → ReLU
- Conv2D(64, 3x3, stride=1) → ReLU
- Flatten → FC(512) → ReLU
- Output: FC(num_actions) (Q-values)

**Key Hyperparameters:**
- Learning rate: 1e-4, 5e-4
- Discount factor (γ): 0.99
- Epsilon decay: 1.0 → 0.01 over 100k steps
- Replay buffer size: 100k
- Batch size: 32, 64
- Target network update: every 1000 steps
- Frame stack: 4 frames

---

## 2. Hyperparameters to Tune

### Critical Parameters (Test Multiple Values)
1. **Learning Rate**
   - Too high: instability, divergence
   - Too low: slow learning
   - Test: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3

2. **Discount Factor (γ)**
   - Higher: more long-term planning
   - Test: 0.95, 0.99, 0.995

3. **Batch Size**
   - Larger: more stable gradients
   - Test: 32, 64, 128, 256

4. **Replay Buffer Size**
   - Larger: more diverse experiences
   - Test: 50k, 100k, 500k, 1M

5. **Network Architecture**
   - Number of conv layers: 3, 4
   - FC layer size: 256, 512, 1024
   - Activation functions: ReLU, ELU, Leaky ReLU

### Secondary Parameters
- Update frequency
- Exploration noise parameters
- Gradient clipping threshold
- Weight initialization scheme

---

## 3. Training Strategy

### Phase 1: Baseline Implementation (Days 1-2)
- Set up environment and basic infrastructure
- Implement one algorithm (DDPG recommended)
- Get first successful training run
- Establish evaluation metrics and logging

### Phase 2: Algorithm Comparison (Days 3-4)
- Implement remaining algorithms (SAC, DQN)
- Run with default hyperparameters
- Compare baseline performance
- Identify most promising algorithm

### Phase 3: Hyperparameter Optimization (Days 5-6)
- Grid search or random search on top algorithm
- Focus on learning rate, batch size, network size
- Document all experiments systematically

### Phase 4: Fine-tuning & Analysis (Day 7)
- Train best configuration for longer
- Generate plots and performance metrics
- Prepare presentation materials
- Record video of best agent

---

## 4. Evaluation Metrics

### Primary Metrics
1. **Average Episode Reward**
   - Track over training episodes
   - Moving average (window=100)
   - Target: >900 (excellent performance)

2. **Success Rate**
   - % of episodes completing the track
   - % of episodes with reward > 500

3. **Sample Efficiency**
   - Reward vs. training steps/episodes
   - Compare across algorithms

### Secondary Metrics
- Training time per episode
- Policy loss, value loss over time
- Exploration vs exploitation balance
- Track completion percentage

### Visualizations to Create
1. **Training Curves**
   - Reward vs episodes (all algorithms)
   - Loss curves
   - Moving average with confidence intervals

2. **Comparison Charts**
   - Algorithm comparison (bar charts)
   - Hyperparameter sensitivity plots
   - Sample efficiency curves

3. **Qualitative Analysis**
   - Video recordings of trained agents
   - Heatmaps of visited track regions
   - Action distribution histograms

---

## 5. Implementation Checklist

### Environment Setup
- [ ] Install gymnasium[box2d]
- [ ] Test environment rendering
- [ ] Implement frame preprocessing (grayscale/color, normalization)
- [ ] Implement frame stacking (if needed)

### Core Components
- [ ] Replay buffer implementation
- [ ] Neural network definitions
- [ ] Training loop with logging
- [ ] Evaluation script
- [ ] Checkpoint saving/loading

### Algorithm Implementations
- [ ] DDPG agent
- [ ] SAC agent
- [ ] DQN agent

### Experimentation
- [ ] Configuration management system
- [ ] Experiment tracking (tensorboard/wandb)
- [ ] Automated hyperparameter search
- [ ] Result aggregation and plotting

### Documentation
- [ ] Training logs for each run
- [ ] Final report with comparisons
- [ ] Code documentation
- [ ] Presentation slides

---

## 6. Technical Considerations

### Preprocessing
- **Image Processing:**
  - Option 1: Keep RGB (96x96x3)
  - Option 2: Convert to grayscale (96x96x1) - faster
  - Normalize pixel values to [0, 1] or [-1, 1]

- **Reward Shaping:**
  - Original reward can be sparse
  - Consider: +penalty for grass, +bonus for speed, +bonus for staying on track

### Stability Tricks
1. Gradient clipping (prevent exploding gradients)
2. Learning rate scheduling (decay over time)
3. Prioritized experience replay (sample important transitions more)
4. Reward normalization/clipping
5. Layer normalization in networks

### Computational Resources
- GPU strongly recommended (CUDA)
- Expect 12-24 hours training for good results
- Save checkpoints frequently
- Log to disk continuously (prevent data loss)

---

## 7. Expected Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1 | Setup, environment testing, DDPG implementation | Working training script |
| 2 | Complete DDPG, first training runs | Baseline DDPG results |
| 3 | Implement SAC | SAC baseline |
| 4 | Implement DQN, algorithm comparison | All 3 algorithms working |
| 5 | Hyperparameter tuning (learning rate, batch size) | Performance improvements |
| 6 | Network architecture experiments, longer training | Best model identified |
| 7 | Final training, plots, documentation, presentation | Final submission |

---

## 8. Success Criteria

### Minimum Viable Product
- ✓ At least 1 algorithm fully implemented
- ✓ Agent learns to improve over random baseline
- ✓ Training/evaluation curves generated
- ✓ Code documented and runnable

### Target Performance
- ✓ Average reward > 700
- ✓ 3 algorithms implemented and compared
- ✓ Systematic hyperparameter study
- ✓ Comprehensive plots and analysis

### Stretch Goals
- ✓ Average reward > 900
- ✓ Track completion in >80% of episodes
- ✓ Novel architecture or technique
- ✓ Ablation studies showing impact of each component

---

## 9. References & Resources

### Papers
- **DDPG:** Lillicrap et al., "Continuous Control with Deep RL" (2015)
- **SAC:** Haarnoja et al., "Soft Actor-Critic" (2018)
- **DQN:** Mnih et al., "Human-Level Control" (2015)

### Code Resources (for inspiration only)
- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- CleanRL: https://docs.cleanrl.dev/

### Documentation
- Gymnasium API: https://gymnasium.farama.org/api/
- PyTorch RL Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
