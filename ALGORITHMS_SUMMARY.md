# Algorithms Summary

## Overview
This document provides a quick reference for the three RL algorithms we'll implement.

---

## 1. DDPG (Deep Deterministic Policy Gradient)

### Type
Actor-Critic, Off-Policy, Model-Free

### Key Characteristics
- **Best for:** Continuous action spaces
- **Learning:** Off-policy (can learn from old experiences)
- **Policy:** Deterministic
- **Exploration:** Uses additive noise (Ornstein-Uhlenbeck)

### Core Components
1. **Actor Network:** Maps state → action
2. **Critic Network:** Maps (state, action) → Q-value
3. **Target Networks:** Slow-moving copies for stability
4. **Replay Buffer:** Stores past experiences

### Pros
✓ Excellent for continuous control
✓ Sample efficient with replay buffer
✓ Stable with target networks
✓ Well-tested algorithm

### Cons
✗ Can be brittle with hyperparameters
✗ May overestimate Q-values
✗ Requires careful exploration tuning

### When to Use
- Primary choice for Car Racing
- When you need deterministic policies
- When sample efficiency matters

---

## 2. SAC (Soft Actor-Critic)

### Type
Actor-Critic, Off-Policy, Model-Free, Maximum Entropy

### Key Characteristics
- **Best for:** Continuous action spaces
- **Learning:** Off-policy
- **Policy:** Stochastic
- **Exploration:** Built into the policy via entropy maximization

### Core Components
1. **Actor Network:** Maps state → action distribution (Gaussian)
2. **Twin Critic Networks:** Two Q-networks (reduce overestimation)
3. **Target Networks:** Slow-moving copies
4. **Temperature Parameter (α):** Controls exploration (auto-tuned)
5. **Replay Buffer:** Stores past experiences

### Pros
✓ State-of-the-art for continuous control
✓ More stable than DDPG
✓ Automatic exploration tuning
✓ Robust to hyperparameters
✓ Prevents premature convergence

### Cons
✗ More complex implementation
✗ Higher computational cost
✗ More hyperparameters to manage

### When to Use
- When you want best performance
- When stability is important
- When you have computational resources

---

## 3. DQN (Deep Q-Network)

### Type
Value-Based, Off-Policy, Model-Free

### Key Characteristics
- **Best for:** Discrete action spaces
- **Learning:** Off-policy
- **Policy:** Epsilon-greedy
- **Exploration:** Epsilon decay

### Core Components
1. **Q-Network:** Maps state → Q-values for all actions
2. **Target Network:** Slow-moving copy
3. **Replay Buffer:** Stores past experiences
4. **Action Discretization:** Convert continuous to discrete

### Pros
✓ Simpler to implement
✓ Well-understood algorithm
✓ Good baseline
✓ Proven track record

### Cons
✗ Not designed for continuous actions (need discretization)
✗ Less sample efficient than actor-critic
✗ Can overestimate Q-values
✗ Limited action precision

### When to Use
- For baseline comparison
- When simplicity is preferred
- For educational purposes

---

## Algorithm Comparison

| Feature | DQN | DDPG | SAC |
|---------|-----|------|-----|
| **Action Space** | Discrete (discretized) | Continuous | Continuous |
| **Policy Type** | Epsilon-greedy | Deterministic | Stochastic |
| **Sample Efficiency** | Medium | High | Highest |
| **Stability** | Medium | Medium | High |
| **Implementation** | Simple | Medium | Complex |
| **Hyperparameter Sensitivity** | Medium | High | Low |
| **Computational Cost** | Low | Medium | High |
| **Best Use Case** | Baseline | Fast continuous | Best performance |

---

## Recommended Implementation Order

### 1. Start with DDPG
- Good balance of performance and complexity
- Native continuous action support
- Excellent learning resource

### 2. Add SAC
- Compare against DDPG
- Likely best performance
- More robust to tuning

### 3. Implement DQN
- Baseline comparison
- Different approach (value-based)
- Educational value

---

## Expected Performance (Car Racing)

Based on literature and benchmarks:

| Algorithm | Expected Avg Reward | Training Episodes | Notes |
|-----------|-------------------|-------------------|-------|
| Random | -50 to 0 | - | Baseline |
| DQN | 400-600 | 1500-2000 | Limited by discretization |
| DDPG | 600-800 | 1000-1500 | Good continuous control |
| SAC | 700-900+ | 1000-1500 | Best overall |

*Note: Actual results depend heavily on hyperparameters and training time*

---

## Key Hyperparameters by Algorithm

### DDPG
- **Critical:** Learning rates (actor & critic), noise parameters
- **Important:** τ (soft update), batch size, buffer size
- **Fine-tune:** Network architecture, gradient clipping

### SAC
- **Critical:** Learning rate, target entropy
- **Important:** Batch size, buffer size, τ
- **Fine-tune:** Network architecture, initial α

### DQN
- **Critical:** Learning rate, epsilon decay schedule
- **Important:** Batch size, target update frequency
- **Fine-tune:** Network architecture, frame stacking

---

## Implementation Tips

### DDPG
1. Start with Ornstein-Uhlenbeck noise (θ=0.15, σ=0.2)
2. Use soft updates (τ=0.005)
3. Normalize observations if possible
4. Clip actions to valid range

### SAC
1. Enable automatic α tuning
2. Use twin Q-networks
3. Start with target_entropy = -dim(action_space)
4. Higher learning rate than DDPG (3e-4)

### DQN
1. Use frame stacking (4 frames)
2. Convert to grayscale to reduce input size
3. Careful action discretization (don't use too many)
4. Longer epsilon decay (100k steps)

---

## Debugging Guide

### Agent Not Learning
- **DQN:** Check epsilon decay, verify discrete actions make sense
- **DDPG:** Adjust noise parameters, check learning rates
- **SAC:** Verify α tuning is working, check entropy values

### Training Unstable
- **DQN:** Reduce learning rate, increase target update frequency
- **DDPG:** Lower learning rates, increase τ, add gradient clipping
- **SAC:** Usually most stable; check critic learning rate

### Poor Final Performance
- **All:** Train longer, tune learning rates, adjust network size
- **DQN:** Refine action discretization
- **DDPG:** Adjust noise schedule
- **SAC:** Tune target entropy

---

## References

### Papers
- **DQN:** Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **DDPG:** Lillicrap et al. (2015) - "Continuous control with deep reinforcement learning"
- **SAC:** Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"

### Code References (Inspiration Only)
- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- CleanRL: https://github.com/vwxyzjn/cleanrl
