"""Main training script for DQN agent.

Trains a DQN agent on the Car Racing environment.
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.config_loader import load_config, save_config
from utils.preprocessing import FrameProcessor
from utils.logger import Logger
from utils.config_adapter import adapt_config
from agents.dqn_agent import DQNAgent


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_config):
    """Get torch device based on config."""
    if device_config == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_config


def evaluate_agent(env, agent, processor, discrete_actions, num_episodes=5):
    """Evaluate agent performance.

    Args:
        env: Gym environment
        agent: DQN agent
        processor: Frame processor
        discrete_actions: List of discrete action mappings
        num_episodes (int): Number of evaluation episodes

    Returns:
        dict: Evaluation metrics
    """
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = processor.reset(state)
        episode_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state, eval_mode=True)
            action = discrete_actions[action_idx]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = processor.step(next_state)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return {
        'eval_mean_reward': np.mean(total_rewards),
        'eval_std_reward': np.std(total_rewards),
        'eval_min_reward': np.min(total_rewards),
        'eval_max_reward': np.max(total_rewards)
    }


def train(config_path, resume_path=None):
    """Main training function.

    Args:
        config_path (str): Path to config file
        resume_path (str): Path to checkpoint to resume from
    """
    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # Adapt config format
    config = adapt_config(config)
    print("Config adapted to implementation format")

    # Set seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Get device
    device = get_device(config['training']['device'])
    print(f"Using device: {device}")

    # Create environment
    env = gym.make(config['env']['name'])
    print(f"Created environment: {config['env']['name']}")

    # Discrete actions
    discrete_actions = config['actions']
    num_actions = len(discrete_actions)

    # Frame processor
    preproc_config = config['preprocessing']
    processor = FrameProcessor(
        frame_size=tuple(preproc_config['frame_size']),
        grayscale=preproc_config['grayscale'],
        normalize=preproc_config['normalize'],
        frame_stack=preproc_config['frame_stack']
    )
    state_shape = processor.get_state_shape()
    print(f"State shape: {state_shape}")

    # Create agent
    agent = DQNAgent(config, state_shape, num_actions, device)
    print(f"Created DQN agent")

    # Resume from checkpoint if specified
    if resume_path:
        agent.load_checkpoint(resume_path)
        print(f"Resumed from checkpoint: {resume_path}")

    # Logger
    log_dir = Path(config['training']['log_dir'])
    logger = Logger(log_dir, use_tensorboard=config['logging']['tensorboard'])

    # Save config
    save_config(config, log_dir / 'config.yaml')

    # Training parameters
    total_episodes = config['training']['total_episodes']
    max_steps = config['training']['max_steps_per_episode']
    eval_freq = config['training']['eval_frequency']
    eval_episodes = config['training']['eval_episodes']
    save_freq = config['training']['save_frequency']

    # Checkpoint directory
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_eval_reward = -float('inf')
    episode_rewards = []

    print(f"\nStarting training for {total_episodes} episodes...")
    print("=" * 80)

    for episode in tqdm(range(total_episodes), desc="Training"):
        # Reset environment
        state, _ = env.reset(seed=seed + episode)
        state = processor.reset(state)

        episode_reward = 0
        episode_steps = 0
        episode_losses = []

        for step in range(max_steps):
            # Select and perform action
            action_idx = agent.select_action(state)
            action = discrete_actions[action_idx]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Process next state
            next_state = processor.step(next_state)

            # Store transition
            agent.store_transition(state, action_idx, reward, next_state, done)

            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        # Update episode counter
        agent.episodes += 1
        episode_rewards.append(episode_reward)

        # Logging
        metrics = {
            'reward': episode_reward,
            'steps': episode_steps,
            'epsilon': agent.epsilon,
            'loss': np.mean(episode_losses) if episode_losses else 0.0,
            'buffer_size': len(agent.replay_buffer),
            'avg_reward_100': np.mean(episode_rewards[-100:])
        }

        logger.log_episode(episode, metrics)

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            print(f"\nEvaluating at episode {episode + 1}...")
            eval_metrics = evaluate_agent(env, agent, processor, discrete_actions, eval_episodes)
            logger.log_episode(episode, eval_metrics)

            # Save best model
            if eval_metrics['eval_mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['eval_mean_reward']
                best_path = checkpoint_dir / 'best_model.pth'
                agent.save_checkpoint(best_path)
                print(f"New best model! Reward: {best_eval_reward:.2f}")

        # Periodic checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode + 1}.pth'
            agent.save_checkpoint(checkpoint_path)

    # Final checkpoint
    final_path = checkpoint_dir / 'final_model.pth'
    agent.save_checkpoint(final_path)

    # Close
    logger.close()
    env.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Models saved to: {checkpoint_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train DQN agent on Car Racing')
    parser.add_argument('--config', type=str, default='configs/dqn_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train(args.config, args.resume)


if __name__ == '__main__':
    main()
