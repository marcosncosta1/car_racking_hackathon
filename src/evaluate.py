"""Evaluation script for trained DQN agent.

Loads a trained model and evaluates its performance.
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path

from utils.config_loader import load_config
from utils.preprocessing import FrameProcessor
from agents.dqn_agent import DQNAgent


def evaluate(model_path, num_episodes=10, render=False, record=False, seed=42):
    """Evaluate a trained agent.

    Args:
        model_path (str): Path to model checkpoint
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render episodes
        record (bool): Whether to record video
        seed (int): Random seed
    """
    # Load checkpoint
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    print(f"Loaded model from {model_path}")
    print(f"Training episodes: {checkpoint['episodes']}")
    print(f"Training steps: {checkpoint['steps']}")

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    render_mode = 'human' if render else None
    if record:
        render_mode = 'rgb_array'

    env = gym.make(config['env']['name'], render_mode=render_mode)

    # Wrap for video recording if requested
    if record:
        from gymnasium.wrappers import RecordVideo
        video_dir = Path(config['training']['video_dir'])
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, str(video_dir), episode_trigger=lambda x: True)
        print(f"Recording videos to: {video_dir}")

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

    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(config, state_shape, num_actions, device)

    # Load weights
    agent.load_checkpoint(model_path)
    print(f"Agent loaded successfully")

    # Evaluation
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 80)

    all_rewards = []
    all_steps = []
    completion_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        state = processor.reset(state)

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # Select action (greedy)
            action_idx = agent.select_action(state, eval_mode=True)
            action = discrete_actions[action_idx]

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = processor.step(next_state)
            episode_reward += reward
            episode_steps += 1
            state = next_state

        all_rewards.append(episode_reward)
        all_steps.append(episode_steps)

        # Check if track was completed (reward > 700 is typical for completion)
        if episode_reward > 700:
            completion_count += 1

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

    # Statistics
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print(f"  Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Min Reward:  {np.min(all_rewards):.2f}")
    print(f"  Max Reward:  {np.max(all_rewards):.2f}")
    print(f"  Mean Steps:  {np.mean(all_steps):.1f}")
    print(f"  Completion Rate: {completion_count}/{num_episodes} ({100*completion_count/num_episodes:.1f}%)")
    print("=" * 80)

    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')
    parser.add_argument('--record', action='store_true',
                        help='Record videos')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render,
        record=args.record,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
