"""Plot training results from logs.

Generates training curves and visualizations.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def moving_average(data, window=100):
    """Calculate moving average.

    Args:
        data (array): Input data
        window (int): Window size

    Returns:
        array: Moving average
    """
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_curves(csv_path, save_dir=None, window=100):
    """Plot training curves from CSV file.

    Args:
        csv_path (str): Path to metrics CSV file
        save_dir (str): Directory to save plots
        window (int): Moving average window size
    """
    # Load data
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")

    # Create save directory
    if save_dir is None:
        save_dir = csv_path.parent / 'plots'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Episode Reward
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
    if len(df) > window:
        ma = moving_average(df['reward'].values, window)
        plt.plot(df['episode'][window-1:], ma, linewidth=2, label=f'{window}-Episode MA')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress: Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'reward_curve.png', dpi=300)
    print(f"Saved: {save_dir / 'reward_curve.png'}")

    # Plot 2: Loss
    if 'loss' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['episode'], df['loss'], alpha=0.3, label='Loss')
        if len(df) > window:
            ma_loss = moving_average(df['loss'].values, window)
            plt.plot(df['episode'][window-1:], ma_loss, linewidth=2, label=f'{window}-Episode MA')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'loss_curve.png', dpi=300)
        print(f"Saved: {save_dir / 'loss_curve.png'}")

    # Plot 3: Epsilon
    if 'epsilon' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['episode'], df['epsilon'])
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'epsilon_curve.png', dpi=300)
        print(f"Saved: {save_dir / 'epsilon_curve.png'}")

    # Plot 4: Combined metrics
    if 'avg_reward_100' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Reward
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3)
        axes[0, 0].plot(df['episode'], df['avg_reward_100'], linewidth=2, color='red')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Reward (with 100-episode average)')
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        if 'loss' in df.columns:
            axes[0, 1].plot(df['episode'], df['loss'], alpha=0.5)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # Epsilon
        if 'epsilon' in df.columns:
            axes[1, 0].plot(df['episode'], df['epsilon'])
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_title('Exploration Rate')
            axes[1, 0].grid(True, alpha=0.3)

        # Buffer size
        if 'buffer_size' in df.columns:
            axes[1, 1].plot(df['episode'], df['buffer_size'])
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Buffer Size')
            axes[1, 1].set_title('Replay Buffer Size')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'combined_metrics.png', dpi=300)
        print(f"Saved: {save_dir / 'combined_metrics.png'}")

    plt.close('all')

    # Print statistics
    print("\n" + "=" * 80)
    print("Training Statistics:")
    print(f"  Total Episodes: {len(df)}")
    print(f"  Final Reward: {df['reward'].iloc[-1]:.2f}")
    print(f"  Best Reward: {df['reward'].max():.2f}")
    print(f"  Mean Reward (last 100): {df['reward'].iloc[-100:].mean():.2f}")
    if 'avg_reward_100' in df.columns:
        print(f"  Final Avg Reward (100): {df['avg_reward_100'].iloc[-1]:.2f}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to metrics CSV file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--window', type=int, default=100,
                        help='Moving average window size')

    args = parser.parse_args()

    plot_training_curves(args.csv, args.save_dir, args.window)


if __name__ == '__main__':
    main()
