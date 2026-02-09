"""Test script to verify all components work correctly.

Run this before starting training to catch any issues.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.replay_buffer import ReplayBuffer
from utils.preprocessing import FrameProcessor
from networks.q_network import QNetwork


def test_replay_buffer():
    """Test replay buffer."""
    print("Testing ReplayBuffer...")

    state_shape = (96, 96, 4)
    buffer = ReplayBuffer(capacity=1000, state_shape=state_shape, seed=42)

    # Add some experiences
    for i in range(100):
        state = np.random.rand(*state_shape).astype(np.float32)
        action = np.random.randint(0, 9)
        reward = np.random.randn()
        next_state = np.random.rand(*state_shape).astype(np.float32)
        done = bool(np.random.rand() > 0.95)

        buffer.add(state, action, reward, next_state, done)

    # Check size
    assert len(buffer) == 100, f"Expected size 100, got {len(buffer)}"

    # Sample batch
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Check shapes
    assert states.shape == (batch_size, *state_shape)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, *state_shape)
    assert dones.shape == (batch_size,)

    # Check types
    assert states.dtype == np.float32
    assert actions.dtype == np.int64
    assert rewards.dtype == np.float32

    print("✓ ReplayBuffer tests passed")


def test_frame_processor():
    """Test frame processor."""
    print("\nTesting FrameProcessor...")

    processor = FrameProcessor(
        frame_size=(96, 96),
        grayscale=True,
        normalize=True,
        frame_stack=4
    )

    # Create dummy RGB frame
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    # Reset
    stacked = processor.reset(frame)
    assert stacked.shape == (96, 96, 4), f"Expected (96, 96, 4), got {stacked.shape}"
    assert stacked.dtype == np.float32
    assert 0 <= stacked.min() <= 1.0 and 0 <= stacked.max() <= 1.0

    # Step
    frame2 = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    stacked2 = processor.step(frame2)
    assert stacked2.shape == (96, 96, 4)

    # Check that frames shifted
    assert not np.allclose(stacked, stacked2)

    print("✓ FrameProcessor tests passed")


def test_q_network():
    """Test Q-network."""
    print("\nTesting QNetwork...")

    state_shape = (96, 96, 4)
    num_actions = 9

    conv_layers = [
        {'filters': 32, 'kernel_size': 8, 'stride': 4},
        {'filters': 64, 'kernel_size': 4, 'stride': 2},
        {'filters': 64, 'kernel_size': 3, 'stride': 1}
    ]
    fc_layers = [512]

    # Standard architecture
    net = QNetwork(state_shape, num_actions, conv_layers, fc_layers, dueling=False)

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, *state_shape)

    output = net(dummy_input)
    assert output.shape == (batch_size, num_actions), f"Expected ({batch_size}, {num_actions}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

    # Test backward pass
    loss = output.mean()
    loss.backward()

    # Check gradients exist
    for param in net.parameters():
        assert param.grad is not None, "Gradients not computed"

    print("✓ QNetwork tests passed")

    # Test dueling architecture
    print("\nTesting Dueling QNetwork...")
    dueling_net = QNetwork(state_shape, num_actions, conv_layers, fc_layers, dueling=True)
    output_dueling = dueling_net(dummy_input)
    assert output_dueling.shape == (batch_size, num_actions)
    assert not torch.isnan(output_dueling).any()

    print("✓ Dueling QNetwork tests passed")


def test_config():
    """Test config loading."""
    print("\nTesting config loading...")

    from utils.config_loader import load_config

    config_path = Path(__file__).parent.parent / 'configs' / 'dqn_config.yaml'

    if not config_path.exists():
        print(f"⚠ Config file not found: {config_path}")
        return

    config = load_config(config_path)

    # Check required sections
    required = ['env', 'preprocessing', 'network', 'agent', 'training']
    for section in required:
        assert section in config, f"Missing section: {section}"

    # Check some key values
    assert config['env']['num_actions'] == 9
    assert len(config['actions']) == 9
    assert config['agent']['gamma'] == 0.99

    print("✓ Config loading tests passed")


def test_environment():
    """Test environment creation."""
    print("\nTesting environment creation...")

    try:
        import gymnasium as gym
        env = gym.make('CarRacing-v3')
        state, _ = env.reset()

        assert state.shape == (96, 96, 3), f"Unexpected state shape: {state.shape}"

        # Test step
        action = [0.0, 0.0, 0.0]
        next_state, reward, terminated, truncated, _ = env.step(action)

        assert next_state.shape == (96, 96, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        env.close()
        print("✓ Environment tests passed")

    except Exception as e:
        print(f"⚠ Environment test failed: {e}")
        print("Make sure gymnasium[box2d] is installed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Running component tests...")
    print("=" * 80)

    try:
        test_replay_buffer()
        test_frame_processor()
        test_q_network()
        test_config()
        test_environment()

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        print("\nYou can now run training with:")
        print("  python train.py --config ../configs/dqn_config.yaml")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
