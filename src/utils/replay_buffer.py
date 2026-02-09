"""Experience Replay Buffer for DQN.

Stores and samples past experiences for training.
Implements a circular buffer for memory efficiency.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity, state_shape, seed=None):
        """Initialize a ReplayBuffer object.

        Args:
            capacity (int): Maximum size of buffer
            state_shape (tuple): Shape of state observations
            seed (int, optional): Random seed for reproducibility
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.position = 0
        self.size = 0

        # Pre-allocate arrays for memory efficiency
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Args:
            state (array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (array): Next state
            done (bool): Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory.

        Args:
            batch_size (int): Size of batch to sample

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        # Random sampling without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size

    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training.

        Args:
            batch_size (int): Required batch size

        Returns:
            bool: True if buffer has enough samples
        """
        return self.size >= batch_size
