"""DQN Agent implementation.

Deep Q-Network agent with experience replay and target network.
Supports Double DQN and epsilon-greedy exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent."""

    def __init__(self, config, state_shape, num_actions, device='cpu'):
        """Initialize DQN agent.

        Args:
            config (dict): Configuration dictionary
            state_shape (tuple): Shape of state observations
            num_actions (int): Number of discrete actions
            device (str): Device to use ('cpu' or 'cuda')
        """
        self.config = config
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device

        # Extract config parameters
        agent_config = config['agent']
        self.gamma = agent_config['gamma']
        self.learning_rate = agent_config['learning_rate']
        self.batch_size = agent_config['batch_size']
        self.learning_starts = agent_config['learning_starts']
        self.target_update_freq = agent_config['target_update_frequency']
        self.train_freq = agent_config['train_frequency']
        self.grad_clip = agent_config.get('grad_clip', None)
        self.double_dqn = agent_config['double_dqn']

        # Exploration parameters
        self.epsilon = agent_config['epsilon_start']
        self.epsilon_start = agent_config['epsilon_start']
        self.epsilon_end = agent_config['epsilon_end']
        self.epsilon_decay_steps = agent_config['epsilon_decay_steps']
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps

        # Networks
        network_config = config['network']
        self.online_net = QNetwork(
            state_shape,
            num_actions,
            network_config['conv_layers'],
            network_config['fc_layers'],
            network_config['dueling']
        ).to(device)

        self.target_net = QNetwork(
            state_shape,
            num_actions,
            network_config['conv_layers'],
            network_config['fc_layers'],
            network_config['dueling']
        ).to(device)

        # Initialize target network
        self.update_target_network()
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            agent_config['replay_buffer_size'],
            state_shape,
            seed=config['training'].get('seed')
        )

        # Training state
        self.steps = 0
        self.episodes = 0
        self.losses = []

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy.

        Args:
            state (array): Current state
            eval_mode (bool): If True, use greedy policy (no exploration)

        Returns:
            int: Selected action
        """
        if eval_mode or random.random() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_tensor)
                action = q_values.argmax(dim=1).item()
        else:
            # Random action
            action = random.randrange(self.num_actions)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer.

        Args:
            state (array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (array): Next state
            done (bool): Whether episode ended
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step.

        Returns:
            float: Loss value (or None if not training yet)
        """
        self.steps += 1

        # Check if we should train
        if self.steps < self.learning_starts:
            return None

        if self.steps % self.train_freq != 0:
            return None

        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q-values
        q_values = self.online_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: select actions with online net, evaluate with target net
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)

        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon
        self.decay_epsilon()

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def update_target_network(self):
        """Copy weights from online network to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.steps < self.epsilon_decay_steps:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def save_checkpoint(self, filepath):
        """Save agent checkpoint.

        Args:
            filepath (str): Path to save checkpoint
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'config': self.config
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load agent checkpoint.

        Args:
            filepath (str): Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']

        print(f"Checkpoint loaded from {filepath}")

    def get_stats(self):
        """Get training statistics.

        Returns:
            dict: Statistics dictionary
        """
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'replay_buffer_size': len(self.replay_buffer)
        }
