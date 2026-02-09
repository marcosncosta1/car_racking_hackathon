"""Q-Network for DQN.

Convolutional neural network that maps states to Q-values for discrete actions.
Supports standard and dueling architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q-Network with convolutional layers."""

    def __init__(self, state_shape, num_actions, conv_layers, fc_layers, dueling=False):
        """Initialize Q-Network.

        Args:
            state_shape (tuple): Shape of state (H, W, C)
            num_actions (int): Number of discrete actions
            conv_layers (list): List of conv layer configs
            fc_layers (list): List of FC layer sizes
            dueling (bool): Use dueling architecture
        """
        super(QNetwork, self).__init__()

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.dueling = dueling

        # Input channels (frame_stack for grayscale)
        in_channels = state_shape[2]

        # Convolutional layers
        conv_modules = []
        for conv_config in conv_layers:
            conv_modules.append(
                nn.Conv2d(
                    in_channels,
                    conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    stride=conv_config['stride']
                )
            )
            conv_modules.append(nn.ReLU())
            in_channels = conv_config['filters']

        self.conv = nn.Sequential(*conv_modules)

        # Calculate conv output size
        conv_output_size = self._get_conv_output_size(state_shape)

        # Fully connected layers
        if dueling:
            # Dueling architecture: separate value and advantage streams
            # Shared feature layer
            self.fc_shared = nn.Linear(conv_output_size, fc_layers[0])

            # Value stream
            self.fc_value = nn.Linear(fc_layers[0], 1)

            # Advantage stream
            self.fc_advantage = nn.Linear(fc_layers[0], num_actions)
        else:
            # Standard DQN architecture
            fc_modules = []
            in_features = conv_output_size

            for fc_size in fc_layers:
                fc_modules.append(nn.Linear(in_features, fc_size))
                fc_modules.append(nn.ReLU())
                in_features = fc_size

            # Output layer
            fc_modules.append(nn.Linear(in_features, num_actions))

            self.fc = nn.Sequential(*fc_modules)

    def _get_conv_output_size(self, state_shape):
        """Calculate the output size of convolutional layers.

        Args:
            state_shape (tuple): Input state shape (H, W, C)

        Returns:
            int: Flattened output size
        """
        # Create dummy input
        dummy_input = torch.zeros(1, state_shape[2], state_shape[0], state_shape[1])
        conv_output = self.conv(dummy_input)
        return int(np.prod(conv_output.shape[1:]))

    def forward(self, state):
        """Forward pass.

        Args:
            state (tensor): State tensor (B, C, H, W) or (B, H, W, C)

        Returns:
            tensor: Q-values for each action (B, num_actions)
        """
        # Ensure correct input format: (B, C, H, W)
        if state.dim() == 3:
            state = state.unsqueeze(0)  # Add batch dimension

        # Convert from (B, H, W, C) to (B, C, H, W) if necessary
        if state.shape[-1] == self.state_shape[2]:
            state = state.permute(0, 3, 1, 2)

        # Convolutional layers
        x = self.conv(state)
        x = x.reshape(x.size(0), -1)  # Flatten

        if self.dueling:
            # Dueling architecture
            x = F.relu(self.fc_shared(x))

            value = self.fc_value(x)
            advantage = self.fc_advantage(x)

            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard architecture
            q_values = self.fc(x)

        return q_values


# Import numpy for conv output calculation
import numpy as np
