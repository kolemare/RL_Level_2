import torch
import torch.nn as nn
from torch.nn import LayerNorm


class DQN(nn.Module):
    def __init__(self, action_size, input_shape=(180,)):
        """
        Initialize the DQN model for LiDAR input.

        Parameters:
        - action_size (int): Number of possible actions.
        - input_shape (tuple): Shape of the input tensor (e.g., (180) for 1D LiDAR data).
        """
        super(DQN, self).__init__()

        input_size = input_shape[0]  # Assuming input_shape is (180)

        # Fully connected layers for feature extraction and Q-value prediction
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, 128),  # Second dense layer
            nn.ReLU(),
            nn.Linear(128, action_size)  # Output layer for Q-values
        )

        # Initialize weights
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        Initialize the weights of the layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, 180).

        Returns:
        - torch.Tensor: Q-values for each action, shape (batch_size, action_size).
        """
        return self.fc(x)  # Pass through fully connected layers
