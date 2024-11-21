import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, action_size, input_shape=(180,)):
        """
        Initialize the Dueling DQN model for LiDAR input.

        Parameters:
        - action_size (int): Number of possible actions.
        - input_shape (tuple): Shape of the input tensor (e.g., (180,) for 1D LiDAR data).
        """
        super(DuelingDQN, self).__init__()

        input_size = input_shape[0]  # Assuming input_shape is (180)

        # Shared feature extraction layer
        self.shared_fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Value stream (outputs a single scalar representing the state value)
        self.value_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for the state value
        )

        # Advantage stream (outputs action advantages)
        self.advantage_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)  # Outputs one value per action
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
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        - torch.Tensor: Q-values for each action, shape (batch_size, action_size).
        """
        shared_features = self.shared_fc(x)

        # Compute state value and action advantages
        state_value = self.value_fc(shared_features)
        action_advantages = self.advantage_fc(shared_features)

        # Combine state value and action advantages to compute Q-values
        q_values = state_value + (action_advantages - action_advantages.mean(dim=1, keepdim=True))
        return q_values
