import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Initialize the Dueling DQN model.

        Parameters:
        - input_shape (Tuple[int, int, int]): Shape of the input (e.g., 1x84x84 for preprocessed grayscale frames).
        - num_actions (int): Number of possible actions.
        """
        super(DuelingDQN, self).__init__()

        print(f"[DEBUG] Initializing DuelingDQN with input shape: {input_shape} and num_actions: {num_actions}")

        # Ensure input_shape is (channels, height, width)
        assert input_shape[0] == 1, "Expected channel-first format for input shape, e.g., (1, 84, 84)"

        # Convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 7)
            nn.ReLU()
        )

        # Calculate the size of the flattened feature map
        conv_out_size = self._get_conv_out(input_shape)

        print(f"[DEBUG] Flattened convolutional output size: {conv_out_size}")

        # Fully connected layers for value and advantage streams
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Value stream outputs a single value
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)  # Advantage stream outputs action-specific values
        )

    def _get_conv_out(self, shape):
        """
        Helper function to calculate the output size of the convolutional layers.

        Parameters:
        - shape (Tuple[int, int, int]): Input shape.

        Returns:
        - int: Flattened size of the convolutional output.
        """
        with torch.no_grad():
            # Adjust dummy input shape to (batch_size, channels, height, width)
            dummy_input = torch.zeros(1, shape[0], shape[1], shape[2])  # (batch_size, channels, height, width)
            try:
                conv_out = self.conv(dummy_input)
            except Exception as e:
                raise
            return int(torch.prod(torch.tensor(conv_out.shape[1:])))

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Q-values for each action, shape (batch_size, num_actions).
        """
        try:
            # Pass input through convolutional layers
            conv_out = self.conv(x).view(x.size(0), -1)
        except Exception as e:
            raise

        # Compute value and advantage streams
        value = self.fc_value(conv_out)

        advantage = self.fc_advantage(conv_out)

        # Combine value and advantage streams to compute Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


if __name__ == "__main__":
    # Quick test to verify the network initialization and forward pass
    input_shape = (1, 84, 84)  # Channels-first grayscale input
    num_actions = 2
    model = DuelingDQN(input_shape, num_actions)

    dummy_input = torch.zeros(1, *input_shape)
    print("[INFO] Running a forward pass with dummy input...")
    output = model(dummy_input)
    print(f"[INFO] Output shape: {output.shape}")
