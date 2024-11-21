import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer
from dueling_dqn import DuelingDQN


class DQNAgent:
    def __init__(self, state_shape, num_actions, config):
        """
        Initialize the DQN agent.

        Parameters:
        - state_shape (Tuple[int, int, int]): Shape of the input frames.
        - num_actions (int): Number of possible actions.
        - config (dict): Configuration parameters.
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.config = config

        # Epsilon parameters for exploration
        self.epsilon = config["epsilon_start"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config["replay_buffer_capacity"],
            batch_size=config["batch_size"]
        )

        # Online and target networks
        self.online_net = DuelingDQN(state_shape, num_actions).to(config["device"])
        self.target_net = DuelingDQN(state_shape, num_actions).to(config["device"])
        self.target_net.eval()  # Target network is used for stable target calculations
        self.sync_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config["learning_rate"])

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Training step counter
        self.training_step = 0

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Parameters:
        - state (np.ndarray): Current game state (preprocessed frame).

        Returns:
        - int: Selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Random action (exploration)
        else:
            # Convert state to a PyTorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config["device"])
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return torch.argmax(q_values).item()  # Best action (exploitation)

    def train(self):
        """
        Train the online network using a batch of experiences from the replay buffer.
        """
        if self.replay_buffer.size() < self.config["batch_size"]:
            return  # Not enough samples to train

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.config["device"])
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.config["device"])
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config["device"])
        next_states = torch.FloatTensor(next_states).to(self.config["device"])
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config["device"])

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]
            target_q_values = rewards + self.config["gamma"] * max_next_q_values * (1 - dones)

        # Compute current Q-values
        current_q_values = self.online_net(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Update training step counter
        self.training_step += 1

        # Sync target network periodically
        if self.training_step % self.config["target_update_frequency"] == 0:
            self.sync_target_network()

    def sync_target_network(self):
        """
        Sync the target network with the online network.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self, path):
        """
        Save the online network to the specified file path.
        """
        torch.save(self.online_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the model from the specified file path.
        """
        self.online_net.load_state_dict(torch.load(path))
        self.online_net.eval()
        print(f"Model loaded from {path}")
