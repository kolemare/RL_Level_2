import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_shape, action_size, config):
        """
        Initialize the DQN agent.

        Parameters:
        - state_shape (int): Size of the flattened input state (e.g., 180 for LiDAR readings).
        - action_size (int): Number of possible actions.
        - config (dict): Hyperparameters for the agent.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.config = config

        # Epsilon parameters for exploration
        self.epsilon = config["epsilon_start"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config["replay_buffer_capacity"],
            batch_size=config["batch_size"]
        )

        # Online and target networks
        self.online_net = DQN(action_size, input_shape=state_shape).to(config["device"])
        self.target_net = DQN(action_size, input_shape=state_shape).to(config["device"])
        self.target_net.eval()  # Ensure the target network is in evaluation mode
        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config["learning_rate"])

        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber Loss

        # Metrics tracking
        self.training_step = 0
        self.loss_history = []

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Parameters:
        - state (np.ndarray): Current state (LiDAR data).

        Returns:
        - int: Selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        else:
            # Convert state to PyTorch tensor and ensure correct dimensions
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config["device"])  # Add batch dim
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay buffer.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self):
        """
        Train the online network using experiences from the replay buffer.
        """
        if self.replay_buffer.size() < self.config["batch_size"]:
            return  # Not enough data to train

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.config["device"])
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.config["device"])
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config["device"])
        next_states = torch.FloatTensor(next_states).to(self.config["device"])
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config["device"])

        # Compute target Q-values
        with torch.no_grad():
            target_q_values = rewards + self.config["gamma"] * torch.max(
                self.target_net(next_states), dim=1, keepdim=True
            )[0] * (1 - dones)

        # Compute current Q-values
        current_q_values = self.online_net(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Track loss and metrics
        self.loss_history.append(loss.item())
        self.training_step += 1

        # Log metrics periodically
        if self.training_step % 100 == 0:  # Log every 100 training steps
            avg_loss = np.mean(self.loss_history[-100:])
            print(
                f"Training Step: {self.training_step} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {self.epsilon:.3f}"
            )

    def update_target_network(self):
        """
        Update the target network weights with the online network weights.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self, path):
        """
        Save the online network to the specified path.
        """
        try:
            torch.save(self.online_net.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, path):
        """
        Load the model weights from the specified path.
        """
        try:
            self.online_net.load_state_dict(torch.load(path))
            self.online_net.eval()
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
