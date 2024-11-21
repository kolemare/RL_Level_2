import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        """
        Initialize the replay buffer.

        Parameters:
        - capacity (int): Maximum number of experiences to store.
        - batch_size (int): Number of experiences to sample for training.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters:
        - state (np.ndarray): The current LiDAR state, shape (lidar_input_size,).
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.ndarray): The next LiDAR state, shape (lidar_input_size,).
        - done (bool): Whether the episode is done.
        """
        # Ensure correct data type for LiDAR states
        state = np.array(state, dtype=np.float32).flatten()  # Flatten the state
        next_state = np.array(next_state, dtype=np.float32).flatten()  # Flatten the next state

        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
        - tuple: Batches of states, actions, rewards, next_states, and dones.
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"Not enough experiences in buffer to sample a batch. Current size: {len(self.buffer)}")

        experiences = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            np.array(states, dtype=np.float32),  # Flattened states, shape (batch_size, lidar_input_size)
            np.array(actions, dtype=np.int64),  # Actions
            np.array(rewards, dtype=np.float32),  # Rewards
            np.array(next_states, dtype=np.float32),  # Flattened next states
            np.array(dones, dtype=np.bool_)  # Done flags
        )

    def size(self):
        """
        Get the current size of the buffer.

        Returns:
        - int: The number of experiences currently in the buffer.
        """
        return len(self.buffer)
