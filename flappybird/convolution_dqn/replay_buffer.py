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
        - state (np.ndarray): Current state (processed frame).
        - action (int): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state (processed frame).
        - done (bool): Whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
        - Tuple[np.ndarray]: Batches of states, actions, rewards, next_states, and dones.
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"Not enough experiences to sample. Current size: {len(self.buffer)}")

        experiences = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def size(self):
        """
        Get the current size of the buffer.

        Returns:
        - int: Number of experiences in the buffer.
        """
        return len(self.buffer)
