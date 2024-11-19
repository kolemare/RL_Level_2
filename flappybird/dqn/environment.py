import gymnasium
import flappy_bird_gymnasium
import numpy as np


class FlappyBirdEnv:
    def __init__(self, render_mode="human", lidar_shape=None):
        """
        Initialize the Flappy Bird environment wrapper with LiDAR input.

        Parameters:
        - render_mode (str): The rendering mode ("human" for visualization or None for no rendering).
        - lidar_shape (tuple or None): Desired shape to reshape the LiDAR data for further processing. If None, no reshaping is applied.
        """
        try:
            self.env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environment: {e}")
        self.lidar_shape = lidar_shape

    def reset(self):
        """
        Reset the environment and preprocess the initial observation.

        Returns:
        - np.ndarray: The preprocessed initial LiDAR observation.
        """
        observation, _ = self.env.reset()
        return self.preprocess_lidar(observation)

    def step(self, action):
        """
        Take a step in the environment with the given action.

        Parameters:
        - action (int): Action to perform (0 = No Flap, 1 = Flap).

        Returns:
        - np.ndarray: Preprocessed next LiDAR observation.
        - float: Reward for the step.
        - bool: Whether the episode is done.
        - dict: Additional info from the environment.
        """
        observation, reward, done, _, info = self.env.step(action)
        processed_observation = self.preprocess_lidar(observation)
        return processed_observation, reward, done, info

    def preprocess_lidar(self, lidar_data):
        """
        Preprocess LiDAR data by normalizing and reshaping (if lidar_shape is specified).

        Parameters:
        - lidar_data (np.ndarray): Raw LiDAR data from the environment.

        Returns:
        - np.ndarray: Preprocessed LiDAR data ready for further processing.
        """
        # Ensure maximum value is not zero to prevent division by zero
        max_value = np.max(lidar_data)
        if max_value <= 0:  # Handle edge case where max_value is 0 or negative
            max_value = 1.0

        # Normalize LiDAR readings to [0, 1]
        normalized_data = lidar_data / max_value

        # Reshape the normalized data if lidar_shape is provided
        if self.lidar_shape:
            reshaped_data = normalized_data.reshape(self.lidar_shape)
            return reshaped_data.astype(np.float32)

        # Otherwise, return the normalized data as is
        return normalized_data.astype(np.float32)

    def render(self):
        """
        Render the current state of the environment.
        """
        self.env.render()

    def close(self):
        """
        Close the environment and release resources.
        """
        self.env.close()

    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
        - gym.spaces.Discrete: The action space object.
        """
        return self.env.action_space

    def observation_space(self):
        """
        Get the observation space of the environment.

        Returns:
        - gym.spaces.Box: The observation space object.
        """
        return self.env.observation_space
