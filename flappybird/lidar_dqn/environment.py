import gymnasium
import flappy_bird_gymnasium
import numpy as np
from typing import Optional, Tuple


class FlappyBirdEnv:
    def __init__(self, render_mode: Optional[str] = "human", lidar_shape: Optional[Tuple[int]] = None):
        """
        Initialize the Flappy Bird environment wrapper with LiDAR input.

        Parameters:
        - render_mode (Optional[str]): The rendering mode ("human" for visualization or None for no rendering).
        - lidar_shape (Optional[Tuple[int]]): Desired shape to reshape the LiDAR data for further processing. Defaults to (180,).
        """
        try:
            self.env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FlappyBird-v0 environment: {e}")

        # Set default LiDAR shape if none provided
        self.lidar_shape = lidar_shape or (180,)

        # Validate observation space dimensions
        obs_shape = self.env.observation_space.shape
        if obs_shape != (180,):
            raise ValueError(
                f"Unexpected observation space shape {obs_shape}. Expected (180,). Ensure the environment is configured correctly."
            )

    def reset(self) -> np.ndarray:
        """
        Reset the environment and preprocess the initial observation.

        Returns:
        - np.ndarray: The preprocessed initial LiDAR observation.
        """
        observation, _ = self.env.reset()
        return self.preprocess_lidar(observation)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment with the given action.

        Parameters:
        - action (int): Action to perform (0 = No Flap, 1 = Flap).

        Returns:
        - Tuple[np.ndarray, float, bool, dict]: Processed LiDAR observation, reward, done flag, and info.
        """
        observation, reward, done, _, info = self.env.step(action)
        print(observation)
        processed_observation = self.preprocess_lidar(observation)
        return processed_observation, reward, done, info

    def preprocess_lidar(self, lidar_data: np.ndarray) -> np.ndarray:
        """
        Preprocess LiDAR data by normalizing and reshaping (if lidar_shape is specified).

        Parameters:
        - lidar_data (np.ndarray): Raw LiDAR data from the environment.

        Returns:
        - np.ndarray: Preprocessed LiDAR data ready for further processing.
        """
        #print("Raw LiDAR data:", lidar_data)  # Debug print to inspect raw data

        # Ensure maximum value is not zero to prevent division by zero
        max_value = np.max(lidar_data)
        if max_value <= 0:  # Handle invalid LiDAR data
            max_value = 1.0  # Avoid division by zero
            print("Warning: LiDAR data contains all zeros or negatives. Normalizing with max_value=1.0.")

        # Normalize LiDAR readings to [0, 1]
        normalized_data = lidar_data / max_value

        # Reshape the normalized data if lidar_shape is provided
        if self.lidar_shape:
            return normalized_data.reshape(self.lidar_shape).astype(np.float32)

        return normalized_data.astype(np.float32)

    def render(self) -> None:
        """Render the current state of the environment."""
        self.env.render()

    def close(self) -> None:
        """Close the environment and release resources."""
        self.env.close()

    def action_space(self) -> gymnasium.spaces.Discrete:
        """
        Get the action space of the environment.

        Returns:
        - gymnasium.spaces.Discrete: The action space object.
        """
        return self.env.action_space

    def observation_space(self) -> gymnasium.spaces.Box:
        """
        Get the observation space of the environment.

        Returns:
        - gymnasium.spaces.Box: The observation space object.
        """
        return self.env.observation_space
