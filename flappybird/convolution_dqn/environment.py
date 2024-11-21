import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from image_video import ImageVideo


class FlappyBirdEnv:
    def __init__(self, render_mode: Optional[str] = "rgb_array", frame_shape: Tuple[int, int] = (84, 84)):
        """
        Initialize the Flappy Bird environment wrapper.

        Parameters:
        - render_mode (Optional[str]): The rendering mode ("rgb_array" for image-based input or None for no rendering).
        - frame_shape (Tuple[int, int]): Shape to resize frames for the CNN (default: 84x84).
        """
        try:
            self.env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environment: {e}")

        self.frame_shape = frame_shape

    def reset(self) -> np.ndarray:
        """
        Reset the environment and preprocess the initial observation.

        Returns:
        - np.ndarray: The preprocessed initial game frame with shape (1, height, width).
        """
        observation, _ = self.env.reset()
        frame = self.render_frame()
        return self.preprocess_frame(frame)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment with the given action.

        Parameters:
        - action (int): Action to perform (0 = No Flap, 1 = Flap).

        Returns:
        - Tuple[np.ndarray, float, bool, dict]: Processed frame, reward, done flag, and info.
        """
        _, reward, done, _, info = self.env.step(action)
        frame = self.render_frame()
        if not ImageVideo.training:
            image = Image.fromarray(frame)
            ImageVideo.add_image(image)
        processed_frame = self.preprocess_frame(frame)
        return processed_frame, reward, done, info

    def render_frame(self) -> np.ndarray:
        """
        Render the current frame of the game.

        Returns:
        - np.ndarray: The raw RGB frame of the game.
        """
        frame = self.env.render()
        if not isinstance(frame, np.ndarray):
            print("[ERROR] Rendered frame is not a valid numpy array.")
            raise ValueError("Rendered frame is not a valid numpy array.")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"[ERROR] Rendered frame shape: {frame.shape} (Expected: 3D RGB image)")
        return frame

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame for CNN input by resizing, converting to grayscale, and adding a channel dimension.

        Parameters:
        - frame (np.ndarray): The raw RGB frame of the game.

        Returns:
        - np.ndarray: The preprocessed frame with shape (1, height, width).
        """
        if not isinstance(frame, np.ndarray):
            print("[ERROR] Frame is not a numpy array.")
            raise ValueError("Frame is not a numpy array.")
        if frame.shape[:2] != (512, 288):
            print(f"[DEBUG] Frame shape before preprocessing: {frame.shape} (Expected: 512x288x3)")

        # Convert to grayscale
        grayscale_frame = Image.fromarray(frame).convert("L")
        # Resize the frame
        resized_frame = grayscale_frame.resize(self.frame_shape, Image.BICUBIC)
        # Normalize pixel values to [0, 1]
        normalized_frame = np.array(resized_frame, dtype=np.float32) / 255.0
        # Add channel dimension (1, height, width)
        processed_frame = np.expand_dims(normalized_frame, axis=0)

        if processed_frame.shape != (1, self.frame_shape[0], self.frame_shape[1]):
            print(f"[ERROR] Preprocessed frame shape: {processed_frame.shape} (Expected: (1, {self.frame_shape[0]}, {self.frame_shape[1]}))")

        return processed_frame

    def render(self) -> None:
        """
        Render the current state of the environment.
        """
        self.env.render()

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.env.close()

    def action_space(self) -> gym.spaces.Discrete:
        """
        Get the action space of the environment.

        Returns:
        - gym.spaces.Discrete: The action space object.
        """
        return self.env.action_space

    def observation_space(self) -> Tuple[int, int, int]:
        """
        Get the observation space of the environment.

        Returns:
        - Tuple[int, int, int]: The shape of the processed frames (channels-first format).
        """
        return (1, self.frame_shape[0], self.frame_shape[1])  # Channels-first grayscale images


if __name__ == "__main__":
    # Quick test to verify functionality
    env = FlappyBirdEnv()
    frame = env.reset()
    print(f"Processed frame shape: {frame.shape}")  # Expected shape: (1, 84, 84)
    env.close()
