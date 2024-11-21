import torch
import numpy as np
from environment import FlappyBirdEnv
from dqn_agent import DQNAgent
import json


def test(config):
    """
    Test the trained DQN agent in the Flappy Bird environment.

    Parameters:
    - config (dict): Configuration dictionary containing hyperparameters and paths.
    """
    # Initialize the environment with "rgb_array" rendering mode
    env = FlappyBirdEnv(render_mode="rgb_array", frame_shape=config["frame_shape"])

    # Initialize the agent
    agent = DQNAgent(
        state_shape=(1, *config["frame_shape"]),
        num_actions=env.action_space().n,
        config=config
    )

    # Load the trained model
    try:
        agent.load_model(config["model_save_path"])
        print(f"Model loaded from {config['model_save_path']}")
    except FileNotFoundError:
        print(f"Error: Model file '{config['model_save_path']}' not found.")
        env.close()
        return

    total_rewards = []
    num_episodes = config.get("test_episodes", 5)  # Default to 5 episodes if not specified

    print(f"Starting evaluation over {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Add batch dimension to the state for the network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config["device"])

            # Select the best action using the trained model
            with torch.no_grad():
                q_values = agent.online_net(state_tensor)
                action = torch.argmax(q_values).item()

            # Step the environment
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            # Render the game to visually debug
            rendered_frame = env.render_frame()
            print(f"[DEBUG] Rendered frame shape: {rendered_frame.shape}")

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    # Compute average reward
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")

    # Save evaluation rewards
    with open(config["test_rewards_path"], "w") as f:
        for i, reward in enumerate(total_rewards, 1):
            f.write(f"Episode {i}: {reward:.2f}\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")

    # Close the environment
    env.close()
    print("Evaluation complete.")


if __name__ == "__main__":
    # Load configuration for testing
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Run testing
    test(config)
