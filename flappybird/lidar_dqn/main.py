import argparse
import torch
from train import train
from environment import FlappyBirdEnv
from dqn_agent import DQNAgent
from config import CONFIG


def test():
    """
    Evaluate the trained DQN agent in the Flappy Bird environment.
    """
    # Initialize the environment with LiDAR input
    env = FlappyBirdEnv(render_mode=CONFIG.get("render_mode", "human"))

    # Initialize the agent
    agent = DQNAgent(
        state_shape=CONFIG["lidar_shape"],  # Pass the flattened LiDAR input size directly
        action_size=env.action_space().n,
        config=CONFIG
    )

    # Load the trained model
    try:
        agent.load_model(CONFIG["model_save_path"])
        print(f"Model loaded from {CONFIG['model_save_path']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    # Ensure the agent uses a greedy policy during testing
    agent.epsilon = 0

    total_reward = 0
    episodes = 10  # Number of episodes to evaluate
    rewards = []

    for episode in range(episodes):
        state = env.reset().flatten()  # Ensure the state is flattened
        done = False
        episode_reward = 0

        while not done:
            # Select the best action (greedy policy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONFIG["device"])  # Add batch dim
            with torch.no_grad():
                action = torch.argmax(agent.online_net(state_tensor)).item()

            # Step the environment
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()  # Ensure the next state is flattened

            # Render if enabled
            if CONFIG.get("render", True):
                env.render()

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        total_reward += episode_reward
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")

    # Report the average reward
    avg_reward = total_reward / episodes
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")

    # Save rewards for analysis
    with open("testing_rewards.txt", "w") as f:
        for i, r in enumerate(rewards, 1):
            f.write(f"Episode {i}: Reward {r:.2f}\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")

    # Close the environment
    env.close()


def main():
    """
    Main entry point for the program.
    """
    parser = argparse.ArgumentParser(description="Train or test a DQN agent on Flappy Bird with LiDAR.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode to run: 'train' or 'test'.")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        train()
    elif args.mode == "test":
        print("Starting evaluation...")
        test()


if __name__ == "__main__":
    main()
