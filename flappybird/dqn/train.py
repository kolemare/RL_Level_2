import time
import torch
from environment import FlappyBirdEnv
from dqn_agent import DQNAgent
from config import CONFIG


def train():
    """
    Train the DQN agent in the Flappy Bird environment.
    """
    # Initialize the environment with LiDAR input
    env = FlappyBirdEnv(render_mode=CONFIG.get("render_mode", None), lidar_shape=CONFIG["lidar_shape"])

    # Flattened input size for fully connected layers
    lidar_input_size = CONFIG["lidar_shape"][0]

    # Initialize the agent
    agent = DQNAgent(
        state_shape=(lidar_input_size,),  # Flattened LiDAR input
        action_size=env.action_space().n,
        config=CONFIG
    )

    # Tracking progress
    total_rewards = []
    best_reward = float('-inf')

    # Training loop
    for episode in range(CONFIG["num_episodes"]):
        state = env.reset().flatten()  # Flatten the LiDAR input
        total_reward = 0
        done = False

        while not done:
            # Select an action
            action = agent.select_action(state)

            # Take the action and observe the result
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()  # Flatten the next state

            # Store the experience in the replay buffer
            agent.store_experience(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

            # Update the state
            state = next_state
            total_reward += reward

            # Optional rendering
            if CONFIG.get("render", False):
                env.render()

        # Update the target network periodically
        if episode % CONFIG["target_update_frequency"] == 0:
            agent.update_target_network()

        # Save the model periodically
        if episode % CONFIG["save_frequency"] == 0:
            try:
                agent.save_model(CONFIG["model_save_path"])
            except Exception as e:
                print(f"Error saving model: {e}")

        # Log the reward for this episode
        total_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        if episode % 10 == 0:  # Periodic logging
            print(
                f"Episode {episode}/{CONFIG['num_episodes']}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Best Reward: {best_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Close the environment
    env.close()

    # Save rewards
    with open("training_rewards.txt", "w") as f:
        for reward in total_rewards:
            f.write(f"{reward}\n")

    # Final save of the model
    try:
        agent.save_model(CONFIG["model_save_path"])
    except Exception as e:
        print(f"Error saving model: {e}")

    print("Training complete. Model saved.")


if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
