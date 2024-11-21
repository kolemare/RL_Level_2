import time
import torch
import json
import os
from environment import FlappyBirdEnv
from dqn_agent import DQNAgent


def train(config):
    """
    Train the DQN agent in the Flappy Bird environment.

    Parameters:
    - config (dict): Configuration dictionary containing hyperparameters and paths.
    """
    # Initialize the environment
    env = FlappyBirdEnv(
        render_mode=config["render_mode"],
        frame_shape=tuple(config["frame_shape"])
    )

    # Ensure `state_shape` is in channel-first format (1, height, width)
    state_shape = (1, *config["frame_shape"])

    # Initialize the agent
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=env.action_space().n,
        config=config
    )

    # Create model save directory if it doesn't exist
    model_dir = os.path.dirname(config["model_save_path"])
    if model_dir:  # Check if a directory is specified
        os.makedirs(model_dir, exist_ok=True)

    # Track training progress
    total_rewards = []
    best_reward = float('-inf')
    start_time = time.time()

    # Training loop
    for episode in range(config["num_episodes"]):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select an action
            action = agent.select_action(state)

            # Perform the action
            next_state, reward, done, _ = env.step(action)

            # Store experience in the replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

            # Update the current state
            state = next_state
            total_reward += reward

        # Track episode rewards
        total_rewards.append(total_reward)
        best_reward = max(best_reward, total_reward)

        # Log progress
        if episode % config["log_frequency"] == 0:
            print(
                f"Episode {episode}/{config['num_episodes']}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Best Reward: {best_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Save the model periodically
        if episode % config["save_frequency"] == 0:
            try:
                agent.save_model(config["model_save_path"])
            except Exception as e:
                print(f"Error saving model: {e}")

    # Save final rewards
    with open("training_rewards.txt", "w") as f:
        f.writelines([f"{reward}\n" for reward in total_rewards])

    # Save the final model
    try:
        agent.save_model(config["model_save_path"])
    except Exception as e:
        print(f"Error saving final model: {e}")

    print(f"Training complete in {time.time() - start_time:.2f} seconds.")
    env.close()
