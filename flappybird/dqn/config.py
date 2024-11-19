import torch

CONFIG = {
    # Environment configuration
    "lidar_shape": (180,),  # Use the raw 1D LiDAR readings
    "replay_buffer_capacity": 10000,  # Increased capacity for diverse experiences

    # Agent configuration
    "batch_size": 128,  # Larger batch size for better gradient estimation
    "learning_rate": 0.0001,  # Learning rate for the optimizer
    "gamma": 0.99,  # Discount factor for future rewards
    "epsilon_start": 1.0,  # Initial epsilon value for exploration
    "epsilon_decay": 0.9995,  # Faster epsilon decay to transition from exploration to exploitation
    "epsilon_min": 0.01,  # Minimum epsilon value for exploration

    # Training configuration
    "num_episodes": 1000,  # Number of episodes to train the agent
    "target_update_frequency": 10,  # Update the target network every 10 episodes
    "save_frequency": 50,  # Save the model every 50 episodes

    # Paths
    "model_save_path": "dqn_flappy_bird_lidar.pth",  # Path to save the trained model
    "model_load_path": "dqn_flappy_bird_lidar.pth",  # Path to load a pre-trained model for testing

    # Rendering options
    "render": True,  # Enable rendering during training
    "render_mode": "human",  # Rendering mode ("human" for visualization, None for no rendering)

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
}
