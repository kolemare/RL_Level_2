import torch

CONFIG = {
    # Environment configuration
    "lidar_shape": (180,),  # Use the raw 1D LiDAR readings
    "replay_buffer_capacity": 100000,  # Increased capacity for diverse experiences

    # Agent configuration
    "batch_size": 128,  # Adjusted for stable gradient estimation and faster updates
    "learning_rate": 0.0001,  # Reduced for finer updates with a deeper model
    "gamma": 0.99,  # Discount factor for future rewards
    "epsilon_start": 1.0,  # Initial epsilon value for exploration
    "epsilon_decay": 0.9995,  # Faster epsilon decay for quicker exploitation
    "epsilon_min": 0.05,  # Minimum epsilon value for exploration

    # Training configuration
    "num_episodes": 30000,  # Number of episodes to train the agent
    "target_update_frequency": 10,  # More frequent updates for the target network
    "save_frequency": 50,  # Save the model every 50 episodes

    # Paths
    "model_save_path": "dqn_flappy_bird_lidar.pth",  # Path to save the trained model
    "model_load_path": "dqn_flappy_bird_lidar.pth",  # Path to load a pre-trained model for testing

    # Rendering options
    "render": False,  # Disable rendering during training for faster execution
    "render_mode": "human",  # Rendering mode ("human" for visualization, None for no rendering)

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
}
