import argparse
import json
from train import train
from test import test


def main():
    """
    Main entry point for training or testing the DQN agent.
    """
    parser = argparse.ArgumentParser(description="Train or test a DQN agent for Flappy Bird.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration once and pass it as a dictionary
    with open(args.config, "r") as f:
        config = json.load(f)

    print(f"Running in {args.mode} mode with configuration:")
    print(json.dumps(config, indent=4))

    # Call train or test with the loaded config dictionary
    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)


if __name__ == "__main__":
    main()
