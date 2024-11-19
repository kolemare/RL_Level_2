#!/bin/bash

# Define the name of the virtual environment
VENV_NAME="venv"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install required libraries
echo "Installing required libraries..."
pip install networkx matplotlib torch ipython numpy opencv-python gymnasium flappy-bird-gymnasium

# Install flappy-bird-gym
echo "Installing flappy-bird-gym..."
pip install git+https://github.com/Talendar/flappy-bird-gym.git

# Verify installation
echo "Verifying installation..."
python -c "import gym, flappy_bird_gym; print('Installation successful!')"

# Deactivate the virtual environment
deactivate

echo "Setup complete. Activate the virtual environment with 'source $VENV_NAME/bin/activate'"
