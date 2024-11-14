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

# Upgrade pip
pip install --upgrade pip
pip install networkx matplotlib torch ipython numpy

# Deactivate the virtual environment
deactivate

echo "Setup complete. Activate the virtual environment with 'source $VENV_NAME/bin/activate'"
