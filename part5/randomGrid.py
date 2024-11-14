import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt

# Network architecture parameters for hidden layers and output layer
L1 = 64  # Input layer size (size of flattened grid state, 4x4 grid -> 16x4 = 64)
L2 = 150  # First hidden layer size
L3 = 100  # Second hidden layer size
L4 = 4  # Output layer size (4 possible actions)

# Defining the neural network model with 2 hidden layers and ReLU activations
model = torch.nn.Sequential(
    torch.nn.Linear(L1, L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4),
)

# Mean Squared Error Loss for Q-learning loss calculation
loss_fn = torch.nn.MSELoss()

# Adam optimizer for updating model parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Q-learning parameters
gamma = 0.9  # Discount factor for future rewards
epsilon = 1.0  # Initial epsilon value for epsilon-greedy policy (exploration-exploitation balance)

# Mapping for action indices to actual movements in the environment
action_set = {
    0: 'u',  # Up
    1: 'd',  # Down
    2: 'l',  # Left
    3: 'r'  # Right
}

# Main loop for running Q-learning over multiple episodes (epochs)
if __name__ == '__main__':
    epochs = 1000
    losses = []  # List to store loss values over epochs for plotting

    for i in range(epochs):
        # Initialize a new Gridworld environment for each epoch
        game = Gridworld(size=4, mode='random')

        # Initial state: get the grid state as a 1x64 vector with slight noise for exploration
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state1 = torch.from_numpy(state_).float()  # Convert state to a torch tensor

        status = 1  # Variable to keep track of the game status (1 means ongoing)

        # Run the game until a terminal state (win or loss) is reached
        while status == 1:
            # Forward pass to get Q-values for current state
            qval = model(state1)
            qval_ = qval.data.numpy()  # Convert Q-values to numpy array

            # Choose an action using epsilon-greedy strategy
            if random.random() < epsilon:
                # Exploration: choose a random action
                action_ = np.random.randint(0, 4)
            else:
                # Exploitation: choose the action with the highest Q-value
                action_ = np.argmax(qval_)

            # Map the chosen action index to the actual action
            action = action_set[action_]

            # Execute the action in the game environment
            game.makeMove(action)

            # Get the new state after the action and add slight noise for exploration
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2 = torch.from_numpy(state2_).float()  # Convert new state to torch tensor

            # Get reward for the action taken
            reward = game.reward()

            # Calculate the target Q-value (Y) based on the reward and max future Q-value
            with torch.no_grad():
                newQ = model(state2.reshape(1, 64))  # Forward pass to get Q-values for next state
            maxQ = torch.max(newQ)  # Get the maximum Q-value for the new state

            # Calculate target Q-value (Y) based on whether we reached terminal or non-terminal state
            if reward == -1:
                # Non-terminal state: use discounted max future Q-value
                Y = reward + (gamma * maxQ)
            else:
                # Terminal state: target is just the reward received
                Y = reward
            Y = torch.Tensor([Y]).detach()  # Convert target to torch tensor and detach from computation graph

            # Current Q-value (X) for the chosen action
            X = qval.squeeze()[action_]

            # Calculate loss between target Q-value (Y) and predicted Q-value (X)
            loss = loss_fn(X, Y)
            print(i, loss.item())  # Print loss for monitoring

            # Backpropagation step
            optimizer.zero_grad()  # Clear gradients from previous step
            loss.backward()  # Compute gradients for the network
            optimizer.step()  # Update model parameters

            # Move to the new state
            state1 = state2

            # If the reward is not -1 (i.e., terminal state), end the game
            if reward != -1:
                status = 0

        # Track the loss for plotting later
        losses.append(loss.item())

        # Decay epsilon to reduce exploration over time
        if epsilon > 0.1:
            epsilon -= (1 / epochs)

    # Plot the loss over epochs
    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.savefig("randomGrid.png")
