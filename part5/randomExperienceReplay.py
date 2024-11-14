import numpy as np
import torch
from Gridworld import Gridworld
import random
from collections import deque
from matplotlib import pylab as plt

# Define network architecture parameters
L1 = 64
L2 = 150
L3 = 100
L4 = 4

# Define the neural network model
model = torch.nn.Sequential(
    torch.nn.Linear(L1, L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4),
)

# Loss function and optimizer
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Q-learning parameters
gamma = 0.9
epsilon = 1.0

# Replay memory settings
mem_size = 1000  # Maximum size of the replay memory
batch_size = 200  # Mini-batch size for training
replay = deque(maxlen=mem_size)  # Replay memory implemented as a deque

# Action mapping
action_set = {
    0: 'u',  # Up
    1: 'd',  # Down
    2: 'l',  # Left
    3: 'r'  # Right
}

# Training loop
epochs = 5000
losses = []  # List to store loss values
max_moves = 50  # Maximum number of moves per game
for i in range(epochs):
    # Initialize a new Gridworld game
    game = Gridworld(size=4, mode='random')

    # Initial state and slight random noise for exploration
    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state1 = torch.from_numpy(state1_).float()

    status = 1  # Status to keep track of game progress
    mov = 0  # Move counter
    while status == 1:
        mov += 1

        # Forward pass to get Q-values for current state
        qval = model(state1)
        qval_ = qval.data.numpy()

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)  # Exploration: random action
        else:
            action_ = np.argmax(qval_)  # Exploitation: choose best action

        # Convert action index to actual action
        action = action_set[action_]

        # Execute the selected action in the environment
        game.makeMove(action)

        # Get the new state and add slight random noise
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state2 = torch.from_numpy(state2_).float()

        # Get reward and check if the game has ended
        reward = game.reward()
        done = True if reward > 0 else False  # True if game won, else False

        # Store the experience in replay memory
        exp = (state1, action_, reward, state2, done)
        replay.append(exp)

        # Move to the new state
        state1 = state2

        # Mini-batch training from replay memory if it contains enough samples
        if len(replay) > batch_size:
            # Sample a mini-batch from replay memory
            minibatch = random.sample(replay, batch_size)

            # Separate each element of experience into mini-batch tensors
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            # Compute predicted Q-values for the mini-batch of states
            Q1 = model(state1_batch)

            # Compute target Q-values for the mini-batch of next states (no gradients)
            with torch.no_grad():
                Q2 = model(state2_batch)

            # Calculate target Q-value (Y) for each experience in the mini-batch
            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])

            # Gather the predicted Q-values for the chosen actions
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            # Calculate the loss and perform backpropagation
            loss = loss_fn(X, Y.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the loss value
            losses.append(loss.item())

        # Check if the game is over due to reaching a terminal state or move limit
        if reward != -1 or mov > max_moves:
            status = 0  # End game if terminal state or move limit reached
            mov = 0  # Reset move counter for next game

    # Decay epsilon for exploration-exploitation balance
    if epsilon > 0.1:
        epsilon -= (1 / epochs)

# Convert losses to a numpy array for plotting
losses = np.array(losses)

# Plotting the loss over training epochs
plt.figure(figsize=(10, 7))
plt.plot(losses)
plt.xlabel("Training Steps", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.savefig("randomExperienceReplay.png")
