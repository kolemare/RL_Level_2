import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt

# Network architecture parameters
L1 = 64
L2 = 150
L3 = 100
L4 = 4

# Defining the neural network model
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

# Mapping actions to movements
action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}

# Training loop
if __name__ == '__main__':
    epochs = 1000
    losses = []
    for i in range(epochs):
        # Initialize new Gridworld game
        game = Gridworld(size=4, mode='static')

        # Initial state
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state1 = torch.from_numpy(state_).float()

        status = 1
        while status == 1:
            # Forward pass to get Q-values
            qval = model(state1)
            qval_ = qval.data.numpy()

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            action = action_set[action_]

            # Execute action
            game.makeMove(action)

            # New state and reward
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()

            # Q-learning target computation
            with torch.no_grad():
                newQ = model(state2.reshape(1, 64))
            maxQ = torch.max(newQ)

            if reward == -1:
                Y = reward + (gamma * maxQ)
            else:
                Y = reward
            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action_]

            # Loss and backpropagation
            loss = loss_fn(X, Y)
            print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state1 = state2
            if reward != -1:
                status = 0

        # Store loss for plotting
        losses.append(loss.item())

        # Decrease epsilon after each epoch
        if epsilon > 0.1:
            epsilon -= (1 / epochs)

    # Plot training loss
    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.show()


    # Test the model after training
    def test_model(model, mode='static', display=True):
        i = 0
        test_game = Gridworld(mode=mode)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()

        if display:
            print("Initial State:")
            print(test_game.display())

        status = 1  # Game status
        while status == 1:
            # Predict Q-values for the current state
            qval = model(state)
            qval_ = qval.data.numpy()

            # Choose action with the highest Q-value
            action_ = np.argmax(qval_)
            action = action_set[action_]

            if display:
                print('Move #: %s; Taking action: %s' % (i, action))

            # Execute action in the game
            test_game.makeMove(action)

            # Update state
            state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state = torch.from_numpy(state_).float()

            if display:
                print(test_game.display())

            # Get reward
            reward = test_game.reward()

            # Check if game is over
            if reward != -1:
                if reward > 0:
                    status = 2  # Game won
                    if display:
                        print("Game won! Reward: %s" % (reward,))
                else:
                    status = 0  # Game lost
                    if display:
                        print("Game LOST. Reward: %s" % (reward,))

            i += 1
            if i > 15:  # Limit moves to avoid infinite loops
                if display:
                    print("Game lost; too many moves.")
                break

        win = True if status == 2 else False
        return win


    # Example call to test_model after training
    test_win = test_model(model, display=True)
    print("Test Game Result:", "Win" if test_win else "Loss")
