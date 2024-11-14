import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt

L1 = 64
L2 = 150
L3 = 100
L4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(L1,L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2,L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3,L4),
)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
epsilon = 1.0

action_set = {
    0:'u',
    1:'d',
    2:'l',
    3:'r'
}

if __name__ == '__main__':
    epochs = 1000
    losses = []
    for i in range(epochs):
        game = Gridworld(size=4,mode='static')
        state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state1 = torch.from_numpy(state_).float()
        status = 1
        while (status == 1):
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            action = action_set[action_]
            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
            state2  = torch.from_numpy(state2_).float()
            reward = game.reward()
            with torch.no_grad():
                newQ = model(state2.reshape(1,64))
            maxQ = torch.max(newQ)

            if reward == -1:
                Y = reward + (gamma*maxQ)
            else:
                Y = reward
            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action_]
            loss = loss_fn(X,Y)
            print(i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state1 = state2
            if reward != -1:
                status = 0
        losses.append(loss.item())
        if epsilon > 0.1:
            epsilon -= (1/epochs)

    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs",fontsize=11)
    plt.ylabel("Loss",fontsize=11)
    plt.show()