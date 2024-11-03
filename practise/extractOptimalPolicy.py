import numpy as np
from time import sleep


class ValueIteration:

    def __init__(self):
        self.row = 9
        self.column = 9
        self.values = np.zeros((self.row, self.column))
        self.policy = np.full((self.row, self.column), '', dtype=str)
        self.step = -1
        self.goal = 10
        self.tolerance = 0.001
        self.goal_indices = (4, 4)
        self.discount = 0.9
        self.withinTolerance = False
        self.policy[self.goal_indices[0], self.goal_indices[1]] = 'G'

    def checkAvailableStates(self, i, j):
        actions = []
        if i < self.row - 1:
            actions.append([1, 0])  # down
        if i > 0:
            actions.append([-1, 0])  # up
        if j < self.column - 1:
            actions.append([0, 1])  # right
        if j > 0:
            actions.append([0, -1])  # left
        return actions

    def isStateTerminal(self, i, j):
        if i is self.goal_indices[0] and j is self.goal_indices[1]:
            return True
        return False

    def doBellman(self):
        while True:
            print(self.values)
            print("------------------------------------------")
            print(self.policy)
            print("------------------------------------------")
            sleep(1)
            if self.withinTolerance:
                break
            self.withinTolerance = True
            for i in range(0, self.row):
                for j in range(0, self.column):
                    actions = self.checkAvailableStates(i, j)
                    if self.isStateTerminal(i, j):
                        continue
                    actionReward = []
                    for item in actions:
                        if (i + item[0], j + item[1]) == self.goal_indices:
                            actionReward.append(self.step + self.goal + self.discount * self.values[i + item[0], j + item[1]])
                        else:
                            actionReward.append(self.step + self.discount * self.values[i + item[0], j + item[1]])
                    value = np.max(actionReward)
                    index = np.argmax(actionReward)
                    if actions[index] == [-1, 0]:  # up
                        self.policy[i, j] = '↑'
                    elif actions[index] == [1, 0]:  # down
                        self.policy[i, j] = '↓'
                    elif actions[index] == [0, 1]:  # right
                        self.policy[i, j] = '→'
                    elif actions[index] == [0, -1]:  # left
                        self.policy[i, j] = '←'

                    if abs(value - self.values[i, j]) > self.tolerance:
                        self.withinTolerance = False
                    self.values[i, j] = value


if __name__ == "__main__":
    valueIter = ValueIteration()
    valueIter.doBellman()
    print(valueIter.values)
