import numpy as np
from time import sleep


class ValueIteration:

    def __init__(self):
        self.row = 9
        self.column = 9
        self.values = np.zeros((self.row, self.column))
        self.values = self.values
        self.step = -1
        self.goal = 10
        self.tolerance = 0.001
        self.goal_indices = (4, 4)
        self.discount = 0.8
        self.withinTolerance = False

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
            if self.withinTolerance:
                break
            print(self.values)
            print("------------------------------------------")
            sleep(1)
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
                    value = max(actionReward)
                    if abs(value - self.values[i, j]) > self.tolerance:
                        self.withinTolerance = False
                    self.values[i, j] = value


if __name__ == "__main__":
    valueIter = ValueIteration()
    valueIter.doBellman()
    print(valueIter.values)
