import numpy as np
from time import sleep
import random


def getActionMapping(move):
    if 1 == move:
        return [1, 0]  # down
    if 2 == move:
        return [-1, 0]  # up
    if 3 == move:
        return [0, 1]  # right
    if 4 == move:
        return [0, -1]  # left


def getMoveMapping(move):
    if [1, 0] == move:
        return 1  # down
    if [-1, 0] == move:
        return 2  # up
    if [0, 1] == move:
        return 3  # right
    if [0, -1] == move:
        return 4  # left


class GPI:

    def __init__(self):
        self.row = 5
        self.column = 5
        self.values = np.zeros((self.row, self.column))
        self.policy = np.random.randint(1, 5, (self.row, self.column))
        self.policyVis = np.full((self.row, self.column), '', dtype=str)
        self.step = -1
        self.goal = 10
        self.tolerance = 0.001
        self.goal_indices = [2, 2]
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

    def getPolicyVisMapping(self, i, j):
        if self.policy[i, j] == 1:
            return '↓'
        elif self.policy[i, j] == 2:
            return '↑'
        elif self.policy[i, j] == 3:
            return '→'
        elif self.policy[i, j] == 4:
            return '←'

    def printPolicy(self):
        for i in range(0, self.row):
            for j in range(0, self.column):
                if [i, j] == self.goal_indices:
                    self.policyVis[i, j] = 'G'
                    continue
                self.policyVis[i, j] = self.getPolicyVisMapping(i, j)
        print(self.policyVis)

    def fixInvalidMoves(self):
        for i in range(0, self.row):
            for j in range(0, self.column):
                if not any(np.array_equal(getActionMapping(self.policy[i, j]), arr) for arr in
                           self.checkAvailableStates(i, j)):
                    self.policy[i, j] = getMoveMapping(random.choice(self.checkAvailableStates(i, j)))

    def calculateActionValue(self, i, j, action):
        # print('Indices: ' + str(i + action[0]) + ':' + str(j + action[1]))
        if [i + action[0], j + action[1]] == self.goal_indices:
            return self.step + self.goal + self.discount * self.values[i + action[0], j + action[1]]
        else:
            return self.step + self.discount * self.values[i + action[0], j + action[1]]

    def doTheGPI(self):
        policyUpdated = True
        while True:
            self.printPolicy()
            print("------------------------------------------")
            print(self.values)
            print("------------------------------------------")
            sleep(1)
            if not policyUpdated:
                break
            policyUpdated = False
            for i in range(0, self.row):
                for j in range(0, self.column):
                    if self.isStateTerminal(i, j):
                        continue
                    value = 0
                    if (i + getActionMapping(self.policy[i, j])[0], j + getActionMapping(self.policy[i, j])[1]) == self.goal_indices:
                        value += self.step + self.goal + self.discount * self.values[i + getActionMapping(self.policy[i, j])[0], j + getActionMapping(self.policy[i, j])[1]]
                    else:
                        value += self.step + self.discount * self.values[i + getActionMapping(self.policy[i, j])[0], j + getActionMapping(self.policy[i, j])[1]]
                    self.values[i, j] = value

            policy_update_threshold_broken = False
            for i in range(0, self.row):
                for j in range(0, self.column):
                    if self.isStateTerminal(i, j):
                        continue
                    actionRewards = []
                    actions = self.checkAvailableStates(i, j)
                    for action in actions:
                        actionRewards.append(self.calculateActionValue(i, j, action))
                    maxIndex = np.argmax(actionRewards)
                    newAction = getMoveMapping(actions[maxIndex])
                    if self.policy[i, j] != newAction:
                        policyUpdated = True
                    self.policy[i, j] = newAction


if __name__ == "__main__":
    gpi = GPI()
    gpi.fixInvalidMoves()
    gpi.doTheGPI()
