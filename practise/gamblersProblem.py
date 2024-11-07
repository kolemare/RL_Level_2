from time import sleep
import numpy as np
from contourpy.util.data import random
import random


class Gambler:

    def __init__(self):
        self.states = np.arange(0, 101)
        self.values = np.zeros(101)
        self.policy = []
        self.current_money = 40
        self.discount = 0.9
        for item in range(1, len(self.states)):
            self.policy.append(random.randint(1, min(item, 100 - item)))
        self.threshold = 0.001

    def gamble(self, money):
        if random.choice([True, False]):
            self.current_money = self.current_money + money
        else:
            self.current_money = self.current_money - money

    def generalizedPolicyIteration(self):
        thresholdBroken = True
        while True:
            if not thresholdBroken:
                break
            thresholdBroken = False
            for index in range(0, len(self.states)):
                if index == 0:
                    continue
                elif index == 100:
                    continue
                reward = 0
                if index + self.policy[index] == 100:
                    reward = reward + 1
                if index - self.policy[index] == 0:
                    reward = reward - 1

                value = 0.5 * (reward + self.discount * self.values[index + self.policy[index]])
                + 0.5 * (reward + self.discount * self.values[index - self.policy[index]])

            for index in range(len(self.states)):
                actions = []
                rewards = []
                for state in range(0, len(self.states)):
                    if index == 0:
                        continue
                    elif index == 100:
                        continue
                    for action in range(1, min(state, 100 - state)):
                        actions.append(action)




