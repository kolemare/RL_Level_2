import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Define the environment
class KnightsTourEnv:
    def __init__(self, start_pos):
        self.board_size = 8
        self.moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        self.start_pos = start_pos
        self.board = None
        self.knight_pos = None
        self.visited = None
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.knight_pos = self.start_pos
        self.board[self.knight_pos] = 1
        self.visited = set()
        self.visited.add(self.knight_pos)
        return self.get_state()

    def get_state(self):
        return self.knight_pos, tuple(sorted(self.visited))

    def is_valid_move(self, pos):
        x, y = pos
        return 0 <= x < self.board_size and 0 <= y < self.board_size and pos not in self.visited

    def step(self, action):
        move = self.moves[action]
        new_pos = (self.knight_pos[0] + move[0], self.knight_pos[1] + move[1])
        if self.is_valid_move(new_pos):
            self.knight_pos = new_pos
            self.board[new_pos] = 1
            self.visited.add(new_pos)
            reward = 1
        else:
            reward = -1

        done = len(self.visited) == self.board_size * self.board_size
        return self.get_state(), reward, done


# Value iteration using Bellman equation (deterministic)
def value_iteration(env, gamma=0.9, theta=1e-5):
    V = defaultdict(float)
    policy = defaultdict(int)
    all_states = []

    for x in range(env.board_size):
        for y in range(env.board_size):
            for visited in range(1 << (env.board_size * env.board_size)):
                visited_set = set()
                for i in range(env.board_size * env.board_size):
                    if visited & (1 << i):
                        visited_set.add((i // env.board_size, i % env.board_size))
                if (x, y) in visited_set:
                    continue
                all_states.append(((x, y), tuple(sorted(visited_set))))

    while True:
        delta = 0
        for state in all_states:
            v = V[state]
            max_value = float('-inf')
            best_action = None
            for action in range(len(env.moves)):
                env.knight_pos = state[0]
                env.visited = set(state[1])
                new_state, reward, _ = env.step(action)
                new_value = reward + gamma * V[new_state]
                if new_value > max_value:
                    max_value = new_value
                    best_action = action
            V[state] = max_value
            policy[state] = best_action
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return policy, V


# Function to visualize the knight's tour
def visualize_tour(path, board_size):
    board = np.zeros((board_size, board_size), dtype=int)
    for step, pos in enumerate(path):
        board[pos] = step + 1

    fig, ax = plt.subplots()
    ax.imshow(board, cmap='viridis', interpolation='none')

    for i in range(board_size):
        for j in range(board_size):
            ax.text(j, i, board[i, j], ha='center', va='center', color='white')

    plt.colorbar(ax.imshow(board, cmap='viridis', interpolation='none'), label='Move number')
    plt.title("Knight's Tour")
    plt.show()


# Main function to run the algorithm
def main():
    start_x = int(input("Enter starting x position (0-7): "))
    start_y = int(input("Enter starting y position (0-7): "))
    start_pos = (start_x, start_y)

    env = KnightsTourEnv(start_pos)
    policy, V = value_iteration(env)

    state = env.reset()
    path = [env.knight_pos]
    while len(env.visited) < env.board_size * env.board_size:
        action = policy[state]
        state, reward, done = env.step(action)
        path.append(env.knight_pos)
        if done:
            break

    visualize_tour(path, env.board_size)


if __name__ == "__main__":
    main()
