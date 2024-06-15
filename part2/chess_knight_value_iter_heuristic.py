import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time


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
        return self.knight_pos

    def is_valid_move(self, pos):
        x, y = pos
        return 0 <= x < self.board_size and 0 <= y < self.board_size and pos not in self.visited

    def get_possible_moves(self, pos):
        return [(pos[0] + move[0], pos[1] + move[1]) for move in self.moves if
                self.is_valid_move((pos[0] + move[0], pos[1] + move[1]))]

    def get_degree(self, pos):
        return len(self.get_possible_moves(pos))


def value_iteration(env, gamma=0.9, theta=1e-5, max_iterations=1000):
    V = defaultdict(float)
    policy = defaultdict(lambda: -1)
    all_states = [(x, y) for x in range(env.board_size) for y in range(env.board_size)]

    for iteration in range(max_iterations):
        delta = 0
        for state in all_states:
            if state in env.visited:
                continue
            v = V[state]
            env.knight_pos = state
            env.visited = {state}
            possible_moves = env.get_possible_moves(state)
            if not possible_moves:
                continue
            min_degree = float('inf')
            best_action = None
            for move in possible_moves:
                degree = env.get_degree(move)
                new_value = -degree + gamma * V[move]
                if degree < min_degree:
                    min_degree = degree
                    best_action = move
            if best_action:
                V[state] = new_value
                policy[state] = best_action
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return policy, V


def heuristic_knights_tour_with_value_iteration(env, policy):
    path = [env.reset()]
    for _ in range(env.board_size * env.board_size - 1):
        current_pos = env.knight_pos
        print(f"My current position is index {current_pos}.")

        if current_pos not in policy or policy[current_pos] == -1:
            print("No valid policy for the current position. Terminating.")
            break

        next_move = policy[current_pos]
        print(f"Policy is recommending me to go to {next_move}.")

        if not env.is_valid_move(next_move):
            print(f"This is an invalid move {next_move}. I have been there already.")
            possible_moves = env.get_possible_moves(current_pos)
            if not possible_moves:
                print("No valid moves available. Terminating.")
                break
            next_move = min(possible_moves, key=lambda move: env.get_degree(move))
            print(f"Finding next valid move. Next move is {next_move}.")
        else:
            print(f"I'm listening to policy and going to {next_move}.")

        env.knight_pos = next_move
        env.board[next_move] = len(path) + 1
        env.visited.add(next_move)
        path.append(next_move)
    return path


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


def main():
    start_x = int(input("Enter starting x position (0-7): "))
    start_y = int(input("Enter starting y position (0-7): "))
    start_pos = (start_x, start_y)

    env = KnightsTourEnv(start_pos)
    start_time = time.time()
    policy, V = value_iteration(env)
    path = heuristic_knights_tour_with_value_iteration(env, policy)
    end_time = time.time()

    print(f"Heuristic knight's tour with value iteration completed in {end_time - start_time:.2f} seconds")
    visualize_tour(path, env.board_size)


if __name__ == "__main__":
    main()
