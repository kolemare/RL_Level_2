import numpy as np
import matplotlib.pyplot as plt
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


def heuristic_knights_tour(env):
    def get_degree(pos):
        return len(env.get_possible_moves(pos))

    path = [env.reset()]
    for _ in range(env.board_size * env.board_size - 1):
        current_pos = env.knight_pos
        possible_moves = env.get_possible_moves(current_pos)
        if not possible_moves:
            print("No valid moves available. Terminating.")
            break
        next_move = min(possible_moves, key=get_degree)
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
    path = heuristic_knights_tour(env)
    end_time = time.time()

    print(f"Heuristic knight's tour completed in {end_time - start_time:.2f} seconds")
    visualize_tour(path, env.board_size)


if __name__ == "__main__":
    main()
