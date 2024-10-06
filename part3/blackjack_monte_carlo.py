import random
from collections import defaultdict
import matplotlib.pyplot as plt


class Deck:
    def __init__(self):                         #10   J   Q   K
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) == 0:
            self.reset_deck()
        return self.cards.pop()

    def reset_deck(self):                       #10   J   Q   K
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.shuffle()


class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def get_total(self):
        total = sum(self.cards)
        if 1 in self.cards and total + 10 <= 21:
            return total + 10, True
        return total, False

    def is_bust(self):
        total, _ = self.get_total()
        return total > 21

    def reset(self):
        self.cards = []


class BlackjackGame:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()

    def deal_initial_cards(self):
        self.player_hand.add_card(self.deck.draw_card())
        self.player_hand.add_card(self.deck.draw_card())
        self.dealer_hand.add_card(self.deck.draw_card())
        self.dealer_hand.add_card(self.deck.draw_card())

    def get_state(self):
        player_total, usable_ace = self.player_hand.get_total()
        dealer_visible_card = self.dealer_hand.cards[0]
        return player_total, dealer_visible_card, usable_ace

    def player_action(self, action):
        if action == 1:
            self.player_hand.add_card(self.deck.draw_card())
            if self.player_hand.is_bust():
                return -1  # Player busts
        return 0  # Game continues

    def dealer_policy(self):
        total, _ = self.dealer_hand.get_total()
        while total < 17:
            self.dealer_hand.add_card(self.deck.draw_card())
            total, _ = self.dealer_hand.get_total()
        return total

    def play_round(self, player_policy):
        self.deal_initial_cards()
        episode = []

        while True:
            state = self.get_state()
            action = player_policy(state)
            episode.append((state, action))
            result = self.player_action(action)
            if result == -1:
                return -1, episode  # Player busts
            if action == 0:
                break

        dealer_total = self.dealer_policy()
        player_total, _ = self.player_hand.get_total()

        if dealer_total > 21 or player_total > dealer_total:
            return 1, episode  # Player wins
        elif player_total == dealer_total:
            return 0, episode  # Draw
        else:
            return -1, episode  # Dealer wins

    def reset(self):
        self.player_hand.reset()
        self.dealer_hand.reset()
        self.deck.reset_deck()


class MonteCarloAgent:
    def __init__(self):
        # Q[state][action]: A dictionary that stores the Q-values for state-action pairs
        self.Q = {}  # Q-values for each state and action
        self.returns = {}  # To keep track of all returns for each (state, action) pair
        self.policy = {}  # Policy that decides the action for each state

    def choose_action(self, state):
        # Epsilon-greedy strategy: choose a random action with a probability of epsilon
        epsilon = 0.1   # 10% chance to choose random action
        if random.uniform(0, 1) < epsilon:
            # Explore: randomly choose either action 0 (stick) or action 1 (hit)
            return random.choice([0, 1])
        else:
            # Exploit: choose the best action based on current Q-values
            if state in self.Q:
                if self.Q[state][0] >= self.Q[state][1]:
                    return 0  # Stick
                else:
                    return 1  # Hit
            else:
                # If this state has never been seen, default to hitting
                return 1

    def update_policy(self, state):
        # Update the policy for a given state based on the Q-values
        if state in self.Q:
            if self.Q[state][0] >= self.Q[state][1]:
                self.policy[state] = 0  # Stick
            else:
                self.policy[state] = 1  # Hit
        else:
            self.policy[state] = 1  # Default to hitting if the state is unseen

    def update_Q(self, episode, reward):
        # Update the Q-values based on the episode and final reward
        G = reward  # The final reward from the game
        for state, action in episode:
            # Initialize returns for the (state, action) pair if not already present
            if (state, action) not in self.returns:
                self.returns[(state, action)] = []

            # Append the final reward to the returns for this (state, action)
            self.returns[(state, action)].append(G)

            # Calculate the average return for this (state, action)
            total_return = sum(self.returns[(state, action)])  # Sum of all returns
            average_return = total_return / len(self.returns[(state, action)])  # Average return

            # Update the Q-value for this (state, action)
            if state not in self.Q:
                self.Q[state] = [0, 0]  # Initialize Q-values if the state is not present
            self.Q[state][action] = average_return

            # Update the policy based on the new Q-values
            self.update_policy(state)


if __name__ == "__main__":
    game = BlackjackGame()
    agent = MonteCarloAgent()
    rounds = 100000

    player_wins, dealer_wins, draws = 0, 0, 0
    win_rate_history = []

    for i in range(rounds):
        result, episode = game.play_round(agent.choose_action)
        if result == 1:
            player_wins += 1
        elif result == -1:
            dealer_wins += 1
        else:
            draws += 1

        agent.update_Q(episode, result)
        game.reset()

        # Record the win rate every 1000 rounds
        if (i + 1) % 1000 == 0:
            win_rate = player_wins / (i + 1)
            win_rate_history.append(win_rate)

    # Plot the win rate over time
    plt.plot(range(1000, rounds + 1, 1000), win_rate_history)
    plt.xlabel("Rounds")
    plt.ylabel("Win Rate")
    plt.title("Player's Win Rate Over Time")
    plt.show()

    print(f"Results after {rounds} rounds:")
    print(f"Player Wins: {player_wins}")
    print(f"Dealer Wins: {dealer_wins}")
    print(f"Draws: {draws}")
