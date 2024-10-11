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
        return (player_total, dealer_visible_card, usable_ace)

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

    def play_round(self, agent):
        self.deal_initial_cards()
        episode = []

        # Choose the initial action
        state = self.get_state()
        action = agent.choose_action(state)
        episode.append((state, action))

        while True:
            result = self.player_action(action)

            # If player busts, we end the game
            if result == -1:
                return -1, episode  # Player busts

            # Player sticks
            if action == 0:
                break

            # Get new state and choose the next action
            state = self.get_state()
            action = agent.choose_action(state)
            episode.append((state, action))

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


class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.Q = defaultdict(lambda: [0.0, 0.0])  # Q-values for each state and action (hit or stick)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def choose_action(self, state):
        epsilon = 0.1  # 10% chance to choose a random action
        if random.uniform(0, 1) < epsilon:
            return random.choice([0, 1])  # Explore (random action)
        else:
            return 0 if self.Q[state][0] >= self.Q[state][1] else 1  # Exploit (best action based on Q-values)

    def update_Q(self, episode, reward):
        for t in range(len(episode) - 1):
            state, action = episode[t]
            next_state, next_action = episode[t + 1]

            # SARSA Update formula
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
            )

        # Final state update (when there is no next state-action pair)
        last_state, last_action = episode[-1]
        self.Q[last_state][last_action] += self.alpha * (reward - self.Q[last_state][last_action])

    def update_policy(self, state):
        # Update the policy for a given state based on the Q-values
        if state in self.Q:
            self.policy[state] = 0 if self.Q[state][0] >= self.Q[state][1] else 1
        else:
            self.policy[state] = 1  # Default to hitting if the state is unseen


if __name__ == "__main__":
    game = BlackjackGame()
    agent = SARSAAgent()
    rounds = 100000

    player_wins, dealer_wins, draws = 0, 0, 0
    win_rate_history = []

    for i in range(rounds):
        result, episode = game.play_round(agent)
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
    plt.xlabel('Rounds')
    plt.ylabel('Win Rate')
    plt.title('Player Win Rate Over Time with SARSA')
    plt.grid()
    plt.savefig("blackjack_sarsa.png")
