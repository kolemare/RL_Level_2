import random
from collections import defaultdict, deque
import matplotlib.pyplot as plt


class Deck:
    def __init__(self):
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) == 0:
            self.reset_deck()
        return self.cards.pop()

    def reset_deck(self):
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

    def determine_final_reward(self):
        dealer_total = self.dealer_policy()
        player_total, _ = self.player_hand.get_total()
        if dealer_total > 21 or player_total > dealer_total:
            return 1  # Player wins
        elif player_total == dealer_total:
            return 0  # Draw
        else:
            return -1  # Dealer wins

    def play_round(self, agent, n):
        self.deal_initial_cards()

        # Initialize state, action, and storage for episode data
        state = self.get_state()
        action = agent.choose_action(state)
        states, actions, rewards = [state], [action], []
        t, T = 0, float('inf')

        while True:
            if t < T:
                # Take action and observe reward
                reward = self.player_action(action)
                next_state = self.get_state() if reward == 0 else None
                next_action = agent.choose_action(next_state) if reward == 0 else None

                # Store reward and transition
                rewards.append(reward)
                if reward == -1:  # Player busts
                    T = t + 1
                elif action == 0:  # Player sticks
                    rewards[-1] = self.determine_final_reward()
                    T = t + 1

                # Record next state and action
                states.append(next_state)
                actions.append(next_action)

                # Prepare for next step
                state, action = next_state, next_action

            # Compute and update Q-values when enough steps are taken
            tau = t - n + 1
            if tau >= 0:
                G = sum(rewards[i] * (agent.gamma ** (i - tau)) for i in range(tau, min(tau + n, T)))
                if tau + n < T:
                    next_state_tau_n = states[tau + n]
                    max_next_Q = max(agent.Q[next_state_tau_n])  # Use max Q-value for off-policy update
                    G += agent.gamma ** n * max_next_Q
                agent.update_Q(states[tau], actions[tau], G)

            if tau == T - 1:
                break
            t += 1

        return rewards[-1] if rewards else 0

    def reset(self):
        self.player_hand.reset()
        self.dealer_hand.reset()
        self.deck.reset_deck()


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, n=3):
        self.Q = defaultdict(lambda: [0.0, 0.0])  # Q-values for each state and action (hit or stick)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.n = n

    def choose_action(self, state):
        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])  # Explore: choose a random action
        else:
            # Exploit: choose the action with the highest Q-value
            return 0 if self.Q[state][0] >= self.Q[state][1] else 1

    def update_Q(self, state, action, G):
        # Standard Q-learning update rule
        self.Q[state][action] += self.alpha * (G - self.Q[state][action])


if __name__ == "__main__":
    n = 3  # Number of steps for n-step Q-learning
    game = BlackjackGame()
    agent = QLearningAgent(n=n)
    rounds = 100000

    player_wins, dealer_wins, draws = 0, 0, 0
    win_rate_history = []

    for i in range(rounds):
        result = game.play_round(agent, n)
        if result == 1:
            player_wins += 1
        elif result == -1:
            dealer_wins += 1
        else:
            draws += 1

        game.reset()

        # Record the win rate every 1000 rounds
        if (i + 1) % 1000 == 0:
            win_rate = player_wins / (i + 1)
            win_rate_history.append(win_rate)

    # Plot the win rate over time
    plt.plot(range(1000, rounds + 1, 1000), win_rate_history)
    plt.xlabel('Rounds')
    plt.ylabel('Win Rate')
    plt.title(f'Player Win Rate Over Time with {n}-Step Q-Learning')
    plt.grid()
    plt.savefig("blackjack_n_step_q.png")
