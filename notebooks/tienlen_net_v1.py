# Cell 1: Installations
!pip install torch numpy tqdm lz4 pyarrow
!mkdir -p logs saved_models

# Cell 2: Imports 
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque, namedtuple
import pickle
import time
import logging
import copy
import json
from datetime import datetime
import lz4.frame
import glob
import gc
import itertools

# Function to convert Card to JSON-serializable dict
def card_to_json(card):
    return {"suit": card.suit, "rank": card.rank}

# Stable mapping from action to id (modulo 200)
def action_to_id(action):
    """Produce a stable index [0,200) for an action tuple ('TYPE', list_of_cards)"""
    action_type, cards = action
    # for 'cards', sort by (suit, rank) for determinism
    key = (action_type, tuple(sorted((card.suit, card.rank) for card in cards)))
    return hash(key) % 200

# Setup logger
def setup_logger(log_id):
    logger = logging.getLogger(f"game_{log_id}")
    logger.setLevel(logging.WARNING)

    if not logger.handlers:
        log_file = f"logs/game_{log_id}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# Cell 3: Device and seed setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Cell 4: TienLenGame implementation with improved bot logic (unchanged)
class TienLenGame:
    RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    SUITS = ['DIAMONDS', 'CLUBS', 'HEARTS', 'SPADES']
    RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}
    SUIT_ORDER = {suit: i for i, suit in enumerate(SUITS)}
    COMBO_TYPES = ['SINGLE', 'PAIR', 'TRIPLE', 'STRAIGHT', 'BOMB']

    class Card:
        __slots__ = ('suit', 'rank')
        def __init__(self, suit, rank):
            self.suit = suit
            self.rank = rank

        def __repr__(self):
            return f"({self.rank} of {self.suit})"

    class Player:
        __slots__ = ('hand', 'player_id')
        def __init__(self, player_id):
            self.hand = []
            self.player_id = player_id

    def __init__(self, game_id):
        self.players = [self.Player(i) for i in range(4)]
        self.current_player = 0
        self.current_combo = None
        self.last_combo_player = 0
        self.done = False
        self.winner = None
        self.game_id = game_id
        self.history = []
        self._initialize_deck()
        self._deal_cards()

    def set_current_combo(self, combo):
        """Helper to consistently update current combo and last player"""
        self.current_combo = combo
        if combo and combo[0] != "PASS":
            self.last_combo_player = self.current_player

    def _initialize_deck(self):
        self.deck = [self.Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]
        random.shuffle(self.deck)

    def _deal_cards(self):
        for i in range(13):
            for player in self.players:
                player.hand.append(self.deck.pop())
        for player in self.players:
            player.hand.sort(key=lambda c: (self.RANK_VALUES[c.rank], self.SUIT_ORDER[c.suit]))

    def get_valid_combos(self, player_idx, use_cache=True):
        """
        Return ALL valid combos for a player, fully expanded and filtered based on current combo on the table, strictly following Tiến Lên rules.
        """
        try:
            player = self.players[player_idx]
            valid_combos = [("PASS", [])]

            # Helper functions for full expansions
            def get_singles():
                return [("SINGLE", [card]) for card in player.hand]

            def get_pairs():
                pairs = []
                cards_by_rank = {}
                for card in player.hand:
                    cards_by_rank.setdefault(card.rank, []).append(card)
                for rank, cards in cards_by_rank.items():
                    if len(cards) >= 2:
                        for comb in itertools.combinations(cards, 2):
                            pairs.append(("PAIR", list(comb)))
                return pairs

            def get_triples():
                triples = []
                cards_by_rank = {}
                for card in player.hand:
                    cards_by_rank.setdefault(card.rank, []).append(card)
                for rank, cards in cards_by_rank.items():
                    if len(cards) >= 3:
                        for comb in itertools.combinations(cards, 3):
                            triples.append(("TRIPLE", list(comb)))
                return triples

            def get_bombs():
                bombs = []
                cards_by_rank = {}
                for card in player.hand:
                    cards_by_rank.setdefault(card.rank, []).append(card)
                for rank, cards in cards_by_rank.items():
                    if len(cards) >= 4:
                        for comb in itertools.combinations(cards, 4):
                            bombs.append(("BOMB", list(comb)))
                return bombs

            def get_straights():
                straights = []
                card_groups = {}
                for card in player.hand:
                    card_groups.setdefault(card.rank, []).append(card)

                # try every possible straight starting position and length
                for start_rank in range(len(TienLenGame.RANKS) - 2):
                    for length in range(3, len(TienLenGame.RANKS) - start_rank + 1):
                        straight = []
                        for i in range(length):
                            rank = TienLenGame.RANKS[start_rank + i]
                            if rank in card_groups and card_groups[rank]:
                                card = min(card_groups[rank], key=lambda c: self.SUIT_ORDER[c.suit])
                                straight.append(card)
                            else:
                                break
                        if len(straight) == length:
                            straights.append(("STRAIGHT", straight))
                return straights

            valid_combos += get_singles()
            valid_combos += get_pairs()
            valid_combos += get_triples()
            valid_combos += get_straights()
            valid_combos += get_bombs()

            # STRICT FILTERING if there's a current combo to compare against
            if self.current_combo and self.current_combo[0] != "PASS":
                combo_type, current_cards = self.current_combo
                current_value = self.get_combo_value(self.current_combo)
                filtered_combos = [("PASS", [])]

                for combo in valid_combos:
                    if combo[0] == "PASS":
                        continue

                    # BOMB always valid if beating non-bomb or can beat current bomb
                    if combo[0] == "BOMB":
                        if combo_type != "BOMB":
                            filtered_combos.append(combo)  # Bomb beats normal
                            continue
                        else:
                            # bomb vs bomb, must be stronger
                            combo_value = self.get_combo_value(combo)
                            if combo_value > current_value:
                                filtered_combos.append(combo)
                            continue

                    # TIẾN LÊN: Prevent playing single "2" on non-2 singles (except bomb)
                    if (
                        combo_type == "SINGLE"
                        and combo[0] == "SINGLE"
                        and len(combo[1]) == 1
                        and len(current_cards) == 1
                        and combo[1][0].rank == "2"
                        and current_cards[0].rank != "2"
                    ):
                        continue

                    # Only strictly same type can be played, and must be strictly greater
                    if combo[0] == combo_type:
                        # For straights, must match length
                        if combo_type == "STRAIGHT" and len(combo[1]) != len(current_cards):
                            continue
                        combo_value = self.get_combo_value(combo)
                        if combo_value > current_value:
                            filtered_combos.append(combo)
                return filtered_combos

            # If starting/leading, allow any (except pass)
            return [c for c in valid_combos if c[0] != "PASS"]

        except Exception as e:
            print(f"[ERROR] get_valid_combos failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return [("PASS", [])]

    def get_combo_value(self, combo):
        combo_type, cards = combo

        base_value = {
            'SINGLE': 0,
            'PAIR': 100,
            'TRIPLE': 200,
            'STRAIGHT': 300,
            'BOMB': 1000  # Bomb always highest
        }[combo_type]

        if combo_type == "SINGLE":
            card = cards[0]
            rank_val = self.RANK_VALUES[card.rank]
            # '2' (Heo) is naturally assigned as rank_val=12, which sorts after all others
            value = base_value + (rank_val * 4 + self.SUIT_ORDER[card.suit])
            return value

        elif combo_type in ["PAIR", "TRIPLE", "BOMB"]:
            rank_val = self.RANK_VALUES[cards[0].rank]
            max_suit = max(cards, key=lambda c: self.SUIT_ORDER[c.suit])
            value = base_value + (rank_val * 4 + self.SUIT_ORDER[max_suit.suit])
            return value

        elif combo_type == "STRAIGHT":
            highest_card = max(cards, key=lambda c: (self.RANK_VALUES[c.rank], self.SUIT_ORDER[c.suit]))
            value = base_value + (self.RANK_VALUES[highest_card.rank] * 4 + self.SUIT_ORDER[highest_card.suit])
            return value

        return 0  # For PASS

    def suggest_bot_action(self, player_idx):
        valid_actions = self.get_valid_combos(player_idx)

        # If no valid actions, return PASS
        if not valid_actions:
            return ("PASS", [])

        player = self.players[player_idx]
        # Immediate win: play all cards if can
        for action in valid_actions:
            if action[0] != "PASS" and len(action[1]) == len(player.hand):
                return action

        # If leading, play smallest possible combo
        if self.current_combo is None or self.current_combo[0] == "PASS":
            sorted_actions = sorted(
                [a for a in valid_actions if a[0] != "PASS"],
                key=lambda a: self.get_combo_value(a)
            )
            return sorted_actions[0] if sorted_actions else valid_actions[0]

        # If not leading, try to beat current combo same type
        same_type_actions = [
            a for a in valid_actions
            if a[0] == self.current_combo[0] and self.get_combo_value(a) > self.get_combo_value(self.current_combo)
        ]
        if same_type_actions:
            return min(same_type_actions, key=lambda a: self.get_combo_value(a))

        # Bomb if available
        bombs = [a for a in valid_actions if a[0] == "BOMB"]
        if bombs:
            return min(bombs, key=lambda a: self.get_combo_value(a))

        # Otherwise PASS
        return ("PASS", [])

    def step(self, action):
        action_type, cards = action
        player = self.players[self.current_player]
        reward = 0

        if action_type != "PASS":
            # Use (suit, rank) matching to verify and remove played cards
            hand_tuples = [(c.suit, c.rank) for c in player.hand]
            # Validate all action cards are in hand
            for card in cards:
                if (card.suit, card.rank) not in hand_tuples:
                    raise ValueError(f"Card not in hand: {card.suit}-{card.rank}")
            # Remove by one-for-one match to avoid removing duplicates
            for card in cards:
                for idx, hand_card in enumerate(player.hand):
                    if (hand_card.suit, hand_card.rank) == (card.suit, card.rank):
                        del player.hand[idx]
                        break  # Only remove one instance
            # Set new current combo using helper
            self.set_current_combo((action_type, cards))
            # Check if player won (immediately after making a move)
            if len(player.hand) == 0:
                self.done = True
                self.winner = self.current_player
                reward = 10 if self.current_player == 0 else -1
                return None, reward, True, {}
        else:
            # Only allow PASS if there's a current combo to pass on
            if self.current_combo is None:
                raise ValueError("Cannot PASS when there's no current combo")

        # Record action
        self.history.append((self.current_player, action))

        # Move to next player
        self.current_player = (self.current_player + 1) % 4

        # Improved: clear combo if 3 passes in a row (rolling, not just last-3)
        pass_count = 0
        for i in range(1, 4):
            if len(self.history) >= i and self.history[-i][1][0] == "PASS":
                pass_count += 1
            else:
                break
        if pass_count >= 3:
            self.current_combo = None
            # Return turn to last_combo_player after reset,
            # but only if it's defined (not None). Otherwise, go to player 0.
            self.current_player = self.last_combo_player if self.last_combo_player is not None else 0

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        """Optimized/Vectorized: Return game state as a numpy array"""
        state = np.zeros((6, 4, 13), dtype=np.float32)

        # Channel 0: Current player's hand (vectorized)
        player = self.players[self.current_player]
        hand_mask = np.zeros((4, 13), dtype=bool)
        for card in player.hand:
            suit_idx = self.SUIT_ORDER[card.suit]
            rank_idx = self.RANK_VALUES[card.rank]
            hand_mask[suit_idx, rank_idx] = True
        state[0] = hand_mask.astype(np.float32)

        # Channel 4: Current combo
        if self.current_combo and self.current_combo[0] != "PASS":
            combo_mask = np.zeros((4, 13), dtype=bool)
            for card in self.current_combo[1]:
                suit_idx = self.SUIT_ORDER[card.suit]
                rank_idx = self.RANK_VALUES[card.rank]
                combo_mask[suit_idx, rank_idx] = True
            state[4] = combo_mask.astype(np.float32)

        # Channel 5: Last played cards from up to last 3 actions (vectorized)
        last_plays = np.zeros((4, 13), dtype=np.float32)
        for i, (player_idx, action) in enumerate(self.history[-3:]):
            if action[0] != "PASS":
                for card in action[1]:
                    suit_idx = self.SUIT_ORDER[card.suit]
                    rank_idx = self.RANK_VALUES[card.rank]
                    last_plays[suit_idx, rank_idx] = 1.0 - (i * 0.3)
        state[5] = last_plays

        return state

    def copy(self):
        """Create a deep copy of the game state"""
        new_game = TienLenGame(self.game_id)
        new_game.players = []
        for player in self.players:
            new_player = new_game.Player(player.player_id)
            new_player.hand = player.hand[:]  # Shallow copy since Cards are immutable
            new_game.players.append(new_player)

        new_game.current_player = self.current_player
        if self.current_combo:
            new_game.current_combo = (self.current_combo[0], self.current_combo[1][:])
        new_game.last_combo_player = self.last_combo_player
        new_game.done = self.done
        new_game.winner = self.winner
        new_game.history = self.history[:]
        return new_game

    def __repr__(self):
        return f"TienLenGame(player={self.current_player}, done={self.done}, winner={self.winner})"

# Cell 5: Network architecture with residual blocks (unchanged)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class TienLenNet(nn.Module):
    def __init__(self):
        super(TienLenNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Policy head
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc1 = nn.Linear(4 * 4 * 13, 512)
        self.policy_fc2 = nn.Linear(512, 256)
        self.policy_out = nn.Linear(256, 200)

        # Value head
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * 4 * 13, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)
        self.value_activation = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = F.relu(self.policy_fc1(p))
        p = F.relu(self.policy_fc2(p))
        p = self.policy_out(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = F.relu(self.value_fc2(v))
        v = self.value_out(v)
        v = self.value_activation(v)

        return p, v

# Cell 6: Node for MCTS with continual resolving (unchanged)
class Node:
    def __init__(self, game_instance, parent=None, action=None, model=None):
        if hasattr(game_instance, "current_player") and game_instance.current_player != 0:
            self.game_instance = game_instance  # No copy for bot
        else:
            self.game_instance = game_instance.copy()
        self.parent = parent
        self.action = action
        self.children = []
        self.w = 0
        self.n = 0
        self.p = 0
        self.valid_actions = None
        self.state_key = None
        self.model = model
        self.resolved = False  # For continual resolving
        self.depth = (parent.depth + 1) if parent is not None and hasattr(parent, 'depth') else 0

    def select_child(self):
        c_puct = 2.0
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            u = c_puct * child.p * np.sqrt(self.n) / (1 + child.n)
            depth_penalty = 0.1 * child.depth
            score = (child.w / (child.n + 1e-5)) + u - depth_penalty
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def update(self, value):
        self.n += 1
        self.w += value
        if self.parent:
            self.parent.update(-value)

    def resolve(self):
        # Only resolve for RL agent (player 0), never for bot nodes!
        if self.game_instance.current_player != 0:
            self.resolved = True
            return 0.0

        try:
            self.valid_actions = self.game_instance.get_valid_combos(0)
            state = self.game_instance.get_state()
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_logits, value = self.model(state_tensor)
            action_probs = F.softmax(action_logits, dim=1).cpu().numpy()[0]

            # Expand all valid children
            for action in self.valid_actions:
                game_copy = self.game_instance.copy()
                try:
                    game_copy.step(action)
                    child = Node(game_copy, self, action, self.model)
                    idx = action_to_id(action)
                    child.p = action_probs[idx]
                    self.children.append(child)
                except Exception as e:
                    print(f"Error in child expansion: {e}")
                    continue

            self.resolved = True
            return value.item()
        except Exception as e:
            print(f"Resolve failed: {e}")
            import traceback
            traceback.print_exc()
            self.resolved = True
            return 0.0

    def is_leaf(self):
        return not self.children

# Cell 7: MCTS with continual resolving and caching (unchanged)
class MCTS:
    def __init__(self, model, num_simulations=5, use_cache=True, use_continual_resolving=True):
        self.model = model
        self.num_simulations = num_simulations
        self.use_cache = use_cache
        self.use_continual_resolving = use_continual_resolving
        self.state_cache = {}
        self.action_cache = {}

    def run(self, game):
        root = Node(game.copy(), model=self.model)
        try:
            root.resolve()
        except Exception as e:
            return np.zeros(200)
        import time
        start_time = time.time()
        for _ in range(self.num_simulations):
            # Break if over 100ms wall time
            if time.time() - start_time > 0.1:
                break
            node = root
            tmp_game = node.game_instance.copy()
            depth = 0
            max_depth = 20  # Was 100
            while not node.is_leaf() and depth < max_depth:
                node = node.select_child()
                if self.use_continual_resolving and not node.resolved:
                    try:
                        node.resolve()
                    except Exception:
                        break
                tmp_game = node.game_instance.copy()
                depth += 1
                if tmp_game.done:
                    break
            if not tmp_game.done:
                state = tmp_game.get_state()
                state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
                if device.type == "cuda":
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                        _, value = self.model(state_tensor)
                else:
                    with torch.no_grad():
                        _, value = self.model(state_tensor)
                value = value.item()
            else:
                value = 1.0 if tmp_game.winner == 0 else -1.0
            node.update(value)
        action_probs = np.zeros(200)
        total_visits = sum(child.n for child in root.children)
        if total_visits > 0:
            for child in root.children:
                action_str = json.dumps({
                    "type": child.action[0],
                    "cards": [card_to_json(c) for c in child.action[1]]
                })
                idx = action_to_id(child.action)
                action_probs[idx] = child.n / total_visits
        else:
            action_probs = np.ones(200) / 200
        return action_probs

# Cell 8: PPO Replay Buffer with trajectory storage (unchanged)
class PPOBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def compute_advantages(self):
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        last_advantage = 0

        # Calculate advantages using GAE
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t+1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def get_trajectory(self):
        advantages = self.compute_advantages()
        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)
        values = np.array(self.values)
        returns = advantages + values

        return states, actions, log_probs, returns, advantages, values

    def __len__(self):
        return len(self.states)

# Cell 9: Replay Buffer for PPO (single-thread, no locks or threading)
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.compression_enabled = True

    def push(self, trajectory):
        if self.compression_enabled:
            compressed = lz4.frame.compress(pickle.dumps(trajectory))
            if len(self.buffer) >= self.capacity:
                self.buffer.popleft()
            self.buffer.append(compressed)
        else:
            if len(self.buffer) >= self.capacity:
                self.buffer.popleft()
            self.buffer.append(trajectory)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        if self.compression_enabled:
            return [pickle.loads(lz4.frame.decompress(s)) for s in samples]
        return samples

    def __len__(self):
        return len(self.buffer)

# Cell 10: PPO Training Function (unchanged)
def ppo_train(model, buffer, optimizer, scaler, clip_epsilon=0.2, ppo_epochs=4, batch_size=128):
    if len(buffer) < batch_size:
        return None

    trajectories = buffer.sample(batch_size)
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0

    for trajectory in trajectories:
        states, actions, old_log_probs, returns, advantages, old_values = trajectory

        # Convert to tensors
        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).long().to(device)
        old_log_probs = torch.tensor(old_log_probs).float().to(device)
        returns = torch.tensor(returns).float().to(device)
        advantages = torch.tensor(advantages).float().to(device)
        old_values = torch.tensor(old_values).float().to(device)

        # PPO update
        for _ in range(ppo_epochs):
            # Get new policy and values
            policy_pred, value_pred = model(states)
            value_pred = value_pred.squeeze()

            # Calculate new log probabilities
            dist = torch.distributions.Categorical(logits=policy_pred)
            new_log_probs = dist.log_prob(actions)

            # Policy loss (clipped surrogate)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            value_pred_clipped = old_values + torch.clamp(value_pred - old_values, -clip_epsilon, clip_epsilon)
            value_loss1 = F.mse_loss(value_pred, returns)
            value_loss2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2).mean()

            # Entropy bonus
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

    num_updates = len(trajectories) * ppo_epochs
    return (
        total_loss / num_updates,
        total_policy_loss / num_updates,
        total_value_loss / num_updates,
        total_entropy_loss / num_updates
    )

# Cell 11: Self-play game function (replaces SelfPlayWorker/Ray)
def self_play_game(model, game_id, model_pool=None, use_continual_resolving=True):
    game = TienLenGame(game_id)
    ppo_buffer = PPOBuffer()
    current_model = model

    if model_pool and random.random() < 0.25:
        model_path = random.choice(model_pool)
        current_model = TienLenNet().to(device)
        current_model.load_state_dict(torch.load(model_path, map_location=device))
        current_model.eval()

    # Use MCTS with optimized default number of simulations (num_simulations=10)
    mcts = MCTS(current_model)

    turn_count = 0
    while not game.done and turn_count < 500:
        turn_count += 1
        current_player = game.current_player
        start_turn_time = time.time()

        if current_player == 0:
            # RL player's turn -- Only here we use MCTS!
            if device.type == "cuda":
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    action_probs = mcts.run(game)
            else:
                action_probs = mcts.run(game)

            valid_actions = game.get_valid_combos(current_player, use_cache=True)
            if valid_actions:
                valid_probs = []
                for action in valid_actions:
                    action_str = json.dumps({
                        "type": action[0],
                        "cards": [card_to_json(c) for c in action[1]]
                    })
                    idx = action_to_id(action)
                    valid_probs.append(action_probs[idx])

                valid_probs = np.array(valid_probs)
                if valid_probs.sum() > 0:
                    valid_probs /= valid_probs.sum()
                else:
                    valid_probs = np.ones(len(valid_actions)) / len(valid_actions)

                action_idx = np.random.choice(len(valid_actions), p=valid_probs)
                action = valid_actions[action_idx]
                log_prob = np.log(valid_probs[action_idx] + 1e-10)

                # Log RL agent action
                action_description = f"{action[0]}: {[f'{c.rank}_{c.suit[:1]}' for c in action[1]]}"
                print(f"[RL] Player 0 action: {action_description}")

                state = game.get_state()
                state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
                if device.type == "cuda":
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        _, value = current_model(state_tensor)
                else:
                    with torch.no_grad():
                        _, value = current_model(state_tensor)
                value = value.item()
                ppo_buffer.store(
                    state,
                    action_idx,
                    log_prob,
                    value,
                    0,  # reward will be filled later
                    False
                )
            else:
                action = ("PASS", [])
            player_tag = "RL"
        else:
            # Bot turn: Rule-based, fast
            action = game.suggest_bot_action(current_player)
            player_tag = f"BOT{current_player}"
            print(f"[BOT] Player {current_player} action: {action}")

        _, reward, done, _ = game.step(action)
        if current_player == 0:
            if ppo_buffer.rewards:
                ppo_buffer.rewards[-1] = reward
            if done:
                ppo_buffer.dones[-1] = True

        turn_time = time.time() - start_turn_time
        print(f"[TURN] Game {game_id} Turn {turn_count}: Player {current_player} ({player_tag}) took {turn_time:.4f}s")

    final_reward = 10.0 if game.winner == 0 else -10.0
    # Only add final reward for RL agent
    if ppo_buffer.rewards and current_player == 0:
        ppo_buffer.rewards[-1] += final_reward

    if len(ppo_buffer) > 0:
        return ppo_buffer.get_trajectory(), game.winner
    return None, game.winner

# Cell 12: Main training loop, sequential, no Ray/multiprocess
# DummyScaler for CPU training compatibility
class DummyScaler:
    """Dummy scaler for CPU training"""
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

def log_gpu_utilization():
    """
    Simple CUDA memory/usage print helper (calls nvidia-smi if available for full info).
    """
    if torch.cuda.is_available():
        print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        print(f"[GPU] Memory reserved: {torch.cuda.memory_reserved()/1e6:.2f} MB")
        try:
            utilization = None
            import subprocess
            # Try to get GPU utilization from nvidia-smi, fallback if not available
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            ).decode().strip().split('\n')[0]
            utilization = float(smi_out)
            print(f"[GPU] Utilization: {utilization:.1f}%")
            if utilization < 60:
                print("[WARNING] GPU utilization low (<60%). Consider increasing batch size or model size.")
        except Exception:
            print("[GPU] Could not get utilization, nvidia-smi not available or failed.")

def main_train_loop():
    print("Initializing training process...")

    # Create model
    model = TienLenNet().to(device)
    print(f"Model architecture:\n{model}")

    # Load model if exists
    model_path = "saved_models/tien_len_net.pth"
    start_epoch = 0
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        start_epoch = int(model_path.split("_")[-1].split(".")[0]) if "epoch" in model_path else 0

    # Replay buffer
    buffer = ReplayBuffer(capacity=50000)

    # Model pool for prev models (epoch checkpoints)
    model_pool = []
    model_pool_size = 5

    # Load existing models
    saved_models = glob.glob("saved_models/tien_len_net_*.pth")
    saved_models.sort(key=os.path.getmtime, reverse=True)
    model_pool = saved_models[:model_pool_size]

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    if device.type == "cuda":
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        scaler = DummyScaler()

    # Optimize episodes & buffer size for speed
    num_episodes = 500
    games_per_episode = 10    # Lower for faster loop (from 30)
    training_steps = 50       # Lower for faster loop (from 100)
    min_buffer_size = 500     # ↓ lower requirement to enable early training
    batch_size = 512          # Increase batch size for better GPU utilization (from 256)
    best_win_rate = 0.0

    for episode in range(start_epoch, num_episodes):
        print(f"\n=== Episode {episode+1}/{num_episodes} | Buffer: {len(buffer)} | LR: {optimizer.param_groups[0]['lr']:.2e} ===")
        start_time = time.time()
        episode_wins = 0

        for i in range(games_per_episode):
            game_id = episode * games_per_episode + i
            print(f"Starting game {i+1}/{games_per_episode} (ID: {game_id})")
            game_start = time.time()
            trajectory, winner = self_play_game(model, game_id, model_pool)
            if winner == 0:
                episode_wins += 1
            game_time = time.time() - game_start
            print(f"Game completed in {game_time:.2f}s. {'RL Agent won!' if winner == 0 else 'Bot won.'}")
            if trajectory is not None:
                buffer.push(trajectory)

        win_rate = episode_wins / games_per_episode
        print(f"Collected {games_per_episode} games | Win rate: {win_rate:.2f} | Buffer size: {len(buffer)}")

        # Show GPU info every 5 episodes
        if (episode+1) % 5 == 0:
            log_gpu_utilization()

        # Only start training if buffer is sufficiently full
        if len(buffer) < min_buffer_size:
            print(f"Skipping training, buffer size {len(buffer)} < {min_buffer_size}")
            continue

        train_losses, policy_losses, value_losses = [], [], []
        for step in range(training_steps):
            step_start = time.time()
            results = ppo_train(model, buffer, optimizer, scaler, batch_size=batch_size)
            if results is None:
                continue

            loss, policy_loss, value_loss, entropy_loss = results
            train_losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

            if step % 10 == 0:
                print(f"Step {step+1}/{training_steps} | Loss: {loss:.4f} | Time: {time.time()-step_start:.2f}s")

        if train_losses:
            avg_loss = sum(train_losses) / len(train_losses)
            scheduler.step(avg_loss)

        if (episode+1) % 10 == 0:
            print("Evaluating model...")
            eval_start = time.time()
            win_rate = evaluate(model, num_games=20)
            eval_time = time.time() - eval_start
            print(f"Evaluation completed in {eval_time:.2f}s | Win rate: {win_rate:.2f}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = f"saved_models/best_tien_len_net_ep{episode+1}_win{win_rate:.2f}.pth"
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model with win rate {win_rate:.2f}!")

        save_path = f"saved_models/tien_len_net_ep{episode+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

        model_pool.append(save_path)
        if len(model_pool) > model_pool_size:
            oldest_model = model_pool.pop(0)
            try:
                os.remove(oldest_model)
            except:
                pass

        episode_time = time.time() - start_time
        buffer_size = len(buffer)
        print(f"Episode completed in {episode_time:.2f}s | Buffer size: {buffer_size}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_gpu_utilization()

    print("Training completed!")

# Cell 13: Evaluation function (unchanged)
def evaluate(model, num_games=20):
    model_wins = 0
    for game_id in range(num_games):
        game = TienLenGame(game_id)
        while not game.done:
            current_player = game.current_player
            state = game.get_state()
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_logits, _ = model(state_tensor)
                action_probs = F.softmax(action_logits, dim=1).cpu().numpy()[0]

            valid_actions = game.get_valid_combos(current_player)
            if valid_actions:
                # Collect probabilities for valid actions only
                valid_probs = []
                for action in valid_actions:
                    action_str = json.dumps({
                        "type": action[0],
                        "cards": [card_to_json(c) for c in action[1]]
                    })
                    idx = action_to_id(action)
                    valid_probs.append(action_probs[idx])

                valid_probs = np.array(valid_probs)
                if valid_probs.sum() > 0:
                    valid_probs /= valid_probs.sum()
                else:
                    valid_probs = np.ones(len(valid_actions)) / len(valid_actions)

                action_idx = np.random.choice(len(valid_actions), p=valid_probs)
                action = valid_actions[action_idx]
            else:
                action = ("PASS", [])

            game.step(action)

        if game.winner == 0:
            model_wins += 1

    return model_wins / num_games

# Cell 14: Run training
# if __name__ == "__main__":
#     # =============================== #
#     #   GPU/Colab Environment Check   #
#     # =============================== #
#     print("\n===== GPU & Torch Environment Info =====")
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"CUDA device count: {torch.cuda.device_count()}")
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"Device name: {torch.cuda.get_device_name(0)}")
#         try:
#             import subprocess
#             print("Running nvidia-smi for GPU details:")
#             smi = subprocess.check_output(["nvidia-smi"]).decode()
#             print(smi)
#         except Exception:
#             print("nvidia-smi not available or failed.")
#         # Test allocation
#         test_tensor = torch.randn(3, 3).to("cuda")
#         print(f"Test tensor device: {test_tensor.device}")
#     else:
#         print("Warning: CUDA not available! Training will run on CPU and be much slower.")
#     print("========================================\n")

#     # ============================== #
#     #   Pick training parameters     #
#     # ============================== #
#     try:
#         n_episodes = int(input("Enter number of training episodes (default 100): ") or 100)
#     except Exception:
#         n_episodes = 100

#     # ===== Run Main Training Loop with interrupt save =====
#     # Patch main_train_loop to accept num_episodes argument if needed
#     try:
#         main_train_loop()
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user (KeyboardInterrupt)!")
#         # Try to save model if available
#         import sys
#         model = None
#         main_globals = sys.modules['__main__'].__dict__
#         # Try to get model from main_train_loop scope if defined globally
#         model = main_globals.get('model', None)
#         try:
#             if model is not None:
#                 torch.save(model.state_dict(), "saved_models/interrupted_model.pth")
#                 print("Model state saved at: saved_models/interrupted_model.pth")
#         except Exception as e:
#             print(f"Could not save interrupted model: {e}")
#     except Exception as e:
#         print(f"\nUnexpected error: {e}")
#         import traceback
#         traceback.print_exc()

# Cell 15: Test Cases for TienLenGame Logic
def run_test_cases():
    print("===== RUNNING GAME LOGIC TEST CASES =====")
    
    # Test 1: Basic game initialization
    print("\nTest 1: Game Initialization")
    game = TienLenGame(0)
    print(f"Players: {len(game.players)}")
    print(f"Player 0 hand size: {len(game.players[0].hand)} cards")
    print(f"Current player: {game.current_player}")
    print(f"Deck size after dealing: {len(game.deck)}")
    assert len(game.players) == 4
    assert all(len(p.hand) == 13 for p in game.players)
    assert len(game.deck) == 0
    
    # Test 2: Valid combo detection (leading)
    print("\nTest 2: Valid Combos (Leading)")
    game = TienLenGame(1)
    valid_combos = game.get_valid_combos(0)
    print(f"Valid combos for player 0 (leading): {len(valid_combos)}")
    assert len(valid_combos) > 0
    assert ("PASS", []) not in valid_combos
    
    # Test 3: Valid Combos (After non-PASS combo)
    print("\nTest 3: Valid Combos (After non-PASS combo)")
    game = TienLenGame(1)
    test_card = game.Card("HEARTS", "5")
    game.current_combo = ("SINGLE", [test_card])
    valid_combos = game.get_valid_combos(1)
    print(f"Valid combos after non-PASS: {len(valid_combos)}")
    assert ("PASS", []) in valid_combos
    assert any(c[0] != "PASS" for c in valid_combos)
    
    # Test 4: Combo value calculation
    print("\nTest 4: Combo Value Calculation")
    card_low = game.Card("DIAMONDS", "3")
    card_high = game.Card("SPADES", "2")
    single_low = ("SINGLE", [card_low])
    single_high = ("SINGLE", [card_high])
    print(f"Value 3♦: {game.get_combo_value(single_low)}")
    print(f"Value 2♠: {game.get_combo_value(single_high)}")
    assert game.get_combo_value(single_low) < game.get_combo_value(single_high)
    
    # Test 5: Bot action suggestion
    print("\nTest 5: Bot Action Suggestion")
    # Setup a scenario where bot should play smallest card
    game = TienLenGame(2)
    bot_action = game.suggest_bot_action(0)
    print(f"Bot suggested action: {bot_action}")
    assert bot_action[0] != "PASS"
    
    # Test 6: Playing a card and state transition
    print("\nTest 6: Playing a Card")
    player0_hand_size = len(game.players[0].hand)
    action = ("SINGLE", [game.players[0].hand[0]])
    _, _, done, _ = game.step(action)
    print(f"Hand size after play: {len(game.players[0].hand)}")
    print(f"Current combo: {game.current_combo}")
    print(f"Next player: {game.current_player}")
    assert len(game.players[0].hand) == player0_hand_size - 1
    assert game.current_player == 1
    assert not done
    
    # Test 7: 3 consecutive passes reset, uses a real played combo to ensure last_combo_player is correct
    print("\nTest 7: 3 Consecutive Passes Reset")
    game = TienLenGame(3)
    # Simulate player 0 playing a card (set last_combo_player)
    card = game.players[0].hand[0]
    game.step(("SINGLE", [card]))  # This sets last_combo_player to 0

    # Now player 1 has the turn, set current_combo for test scenario
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "5")])
    game.last_combo_player = 0  # Explicitly for clarity

    # Simulate 3 passes starting from player 1
    game.current_player = 1
    game.step(("PASS", []))
    game.step(("PASS", []))
    game.step(("PASS", []))

    print(f"After 3 passes: Combo={game.current_combo}, Player={game.current_player}")
    assert game.current_combo is None
    assert game.current_player == 0  # Should return to last combo player

    # Test 8: Winning condition
    print("\nTest 8: Winning Condition")
    game = TienLenGame(4)
    # Simulate player 0 playing last card
    game.players[0].hand = [game.Card("SPADES", "A")]
    action = ("SINGLE", game.players[0].hand)
    _, reward, done, _ = game.step(action)
    print(f"Game done: {done}, Winner: {game.winner}, Reward: {reward}")
    assert done
    assert game.winner == 0
    assert reward == 10
    
    # Test 9: Bomb beats non-bomb
    print("\nTest 9: Bomb vs Non-Bomb")
# Cell 15b: Additional Test Cases for Game Logic
def run_additional_test_cases():
    print("===== RUNNING ADDITIONAL GAME LOGIC TEST CASES =====")
    
    # Test 11: 3 consecutive passes should clear current combo
    print("\nTest 11: 3 Consecutive Passes Reset Combo")
    game = TienLenGame(11)
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "7")])
    
    # Simulate 3 passes from players 1,2,3
    game.current_player = 1
    game.step(("PASS", []))
    game.step(("PASS", []))
    game.step(("PASS", []))
    
    print(f"After 3 passes: Combo={game.current_combo}, Current Player={game.current_player}")
    assert game.current_combo is None
    assert game.current_player == 0  # Should return to last combo player
    
    # Test 12: Valid combos after reset should not include PASS
    print("\nTest 12: Valid Combos After Reset")
    valid_combos = game.get_valid_combos(0)
    print(f"Valid combos for player 0: {len(valid_combos)}")
    assert ("PASS", []) not in valid_combos
    
    # Test 13: Valid combos for bomb vs higher bomb
    print("\nTest 13: Bomb vs Higher Bomb")
    game = TienLenGame(13)
    game.current_combo = ("BOMB", [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4")
    ])
    # Player has higher bomb
    game.players[1].hand = [
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "5"),
        game.Card("HEARTS", "5"),
        game.Card("SPADES", "5")
    ]
    valid_combos = game.get_valid_combos(1)
    bomb_combos = [c for c in valid_combos if c[0] == "BOMB"]
    print(f"Valid bomb combos: {len(bomb_combos)}")
    assert len(bomb_combos) > 0
    
    # Test 14: Valid straight combos
    print("\nTest 14: Straight Combos Validation")
    game = TienLenGame(14)
    game.current_combo = ("STRAIGHT", [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "5")
    ])
    # Player has higher straight
    game.players[1].hand = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "5"),
        game.Card("HEARTS", "6")
    ]
    valid_combos = game.get_valid_combos(1)
    straight_combos = [c for c in valid_combos if c[0] == "STRAIGHT"]
    print(f"Valid straight combos: {len(straight_combos)}")
    assert len(straight_combos) > 0
    
    # Test 15: RL agent action selection
    print("\nTest 15: RL Agent Action Selection")
    game = TienLenGame(15)
    model = TienLenNet().to(device)
    action_probs = np.ones(200) / 200  # Uniform probabilities
    
    # Get valid actions for RL agent
    valid_actions = game.get_valid_combos(0)
    valid_probs = []
    for action in valid_actions:
        idx = action_to_id(action)
        valid_probs.append(action_probs[idx])
    
    valid_probs = np.array(valid_probs)
    if valid_probs.sum() > 0:
        valid_probs /= valid_probs.sum()
    else:
        valid_probs = np.ones(len(valid_actions)) / len(valid_actions)
    
    # Select action
    action_idx = np.random.choice(len(valid_actions), p=valid_probs)
    action = valid_actions[action_idx]
    print(f"RL Agent selected action: {action}")
    assert action in valid_actions
    
    # Test 16: Card removal after play
    print("\nTest 16: Card Removal After Play")
    game = TienLenGame(16)
    player = game.players[0]
    original_hand = player.hand.copy()
    card_to_play = original_hand[0]
    action = ("SINGLE", [card_to_play])
    
    game.step(action)
    print(f"Hand size after play: {len(player.hand)}")
    assert len(player.hand) == len(original_hand) - 1
    assert card_to_play not in player.hand
    
    # Test 17: Game end when player has no cards
    print("\nTest 17: Game End Condition")
    game = TienLenGame(17)
    game.players[0].hand = [game.Card("SPADES", "A")]
    action = ("SINGLE", game.players[0].hand)
    _, _, done, _ = game.step(action)
    print(f"Game done: {done}, Winner: {game.winner}")
    assert done
    assert game.winner == 0
    
    # Test 18: Multiple passes without reset
    print("\nTest 18: Two Passes Should Not Reset")
    game = TienLenGame(18)
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "8")])
    
    # Simulate 2 passes
    game.step(("PASS", []))
    game.step(("PASS", []))
    
    print(f"After 2 passes: Combo={game.current_combo}")
    assert game.current_combo is not None
    
    print("\n===== ALL ADDITIONAL TEST CASES PASSED =====")

# Run all tests
if __name__ == "__main__":
    run_test_cases()
    run_additional_test_cases()
    game = TienLenGame(5)
    game.current_combo = ("SINGLE", [game.Card("SPADES", "2")])
    # Create a bomb in player's hand
    bomb_cards = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4")
    ]
    game.players[1].hand = bomb_cards
    valid_combos = game.get_valid_combos(1)
    print("Valid combos with bomb:", [c[0] for c in valid_combos])
    assert "BOMB" in [c[0] for c in valid_combos]
    
    # Test 10: Action to ID stability
    print("\nTest 10: Action to ID Stability")
    action1 = ("PAIR", [
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "5")
    ])
    action2 = ("PAIR", [
        game.Card("CLUBS", "5"),
        game.Card("DIAMONDS", "5")
    ])  # Same cards, different order
    id1 = action_to_id(action1)
    id2 = action_to_id(action2)
    print(f"ID1: {id1}, ID2: {id2}")
    assert id1 == id2
    
    print("\n===== ALL TEST CASES PASSED =====")

# Run the tests
if __name__ == "__main__":
    run_test_cases()
# === ADVANCED TEST CASES (AI Upgraded) ===

def test3():
    print("\n=== Test 3: Valid Combos After Non-PASS Combo ===")
    game = TienLenGame(3)
    test_card = game.Card("HEARTS", "5")
    game.current_combo = ("SINGLE", [test_card])

    print(f"Current combo: {game.current_combo[0]} {[f'{c.rank}-{c.suit[:1]}' for c in game.current_combo[1]]}")

    valid_combos = game.get_valid_combos(1)
    print("\nValid combos for player 1:")
    for i, combo in enumerate(valid_combos[:5]):
        cards_str = ", ".join([f"{c.rank}-{c.suit[:1]}" for c in combo[1]])
        print(f"{i+1}. {combo[0]}: [{cards_str}]")

    has_pass = any(c[0] == "PASS" for c in valid_combos)
    has_higher_single = any(
        c[0] == "SINGLE" and 
        game.get_combo_value(c) > game.get_combo_value(game.current_combo)
        for c in valid_combos
    )
    print(f"\nPASS available: {has_pass}")
    print(f"Higher SINGLE available: {has_higher_single}")
    assert has_pass and has_higher_single

def test4():
    print("\n=== Test 4: Combo Value Calculation ===")
    game = TienLenGame(4)

    test_combos = [
        ("SINGLE", [game.Card("DIAMONDS", "3")]),
        ("SINGLE", [game.Card("SPADES", "2")]),
        ("PAIR", [game.Card("DIAMONDS", "5"), game.Card("CLUBS", "5")]),
        ("STRAIGHT", [
            game.Card("DIAMONDS", "3"),
            game.Card("CLUBS", "4"),
            game.Card("HEARTS", "5")
        ]),
        ("BOMB", [
            game.Card("DIAMONDS", "4"),
            game.Card("CLUBS", "4"),
            game.Card("HEARTS", "4"),
            game.Card("SPADES", "4")
        ])
    ]

    print("Combo values:")
    for combo in test_combos:
        value = game.get_combo_value(combo)
        cards = ", ".join([f"{c.rank}-{c.suit[:1]}" for c in combo[1]])
        print(f"- {combo[0]}[{cards}]: {value}")
    single3 = game.get_combo_value(test_combos[0])
    pair5 = game.get_combo_value(test_combos[2])
    straight345 = game.get_combo_value(test_combos[3])
    bomb4 = game.get_combo_value(test_combos[4])
    assert single3 < pair5 < straight345 < bomb4

def test19():
    print("\n=== Test 19: RL Agent Action Selection ===")
    game = TienLenGame(19)
    model = TienLenNet().to(device)
    game.current_player = 0
    game.players[0].hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "3"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "5")
    ]
    valid_actions = game.get_valid_combos(0)
    print("Valid actions available:")
    for i, action in enumerate(valid_actions):
        cards = ", ".join([f"{c.rank}-{c.suit[:1]}" for c in action[1]])
        print(f"{i+1}. {action[0]}[{cards}]")
    action_probs = np.zeros(200)
    for action in valid_actions:
        idx = action_to_id(action)
        action_probs[idx] = 10.0 if action[0] == "PAIR" else 1.0
    valid_probs = []
    for action in valid_actions:
        idx = action_to_id(action)
        valid_probs.append(action_probs[idx])
    valid_probs = np.array(valid_probs)
    valid_probs /= valid_probs.sum()
    action_idx = np.random.choice(len(valid_actions), p=valid_probs)
    selected_action = valid_actions[action_idx]
    print(f"\nSelected action: {selected_action[0]}[{', '.join([f'{c.rank}-{c.suit[:1]}' for c in selected_action[1]])}]")
    assert selected_action[0] == "PAIR"

def test20():
    print("\n=== Test 20: Complex Game Scenario ===")
    game = TienLenGame(20)
    # Assign hand for player 0 using real objects tracked in hand list
    cards_p0 = [
        game.Card("SPADES", "A"),
        game.Card("HEARTS", "A"),
        game.Card("CLUBS", "A"),
        game.Card("DIAMONDS", "K"),
    ]
    game.players[0].hand = cards_p0
    # Assign hand for player 1 using real objects tracked in hand list
    cards_p1 = [
        game.Card("SPADES", "2"),
        game.Card("HEARTS", "Q"),
        game.Card("CLUBS", "Q"),
    ]
    game.players[1].hand = cards_p1
    # When creating the action, use cards from player 0's actual hand: (cards_p0[:3])
    action = ("TRIPLE", cards_p0[:3])
    game.step(action)
    valid_actions = game.get_valid_combos(1)
    print("Valid responses to TRIPLE A-A-A:")
    for action in valid_actions:
        cards = ", ".join([f"{c.rank}-{c.suit[:1]}" for c in action[1]])
        print(f"- {action[0]}[{cards}]")
    has_bomb = any(a[0] == "BOMB" for a in valid_actions)
    has_pass = any(a[0] == "PASS" for a in valid_actions)
    assert not has_bomb
    assert has_pass

def run_advanced_tests():
    print("\n==== RUNNING ADVANCED AI-UPGRADED TEST CASES ====")
    test3()
    test4()
    test19()
    test20()
    print("\n==== ALL ADVANCED TESTS PASSED ====")
if __name__ == "__main__":
    # Run main test suites
    run_test_cases()
    run_additional_test_cases()
    # Run advanced AI-upgraded tests
    run_advanced_tests()
def run_advanced_tests():
    print("\n===== RUNNING ADVANCED TEST CASES (20) =====")

    # Test 1: Straight với độ dài khác nhau
    print("\nTest 1: Straight Length Mismatch")
    game = TienLenGame(101)
    game.current_combo = ("STRAIGHT", [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "5")
    ])
    game.players[1].hand = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "5"),
        game.Card("HEARTS", "6"),
        game.Card("SPADES", "7")
    ]
    valid_combos = game.get_valid_combos(1)
    straight_combos = [c for c in valid_combos if c[0] == "STRAIGHT" and len(c[1]) == 4]
    print(f"Valid 4-card straights: {len(straight_combos)}")
    assert len(straight_combos) == 0  # Không thể đánh straight dài hơn

    # Test 2: Đôi 2 (mạnh nhất)
    print("\nTest 2: Highest Pair vs Lowest Pair")
    game = TienLenGame(102)
    game.current_combo = ("PAIR", [
        game.Card("DIAMONDS", "2"),
        game.Card("CLUBS", "2")
    ])
    game.players[1].hand = [
        game.Card("HEARTS", "3"),
        game.Card("SPADES", "3")
    ]
    valid_combos = game.get_valid_combos(1)
    has_valid_pair = any(
        c[0] == "PAIR" and
        game.get_combo_value(c) > game.get_combo_value(game.current_combo)
        for c in valid_combos
    )
    print(f"Valid higher pair: {has_valid_pair}")
    assert not has_valid_pair

    # Test 3: Tứ quý heo (bomb) vs Đôi 2
    print("\nTest 3: Bomb 2s vs Pair 2s")
    game = TienLenGame(103)
    game.current_combo = ("PAIR", [
        game.Card("DIAMONDS", "2"),
        game.Card("CLUBS", "2")
    ])
    bomb_cards = [
        game.Card("DIAMONDS", "2"),
        game.Card("CLUBS", "2"),
        game.Card("HEARTS", "2"),
        game.Card("SPADES", "2")
    ]
    game.players[1].hand = bomb_cards
    valid_combos = game.get_valid_combos(1)
    bomb_combos = [c for c in valid_combos if c[0] == "BOMB"]
    print(f"Bomb combos: {len(bomb_combos)}")
    assert len(bomb_combos) > 0

    # Test 4: 3 đôi thông (không hợp lệ)
    print("\nTest 4: Three Consecutive Pairs (Invalid)")
    game = TienLenGame(104)
    game.players[0].hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "3"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4"),
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "5")
    ]
    valid_combos = game.get_valid_combos(0)
    three_pair_combos = [c for c in valid_combos if c[0] == "THREE_PAIR"]
    print(f"Three pair combos: {len(three_pair_combos)}")
    assert len(three_pair_combos) == 0

    # Test 5: Sảnh dài 5 lá
    print("\nTest 5: 5-Card Straight")
    game = TienLenGame(105)
    game.players[0].hand = [
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "6"),
        game.Card("HEARTS", "7"),
        game.Card("SPADES", "8"),
        game.Card("DIAMONDS", "9")
    ]
    valid_combos = game.get_valid_combos(0)
    five_card_straights = [c for c in valid_combos if c[0] == "STRAIGHT" and len(c[1]) == 5]
    print(f"5-card straights: {len(five_card_straights)}")
    assert len(five_card_straights) > 0

    # Test 6: So sánh 2 sảnh cùng độ dài
    print("\nTest 6: Straight Comparison - Same Length")
    game = TienLenGame(106)
    game.current_combo = ("STRAIGHT", [
        game.Card("DIAMONDS", "6"),
        game.Card("CLUBS", "7"),
        game.Card("HEARTS", "8")
    ])
    game.players[1].hand = [
        game.Card("SPADES", "7"),
        game.Card("DIAMONDS", "8"),
        game.Card("CLUBS", "9")
    ]
    valid_combos = game.get_valid_combos(1)
    valid_straights = [c for c in valid_combos if c[0] == "STRAIGHT"]
    print(f"Valid straights: {len(valid_straights)}")
    assert len(valid_straights) > 0

    # Test 7: Bomb khi không phải lượt đè
    print("\nTest 7: Bomb When Not Required")
    game = TienLenGame(107)
    game.current_combo = None
    bomb_cards = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4")
    ]
    game.players[0].hand = bomb_cards
    valid_combos = game.get_valid_combos(0)
    bomb_combos = [c for c in valid_combos if c[0] == "BOMB"]
    print(f"Bomb combos when leading: {len(bomb_combos)}")
    assert len(bomb_combos) > 0

    # Test 8: Bomb cơ (mạnh nhất) vs Bomb chuồn (yếu nhất)
    print("\nTest 8: Highest Bomb vs Lowest Bomb")
    high_bomb = ("BOMB", [
        game.Card("HEARTS", "A"),
        game.Card("DIAMONDS", "A"),
        game.Card("CLUBS", "A"),
        game.Card("SPADES", "A")
    ])
    low_bomb = ("BOMB", [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "3"),
        game.Card("HEARTS", "3"),
        game.Card("SPADES", "3")
    ])
    print(f"High bomb value: {game.get_combo_value(high_bomb)}")
    print(f"Low bomb value: {game.get_combo_value(low_bomb)}")
    assert game.get_combo_value(high_bomb) > game.get_combo_value(low_bomb)

    # Test 9: Heo lẻ không đè được 3
    print("\nTest 9: Single 2 Cannot Beat Single 3")
    game = TienLenGame(109)
    game.current_combo = ("SINGLE", [game.Card("DIAMONDS", "3")])
    game.players[1].hand = [game.Card("SPADES", "2")]
    valid_combos = game.get_valid_combos(1)
    has_valid_single = any(
        c[0] == "SINGLE" and
        game.get_combo_value(c) > game.get_combo_value(game.current_combo)
        for c in valid_combos
    )
    print(f"Valid higher single: {has_valid_single}")
    assert not has_valid_single

    # Test 10: Pass 3 lần không liên tiếp
    print("\nTest 10: Non-Consecutive Passes")
    game = TienLenGame(110)
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "7")])
    game.step(("PASS", []))
    game.step(("SINGLE", [game.Card("SPADES", "8")]))
    game.step(("PASS", []))
    game.step(("PASS", []))
    print(f"After non-consecutive passes: Combo={game.current_combo}")
    assert game.current_combo is not None

    # Test 11: Lượt đánh sau reset
    print("\nTest 11: Turn After Reset")
    game = TienLenGame(111)
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "7")])
    game.last_combo_player = 0
    game.step(("PASS", []))
    game.step(("PASS", []))
    game.step(("PASS", []))
    print(f"After reset: Current player={game.current_player}")
    assert game.current_player == 0

    # Test 12: Không có action nào ngoài PASS
    print("\nTest 12: Only PASS Available")
    game = TienLenGame(112)
    game.current_combo = ("SINGLE", [game.Card("SPADES", "2")])
    game.players[1].hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "4")
    ]
    valid_combos = game.get_valid_combos(1)
    print(f"Valid combos: {[c[0] for c in valid_combos]}")
    assert len(valid_combos) == 1 and valid_combos[0][0] == "PASS"

    # Test 13: Thắng bằng bomb
    print("\nTest 13: Win with Bomb")
    game = TienLenGame(113)
    bomb_cards = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4")
    ]
    game.players[0].hand = bomb_cards
    action = ("BOMB", bomb_cards)
    _, _, done, _ = game.step(action)
    print(f"Game done: {done}, Winner: {game.winner}")
    assert done and game.winner == 0

    # Test 14: Thắng bằng đôi
    print("\nTest 14: Win with Pair")
    game = TienLenGame(114)
    pair_cards = [
        game.Card("DIAMONDS", "K"),
        game.Card("CLUBS", "K")
    ]
    game.players[0].hand = pair_cards
    action = ("PAIR", pair_cards)
    _, _, done, _ = game.step(action)
    print(f"Game done: {done}, Winner: {game.winner}")
    assert done and game.winner == 0

    # Test 15: Tứ quý chặt đôi heo
    print("\nTest 15: Four Aces Chop Pair of 2s")
    game = TienLenGame(115)
    game.current_combo = ("PAIR", [
        game.Card("DIAMONDS", "2"),
        game.Card("CLUBS", "2")
    ])
    four_aces = [
        game.Card("DIAMONDS", "A"),
        game.Card("CLUBS", "A"),
        game.Card("HEARTS", "A"),
        game.Card("SPADES", "A")
    ]
    game.players[1].hand = four_aces
    valid_combos = game.get_valid_combos(1)
    bomb_combos = [c for c in valid_combos if c[0] == "BOMB"]
    print(f"Bomb combos: {len(bomb_combos)}")
    assert len(bomb_combos) > 0

    # Test 16: Tứ quý chặt tứ quý nhỏ hơn
    print("\nTest 16: Four Aces Chop Four Kings")
    game = TienLenGame(116)
    game.current_combo = ("BOMB", [
        game.Card("DIAMONDS", "K"),
        game.Card("CLUBS", "K"),
        game.Card("HEARTS", "K"),
        game.Card("SPADES", "K")
    ])
    four_aces = [
        game.Card("DIAMONDS", "A"),
        game.Card("CLUBS", "A"),
        game.Card("HEARTS", "A"),
        game.Card("SPADES", "A")
    ]
    game.players[1].hand = four_aces
    valid_combos = game.get_valid_combos(1)
    valid_bombs = [c for c in valid_combos if c[0] == "BOMB" and
                  game.get_combo_value(c) > game.get_combo_value(game.current_combo)]
    print(f"Valid higher bombs: {len(valid_bombs)}")
    assert len(valid_bombs) > 0

    # Test 17: Bài chỉ còn 1 lá
    print("\nTest 17: Last Card Win")
    game = TienLenGame(117)
    last_card = game.Card("SPADES", "A")
    game.players[0].hand = [last_card]
    action = ("SINGLE", [last_card])
    _, _, done, _ = game.step(action)
    print(f"Game done: {done}, Winner: {game.winner}")
    assert done and game.winner == 0

    # Test 18: Heo chặt heo - chất cao hơn thắng
    print("\nTest 18: 2 vs 2 - Higher Suit Wins")
    heart_2 = ("SINGLE", [game.Card("HEARTS", "2")])
    spade_2 = ("SINGLE", [game.Card("SPADES", "2")])
    print(f"Heart 2 value: {game.get_combo_value(heart_2)}")
    print(f"Spade 2 value: {game.get_combo_value(spade_2)}")
    assert game.get_combo_value(spade_2) > game.get_combo_value(heart_2)

    # Test 19: Sảnh A-2-3 (không hợp lệ)
    print("\nTest 19: Invalid Straight (A-2-3)")
    game = TienLenGame(119)
    game.players[0].hand = [
        game.Card("DIAMONDS", "A"),
        game.Card("CLUBS", "2"),
        game.Card("HEARTS", "3")
    ]
    valid_combos = game.get_valid_combos(0)
    invalid_straights = [c for c in valid_combos if c[0] == "STRAIGHT" and
                         "A" in [card.rank for card in c[1]] and
                         "2" in [card.rank for card in c[1]]]
    print(f"Invalid A-2-3 straights: {len(invalid_straights)}")
    assert len(invalid_straights) == 0

    # Test 20: Sảnh Q-K-A hợp lệ
    print("\nTest 20: Valid Straight (Q-K-A)")
    game = TienLenGame(120)
    game.players[0].hand = [
        game.Card("DIAMONDS", "Q"),
        game.Card("CLUBS", "K"),
        game.Card("HEARTS", "A")
    ]
    valid_combos = game.get_valid_combos(0)
    valid_straights = [c for c in valid_combos if c[0] == "STRAIGHT" and
                      {"Q", "K", "A"} == {card.rank for card in c[1]}]
    print(f"Valid Q-K-A straights: {len(valid_straights)}")
    assert len(valid_straights) > 0

    print("\n===== ALL 20 ADVANCED TEST CASES PASSED =====\n")


def run_rl_agent_tests():
    print("\n===== ADVANCED RL AGENT TEST CASES (20) =====")
    # Test 1: Prefer Low Cards When Leading
    print("\nTest 1: Prefer Low Cards When Leading")
    game = TienLenGame(201)
    agent_hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "5"),
        game.Card("SPADES", "A"),
        game.Card("DIAMONDS", "2")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert "3" in [c.rank for c in action[1]] or "4" in [c.rank for c in action[1]]

    # Test 2: PASS When Cannot Beat
    print("\nTest 2: PASS When Cannot Beat")
    game = TienLenGame(202)
    game.current_combo = ("SINGLE", [game.Card("SPADES", "2")])
    agent_hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "4")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] == "PASS"

    # Test 3: Use Bomb When Necessary
    print("\nTest 3: Use Bomb When Necessary")
    game = TienLenGame(203)
    game.current_combo = ("SINGLE", [game.Card("SPADES", "2")])
    bomb_cards = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4")
    ]
    agent_hand = bomb_cards + [game.Card("HEARTS", "5")]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert action[0] == "BOMB"

    # Test 4: Save Bomb When Not Necessary
    print("\nTest 4: Save Bomb When Not Necessary")
    game = TienLenGame(204)
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "7")])
    bomb_cards = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4")
    ]
    agent_hand = bomb_cards + [game.Card("SPADES", "8")]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert action[0] == "SINGLE" and "8" in [c.rank for c in action[1]]

    # Test 5: Win Immediately When Possible
    print("\nTest 5: Win Immediately When Possible")
    game = TienLenGame(205)
    winning_cards = [
        game.Card("SPADES", "A"),
        game.Card("HEARTS", "A")
    ]
    game.players[0].hand = winning_cards
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert action[0] == "PAIR" and len(action[1]) == 2

    # Test 6: No Bomb if Not Possible
    print("\nTest 6: No Invalid Bomb")
    game = TienLenGame(206)
    agent_hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "3"),
        game.Card("HEARTS", "3"),
        game.Card("SPADES", "4"),
        game.Card("CLUBS", "5")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]} (Should NOT be BOMB)")
    assert action[0] != "BOMB"

    # Test 7: Winning with a Single 2
    print("\nTest 7: Win with Single 2")
    game = TienLenGame(207)
    agent_hand = [
        game.Card("SPADES", "2")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert action[0] == "SINGLE" and "2" in [c.rank for c in action[1]]

    # Test 8: Bomb Overweighs Pair 2 Combo
    print("\nTest 8: Bomb Over Pair 2")
    game = TienLenGame(208)
    game.current_combo = ("PAIR", [
        game.Card("HEARTS", "2"),
        game.Card("SPADES", "2")
    ])
    bomb_cards = [
        game.Card("DIAMONDS", "7"),
        game.Card("CLUBS", "7"),
        game.Card("HEARTS", "7"),
        game.Card("SPADES", "7")
    ]
    game.players[0].hand = bomb_cards
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert action[0] == "BOMB"

    # Test 9: No PASS if Can Win
    print("\nTest 9: No PASS if Can Play All Cards")
    game = TienLenGame(209)
    hand = [
        game.Card("DIAMONDS", "6"),
        game.Card("CLUBS", "6"),
        game.Card("HEARTS", "6"),
        game.Card("SPADES", "6")
    ]
    game.players[0].hand = hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] != "PASS"

    # Test 10: Bomb Option Available When Beating Bomb
    print("\nTest 10: Bomb Option When Necessary")
    game = TienLenGame(210)
    game.current_combo = ("BOMB", [
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "5"),
        game.Card("HEARTS", "5"),
        game.Card("SPADES", "5")
    ])
    bomb_cards = [
        game.Card("DIAMONDS", "6"),
        game.Card("CLUBS", "6"),
        game.Card("HEARTS", "6"),
        game.Card("SPADES", "6")
    ]
    game.players[0].hand = bomb_cards
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]} (Should be BOMB)")
    assert action[0] == "BOMB"

    # Test 11: PASS if Only Option Is PASS
    print("\nTest 11: Only PASS Available")
    game = TienLenGame(211)
    game.current_combo = ("SINGLE", [game.Card("SPADES", "2")])
    agent_hand = [
        game.Card("DIAMONDS", "3"),
        game.Card("CLUBS", "4")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] == "PASS"

    # Test 12: Winning with Bomb Recognized
    print("\nTest 12: Winning with Bomb Recognized")
    game = TienLenGame(212)
    bomb_cards = [
        game.Card("HEARTS", "K"),
        game.Card("DIAMONDS", "K"),
        game.Card("CLUBS", "K"),
        game.Card("SPADES", "K")
    ]
    game.players[0].hand = bomb_cards
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert action[0] == "BOMB" and len(action[1]) == 4

    # Test 13: Rule-Based Bot Avoids Bomb Waste
    print("\nTest 13: Rule-Based Bot Avoids Bomb Waste")
    game = TienLenGame(213)
    game.current_combo = ("SINGLE", [game.Card("HEARTS", "3")])
    bomb_cards = [
        game.Card("HEARTS", "J"),
        game.Card("DIAMONDS", "J"),
        game.Card("CLUBS", "J"),
        game.Card("SPADES", "J")
    ]
    random_card = game.Card("DIAMONDS", "5")
    game.players[0].hand = [random_card] + bomb_cards
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] == "SINGLE" and random_card in action[1]

    # Test 14: Play Largest if Only Possible
    print("\nTest 14: Play Largest Card If Only One")
    game = TienLenGame(214)
    agent_hand = [
        game.Card("SPADES", "2")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert "2" in [c.rank for c in action[1]]

    # Test 15: STRAIGHT Detected in Hand
    print("\nTest 15: STRAIGHT Detected")
    game = TienLenGame(215)
    agent_hand = [
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "6"),
        game.Card("HEARTS", "7"),
        game.Card("SPADES", "8"),
        game.Card("DIAMONDS", "9")
    ]
    game.players[0].hand = agent_hand
    combos = game.get_valid_combos(0)
    any_straight = any(c[0] == "STRAIGHT" for c in combos)
    print(f"STRAIGHT found? {any_straight}")
    assert any_straight

    # Test 16: AI Avoids Bomb as First Play Without Threat
    print("\nTest 16: Avoid Bomb as First Play")
    game = TienLenGame(216)
    bomb_cards = [
        game.Card("DIAMONDS", "4"),
        game.Card("CLUBS", "4"),
        game.Card("HEARTS", "4"),
        game.Card("SPADES", "4"),
    ]
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] != "BOMB"

    # Test 17: Immediate Win for Any Combo
    print("\nTest 17: Win When Only Combo Left")
    game = TienLenGame(217)
    agent_hand = [
        game.Card("DIAMONDS", "K"),
        game.Card("CLUBS", "K")
    ]
    game.players[0].hand = agent_hand
    combos = game.get_valid_combos(0)
    win_combo = [combo for combo in combos if len(combo[1]) == 2]
    assert win_combo
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, cards: {[c.rank for c in action[1]]}")
    assert all(card in agent_hand for card in action[1])

    # Test 18: Leading If All Pass
    print("\nTest 18: Lead If All PASS")
    game = TienLenGame(218)
    game.current_combo = None  # All pass
    agent_hand = [
        game.Card("DIAMONDS", "9")
    ]
    game.players[0].hand = agent_hand
    combos = game.get_valid_combos(0)
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] != "PASS"

    # Test 19: Play Smallest When Leading
    print("\nTest 19: Smallest Combo When Leading")
    game = TienLenGame(219)
    agent_hand = [
        game.Card("DIAMONDS", "5"),
        game.Card("CLUBS", "7"),
        game.Card("HEARTS", "9")
    ]
    game.players[0].hand = agent_hand
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}, {[(c.rank, c.suit) for c in action[1]]}")
    assert action[0] != "PASS"

    # Test 20: Guarantee No PASS on Only Possible Win
    print("\nTest 20: No PASS if Only Win Combo Left")
    game = TienLenGame(220)
    agent_hand = [
        game.Card("HEARTS", "A")
    ]
    game.players[0].hand = agent_hand
    combos = game.get_valid_combos(0)
    assert any(len(c[1]) == 1 for c in combos)
    action = game.suggest_bot_action(0)
    print(f"Suggested action: {action[0]}")
    assert action[0] != "PASS"

    print("\n===== ALL 20 RL AGENT TEST CASES PASSED =====\n")


# Đăng ký gọi các test case mới vào main block
if __name__ == "__main__":
    run_test_cases()
    run_additional_test_cases()
    run_advanced_tests()
    run_rl_agent_tests()