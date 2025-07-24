!pip install torch numpy tqdm lz4 pyarrow
!mkdir -p logs saved_models

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
import sys
import multiprocessing

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

# Device and seed setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# TienLenGame implementation with improved bot logic
class TienLenGame:
    RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    SUITS = ['DIAMONDS', 'CLUBS', 'HEARTS', 'SPADES']
    RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}
    SUIT_ORDER = {suit: i for i, suit in enumerate(SUITS)}
    COMBO_TYPES = ['SINGLE', 'PAIR', 'TRIPLE', 'STRAIGHT', 'BOMB', 'CONSEC_PAIRS']

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
        self.broke_combo = False  # Track if player broke opponent's combo

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
                for start_idx in range(len(TienLenGame.RANKS) - 2):
                    for length in range(3, len(TienLenGame.RANKS) - start_idx + 1):
                        straight = []
                        for i in range(length):
                            rank = TienLenGame.RANKS[start_idx + i]
                            if rank in card_groups and card_groups[rank]:
                                card = min(card_groups[rank], key=lambda c: self.SUIT_ORDER[c.suit])
                                straight.append(card)
                            else:
                                break
                        if len(straight) == length:
                            straights.append(("STRAIGHT", straight))
                return straights

            # --- Insert improved combo logic here ---
            valid_combos += get_singles()
            valid_combos += get_pairs()
            valid_combos += get_triples()
            valid_combos += get_straights()
            valid_combos += get_bombs()
            valid_combos += self.get_consecutive_pairs(player)

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

    def get_consecutive_pairs(self, player):
        # "3 đôi thông" = 3 consecutive pairs (6 cards, ranks must be consecutive, and each pair same rank)
        rank_values = sorted(set(self.RANK_VALUES[card.rank] for card in player.hand))
        available_pairs = {}
        for rank_val in rank_values:
            rank = self.RANKS[rank_val]
            cards_of_rank = [card for card in player.hand if card.rank == rank]
            if len(cards_of_rank) >= 2:
                available_pairs[rank_val] = cards_of_rank
        
        combos = []
        sorted_ranks = sorted(available_pairs.keys())
        for i in range(len(sorted_ranks) - 2):
            r0, r1, r2 = sorted_ranks[i], sorted_ranks[i+1], sorted_ranks[i+2]
            if r1 == r0 + 1 and r2 == r1 + 1:
                cards = []
                for r in [r0, r1, r2]:
                    # Add lowest 2 cards for each rank
                    cards += sorted(available_pairs[r], key=lambda c: self.SUIT_ORDER[c.suit])[:2]
                combos.append(("CONSEC_PAIRS", cards))
        return combos

    def get_combo_value(self, combo):
        combo_type, cards = combo

        base_value = {
            'SINGLE': 0,
            'PAIR': 100,
            'TRIPLE': 200,
            'STRAIGHT': 300,
            'BOMB': 1000,
            'CONSEC_PAIRS': 400
        }[combo_type]

        if combo_type == "SINGLE":
            card = cards[0]
            rank_val = self.RANK_VALUES[card.rank]
            value = base_value + (rank_val * 4 + self.SUIT_ORDER[card.suit])
            return value

        elif combo_type == "CONSEC_PAIRS":
            # Highest rank among the 3 pairs
            highest_rank_val = max(self.RANK_VALUES[card.rank] for card in cards)
            # Get all cards of this rank
            highest_rank_cards = [c for c in cards if self.RANK_VALUES[c.rank] == highest_rank_val]
            # Pick the one with the highest suit
            max_suit = max(highest_rank_cards, key=lambda c: self.SUIT_ORDER[c.suit])
            return base_value + (highest_rank_val * 4 + self.SUIT_ORDER[max_suit.suit])

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
        player = self.players[player_idx]
        hand = player.hand

        # Immediate win condition
        for action in valid_actions:
            if action[0] != "PASS" and len(action[1]) == len(hand):
                return action

        # Card counting and probability tracking
        played_cards = self.get_played_cards()
        remaining_cards = self.get_remaining_cards(played_cards)
        opponent_profile = self.estimate_opponent_hands(remaining_cards)

        # Hand analysis metrics
        hand_strength = self.calculate_hand_strength(hand, opponent_profile)
        high_card_count = sum(1 for card in hand if card.rank in ['A', '2'] or card.rank in ['K', 'Q'])
        low_card_count = sum(1 for card in hand if card.rank in ['3', '4', '5'])
        chain_potential = self.identify_chain_potential(hand)
        bomb_potential = self.identify_bomb_potential(hand)

        # Game phase detection
        total_cards_played = 52 - sum(len(p.hand) for p in self.players)
        game_phase = "early" if total_cards_played < 15 else "mid" if total_cards_played < 35 else "late"

        # === Improved action logic for bot ===
        if self.current_combo is None or self.current_combo[0] == "PASS":
            if game_phase == "early" and random.random() < 0.3:
                non_pass_actions = [a for a in valid_actions if a[0] != "PASS"]
                if non_pass_actions:
                    return random.choice(non_pass_actions)
            if game_phase == "early":
                low_pairs = [a for a in valid_actions
                            if a[0] in ["PAIR", "TRIPLE"]
                            and max(self.RANK_VALUES[c.rank] for c in a[1]) < 5]
                if low_pairs:
                    return random.choice(low_pairs)
                low_singles = [a for a in valid_actions
                               if a[0] == "SINGLE" and self.RANK_VALUES[a[1][0].rank] < 5]
                if low_singles:
                    return min(low_singles, key=lambda a: self.get_combo_value(a))
                short_chains = [a for a in valid_actions if a[0] == "STRAIGHT" and 3 <= len(a[1]) <= 4]
                if short_chains:
                    return min(short_chains, key=lambda a: self.get_combo_value(a))

            long_chains = [a for a in valid_actions if a[0] == "STRAIGHT" and len(a[1]) > 4]
            if long_chains:
                other_non_pass = [a for a in valid_actions if a[0] != "PASS" and a[0] != "STRAIGHT"]
                if other_non_pass:
                    return min(other_non_pass, key=lambda a: self.get_combo_value(a))

            consecutive_pairs = [a for a in valid_actions if a[0] == "CONSEC_PAIRS"]
            if consecutive_pairs:
                return consecutive_pairs[0]

            if game_phase == "early" or game_phase == "mid":
                low_actions = [a for a in valid_actions
                               if max(self.RANK_VALUES[c.rank] for c in a[1]) < 5]
                if low_actions:
                    return random.choice(low_actions)

            if game_phase == "early":
                return self.early_game_lead(hand, valid_actions, hand_strength, chain_potential)
            elif game_phase == "mid":
                return self.mid_game_lead(hand, valid_actions, opponent_profile, bomb_potential)
            else:
                return self.late_game_lead(hand, valid_actions, opponent_profile)

        # Non-leading/responding to a current combo:
        if self.current_combo and self.current_combo[0] != "BOMB":
            bombs = [a for a in valid_actions if a[0] == "BOMB"]
            if bombs and random.random() < 0.3:
                return min(bombs, key=lambda a: self.get_combo_value(a))

        return self.response_strategy(player_idx, valid_actions, game_phase, hand_strength)

    def get_played_cards(self):
        played = []
        for _, action in self.history:
            if action[0] != "PASS":
                played.extend(action[1])
        return played

    def get_remaining_cards(self, played_cards):
        all_cards = [self.Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]
        remaining = []
        for card in all_cards:
            found = False
            for played in played_cards:
                if card.suit == played.suit and card.rank == played.rank:
                    found = True
                    break
            if not found:
                remaining.append(card)
        return remaining

    def estimate_opponent_hands(self, remaining_cards):
        opponent_hands = {i: [] for i in range(4)}
        card_counts = {}
        for card in remaining_cards:
            key = (card.suit, card.rank)
            card_counts[key] = card_counts.get(key, 0) + 1
        
        # Distribute cards to opponents
        for card in remaining_cards:
            possible_owners = [i for i in range(4) if i != self.current_player]
            owner = random.choice(possible_owners)
            opponent_hands[owner].append(card)
            
        return opponent_hands

    def calculate_hand_strength(self, hand, opponent_profile):
        card_values = sum(self.RANK_VALUES[card.rank] for card in hand)
        combo_potential = self.identify_combo_potential(hand)
        flexibility = self.calculate_hand_flexibility(hand)
        opponent_pressure = 0
        for player_id, cards in opponent_profile.items():
            if player_id != self.current_player:
                opponent_pressure += sum(1 for card in cards if card.rank in ['A', '2'])
        strength = (card_values * 0.3 + combo_potential * 0.4 + flexibility * 0.2 -
                    opponent_pressure * 0.1)
        return strength

    def identify_combo_potential(self, hand):
        score = 0
        rank_counts = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        for count in rank_counts.values():
            if count >= 2: score += 2
            if count >= 3: score += 3
            if count >= 4: score += 5
        
        sorted_ranks = sorted(set(self.RANK_VALUES[card.rank] for card in hand))
        chain_length = 0
        max_chain = 0
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                chain_length += 1
                max_chain = max(max_chain, chain_length)
            else:
                chain_length = 0
        if max_chain >= 2:
            score += max_chain * 3
        return score

    def calculate_hand_flexibility(self, hand):
        flexibility = len(hand)
        rank_counts = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        for count in rank_counts.values():
            if count >= 2: flexibility += 2
            if count >= 3: flexibility += 3
            
        for length in range(3, 6):
            if self.can_make_chain(hand, length):
                flexibility += length * 2
        return flexibility

    def early_game_lead(self, hand, valid_actions, hand_strength, chain_potential):
        if hand_strength > 50:
            return self.aggressive_lead(hand, valid_actions)
        else:
            return self.conservative_lead(hand, valid_actions, chain_potential)

    def mid_game_lead(self, hand, valid_actions, opponent_profile, bomb_potential):
        weak_opponents = sum(1 for i, p in enumerate(self.players) 
                          if i != self.current_player and len(p.hand) > 8)
        if weak_opponents >= 2:
            return self.aggressive_lead(hand, valid_actions)
        elif bomb_potential:
            non_bomb_actions = [a for a in valid_actions if a[0] != "BOMB"]
            if non_bomb_actions:
                return min(non_bomb_actions, key=lambda a: self.get_combo_value(a))
        return self.conservative_lead(hand, valid_actions, None)

    def late_game_lead(self, hand, valid_actions, opponent_profile):
        opponent_hand_sizes = [len(p.hand) for i, p in enumerate(self.players) if i != self.current_player]
        shortest_opponent = min(opponent_hand_sizes) if opponent_hand_sizes else 0
        if len(hand) <= shortest_opponent + 2:
            return self.aggressive_lead(hand, valid_actions)
        else:
            return self.conservative_lead(hand, valid_actions, None)

    def aggressive_lead(self, hand, valid_actions):
        actions = [a for a in valid_actions if a[0] != "PASS"]
        high_card_actions = []
        for action in actions:
            if any(card.rank in ['A', '2'] for card in action[1]):
                high_card_actions.append(action)
        if high_card_actions:
            return max(high_card_actions, key=lambda a: self.get_combo_value(a))
        elif actions:
            return max(actions, key=lambda a: self.get_combo_value(a))
        else:
            return ("PASS", [])

    def conservative_lead(self, hand, valid_actions, chain_potential):
        actions = [a for a in valid_actions if a[0] != "PASS"]
        if chain_potential:
            chain_actions = [a for a in actions if a[0] == "STRAIGHT"]
            if chain_actions:
                longest_chain = max(len(a[1]) for a in chain_actions)
                long_chains = [a for a in chain_actions if len(a[1]) == longest_chain]
                return min(long_chains, key=lambda a: self.get_combo_value(a))
        low_card_actions = []
        for action in actions:
            if all(card.rank in ['3', '4', '5', '6'] for card in action[1]):
                low_card_actions.append(action)
        if low_card_actions:
            return min(low_card_actions, key=lambda a: self.get_combo_value(a))
        elif actions:
            return min(actions, key=lambda a: self.get_combo_value(a))
        else:
            return ("PASS", [])

    def response_strategy(self, player_idx, valid_actions, game_phase, hand_strength):
        current_value = self.get_combo_value(self.current_combo)
        combo_type, current_cards = self.current_combo
        player = self.players[player_idx]
        same_type_actions = [
            a for a in valid_actions
            if a[0] == combo_type and self.get_combo_value(a) > current_value
        ]
        if same_type_actions:
            if game_phase == "early":
                return min(same_type_actions, key=lambda a: self.get_combo_value(a))
            elif game_phase == "late" or hand_strength > 60:
                return max(same_type_actions, key=lambda a: self.get_combo_value(a))
            else:
                return min(
                    same_type_actions,
                    key=lambda a: self.get_combo_value(a) - current_value
                )
        bombs = [a for a in valid_actions if a[0] == "BOMB"]
        if bombs:
            bomb_value = min(self.get_combo_value(bomb) for bomb in bombs)
            should_bomb = False
            bomb_benefit = self.calculate_bomb_benefit(player_idx)
            if combo_type == "BOMB":
                stronger_bombs = [b for b in bombs if self.get_combo_value(b) > current_value]
                if stronger_bombs and bomb_benefit > 3:
                    return min(stronger_bombs, key=lambda a: self.get_combo_value(a))
            else:
                if game_phase == "late" and bomb_benefit > 2:
                    should_bomb = True
                elif current_value > 200:
                    should_bomb = True
                elif bomb_benefit > 4:
                    should_bomb = True
                if should_bomb:
                    return min(bombs, key=lambda a: self.get_combo_value(a))
        if combo_type == "STRAIGHT" and len(current_cards) > 3:
            for action in valid_actions:
                if action[0] == "STRAIGHT" and len(action[1]) < len(current_cards):
                    if self.is_chain_subset(action[1], current_cards):
                        return action
        if self.should_pass_strategically(player_idx, game_phase):
            return ("PASS", [])
        return ("PASS", [])

    def calculate_bomb_benefit(self, player_idx):
        benefit = 0
        current_value = self.get_combo_value(self.current_combo)
        if current_value > 150:
            benefit += 2
        players_after = (self.current_player - player_idx) % 4
        if players_after == 3:
            benefit += 1
        opponent_cards = sum(len(p.hand) for i, p in enumerate(self.players) if i != player_idx)
        if len(self.players[player_idx].hand) < opponent_cards / 3:
            benefit += 2
        total_played = 52 - sum(len(p.hand) for p in self.players)
        if total_played > 40:
            benefit += 2
        return benefit

    def should_pass_strategically(self, player_idx, game_phase):
        hand_size = len(self.players[player_idx].hand)
        next_players = [(self.current_player + i) % 4 for i in range(1, 4)]
        if player_idx in next_players:
            last_player = next_players[-1]
            if last_player == player_idx:
                return False
        if game_phase == "early":
            high_cards = sum(1 for card in self.players[player_idx].hand
                            if card.rank in ['A', '2'] or card.rank in ['K', 'Q'])
            if hand_size > 0 and high_cards / hand_size > 0.4:
                return True
        bomb_potential = self.identify_bomb_potential(self.players[player_idx].hand)
        if bomb_potential and self.current_combo[0] != "BOMB":
            return True
        return False

    def identify_chain_potential(self, hand):
        sorted_ranks = sorted(set(self.RANK_VALUES[card.rank] for card in hand))
        chains = []
        if not sorted_ranks:
            return chains
        current_chain = [sorted_ranks[0]]
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == current_chain[-1] + 1:
                current_chain.append(sorted_ranks[i])
            else:
                if len(current_chain) >= 3:
                    chains.append(current_chain)
                current_chain = [sorted_ranks[i]]
        if len(current_chain) >= 3:
            chains.append(current_chain)
        return chains

    def identify_bomb_potential(self, hand):
        rank_counts = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        return any(count >= 4 for count in rank_counts.values())

    def can_make_chain(self, hand, length):
        ranks = sorted(set(self.RANK_VALUES[card.rank] for card in hand))
        for i in range(len(ranks) - length + 1):
            if ranks[i + length - 1] - ranks[i] == length - 1:
                return True
        return False

    def is_chain_subset(self, subset, full_chain):
        sub_ranks = sorted(self.RANK_VALUES[c.rank] for c in subset)
        full_ranks = sorted(self.RANK_VALUES[c.rank] for c in full_chain)
        for i in range(len(full_ranks) - len(sub_ranks) + 1):
            if full_ranks[i:i+len(sub_ranks)] == sub_ranks:
                return True
        return False

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over")
        action_type, cards = action
        player = self.players[self.current_player]
        reward = 0

        # Save previous combo before action for reward shaping
        prev_combo = self.current_combo
        prev_last_combo_player = getattr(self, "last_combo_player", None)

        if action_type != "PASS":
            # Remove card by value (suit and rank)
            for card in cards:
                found = False
                for idx, hand_card in enumerate(player.hand):
                    if hand_card.suit == card.suit and hand_card.rank == card.rank:
                        del player.hand[idx]
                        found = True
                        break
                if not found:
                    raise ValueError(f"Card not in hand: {card} (suit={card.suit}, rank={card.rank})")

            # WIN CHECK: as soon as hand is empty
            if len(player.hand) == 0:
                self.done = True
                self.winner = self.current_player
                reward = 10 if self.current_player == 0 else -1
                self.history.append((self.current_player, action))
                return self.get_state(), reward, True, {}
            # Only set current combo if not win
            self.set_current_combo((action_type, cards))
            self.broke_combo = False
        else:
            # Only allow PASS if there's a current combo to pass on
            if self.current_combo is None:
                raise ValueError("Cannot PASS when there's no current combo")
            self.broke_combo = False

        # ======= Reward Shaping: Encourage playing cards and blocking combos =======
        if action_type != "PASS":
            # Small reward for playing cards
            reward += 0.1 * len(cards)
            
            # Reward for breaking combo: if responding to another player's combo and beating it
            broke_combo = False
            if (
                prev_combo is not None
                and prev_combo[0] != "PASS"
                and prev_last_combo_player is not None
                and prev_last_combo_player != self.current_player
                and self.get_combo_value((action_type, cards)) > self.get_combo_value(prev_combo)
                and action_type == prev_combo[0]
            ):
                broke_combo = True
                reward += 0.5
                self.broke_combo = True
                
            # Bonus for playing low cards in early game
            total_played = 52 - sum(len(p.hand) for p in self.players)
            if total_played < 20 and all(card.rank in ['3','4','5','6'] for card in cards):
                reward += 0.2
                
            # Penalty for playing high cards too early
            if total_played < 10 and any(card.rank in ['A','2'] for card in cards):
                reward -= 0.3
                
            # Bonus for saving bombs
            if action_type == "BOMB" and total_played < 30:
                reward -= 0.4  # Penalize early bomb use
            elif action_type == "BOMB" and total_played > 40:
                reward += 0.6  # Reward late bomb use

        # Record action
        self.history.append((self.current_player, action))

        # Move to next player
        self.current_player = (self.current_player + 1) % 4

        # Clear combo if 3 passes in a row (rolling)
        pass_count = 0
        for i in range(1, 4):
            if len(self.history) >= i and self.history[-i][1][0] == "PASS":
                pass_count += 1
            else:
                break
        if pass_count >= 3:
            self.current_combo = None
            # Return turn to last_combo_player after reset
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
            new_player.hand = [new_game.Card(c.suit, c.rank) for c in player.hand]
            new_game.players.append(new_player)

        new_game.current_player = self.current_player
        if self.current_combo:
            new_game.current_combo = (self.current_combo[0], 
                                      [new_game.Card(c.suit, c.rank) for c in self.current_combo[1]])
        new_game.last_combo_player = self.last_combo_player
        new_game.done = self.done
        new_game.winner = self.winner
        new_game.history = copy.deepcopy(self.history)
        new_game.broke_combo = self.broke_combo
        return new_game

    def __repr__(self):
        return f"TienLenGame(player={self.current_player}, done={self.done}, winner={self.winner})"

# Network architecture with residual blocks
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

# Rewind Mechanism
class RewindBuffer:
    def __init__(self, max_depth=5):
        self.buffer = deque(maxlen=max_depth)
        self.current_index = -1
    
    def push(self, game_state, action):
        self.buffer.append((copy.deepcopy(game_state), action))
        self.current_index = len(self.buffer) - 1
    
    def rewind(self, steps=1):
        if self.current_index - steps >= 0:
            self.current_index -= steps
            return self.buffer[self.current_index][0]
        return None

    def get_current_state(self):
        return self.buffer[self.current_index][0] if self.current_index >= 0 else None

# Base ReplayBuffer for experience storage
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)

# Strategic Experience Replay
# Strategic Experience Replay
class StrategicReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=100000):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.win_probs = deque(maxlen=capacity)

    def push(self, trajectory, win_prob):
        compressed = lz4.frame.compress(pickle.dumps(trajectory))
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
            self.priorities.popleft()
            self.win_probs.popleft()
        self.buffer.append(compressed)
        self.priorities.append(win_prob)
        self.win_probs.append(win_prob)

    def sample(self, batch_size, win_sample_ratio=0.7):
        # Separate wins and losses
        win_indices = [i for i, wp in enumerate(self.win_probs) if wp > 0.7]
        loss_indices = [i for i, wp in enumerate(self.win_probs) if wp <= 0.7]
        
        # Calculate sample sizes
        win_sample_size = min(int(batch_size * win_sample_ratio), len(win_indices))
        loss_sample_size = min(batch_size - win_sample_size, len(loss_indices))
        
        # Sample from wins and losses
        win_samples = random.sample(win_indices, win_sample_size) if win_indices else []
        loss_samples = random.sample(loss_indices, loss_sample_size) if loss_indices else []
        
        # Combine samples
        samples = win_samples + loss_samples
        random.shuffle(samples)
        
        # Decompress and return
        return [pickle.loads(lz4.frame.decompress(self.buffer[i])) for i in samples]

# Parallel MCTS with Progressive Widening and detailed Multi Way analysis
class ParallelMCTS:
    def __init__(self, model, num_simulations=500, num_workers=4):
        self.model = model
        self.num_simulations = num_simulations
        self.num_workers = num_workers
        self.pool = multiprocessing.Pool(num_workers)
        self.analysis_log = []
        self.win_paths = []

    def simulate(self, root, sim_id):
        """Run one simulation with detailed analysis"""
        node = root
        tmp_game = node.game_instance.copy()
        depth = 0
        max_depth = 200
        path = []
        start_time = time.time()

        while not node.is_leaf() and depth < max_depth:
            node = node.select_child()
            if not node.resolved:
                try:
                    node.resolve()
                except:
                    break
            tmp_game = node.game_instance.copy()

            # Record move path
            action_desc = f"{node.action[0]}" if node.action and node.action[0] != "PASS" else "PASS"
            if node.action and node.action[0] != "PASS":
                cards = [f"{c.rank}_{c.suit[:1]}" for c in node.action[1]]
                action_desc += f":{','.join(cards)}"
            path.append({
                "player": tmp_game.current_player,
                "action": action_desc,
                "depth": depth,
                "timestamp": time.time() - start_time
            })

            depth += 1
            if tmp_game.done:
                break

        if not tmp_game.done:
            state = tmp_game.get_state()
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                _, value = self.model(state_tensor)
            result = value.item()
        else:
            result = 1.0 if tmp_game.winner == 0 else -1.0

        return {
            "id": sim_id,
            "path": path,
            "result": result,
            "depth": depth,
            "duration": time.time() - start_time,
            "win": result > 0
        }

    def run(self, game):
        """Run Multi Way MCTS with comprehensive analysis"""
        root = Node(game.copy(), model=self.model)
        root.resolve()

        # Run simulations in parallel
        results = []
        asyncs = [self.pool.apply_async(self.simulate, (root, i)) for i in range(self.num_simulations)]
        for a in asyncs:
            try:
                res = a.get(timeout=2.0)
                results.append(res)
            except multiprocessing.TimeoutError:
                continue

        # Analyze results
        win_count = sum(1 for r in results if r["win"])
        loss_count = len(results) - win_count
        win_rate = win_count / len(results) if results else 0

        # Gather stats for first-step moves
        action_stats = {}
        for r in results:
            for step in r["path"]:
                if step["depth"] == 0:
                    act = step["action"]
                    stats = action_stats.setdefault(act, {"wins": 0, "losses": 0, "visits": 0})
                    stats["visits"] += 1
                    if r["win"]:
                        stats["wins"] += 1
                    else:
                        stats["losses"] += 1

        # Compute action probabilities array
        action_probs = np.zeros(200)
        if action_stats:
            for act, stats in action_stats.items():
                # Extract action type for id mapping
                act_type = act.split(":")[0]
                idx = action_to_id((act_type, []))
                win_ratio = stats["wins"] / stats["visits"]
                action_probs[idx] = win_ratio * stats["visits"]
            total = action_probs.sum()
            if total > 0:
                action_probs /= total
        else:
            action_probs = np.ones(200) / 200

        # Log analysis entry
        self.analysis_log.append({
            "timestamp": datetime.now().isoformat(),
            "total_simulations": len(results),
            "win_rate": win_rate,
            "win_count": win_count,
            "loss_count": loss_count,
            "action_stats": action_stats
        })

        # Print summary
        print("\n===== MULTI WAY ANALYSIS =====")
        print(f"Total simulations: {len(results)}")
        print(f"Win rate: {win_rate:.2%} ({win_count}/{len(results)})")
        print("=============================\n")

        return action_probs, win_rate

# Original MCTS kept for fallback
class MCTS:
    def __init__(self, model, num_simulations=500, use_cache=True, use_continual_resolving=True):
        self.model = model
        self.num_simulations = num_simulations
        self.use_cache = use_cache
        self.use_continual_resolving = use_continual_resolving
        self.state_cache = {}
        self.action_cache = {}

    def run(self, game):
        root = Node(game.copy(), model=self.model)
        self.root = root  # store root for post-game summary
        try:
            root.resolve()
        except Exception as e:
            return np.zeros(200)
        
        start_time = time.time()
        for _ in range(self.num_simulations):
            if time.time() - start_time > 0.5:
                break
                
            node = root
            tmp_game = node.game_instance.copy()
            depth = 0
            max_depth = 200
            
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
                idx = action_to_id(child.action)
                action_probs[idx] = child.n / total_visits
        else:
            action_probs = np.ones(200) / 200
            
        return action_probs

# Node for MCTS with continual resolving
class Node:
    def __init__(self, game_instance, parent=None, action=None, model=None):
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
        self.resolved = False
        self.depth = (parent.depth + 1) if parent is not None and hasattr(parent, 'depth') else 0
        self.expansion_threshold = 5  # visits needed before expanding further

    def select_child(self):
        # progressive widening: expand when visits exceed threshold
        if not self.resolved and self.n >= self.expansion_threshold:
            self.resolve()
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
        if self.resolved or self.game_instance.current_player != 0:
            self.resolved = True
            return 0.0
        try:
            self.valid_actions = self.game_instance.get_valid_combos(0)
            state = self.game_instance.get_state()
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_logits, value = self.model(state_tensor)
            action_probs = F.softmax(action_logits, dim=1).cpu().numpy()[0]
            # progressive widening: expand top-k only initially
            k = min(5, len(self.valid_actions))
            scores = [action_probs[action_to_id(a)] for a in self.valid_actions]
            top_idxs = np.argsort(scores)[-k:]
            for i in top_idxs:
                action = self.valid_actions[i]
                game_copy = self.game_instance.copy()
                try:
                    game_copy.step(action)
                    child = Node(game_copy, self, action, self.model)
                    child.p = scores[i]
                    self.children.append(child)
                except:
                    continue
            self.resolved = True
            return value.item()
        except Exception:
            self.resolved = True
            return 0.0

    def is_leaf(self):
        return not self.children

# PPO Replay Buffer with trajectory storage
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
        last_value = 0

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

        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'returns': returns,
            'advantages': advantages,
            'values': values
        }

    def __len__(self):
        return len(self.states)

# DummyScaler for CPU training compatibility
class DummyScaler:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

def log_gpu_utilization():
    if torch.cuda.is_available():
        print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        print(f"[GPU] Memory reserved: {torch.cuda.memory_reserved()/1e6:.2f} MB")
        try:
            import subprocess
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            ).decode().strip().split('\n')[0]
            utilization = float(smi_out)
            print(f"[GPU] Utilization: {utilization:.1f}%")
        except Exception:
            print("[GPU] Could not get utilization, nvidia-smi not available or failed.")

# Guided Exploration
def guided_exploration(game_state, model, valid_actions, game):
    # Use model to predict potential actions
    state_tensor = torch.tensor(game_state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action_logits, _ = model(state_tensor)
        action_probs = F.softmax(action_logits, dim=1).cpu().numpy()[0]
    
    # Create action mapping
    action_ids = [action_to_id(action) for action in valid_actions]
    valid_probs = [action_probs[aid] for aid in action_ids]
    
    if sum(valid_probs) > 0:
        valid_probs = np.array(valid_probs) / sum(valid_probs)
    else:
        valid_probs = np.ones(len(valid_actions)) / len(valid_actions)
    
    # Heuristic: Prefer playing pairs/triples of low cards in early game
    total_played = 52 - sum(len(p.hand) for p in game.players)
    if total_played < 15:
        low_actions = []
        for i, action in enumerate(valid_actions):
            if action[0] in ["PAIR", "TRIPLE"]:
                if all(card.rank in ['3','4','5','6'] for card in action[1]):
                    low_actions.append(i)
        if low_actions:
            probs = np.zeros(len(valid_actions))
            for idx in low_actions:
                probs[idx] = valid_probs[idx] * 2.0  # Double probability
            if probs.sum() > 0:
                probs /= probs.sum()
                valid_probs = probs
                return np.random.choice(len(valid_actions), p=probs), valid_probs
    
    # Heuristic: Save bombs for late game
    if total_played < 30:
        bomb_actions = [i for i, action in enumerate(valid_actions) if action[0] == "BOMB"]
        if bomb_actions:
            probs = np.array(valid_probs)
            for idx in bomb_actions:
                probs[idx] *= 0.3  # Reduce bomb probability
            probs /= probs.sum()
            valid_probs = probs
            return np.random.choice(len(valid_actions), p=probs), valid_probs
    
    # Default: Use model probabilities
    return np.random.choice(len(valid_actions), p=valid_probs), valid_probs

# Enhanced self-play with Multi Way analysis using ParallelMCTS
def self_play_game(model, game_id, model_pool=None):
    game = TienLenGame(game_id)
    ppo_buffer = PPOBuffer()
    rewind_buffer = RewindBuffer(max_depth=5)

    # Model selection
    current_model = model
    if model_pool and random.random() < 0.25:
        model_path = random.choice(model_pool)
        current_model = TienLenNet().to(device)
        current_model.load_state_dict(torch.load(model_path, map_location=device))
        current_model.eval()

    # Use ParallelMCTS for Multi Way analysis
    mcts = ParallelMCTS(current_model, num_simulations=500, num_workers=4)
    win_prob = 0.5  # Default win probability

    # Play until game ends
    while not game.done:
        if game.current_player == 0:  # RL player
            state = game.get_state()
            rewind_buffer.push(state, None)

            # Run Multi Way MCTS analysis
            action_probs, win_prob = mcts.run(game)

            # Filter valid actions
            valid_actions = game.get_valid_combos(0)
            valid_probs = [action_probs[action_to_id(a)] for a in valid_actions]
            if sum(valid_probs) > 0:
                valid_probs = np.array(valid_probs) / sum(valid_probs)
            else:
                valid_probs = np.ones(len(valid_actions)) / len(valid_actions)

            # Sample action
            action_idx = np.random.choice(len(valid_actions), p=valid_probs)
            action = valid_actions[action_idx]
            log_prob = np.log(valid_probs[action_idx] + 1e-10)

            # Store experience
            ppo_buffer.store(state, action_idx, log_prob, win_prob, 0, False)
        else:
            action = game.suggest_bot_action(game.current_player)

        # Execute and record
        _, reward, done, _ = game.step(action)
        if game.current_player == 0 and ppo_buffer.rewards:
            ppo_buffer.rewards[-1] = reward
            if done:
                ppo_buffer.dones[-1] = True
                win_prob = 1.0 if game.winner == 0 else 0.0

    # Final reward shaping
    final_reward = 10.0 if game.winner == 0 else -10.0
    if ppo_buffer.rewards:
        ppo_buffer.rewards[-1] += final_reward

    # Build trajectory and attach latest analysis
    trajectory = ppo_buffer.get_trajectory()
    trajectory["mcts_analysis"] = mcts.analysis_log[-1] if hasattr(mcts, 'analysis_log') and mcts.analysis_log else {}

    return trajectory, game.winner, win_prob
    
    # Model selection
    current_model = model
    if model_pool and random.random() < 0.25:
        model_path = random.choice(model_pool)
        current_model = TienLenNet().to(device)
        current_model.load_state_dict(torch.load(model_path, map_location=device))
        current_model.eval()

    mcts = MCTS(current_model, num_simulations=500)
    turn_count = 0
    win_prob = 0.5  # Default win probability
    
    while not game.done and turn_count < 500:
        turn_count += 1
        current_player = game.current_player
        start_turn_time = time.time()

        if current_player == 0:  # RL player
            state = game.get_state()
            rewind_buffer.push(state, None)
            
            # Rewind Mechanism (40% chance to rewind)
            if np.random.rand() < 0.4:
                rewind_steps = np.random.randint(1, 4)
                rewind_state = rewind_buffer.rewind(rewind_steps)
                if rewind_state is not None:
                    state = rewind_state
            
            # Log previous BOT action
            if game.history:
                prev_player, prev_action = game.history[-1]
                if prev_player != 0:
                    bot_cards = ", ".join(f"{c.rank}_{c.suit[:1]}" for c in prev_action[1])
                    print(f"[BOT{prev_player}] played {prev_action[0]}: [{bot_cards}]", flush=True)
# Log RL current hand
            hand_cards = ", ".join(f"{c.rank}_{c.suit[:1]}" for c in game.players[0].hand)
            print(f"[RL] Hand: [{hand_cards}]", flush=True)
            valid_actions = game.get_valid_combos(current_player, use_cache=True)
            print(f"[RL] Valid moves count: {len(valid_actions)}")
            print("[RL] Valid moves list: " + ", ".join(
                f"{action[0]}: {[f'{c.rank}_{c.suit[:1]}' for c in action[1]]}"
                for action in valid_actions
            ))
            
            # Guided Exploration (30% chance)
            if np.random.rand() < 0.3:
                print("[LOG] Guided exploration triggered")  # Debug guided path
                action_idx, valid_probs = guided_exploration(state, current_model, valid_actions, game)
                log_prob = np.log(valid_probs[action_idx] + 1e-10)
                action = valid_actions[action_idx]
                print(f"[Guided] action={action[0]}: {[f'{c.rank}_{c.suit[:1]}' for c in action[1]]}, prob={valid_probs[action_idx]:.4f}")
            else:
                # Standard MCTS
                print("[LOG] Running MCTS simulations")  # Debug MCTS start
                action_probs = mcts.run(game)
                print(f"[MCTS] total_prob_sum={action_probs.sum():.4f}")
                valid_probs = []
                for action in valid_actions:
                    idx = action_to_id(action)
                    valid_probs.append(action_probs[idx])
                # Log full valid-action probabilities
                print("[MCTS] All valid-action probs: " + ", ".join(
                    f"{act[0]}:{prob:.4f}" for act, prob in zip(valid_actions, valid_probs)
                ))
                top5 = sorted(zip(valid_actions, valid_probs), key=lambda x: -x[1])[:5]
                print("[MCTS] Top5 candidates:", ", ".join(f"{a[0]}:{p:.4f}" for a,p in top5))
                
                if sum(valid_probs) > 0:
                    valid_probs = np.array(valid_probs) / sum(valid_probs)
                else:
                    valid_probs = np.ones(len(valid_actions)) / len(valid_actions)
                
                action_idx = np.random.choice(len(valid_actions), p=valid_probs)
                action = valid_actions[action_idx]
                log_prob = np.log(valid_probs[action_idx] + 1e-10)
            
            # Get value estimate
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, value = current_model(state_tensor)
            # Compute and log model policy over valid moves
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            valid_model_probs = [policy_probs[action_to_id(act)] for act in valid_actions]
            print("[PPO] Model policy for valid moves: " + ", ".join(
                f"{act[0]}:{prob:.4f}" for act, prob in zip(valid_actions, valid_model_probs)
            ))
            value = value.item()
            print(f"[PPO] log_prob={log_prob:.4f}, value={value:.4f}")  # Debug PPO metrics
            
            # Store in buffer
            ppo_buffer.store(state, action_idx, log_prob, value, 0, False)
            
            # Log action
            action_desc = f"{action[0]}: {[f'{c.rank}_{c.suit[:1]}' for c in action[1]]}"
            print(f"[RL] Player 0 action: {action_desc}")
            player_tag = "RL"
        else:  # Bot player
            action = game.suggest_bot_action(current_player)
            # Log every BOT action
            cards_str = ", ".join(f"{c.rank}_{c.suit[:1]}" for c in action[1])
            print(f"[BOT{current_player}] played {action[0]}: [{cards_str}]")
            player_tag = f"BOT{current_player}"
            # BOT action log removed
        
        # Execute action
        _, reward, done, _ = game.step(action)
        
        # Update reward for RL player
        if current_player == 0 and ppo_buffer.rewards:
            ppo_buffer.rewards[-1] = reward
            if done:
                ppo_buffer.dones[-1] = True
                win_prob = 1.0 if game.winner == 0 else 0.0
        
        turn_time = time.time() - start_turn_time
        # Removed detailed TURN logs to only show RL actions
    
    # Final reward for RL player
    if game.winner == 0:
        final_reward = 10.0
        win_prob = 1.0
    else:
        final_reward = -10.0
        win_prob = 0.0
        
    if ppo_buffer.rewards and game.current_player == 0:
        ppo_buffer.rewards[-1] += final_reward
    
# Log final game result
# Post-game summary of MCTS root children stats
    if hasattr(mcts, 'root') and mcts.root and mcts.root.children:
        print("[MCTS Summary] Root action stats (visits, avg_value):")
        for child in mcts.root.children:
            visits = child.n
            avg_value = (child.w / visits) if visits > 0 else 0.0
            action_desc = child.action[0]
            if child.action[1]:
                cards = [f"{c.rank}_{c.suit[:1]}" for c in child.action[1]]
                action_desc += ":" + ",".join(cards)
            print(f"  Action {action_desc} -> visits={visits}, avg_value={avg_value:.3f}")
    print(f"[GAME {game_id}] Winner: Player {game.winner}, win_prob={win_prob:.2f}")
    trajectory = None
    if len(ppo_buffer) > 0:
        trajectory = ppo_buffer.get_trajectory()
    
    return trajectory, game.winner, win_prob

# Main training loop
def main_train_loop(total_episodes=500):
    print("Initializing training process...")

    # Create model
    model = TienLenNet().to(device)
    print(f"Model architecture:\n{model}")

    # Load model if exists
    model_path = "saved_models/TienLenNetV1.pth"
    start_epoch = 0
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        start_epoch = int(model_path.split("_")[-1].split(".")[0]) if "epoch" in model_path else 0

    # Strategic replay buffer
    buffer = StrategicReplayBuffer(capacity=50000)

    # Model pool for prev models
    model_pool = []
    model_pool_size = 5
    saved_models = glob.glob("saved_models/tien_len_net_*.pth")
    saved_models.sort(key=os.path.getmtime, reverse=True)
    model_pool = saved_models[:model_pool_size]

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = DummyScaler()

    # Training parameters
    games_per_episode = 100
    training_steps = 50
    min_buffer_size = 200
    batch_size = 512
    best_win_rate = 0.0
    strong_performance_count = 0

    for episode in tqdm(range(start_epoch, total_episodes), desc="Training Episodes", unit="ep", initial=start_epoch, total=total_episodes):
        print(f"\n=== Episode {episode+1}/{total_episodes} | Buffer: {len(buffer)} | LR: {optimizer.param_groups[0]['lr']:.2e} ===")
        start_time = time.time()
        episode_wins = 0

        for i in range(games_per_episode):
            game_id = episode * games_per_episode + i
            print(f"Starting game {i+1}/{games_per_episode} (ID: {game_id})")
            trajectory, winner, win_prob = self_play_game(model, game_id, model_pool)
            if winner == 0:
                episode_wins += 1
            print(f"Game completed. {'RL Agent won!' if winner == 0 else 'Bot won.'}")
            if trajectory is not None:
                buffer.push(trajectory, win_prob)

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
        for step in tqdm(range(training_steps), desc=f"Training Episode {episode+1}", unit="step"):
            step_start = time.time()
            
            # Sample with priority to winning games
            trajectories = buffer.sample(batch_size, win_sample_ratio=0.7)
            
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy_loss = 0.0
            
            for traj in trajectories:
                states = torch.tensor(traj['states']).float().to(device)
                actions = torch.tensor(traj['actions']).long().to(device)
                old_log_probs = torch.tensor(traj['log_probs']).float().to(device)
                returns = torch.tensor(traj['returns']).float().to(device)
                advantages = torch.tensor(traj['advantages']).float().to(device)
                old_values = torch.tensor(traj['values']).float().to(device)
                
                # PPO update
                for _ in range(4):  # PPO epochs
                    # Get new policy and values
                    policy_pred, value_pred = model(states)
                    value_pred = value_pred.squeeze()

                    # Calculate new log probabilities
                    dist = torch.distributions.Categorical(logits=policy_pred)
                    new_log_probs = dist.log_prob(actions)

                    # Policy loss (clipped surrogate)
                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss (clipped)
                    value_pred_clipped = old_values + torch.clamp(value_pred - old_values, -0.2, 0.2)
                    value_loss1 = F.mse_loss(value_pred, returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, returns)
                    value_loss = torch.max(value_loss1, value_loss2).mean()

                    # Entropy bonus
                    entropy_loss = -dist.entropy().mean()

                    # Total loss
                    loss = policy_loss + 0.5 * value_loss + 0.05 * entropy_loss

                    # Backpropagation
                    optimizer.zero_grad()
                    if device.type == "cuda":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    
            avg_loss = total_loss / len(trajectories)
            train_losses.append(avg_loss)
            
            if step % 10 == 0:
                # calculate ETA for remaining training steps
                step_time = time.time() - step_start
                remaining_steps = training_steps - step - 1
                eta = remaining_steps * step_time
                eta_min = eta / 60.0
                print(f"Step {step+1}/{training_steps} | Loss: {avg_loss:.4f} | Time: {step_time:.2f}s | ETA: {eta_min:.2f}min")

        if train_losses:
            avg_epoch_loss = sum(train_losses) / len(train_losses)
            scheduler.step(avg_epoch_loss)

        # Early stopping if strong performance for 3 evals > 60%
        if (episode+1) % 10 == 0:
            print("Evaluating model...")
            eval_start = time.time()
            win_rate_eval = evaluate(model, num_games=20)
            eval_time = time.time() - eval_start
            print(f"Evaluation completed in {eval_time:.2f}s | Win rate: {win_rate_eval:.2f}")

            if win_rate_eval >= 0.6:
                strong_performance_count += 1
                print(f"Strong eval performance count: {strong_performance_count}/3")
                if strong_performance_count >= 3:
                    print("Stopping early - consistent strong performance!")
                    break
            else:
                strong_performance_count = 0

            if win_rate_eval > best_win_rate:
                best_win_rate = win_rate_eval
                best_path = f"saved_models/best_tien_len_net_ep{episode+1}_win{win_rate_eval:.2f}.pth"
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model with win rate {win_rate_eval:.2f}!")

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

# Evaluation function
def evaluate(model, num_games=20):
    model_wins = 0
    for game_id in range(num_games):
        game = TienLenGame(game_id)
        while not game.done:
            current_player = game.current_player
            state = game.get_state()
            
            if current_player == 0:  # Model's turn
                state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action_logits, _ = model(state_tensor)
                    action_probs = F.softmax(action_logits, dim=1).cpu().numpy()[0]
                
                valid_actions = game.get_valid_combos(current_player)
                if valid_actions:
                    valid_probs = []
                    for action in valid_actions:
                        idx = action_to_id(action)
                        valid_probs.append(action_probs[idx])
                    
                    if sum(valid_probs) > 0:
                        valid_probs = np.array(valid_probs) / sum(valid_probs)
                    else:
                        valid_probs = np.ones(len(valid_actions)) / len(valid_actions)
                    
                    action_idx = np.argmax(valid_probs)
                    action = valid_actions[action_idx]
                else:
                    action = ("PASS", [])
            else:
                action = game.suggest_bot_action(current_player)
                
            game.step(action)
            
        if game.winner == 0:
            model_wins += 1

    return model_wins / num_games

# Run training
if False and __name__ == "__main__":
    print("\n===== GPU & Torch Environment Info =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        try:
            import subprocess
            print("Running nvidia-smi for GPU details:")
            smi = subprocess.check_output(["nvidia-smi"]).decode()
            print(smi)
        except Exception:
            print("nvidia-smi not available or failed.")
    else:
        print("Warning: CUDA not available! Training will run on CPU and be much slower.")
    print("========================================\n")

    try:
        n_episodes = int(input("Enter number of training episodes (default 100): ") or 100)
    except Exception:
        n_episodes = 100

    try:
        main_train_loop(n_episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt)!")
        import sys
        main_globals = sys.modules['__main__'].__dict__
        model = main_globals.get('model', None)
        try:
            if model is not None:
                torch.save(model.state_dict(), "saved_models/interrupted_model.pth")
                print("Model state saved at: saved_models/interrupted_model.pth")
        except Exception as e:
            print(f"Could not save interrupted model: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
# Single-game test entrypoint
if __name__ == "__main__" and False:
    pass  # Original training entry disabled

if __name__ == "__main__":
    # Initialize a fresh model for testing
    model = TienLenNet().to(device)
    print("===== Single Test Game =====")
    model.eval()
    trajectory, winner, win_prob = self_play_game(model, game_id=0, model_pool=[])
    print(f"Test game finished. Winner: {winner}, Win probability: {win_prob:.2f}")