from .player import Player
from .card import Card
from random import shuffle

class GameLogic:
    @staticmethod
    def create_deck():
        deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                deck.append(Card(suit, rank))
        shuffle(deck)
        return deck

    @staticmethod
    def find_starting_player(players):
        """Find the player with 3♠ to start the game"""
        for i, player in enumerate(players):
            for card in player.hand:
                if card.rank == '3' and card.suit == 'spades':
                    return i
        return 0  # fallback

    @staticmethod
    def check_instant_win(player: Player) -> bool:
        cards = player.hand

        # Check for Sảnh Rồng (12 consecutive cards)
        sorted_cards = sorted(cards, key=lambda x: Card.RANKS.index(x.rank))
        ranks = [Card.RANKS.index(c.rank) for c in sorted_cards]
        if len(ranks) >= 12:
            for i in range(len(ranks) - 11):
                if all(ranks[j] + 1 == ranks[j + 1] for j in range(i, i + 11)):
                    return True

        # Check for 5 Đôi Thông (5 consecutive pairs)
        rank_groups = {}
        for card in cards:
            rank_groups.setdefault(card.rank, []).append(card)
        pairs = [r for r, group in rank_groups.items() if len(group) == 2]
        if len(pairs) >= 5:
            pairs.sort(key=lambda x: Card.RANKS.index(x))
            for i in range(len(pairs) - 4):
                if all(Card.RANKS.index(pairs[j]) + 1 == Card.RANKS.index(pairs[j + 1])
                       for j in range(i, i + 4)):
                    return True

        # Check for 6 Đôi Bất Kỳ (6 pairs of any rank)
        if len([r for r, group in rank_groups.items() if len(group) == 2]) >= 6:
            return True

        # Check for Tứ Quý 2 (four 2s)
        if len(rank_groups.get('2', [])) == 4:
            return True

        # Check for 6 Lá Cùng Số (6 cards of same rank)
        if any(len(group) >= 6 for group in rank_groups.values()):
            return True

        # Check for Đồng Chất (all cards of same suit)
        suit_groups = {}
        for card in cards:
            suit_groups.setdefault(card.suit, []).append(card)
        if any(len(group) == 13 for group in suit_groups.values()):
            return True

        return False