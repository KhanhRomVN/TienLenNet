import numpy as np
from src.game.card import Card
from src.game.player import Player

class StateEncoder:
    @staticmethod
    def encode(game_state, player_index):
        """
        Encode game state to 128-dim vector for RL agent (compact version).
        [0-51]: player's hand (one-hot for each card)
        [52-75]: center cards (one-hot for each card, or truncated to 24-dim)
        [76-99]: last played cards (one-hot or compressed)
        [100-115]: summary for other players (hand count ratio & passed, up to 4 players)
        [116-127]: misc. turn/state summary
        """
        print(f"[STATE] Encoding for player {player_index}")
        try:
            encoded = np.zeros(128, dtype=np.float32)
            idx = 0

            # 0-51: Player hand (52 bits) --- SỬA lại encode: đánh dấu từng lá bài đúng vị trí!
            player = game_state.players[player_index]
            for card in player.hand:
                rank_idx = Card.RANKS.index(card.rank)
                suit_idx = Card.SUITS.index(card.suit)
                card_idx = rank_idx * 4 + suit_idx
                if 0 <= card_idx < 52:
                    encoded[card_idx] = 1.0

            # DEBUG print encode state
            print(f"[STATE_ENCODE] Player {player_index} - Hand: {sum(encoded[:52])} cards, Center: {sum(encoded[52:76])}, Last: {sum(encoded[76:100])}")

            # 52-75: center cards (truncate after first 24, or aggregate if more)
            if hasattr(game_state, 'center_cards'):
                for card in game_state.center_cards[:24]:
                    idx2 = Card.RANKS.index(card.rank)*4 + Card.SUITS.index(card.suit)
                    encoded[52 + idx2 % 24] = 1  # overlap possible for more than 24, rare
            # 76-99: last played cards (also up to 24)
            # 76-81: type of last_played combo, 82-94: rank info
            if hasattr(game_state, 'last_played_cards') and game_state.last_played_cards:
                last_cards = game_state.last_played_cards
                # Use Player method to get combo type, fallback = "SINGLE"
                try:
                    combo_type = Player.get_combo_type(Player, last_cards)
                except Exception:
                    combo_type = "SINGLE"
                combo_type_list = ["SINGLE", "PAIR", "THREE", "STRAIGHT", "FOUR_OF_A_KIND", "PAIRS_STRAIGHT"]
                if combo_type in combo_type_list:
                    type_idx = combo_type_list.index(combo_type)
                    encoded[76 + type_idx] = 1
                # Encode rank of max card for last_played
                max_card = max(last_cards, key=lambda c: (Card.RANKS.index(c.rank), Card.SUITS.index(c.suit)))
                rank_idx = Card.RANKS.index(max_card.rank)
                encoded[82 + rank_idx] = 1
            # 100-115: other players' summary (2 per player except self: hand count ratio, passed)
            other_idx = 100
            for i, p in enumerate(game_state.players):
                if i == player_index:
                    continue
                encoded[other_idx] = len(p.hand) / 13
                encoded[other_idx + 1] = 1 if hasattr(p, 'passed') and p.passed else 0
                other_idx += 2
                if other_idx > 115:
                    break
            # 116-127: misc. turn/game summary (12 bits)
            encoded[116] = player_index
            encoded[117] = getattr(game_state, 'current_player_index', 0)
            encoded[118] = len(getattr(game_state, 'last_played_cards', []))
            encoded[119] = getattr(game_state, 'consecutive_passes', 0)
            # 120: số lá bài còn lại của player này (bình thường 1.0, nhỏ dần)
            encoded[120] = len(player.hand) / 13.0

            # 121-123: số lá bài còn lại của các player khác (tối đa 3), normalized
            idx_other = 121
            for i, p in enumerate(game_state.players):
                if i == player_index:
                    continue
                encoded[idx_other] = len(p.hand) / 13.0
                idx_other += 1
                if idx_other > 123:
                    break

            # 124-127: tổng số bài đã đánh của từng player (tối đa 4, normalized)
            # Nếu cần có thể custom theo những gì agent cần nhất
            total_played = [13 - len(p.hand) for p in game_state.players]
            for i in range(4):
                if 124 + i < 128 and i < len(total_played):
                    encoded[124 + i] = total_played[i] / 13.0

            return encoded
        except Exception as e:
            print(f"[ENCODE ERROR] {str(e)}")
            return np.zeros(128, dtype=np.float32)