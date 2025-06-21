# Patch: import for test vs module run
if __name__ == "__main__":
    try:
        from .model import DQNAgent
        from .state_encoder import StateEncoder
        from .agent import AIAgent
    except (ImportError, SystemError):
        # Fallback absolute import if run as script
        from src.ai.model import DQNAgent
        from src.ai.state_encoder import StateEncoder
        from src.ai.agent import AIAgent
import numpy as np
import torch
from src.game.card import Card
from .model import DQNAgent
from .state_encoder import StateEncoder
from .virtual_chef import VirtualChef
import random

class AIAgent:
    def __init__(self, player_index, model_path=None):
        self.player_index = player_index
        self.state_encoder = StateEncoder()
        self.action_size = 100  # Số lượng hành động tối đa (chỉnh tuỳ complexity)

        # Sử dụng GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use 128 for state dim to match state encoder
        self.agent = DQNAgent(128, self.action_size, device=self.device)

        if model_path:
            self.agent.load(model_path)

    def get_valid_actions(self, game_state):
        """
        Trả về danh sách các action_id hợp lệ cho RL agent.
        Sửa: không chỉ dựa vào bài trên tay, mà còn kiểm tra từng action có thực sự hợp lệ so với luật chặn.
        """
        player = game_state.players[self.player_index]
        last_played = getattr(game_state, "last_played_cards", [])
        must_play_first = not last_played
        has_three_spades = any(card.rank == '3' and card.suit == 'spades' for card in player.hand)

        # SỬA QUAN TRỌNG: Nếu là lượt đầu và có 3♠, chỉ phải đánh đúng 3♠, loại bỏ PASS hoàn toàn
        if must_play_first and has_three_spades:
            valid_actions = []
            for card in player.hand:
                if card.rank == '3' and card.suit == 'spades':
                    idx = Card.RANKS.index(card.rank) * 4 + Card.SUITS.index(card.suit)
                    valid_actions.append(1 + idx)
            print(f"[VALID_ACTIONS] Player {self.player_index} MUST_PLAY_FIRST=True, has_3♠: {has_three_spades}, valid={valid_actions}")
            return valid_actions  # Không cho PASS

        # Mọi trường hợp khác: luôn cho phép PASS + check đủ mọi bài hợp lệ
        valid_actions = [0]

        from itertools import combinations

        # Sinh mọi tổ hợp combo hợp lệ:
        # - single (1), đôi (2), bộ ba (3), tứ quý (4)
        # - sảnh liên tiếp [min=3], đôi thông [min=6]
        # (nâng tầm: len_hand+1 để xét full combos nhiều lá)
        max_n = min(13, len(player.hand))  # tối đa 13 lá
        for n in range(1, max_n+1):
            for cards in combinations(player.hand, n):
                # Đảm bảo tính các combo hợp lệ lớn hơn 4: sảnh (≥3), đôi thông (≥6, chẵn)
                # Bỏ qua tổ hợp quá lớn (không đúng luật tiến lên)
                if n > 4 and n < 6:  # skip n=5 (ko có loại combo hợp lệ)
                    continue
                if player.can_play_cards(list(cards), last_played):
                    if len(cards) == 1:
                        # Single (1..52)
                        c = cards[0]
                        idx = Card.RANKS.index(c.rank) * 4 + Card.SUITS.index(c.suit)
                        action_id = 1 + idx
                        if action_id not in valid_actions:
                            valid_actions.append(action_id)
                    else:
                        combo_key = tuple(sorted([(Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)) for card in cards]))
                        combo_hash = abs(hash(combo_key)) % 10000
                        action_id = 53 + (combo_hash % 47)
                        if action_id not in valid_actions:
                            # DEBUG cho các combo dài hơn 4
                            if len(cards) > 4:
                                print(f"[DEBUG] ADD combo n={len(cards)}: {[(c.rank, c.suit) for c in cards]}")
                            valid_actions.append(action_id)

        print(f"[VALID_ACTIONS] Player {self.player_index} must_play_first={must_play_first}, has_3♠={has_three_spades}, last_played={len(last_played)}, valid_count={len(valid_actions)}, valid={valid_actions}")

        return valid_actions

    def can_beat(self, last_cards, new_cards):
        """Chỉ so sánh lá đơn: new_cards có lớn hơn last_cards không?"""
        if not last_cards or not new_cards:
            return False
        if len(new_cards) != len(last_cards):
            return False
        # So sánh theo luật — chỉ áp dụng single card, có thể mở rộng combo về sau
        def card_value(card):
            return Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)
        return card_value(new_cards[0]) > card_value(last_cards[0])
    def select_action(self, game_state, exploration_eps=0.05):
        """Chọn hành động: Khám phá dùng VirtualChef, khai thác dùng Q-value"""
        state = self.state_encoder.encode(game_state, self.player_index)
        if len(state) != 128:
            print(f"[ERROR] Invalid state shape: {len(state)}")
            state = np.zeros(128, dtype=np.float32)
        valid_actions = self.get_valid_actions(game_state)
        print(f"[SELECT_ACTION] Player {self.player_index}, Valid actions: {valid_actions}")

        # Epsilon-greedy: khám phá với VirtualChef
        eps = self.agent.epsilon if hasattr(self.agent, "epsilon") else exploration_eps
        if random.random() < eps:
            chosen = VirtualChef.suggest_random_valid_move(valid_actions)
            print(f"[SELECT_ACTION][EXPLORE] VirtualChef chọn: {chosen}")
            return chosen

        # Khai thác Q-value
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.policy_net(state_tensor).cpu().numpy()[0]
        print(f"[SELECT_ACTION] Player {self.player_index}, Q-values: {q_values}")

        # Bảo vệ trường hợp Q-value thiếu chỉ số action
        valid_q = [
            q_values[a] if a < len(q_values) else -np.inf
            for a in valid_actions
        ]
        best_action_idx = np.argmax(valid_q)
        print(f"[SELECT_ACTION][EXPLOIT] Player {self.player_index}, Chose: {valid_actions[best_action_idx]}")
        return valid_actions[best_action_idx]

    def interpret_action(self, action_id, game_state):
        """Chuyển action_id thành action game"""
        if action_id == 0:
            return "PASS"
        # Chỉ giải mã bài đơn nếu action_id thuộc khoảng hợp lệ (1–52)
        if 1 <= action_id <= 52:
            action_id -= 1
            rank_idx = action_id // 4
            suit_idx = action_id % 4
            if (0 <= rank_idx < len(Card.RANKS) and
                0 <= suit_idx < len(Card.SUITS)):
                rank = Card.RANKS[rank_idx]
                suit = Card.SUITS[suit_idx]
                # Chỉ tìm bài nằm trong tay người chơi
                player = game_state.players[self.player_index]
                for card in player.hand:
                    if card.rank == rank and card.suit == suit:
                        return [card]
        return "PASS"
# TEST FOR get_valid_actions - debug full combos
if __name__ == "__main__":
    from src.game.card import Card
    from src.game.player import Player
    class DummyGame:
        pass
    # Tạo tay bài có single, đôi, sảnh, ba lá, tứ quý 
    hand = [
        Card('spades', '3'),
        Card('spades', '4'),
        Card('hearts', '4'),
        Card('spades', '5'),
        Card('diamonds', '5'),
        Card('clubs', '5'),
        Card('hearts', '5'),  # tứ quý 5
        Card('spades', '6'),
        Card('spades', '7'),
        Card('clubs', '4'),  # tạo đủ 3 đôi, sảnh 3-4-5-6-7, ba lá, tứ quý
    ]
    player = Player("Test", "bottom", is_ai=True)
    player.hand = hand
    game_state = DummyGame()
    game_state.players = [player]
    game_state.last_played_cards = []  # Giả lập lượt đầu 
    agent = AIAgent(0)
    actions = agent.get_valid_actions(game_state)
    print("[TEST] Valid actions:", actions)
    print("[TEST] Action count:", len(actions))
    # Test giả lập tình huống phải chặn một đôi 4
    game_state.last_played_cards = [Card('spades', '4'), Card('clubs', '4')]
    actions2 = agent.get_valid_actions(game_state)
    print("[TEST] Valid vs pair of 4:", actions2)
    print("[TEST] Action count:", len(actions2))
    # Test với sảnh vừa vừa
    game_state.last_played_cards = [Card('spades', '3'), Card('spades', '4'), Card('spades', '5'), Card('spades', '6'), Card('spades', '7')]
    actions3 = agent.get_valid_actions(game_state)
    print("[TEST] Valid vs 3-4-5-6-7:", actions3)
    print("[TEST] Action count:", len(actions3))
    # Đảm bảo log các combo đúng