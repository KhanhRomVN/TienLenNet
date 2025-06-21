from typing import List
from .card import Card
import pygame

class Player:
    def __init__(self, id: str, name: str, position: str, is_ai=False):
        self.id = id
        self.name = name
        self.position = position  # 'bottom', 'left', 'top', 'right'
        self.hand: List[Card] = []
        self.is_turn = False
        self.passed = False
        self.z_index_counter = 0
        self.is_ai = is_ai  # AI flag

        # RL reward system variables
        self.score = 0  # Số điểm hiện tại
        self.move_count = 0
        self.cards_at_start = 13
        self.high_cards_played = 0
        self.bombs_used = 0
        self.consecutive_passes_forced = 0
        self.low_cards_played_early = 0
        self.invalid_moves = 0
        self.cards_lost = 0
        self.last_move_type = None
        self.consecutive_pass_count = 0

    def add_card(self, card: Card):
        self.hand.append(card)
        self.sort_hand()

    def remove_cards(self, cards: List[Card]):
        for card in cards:
            if card in self.hand:
                self.hand.remove(card)
        self.cards_lost += len(cards)

    def sort_hand(self):
        self.hand.sort(key=lambda x: (Card.RANKS.index(x.rank), Card.SUITS.index(x.suit)))

    def get_selected_cards(self) -> List[Card]:
        return [card for card in self.hand if card.selected]

    def update_invalid_states(self, last_played_cards=None, game_first_turn=False):
        """
        Validate selected combo, set card.invalid for each card;
        - game_first_turn: first turn of first game (requires 3♠ if no card on table)
        """
        selected = [card for card in self.hand if card.selected]
        if last_played_cards is None:
            last_played_cards = getattr(self, '_last_played_cards', None)
        valid = False
        if selected:
            if game_first_turn and not last_played_cards:
                valid = any(card.rank == '3' and card.suit == 'spades' for card in selected)
            else:
                valid = self.can_play_cards(selected, last_played_cards)
        for card in self.hand:
            card.invalid = card.selected and not valid

    def clear_selection(self):
        for card in self.hand:
            card.selected = False

    def reset_state(self):
        self.passed = False
        self.clear_selection()

    # RL Reward functions -----------------------------
    def add_reward(self, value, reason=""):
        # Hệ số thưởng/phạt nâng cao:
        multiplier = 1.0
        # Nếu gần hết bài
    def get_dynamic_reward(self, action, game_state, total_moves=0, explored_combos=None):
        """
        Tặng thưởng/phạt linh hoạt cho AI tùy tình huống game và giai đoạn học
        """
        base_reward = self.calculate_base_reward(action)
        explored_combos = explored_combos or set()

        # Bước 1: Thưởng cho đánh bài sớm/làm chủ – giai đoạn học vỡ lòng
        if total_moves < 100:
            if action != "PASS":
                base_reward += 0.5 * (len(action) if isinstance(action, list) else 1)
            else:
                base_reward -= 0.3

        # Bước 2: Thưởng nếu đánh ra loại combo mới với AI
        if hasattr(self, "combo_history"):
            combo_key = tuple(sorted([(c.rank, c.suit) for c in action])) if isinstance(action, list) else tuple()
            if combo_key and combo_key not in self.combo_history:
                base_reward += 1.0
                self.combo_history.add(combo_key)
        if explored_combos and isinstance(action, list):
            combo_key = tuple(sorted([(c.rank, c.suit) for c in action]))
            if combo_key not in explored_combos:
                base_reward += 1.0
                explored_combos.add(combo_key)
        
        return base_reward

    def calculate_base_reward(self, action):
        # Hưởng cơ bản: có thể tăng cường logic recognition
        # Đánh bài được thưởng, pass bị phạt nhẹ
        if action == "PASS":
            return -0.1
        return 0.2 * (len(action) if isinstance(action, list) else 1)
        if len(self.hand) <= 3:
            multiplier = 2.0
        # "Phục hồi thế cờ" — vừa mất ≥5 bài mà còn ≤5 lá
        elif self.cards_lost >= 5 and len(self.hand) <= 5:
            multiplier = 1.5
        self.score += value * multiplier
        print(f"[REWARD] {self.name}: {value:.2f} x {multiplier:.1f} = {value * multiplier:.2f}   ({reason})")

    def update_after_move(self, cards_played):
        self.move_count += 1
        if cards_played:
            self.add_reward(0.2 * len(cards_played), "Chơi bài hợp lệ")
            if len(cards_played) == 1:
                self.add_reward(0.3, "Thoát bài khó")
            if getattr(self, "last_played_by", None) is None and len(cards_played) <= 2:
                self.add_reward(0.8, "Dẫn đầu bằng combo nhỏ")
        high_ranks = ['2', 'A', 'K']
        for card in cards_played or []:
            if card.rank in high_ranks:
                self.high_cards_played += 1
                if self.move_count <= 5:
                    self.add_reward(-1.0, "Lãng phí bài cao sớm")
        low_ranks = ['3', '4', '5', '6', '7']
        for card in cards_played or []:
            if card.rank in low_ranks and self.move_count <= 10:
                self.low_cards_played_early += 1
                self.add_reward(0.2, "Thoát bài thấp sớm")
        self.consecutive_pass_count = 0

        # Layer reward: tích hợp phần thưởng chiến thuật/multi-objective
        self.update_strategy_rewards()

    def update_after_pass(self):
        self.consecutive_pass_count += 1
        if self.consecutive_pass_count >= 3:
            self.add_reward(-0.2, "Bế tắc")

    def update_after_block(self, success=True):
        if success:
            self.add_reward(2.0, "Chặn thành công")
        else:
            self.add_reward(-1.5, "Bị chặn")
        print(f"Player {self.id} {'chặn thành công' if success else 'bị chặn'}")

    def update_strategy_rewards(self):
        # --- Các tín hiệu chiến thuật reward theo REWARD.md ---
        # Giảm 50% bài
        if len(self.hand) <= 6 and self.cards_at_start == 13:
            self.add_reward(8.0, "Giảm 50% số bài")
            self.cards_at_start = len(self.hand)
        high_ranks = ['2', 'A', 'K']
        # Giữ bài cao đến cuối
        if len(self.hand) <= 5 and any(card.rank in high_ranks for card in self.hand):
            self.add_reward(0.5, "Giữ bài cao đến cuối")
        # Tích lũy combo tiềm năng
        if self.count_available_combos() >= 3:
            self.add_reward(0.3, "Tích lũy combo tiềm năng")
        # Tiết kiệm bài cao
        if self.saved_high_card():
            self.add_reward(0.4, "Tiết kiệm bài cao")
        # Mắc kẹt bài lẻ
        if self.count_orphan_cards() >= 3:
            self.add_reward(-0.7, "Mắc kẹt bài lẻ")
        # Tối ưu hóa sảnh
        if self.created_long_sequence():
            self.add_reward(0.6, "Tối ưu hóa sảnh")
        # Thưởng phá vỡ bế tắc
        if self.consecutive_pass_count >= 2:
            self.add_reward(0.5, "Phá vỡ bế tắc")
        # Thưởng tạo combo dài liên tục
        if self.created_long_sequence():
            self.add_reward(0.8, "Tạo combo dài")
        # Phạt giữ bài cao khi còn nhiều bài
        if len(self.hand) >= 8 and any(card.rank in high_ranks for card in self.hand):
            self.add_reward(-0.3, "Giữ bài cao không cần thiết")


    def count_available_combos(self):
        return 2
    def count_orphan_cards(self):
        return 0
    def saved_high_card(self):
        return False
    def created_long_sequence(self):
        return False

    # ---------------- Gameplay/core logic -----------------
    def validate_selection(self, selected: List["Card"]) -> bool:
        """
        Kiểm tra sự hợp lệ của bộ bài được chọn, PHẢI xét luôn luật chặn: nếu trên bàn đã ra sảnh hay đôi,
        phải bắt buộc lớn hơn, cùng số lá, đúng loại combo, đúng chất với sảnh nếu cùng loại.
        """
        if not selected:
            return False
        from collections import Counter
        n = len(selected)
        ranks = [Card.RANKS.index(c.rank) for c in selected]
        unique_ranks = set(ranks)

        # Nếu đang phải chặn sảnh hoặc combo đặc biệt, phải sử dụng can_play_cards để chuẩn
        try:
            from src.game.main import TienLenGame
        except ImportError:
            TienLenGame = None

        # Tìm context bàn hiện tại (nếu có tham chiếu _last_played_cards)
        last_played = getattr(self, '_last_played_cards', None)

        # Nếu trên bàn đã có combo, kiểm tra luôn bằng can_play_cards (sẽ so sánh luật chặn chuẩn)
        if last_played:
            return self.can_play_cards(selected, last_played)
        
        # Single card
        if n == 1: return True
        # Pair/triple/quad
        if n in (2, 3, 4) and len(unique_ranks) == 1: return True
        # Straight
        if n >= 3:
            if len(unique_ranks) != n: return False
            sorted_ranks = sorted(unique_ranks)
            if all(sorted_ranks[i]+1==sorted_ranks[i+1] for i in range(n-1)):
                if Card.RANKS.index('2') in sorted_ranks[:-1]:
                    return False
                return True
        # Pairs straight (đôi thông)
        if n >= 6 and n % 2 == 0:
            rank_groups = {}
            for card in selected:
                rank_groups.setdefault(card.rank, []).append(card)
            if not all(len(group) == 2 for group in rank_groups.values()):
                return False
            sorted_ranks = sorted(rank_groups.keys(), key=lambda r: Card.RANKS.index(r))
            rank_indices = [Card.RANKS.index(r) for r in sorted_ranks]
            if all(rank_indices[i]+1==rank_indices[i+1] for i in range(len(rank_indices)-1)):
                if '2' in sorted_ranks[:-1]:
                    return False
                return True
        return False

    def can_play_cards(self, selected_cards: List[Card], last_played_cards: List[Card] = None) -> bool:
        if not selected_cards:
            return False

        # Kiểm tra combo hợp lệ cho lượt đầu ván (nếu chưa có bài trên bàn)
        if not last_played_cards:
            # SỬA: Chỉ yêu cầu combo hợp lệ, không yêu cầu 3♠
            return self.validate_selection(selected_cards)

        combo_type = self.get_combo_type(selected_cards)
        if combo_type == "INVALID":
            return False
        prev_combo_type = self.get_combo_type(last_played_cards)
        # Special beating rules
        if prev_combo_type == "FOUR_OF_A_KIND":
            if combo_type not in ["FOUR_OF_A_KIND", "PAIRS_STRAIGHT"]:
                return False
        elif prev_combo_type == "PAIRS_STRAIGHT":
            if combo_type not in ["PAIRS_STRAIGHT", "FOUR_OF_A_KIND"]:
                return False
        elif combo_type != prev_combo_type:
            return False
        return self.compare_combinations(selected_cards, last_played_cards) > 0

    def get_combo_type(self, cards: List[Card]) -> str:
        if not cards: return "INVALID"
        from collections import Counter
        n = len(cards)
        rank_counts = Counter(c.rank for c in cards)
        unique_ranks = set(rank_counts.keys())
        rank_indices = sorted([Card.RANKS.index(r) for r in unique_ranks])
        if n == 1: return "SINGLE"
        if n == 2 and len(unique_ranks) == 1: return "PAIR"
        if n == 3 and len(unique_ranks) == 1: return "THREE"
        if n == 4 and len(unique_ranks) == 1: return "FOUR_OF_A_KIND"
        if n >= 3:
            if (len(unique_ranks) == n and all(rank_indices[i]+1==rank_indices[i+1] for i in range(len(rank_indices)-1))):
                if Card.RANKS.index('2') in rank_indices[:-1]:
                    return "INVALID"
                return "STRAIGHT"
        if n >= 6 and n % 2 == 0:
            if all(count == 2 for count in rank_counts.values()):
                if all(rank_indices[i]+1==rank_indices[i+1] for i in range(len(rank_indices)-1)):
                    if Card.RANKS.index('2') in rank_indices[:-1]:
                        return "INVALID"
                    return "PAIRS_STRAIGHT"
        return "INVALID"

    def compare_combinations(self, cards1: List[Card], cards2: List[Card]) -> int:
        if len(cards1) in [1, 2, 3]:
            max1 = max(cards1)
            max2 = max(cards2)
            if Card.RANKS.index(max1.rank) > Card.RANKS.index(max2.rank):
                return 1
            elif Card.RANKS.index(max1.rank) < Card.RANKS.index(max2.rank):
                return -1
            return 1 if Card.SUITS.index(max1.suit) > Card.SUITS.index(max2.suit) else -1
        if len(cards1) >= 3:
            max1 = max(cards1)
            max2 = max(cards2)
            if Card.RANKS.index(max1.rank) > Card.RANKS.index(max2.rank):
                return 1
            elif Card.RANKS.index(max1.rank) < Card.RANKS.index(max2.rank):
                return -1
            return 1 if Card.SUITS.index(max1.suit) > Card.SUITS.index(max2.suit) else -1
        if len(cards1) == 4:
            rank1 = cards1[0].rank
            rank2 = cards2[0].rank
            return 1 if Card.RANKS.index(rank1) > Card.RANKS.index(rank2) else -1
        return 0

    # UI/interaction -------------
    def handle_click(self, pos, last_played_cards=None, game_first_turn=False):
        """
        Khi click card: toggle chọn, validate lại các lá đang chọn (border HỢP LỆ/ERROR - teal/red)
        Thêm debug log các trạng thái khi click.
        """
        print(f"[CLICK] pos={pos} | is_turn={self.is_turn} | passed={self.passed} | is_ai={getattr(self,'is_ai',False)}")
        if self.is_turn and not self.passed and not getattr(self, 'is_ai', False):
            # THÊM: Reset trạng thái invalid trước khi xử lý click
            for card in self.hand:
                card.invalid = False

            for card in sorted(self.hand, key=lambda c: c.z_index, reverse=True):
                print(f"[DEBUG] Card {card} - rect={card.rect} - selected={card.selected}")
                if card.rect.collidepoint(pos):
                    card.toggle_selected()
                    print(f"[DEBUG] => Card {card} được {'chọn' if card.selected else 'bỏ chọn'}!")
                    # CẬP NHẬT: Validate ngay sau khi click
                    self.update_invalid_states(last_played_cards, game_first_turn)
                    # In trạng thái từng card sau khi validate
                    for c in self.hand:
                        print(f"[STATE] {c}: selected={c.selected} | invalid={getattr(c, 'invalid', None)}")
                    return True
        return False

    def update_hover(self, mouse_pos, last_played_cards=None, game_first_turn=False):
        if self.is_turn:
            for card in self.hand:
                card.update_hover(mouse_pos)
            self.update_invalid_states(last_played_cards, game_first_turn)

    def draw_hand(self, screen, card_spacing=None):
        if not self.hand:
            return

        if self.position == 'bottom':
            self.z_index_counter = 0
            if card_spacing is None:
                max_spacing = min(Card.CARD_WIDTH * 0.6, (screen.get_width() * 0.9) / len(self.hand))
                card_spacing = max(Card.CARD_WIDTH * 0.3, max_spacing)
            total_width = (len(self.hand) - 1) * card_spacing + Card.CARD_WIDTH
            start_x = (screen.get_width() - total_width) // 2
            y = screen.get_height() - Card.CARD_HEIGHT - 30
            for i, card in enumerate(self.hand):
                card.z_index = self.z_index_counter
                x = start_x + i * card_spacing
                card.face_up = True  # Always show player's hand face up
                card.draw(screen, (x + Card.CARD_WIDTH//2, y + Card.CARD_HEIGHT//2))
                self.z_index_counter += 1
        else:
            # Enlarged card display (non-bottom): 90% size, edges closer
            card_back_color = (255, 255, 255)
            border_color = (0, 0, 0)
            card_scale = 0.9  # 90% of base card size
            w = int(Card.CARD_WIDTH * card_scale)
            h = int(Card.CARD_HEIGHT * card_scale)
            font = pygame.font.SysFont('Arial', int(21 * card_scale), bold=True)

            edge_margin = 20
            vertical_center = screen.get_height() // 2

            if self.position == 'top':
                card_x = screen.get_width() // 2 - w // 2
                card_y = 40
                card_rect_pos = (card_x, card_y)
            elif self.position == 'left':
                card_x = edge_margin
                card_y = vertical_center - h // 2
                card_rect_pos = (card_x, card_y)
            elif self.position == 'right':
                card_x = screen.get_width() - w - edge_margin
                card_y = vertical_center - h // 2
                card_rect_pos = (card_x, card_y)
            else:
                card_x = screen.get_width() // 2 - w // 2
                card_y = 40
                card_rect_pos = (card_x, card_y)

            card_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card_surf, card_back_color, (0, 0, w, h), border_radius=8)
            pygame.draw.rect(card_surf, border_color, (0, 0, w, h), 2, border_radius=8)
            count_text = font.render(str(len(self.hand)), True, (100, 100, 100))
            tx = w // 2 - count_text.get_width() // 2
            ty = h // 2 - count_text.get_height() // 2
            card_surf.blit(count_text, (tx, ty))
            screen.blit(card_surf, card_rect_pos)
            return