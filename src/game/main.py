import time
def debug_log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("debug_tienlen.log", "a") as f:
        f.write(f"[{t}] {msg}\n")

def history_log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("tienlen_history.log", "a") as f:
        f.write(f"[{t}] {msg}\n")
from dotenv import load_dotenv
load_dotenv()
import sys
from .card import Card
from .player import Player
from .game_logic import GameLogic
from .ui import constants as UI
import os
import json
from src.game.gemini_agent import GeminiAgent

class TienLenGame:
    def arrange_center_cards(self):
        """Sắp xếp/vị trí các lá bài ở giữa bàn chơi cho hiển thị"""
        n = len(self.center_cards)
        if n == 0:
            self.center_cards_pos = []
            return

        card_w = Card.CARD_WIDTH
        spacing = min(card_w * 0.6, (self.WINDOW_WIDTH * 0.9) / n)
        spacing = max(card_w * 0.3, spacing)
        total_width = (n - 1) * spacing + card_w
        start_x = (self.WINDOW_WIDTH - total_width) // 2
        y = self.WINDOW_HEIGHT // 2 - Card.CARD_HEIGHT // 2

        self.center_cards_pos = []
        for i in range(n):
            x = int(start_x + i * spacing)
            self.center_cards_pos.append((x, y))
    def __init__(self, render=True):
        self.render = render
        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 720
        if self.render:
            import pygame
            pygame.init()
            # Set fullscreen mode
            display_info = pygame.display.Info()
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT = display_info.current_w, display_info.current_h
            self.screen = pygame.display.set_mode(
                (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.FULLSCREEN
            )
            pygame.display.set_caption("Tiến Lên - Card Game")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        self.center_cards = []
        self.center_cards_pos = []
        self.current_player_id = None
        self.last_played_cards = []
        self.last_played_by = None
        self.ai_agents = []
        self.move_history = []  # Thêm dòng này để khởi tạo lịch sử

        # LLM AI toggles (auto-read from environment)
        self.USE_LLM_AGENT = os.environ.get("USE_LLM_AGENT", "False").lower() in ["1", "true", "yes"]

    def get_player_at_position(self, position):
        """Return the player object matching the given position ('bottom', 'top', etc). Return None if not found."""
        for player in self.players:
            if player.position == position:
                return player
        return None

    def get_llm_model(self):
        """Ensure LLM_MODEL is always initialized, fallback to default if missing."""
        return getattr(self, "LLM_MODEL", os.environ.get("LLM_MODEL", "gemini-2.0-flash"))

    def __post_init__(self):
        """Explicit post-init to ensure LLM_MODEL set before player usage."""
        self.LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

    # re-insert correct __init__ fields after get_player_at_position
    def __init__(self, render=True):
        self.render = render
        self.WINDOW_WIDTH = 1920
        self.WINDOW_HEIGHT = 1000
        if self.render:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Tiến Lên - Card Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
            self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        else:
            self.screen = None
            self.clock = None
        self.center_cards = []
        self.center_cards_pos = []
        self.current_player_id = None
        self.last_played_cards = []
        self.last_played_by = None
        self.ai_agents = []

        # LLM AI toggles (auto-read from environment)
        self.USE_LLM_AGENT = os.environ.get("USE_LLM_AGENT", "False").lower() in ["1", "true", "yes"]
        self.LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        self.consecutive_passes = 0
        self.players = []
        self.passed_player_ids = []
        self.running = True
        self.game_count = 0
        self.player_rankings = []
        self.move_history = []

        # Ensure LLM_MODEL is always set before initialize_players
        self.initialize_players()
        for player in self.players:
            agent = getattr(self, 'ai_agents', {}).get(player.id, None)

    def get_player_by_id(self, player_id):
        """Return the player object matching player_id. Return None if not found."""
        for player in self.players:
            if player.id == player_id:
                return player
        return None


    def initialize_players(self):
        self.players = []  # Always reset player list
        positions = ['bottom', 'right', 'top', 'left']
        with open("player.json", "r") as f:
            player_list = json.load(f)
        for i, (player_info, position) in enumerate(zip(player_list, positions)):
            is_ai = self.USE_LLM_AGENT
            player = Player(
                id=player_info["id"],
                name=player_info["name"],
                position=position,
                is_ai=is_ai
            )
            self.players.append(player)

        # Save canonical seat order (list of IDs in order)
        self.seat_order = [player.id for player in self.players]

        # Assign GeminiAgent to all AI players if LLM agent is enabled; otherwise None.
        self.ai_agents = {}
        for player in self.players:
            if player.is_ai and self.LLM_MODEL == "gemini-2.0-flash":
                self.ai_agents[player.id] = GeminiAgent()
            elif player.is_ai:
                self.ai_agents[player.id] = None

        if self.game_count == 1:
            new_order = [starter]
            for i in range(1, 4):
                idx = (self.players.index(starter) + i) % 4
                new_order.append(self.players[idx])
                
            positions = ['bottom', 'right', 'top', 'left']
            for i, player in enumerate(new_order):
                player.position = positions[i]
                player.is_turn = (i == 0)
            self.players = new_order
            self.current_player_id = starter.id


    def deal_cards(self):
        for i in range(13):
            for idx, player in enumerate(self.players):
                if self.deck:
                    c = self.deck.pop()
                    player.add_card(c)

    def show_mode_menu(self):
        import pygame
        WHITE = (250, 250, 250)
        BLUE = (70, 130, 200)
        GRAY = (180, 180, 180)
        btn_w, btn_h = 350, 70
        gap = 50
        modes = [("4 Human", "4_Humans"), ("4 AI (disabled)", "4_AI")]

        buttons = []
        total_h = len(modes)*btn_h + (len(modes)-1)*gap
        y0 = (self.WINDOW_HEIGHT//2) - total_h//2 + 80

        choosing = True
        while choosing:
            self.screen.fill(WHITE)
            # draw title every frame
            try:
                title = self.title_font.render("Chọn chế độ chơi", True, BLUE)
            except Exception:
                font_fallback = pygame.font.SysFont('Arial', 36, bold=True)
                title = font_fallback.render("Chọn chế độ chơi", True, BLUE)
            self.screen.blit(title, (self.WINDOW_WIDTH//2 - title.get_width()//2, 100))

            buttons.clear()
            for idx, (label, code) in enumerate(modes):
                rect = pygame.Rect(self.WINDOW_WIDTH//2 - btn_w//2, y0 + idx*(btn_h+gap), btn_w, btn_h)
                # Disable option for "4 AI"
                disabled = (code == "4_AI")
                color = GRAY if disabled else BLUE
                pygame.draw.rect(self.screen, color, rect, border_radius=15)
                try:
                    text = self.font.render(label, True, WHITE)
                except Exception:
                    font_fallback = pygame.font.SysFont('Arial', 24)
                    text = font_fallback.render(label, True, WHITE)
                self.screen.blit(text, (rect.centerx - text.get_width()//2, rect.centery - text.get_height()//2))
                buttons.append((rect, code, disabled))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse = pygame.mouse.get_pos()
                    for rect, code, disabled in buttons:
                        if rect.collidepoint(mouse) and not disabled:
                            choosing = False
                            return code
            self.clock.tick(30)

        for player in self.players:
            if GameLogic.check_instant_win(player):
                self.end_game(player.id)
                return

    def play_cards(self):
        import traceback
        traceback.print_stack(limit=8)
        print(f"[PLAY LOG] --> Called for current_player_id={self.current_player_id}")
        current_player = self.get_player_by_id(self.current_player_id)
        selected_cards = current_player.get_selected_cards()

        # Nếu không chọn lá bài nào thì không làm gì cả, tránh clear bàn do call thừa
        if not selected_cards:
            print("[PLAY LOG] No selected_cards, returning immediately.")
            return

        # Prevent crash: log_move may not be defined
        if hasattr(self, "log_move"):
            self.log_move(current_player, selected_cards if selected_cards else None)
        
        # Validate lại trước khi thực thi
        game_first_turn = (self.game_count == 1 and not self.last_played_cards)
        if not current_player.can_play_cards(selected_cards, self.last_played_cards, game_first_turn):
            # >> Log all invalid move attempts for LLM debugging
            print("[INVALID MOVE] Detected!")
            print(f"  [INVALID MOVE] Player: {current_player.name} (AI={current_player.is_ai})")
            print(f"  [INVALID MOVE] Attempted Cards: {[str(c) for c in selected_cards]}")
            print(f"  [INVALID MOVE] Table: {[str(c) for c in self.last_played_cards]}")
            print(f"  [INVALID MOVE] Game First Turn: {game_first_turn}")
            print(f"  [INVALID MOVE] Cards on hand: {[str(c) for c in current_player.hand]}")
            # Always clear selection after an invalid move, both for AI and human
            current_player.clear_selection()
            # Nếu không hợp lệ nhưng là lượt đầu, reset lại hiển thị và selection
            if game_first_turn:
                self.center_cards = []
                self.center_cards_pos = []
                self.arrange_center_cards()
                return
            # Chỉ clear center_cards (hiển thị bài sai) nếu đã từng có bài trên bàn/lượt chặn,
            # Còn lượt đầu, không clear tức là giữ nguyên trạng thái bàn
            if self.last_played_cards:
                self.center_cards = []
                self.center_cards_pos = []
                self.arrange_center_cards()
                self.last_played_cards = []
            return

        # Số lượng bài hợp lệ tối đa allowed
        if len(selected_cards) not in [1, 2, 3, 4] and not self.is_valid_straight(selected_cards):
            # Không cho phép đánh combo lạ, bỏ qua thao tác
            return

        # Ghi lại lịch sử
        move_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in selected_cards])
        self.move_history.append({
            'player': current_player.name,
            'move': [f"{c.rank}{c.suit[0].upper()}" for c in selected_cards]
        })
        # Log player hand before playing
        hand_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in current_player.hand])
        history_log(f"{current_player.name} hand: {hand_str}")
        current_player.remove_cards(selected_cards)
        self.last_played_cards = selected_cards
        self.last_played_by = current_player.id
        move_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in selected_cards])
        history_log(f"{current_player.name} played: {move_str}")
        print(f"[PLAY LOG] Setting last_played_by={self.last_played_by} after play by {current_player.name} (id={current_player.id})")
        self.center_cards = selected_cards
        self.arrange_center_cards()  # luôn cập nhật vị trí bài

        current_player.passed = False
        self.consecutive_passes = 0

        if not current_player.hand:
            self.reset_table()
            self.end_game(current_player.id)
            return  # QUAN TRỌNG: dừng, không xoay lượt

        # Sau một lần đánh, mọi người có thể chơi tiếp -> reset hết lượt pass!
        self.passed_player_ids = []
        for p in self.players:
            p.passed = False

        # Xoay lượt: người tiếp theo liền kề theo chiều kim đồng hồ, update current_player_id
        current_player.is_turn = False
        curr_idx = self.players.index(current_player)
        next_idx = (curr_idx + 1) % len(self.players)
        self.current_player_id = self.players[next_idx].id
        self.players[next_idx].is_turn = True

        self.rotate_player_positions()

    def is_valid_straight(self, cards):
        """Kiểm tra xem cards có phải là sảnh hợp lệ (hoặc đôi thông) đúng luật."""
        from collections import Counter
        if not cards or len(cards) < 3:
            return False
        # Sảnh lẻ
        values = [Card.RANKS.index(c.rank) for c in cards]
        if len(set(values)) == len(values) and all(values[i]+1 == values[i+1] for i in range(len(values)-1)):
            return True
        # Đôi thông
        if len(cards) >= 6 and len(cards) % 2 == 0:
            rank_groups = {}
            for card in cards:
                rank_groups.setdefault(card.rank, []).append(card)
            if not all(len(group) == 2 for group in rank_groups.values()):
                return False
            sorted_ranks = sorted(rank_groups.keys(), key=lambda r: Card.RANKS.index(r))
            rank_indices = [Card.RANKS.index(r) for r in sorted_ranks]
            if all(rank_indices[i]+1==rank_indices[i+1] for i in range(len(rank_indices)-1)):
                return True
        return False

    def pass_turn(self):
        import traceback
        print("[PASS LOG] ------- pass_turn CALLED -------")
        traceback.print_stack(limit=8)
        print(f"[PASS LOG] --> Called with current_player_id={self.current_player_id}")
        current_player = self.get_player_by_id(self.current_player_id)
        if current_player is None:
            print("[PASS LOG] current_player is None, cannot pass turn.")
            return

        # Không cho PASS nếu bàn đang trống (không có bài trên bàn)!
        if not self.last_played_cards:
            print("[PASS LOG] Forbidden: Attempted to PASS when no cards on table")
            # Force player to play a card (fallback)
            if current_player and current_player.hand:
                current_player.sort_hand()
                card = current_player.hand[0]
                card.selected = True
                print(f"[PASS LOG] Fallback: Auto-playing smallest card {card.rank}{card.suit[0].upper()} for {current_player.name}")
                self.play_cards()
            return

        print(f"[PASS LOG] Player {current_player.name} (id={current_player.id}) initiates PASS.")
        print(f"  [PASS LOG] Before pass: passed_player_ids={self.passed_player_ids}, current_player.passed={current_player.passed}")

        # Đánh dấu player này đã pass nếu chưa có trong danh sách
        if current_player.id not in self.passed_player_ids:
            self.passed_player_ids.append(current_player.id)
            print(f"  [PASS LOG] -> Added to passed_player_ids: {self.passed_player_ids}")
        current_player.passed = True
        current_player.clear_selection()
        current_player.update_after_pass()
        # Log PASS action in history log
        hand_str = ", ".join([f"{c.rank}{c.suit[0].upper()}" for c in current_player.hand])
        history_log(f"{current_player.name} hand: {hand_str}")
        history_log(f"{current_player.name} played: PASS")

        print(f"  [PASS LOG] After pass flag: passed_player_ids={self.passed_player_ids}")
        print(f"  [PASS LOG] Players state:")
        for idx, player in enumerate(self.players):
            print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn} | passed={player.passed} | in_passed_ids={player.id in self.passed_player_ids}")

        # Nếu đủ n-1 người pass, reset bàn: người cuối cùng đánh gần nhất sẽ giữ lượt
        if len(self.passed_player_ids) >= len(self.players) - 1:
            print(f"[PASS LOG] Enough ({len(self.passed_player_ids)}) players have passed, resetting table.")
            # Extra logging before reset
            print("[PASS LOG] --- BEFORE reset_table() ---")
            for idx, player in enumerate(self.players):
                print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn}")
            # --- FIX: Save last_played_by before table reset ---
            last_valid_player = self.last_played_by
            self.reset_table()
            # Log after reset_table
            print("[PASS LOG] --- AFTER reset_table() ---")
            for idx, player in enumerate(self.players):
                print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn}")
            print(f"[PASS LOG] last_valid_player (saved last_played_by)={last_valid_player} current_player.id={current_player.id}")
            if not last_valid_player:
                print("[ERROR] last_played_by is None during mass pass table reset! Assigning next in seating order to prevent crash.")
                # Safety fallback: assign first player as current if last_played_by is None
                if self.players:
                    self.current_player_id = self.players[0].id
                    print(f"[PASS LOG] Fallback: set current_player_id to first player: {self.current_player_id}")
                else:
                    self.current_player_id = None
            else:
                self.current_player_id = last_valid_player
                print(f"[PASS LOG] Now setting current_player_id={self.current_player_id} after table reset.")
            # FIX: Do NOT clear last_played_by here!
            self.passed_player_ids = []
            for p in self.players:
                p.passed = False
            # Cập nhật trạng thái turn cho người chơi mới
            print("[PASS LOG] is_turn assignment pass:")
            for idx, player in enumerate(self.players):
                print(f" before assign: [{idx}] {player.name} (id={player.id}) | is_turn={player.is_turn}")
            for player in self.players:
                player.is_turn = (player.id == self.current_player_id)
            for idx, player in enumerate(self.players):
                print(f" after assign: [{idx}] {player.name} (id={player.id}) | is_turn={player.is_turn}")
            print("[PASS LOG] After all passed: now rotating table to current player at bottom.")
            self.rotate_player_positions()
            # Log after table reset and rotate
            print("[PASS LOG] --- AFTER ROTATE ---")
            for idx, player in enumerate(self.players):
                print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn}")
            print("[PASS LOG] Table reset: Player order after reset:")
            for idx, player in enumerate(self.players):
                print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn}")
            return

        # Tìm người chơi tiếp theo CHƯA PASS
        current_idx = self.players.index(current_player)
        next_player = None

        # Duyệt tất cả người chơi theo chiều kim đồng hồ
        print("[PASS LOG] Finding next player who has NOT passed.")
        for i in range(1, len(self.players) + 1):
            next_idx = (current_idx + i) % len(self.players)
            candidate = self.players[next_idx]
            print(f"    [PASS LOG] Checking candidate {candidate.name} (id={candidate.id}) passed={candidate.passed}")
            # Bỏ qua người đã pass
            if candidate.id in self.passed_player_ids:
                continue
            next_player = candidate
            print(f"    [PASS LOG] -> Next player chosen: {candidate.name} (id={candidate.id}) at idx={next_idx}")
            break

        # Nếu tìm thấy người chơi tiếp theo
        if next_player:
            current_player.is_turn = False
            next_player.is_turn = True
            self.current_player_id = next_player.id
            print(f"[PASS LOG] Before rotate: players order/positions:")
            for idx, player in enumerate(self.players):
                print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn}")
            self.rotate_player_positions()
            print(f"[PASS LOG] After rotate: players order/positions:")
            for idx, player in enumerate(self.players):
                print(f"    [{idx}] {player.name} (id={player.id}) | pos={player.position} | is_turn={player.is_turn}")

    def end_game(self, winner_id):
        winner = self.get_player_by_id(winner_id)
        remaining = [p for p in self.players if p.id != winner_id]
        sorted_other = sorted(remaining, key=lambda p: len(p.hand))
        
        self.player_rankings = [winner_id] + [p.id for p in sorted_other]
        
        rewards = [200, 50, -20, -60]
        for i, player in enumerate(self.players):
            rank = self.player_rankings.index(player.id)
            player.add_reward(rewards[rank], f"Xếp hạng {rank+1}")
            multiplier = 1.5 if rank == 0 else 0.7 if rank == 3 else 1.0
            player.score *= multiplier

        for i, player_id in enumerate(self.player_rankings):
            player = self.get_player_by_id(player_id)
        
        self.prepare_new_game()


    def handle_events(self):
        """
        Handle all events (click/hover) for Pygame;
        - enable hover effect for bottom player's cards.
        - enable card select (click)
        """
        import pygame
        # Find the human bottom player whose turn it is
        bottom_player = None
        for player in self.players:
            if player.position == "bottom" and player.is_turn and not player.passed and not getattr(player, "is_ai", False):
                bottom_player = player
                break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEMOTION and bottom_player is not None:
                bottom_player.update_hover(event.pos)
            # THÊM xử lý click card
            if event.type == pygame.MOUSEBUTTONDOWN and bottom_player is not None:
                mouse_pos = pygame.mouse.get_pos()
                # Lấy trạng thái bàn
                last_played_cards = self.last_played_cards
                game_first_turn = (self.game_count == 1 and not self.last_played_cards)
                clicked = bottom_player.handle_click(mouse_pos, last_played_cards=last_played_cards, game_first_turn=game_first_turn)

        if self.clock:
            self.clock.tick(30)


    def setup_ai_players(self):
        """
        Initialize ai_agents dict for all AI players.
        Fills with None as placeholder (to be replaced by actual agents in RL mode).
        """
        # Tạo self.ai_agents sau khi đã gán is_ai cho tất cả player
        self.ai_agents = {}
        for player in self.players:
            if player.is_ai and getattr(self, "USE_LLM_AGENT", False) and getattr(self, "LLM_MODEL", "") == "gemini-2.0-flash":
                self.ai_agents[player.id] = GeminiAgent()
            elif player.is_ai:
                self.ai_agents[player.id] = None
    def draw(self):
        """Draw the full game state (cards, play area, UI buttons)"""
        import pygame
        from .ui import constants as UI

        if not self.render or not self.screen:
            return

        # Background
        self.screen.fill((246, 246, 240))  # Cream, matches card design

        # Draw cards currently on the table (no center table box anymore)
        for card, pos in zip(self.center_cards, self.center_cards_pos):
            card_surface = card.get_face_up_surface()
            self.screen.blit(card_surface, pos)

        # Draw all player hands
        for player in self.players:
            player.draw_hand(self.screen)

        # --- Draw player info card at bottom left ---
        bottom_player = self.get_player_at_position('bottom')
        if bottom_player:
            info_w, info_h = 195, 66
            info_x, info_y = 20, self.WINDOW_HEIGHT - info_h - 20
            radius = 8
            info_bg = (255, 255, 255)
            info_border = (82, 170, 165)
            text_color = (20, 22, 22)
            pygame.draw.rect(self.screen, info_bg, (info_x, info_y, info_w, info_h), border_radius=radius)
            pygame.draw.rect(self.screen, info_border, (info_x, info_y, info_w, info_h), 2, border_radius=radius)
            font = pygame.font.SysFont('Arial', 20, bold=True)
            subfont = pygame.font.SysFont('Arial', 14)
            name_text = font.render(bottom_player.name, True, text_color)
            id_text = subfont.render("id: " + str(bottom_player.id), True, (110, 120, 125))
            self.screen.blit(name_text, (info_x + 16, info_y + 13))
            self.screen.blit(id_text, (info_x + 16, info_y + 36))

        # Debounce state for mouse buttons (to avoid frame-repeat clicks)
        if not hasattr(self, "_prev_mouse_down"):
            self._prev_mouse_down = False

        if hasattr(self, 'play_button') and hasattr(self, 'pass_button'):
            current_player = self.get_player_by_id(self.current_player_id) if hasattr(self, "current_player_id") else None
            can_play = False
            if current_player:
                selected_cards = current_player.get_selected_cards()
                # SỬA: luôn kiểm tra can_play_cards (bất kể có bài trên bàn hay không)
                if hasattr(current_player, "can_play_cards"):
                    can_play = current_player.can_play_cards(selected_cards, self.last_played_cards)
            # PLAY button
            play_color = (100, 200, 100) if can_play else (150, 150, 150)
            play_border = (50, 150, 50) if can_play else (110, 110, 110)
            pygame.draw.rect(self.screen, play_color, self.play_button, border_radius=8)
            pygame.draw.rect(self.screen, play_border, self.play_button, 2, border_radius=8)
            play_text = self.font.render("PLAY", True, (255, 255, 255))
            play_rect = play_text.get_rect(center=self.play_button.center)
            self.screen.blit(play_text, play_rect)
            # PASS button
            pygame.draw.rect(self.screen, (240, 100, 100), self.pass_button, border_radius=8)
            pygame.draw.rect(self.screen, (180, 50, 50), self.pass_button, 2, border_radius=8)
            pass_text = self.font.render("PASS", True, (255, 255, 255))
            pass_rect = pass_text.get_rect(center=self.pass_button.center)
            self.screen.blit(pass_text, pass_rect)
            # Button click actions: DE-BOUNCED so triggers only on mouse DOWN edge
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            mouse_down = mouse_buttons[0]
            # Only trigger event if not held last frame
            if mouse_down and not self._prev_mouse_down:
                if self.play_button.collidepoint(mouse_pos):
                    self.play_cards()
                elif self.pass_button.collidepoint(mouse_pos):
                    self.pass_turn()
            self._prev_mouse_down = mouse_down

        pygame.display.flip()

    def run(self):
        try:
            # Always use 4_Humans mode, do not display mode menu
            # Mode selection removed: always use human/LLM fallback as set above

            self.initialize_players()
            print("[DEBUG] USE_LLM_AGENT =", self.USE_LLM_AGENT, "| LLM_MODEL =", self.LLM_MODEL)
            self.setup_ai_players()
            print("[DEBUG] ai_agents =", self.ai_agents)
            self.prepare_new_game()

            while self.running:
                step_start = time.time()
                current_player = self.get_player_by_id(self.current_player_id)

                if current_player.is_ai:
                    try:
                        agent = self.ai_agents[current_player.id]
                        # Determine if this is the first turn for move validation
                        game_first_turn = (self.game_count == 1 and not self.last_played_cards)
                        if isinstance(agent, GeminiAgent):
                            retry_count = 0
                            max_retries = 2
                            while retry_count < max_retries:
                                llm_action = agent.select_action(self, current_player)
                                if llm_action == "PASS":
                                    self.pass_turn()
                                    break
                                elif current_player.can_play_cards(llm_action, self.last_played_cards, game_first_turn):
                                    for card in current_player.hand:
                                        card.selected = (card in llm_action)
                                    self.play_cards()
                                    break
                                else:
                                    print(f"[MAIN] Invalid move by {current_player.name}, retrying... ({retry_count+1} of {max_retries})")
                                    retry_count += 1
                            if retry_count >= max_retries:
                                print(f"[MAIN] Maximum retries exceeded for {current_player.name}, forcing PASS")
                                self.pass_turn()
                        else:
                            # RL or other agent logic
                            action_id = agent.select_action(self)
                            action = agent.interpret_action(action_id, self)
                            if action == "PASS":
                                self.pass_turn()
                            else:
                                # Validate move before playing
                                if current_player.can_play_cards(action, self.last_played_cards, game_first_turn):
                                    for card in current_player.hand:
                                        card.selected = (card in action)
                                    self.play_cards()
                                else:
                                    print(f"[MAIN] Invalid move by {current_player.name} (RL/Other), forcing PASS")
                                    self.pass_turn()
                    except Exception as e:
                        print(f"Error in AI move: {str(e)}")
                        self.pass_turn()
                else:
                    self.handle_events()

                if self.render:
                    import pygame
                    self.draw()
                    if self.clock:
                        self.clock.tick(60)

        except Exception as e:
            import traceback
            traceback.print_exc()

    def rotate_player_positions(self):
        # Safeguard against missing players
        if not hasattr(self, 'players') or not self.players:
            print("[ERROR] Cannot rotate positions: players not initialized")
            return

        # Always rotate so that current player is first
        positions = ['bottom', 'right', 'top', 'left']
        num_players = len(self.players)
        idx = next((i for i, p in enumerate(self.players) if p.id == self.current_player_id), 0)
        self.players = self.players[idx:] + self.players[:idx]
        # Assign new positions (circular)
        for i, player in enumerate(self.players):
            if i < len(positions):
                player.position = positions[i]
        # Log player positions after rotation
        print("[LOG] Vị trí người chơi sau khi xoay:")
        for p in self.players:
            print(f"    {p.name} (id={p.id}): {p.position} | is_turn={p.is_turn}")

    def reset_table(self):
        print("[RESET LOG] ------- reset_table CALLED -------")
        print(f"[RESET LOG] Table state before reset: passed_player_ids={self.passed_player_ids}")
        # CHỈ reset state, không thay đổi thứ tự người chơi
        self.last_played_cards = []
        # KHÔNG reset last_played_by
        self.center_cards = []
        self.arrange_center_cards()
        self.consecutive_passes = 0
        self.passed_player_ids = []
        for player in self.players:
            player.passed = False
        print(f"[RESET LOG] Table state after reset: passed_player_ids={self.passed_player_ids}")

    def prepare_new_game(self):
        """Prepare a new game: deal cards, reset state, find starter."""
        # Clear history log on new game
        with open("tienlen_history.log", "w") as f:
            pass
        self.game_count += 1
        self.reset_table()
        self.deck = GameLogic.create_deck()
        self.deal_cards()
        starter = GameLogic.find_starting_player(self.players)
        # Set the starting player as current
        self.current_player_id = starter.id
        # Reset turn flags: only starter has turn
        for player in self.players:
            player.is_turn = (player.id == starter.id)
            player.reset_state()
        # Reset game state
        self.consecutive_passes = 0
        self.last_played_by = None
        self.passed_player_ids = []
        self.center_cards = []
        self.last_played_cards = []
        # Arrange player positions so that starter is at bottom
        self.rotate_player_positions()
        # Place Play and Pass buttons in the bottom-right corner
        if self.render and self.screen:
            import pygame
            btn_w, btn_h = 150, 55
            gap = 20
            right_margin = 30
            bottom_margin = 30
            self.play_button = pygame.Rect(
                self.WINDOW_WIDTH - btn_w - right_margin,
                self.WINDOW_HEIGHT - 2 * btn_h - gap - bottom_margin,
                btn_w,
                btn_h
            )
            self.pass_button = pygame.Rect(
                self.WINDOW_WIDTH - btn_w - right_margin,
                self.WINDOW_HEIGHT - btn_h - bottom_margin,
                btn_w,
                btn_h
            )

if __name__ == '__main__':
    game = TienLenGame()
    game.run()
    try:
        import pygame
        pygame.quit()
    except ImportError:
        pass
    sys.exit()