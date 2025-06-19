import sys
from .card import Card
from .player import Player
from .game_logic import GameLogic
from .ui import constants as UI
# import src.ai.agent import AIAgent # Moved below to avoid circular import for unit tests
import os

class TienLenGame:
    def __init__(self, render=True):
        self.render = render
        self.center_cards = []
        self.center_cards_pos = []
        self.current_player_index = 0
        self.last_played_cards = []
        self.last_played_by = None
        self.game_mode = None   # "4_Humans", "1_AI_3_Humans", "4_AI"
        self.ai_players = []    # List index các player là AI
        self.ai_agents = []     # List AIAgent tương ứng index (dưới)

        if self.render:
            import pygame
            pygame.init()
            pygame.font.init()
            screen_info = pygame.display.Info()
            self.WINDOW_WIDTH = screen_info.current_w
            self.WINDOW_HEIGHT = screen_info.current_h
            self.screen = pygame.display.set_mode((screen_info.current_w, screen_info.current_h), pygame.FULLSCREEN)
            pygame.display.set_caption("Tiến Lên - Modern Edition")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
            self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
            # UI buttons use latest window size
            button_width, button_height = 120, 45
            self.play_button = pygame.Rect(self.WINDOW_WIDTH - button_width - 20,
                                      self.WINDOW_HEIGHT - button_height - 20,
                                      button_width, button_height)
            self.pass_button = pygame.Rect(self.WINDOW_WIDTH - button_width*2 - 30,
                                      self.WINDOW_HEIGHT - button_height - 20,
                                      button_width, button_height)
        else:
            self.WINDOW_WIDTH = 1920  # Default size khi không render
            self.WINDOW_HEIGHT = 1080
            self.screen = None
            self.clock = None
            self.font = None
            self.title_font = None
            self.play_button = None
            self.pass_button = None

        self.running = True
        self.game_count = 0
        self.player_rankings = []
        self.consecutive_passes = 0  # Pass combo reward

        # Initialize players
        self.players = []
        positions = ['bottom', 'right', 'top', 'left']
        for i, position in enumerate(positions):
            if self.game_mode == "1_AI_3_Humans":
                is_ai = (i == 0)
            elif self.game_mode == "4_AI":
                is_ai = True
            else:
                is_ai = False
            player = Player(f"Player {i+1}", position, is_ai=is_ai)
            player.player_index = i
            self.players.append(player)

    def get_player_at_position(self, position: str):
        for player in self.players:
            if player.position == position:
                return player
        return None

    def prepare_new_game(self):
        import os, datetime
        self.game_count += 1
        self.last_played_cards = []
        self.last_played_by = None
        self.center_cards = []
        self.center_cards_pos = []
        # Log file setup
        os.makedirs("logs", exist_ok=True)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = f"logs/game_{now}.log"

        # Reset player states
        for player in self.players:
            player.hand = []
            player.is_turn = False
            player.passed = False

        # Shuffle and deal new cards
        self.deck = GameLogic.create_deck()
        self.deal_cards()

        # THÊM: In bài người chơi để debug
        print(f"\n=== GAME {self.game_count} STARTING ===")
        for i, player in enumerate(self.players):
            cards = ", ".join([f"{card.rank}{card.suit[0]}" for card in player.hand])
            print(f"Player {i} cards: {cards}")

        # SỬA: xác định starter đúng, dùng đúng index cho ván đầu!
        self.current_player_index = GameLogic.find_starting_player(self.players)
        starter = self.players[self.current_player_index]
        has_three = any(card.rank == '3' and card.suit == 'spades' for card in starter.hand)
        print(f"[GAME] Starter is player {self.current_player_index}, has 3♠: {has_three}")

        if self.game_count == 1:
            # ĐẢM BẢO index đúng
            self.arrange_positions_by_starting_player(self.current_player_index)
            self.current_player_index = 0
        elif self.game_count > 1:
            self.rearrange_players()

        # Cập nhật ai đi đầu
        self.current_player_index = GameLogic.find_starting_player(self.players)
        self.players[self.current_player_index].is_turn = True
        self.round_starter = self.current_player_index

    def arrange_positions_by_starting_player(self, starting_index_old):
        positions = ['bottom', 'right', 'top', 'left']
        new_players = []
        for i in range(4):
            idx = (starting_index_old + i) % 4
            new_players.append(self.players[idx])
        for i, player in enumerate(new_players):
            player.position = positions[i]
            player.player_index = i
        self.players = new_players

    def rearrange_players(self):
        ranked_players = [self.players[i] for i in self.player_rankings]
        positions = ['bottom', 'right', 'top', 'left']
        for i, player in enumerate(ranked_players):
            player.position = positions[i]
            player.player_index = i
        self.players = ranked_players

    def end_game(self, winner_index):
        self.determine_rankings(winner_index)
        # Áp dụng phần thưởng kết thúc theo xếp hạng
        rewards = [200, 50, -20, -60]  # [Top 1, Top 2, Top 3, Top 4]
        for i, player in enumerate(self.players):
            rank = self.player_rankings.index(i)
            player.add_reward(rewards[rank], f"Xếp hạng {rank+1}")
            multiplier = 1.5 if rank == 0 else 0.7 if rank == 3 else 1.0
            player.score *= multiplier
        # +2.0 cho phục hồi thế cờ
        for player in self.players:
            if player.cards_lost >= 5 and len(player.hand) <= 5:
                player.add_reward(2.0, "Phục hồi thế cờ")
        # Hiển thị bảng điểm
        print("\n=== KẾT QUẢ VÁN ĐẤU ===")
        for i, player in enumerate(sorted(self.players, key=lambda p: self.player_rankings.index(self.players.index(p)))):
            rank = i + 1
            print(f"Top {rank}: {player.name} - Điểm: {player.score}")
        self.prepare_new_game()
    
    def rearrange_players(self):
        print("[PLAYERS] Current order:")
        for i, p in enumerate(self.players):
            print(f"  {i}: {p.name} ({p.position})")
        # original code
        ranked_players = [self.players[i] for i in self.player_rankings]
        positions = ['bottom', 'right', 'top', 'left']
        for i, player in enumerate(ranked_players):
            player.position = positions[i]
        self.players = ranked_players

    def determine_rankings(self, winner_index):
        winner = self.players[winner_index]
        remaining = [(i, p) for i, p in enumerate(self.players) if i != winner_index]
        start_index = (winner_index + 1) % 4
        for offset in range(3):
            idx = (start_index + offset) % 4
            if idx != winner_index:
                pass
        sorted_other = sorted(remaining, key=lambda t: (len(t[1].hand), 0))
        self.player_rankings = [winner_index] + [idx for (idx, _) in sorted_other]

    def create_gradient_background(self):
        import pygame
        background = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        background.fill(UI.BACKGROUND_COLOR)
        for x in range(0, self.WINDOW_WIDTH, 40):
            for y in range(0, self.WINDOW_HEIGHT, 40):
                if (x//40 + y//40) % 2 == 0:
                    pygame.draw.circle(background, (235, 235, 235), (x, y), 2)
        return background

    def deal_cards(self):
        for i in range(13):
            for player in self.players:
                if self.deck:
                    player.add_card(self.deck.pop())

        # Kiểm tra thắng trắng (instant win)
        for player in self.players:
            if GameLogic.check_instant_win(player):
                print(f"{player.name} wins with instant win condition!")
                winner_index = self.players.index(player)
                self.end_game(winner_index)
                return

    def handle_events(self):
        import pygame
        mouse_pos = pygame.mouse.get_pos()
        current_player = self.players[0]
        current_player.update_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(f"[DEBUG] Mouse button down at {event.pos} button={event.button}")
                if event.button == 1:  # Left click
                    if current_player.is_turn:
                        print(f"[DEBUG] Handling card click for player at pos={event.pos}")
                        current_player.handle_click(event.pos)
                        if self.play_button and self.play_button.collidepoint(event.pos):
                            print("[DEBUG] Play button clicked")
                            # Chỉ cho phép Play khi hợp lệ
                            selected_cards = current_player.get_selected_cards()
                            if current_player.validate_selection(selected_cards):
                                self.play_cards()
                        elif self.pass_button and self.pass_button.collidepoint(event.pos):
                            print("[DEBUG] Pass button clicked")
                            self.pass_turn()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if current_player.is_turn:
                        selected_cards = current_player.get_selected_cards()
                        if current_player.validate_selection(selected_cards):
                            self.play_cards()
                elif event.key == pygame.K_p:
                    if current_player.is_turn:
                        self.pass_turn()

    def play_cards(self):
        current_player = self.players[0]
        selected_cards = current_player.get_selected_cards()

        # --- LOG lượt chơi ---
        self.log_move(current_player, selected_cards if selected_cards else None)

        if current_player.can_play_cards(selected_cards, self.last_played_cards):
            current_player.remove_cards(selected_cards)
            self.last_played_cards = selected_cards
            self.last_played_by = self.current_player_index
            self.center_cards = selected_cards
            self.arrange_center_cards()

            if not current_player.hand:
                print(f"{current_player.name} wins!")
                self.end_game(self.current_player_index)
                return

            # Sau mỗi lượt đánh thành công
            if self.game_mode == "4_AI":
                self.rotate_players()
                self.current_player_index = 0
                self.players[self.current_player_index].is_turn = True
            else:
                # Advance to next player in sequence (modulo 4), set their is_turn
                self.players[self.current_player_index].is_turn = False
                self.current_player_index = (self.current_player_index + 1) % 4
                self.players[self.current_player_index].is_turn = True
                self.arrange_players_for_ui(self.current_player_index)
        else:
            current_player.clear_selection()

    def log_move(self, player, cards, is_pass=False):
        """Ghi log 1 lượt chơi ra file log"""
        try:
            with open(self.log_file_path, "a") as f:
                info = "[AI]" if getattr(player, "is_ai", False) else "[Human]"
                if is_pass:
                    log = f"{player.name} {info} - PASS\n"
                else:
                    if cards:
                        cards_str = ", ".join([f"{card.rank}{card.suit[0].upper() if hasattr(card.suit,'upper') else card.suit}" for card in cards])
                        log = f"{player.name} {info} - {cards_str}\n"
                    else:
                        log = f"{player.name} {info} - (No Card)\n"
                f.write(log)
        except Exception as e:
            print(f"[LOG ERROR] {e}")

    def rotate_players(self):
        """Đưa người được quyền chơi tiếp vào vị trí bottom và cập nhật lại view UI"""
        self.arrange_players_for_ui(self.current_player_index)

    def arrange_players_for_ui(self, focused_index):
        """Chuyển self.players sao cho focused_index lên đầu, gán lại vị trí UI cho đúng, bottom luôn là người đến lượt"""
        import src.game.ui.constants as UI
        n = len(self.players)
        old_players = self.players
        new_players = [self.players[(focused_index + i) % n] for i in range(n)]
        for i, player in enumerate(new_players):
            player.position = UI.PLAYER_POSITIONS[i]
            player.is_turn = (i == 0)
            player.player_index = i
        self.players = new_players
        self.current_player_index = 0  # always: bottom là người đến lượt (player vừa chuyển tới lượt)

        # LOG xoay vị trí
        print("[ARRANGE_FOR_UI] Player order after rotation:")
        for i, player in enumerate(self.players):
            print(f"  {i}: {player.name} at {player.position}")

        # SỬA: cập nhật lại agent mapping chính xác
        if isinstance(self.ai_agents, dict) and self.ai_agents:
            new_agents = {}
            for i, player in enumerate(new_players):
                original_idx = old_players.index(player)
                if original_idx in self.ai_agents:
                    agent = self.ai_agents[original_idx]
                    agent.player_index = i  # update index for new arrangement
                    new_agents[i] = agent
            self.ai_agents = new_agents
            self.ai_players = sorted(self.ai_agents.keys())
        elif isinstance(self.ai_agents, list):
            for agent in self.ai_agents:
                agent.player_index = (agent.player_index - focused_index) % 4
            self.ai_players = [(i-focused_index) % 4 for i in self.ai_players]


    def pass_turn(self):
        current_player = self.players[0]

        # Rule: In the first round of the first game, the player holding 3♠ MUST play, cannot pass
        if self.game_count == 1 and not self.last_played_cards:
            has_three_spades = any(card.rank == '3' and card.suit == 'spades' for card in current_player.hand)
            if has_three_spades:
                return
        # Rule: From round 2 onwards, top 1 from previous game (now at bottom, player index 0) must play first, can't pass at first turn
        if self.game_count > 1 and not self.last_played_cards and self.current_player_index == 0:
            return

        current_player.passed = True
        current_player.clear_selection()
        current_player.update_after_pass()
        self.consecutive_passes += 1

        # --- Sửa deadlock ALL PASS: nếu tất cả đều passed thì reset center & bàn ---
        if all(p.passed for p in self.players):
            print("[GAME] All players passed! Resetting center and pass state.")
            for player in self.players:
                player.passed = False
            self.last_played_cards = []
            self.center_cards = []
            self.consecutive_passes = 0
            # Đúng luật: người cuối cùng đánh được đi tiếp, fallback sang next nếu chưa có
            if self.last_played_by is not None:
                self.current_player_index = self.last_played_by
            else:
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
            for i, player in enumerate(self.players):
                player.is_turn = (i == 0)
            self.arrange_players_for_ui(self.current_player_index)
            return

        # Chỉ xoay người chơi nếu chưa trigger deadlock
        self.rotate_players()
        self.current_player_index = 0
        for i, player in enumerate(self.players):
            player.is_turn = (i == 0)

    def next_turn(self):
        self.players[self.current_player_index].is_turn = False
        self.current_player_index = (self.current_player_index + 1) % 4
        self.players[self.current_player_index].is_turn = True
        self.arrange_players_for_ui(self.current_player_index)

    def arrange_center_cards(self):
        if not self.center_cards:
            return
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2
        card_spacing = UI.CARD_WIDTH // 2
        total_width = (len(self.center_cards) - 1) * card_spacing + UI.CARD_WIDTH
        start_x = center_x - total_width // 2
        self.center_cards_pos = []
        for i in range(len(self.center_cards)):
            x = start_x + i * card_spacing
            self.center_cards_pos.append((x, center_y))

    def draw(self):
        import pygame
        if not self.render:
            return
        self.screen.fill(UI.BACKGROUND_COLOR)
        # Center card area
        center_rect = pygame.Rect(self.WINDOW_WIDTH//2 - 200, self.WINDOW_HEIGHT//2 - 100, 400, 200)
        pygame.draw.rect(self.screen, UI.CARD_AREA_COLOR, center_rect, border_radius=15)
        pygame.draw.rect(self.screen, (200, 200, 200), center_rect, 2, border_radius=15)
        # Center cards
        for card, pos in zip(self.center_cards, self.center_cards_pos):
            card.face_up = True
            card.draw(self.screen, pos)
        # Draw player hands
        for player in self.players:
            player.draw_hand(self.screen)
        self.draw_ui()
        pygame.display.flip()

    def draw_ui(self):
        import pygame
        # UI tối giản, không hiển thị title, current turn, last played, player name các góc nữa
        if self.players[self.current_player_index].is_turn:
            current_player = self.players[self.current_player_index]
            selected_cards = current_player.get_selected_cards()
            # Cập nhật trạng thái bàn cho update_invalid_states/can_play_cards
            current_player._last_played_cards = self.last_played_cards if self.last_played_cards else None
            can_play = current_player.can_play_cards(selected_cards, self.last_played_cards)
            # Gray out button if move isn't valid
            if can_play:
                play_color = (100, 200, 100)
                play_border = (50, 150, 50)
            else:
                play_color = (150, 150, 150)
                play_border = (110, 110, 110)
            pygame.draw.rect(self.screen, play_color, self.play_button, border_radius=8)
            pygame.draw.rect(self.screen, play_border, self.play_button, 2, border_radius=8)
            play_text = self.font.render("PLAY", True, (255, 255, 255))
            play_rect = play_text.get_rect(center=self.play_button.center)
            self.screen.blit(play_text, play_rect)
            pygame.draw.rect(self.screen, (240, 100, 100), self.pass_button, border_radius=8)
            pygame.draw.rect(self.screen, (180, 50, 50), self.pass_button, 2, border_radius=8)
            pass_text = self.font.render("PASS", True, (255, 255, 255))
            pass_rect = pass_text.get_rect(center=self.pass_button.center)
            self.screen.blit(pass_text, pass_rect)

    def show_mode_menu(self):
        """Hiển thị menu chọn chế độ chơi, return mode string"""
        pygame = __import__("pygame")
        WHITE = (250, 250, 250)
        BLUE = (70, 130, 200)
        GRAY = (180, 180, 180)
        btn_w, btn_h = 350, 70
        gap = 50

        modes = [("4 Humans", "4_Humans"),
                 ("1 AI 3 Human", "1_AI_3_Humans"),
                 ("4 AI", "4_AI"),
                 ("4 AI Auto Train (500 games)", "4_AI_AUTO_TRAIN")]
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
            for i, (label, code) in enumerate(modes):
                rect = pygame.Rect(self.WINDOW_WIDTH//2 - btn_w//2, y0 + i*(btn_h+gap), btn_w, btn_h)
                pygame.draw.rect(self.screen, BLUE if i==1 else GRAY, rect, border_radius=15)
                try:
                    text = self.font.render(label, True, WHITE)
                except Exception:
                    font_fallback = pygame.font.SysFont('Arial', 24)
                    text = font_fallback.render(label, True, WHITE)
                self.screen.blit(text, (rect.centerx - text.get_width()//2, rect.centery - text.get_height()//2))
                buttons.append((rect, code))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse = pygame.mouse.get_pos()
                    for rect, code in buttons:
                        if rect.collidepoint(mouse):
                            print(f"[DEBUG] Chọn mode: {code}")
                            choosing = False
                            return code
            self.clock.tick(30)

    def setup_ai_players(self):
        """Đồng bộ ai_players và ai_agents dựa trên self.players"""
        self.ai_players = []
        self.ai_agents = {}
        # Import AIAgent here only when actually running game, to avoid circular import for standalone agent test
        from src.ai.agent import AIAgent
        for idx, player in enumerate(self.players):
            if getattr(player, "is_ai", False):
                self.ai_players.append(idx)
                self.ai_agents[idx] = AIAgent(idx, model_path=None)

    def run(self):
        # Optional: check memory before starting in non-render mode
        if not self.render:
            try:
                import psutil
                mem = psutil.virtual_memory()
                if mem.available < 1024 * 1024 * 500:  # 500MB
                    print("Not enough memory! Exiting...")
                    return
            except ImportError:
                print("psutil not available, skipping memory check.")
        print("[DEBUG] Starting game with mode:", self.game_mode)
        try:
            # Thêm chế độ 4_AI_AUTO_TRAIN
            if self.render:
                self.game_mode = self.show_mode_menu()
            else:
                self.game_mode = "1_AI_3_Humans"
            if self.game_mode == "4_AI_AUTO_TRAIN":
                # Khởi tạo 4 AI, chơi 500 ván không render
                self.render = False
                from tqdm import trange
                self.game_mode = "4_AI"
                self.players = []
                positions = ['bottom', 'right', 'top', 'left']
                for i, position in enumerate(positions):
                    is_ai = True
                    player = Player(f"Player {i+1}", position, is_ai=is_ai)
                    self.players.append(player)
                self.setup_ai_players()
                n_round = 500
                for gi in range(n_round):
                    print(f"[AUTO TRAIN] Game {gi + 1}/{n_round}")
                    self.prepare_new_game()
                    self.running = True
                    step = 0
                    stuck_counter = 0
                    stuck_limit = 200  # Nếu stuck 200 lượt vẫn không ai đánh -> force play
                    prev_hand = [-1 for _ in range(4)]
                    while self.running:
                        current_player = self.players[0]
                        print(f"[AUTO] Step {step}: Player {current_player.player_index} {current_player.name} - Hand: {len(current_player.hand)} cards")
                        try:
                            # Lấy đúng agent cho player_index hiện tại
                            agent = self.ai_agents[current_player.player_index]
                            # Log encode state
                            state_enc = agent.state_encoder.encode(self, current_player.player_index)
                            print(f"[AUTO-DBG] State head: {state_enc[:8]} ... tail: {state_enc[-8:]} (len={len(state_enc)})")
                            valid_actions = agent.get_valid_actions(self)
                            print(f"[AUTO-DBG] Valid actions: {valid_actions}")
                            action_id = agent.select_action(self)
                            action = agent.interpret_action(action_id, self)
                            print(f"[AUTO-DBG] SELECTED: action_id={action_id}, action={action}")
                            can_play = False
                            if action != "PASS":
                                can_play = current_player.can_play_cards(action if isinstance(action, list) else [], self.last_played_cards)
                                print(f"[AUTO-DBG] can_play_cards={can_play}")
                            # Action thực hiện
                            if action == "PASS" or not can_play:
                                print(f"[AUTO-DBG] Agent decided PASS or can't play -> pass_turn()")
                                self.pass_turn()
                                stuck_counter += 1
                            else:
                                print(f"[AUTO-DBG] Agent plays action -> play_cards()")
                                for card in current_player.hand:
                                    card.selected = card in action
                                self.play_cards()
                                stuck_counter = 0
                            print(f"[AUTO-DBG] After action: Player {current_player.player_index} - Hand: {len(current_player.hand)} cards")
                        except Exception as e:
                            print(f"[AI ERROR][AUTO] {str(e)}")
                            self.pass_turn()
                            stuck_counter += 1
                        self.update_game_state()
                        # Nếu stuck quá lâu không ai đánh, AI sẽ ép play random action hợp lệ
                        if stuck_counter > stuck_limit:
                            print(f"[AUTO] Force random play to break stuck after {stuck_counter} passes.")
                            valid_actions = agent.get_valid_actions(self)
                            print(f"[AUTO-DEBUG] Valid actions: {valid_actions}")
                            valid_no_pass = [a for a in valid_actions if a != 0]
                            if valid_no_pass:
                                action_id = valid_no_pass[0]
                                action = agent.interpret_action(action_id, self)
                                print(f"[AUTO-DEBUG] Forced action_id: {action_id} => action: {action}")
                                for card in current_player.hand:
                                    card.selected = card in action
                                self.play_cards()
                                print(f"[AUTO-DEBUG] After forced play, Player {current_player.player_index} - Hand: {len(current_player.hand)}")
                            else:
                                # Không có action hợp lệ, vẫn pass (avoid infinite loop)
                                print(f"[AUTO-DEBUG] NO valid non-pass action, forced pass.")
                                self.pass_turn()
                                print(f"[AUTO-DEBUG] After forced pass, Player {current_player.player_index} - Hand: {len(current_player.hand)}")
                            stuck_counter = 0
                        if len(self.players[0].hand) == 0:
                            self.running = False
                        step += 1
                    # --- Sau mỗi ván có thể gọi học/ghi model từng agent ---
                    for ai_idx, agent in self.ai_agents.items():
                        if hasattr(agent, "agent") and hasattr(agent.agent, "save"):
                            model_name = f"models/ai{ai_idx}_after_{gi+1}.pth"
                            os.makedirs("models", exist_ok=True)
                            agent.agent.save(model_name)
                print("4 AI AUTO TRAIN FINISHED.")
                return
            # --- các chế độ khác giữ nguyên ---
            self.players = []
            positions = ['bottom', 'right', 'top', 'left']
            for i, position in enumerate(positions):
                if self.game_mode == "1_AI_3_Humans":
                    is_ai = (i == 0)
                elif self.game_mode == "4_AI":
                    is_ai = True
                else:
                    is_ai = False
                player = Player(f"Player {i+1}", position, is_ai=is_ai)
                self.players.append(player)
            self.setup_ai_players()
            self.prepare_new_game()
            ai_action_cooldown = 0
            while self.running:
                current_player = self.players[self.current_player_index]
                if getattr(current_player, "is_ai", False):
                    print(f"[AI TURN] Player {self.current_player_index}")
                    try:
                        agent = self.ai_agents[self.current_player_index]
                        print("[DEBUG] Before ai_agent.select_action")
                        action_id = agent.select_action(self)
                        print(f"[AI ACTION] ID: {action_id}")
                        action = agent.interpret_action(action_id, self)
                        if action == "PASS":
                            print("[DEBUG] AI decided to pass.")
                            self.pass_turn()
                            print(f"[DEBUG] After pass_turn. current_player_index={self.current_player_index}")
                        else:
                            for card in self.players[self.current_player_index].hand:
                                card.selected = card in action
                            print(f"[DEBUG] AI decided to play. action={action}")
                            self.play_cards()
                            print(f"[DEBUG] After play_cards. current_player_index={self.current_player_index}")
                        ai_action_cooldown = 0
                        # Debug: print all player turns
                        print("[DEBUG] Player turn states:")
                        for idx, player in enumerate(self.players):
                            print(f"    Player idx={idx}, name={player.name}, is_turn={player.is_turn}, passed={player.passed}")
                    except Exception as e:
                        print(f"[AI ERROR] {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: pass automatically
                        self.pass_turn()
                else:
                    self.handle_events()
                # Always update game state
                self.update_game_state()
                if self.render:
                    import pygame
                    self.draw()
                    self.clock.tick(60)
        except Exception as e:
            print(f"Game crashed: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_game_state(self):
        """Stub for custom game state update logic if needed."""
        pass
if __name__ == '__main__':
    game = TienLenGame()
    game.run()
    try:
        import pygame
        pygame.quit()
    except ImportError:
        pass
    sys.exit()