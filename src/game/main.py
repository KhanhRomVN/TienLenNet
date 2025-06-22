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
        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 720
        if self.render:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
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

        # LLM AI toggles (auto-read from environment)
        self.USE_LLM_AGENT = os.environ.get("USE_LLM_AGENT", "False").lower() in ["1", "true", "yes"]
        self.LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        self.consecutive_passes = 0
        self.players = []
        self.passed_player_ids = []
        self.running = True
        self.game_count = 0
        self.player_rankings = []

        # Ensure LLM_MODEL is always set before initialize_players
        self.initialize_players()
        # Debug print for each player and ai_agent
        print("[DEBUG] Player assignments after init:")
        for player in self.players:
            agent = getattr(self, 'ai_agents', {}).get(player.id, None)
            print(f"  Player {player.name} (id={player.id}) | is_ai={player.is_ai} | ai_agent={type(agent)}")

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
        current_player = self.get_player_by_id(self.current_player_id)
        selected_cards = current_player.get_selected_cards()

        # Nếu không chọn lá bài nào thì không làm gì cả, tránh clear bàn do call thừa
        if not selected_cards:
            return

        # Prevent crash: log_move may not be defined
        if hasattr(self, "log_move"):
            self.log_move(current_player, selected_cards if selected_cards else None)
        
        if current_player.can_play_cards(selected_cards, self.last_played_cards):
            current_player.remove_cards(selected_cards)
            self.last_played_cards = selected_cards
            self.last_played_by = current_player.id
            self.center_cards = selected_cards
            self.arrange_center_cards()  # luôn cập nhật vị trí bài

            current_player.passed = False
            self.consecutive_passes = 0
        else:
            # Chỉ clear center_cards (hiển thị bài sai) nếu đã từng có bài trên bàn/lượt chặn,
            # Còn lượt đầu, không clear tức là giữ nguyên trạng thái bàn
            if self.last_played_cards:
                self.center_cards = []
                self.center_cards_pos = []
                self.arrange_center_cards()
                self.last_played_cards = []

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

        # Xoay vị trí vật lý các player sau mỗi lượt
        self.rotate_player_positions()

    def pass_turn(self):
        current_player = self.get_player_by_id(self.current_player_id)
        if current_player is None:
            return

        # Đánh dấu player này đã pass nếu chưa có trong danh sách
        if current_player.id not in self.passed_player_ids:
            self.passed_player_ids.append(current_player.id)
        current_player.passed = True
        current_player.clear_selection()
        current_player.update_after_pass()

        # Nếu đủ n-1 người pass, reset bàn: người cuối cùng đánh gần nhất sẽ giữ lượt
        if len(self.passed_player_ids) >= len(self.players) - 1:
            self.reset_table()
            self.current_player_id = self.last_played_by if self.last_played_by else current_player.id
            self.last_played_by = None
            self.passed_player_ids = []
            for p in self.players:
                p.passed = False
            self.rotate_player_positions()
            return

        # Xác định player chưa pass tiếp theo
        current_player.is_turn = False
        curr_idx = self.players.index(current_player)
        for i in range(1, len(self.players) + 1):
            nx = (curr_idx + i) % len(self.players)
            next_player = self.players[nx]
            if next_player.id not in self.passed_player_ids:
                self.current_player_id = next_player.id
                self.players[nx].is_turn = True
                break
        self.rotate_player_positions()

        if self.game_count == 1 and not self.last_played_cards:
            has_three_spades = any(card.rank == '3' and card.suit == 'spades' for card in current_player.hand)
            if has_three_spades:
                return

        current_player.passed = True
        current_player.clear_selection()
        current_player.update_after_pass()
        self.consecutive_passes += 1

        if self.last_played_by is not None:
            all_others_passed = True
            for player in self.players:
                if player.id == self.last_played_by:
                    continue
                if not player.passed:
                    all_others_passed = False
                    break

            if all_others_passed:
                self.reset_table()
                self.current_player_id = self.last_played_by
                self.last_played_by = None
                for player in self.players:
                    player.passed = False
                self.consecutive_passes = 0
                return

        current_player.is_turn = False
        next_player_idx = (self.players.index(current_player) + 1) % 4
        self.current_player_id = self.players[next_player_idx].id
        self.players[next_player_idx].is_turn = True

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
            # Button click actions
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            if mouse_buttons[0]:
                if self.play_button.collidepoint(mouse_pos):
                    self.play_cards()
                elif self.pass_button.collidepoint(mouse_pos):
                    self.pass_turn()

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
                current_player = self.get_player_by_id(self.current_player_id)
                
                if current_player.is_ai:
                    # ... AI logic ...
                    try:
                        agent = self.ai_agents[current_player.id]
                        if isinstance(agent, GeminiAgent):
                            llm_action = agent.select_action(self, current_player)
                            if llm_action == "PASS":
                                self.pass_turn()
                            else:
                                for card in current_player.hand:
                                    card.selected = card in llm_action
                                self.play_cards()
                        else:
                            action_id = agent.select_action(self)
                            action = agent.interpret_action(action_id, self)
                            if action == "PASS":
                                self.pass_turn()
                            else:
                                for card in current_player.hand:
                                    card.selected = card in action
                                self.play_cards()
                    except Exception as e:
                        self.pass_turn()
                else:
                    self.handle_events()

                if self.render:
                    # ... drawing logic ...
                    import pygame
                    self.draw()
                    self.clock.tick(60)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()

    def rotate_player_positions(self):
        # Gán lại player.position sao cho current_player luôn là 'bottom', xoay visual đúng.
        positions = ['bottom', 'right', 'top', 'left']
        num_players = len(self.players)
        if num_players == 4:
            # Tìm index của current_player
            idx = None
            for i, player in enumerate(self.players):
                if player.id == self.current_player_id:
                    idx = i
                    break
            if idx is not None:
                for offset in range(4):
                    player = self.players[(idx + offset) % 4]
                    player.position = positions[offset]

    def reset_table(self):
        self.last_played_cards = []
        self.last_played_by = None
        self.center_cards = []
        self.arrange_center_cards()
        self.consecutive_passes = 0
        self.passed_player_ids = []
        for player in self.players:
            player.passed = False

    def prepare_new_game(self):
        """Prepare a new game: deal cards, reset state, find starter."""
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

if __name__ == '__main__':
    game = TienLenGame()
    game.run()
    try:
        import pygame
        pygame.quit()
    except ImportError:
        pass
    sys.exit()