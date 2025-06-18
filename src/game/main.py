import pygame
import sys
from .card import Card
from .player import Player
from .game_logic import GameLogic
from .ui import constants as UI

# Initialize Pygame
pygame.init()
pygame.font.init()

# Get the screen info for responsive sizing
screen_info = pygame.display.Info()
WINDOW_WIDTH = screen_info.current_w
WINDOW_HEIGHT = screen_info.current_h

class TienLenGame:
    def __init__(self):
        self.center_cards = []
        self.center_cards_pos = []
        self.current_player_index = 0
        self.last_played_cards = []
        self.last_played_by = None
        screen_info = pygame.display.Info()
        self.screen = pygame.display.set_mode((screen_info.current_w, screen_info.current_h), pygame.FULLSCREEN)
        pygame.display.set_caption("Tiến Lên - Modern Edition")
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        self.game_count = 0
        self.player_rankings = []

        # Initialize players
        self.players = []
        for i, position in enumerate(UI.PLAYER_POSITIONS):
            player = Player(f"Player {i+1}", position)
            self.players.append(player)

        # UI buttons use latest window size
        button_width, button_height = 120, 45
        self.play_button = pygame.Rect(WINDOW_WIDTH - button_width - 20,
                                      WINDOW_HEIGHT - button_height - 20,
                                      button_width, button_height)
        self.pass_button = pygame.Rect(WINDOW_WIDTH - button_width*2 - 30,
                                      WINDOW_HEIGHT - button_height - 20,
                                      button_width, button_height)

    def get_player_at_position(self, position: str):
        for player in self.players:
            if player.position == position:
                return player
        return None

    def prepare_new_game(self):
        self.game_count += 1
        self.last_played_cards = []
        self.last_played_by = None
        self.center_cards = []
        self.center_cards_pos = []

        # Reset player states
        for player in self.players:
            player.hand = []
            player.is_turn = False
            player.passed = False

        # Shuffle and deal new cards
        self.deck = GameLogic.create_deck()
        self.deal_cards()

        if self.game_count == 1:
            starting_index_old = GameLogic.find_starting_player(self.players)
            self.arrange_positions_by_starting_player(starting_index_old)
        elif self.game_count > 1:
            self.rearrange_players()

        # Ván nào cũng set index 0 là đáy và là người đi đầu
        self.current_player_index = 0
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
        self.players = new_players

    def rearrange_players(self):
        ranked_players = [self.players[i] for i in self.player_rankings]
        positions = ['bottom', 'right', 'top', 'left']
        for i, player in enumerate(ranked_players):
            player.position = positions[i]
        self.players = ranked_players

    def end_game(self, winner_index):
        self.determine_rankings(winner_index)
        self.prepare_new_game()

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
        background = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        background.fill(UI.BACKGROUND_COLOR)
        for x in range(0, WINDOW_WIDTH, 40):
            for y in range(0, WINDOW_HEIGHT, 40):
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
        mouse_pos = pygame.mouse.get_pos()
        current_player = self.players[self.current_player_index]
        current_player.update_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if current_player.is_turn:
                        current_player.handle_click(event.pos)
                        if self.play_button.collidepoint(event.pos):
                            # Chỉ cho phép Play khi hợp lệ
                            selected_cards = current_player.get_selected_cards()
                            if current_player.validate_selection(selected_cards):
                                self.play_cards()
                        elif self.pass_button.collidepoint(event.pos):
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
        current_player = self.players[self.current_player_index]
        selected_cards = current_player.get_selected_cards()

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

            # Sau mỗi lượt đánh thành công, rotate vị trí ghế (bottom→left, right→bottom, v.v...)
            self.rotate_players()
            self.current_player_index = 0
            self.players[self.current_player_index].is_turn = True
        else:
            current_player.clear_selection()

    def rotate_players(self):
        """Rotate seat positions left after each turn. Bottom->Left, Right->Bottom, Top->Right, Left->Top"""
        self.players = self.players[1:] + [self.players[0]]
        positions = ['bottom', 'right', 'top', 'left']
        for i, player in enumerate(self.players):
            player.position = positions[i]
        # Cập nhật trạng thái lượt
        for i, player in enumerate(self.players):
            player.is_turn = (i == 0)

    def pass_turn(self):
        current_player = self.players[self.current_player_index]

        # Rule: In the first round of the first game, the player holding 3♠ MUST play, cannot pass
        if self.game_count == 1 and not self.last_played_cards:
            has_three_spades = any(card.rank == '3' and card.suit == 'spades' for card in current_player.hand)
            if has_three_spades:
                # Optionally: Show a message/feedback
                # print("Phải đánh 3♠ lượt đầu, không được Pass.")
                return

        # Rule: From round 2 onwards, top 1 from previous game (now at bottom, player index 0) must play first, can't pass at first turn
        if self.game_count > 1 and not self.last_played_cards and self.current_player_index == 0:
            # print("Top 1 ván trước phải đánh lượt đầu ván mới, không được Pass.")
            return

        current_player.passed = True
        current_player.clear_selection()
        passed_count = sum(1 for p in self.players if p.passed)
        if passed_count >= 3:
            for player in self.players:
                player.passed = False
            self.last_played_cards = []
            self.center_cards = []
        self.next_turn()

    def next_turn(self):
        self.players[self.current_player_index].is_turn = False
        self.current_player_index = (self.current_player_index + 1) % 4
        self.players[self.current_player_index].is_turn = True

    def arrange_center_cards(self):
        if not self.center_cards:
            return
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        card_spacing = UI.CARD_WIDTH // 2
        total_width = (len(self.center_cards) - 1) * card_spacing + UI.CARD_WIDTH
        start_x = center_x - total_width // 2
        self.center_cards_pos = []
        for i in range(len(self.center_cards)):
            x = start_x + i * card_spacing
            self.center_cards_pos.append((x, center_y))

    def draw(self):
        self.screen.fill(UI.BACKGROUND_COLOR)
        # Center card area
        center_rect = pygame.Rect(WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 - 100, 400, 200)
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
        title = self.title_font.render("TIẾN LÊN", True, UI.ACCENT_COLOR)
        self.screen.blit(title, (20, 20))
        turn_text = self.font.render(f"Current turn: {self.players[self.current_player_index].name}", True, UI.TEXT_COLOR)
        self.screen.blit(turn_text, (20, 70))
        if self.last_played_cards:
            last_play_text = self.font.render(
                f"Last played by {self.players[self.last_played_by].name}: {len(self.last_played_cards)} cards", True, UI.TEXT_COLOR
            )
            self.screen.blit(last_play_text, (20, 100))
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

    def run(self):
        self.prepare_new_game()
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

if __name__ == '__main__':
    game = TienLenGame()
    game.run()
    pygame.quit()
    sys.exit()