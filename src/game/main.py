import pygame
import sys
from pathlib import Path
from random import shuffle
from .card import Card
from .player import Player

# Initialize Pygame
pygame.init()
pygame.font.init()

# Get the screen info
screen_info = pygame.display.Info()
WINDOW_WIDTH = screen_info.current_w
WINDOW_HEIGHT = screen_info.current_h
PLAYER_POSITIONS = ['bottom', 'left', 'top', 'right']

# Modern color scheme
BACKGROUND_COLOR = (245, 245, 245)  # Light gray
CARD_AREA_COLOR = (230, 230, 230)   # Slightly darker gray
TEXT_COLOR = (50, 50, 50)           # Dark gray
ACCENT_COLOR = (70, 130, 180)       # Steel blue

class TienLenGame:
    def __init__(self):
        self.center_cards = []
        self.center_cards_pos = []
        self.current_player_index = 0  # Fix: Init current player index
        self.last_played_cards = []
        self.last_played_by = None
        screen_info = pygame.display.Info()
        self.screen = pygame.display.set_mode((screen_info.current_w, screen_info.current_h), pygame.FULLSCREEN)
        pygame.display.set_caption("Tiến Lên - Modern Edition")
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        self.game_count = 0  # Track number of games played
        self.player_rankings = []  # Store ranking indices after each game

        # Initialize players
        self.players = []
        for i, position in enumerate(PLAYER_POSITIONS):
            player = Player(f"Player {i+1}", position)
            self.players.append(player)

        # Modern UI elements
        button_width, button_height = 120, 45
        self.play_button = pygame.Rect(WINDOW_WIDTH - button_width - 20,
                                      WINDOW_HEIGHT - button_height - 20,
                                      button_width, button_height)
        self.pass_button = pygame.Rect(WINDOW_WIDTH - button_width*2 - 30,
                                      WINDOW_HEIGHT - button_height - 20,
                                      button_width, button_height)

    def get_player_at_position(self, position: str):
        """Get player by their position (bottom, left, top, right)"""
        for player in self.players:
            if player.position == position:
                return player
        return None

    def prepare_new_game(self):
        """Prepare for a new game based on previous results"""
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
        self.deck = self.create_deck()
        self.deal_cards()

        if self.game_count == 1:
            # First game: AFTER dealing, arrange so player with 3♠ is bottom
            starting_index_old = self.find_starting_player()
            self.arrange_positions_by_starting_player(starting_index_old)
        elif self.game_count > 1:
            # Rearrange by previous game rankings
            self.rearrange_players()

        # Ván nào cũng set index 0 là đáy và là người đi đầu
        self.current_player_index = 0
        self.players[self.current_player_index].is_turn = True
        self.round_starter = self.current_player_index

    def arrange_positions_by_starting_player(self, starting_index_old):
        """Arrange the players so that the starting player (with 3 of spades) is at bottom, and others in counter-clockwise order."""
        positions = ['bottom', 'right', 'top', 'left']
        new_players = []
        for i in range(4):
            idx = (starting_index_old + i) % 4
            new_players.append(self.players[idx])
        for i, player in enumerate(new_players):
            player.position = positions[i]
        self.players = new_players

    def rearrange_players(self):
        """Rearrange player positions based on previous ranking"""
        # Make a new player list in rank order
        ranked_players = [self.players[i] for i in self.player_rankings]
        # Assign new positions (bottom, right, top, left)
        positions = ['bottom', 'right', 'top', 'left']
        for i, player in enumerate(ranked_players):
            player.position = positions[i]
        self.players = ranked_players

    def end_game(self, winner_index):
        """Handle the end of a round, updating rankings and preparing new game"""
        self.determine_rankings(winner_index)
        self.prepare_new_game()

    def determine_rankings(self, winner_index):
        """Determine player rankings after a game (winner first, rest by hand count then play order)"""
        winner = self.players[winner_index]
        # Get (index, player) pairs for all others
        remaining = [(i, p) for i, p in enumerate(self.players) if i != winner_index]
        # Sort by fewest cards, breaking ties by order after winner
        start_index = (winner_index + 1) % 4
        for offset in range(3):
            idx = (start_index + offset) % 4
            if idx != winner_index:
                pass  # order priority if needed
        
        # You can refine the tie-break order here if needed
        sorted_other = sorted(remaining, key=lambda t: (len(t[1].hand), 0))  # 0 is placeholder, you can adjust
        # New player_rankings is [winner_index, ...indices...]
        self.player_rankings = [winner_index] + [idx for (idx, _) in sorted_other]

    def create_gradient_background(self):
        """Create a light modern background"""
        background = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        background.fill(BACKGROUND_COLOR)
        
        # Add subtle pattern
        for x in range(0, WINDOW_WIDTH, 40):
            for y in range(0, WINDOW_HEIGHT, 40):
                if (x//40 + y//40) % 2 == 0:
                    pygame.draw.circle(background, (235, 235, 235), (x, y), 2)
        
        return background

    def create_deck(self):
        """Create and return a shuffled deck of cards"""
        deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                card = Card(suit, rank)
                deck.append(card)
        shuffle(deck)
        return deck

    def find_starting_player(self) -> int:
        """Find the player with 3♠ to start the game"""
        for i, player in enumerate(self.players):
            for card in player.hand:
                if card.rank == '3' and card.suit == 'spades':
                    # Player who has the 3♠ starts, but do not auto-select the card
                    return i
        return 0  # Fallback to first player if 3♠ not found

    def deal_cards(self):
        """Deal 13 cards to each player and check for instant win conditions"""
        for i in range(13):
            for player in self.players:
                if self.deck:
                    player.add_card(self.deck.pop())

        # Check for instant win conditions after dealing
        for player in self.players:
            if self.check_instant_win(player):
                print(f"{player.name} wins with instant win condition!")
                winner_index = self.players.index(player)
                self.end_game(winner_index)
                return

    def check_instant_win(self, player: Player) -> bool:
        """Check for instant win conditions (Tới Trắng)"""
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
        pairs = [r for r, cards in rank_groups.items() if len(cards) == 2]
        if len(pairs) >= 5:
            pairs.sort(key=lambda x: Card.RANKS.index(x))
            for i in range(len(pairs) - 4):
                if all(Card.RANKS.index(pairs[j]) + 1 == Card.RANKS.index(pairs[j + 1]) 
                      for j in range(i, i + 4)):
                    return True
                    
        # Check for 6 Đôi Bất Kỳ (6 pairs of any rank)
        if len([r for r, cards in rank_groups.items() if len(cards) == 2]) >= 6:
            return True
            
        # Check for Tứ Quý 2 (four 2s)
        if len(rank_groups.get('2', [])) == 4:
            return True
            
        # Check for 6 Lá Cùng Số (6 cards of same rank)
        if any(len(cards) >= 6 for cards in rank_groups.values()):
            return True
            
        # Check for Đồng Chất (all cards of same suit)
        suit_groups = {}
        for card in cards:
            suit_groups.setdefault(card.suit, []).append(card)
        if any(len(cards) == 13 for cards in suit_groups.values()):
            return True
            
        return False

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        
        # Update hover/click only for current_player
        mouse_pos = pygame.mouse.get_pos()
        current_player = self.players[self.current_player_index]
        current_player.update_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if current_player.is_turn:
                        # Handle card selection for current player
                        current_player.handle_click(event.pos)
                        # Handle button clicks
                        if self.play_button.collidepoint(event.pos):
                            self.play_cards()
                        elif self.pass_button.collidepoint(event.pos):
                            self.pass_turn()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if current_player.is_turn:
                        self.play_cards()
                elif event.key == pygame.K_p:
                    if current_player.is_turn:
                        self.pass_turn()

    def play_cards(self):
        """Play selected cards if valid"""
        current_player = self.players[self.current_player_index]
        selected_cards = current_player.get_selected_cards()

        if current_player.can_play_cards(selected_cards, self.last_played_cards):
            # Remove cards from hand
            current_player.remove_cards(selected_cards)

            # Update game state
            self.last_played_cards = selected_cards
            self.last_played_by = self.current_player_index

            # Update center cards display
            self.center_cards = selected_cards
            self.arrange_center_cards()

            # Check for win condition
            if not current_player.hand:
                print(f"{current_player.name} wins!")
                self.end_game(self.current_player_index)  # update rankings and start new game
                return

            self.next_turn()
        else:
            current_player.clear_selection()

    def pass_turn(self):
        """Pass the current turn"""
        current_player = self.players[self.current_player_index]
        current_player.passed = True
        current_player.clear_selection()
        
        # Check if everyone has passed
        passed_count = sum(1 for p in self.players if p.passed)
        if passed_count >= 3:
            # Reset for new round
            for player in self.players:
                player.passed = False
            self.last_played_cards = []
            self.center_cards = []
            
        self.next_turn()

    def next_turn(self):
        """Move to the next player's turn"""
        self.players[self.current_player_index].is_turn = False
        self.current_player_index = (self.current_player_index + 1) % 4
        self.players[self.current_player_index].is_turn = True

    def arrange_center_cards(self):
        """Arrange cards in the center of the screen"""
        if not self.center_cards:
            return
            
        # Calculate positions for center cards
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        card_spacing = Card.CARD_WIDTH // 2
        
        total_width = (len(self.center_cards) - 1) * card_spacing + Card.CARD_WIDTH
        start_x = center_x - total_width // 2
        
        self.center_cards_pos = []
        for i in range(len(self.center_cards)):
            x = start_x + i * card_spacing
            self.center_cards_pos.append((x, center_y))

    def draw(self):
        # Draw modern background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw center card area
        center_rect = pygame.Rect(WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 - 100, 400, 200)
        pygame.draw.rect(self.screen, CARD_AREA_COLOR, center_rect, border_radius=15)
        pygame.draw.rect(self.screen, (200, 200, 200), center_rect, 2, border_radius=15)
        
        # Draw center cards
        for card, pos in zip(self.center_cards, self.center_cards_pos):
            card.face_up = True
            card.draw(self.screen, pos)

        # Rotate players so current player is at bottom
        rotated_positions = PLAYER_POSITIONS[:]  # Copy the positions list
        bottom_idx = PLAYER_POSITIONS.index('bottom')
        rotation = (4 - ((self.current_player_index - bottom_idx) % 4)) % 4
        rotated_positions = rotated_positions[rotation:] + rotated_positions[:rotation]
        
        # Save original positions
        original_positions = {player: player.position for player in self.players}
        
        # Assign rotated positions
        # No longer rotate positions -- keep players in fixed positions
        # for player, new_pos in zip(self.players, rotated_positions):
        #     player.position = new_pos
            
        # Draw player hands
        for player in self.players:
            player.draw_hand(self.screen)
            
        # Restore original positions
        for player in self.players:
            player.position = original_positions[player]

        # Draw UI elements
        self.draw_ui()
        
        pygame.display.flip()

    def draw_ui(self):
        # Draw game title
        title = self.title_font.render("TIẾN LÊN", True, ACCENT_COLOR)
        self.screen.blit(title, (20, 20))
        
        # Draw current turn indicator
        turn_text = self.font.render(f"Current turn: {self.players[self.current_player_index].name}", 
                                   True, TEXT_COLOR)
        self.screen.blit(turn_text, (20, 70))
        
        # Draw last played info if exists
        if self.last_played_cards:
            last_play_text = self.font.render(
                f"Last played by {self.players[self.last_played_by].name}: {len(self.last_played_cards)} cards",
                True, TEXT_COLOR
            )
            self.screen.blit(last_play_text, (20, 100))
        
        # Draw buttons for current player
        if self.players[self.current_player_index].is_turn:
            # Play button
            pygame.draw.rect(self.screen, (100, 200, 100), self.play_button, border_radius=8)
            pygame.draw.rect(self.screen, (50, 150, 50), self.play_button, 2, border_radius=8)
            play_text = self.font.render("PLAY", True, (255, 255, 255))
            play_rect = play_text.get_rect(center=self.play_button.center)
            self.screen.blit(play_text, play_rect)
            
            # Pass button
            pygame.draw.rect(self.screen, (240, 100, 100), self.pass_button, border_radius=8)
            pygame.draw.rect(self.screen, (180, 50, 50), self.pass_button, 2, border_radius=8)
            pass_text = self.font.render("PASS", True, (255, 255, 255))
            pass_rect = pass_text.get_rect(center=self.pass_button.center)
            self.screen.blit(pass_text, pass_rect)

    def run(self):
        # Prepare the first game (reset & deal etc.)
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