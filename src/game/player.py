from typing import List
from .card import Card
import pygame

class Player:
    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position  # 'bottom', 'left', 'top', 'right'
        self.hand: List[Card] = []
        self.is_turn = False
        self.passed = False  # Track if player passed their turn
        # Track z-index for overlapping cards
        self.z_index_counter = 0
        
    def add_card(self, card: Card):
        """Add a card to player's hand"""
        self.hand.append(card)
        self.sort_hand()
        
    def remove_cards(self, cards: List[Card]):
        """Remove cards from player's hand"""
        for card in cards:
            if card in self.hand:
                self.hand.remove(card)
                
    def sort_hand(self):
        """Sort cards with proper suit order"""
        self.hand.sort(key=lambda x: (Card.RANKS.index(x.rank), Card.SUITS.index(x.suit)))
        
    def get_selected_cards(self) -> List[Card]:
        """Get all selected cards from player's hand."""
        return [card for card in self.hand if card.selected]

    def update_invalid_states(self):
        """Update each card's invalid state based on current selection"""
        selected = [card for card in self.hand if card.selected]
        valid = self.validate_selection(selected)
        for card in self.hand:
            card.invalid = card.selected and not valid

    def validate_selection(self, selected: List["Card"]) -> bool:
        """Improved validation with Tiến Lên rules"""
        if not selected:
            return False

        from collections import Counter
        n = len(selected)
        ranks = [Card.RANKS.index(c.rank) for c in selected]
        unique_ranks = set(ranks)

        # Single card - always valid
        if n == 1:
            return True

        # Pair/triple/quad - must be same rank
        if n in (2, 3, 4) and len(unique_ranks) == 1:
            return True

        # Straight validation
        if n >= 3:
            # Must have unique ranks
            if len(unique_ranks) != n:
                return False

            sorted_ranks = sorted(unique_ranks)
            # Check consecutive ranks
            if all(sorted_ranks[i] + 1 == sorted_ranks[i+1] for i in range(len(sorted_ranks)-1)):
                # 2 can't be in the middle of straight
                if Card.RANKS.index('2') in sorted_ranks[:-1]:
                    return False
                return True

        # Pairs straight validation (đôi thông)
        if n >= 6 and n % 2 == 0:
            # Group cards by rank
            rank_groups = {}
            for card in selected:
                rank_groups.setdefault(card.rank, []).append(card)

            # Each rank must have exactly 2 cards
            if not all(len(group) == 2 for group in rank_groups.values()):
                return False

            # Check consecutive ranks
            sorted_ranks = sorted(rank_groups.keys(), key=lambda r: Card.RANKS.index(r))
            rank_indices = [Card.RANKS.index(r) for r in sorted_ranks]
            if all(rank_indices[i] + 1 == rank_indices[i+1] for i in range(len(rank_indices)-1)):
                # 2 can't be in the middle of pairs straight
                if '2' in sorted_ranks[:-1]:
                    return False
                return True

        return False
        
    def clear_selection(self):
        """Clear all card selections"""
        for card in self.hand:
            card.selected = False
            
    def draw_hand(self, screen, card_spacing=None):
        """Draw the player's hand at its fixed position. Always show current player's cards face-up."""
        if not self.hand:
            return

        # Reset z-index counter
        self.z_index_counter = 0

        if card_spacing is None:
            if self.position in ['bottom', 'top']:
                max_spacing = min(Card.CARD_WIDTH * 0.6,
                                (screen.get_width() * 0.9) / len(self.hand))
                card_spacing = max(Card.CARD_WIDTH * 0.3, max_spacing)
            else:
                max_spacing = min(Card.CARD_HEIGHT * 0.6,
                                (screen.get_height() * 0.9) / len(self.hand))
                card_spacing = max(Card.CARD_HEIGHT * 0.3, max_spacing)

        if self.position == 'bottom':
            total_width = (len(self.hand) - 1) * card_spacing + Card.CARD_WIDTH
            start_x = (screen.get_width() - total_width) // 2
            y = screen.get_height() - Card.CARD_HEIGHT - 30
            
            for i, card in enumerate(self.hand):
                card.z_index = self.z_index_counter
                x = start_x + i * card_spacing
                card.face_up = self.is_turn
                card.draw(screen, (x + Card.CARD_WIDTH//2, y + Card.CARD_HEIGHT//2))
                self.z_index_counter += 1

        elif self.position == 'top':
            total_width = (len(self.hand) - 1) * card_spacing + Card.CARD_WIDTH
            start_x = (screen.get_width() - total_width) // 2
            y = 30

            for i, card in enumerate(self.hand):
                card.z_index = self.z_index_counter
                x = start_x + i * card_spacing
                card.face_up = self.is_turn
                card.draw(screen, (x + Card.CARD_WIDTH//2, y + Card.CARD_HEIGHT//2))
                self.z_index_counter += 1

        elif self.position == 'left':
            total_height = (len(self.hand) - 1) * card_spacing + Card.CARD_HEIGHT
            start_y = (screen.get_height() - total_height) // 2
            x = 30

            for i, card in enumerate(self.hand):
                card.z_index = self.z_index_counter
                y = start_y + i * card_spacing
                card.face_up = self.is_turn
                card.draw(screen, (x + Card.CARD_WIDTH//2, y + Card.CARD_HEIGHT//2))
                self.z_index_counter += 1

        elif self.position == 'right':
            total_height = (len(self.hand) - 1) * card_spacing + Card.CARD_HEIGHT
            start_y = (screen.get_height() - total_height) // 2
            x = screen.get_width() - Card.CARD_WIDTH - 30

            for i, card in enumerate(self.hand):
                card.z_index = self.z_index_counter
                y = start_y + i * card_spacing
                card.face_up = self.is_turn
                card.draw(screen, (x + Card.CARD_WIDTH//2, y + Card.CARD_HEIGHT//2))
                self.z_index_counter += 1
        
        # Draw player name with modern styling
        font = pygame.font.SysFont('Arial', 24)
        status = " (Turn)" if self.is_turn else " (Passed)" if self.passed else ""
        name_text = f"{self.name}{status}"
        text_color = (50, 120, 200) if self.is_turn else (150, 150, 150) if self.passed else (80, 80, 80)
        text = font.render(name_text, True, text_color)
        
        if self.position == 'bottom':
            text_rect = text.get_rect(center=(screen.get_width()//2, screen.get_height() - 10))
        elif self.position == 'top':
            text_rect = text.get_rect(center=(screen.get_width()//2, 20))
        elif self.position == 'left':
            text_rect = text.get_rect(center=(50, screen.get_height()//2))
            text = pygame.transform.rotate(text, 90)
        else:  # right
            text_rect = text.get_rect(center=(screen.get_width() - 50, screen.get_height()//2))
            text = pygame.transform.rotate(text, -90)
            
        screen.blit(text, text_rect)
                
    def handle_click(self, pos):
        """Handle mouse click with z-index priority for current player"""
        if self.is_turn and not self.passed:
            # Check cards from highest z-index (topmost) to lowest
            for card in sorted(self.hand, key=lambda c: c.z_index, reverse=True):
                if card.rect.collidepoint(pos):
                    card.toggle_selected()
                    self.update_invalid_states()
                    return True
        return False
        
    def update_hover(self, mouse_pos):
        """Update hover state for cards of current player"""
        if self.is_turn:
            for card in self.hand:
                card.update_hover(mouse_pos)
            self.update_invalid_states()
                
    def can_play_cards(self, selected_cards: List[Card], last_played_cards: List[Card] = None) -> bool:
        """Complete Tiến Lên rules implementation"""
        if not selected_cards:
            return False

        combo_type = self.get_combo_type(selected_cards)
        if combo_type == "INVALID":
            return False

        # First turn must include 3♠
        if not last_played_cards:
            return any(card.rank == '3' and card.suit == 'spades' for card in selected_cards)

        prev_combo_type = self.get_combo_type(last_played_cards)

        # Special beating rules
        if prev_combo_type == "FOUR_OF_A_KIND":
            # Only higher four of a kind or pairs straight can beat
            if combo_type not in ["FOUR_OF_A_KIND", "PAIRS_STRAIGHT"]:
                return False
        elif prev_combo_type == "PAIRS_STRAIGHT":
            # Only higher pairs straight or four of a kind can beat
            if combo_type not in ["PAIRS_STRAIGHT", "FOUR_OF_A_KIND"]:
                return False
        elif combo_type != prev_combo_type:
            # Different combo types not allowed
            return False

        # Compare same combo types
        return self.compare_combinations(selected_cards, last_played_cards) > 0
        
    def get_combo_type(self, cards: List[Card]) -> str:
        """Improved combo identification"""
        if not cards:
            return "INVALID"

        from collections import Counter
        n = len(cards)
        rank_counts = Counter(c.rank for c in cards)
        unique_ranks = set(rank_counts.keys())
        rank_indices = sorted([Card.RANKS.index(r) for r in unique_ranks])

        # Single card
        if n == 1:
            return "SINGLE"

        # Pair
        if n == 2 and len(unique_ranks) == 1:
            return "PAIR"

        # Three of a kind
        if n == 3 and len(unique_ranks) == 1:
            return "THREE"

        # Four of a kind
        if n == 4 and len(unique_ranks) == 1:
            return "FOUR_OF_A_KIND"

        # Straight
        if n >= 3:
            # Check consecutive unique ranks
            if (len(unique_ranks) == n and
                all(rank_indices[i] + 1 == rank_indices[i+1] for i in range(len(rank_indices)-1))):
                # 2 can't be in the middle
                if Card.RANKS.index('2') in rank_indices[:-1]:
                    return "INVALID"
                return "STRAIGHT"

        # Pairs straight
        if n >= 6 and n % 2 == 0:
            # Each rank must have exactly 2 cards
            if all(count == 2 for count in rank_counts.values()):
                # Check consecutive ranks
                if all(rank_indices[i] + 1 == rank_indices[i+1] for i in range(len(rank_indices)-1)):
                    # 2 can't be in the middle
                    if Card.RANKS.index('2') in rank_indices[:-1]:
                        return "INVALID"
                    return "PAIRS_STRAIGHT"

        return "INVALID"
        
    def compare_combinations(self, cards1: List[Card], cards2: List[Card]) -> int:
        """Compare two combinations of the same type"""
        # For singles, pairs, three of a kind - compare highest card
        if len(cards1) in [1, 2, 3]:
            max1 = max(cards1)
            max2 = max(cards2)
            
            # Compare ranks first
            if Card.RANKS.index(max1.rank) > Card.RANKS.index(max2.rank):
                return 1
            elif Card.RANKS.index(max1.rank) < Card.RANKS.index(max2.rank):
                return -1
                
            # If same rank, compare suits
            return 1 if Card.SUITS.index(max1.suit) > Card.SUITS.index(max2.suit) else -1
            
        # For straights and pairs straights - compare by highest card
        if len(cards1) >= 3:
            max1 = max(cards1)
            max2 = max(cards2)
            
            if Card.RANKS.index(max1.rank) > Card.RANKS.index(max2.rank):
                return 1
            elif Card.RANKS.index(max1.rank) < Card.RANKS.index(max2.rank):
                return -1
                
            # For straights with same highest card, compare suit of highest card
            return 1 if Card.SUITS.index(max1.suit) > Card.SUITS.index(max2.suit) else -1
            
        # For four of a kind - compare rank
        if len(cards1) == 4:
            rank1 = cards1[0].rank
            rank2 = cards2[0].rank
            return 1 if Card.RANKS.index(rank1) > Card.RANKS.index(rank2) else -1
            
        return 0