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
        """Get all selected cards from player's hand"""
        return [card for card in self.hand if card.selected]
        
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
                    return True
        return False
        
    def update_hover(self, mouse_pos):
        """Update hover state for cards of current player"""
        if self.is_turn:
            for card in self.hand:
                card.update_hover(mouse_pos)
                
    def can_play_cards(self, selected_cards: List[Card], last_played_cards: List[Card] = None) -> bool:
        """Improved rules implementation"""
        if not selected_cards:
            return False
            
        # Validate the combination
        combo_type = self.get_combo_type(selected_cards)
        if combo_type == "INVALID":
            return False
            
        # First turn must include 3♠
        if not last_played_cards:
            has_three_spades = any(card.rank == '3' and card.suit == 'spades' for card in selected_cards)
            if not has_three_spades:
                return False
            return True
            
        # Get previous combo type
        prev_combo_type = self.get_combo_type(last_played_cards)
        
        # Special cases
        if prev_combo_type == "FOUR_OF_A_KIND":
            # Only higher four of a kind or pairs straight can beat
            return combo_type in ["FOUR_OF_A_KIND", "PAIRS_STRAIGHT"] and \
                   self.compare_combinations(selected_cards, last_played_cards) > 0
                   
        if prev_combo_type == "PAIRS_STRAIGHT":
            # Only higher pairs straight or four of a kind can beat
            return combo_type in ["PAIRS_STRAIGHT", "FOUR_OF_A_KIND"] and \
                   self.compare_combinations(selected_cards, last_played_cards) > 0
                   
        # Same combo type comparison
        if combo_type == prev_combo_type:
            return self.compare_combinations(selected_cards, last_played_cards) > 0
                   
        return False
        
    def get_combo_type(self, cards: List[Card]) -> str:
        """Identify the combination type"""
        if not cards:
            return "INVALID"
            
        # Sort cards by rank
        sorted_cards = sorted(cards, key=lambda x: Card.RANKS.index(x.rank))
        
        # Single card
        if len(cards) == 1:
            return "SINGLE"
            
        # Get rank counts
        rank_counts = {}
        for card in cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
            
        # Pair
        if len(cards) == 2 and len(rank_counts) == 1:
            return "PAIR"
            
        # Three of a kind
        if len(cards) == 3 and len(rank_counts) == 1:
            return "THREE"
            
        # Four of a kind
        if len(cards) == 4 and len(rank_counts) == 1:
            return "FOUR_OF_A_KIND"
            
        # Straight (3+ consecutive cards)
        if len(cards) >= 3:
            ranks = [Card.RANKS.index(c.rank) for c in sorted_cards]
            
            # Check if consecutive and no 2 in the middle
            if all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1)):
                # 2 can only be at the end
                if any(r == Card.RANKS.index('2') for r in ranks[:-1]):
                    return "INVALID"
                return "STRAIGHT"
                
        # Pairs straight (even number of cards, 4+)
        if len(cards) >= 4 and len(cards) % 2 == 0:
            # All ranks must have exactly 2 cards
            if all(count == 2 for count in rank_counts.values()):
                sorted_ranks = sorted(rank_counts.keys(), key=lambda x: Card.RANKS.index(x))
                rank_indices = [Card.RANKS.index(r) for r in sorted_ranks]
                
                # Check consecutive and no 2 in the middle
                if all(rank_indices[i] + 1 == rank_indices[i + 1] for i in range(len(rank_indices) - 1)):
                    if any(idx == Card.RANKS.index('2') for idx in rank_indices[:-1]):
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