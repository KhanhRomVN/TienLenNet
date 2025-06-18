import pygame
from pathlib import Path

class Card:
    SUITS = ['spades', 'clubs', 'diamonds', 'hearts']  # Updated suit order: ♠️ < ♣️ < ♦️ < ♥️
    RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    
    # Updated dimensions for better aspect ratio
    CARD_WIDTH = 100
    CARD_HEIGHT = 140
    
    # Modern color scheme
    BACKGROUND_COLOR = (255, 255, 255)  # Pure white
    BORDER_COLOR = (200, 200, 200)     # Light gray
    TEXT_COLOR = (40, 40, 40)           # Dark gray

    # Suit image mapping: class variable, loaded only once
    CARD_SUIT_IMAGES = {}
    CARD_SUIT_FILEPATHS = {
        'spades':   'src/assets/logos/bich.png',
        'clubs':    'src/assets/logos/chuon.png',
        'diamonds': 'src/assets/logos/ro.png',
        'hearts':   'src/assets/logos/co.png'
    }

    SUIT_SYMBOLS = {
        'hearts': '♥',
        'diamonds': '♦',
        'clubs': '♣',
        'spades': '♠'
    }
    
    SUIT_COLORS = {
        'hearts': (220, 20, 60),    # Crimson Red
        'diamonds': (220, 20, 60),  # Crimson Red
        'clubs': (40, 40, 40),      # Dark Gray
        'spades': (40, 40, 40)      # Dark Gray
    }

    @classmethod
    def load_suit_images(cls):
        if not cls.CARD_SUIT_IMAGES:
            for suit, path in cls.CARD_SUIT_FILEPATHS.items():
                # Convert path to universal (in case run from root or other OS)
                file_path = Path(path)
                try:
                    img = pygame.image.load(str(file_path)).convert_alpha()
                    cls.CARD_SUIT_IMAGES[suit] = img
                except Exception as e:
                    print(f"Cannot load suit image for {suit} at {path}: {e}")
                    cls.CARD_SUIT_IMAGES[suit] = None

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.face_up = False
        self.selected = False
        self.width = self.CARD_WIDTH
        self.height = self.CARD_HEIGHT
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.hover = False
        self.selection_animation = 0
        self.z_index = 0  # For handling overlapping cards

    def draw(self, screen, position, size=None):
        Card.load_suit_images()
        if size:
            self.width, self.height = size
        
        # Create surface with per-pixel alpha for advanced effects
        card_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Selection animation
        if self.selected:
            self.selection_animation = min(self.selection_animation + 0.2, 1.0)
        else:
            self.selection_animation = max(self.selection_animation - 0.2, 0.0)
        
        # Apply hover and selection effects
        offset_y = 0
        if self.hover and not self.selected:
            offset_y = -8
        if self.selection_animation > 0:
            offset_y = -15 * self.selection_animation
            
        # Update position with offset
        self.rect.center = (position[0], position[1] + offset_y)
        
        if self.face_up:
            # Draw card face with modern styling
            pygame.draw.rect(card_surface, self.BACKGROUND_COLOR, (0, 0, self.width, self.height),
                            border_radius=10)
            pygame.draw.rect(card_surface, self.BORDER_COLOR, (0, 0, self.width, self.height),
                            2, border_radius=10)
            
            # Get suit color
            suit_color = self.SUIT_COLORS[self.suit]
            
            # Draw rank with modern typography
            # Smaller font for modern look
            font = pygame.font.SysFont('Arial', int(self.height * 0.18), bold=True)
            # Main rank text (smaller)
            rank_text = font.render(self.rank, True, suit_color)
            rank_pos = (10, 10)
            card_surface.blit(rank_text, rank_pos)

            # Center suit icon smaller
            suit_img = self.CARD_SUIT_IMAGES.get(self.suit)
            if suit_img:
                center_h = int(self.height * 0.22)
                center_w = int(suit_img.get_width() * center_h / suit_img.get_height())
                suit_img_center = pygame.transform.smoothscale(suit_img, (center_w, center_h))
                center_pos = (self.width//2 - center_w//2,
                             self.height//2 - center_h//2)
                card_surface.blit(suit_img_center, center_pos)
            else:
                center_suit_size = int(self.height * 0.22)
                center_suit_font = pygame.font.SysFont('Arial', center_suit_size)
                center_suit = center_suit_font.render(self.SUIT_SYMBOLS[self.suit], True, suit_color)
                center_pos = (self.width//2 - center_suit.get_width()//2,
                             self.height//2 - center_suit.get_height()//2)
                card_surface.blit(center_suit, center_pos)
            
        else:
            # Modern card back design with white background and inset border
            # Main white background
            pygame.draw.rect(card_surface, (255, 255, 255), (0, 0, self.width, self.height),
                            border_radius=10)
            
            # Outer border (light gray)
            pygame.draw.rect(card_surface, (220, 220, 220), (0, 0, self.width, self.height),
                            2, border_radius=10)
            
            # Inner border (black, inset)
            inset = 10  # Inset amount in pixels
            inner_rect = pygame.Rect(inset, inset, self.width - 2*inset, self.height - 2*inset)
            pygame.draw.rect(card_surface, (40, 40, 40), inner_rect, 2, border_radius=8)
            
            # Subtle pattern inside the inner border
            pattern_rect = pygame.Rect(inset + 4, inset + 4,
                                     self.width - 2*(inset + 4),
                                     self.height - 2*(inset + 4))
            pattern_color = (245, 245, 245)  # Very light gray
            
            # Draw diagonal lines pattern
            for i in range(0, pattern_rect.width + pattern_rect.height, 15):
                start_x = max(pattern_rect.left, pattern_rect.left + i - pattern_rect.height)
                end_x = min(pattern_rect.right, pattern_rect.left + i)
                start_y = min(pattern_rect.bottom, pattern_rect.bottom - (pattern_rect.left + i - start_x))
                end_y = max(pattern_rect.top, pattern_rect.bottom - i)
                pygame.draw.line(card_surface, pattern_color, (start_x, start_y), (end_x, end_y), 1)
        
        # Draw selection/hover effects
        # Bordered highlight with optimal width/colors
        border_width = 2 if (self.selected or self.hover) else 0
        if border_width > 0:
            # Border color priority: invalid > select > hover
            if hasattr(self, "invalid") and self.invalid:
                border_color = (220, 50, 50)  # Red for invalid (always highest priority)
            elif self.selected:
                border_color = (60, 180, 75)  # Green for select
            elif self.hover:
                border_color = (255, 215, 0)    # Vàng gold cho hover
            else:
                border_color = (70, 130, 180)
            pygame.draw.rect(card_surface, border_color,
                            (0, 0, self.width, self.height), border_width, border_radius=10)
        
        # Draw card on screen
        screen.blit(card_surface, self.rect.topleft)

    def toggle_selected(self):
        """Toggle card selection state"""
        self.selected = not self.selected
        
    def update_hover(self, mouse_pos):
        """Update hover state based on mouse position"""
        self.hover = self.rect.collidepoint(mouse_pos)
        
    def __str__(self):
        return f"{self.rank} of {self.suit}"
        
    def __lt__(self, other):
        """Compare cards based on Tiến Lên rules"""
        if not isinstance(other, Card):
            return NotImplemented
        rank_order = {rank: idx for idx, rank in enumerate(self.RANKS)}
        return rank_order[self.rank] < rank_order[other.rank] 