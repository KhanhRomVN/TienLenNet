import pygame
from pathlib import Path

class Card:
    SUITS = ['spades', 'clubs', 'diamonds', 'hearts']  # Updated suit order: ♠️ < ♣️ < ♦️ < ♥️
    RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    
    # Updated dimensions for better aspect ratio
    CARD_WIDTH = 100
    CARD_HEIGHT = 140
    
    # Modern color scheme
    BACKGROUND_COLOR = (246, 246, 240)  # #f6f6f0 cream
    BORDER_COLOR = (82, 170, 165)       # #52aaa5 teal
    TEXT_COLOR = (0, 0, 0)              # Black

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
        'clubs': (0, 0, 0),         # Black
        'spades': (0, 0, 0)         # Black
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
        self.hover_animation = 0  # Animation progress for hover lift [0...1]
        self.z_index = 0  # For handling overlapping cards
        self._face_up_cache = None  # Cache for face up rendering

    def draw(self, screen, position, size=None):
        Card.load_suit_images()
        if size:
            self.width, self.height = size

        # --- Selection animation ---
        if self.selected:
            self.selection_animation = min(self.selection_animation + 0.20, 1.0)
        else:
            self.selection_animation = max(self.selection_animation - 0.20, 0.0)

        # --- Hover animation (independent, smooth) ---
        if self.hover and not self.selected:
            self.hover_animation = min(self.hover_animation + 0.40, 1.0)
        else:
            self.hover_animation = max(self.hover_animation - 0.40, 0.0)

        # --- Calculate translateY offset_y ---
        if self.selected:
            offset_y = -21
        elif self.hover_animation > 0:
            offset_y = -int(18 * self.hover_animation)
        else:
            offset_y = 0

        # Update position with offset
        self.rect.center = (position[0], position[1] + offset_y)

        # Draw the card's appearance
        if self.face_up:
            card_surface = self.get_face_up_surface()
        else:
            # Use normal drawing code for back design (not cached)
            card_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            # Modern card back design with white background and inset border
            pygame.draw.rect(card_surface, (255, 255, 255), (0, 0, self.width, self.height), border_radius=10)
            # Outer border (light gray)
            pygame.draw.rect(card_surface, (220, 220, 220), (0, 0, self.width, self.height), 2, border_radius=10)
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

        # Selection/hover effects and borders
        pygame.draw.rect(card_surface, (0, 0, 0), (0, 0, self.width, self.height), 2, border_radius=10)
        border_effect_width = 0
        if hasattr(self, "invalid") and self.invalid and self.selected:
            border_effect_width = 4
            border_color = (220, 50, 50)  # Red for invalid
        elif self.selected:
            border_effect_width = 4
            border_color = (82, 170, 165)  # Teal for select
        if border_effect_width > 0:
            pygame.draw.rect(card_surface, border_color,
                             (0, 0, self.width, self.height), border_effect_width, border_radius=10)
        screen.blit(card_surface, self.rect.topleft)

    def get_face_up_surface(self):
        """Return a cached surface of the card's face (front) for fast rendering."""
        if self._face_up_cache is not None:
            return self._face_up_cache

        # Ensure suit images are loaded
        Card.load_suit_images()
        # Draw surface for the card face
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(surface, self.BACKGROUND_COLOR, (0, 0, self.width, self.height), border_radius=10)
        suit_color = self.SUIT_COLORS[self.suit]
        font = pygame.font.SysFont('Arial', int(self.height * 0.18), bold=True)
        rank_text = font.render(self.rank, True, suit_color)
        rank_pos = (10, 10)
        surface.blit(rank_text, rank_pos)

        suit_img = self.CARD_SUIT_IMAGES.get(self.suit)
        if suit_img:
            center_h = int(self.height * 0.22)
            center_w = int(suit_img.get_width() * center_h / suit_img.get_height())
            suit_img_center = pygame.transform.smoothscale(suit_img, (center_w, center_h))
            center_pos = (self.width//2 - center_w//2, self.height//2 - center_h//2)
            surface.blit(suit_img_center, center_pos)
        else:
            center_suit_size = int(self.height * 0.22)
            center_suit_font = pygame.font.SysFont('Arial', center_suit_size)
            center_suit = center_suit_font.render(self.SUIT_SYMBOLS[self.suit], True, suit_color)
            center_pos = (self.width//2 - center_suit.get_width()//2, self.height//2 - center_suit.get_height()//2)
            surface.blit(center_suit, center_pos)
        self._face_up_cache = surface
        return surface

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