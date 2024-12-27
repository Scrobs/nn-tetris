# =============================
# File: render/render_config.py
# =============================

from typing import Dict, Optional
import pygame
from config.game_config import GameConfig
from render.colors import Colors
from utils.logging_setup import setup_logging

loggers = setup_logging()
resource_logger = loggers['resource']

class RenderConfig:
    """
    Configuration for rendering-specific parameters, with improvements:
    - Optionally override font sizes via config or constants
    - Provide more robust error handling/logging
    - Encourage use of dynamic layout
    """

    def __init__(self, game_config: GameConfig):
        """Initialize render configuration with logging."""
        self.colors = Colors()
        self.grid_size = game_config.grid_size
        self.cell_padding = game_config.cell_padding
        self.grid_line_width = game_config.grid_line_width
        self.preview_margin = game_config.preview_margin
        self.preview_pieces = game_config.preview_pieces
        # Increase if UI is still too cramped (e.g., 350, 400)
        self.ui_width = game_config.ui_width  
        self.screen_width = game_config.screen_width
        self.screen_height = game_config.screen_height
        self.debug_mode = game_config.debug_mode
        self.particle_effects = game_config.particle_effects
        
        # Further reduce font sizes to help avoid overlap
        self.font_small_size = 12
        self.font_medium_size = 18
        self.font_large_size = 26

        self.fonts: Dict[str, Optional[pygame.font.Font]] = self._initialize_fonts()
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = self._initialize_sounds()

    def _initialize_fonts(self) -> Dict[str, Optional[pygame.font.Font]]:
        """
        Initialize fonts with custom font files and fallback logic.
        Improved to allow dynamic font sizes if desired.
        """
        fonts = {}
        try:
            # Adjust paths & sizes if needed
            fonts['small'] = pygame.font.Font(
                'resources/fonts/PressStart2P-Regular.ttf', self.font_small_size
            )
            fonts['medium'] = pygame.font.Font(
                'resources/fonts/PressStart2P-Regular.ttf', self.font_medium_size
            )
            fonts['large'] = pygame.font.Font(
                'resources/fonts/PressStart2P-Regular.ttf', self.font_large_size
            )
            resource_logger.debug("Fonts initialized successfully with PressStart2P-Regular.ttf")
        except FileNotFoundError as e:
            resource_logger.error("Font file not found: %s", str(e))
            # Use fallback fonts with approximate sizes
            fonts['small'] = pygame.font.SysFont(None, self.font_small_size)
            fonts['medium'] = pygame.font.SysFont(None, self.font_medium_size)
            fonts['large'] = pygame.font.SysFont(None, self.font_large_size)
        except Exception as e:
            resource_logger.error("Error initializing fonts: %s", str(e))
            fonts = {key: None for key in ['small', 'medium', 'large']}
        return fonts

    def _initialize_sounds(self) -> Dict[str, Optional[pygame.mixer.Sound]]:
        """Load sound effects with error handling."""
        sounds = {}
        try:
            sounds['move'] = pygame.mixer.Sound('resources/sounds/move.wav')
            sounds['rotate'] = pygame.mixer.Sound('resources/sounds/rotate.wav')
            sounds['clear'] = pygame.mixer.Sound('resources/sounds/clear.wav')
            sounds['hold'] = pygame.mixer.Sound('resources/sounds/hold.wav')
            sounds['drop'] = pygame.mixer.Sound('resources/sounds/drop.wav')
            sounds['game_over'] = pygame.mixer.Sound('resources/sounds/game_over.wav')
            sounds['lock'] = pygame.mixer.Sound('resources/sounds/lock.wav')
            resource_logger.debug("Sound effects loaded successfully")
        except pygame.error as e:
            resource_logger.error("Failed to load sound effects: %s", str(e))
            # Provide Nones so code won't crash if resources fail
            sounds = {
                'move': None,
                'rotate': None,
                'clear': None,
                'hold': None,
                'drop': None,
                'game_over': None,
                'lock': None
            }
        return sounds
