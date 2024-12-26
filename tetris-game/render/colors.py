# File: render/colors.py

from typing import Dict, Tuple

# Type aliases for clarity
Color = Tuple[int, int, int]
ColorAlpha = Tuple[int, int, int, int]

# Basic color definitions
BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)
GRAY: Color = (40, 40, 40)
DARK_GRAY: Color = (50, 50, 50)
RED: Color = (255, 0, 0)
GREEN: Color = (0, 255, 0)
BLUE: Color = (0, 0, 255)
YELLOW: Color = (255, 255, 0)
PURPLE: Color = (128, 0, 128)
ORANGE: Color = (255, 165, 0)
CYAN: Color = (0, 255, 255)
MAGENTA: Color = (255, 0, 255)


class Colors:
    """
    Class to hold color mappings for Tetris pieces and UI elements.
    """

    # Mapping of Tetrimino types to their colors
    PIECE_COLORS: Dict[str, Color] = {
        'I': CYAN,
        'O': YELLOW,
        'T': PURPLE,
        'S': GREEN,
        'Z': RED,
        'J': BLUE,
        'L': ORANGE,
    }

    # Mapping of UI elements to their colors
    UI_COLORS: Dict[str, Color] = {
        'background': BLACK,
        'text': WHITE,
        'highlight': YELLOW,
        'grid_lines': GRAY,
        'shadow': DARK_GRAY,
    }

    def get_piece_color(self, piece_type: str) -> Color:
        """
        Get the color associated with a Tetrimino type.
        
        :param piece_type: The type of the Tetrimino (e.g., 'I', 'O', 'T', ...)
        :return: A tuple representing the RGB color
        """
        return self.PIECE_COLORS.get(piece_type, WHITE)  # Default to WHITE if type not found
