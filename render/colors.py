# File: render/colors.py

from typing import Dict, Tuple

Color = Tuple[int, int, int]
ColorAlpha = Tuple[int, int, int, int]

# Define basic colors
BLACK: Color = (18, 18, 18)
WHITE: Color = (255, 255, 255)
GRAY: Color = (50, 50, 50)
DARK_GRAY: Color = (60, 60, 60)
RED: Color = (255, 85, 85)
GREEN: Color = (85, 255, 85)
BLUE: Color = (85, 85, 255)
YELLOW: Color = (255, 255, 85)
PURPLE: Color = (255, 85, 255)
ORANGE: Color = (255, 170, 85)
CYAN: Color = (85, 255, 255)
MAGENTA: Color = (255, 85, 255)
FLASH_COLOR: Color = (255, 255, 255)
NORMAL_COLOR: Color = (0, 0, 0)

class Colors:
    """
    Class to hold color mappings for Tetris pieces and UI elements.
    """
    PIECE_COLORS: Dict[str, Color] = {
        'I': CYAN,
        'O': YELLOW,
        'T': PURPLE,
        'S': GREEN,
        'Z': RED,
        'J': BLUE,
        'L': ORANGE,
    }
    UI_COLORS: Dict[str, Color] = {
        'background': BLACK,
        'text': WHITE,
        'highlight': YELLOW,
        'grid_lines': GRAY,
        'shadow': DARK_GRAY,
    }
    FLASH_COLOR: Color = FLASH_COLOR
    NORMAL_COLOR: Color = NORMAL_COLOR

    def get_piece_color(self, piece_type: str) -> Color:
        """
        Get the color associated with a Tetrimino type.
        
        :param piece_type: The type of the Tetrimino (e.g., 'I', 'O', 'T', ...)
        :return: A tuple representing the RGB color
        """
        return self.PIECE_COLORS.get(piece_type, WHITE)
