# File: core/game_state.py

from enum import Enum, auto

class GameState(Enum):
    """Enumeration of possible game states with logging."""
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()
    ANIMATING = auto()

    def __str__(self) -> str:
        return self.name
