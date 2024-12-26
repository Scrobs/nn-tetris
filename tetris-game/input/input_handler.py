# File: input/input_handler.py

import pygame
from typing import Callable, Dict, Any
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']


class InputHandler:
    """Handles user input and maps it to game actions through callbacks."""
    
    def __init__(self, config: Any, callbacks: Dict[str, Callable]):
        """
        Initialize the InputHandler with configuration and callback functions.
        
        :param config: Game configuration settings
        :param callbacks: A dictionary mapping action names to callback functions
        """
        self.config = config
        self.callbacks = callbacks

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle individual Pygame events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.callbacks.get('move_left', lambda: None)()
            elif event.key == pygame.K_RIGHT:
                self.callbacks.get('move_right', lambda: None)()
            elif event.key == pygame.K_DOWN:
                self.callbacks.get('soft_drop', lambda: None)()
            elif event.key == pygame.K_UP:
                self.callbacks.get('rotate_cw', lambda: None)()
            elif event.key == pygame.K_z:
                self.callbacks.get('rotate_ccw', lambda: None)()
            elif event.key == pygame.K_SPACE:
                self.callbacks.get('hard_drop', lambda: None)()
            elif event.key == pygame.K_c:
                self.callbacks.get('hold', lambda: None)()
            elif event.key == pygame.K_p:
                self.callbacks.get('toggle_pause', lambda: None)()
            elif event.key == pygame.K_r:
                self.callbacks.get('restart', lambda: None)()
            elif event.key == pygame.K_q:
                self.callbacks.get('quit', lambda: None)()


    def move_left(self, piece: Any, grid: Any) -> None:
        """Move the piece left if possible."""
        if piece and grid.can_move(piece, piece.x - 1, piece.y):
            piece.x -= 1
            # Play move sound
            # grid.update_position(piece)
            game_logger.debug("Moved piece left to x=%d", piece.x)

    def move_right(self, piece: Any, grid: Any) -> None:
        """Move the piece right if possible."""
        if piece and grid.can_move(piece, piece.x + 1, piece.y):
            piece.x += 1
            # Play move sound
            # grid.update_position(piece)
            game_logger.debug("Moved piece right to x=%d", piece.x)

    def soft_drop(self, piece: Any, grid: Any) -> None:
        """Soft drop the piece if possible."""
        if piece and grid.can_move(piece, piece.x, piece.y + 1):
            piece.y += 1
            # Play soft drop sound
            game_logger.debug("Soft dropped piece to y=%d", piece.y)

    def rotate_cw(self, piece: Any, grid: Any) -> None:
        """Rotate the piece clockwise."""
        if piece:
            rotated = piece.try_rotation(grid, clockwise=True)
            if rotated:
                # Play rotate sound
                game_logger.debug("Rotated piece clockwise")
            else:
                game_logger.debug("Rotation clockwise failed")

    def rotate_ccw(self, piece: Any, grid: Any) -> None:
        """Rotate the piece counterclockwise."""
        if piece:
            rotated = piece.try_rotation(grid, clockwise=False)
            if rotated:
                # Play rotate sound
                game_logger.debug("Rotated piece counterclockwise")
            else:
                game_logger.debug("Rotation counterclockwise failed")

    def hard_drop(self, piece: Any, grid: Any) -> None:
        """Hard drop the piece to the lowest possible position."""
        if piece:
            while grid.can_move(piece, piece.x, piece.y + 1):
                piece.y += 1
            # Play hard drop sound
            game_logger.debug("Hard dropped piece to y=%d", piece.y)

    def hold(self, piece: Any, grid: Any) -> None:
        """Hold the current piece."""
        # Implement hold logic
        game_logger.debug("Hold functionality triggered")
