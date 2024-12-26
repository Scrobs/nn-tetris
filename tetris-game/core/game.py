# File: core/game.py

import os
import pygame
import sys
import random
import time
from typing import List, Optional, Callable, Dict
from config.game_config import GameConfig
from core.game_state import GameState
from core.grid import Grid
from core.score import Score
from core.tetrimino import Tetrimino, TetriminoData
from render.render_config import RenderConfig
from render.render_system import RenderSystem
from audio.sound_manager import SoundManager
from input.input_handler import InputHandler
from utils.performance_monitor import PerformanceMonitor
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class TetrisGame:
    """
    Main game class handling game logic, state, and interactions.
    Manages the core game loop, piece movement, scoring, and state transitions.
    """
    def __init__(self, config: GameConfig):
        """Initialize the TetrisGame with configuration and subsystems."""
        self.config = config
        game_logger.info("Initializing Tetris game")
        
        # Initialize pygame display first
        pygame.init()
        pygame.display.init()
        
        # Set up the display
        flags = pygame.DOUBLEBUF
        if self.config.fullscreen:
            flags |= pygame.FULLSCREEN
        if self.config.vsync:
            flags |= pygame.HWSURFACE | pygame.SCALED
            
        self.screen = pygame.display.set_mode(
            (self.config.screen_width + self.config.ui_width, 
            self.config.screen_height),
            flags
        )
        pygame.display.set_caption("Tetris")
        
        self._initialize_subsystems()
        self.reset()

    def _initialize_subsystems(self) -> None:
        """Initialize all game subsystems including rendering, input, sound, etc."""
        try:
            # Initialize rendering subsystems
            self.render_config = RenderConfig(self.config)
            self.renderer = RenderSystem(self.config, self.render_config)
            self.sound_manager = SoundManager(self.render_config)

            # Define input callbacks
            callbacks = {
                'move_left': self.move_left,
                'move_right': self.move_right,
                'soft_drop': self.soft_drop,
                'rotate_cw': self.rotate_cw,
                'rotate_ccw': self.rotate_ccw,
                'hard_drop': self.hard_drop,
                'hold': self.hold,
                'toggle_pause': self.toggle_pause,
                'restart': self.reset,
                'quit': self.quit_game
            }

            # Initialize remaining subsystems
            self.input_handler = InputHandler(self.config, callbacks)
            self.performance_monitor = PerformanceMonitor()
            self.score_handler = Score(max_score=self.config.max_score)
            self.grid = Grid(self.config)

            # Initialize game state variables
            self.state = GameState.MENU
            self.frame_counter = 0
            self.level = 1
            self.lines_cleared = 0
            self.combo_counter = 0
            self.held_piece: Optional[Tetrimino] = None
            self.bag: List[str] = []
            self.next_pieces: List[Tetrimino] = []
            self.current_piece: Optional[Tetrimino] = None
            self.running = False
            self.last_move_time = time.time()
            self.lock_delay_start = None

            game_logger.info("All subsystems initialized successfully")
        except Exception as e:
            game_logger.error("Failed to initialize game: %s", str(e))
            raise

    def reset(self) -> None:
        """Reset the game state to start a new game."""
        try:
            self.grid.reset()
            self.score_handler.reset()
            self.level = 1
            self.lines_cleared = 0
            self.combo_counter = 0
            self.held_piece = None
            self.bag = []
            self.next_pieces = [self._get_next_piece() for _ in range(self.config.preview_pieces)]
            self.current_piece = self._get_next_piece()
            self.state = GameState.PLAYING
            self.lock_delay_start = None
            game_logger.info("Game state reset completed")
        except Exception as e:
            game_logger.error("Failed to reset game state: %s", str(e))
            raise

    def move_left(self) -> None:
        """Move the current piece left if possible."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        if self.grid.can_move(self.current_piece, self.current_piece.x - 1, self.current_piece.y):
            self.current_piece.x -= 1
            self.sound_manager.play_sound('move')
            self.lock_delay_start = time.time()  # Reset lock delay on movement
            game_logger.debug("Moved piece left to x=%d", self.current_piece.x)

    def move_right(self) -> None:
        """Move the current piece right if possible."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        if self.grid.can_move(self.current_piece, self.current_piece.x + 1, self.current_piece.y):
            self.current_piece.x += 1
            self.sound_manager.play_sound('move')
            self.lock_delay_start = time.time()  # Reset lock delay on movement
            game_logger.debug("Moved piece right to x=%d", self.current_piece.x)

    def soft_drop(self) -> None:
        """Soft drop the current piece if possible."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        if self.grid.can_move(self.current_piece, self.current_piece.x, self.current_piece.y + 1):
            self.current_piece.y += 1
            self.score_handler.add(1)  # Small score bonus for soft drop
            self.sound_manager.play_sound('move')
            self.lock_delay_start = time.time()  # Reset lock delay on movement
            game_logger.debug("Soft dropped piece to y=%d", self.current_piece.y)

    def rotate_cw(self) -> None:
        """Rotate the current piece clockwise."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        if self.current_piece.try_rotation(self.grid.grid, clockwise=True):
            self.sound_manager.play_sound('rotate')
            self.lock_delay_start = time.time()  # Reset lock delay on rotation
            game_logger.debug("Rotated piece clockwise")

    def rotate_ccw(self) -> None:
        """Rotate the current piece counterclockwise."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        if self.current_piece.try_rotation(self.grid.grid, clockwise=False):
            self.sound_manager.play_sound('rotate')
            self.lock_delay_start = time.time()  # Reset lock delay on rotation
            game_logger.debug("Rotated piece counterclockwise")

    def hard_drop(self) -> None:
        """Hard drop the current piece to the lowest possible position."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        initial_y = self.current_piece.y
        while self.grid.can_move(self.current_piece, self.current_piece.x, self.current_piece.y + 1):
            self.current_piece.y += 1

        drop_distance = self.current_piece.y - initial_y
        self.score_handler.add(drop_distance * 2)  # Score bonus for hard drop
        self._lock_piece()
        self.sound_manager.play_sound('drop')
        game_logger.debug("Hard dropped piece to y=%d", self.current_piece.y)

    def hold(self) -> None:
        """Hold the current piece and swap with previously held piece."""
        if self.state != GameState.PLAYING or not self.current_piece:
            return

        if self.held_piece:
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            # Reset position for the newly active piece
            self.current_piece.x = (self.config.columns - len(self.current_piece.shape[0])) // 2
            self.current_piece.y = 0
        else:
            self.held_piece = self.current_piece
            self._spawn_new_piece()

        self.sound_manager.play_sound('hold')
        self.lock_delay_start = None
        game_logger.debug("Held piece swapped")

    def _fill_bag(self) -> None:
        """Fill the bag with all piece types if it's empty."""
        if not self.bag:
            self.bag = list(TetriminoData.SHAPES.keys())
            random.shuffle(self.bag)
            game_logger.debug("Filled piece bag: %s", self.bag)

    def _get_next_piece(self) -> Tetrimino:
        """
        Retrieve the next piece from the bag, refilling if necessary.
        
        Returns:
            Tetrimino: A new piece instance
        """
        self._fill_bag()
        piece_type = self.bag.pop()
        game_logger.debug("Got next piece: %s", piece_type)
        return Tetrimino(piece_type, self.config.grid_size, self.config.columns)

    def run(self) -> None:
        """Main game loop with proper error handling and cleanup."""
        try:
            self.running = True
            clock = pygame.time.Clock()
            selected_menu_index = 0
            menu_options = ["Start Game", "Settings", "High Scores", "Quit"]

            while self.running:
                try:
                    self.performance_monitor.start_frame()

                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        else:
                            self.input_handler.handle_event(event)

                    # Update game state based on current state
                    if self.state == GameState.MENU:
                        selected_menu_index = self._handle_menu_input(menu_options, selected_menu_index)
                        self._render_menu(menu_options, selected_menu_index)
                    elif self.state == GameState.PLAYING:
                        self._handle_input()
                        self._update_game_state()
                        self._render_frame()
                    elif self.state == GameState.PAUSED:
                        self.renderer.draw_pause_screen()
                        screen = pygame.display.get_surface()
                        if screen:
                            self.renderer.compose_frame(screen, self.frame_counter)
                    elif self.state == GameState.GAME_OVER:
                        self.renderer.draw_game_over(self.score_handler.value_display)
                        screen = pygame.display.get_surface()
                        if screen:
                            self.renderer.compose_frame(screen, self.frame_counter)

                    # Maintain frame rate
                    delta_time = clock.tick(self.config.fps) / 1000.0
                    self.performance_monitor.end_frame()
                    self.performance_monitor.log_performance()

                except Exception as e:
                    game_logger.error("Error in game loop: %s", str(e), exc_info=True)
                    if self.state != GameState.GAME_OVER:
                        game_logger.info("Setting game state to GAME_OVER due to error")
                        self.state = GameState.GAME_OVER

        except Exception as e:
            game_logger.critical("Fatal error in game loop: %s", str(e), exc_info=True)
            raise
        finally:
            self.cleanup()

    def _handle_menu_input(self, menu_options: List[str], selected_index: int) -> int:
        """
        Handle user input in the menu.
        
        Args:
            menu_options: List of menu option strings
            selected_index: Currently selected menu item index
            
        Returns:
            int: Updated selected index
        """
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_DOWN]:
            selected_index = (selected_index + 1) % len(menu_options)
            time.sleep(0.2)  # Prevent too rapid selection changes
        
        if keys[pygame.K_UP]:
            selected_index = (selected_index - 1) % len(menu_options)
            time.sleep(0.2)
        
        if keys[pygame.K_RETURN]:
            selected_option = menu_options[selected_index]
            game_logger.info("Selected menu option: %s", selected_option)
            
            if selected_option == "Start Game":
                self.reset()
            elif selected_option == "Quit":
                self.running = False
                
        return selected_index

    def _handle_input(self) -> None:
        """Handle continuous input like holding keys."""
        if self.state != GameState.PLAYING:
            return
            
        keys = pygame.key.get_pressed()
        current_time = time.time()
        
        # Handle continuous movement with DAS (Delayed Auto Shift)
        if current_time - self.last_move_time >= self.config.das_delay / 1000.0:
            if keys[pygame.K_LEFT]:
                self.move_left()
            if keys[pygame.K_RIGHT]:
                self.move_right()
            if keys[pygame.K_DOWN]:
                self.soft_drop()
            self.last_move_time = current_time

    def _update_game_state(self) -> None:
        """Update game state with proper synchronization and logging."""
        try:
            if self.state != GameState.PLAYING:
                return

            current_time = time.time()

            # Handle natural piece falling
            if not self._apply_gravity():
                # Piece can't move down, start/continue lock delay
                if self.lock_delay_start is None:
                    self.lock_delay_start = current_time
                elif current_time - self.lock_delay_start >= self.config.lock_delay / 1000.0:
                    self._lock_piece()
                    cleared_lines = self._clear_lines()
                    reward = self._calculate_reward(cleared_lines)
                    self.score_handler.add(reward)
                    
                    if not self._spawn_new_piece():
                        game_logger.info("Game over - cannot spawn new piece")
                        self.state = GameState.GAME_OVER
            else:
                # Piece moved down successfully, reset lock delay
                self.lock_delay_start = None

        except Exception as e:
            game_logger.error("Error updating game state: %s", str(e), exc_info=True)
            raise

    def _apply_gravity(self) -> bool:
        """
        Apply gravity to the current piece.
        
        Returns:
            bool: True if the piece moved down, False if blocked
        """
        if self.current_piece and self.grid.can_move(
            self.current_piece,
            self.current_piece.x,
            self.current_piece.y + 1
        ):
            self.current_piece.y += 1
            game_logger.debug("Applied gravity to piece at y=%d", self.current_piece.y)
            return True
        return False

    def _lock_piece(self) -> None:
        """Lock the current piece into the grid."""
        if self.current_piece:
            self.grid.lock_piece(self.current_piece)
            self.sound_manager.play_sound('lock')
            game_logger.debug("Locked piece at position (%d, %d)",
                            self.current_piece.x, self.current_piece.y)

    def _clear_lines(self) -> int:
        """
        Clear completed lines and calculate score rewards.
        
        Handles line clearing mechanics including:
        - Detection of completed lines
        - Removal of completed lines
        - Score calculation based on number of lines cleared
        - Combo system management
        
        Returns:
            int: Number of lines cleared in this operation
        """
        try:
            lines_cleared = self.grid.clear_lines()
            if lines_cleared > 0:
                self.sound_manager.play_sound('clear')
                self.lines_cleared += lines_cleared
                
                # Level progression based on lines cleared
                self.level = (self.lines_cleared // 10) + 1
                
                game_logger.debug("Cleared %d lines, total: %d, level: %d",
                                lines_cleared, self.lines_cleared, self.level)
            return lines_cleared
        except Exception as e:
            game_logger.error("Error clearing lines: %s", str(e))
            return 0

    def _calculate_reward(self, lines_cleared: int) -> int:
        """
        Calculate score reward based on number of lines cleared and current level.
        
        Implements standard Tetris scoring system:
        - Single line: 100 * level
        - Double lines: 300 * level
        - Triple lines: 500 * level
        - Tetris (4 lines): 800 * level
        - Additional combo bonuses
        
        Args:
            lines_cleared: Number of lines cleared in current move
            
        Returns:
            int: Calculated score reward
        """
        try:
            base_rewards = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
            reward = base_rewards.get(lines_cleared, 0) * self.level
            
            # Combo system
            if lines_cleared > 1:
                self.combo_counter += 1
                reward += self.combo_counter * 50 * self.level
            else:
                self.combo_counter = 0
                
            game_logger.debug("Calculated reward: %d for %d lines (combo: %d)",
                            reward, lines_cleared, self.combo_counter)
            return reward
        except Exception as e:
            game_logger.error("Error calculating reward: %s", str(e))
            return 0

    def _spawn_new_piece(self) -> bool:
        """
        Spawn a new tetrimino piece and update the next piece queue.
        
        Implements piece spawning mechanics:
        - Takes next piece from preview queue
        - Adds new piece to preview queue
        - Checks for spawn collision (game over condition)
        - Updates piece position and orientation
        
        Returns:
            bool: True if piece spawned successfully, False if spawn blocked (game over)
        """
        try:
            self.current_piece = self.next_pieces.pop(0)
            new_piece = self._get_next_piece()
            self.next_pieces.append(new_piece)
            
            # Check if spawn position is valid
            if not self.grid.can_place(self.current_piece,
                                     self.current_piece.x,
                                     self.current_piece.y):
                game_logger.info("Game over: spawn position blocked")
                return False
                
            self.lock_delay_start = None
            game_logger.debug("Spawned new piece: %s", self.current_piece.piece_type)
            return True
        except Exception as e:
            game_logger.error("Error spawning new piece: %s", str(e))
            return False

    def _render_menu(self, menu_options: List[str], selected_index: int) -> None:
        """
        Render the main menu interface with proper highlighting and animation.
        
        Handles menu visualization including:
        - Menu option rendering
        - Selection highlighting
        - Background elements
        - Transition effects
        
        Args:
            menu_options: List of available menu options
            selected_index: Currently selected option index
        """
        try:
            screen = pygame.display.get_surface()
            if not screen:
                game_logger.error("No display surface available")
                return
                
            self.renderer.clear_surfaces()
            self.renderer.draw_menu(menu_options, selected_index)
            self.renderer.compose_frame(screen, self.frame_counter)
            game_logger.debug("Menu rendered successfully")
        except Exception as e:
            game_logger.error("Error rendering menu: %s", str(e))

    def _render_frame(self) -> None:
        """
        Render a complete game frame with all visual elements.
        
        Implements the rendering pipeline:
        - Clear previous frame
        - Draw game grid
        - Draw current piece and ghost piece
        - Draw UI elements (score, next pieces, held piece)
        - Draw particle effects
        - Compose final frame
        """
        try:
            self.renderer.clear_surfaces()
            game_logger.debug("Starting frame render %d", self.frame_counter)
            
            # Draw game elements
            self.renderer.draw_grid(self.grid)
            
            if self.current_piece:
                try:
                    ghost_y = self.current_piece.get_ghost_position(self.grid)
                    self.renderer.draw_piece(self.current_piece, ghost=True,
                                          override_y=ghost_y)
                except ValueError as e:
                    game_logger.error("Error getting ghost position: %s", str(e))
                
                self.renderer.draw_piece(self.current_piece)
            
            # Draw UI elements
            self.renderer.draw_ui(
                score=self.score_handler.value_display,
                high_score=self.score_handler.high_score_display,
                level=self.level,
                lines=self.lines_cleared,
                next_pieces=self.next_pieces,
                held_piece=self.held_piece,
                combo=self.combo_counter
            )
            
            # Draw debug information if enabled
            if self.config.debug_mode:
                fps = self.performance_monitor.get_fps()
                self.renderer.draw_debug_info(self.frame_counter, fps)
            
            # Compose and display final frame
            screen = pygame.display.get_surface()
            if screen:
                self.renderer.compose_frame(screen, self.frame_counter)
            else:
                game_logger.error("No display surface available")
            
            self.frame_counter += 1
            game_logger.debug("Frame %d rendered successfully", self.frame_counter)
        except Exception as e:
            game_logger.error("Error rendering frame: %s", str(e))

    def cleanup(self) -> None:
        """
        Perform cleanup operations before game exit.
        
        Handles:
        - Resource deallocation
        - High score saving
        - Pygame shutdown
        - Logging cleanup
        """
        try:
            # Save high score
            self.score_handler.save_high_score()
            
            # Cleanup pygame resources
            pygame.quit()
            game_logger.info("Game cleanup completed successfully")
        except Exception as e:
            game_logger.error("Error during cleanup: %s", str(e))

    def quit_game(self) -> None:
        """
        Initiate graceful game shutdown sequence.
        
        Handles:
        - State saving
        - Resource cleanup
        - Exit sequence
        """
        self.running = False
        game_logger.info("Initiating game shutdown")

    def toggle_pause(self) -> None:
        """
        Toggle game pause state.
        
        Implements pause functionality:
        - State transition management
        - Timer suspension
        - UI updates
        """
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
            game_logger.info("Game paused")
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
            game_logger.info("Game resumed")