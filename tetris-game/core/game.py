# core/game.py

"""
core/game.py - Part 1: Core Game Implementation
Implements the main Tetris game engine with fixed timestep physics and interpolated rendering.
This module handles:
- Game initialization and resource management
- Physics simulation with fixed timestep updates
- State management and game loop implementation
- Performance monitoring and optimization
Architecture Overview:
- Follows Component-based design for modularity
- Implements fixed timestep pattern for consistent physics
- Uses interpolated rendering for smooth animation
- Maintains clear separation between update and render cycles
"""

import os
import pygame
import sys
import random
import time
from typing import List, Optional, Callable, Dict, Tuple
from dataclasses import dataclass
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

@dataclass
class GameMetrics:
    """
    Container for comprehensive game performance metrics and state information.
    Attributes:
    frame_count: Total number of rendered frames
    physics_updates: Total number of physics simulation steps
    render_time: Time spent in rendering pipeline (ms)
    physics_time: Time spent in physics updates (ms)
    total_time: Total elapsed game time (s)
    """
    frame_count: int
    physics_updates: int
    render_time: float
    physics_time: float
    total_time: float

class TetrisGame:
    """
    Main game class handling game logic, state, and interactions.
    Implements fixed timestep physics with interpolated rendering.
    Key Features:
    - Fixed timestep physics simulation (60 Hz)
    - Interpolated rendering for smooth animation
    - Comprehensive state management
    - Performance-optimized update cycles
    """
    def __init__(self, config: GameConfig):
        """
        Initialize the TetrisGame with configuration and subsystems.
        Args:
        config: GameConfig instance containing game parameters
        Raises:
        pygame.error: If display initialization fails
        Exception: If subsystem initialization fails
        Implementation Details:
        1. Core Configuration
           - Game parameters
           - Fixed timestep physics setup
           - Display configuration
        2. Initialization Sequence
           - Physics timing parameters
           - Pygame setup
           - Subsystem initialization
           - Game state setup
        """
        try:
            self.config = config
            self.fixed_timestep = config.fixed_timestep
            self.max_frame_time = config.max_frame_time
            self.physics_accumulator = 0.0
            self.physics_alpha = 0.0
            self.gravity_timer = 0.0  # Initialize gravity timer
            self.start_time = time.perf_counter()
            game_logger.info("Initializing Tetris game")
            self._initialize_pygame()
            self._initialize_subsystems()
            self._initialize_game_state()
            self.reset()
            game_logger.debug(
                "Game initialized with timestep=%.3fs, max_frame=%.3fs",
                self.fixed_timestep,
                self.max_frame_time
            )
        except Exception as e:
            game_logger.critical(
                "Fatal error during game initialization: %s", str(e)
            )
            raise

    def _initialize_pygame(self) -> None:
        """
        Initialize Pygame with proper display configuration.
        Configures:
        - Display mode and resolution
        - Double buffering
        - VSync (if enabled)
        - Fullscreen mode (if enabled)
        Raises:
        pygame.error: If display initialization fails
        """
        pygame.init()
        pygame.display.init()
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
        game_logger.debug("Pygame initialized with display mode: %dx%d",
                          self.config.screen_width + self.config.ui_width,
                          self.config.screen_height)

    def _initialize_game_controls(self) -> None:
        """
        Initialize all game control methods and movement handlers.
        This ensures all necessary control methods exist before input handler setup.
        """
        self.move_left = lambda: self._move_piece(-1, 0)
        self.move_right = lambda: self._move_piece(1, 0)
        self.soft_drop = lambda: self._move_piece(0, 1)
        self.rotate_cw = lambda: self._rotate_piece(True)
        self.rotate_ccw = lambda: self._rotate_piece(False)
        self.hard_drop = self._hard_drop
        self.hold = self._hold_piece
        game_logger.debug("Game controls initialized")

    def _initialize_subsystems(self) -> None:
        """Initialize all game subsystems with comprehensive error handling."""
        initialized_systems = []
        try:
            self._initialize_game_controls()
            initialized_systems.append('game_controls')

            self.render_config = RenderConfig(self.config)
            initialized_systems.append('render_config')

            self.renderer = RenderSystem(self.config, self.render_config)
            initialized_systems.append('renderer')

            self.sound_manager = SoundManager(self.render_config)
            initialized_systems.append('sound_manager')

            self.performance_monitor = PerformanceMonitor(
                fixed_timestep=self.fixed_timestep,
                vsync_enabled=self.config.vsync
            )
            initialized_systems.append('performance_monitor')

            self.score_handler = Score(max_score=self.config.max_score)
            initialized_systems.append('score_handler')

            self.grid = Grid(self.config)
            initialized_systems.append('grid')

            self._setup_input_handler()
            initialized_systems.append('input_handler')

            game_logger.info("All subsystems initialized successfully")
        except Exception as e:
            game_logger.error(
                "Failed to initialize game subsystems at stage '%s': %s",
                initialized_systems[-1] if initialized_systems else 'unknown',
                str(e)
            )
            self._cleanup_partial_initialization(initialized_systems)
            raise

    def _setup_input_handler(self) -> None:
        """
        Configure input handling system with callback mappings.
        Maps game actions to their corresponding handler methods.
        Configures:
        - Movement controls
        - Rotation controls
        - Game state controls
        - Debug controls
        """
        callbacks = {
            'MOVE_LEFT': self.move_left,
            'MOVE_RIGHT': self.move_right,
            'SOFT_DROP': self.soft_drop,
            'ROTATE_CW': self.rotate_cw,
            'ROTATE_CCW': self.rotate_ccw,
            'HARD_DROP': self.hard_drop,
            'HOLD': self.hold,
            'PAUSE': self.toggle_pause,
            'RESTART': self.reset,
            'QUIT': self.quit_game,
            'TOGGLE_DEBUG': self.toggle_debug,
            'SET_LOG_LEVEL': self.set_log_level
        }
        self.input_handler = InputHandler("input/input_config.json", callbacks)
        game_logger.debug("Input handler configured with %d callbacks",
                          len(callbacks))

    def _initialize_game_state(self) -> None:
        """
        Initialize core game state variables and counters.
        Initializes:
        - Game state enum
        - Performance counters
        - Level and scoring state
        - Piece management state
        - Timing variables
        """
        self.state = GameState.MENU
        self.frame_counter = 0
        self.physics_updates = 0
        self.level = self.config.initial_level
        self.lines_cleared = 0
        self.combo_counter = 0
        self.held_piece: Optional[Tetrimino] = None
        self.bag: List[str] = []
        self.next_pieces: List[Tetrimino] = []
        self.current_piece: Optional[Tetrimino] = None
        self.running = False
        self.last_move_time = time.time()
        self.lock_delay_start: Optional[float] = None
        self.lock_delay_accumulated = 0.0
        self.selected_menu_index = 0
        game_logger.debug("Game state initialized")

    def reset(self) -> None:
        """
        Reset the game state to start a new game.
        Resets:
        - Grid state
        - Score system
        - Level progression
        - Piece management
        - Timing variables
        Raises:
        Exception: If state reset fails
        """
        try:
            self.grid.reset()
            self.score_handler.reset()
            self.level = self.config.initial_level
            self.lines_cleared = 0
            self.combo_counter = 0
            self.held_piece = None
            self.bag = []
            self.next_pieces = [
                self._get_next_piece()
                for _ in range(self.config.preview_pieces)
            ]
            self.current_piece = self._get_next_piece()
            self.state = GameState.PLAYING
            self.lock_delay_start = None
            self.lock_delay_accumulated = 0.0
            self.physics_accumulator = 0.0
            self.gravity_timer = 0.0  # Reset gravity timer
            game_logger.info("Game state reset completed")
        except Exception as e:
            game_logger.error("Failed to reset game state: %s", str(e))
            raise

    def _move_piece(self, dx: int, dy: int) -> None:
        """
        Move the current piece by the specified delta.
        Args:
        dx: Horizontal movement (-1 for left, 1 for right)
        dy: Vertical movement (1 for down)
        """
        if self.current_piece and self.state == GameState.PLAYING:
            if self.grid.can_move(
                self.current_piece,
                self.current_piece.x + dx,
                self.current_piece.y + dy
            ):
                self.current_piece.x += dx
                self.current_piece.y += dy
                self._reset_lock_delay()
                if dy > 0:
                    self.score_handler.add(self.config.scoring_values['soft_drop'] * dy)
                elif dx != 0:
                    self.sound_manager.play_sound('move')
                game_logger.debug(
                    "Piece moved to (%d, %d)",
                    self.current_piece.x,
                    self.current_piece.y
                )

    def _rotate_piece(self, clockwise: bool) -> None:
        """
        Rotate the current piece with wall kick validation.
        Args:
        clockwise: Direction of rotation
        """
        if self.current_piece and self.state == GameState.PLAYING:
            if self.current_piece.try_rotation(self.grid, clockwise):
                self._reset_lock_delay()
                self.sound_manager.play_sound('rotate')
                game_logger.debug(
                    "Piece rotated %s to rotation %d",
                    "clockwise" if clockwise else "counter-clockwise",
                    self.current_piece.rotation
                )

    def _hard_drop(self) -> None:
        """Instantly drop the current piece to its final position."""
        if self.current_piece and self.state == GameState.PLAYING:
            drop_distance = 0
            while self.grid.can_move(
                self.current_piece,
                self.current_piece.x,
                self.current_piece.y + 1
            ):
                self.current_piece.y += 1
                drop_distance += 1
            self.score_handler.add(
                drop_distance * self.config.scoring_values['hard_drop']
            )
            self._lock_piece()
            self.sound_manager.play_sound('drop')
            game_logger.debug(
                "Piece hard dropped %d spaces",
                drop_distance
            )

    def _hold_piece(self) -> None:
        """Hold the current piece or swap with held piece."""
        if not self.current_piece or self.state != GameState.PLAYING:
            return
        if self.held_piece is None:
            self.held_piece = self.current_piece
            self._reset_piece_position(self.held_piece)
            self._spawn_new_piece()
        else:
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            self._reset_piece_position(self.current_piece)
        self.sound_manager.play_sound('hold')
        game_logger.debug("Piece held")

    def run(self) -> None:
        """
        Main game loop implementing fixed timestep physics and interpolated rendering.
        Features:
        - Fixed timestep physics updates (60 Hz)
        - Frame time clamping to prevent spiral of death
        - Performance monitoring and logging
        - Proper resource cleanup
        Raises:
        Exception: If fatal error occurs in game loop
        """
        try:
            self.running = True
            last_time = time.perf_counter()
            while self.running:
                current_time = time.perf_counter()
                frame_time = min(current_time - last_time, self.max_frame_time)
                last_time = current_time
                self.performance_monitor.start_frame()
                self._handle_events()
                self.physics_accumulator += frame_time
                self.gravity_timer += frame_time  # Increment gravity timer

                # Apply gravity based on gravity_delay
                while self.gravity_timer >= self.config.gravity_delay:
                    if not self._apply_gravity():
                        if self.lock_delay_start is None:
                            self.lock_delay_start = time.perf_counter()
                            self.lock_delay_accumulated = 0.0
                        else:
                            self.lock_delay_accumulated += self.config.fixed_timestep
                            if self.lock_delay_accumulated >= self.config.lock_delay / 1000.0:
                                self._lock_piece()
                                self._clear_lines()
                                if not self._spawn_new_piece():
                                    game_logger.info("Game over - cannot spawn new piece")
                                    self.state = GameState.GAME_OVER
                                    self.lock_delay_start = None
                                    self.lock_delay_accumulated = 0.0
                                else:
                                    self.lock_delay_start = None
                                    self.lock_delay_accumulated = 0.0
                    self.gravity_timer -= self.config.gravity_delay

                # Fixed timestep physics updates
                while self.physics_accumulator >= self.fixed_timestep:
                    self._update_physics(self.fixed_timestep)
                    self.physics_accumulator -= self.fixed_timestep
                    self.physics_updates += 1

                self.physics_alpha = self.physics_accumulator / self.fixed_timestep
                self._render_frame()
                self.performance_monitor.end_frame()
                self.performance_monitor.log_performance()
        except Exception as e:
            game_logger.critical("Fatal error in game loop: %s", str(e), exc_info=True)
            raise
        finally:
            self.cleanup()

    def _render_frame(self) -> None:
        """
        Render a complete frame with interpolation for smooth animation.
        This method handles:
        - Clearing render surfaces
        - Drawing the game grid
        - Rendering active piece and ghost piece
        - Drawing UI elements
        - Composing the final frame
        - Debug information (if enabled)
        Implementation follows the established rendering pipeline with proper
        state handling and performance monitoring.
        """
        try:
            self.renderer.clear_surfaces()
            if self.state == GameState.MENU:
                menu_options = ["Start Game", "Options", "High Scores", "Quit"]
                self.renderer.draw_menu(menu_options, self.selected_menu_index)
            else:
                self.renderer.draw_grid(self.grid)
                if self.current_piece and self.state == GameState.PLAYING:
                    ghost_y = self.current_piece.get_ghost_position(self.grid.grid)
                    self.renderer.draw_piece(self.current_piece, ghost=True, override_y=ghost_y)
                    self.renderer.draw_piece(self.current_piece)
                self.renderer.update_particles(self.fixed_timestep)
                self.renderer.draw_particles()
                self.renderer.draw_ui(
                    score=self.score_handler.value,
                    high_score=self.score_handler.high_score,
                    level=self.level,
                    lines=self.lines_cleared,
                    next_pieces=self.next_pieces,
                    held_piece=self.held_piece,
                    combo=self.combo_counter
                )
                if self.state == GameState.PAUSED:
                    self.renderer.draw_pause_screen()
                elif self.state == GameState.GAME_OVER:
                    self.renderer.draw_game_over(self.score_handler.value)
            if self.config.debug_mode:
                metrics = self.performance_monitor.get_metrics()
                self.renderer.draw_debug_info(
                    frame_counter=self.frame_counter,
                    fps=metrics.fps
                )
            self.renderer.compose_frame(
                self.screen,
                self.frame_counter
            )
            self.frame_counter += 1
        except Exception as e:
            game_logger.error("Error rendering frame: %s", str(e), exc_info=True)
            raise

    def _handle_events(self) -> None:
        """
        Process all pending input events.
        Handles:
        - Window events (quit)
        - Keyboard input
        - Game state transitions
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.state == GameState.MENU:
                self._handle_menu_input(event)
            else:
                self.input_handler.handle_event(event)

    def _handle_menu_input(self, event: pygame.event.Event) -> None:
        """
        Process input events for menu navigation.
        Args:
        event: Pygame event to process
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_menu_index = (self.selected_menu_index - 1) % 4
            elif event.key == pygame.K_DOWN:
                self.selected_menu_index = (self.selected_menu_index + 1) % 4
            elif event.key == pygame.K_RETURN:
                self._handle_menu_selection()

    def _handle_menu_selection(self) -> None:
        """Handle menu option selection and trigger appropriate actions."""
        if self.selected_menu_index == 0:
            self.reset()
            self.state = GameState.PLAYING
        elif self.selected_menu_index == 3:
            self.running = False

    def _update_physics(self, delta_time: float) -> None:
        """
        Update game physics with fixed timestep.
        Implements:
        - Piece gravity based on gravity_delay
        - Lock delay system
        - Line clearing
        - Game over detection
        Args:
        delta_time: Fixed physics timestep duration
        """
        if self.state != GameState.PLAYING:
            return
        try:
            # Since gravity is now handled separately, physics updates can handle other mechanics
            pass  # Placeholder for any additional physics updates
        except Exception as e:
            game_logger.error("Error in physics update: %s", str(e))
            raise

    def toggle_pause(self) -> None:
        """Toggle game pause state with proper state transitions."""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
            game_logger.info("Game paused")
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
            game_logger.info("Game resumed")

    def toggle_debug(self) -> None:
        """
        Toggle debug mode for performance monitoring and visualization.
        Updates configuration and triggers appropriate rendering changes.
        """
        self.config.debug_mode = not self.config.debug_mode
        game_logger.info(f"Debug mode {'enabled' if self.config.debug_mode else 'disabled'}")

    def set_log_level(self) -> None:
        """
        Toggle logging level between DEBUG and INFO.
        Affects granularity of performance and game state logging.
        """
        current_level = game_logger.level
        new_level = logging.DEBUG if current_level != logging.DEBUG else logging.INFO
        game_logger.setLevel(new_level)
        game_logger.info(f"Log level set to {logging.getLevelName(new_level)}")

    def quit_game(self) -> None:
        """
        Initiate graceful game shutdown sequence.
        Handles state saving, resource cleanup, and exit procedures.
        """
        self.running = False
        game_logger.info("Initiating game shutdown")

    def _apply_gravity(self) -> bool:
        """
        Apply gravity to the current piece.
        Returns:
        bool: True if the piece moved down, False if blocked
        """
        if not self.current_piece:
            return False
        if self.grid.can_move(
            self.current_piece,
            self.current_piece.x,
            self.current_piece.y + 1
        ):
            self.current_piece.y += 1
            game_logger.debug("Applied gravity to piece at y=%d", self.current_piece.y)
            return True
        return False

    def _reset_lock_delay(self) -> None:
        """Reset piece lock delay after valid movement."""
        self.lock_delay_start = time.perf_counter()
        self.lock_delay_accumulated = 0.0

    def _reset_piece_position(self, piece: Tetrimino) -> None:
        """
        Reset a piece to its initial spawn position.
        Args:
        piece: The Tetrimino to reset
        """
        if piece.piece_type == 'I':
            piece.x = (self.config.columns - 4)
        elif piece.piece_type == 'O':
            piece.x = (self.config.columns - 2)
        else:
            piece.x = (self.config.columns - 3)
        piece.y = 0
        piece.rotation = 0

    def _lock_piece(self) -> None:
        """
        Lock the current piece into the grid and handle post-lock operations.
        Updates game state and triggers appropriate sound effects.
        """
        if self.current_piece:
            self.grid.lock_piece(self.current_piece)
            self.sound_manager.play_sound('lock')
            game_logger.debug("Locked piece at position (%d, %d)",
                              self.current_piece.x, self.current_piece.y)

    def _clear_lines(self) -> None:
        """
        Clear completed lines and update score/level progression.
        Handles line clearing mechanics including:
        - Detection of completed lines
        - Removal of completed lines
        - Score calculation and combo system
        - Level progression
        """
        try:
            lines_cleared = self.grid.clear_lines()
            if lines_cleared > 0:
                self.sound_manager.play_sound('clear')
                self.lines_cleared += lines_cleared
                self.score_handler.add(self._calculate_reward(lines_cleared))

                # Update level based on lines cleared
                new_level = (self.lines_cleared // self.config.lines_per_level) + 1
                if new_level != self.level:
                    self.level = new_level
                    self._update_gravity_speed()
                    game_logger.info(f"Level up! New level: {self.level}")

                game_logger.debug(
                    "Cleared %d lines, total: %d, level: %d",
                    lines_cleared, self.lines_cleared, self.level
                )
        except Exception as e:
            game_logger.error("Error clearing lines: %s", str(e))

    def _calculate_reward(self, lines_cleared: int) -> int:
        """
        Calculate score reward based on lines cleared and current level.
        Args:
        lines_cleared: Number of lines cleared in current move
        Returns:
        int: Calculated score reward
        """
        try:
            reward = self.config.scoring_values.get(
                'tetris' if lines_cleared == 4 else
                'triple' if lines_cleared == 3 else
                'double' if lines_cleared == 2 else
                'single', 0
            ) * self.level
            if lines_cleared > 1:
                self.combo_counter += 1
                reward += (self.combo_counter *
                           self.config.scoring_values['combo'] *
                           self.level)
            else:
                self.combo_counter = 0
            game_logger.debug(
                "Calculated reward: %d for %d lines (combo: %d)",
                reward, lines_cleared, self.combo_counter
            )
            return reward
        except Exception as e:
            game_logger.error("Error calculating reward: %s", str(e))
            return 0

    def _update_gravity_speed(self) -> None:
        """
        Update gravity_delay based on the current level.
        Lower levels have slower gravity; higher levels have faster gravity.
        """
        # Example formula: gravity_delay decreases by 0.05 seconds each level, minimum 0.1 seconds
        new_gravity_delay = max(0.1, 1.0 - (self.level - 1) * 0.05)
        self.config.gravity_delay = new_gravity_delay
        game_logger.info(f"Gravity speed updated: gravity_delay = {self.config.gravity_delay:.2f}s")

    def _fill_bag(self) -> None:
        """
        Fill the piece bag using 7-bag randomization system.
        Ensures fair piece distribution and predictable randomization.
        """
        if not self.bag:
            self.bag = list(TetriminoData.SHAPES.keys())
            random.shuffle(self.bag)
            game_logger.debug("Filled piece bag: %s", self.bag)

    def _get_next_piece(self) -> Tetrimino:
        """
        Retrieve the next piece from the bag with 7-bag system.
        Returns:
        Tetrimino: A new piece instance
        """
        self._fill_bag()
        piece_type = self.bag.pop()
        game_logger.debug("Got next piece: %s", piece_type)
        return Tetrimino(piece_type, self.config.grid_size, self.config.columns)

    def _spawn_new_piece(self) -> bool:
        """
        Spawn a new tetrimino piece and update the preview queue.
        Returns:
        bool: True if piece spawned successfully, False if spawn blocked
        """
        try:
            self.current_piece = self.next_pieces.pop(0)
            new_piece = self._get_next_piece()
            self.next_pieces.append(new_piece)
            if not self.grid.can_place(
                self.current_piece,
                self.current_piece.x,
                self.current_piece.y
            ):
                game_logger.info("Game over: spawn position blocked")
                return False
            self.lock_delay_start = None
            self.lock_delay_accumulated = 0.0
            game_logger.debug("Spawned new piece: %s", self.current_piece.piece_type)
            return True
        except Exception as e:
            game_logger.error("Error spawning new piece: %s", str(e))
            return False

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
            self.score_handler.save_high_score()
            pygame.quit()
            game_logger.info("Game cleanup completed successfully")
        except Exception as e:
            game_logger.error("Error during cleanup: %s", str(e))

    def get_metrics(self) -> GameMetrics:
        """
        Get current game performance metrics.
        Returns:
        GameMetrics: Container with current game metrics
        """
        return GameMetrics(
            frame_count=self.frame_counter,
            physics_updates=self.physics_updates,
            render_time=self.renderer.last_render_time,
            physics_time=self.physics_accumulator,
            total_time=time.perf_counter() - self.start_time
        )

    def _cleanup_partial_initialization(self, initialized_systems: List[str]) -> None:
        """
        Clean up partially initialized systems after initialization failure.
        Args:
        initialized_systems: List of system names that were initialized
        Implementation:
        - Reverse order cleanup
        - Resource deallocation
        - State reset
        """
        game_logger.debug(
            "Cleaning up partial initialization: %s",
            ', '.join(initialized_systems)
        )
        cleanup_order = reversed(initialized_systems)
        for system in cleanup_order:
            try:
                if hasattr(self, system):
                    if system == 'renderer':
                        pygame.quit()
                    delattr(self, system)
                    game_logger.debug("Cleaned up system: %s", system)
            except Exception as e:
                game_logger.error(
                    "Error cleaning up system '%s': %s",
                    system, str(e)
                )
