"""
core/game.py - Tetris Game Core Implementation
Implements the main Tetris game engine with fixed timestep physics and interpolated rendering.
"""

import os
import pygame
import sys
import random
import time
import logging
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
from render.colors import Color

loggers = setup_logging()
game_logger = loggers['game']

@dataclass
class GameMetrics:
    frame_count: int
    physics_updates: int
    render_time: float
    physics_time: float
    total_time: float

class TetrisGame:
    def __init__(self, config: GameConfig):
        try:
            self.config = config
            self.fixed_timestep = config.fixed_timestep
            self.max_frame_time = config.max_frame_time
            self.physics_accumulator = 0.0
            self.physics_alpha = 0.0
            self.gravity_timer = 0.0
            self.start_time = time.perf_counter()

            self.info_log("Initializing Tetris game")
            self._initialize_pygame()
            self._initialize_subsystems()
            self._initialize_game_state()
            self.reset()

            self.info_log(
                "Game initialized with timestep=%.3fs, max_frame=%.3fs",
                self.fixed_timestep,
                self.max_frame_time
            )
        except Exception as e:
            self.info_log("Fatal error during game initialization: %s", str(e))
            raise

    def _initialize_pygame(self) -> None:
        pygame.init()
        pygame.display.init()

        flags = pygame.DOUBLEBUF
        if self.config.fullscreen:
            flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode(
            (self.config.screen_width + self.config.ui_width, self.config.screen_height),
            flags
        )
        pygame.display.set_caption("Tetris")

        self.info_log(
            "Pygame initialized with display mode: %dx%d",
            self.config.screen_width + self.config.ui_width,
            self.config.screen_height
        )

    def _initialize_game_controls(self) -> None:
        self.move_left = lambda: self._move_piece(-1, 0)
        self.move_right = lambda: self._move_piece(1, 0)
        self.soft_drop = lambda: self._move_piece(0, 1)
        self.rotate_cw = lambda: self._rotate_piece(True)
        self.rotate_ccw = lambda: self._rotate_piece(False)
        self.hard_drop = self._hard_drop
        self.hold = self._hold_piece
        self.info_log("Game controls initialized")

    def reset(self) -> None:
        try:
            self.grid.reset()
            self.score_handler.reset()

            self.level = self.config.initial_level
            self.lines_cleared = 0
            self.combo_counter = 0
            self.held_piece = None
            self.bag = []
            self.next_pieces = [
                self._get_next_piece() for _ in range(self.config.preview_pieces)
            ]
            self.current_piece = self._get_next_piece()

            if not self.grid.can_place(self.current_piece, self.current_piece.x, self.current_piece.y):
                self.info_log("Game over - cannot spawn new piece")
                self.state = GameState.GAME_OVER
            else:
                self.state = GameState.PLAYING

            self.lock_delay_start = None
            self.lock_delay_accumulated = 0.0
            self.physics_accumulator = 0.0
            self.gravity_timer = 0.0

            self.info_log("Game state reset completed")

        except Exception as e:
            self.info_log("Failed to reset game state: %s", str(e))
            raise

    def _initialize_subsystems(self) -> None:
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

            self.info_log("All subsystems initialized successfully")

        except Exception as e:
            self.info_log(
                "Failed to initialize game subsystems at stage '%s': %s",
                initialized_systems[-1] if initialized_systems else 'unknown',
                str(e)
            )
            self._cleanup_partial_initialization(initialized_systems)
            raise

    def _setup_input_handler(self) -> None:
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
        self.info_log("Input handler configured with %d callbacks", len(callbacks))

    def _initialize_game_state(self) -> None:
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

        self.info_log("Game state initialized")

    def debug_log(self, message: str, *args) -> None:
        if self.config.debug_mode:
            game_logger.debug(message, *args)

    def info_log(self, message: str, *args) -> None:
        game_logger.info(message, *args)

    def _move_piece(self, dx: int, dy: int) -> None:
        if self.current_piece and self.state == GameState.PLAYING:
            if self.grid.can_move(
                self.current_piece,
                self.current_piece.x + dx,
                self.current_piece.y + dy
            ):
                self.current_piece.x += dx
                self.current_piece.y += dy

                if dy > 0:
                    self.score_handler.add(
                        self.config.scoring_values['soft_drop'] * dy
                    )
                elif dx != 0:
                    self.sound_manager.play_sound('move')

                self.debug_log(
                    "Piece moved to (%d, %d)",
                    self.current_piece.x,
                    self.current_piece.y
                )

    def _rotate_piece(self, clockwise: bool) -> None:
        if self.current_piece and self.state == GameState.PLAYING:
            if self.current_piece.try_rotation(self.grid, clockwise):
                self.sound_manager.play_sound('rotate')
                self.debug_log(
                    "Piece rotated %s to rotation %d",
                    "clockwise" if clockwise else "counter-clockwise",
                    self.current_piece.rotation
                )
            else:
                self.debug_log("Rotation failed for piece '%s'", self.current_piece.piece_type)

    def _hard_drop(self) -> None:
        if self.current_piece and self.state == GameState.PLAYING:
            drop_distance = 0
            while self.grid.can_move(
                self.current_piece,
                self.current_piece.x,
                self.current_piece.y + 1
            ):
                self.current_piece.y += 1
                drop_distance += 1

            if drop_distance > 0:
                self.score_handler.add(
                    drop_distance * self.config.scoring_values['hard_drop']
                )
                self.renderer.trigger_particle_effect((self.current_piece.x, self.current_piece.y))

            self._lock_piece()
            self.sound_manager.play_sound('drop')
            self.debug_log(
                "Piece hard dropped %d spaces",
                drop_distance
            )

    def _hold_piece(self) -> None:
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
        self.debug_log("Piece held")

    def run(self) -> None:
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
                self.gravity_timer += frame_time

                while self.gravity_timer >= self.config.gravity_delay:
                    if not self._apply_gravity():
                        if self.lock_delay_start is None:
                            self.lock_delay_start = time.perf_counter()
                            self.lock_delay_accumulated = 0.0
                        else:
                            self.lock_delay_accumulated += self.fixed_timestep

                        if self.lock_delay_accumulated >= self.config.lock_delay / 1000.0:
                            self._lock_piece()
                            self._clear_lines()
                            if not self._spawn_new_piece():
                                self.info_log("Game over - cannot spawn new piece")
                                self.state = GameState.GAME_OVER
                            self.lock_delay_start = None
                            self.lock_delay_accumulated = 0.0
                    else:
                        self.lock_delay_start = None
                        self.lock_delay_accumulated = 0.0

                    self.gravity_timer -= self.config.gravity_delay

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
        try:
            self.renderer.clear_surfaces()

            if self.state == GameState.MENU:
                menu_options = ["Start Game", "Options", "High Scores", "Quit"]
                self.renderer.draw_menu(menu_options, self.selected_menu_index)
            else:
                self.renderer.draw_grid(self.grid)

                if self.current_piece and self.state == GameState.PLAYING:
                    ghost_y = self.current_piece.get_ghost_position(self.grid)
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.state == GameState.MENU:
                self._handle_menu_input(event)
            else:
                self.input_handler.handle_event(event)

    def _handle_menu_input(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_menu_index = (self.selected_menu_index - 1) % 4
                self.info_log(f"Menu selection moved to index {self.selected_menu_index}")
            elif event.key == pygame.K_DOWN:
                self.selected_menu_index = (self.selected_menu_index + 1) % 4
                self.info_log(f"Menu selection moved to index {self.selected_menu_index}")
            elif event.key == pygame.K_RETURN:
                self._handle_menu_selection()

    def _handle_menu_selection(self) -> None:
        menu_options = ["Start Game", "Options", "High Scores", "Quit"]
        selected_option = menu_options[self.selected_menu_index]
        self.info_log(f"Menu option selected: {selected_option}")

        if selected_option == "Start Game":
            self.reset()
            self.state = GameState.PLAYING
        elif selected_option == "Options":
            pass
        elif selected_option == "High Scores":
            pass
        elif selected_option == "Quit":
            self.running = False

    def _update_physics(self, delta_time: float) -> None:
        if self.state != GameState.PLAYING:
            return

    def toggle_pause(self) -> None:
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
            self.info_log("Game paused")
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
            self.info_log("Game resumed")

    def toggle_debug(self) -> None:
        self.config.debug_mode = not self.config.debug_mode
        self.info_log(f"Debug mode {'enabled' if self.config.debug_mode else 'disabled'}")

    def set_log_level(self) -> None:
        current_level = game_logger.level
        new_level = logging.DEBUG if current_level != logging.DEBUG else logging.INFO
        game_logger.setLevel(new_level)
        game_logger.info(f"Log level set to {logging.getLevelName(new_level)}")

    def quit_game(self) -> None:
        self.running = False
        self.info_log("Initiating game shutdown")

    def _apply_gravity(self) -> bool:
        if not self.current_piece:
            return False

        if self.grid.can_move(
            self.current_piece,
            self.current_piece.x,
            self.current_piece.y + 1
        ):
            self.current_piece.y += 1
            self.debug_log("Applied gravity to piece at y=%d", self.current_piece.y)
            return True
        return False

    def _reset_piece_position(self, piece: Tetrimino) -> None:
        if piece.piece_type == 'I':
            piece.x = (self.config.columns - 4)
        elif piece.piece_type == 'O':
            piece.x = (self.config.columns - 2)
        else:
            piece.x = (self.config.columns - 3)

        piece.y = 0
        piece.rotation = 0
        piece.shape = TetriminoData.get_initial_shape(piece.piece_type, piece.rotation)
        self.debug_log(
            "Reset piece '%s' to initial position (%d, %d)",
            piece.piece_type, piece.x, piece.y
        )

    def _lock_piece(self) -> None:
        if self.current_piece:
            self.grid.lock_piece(self.current_piece)
            self.sound_manager.play_sound('lock')
            self.debug_log(
                "Locked piece at position (%d, %d)",
                self.current_piece.x, self.current_piece.y
            )

    def _clear_lines(self) -> None:
        try:
            cleared_lines = self.grid.detect_cleared_lines()
            lines_cleared = len(cleared_lines)

            if lines_cleared > 0:
                self.sound_manager.play_sound('clear')
                self.lines_cleared += lines_cleared
                self.score_handler.add(self._calculate_reward(lines_cleared))
                self._spawn_particle_effects_for_cleared_lines()
                self._animate_line_clears(cleared_lines)
                self.grid.clear_cleared_lines()

                new_level = (self.lines_cleared // self.config.lines_per_level) + self.config.initial_level
                if new_level != self.level:
                    self.level = new_level
                    self._update_gravity_speed()
                    self.info_log(f"Level up! New level: {self.level}")

                self.debug_log(
                    "Cleared %d lines, total: %d, level: %d",
                    lines_cleared, self.lines_cleared, self.level
                )
        except Exception as e:
            game_logger.error("Error clearing lines: %s", str(e))

    def _calculate_reward(self, lines_cleared: int) -> int:
        try:
            if lines_cleared == 1:
                reward = self.config.scoring_values['single'] * self.level
            elif lines_cleared == 2:
                reward = self.config.scoring_values['double'] * self.level
            elif lines_cleared == 3:
                reward = self.config.scoring_values['triple'] * self.level
            elif lines_cleared == 4:
                reward = self.config.scoring_values['tetris'] * self.level
            else:
                reward = 0

            if lines_cleared > 1:
                self.combo_counter += 1
                reward += (self.combo_counter *
                           self.config.scoring_values['combo'] *
                           self.level)
            else:
                self.combo_counter = 0

            popup_position = (self.config.screen_width + 20, 250)
            self.renderer.add_score_popup(reward, popup_position)
            return reward

        except Exception as e:
            game_logger.error("Error calculating reward: %s", str(e))
            return 0

    def _update_gravity_speed(self) -> None:
        new_gravity_delay = max(0.1, 1.0 - (self.level - 1) * 0.05)
        self.config.gravity_delay = new_gravity_delay
        self.info_log(f"Gravity speed updated: gravity_delay = {self.config.gravity_delay:.2f}s")

    def _fill_bag(self) -> None:
        if not self.bag:
            self.bag = list(TetriminoData.SHAPES.keys())
            random.shuffle(self.bag)
            self.debug_log("Filled piece bag: %s", self.bag)

    def _get_next_piece(self) -> Tetrimino:
        self._fill_bag()
        piece_type = self.bag.pop()
        self.debug_log("Got next piece: %s", piece_type)
        return Tetrimino(piece_type, self.config.grid_size, self.config.columns)

    def _spawn_new_piece(self) -> bool:
        try:
            self.current_piece = self.next_pieces.pop(0)
            new_piece = self._get_next_piece()
            self.next_pieces.append(new_piece)

            if not self.grid.can_place(self.current_piece, self.current_piece.x, self.current_piece.y):
                self.info_log("Game over: spawn position blocked")
                return False

            self.lock_delay_start = None
            self.lock_delay_accumulated = 0.0
            self.debug_log("Spawned new piece: %s", self.current_piece.piece_type)
            return True

        except Exception as e:
            self.debug_log("Error spawning new piece: %s", str(e))
            return False

    def _spawn_particle_effects_for_cleared_lines(self) -> None:
        cleared_lines = self.grid.get_cleared_lines()
        for y in cleared_lines:
            for x in range(self.config.columns):
                self.renderer.trigger_particle_effect((x, y))
        self.debug_log("Spawned particle effects for cleared lines")

    def _animate_line_clears(self, cleared_lines: List[int]) -> None:
        animation_duration = 0.5
        animation_steps = 30
        for step in range(animation_steps):
            for y in cleared_lines:
                for x in range(self.config.columns):
                    if step % 2 == 0:
                        self.grid.grid[y][x] = 'flash_color'
                    else:
                        self.grid.grid[y][x] = 'normal_color'

            self.renderer.draw_grid(self.grid)
            self.renderer.draw_score_popups()
            self.renderer.compose_frame(self.screen, self.frame_counter)
            pygame.display.flip()
            pygame.time.delay(int(animation_duration * 1000 / animation_steps))

    def cleanup(self) -> None:
        try:
            self.score_handler.save_high_score()
            pygame.quit()
            self.info_log("Game cleanup completed successfully")
        except Exception as e:
            self.info_log("Error during cleanup: %s", str(e))

    def get_metrics(self) -> GameMetrics:
        return GameMetrics(
            frame_count=self.frame_counter,
            physics_updates=self.physics_updates,
            render_time=self.renderer.last_render_time,
            physics_time=self.physics_accumulator,
            total_time=time.perf_counter() - self.start_time
        )

    def _cleanup_partial_initialization(self, initialized_systems: List[str]) -> None:
        self.debug_log(
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
                self.debug_log("Cleaned up system: %s", system)
            except Exception as e:
                self.info_log(
                    "Error cleaning up system '%s': %s",
                    system, str(e)
                )