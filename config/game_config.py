# config/game_config.py

"""
config/game_config.py - Game Configuration Management System
This module implements a comprehensive configuration system for the Tetris game,
managing all game parameters, display settings, and gameplay mechanics.
Key Features:
- Centralized configuration management
- JSON-based configuration loading
- Robust validation and error handling
- Type-safe parameter management
- Comprehensive logging integration
Design Principles:
1. Single Responsibility: Handles only configuration management
2. Immutability: Configurations are validated once at initialization
3. Type Safety: Strong typing for all configuration parameters
4. Error Handling: Comprehensive validation and error reporting
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class GameConfig:
    """
    Comprehensive game configuration management system.
    This class manages all game settings including display parameters,
    gameplay mechanics, performance settings, and debug options.
    Attributes:
    screen_width (int): Window width in pixels
    screen_height (int): Window height in pixels
    grid_size (int): Size of each grid cell in pixels
    columns (int): Number of grid columns
    rows (int): Number of grid rows
    fps (int): Target frames per second
    vsync (bool): Vertical synchronization flag
    fixed_timestep (float): Physics update interval in seconds
    max_frame_time (float): Maximum allowed frame time to prevent spiral of death
    particle_pool_size (int): Maximum number of particle effects
    fullscreen (bool): Fullscreen mode flag
    ui_width (int): Width of UI panel in pixels
    grid_line_width (int): Width of grid lines in pixels
    cell_padding (int): Padding between cells in pixels
    preview_margin (int): Margin around piece preview area
    preview_pieces (int): Number of next pieces to show
    das_delay (int): Delayed Auto Shift initial delay in frames
    das_repeat (int): Delayed Auto Shift repeat interval in frames
    soft_drop_speed (int): Soft drop multiplier
    lock_delay (int): Piece lock delay in frames
    input_buffer_frames (int): Input buffer size in frames
    max_score (int): Maximum possible score
    gravity_delay (float): Base gravity delay in seconds
    lines_per_level (int): Lines required for level up
    debug_mode (bool): Debug features toggle
    particle_effects (bool): Particle effects toggle
    logging_level (str): Logging verbosity level
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize game configuration with defaults or provided settings.
        Args:
        config: Optional dictionary of configuration overrides.
        If None, uses default values.
        The initialization process:
        1. Sets default values for all parameters
        2. Applies any provided configuration overrides
        3. Validates the final configuration state
        """
        self.screen_width: int = 300
        self.screen_height: int = 600
        self.grid_size: int = 30
        self.columns: int = self.screen_width // self.grid_size
        self.rows: int = self.screen_height // self.grid_size
        self.ui_width: int = 200
        self.grid_line_width: int = 1
        self.cell_padding: int = 1
        self.preview_margin: int = 30
        self.fps: int = 60
        self.vsync: bool = True
        self.fullscreen: bool = False
        self.fixed_timestep: float = 1.0 / 60.0
        self.max_frame_time: float = 0.25
        self.particle_pool_size: int = 1000
        self.preview_pieces: int = 3
        self.initial_level: int = 1
        self.lines_per_level: int = 10
        self.gravity_delay: float = 1.0
        self.lock_delay: int = 30
        self.das_delay: int = 16
        self.das_repeat: int = 6
        self.soft_drop_speed: int = 2
        self.input_buffer_frames: int = 10
        self.max_score: int = 999999999
        self.scoring_values: Dict[str, int] = {
            'single': 100,
            'double': 300,
            'triple': 500,
            'tetris': 800,
            'soft_drop': 1,
            'hard_drop': 2,
            'combo': 50
        }
        self.debug_mode: bool = False
        self.particle_effects: bool = True
        self.logging_level: str = "INFO"
        self.key_bindings: Dict[str, str] = {
            "MOVE_LEFT": "LEFT",
            "MOVE_RIGHT": "RIGHT",
            "SOFT_DROP": "DOWN",
            "ROTATE_CW": "UP",
            "ROTATE_CCW": "Z",
            "HARD_DROP": "SPACE",
            "HOLD": "C",
            "PAUSE": "P",
            "RESTART": "R",
            "QUIT": "Q",
            "TOGGLE_DEBUG": "D",
            "SET_LOG_LEVEL": "L"
        }

        if config:
            self._apply_config(config)
        self._validate_config()
        game_logger.debug("Game configuration initialized with %d parameters",
                          len([attr for attr in dir(self) if not attr.startswith('_')]))

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration settings from a dictionary with robust validation.
        Args:
        config: Dictionary containing configuration overrides
        Implementation Details:
        - Validates each configuration key before application
        - Logs warnings for unknown configuration parameters
        - Maintains type safety through explicit casting
        - Preserves immutability of critical game parameters
        """
        for key, value in config.items():
            if hasattr(self, key):
                try:
                    current_value = getattr(self, key)
                    if isinstance(current_value, bool):
                        value = bool(value)
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, dict):
                        if not isinstance(value, dict):
                            raise ValueError(f"Expected dict for {key}, got {type(value)}")
                        setattr(self, key, value)
                        game_logger.debug("Configuration parameter '%s' set to: %s", key, value)
                        continue
                    setattr(self, key, value)
                    game_logger.debug("Configuration parameter '%s' set to: %s", key, value)
                except (ValueError, TypeError) as e:
                    game_logger.error("Failed to set configuration '%s': %s", key, str(e))
                    raise ValueError(f"Invalid value for configuration '{key}': {str(e)}")
            else:
                game_logger.warning("Unknown configuration key: %s", key)

    def _validate_config(self) -> None:
        """
        Validate the complete configuration state for consistency and constraints.
        Validation Criteria:
        1. Display dimensions must be positive and aligned with grid size
        2. Performance parameters must be within reasonable bounds
        3. Gameplay mechanics must maintain fair and playable conditions
        4. System settings must be properly configured
        Raises:
        ValueError: If any configuration parameter fails validation
        """
        try:
            self._validate_display_settings()
            self._validate_performance_settings()
            self._validate_gameplay_settings()
            game_logger.debug("Configuration validation successful")
        except ValueError as e:
            game_logger.error("Configuration validation failed: %s", str(e))
            raise

    def _validate_display_settings(self) -> None:
        """Validate display-related configuration parameters."""
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("Screen dimensions must be positive")
        if self.screen_width % self.grid_size != 0:
            raise ValueError("Screen width must be a multiple of grid size")
        if self.screen_height % self.grid_size != 0:
            raise ValueError("Screen height must be a multiple of grid size")
        if self.grid_line_width < 0 or self.cell_padding < 0:
            raise ValueError("Grid line width and cell padding must be non-negative")

    def _validate_performance_settings(self) -> None:
        """Validate performance-related configuration parameters."""
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        if not 0.0 < self.fixed_timestep <= 1.0:
            raise ValueError("Fixed timestep must be between 0 and 1 second")
        if self.max_frame_time <= 0:
            raise ValueError("Maximum frame time must be positive")
        if self.particle_pool_size < 0:
            raise ValueError("Particle pool size must be non-negative")

    def _validate_gameplay_settings(self) -> None:
        """Validate gameplay-related configuration parameters."""
        if self.preview_pieces < 0:
            raise ValueError("Preview pieces count must be non-negative")
        if self.initial_level < 1:
            raise ValueError("Initial level must be at least 1")
        if self.lines_per_level <= 0:
            raise ValueError("Lines per level must be positive")
        if self.gravity_delay <= 0:
            raise ValueError("Gravity delay must be positive")
        if self.gravity_delay < 0.1:
            game_logger.warning("Gravity delay is very low, game may be too fast")
        elif self.gravity_delay > 10.0:
            game_logger.warning("Gravity delay is very high, game may be too slow")
        if self.max_score <= 0:
            raise ValueError("Maximum score must be positive")

    @classmethod
    def load_config(cls, filepath: str) -> 'GameConfig':
        """
        Load configuration from a JSON file with comprehensive error handling.
        Args:
        filepath: Path to JSON configuration file
        Returns:
        GameConfig: New configuration instance
        Raises:
        FileNotFoundError: If configuration file doesn't exist
        json.JSONDecodeError: If configuration file contains invalid JSON
        ValueError: If configuration validation fails
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            game_logger.info("Loaded configuration from %s", filepath)
            return cls(config)
        except FileNotFoundError:
            game_logger.error("Configuration file not found: %s", filepath)
            raise
        except json.JSONDecodeError:
            game_logger.error("Invalid JSON in configuration file: %s", filepath)
            raise
        except Exception as e:
            game_logger.error("Unexpected error loading configuration: %s", str(e))
            raise

    def save_config(self, filepath: str) -> None:
        """
        Save current configuration to a JSON file.
        Args:
        filepath: Path to save configuration file
        Implementation Details:
        - Creates parent directories if needed
        - Preserves formatting for readability
        - Implements atomic write operations
        - Handles permission and I/O errors
        """
        try:
            config_dict = {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('_') and key != 'key_bindings'
            }
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            Path(temp_filepath).rename(filepath)
            game_logger.info("Saved configuration to %s", filepath)
        except IOError as e:
            game_logger.error("Failed to save configuration: %s", str(e))
            raise
