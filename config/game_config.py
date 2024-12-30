# config/game_config.py

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class GameConfig:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        try:
            self._validate_display_settings()
            self._validate_performance_settings()
            self._validate_gameplay_settings()
            game_logger.debug("Configuration validation successful")
        except ValueError as e:
            game_logger.error("Configuration validation failed: %s", str(e))
            raise

    def _validate_display_settings(self) -> None:
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("Screen dimensions must be positive")
        if self.screen_width % self.grid_size != 0:
            raise ValueError("Screen width must be a multiple of grid size")
        if self.screen_height % self.grid_size != 0:
            raise ValueError("Screen height must be a multiple of grid size")
        if self.grid_line_width < 0 or self.cell_padding < 0:
            raise ValueError("Grid line width and cell padding must be non-negative")

    def _validate_performance_settings(self) -> None:
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        if not 0.0 < self.fixed_timestep <= 1.0:
            raise ValueError("Fixed timestep must be between 0 and 1 second")
        if self.max_frame_time <= 0:
            raise ValueError("Maximum frame time must be positive")
        if self.particle_pool_size < 0:
            raise ValueError("Particle pool size must be non-negative")

    def _validate_gameplay_settings(self) -> None:
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
