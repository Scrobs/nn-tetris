# File: config/game_config.py

import json
from typing import Dict, Any, Optional
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class GameConfig:
    """
    Comprehensive game configuration class managing all Tetris game settings.
    
    Handles:
    - Display settings (screen dimensions, UI layout)
    - Game mechanics (grid size, preview pieces)
    - Performance settings (FPS, vsync)
    - Input configuration (key bindings, DAS settings)
    - Debug and particle effect settings
    
    Configuration can be loaded from JSON or initialized with defaults.
    All critical game dimensions and timings are centrally managed here.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize game configuration with defaults or provided settings.
        
        Args:
            config: Optional dictionary of configuration overrides
        """
        # Core game dimensions
        self.screen_width: int = 300
        self.screen_height: int = 600
        self.grid_size: int = 30
        self.columns: int = self.screen_width // self.grid_size
        self.rows: int = self.screen_height // self.grid_size
        
        # Display settings
        self.fps: int = 60
        self.vsync: bool = True
        self.fullscreen: bool = False
        self.ui_width: int = 200
        
        # Grid appearance
        self.grid_line_width: int = 1
        self.cell_padding: int = 1
        
        # Game mechanics
        self.preview_margin: int = 30
        self.preview_pieces: int = 3
        self.initial_level: int = 1
        self.lines_per_level: int = 10
        self.gravity_delay: float = 1.0  # Base falling speed in seconds
        self.lock_delay: int = 30  # Frames before piece locks in place
        
        # Input handling
        self.das_delay: int = 16  # Delayed Auto Shift initial delay (frames)
        self.das_repeat: int = 6  # DAS repeat rate (frames)
        self.soft_drop_speed: int = 2  # Soft drop multiplier
        self.input_buffer_frames: int = 10  # Input buffer window
        
        # Scoring
        self.max_score: int = 999999999
        self.lines_per_level: int = 10
        self.scoring_values: Dict[str, int] = {
            'single': 100,
            'double': 300,
            'triple': 500,
            'tetris': 800,
            'soft_drop': 1,
            'hard_drop': 2,
            'combo': 50
        }
        
        # System settings
        self.debug_mode: bool = False
        self.particle_effects: bool = True
        self.logging_level: str = "INFO"
        
        # Key bindings
        self.key_bindings: Dict[str, str] = {
            'MOVE_LEFT': 'LEFT',
            'MOVE_RIGHT': 'RIGHT',
            'SOFT_DROP': 'DOWN',
            'ROTATE_CW': 'UP',
            'ROTATE_CCW': 'Z',
            'HARD_DROP': 'SPACE',
            'HOLD': 'C',
            'PAUSE': 'P',
            'RESTART': 'R',
            'QUIT': 'Q',
            'TOGGLE_DEBUG': 'D',
            'SET_LOG_LEVEL': 'L'
        }
        
        # Apply custom configuration if provided
        if config:
            self._apply_config(config)
            
        # Validate configuration after initialization
        self._validate_config()
        
        game_logger.debug("Game configuration initialized")

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration settings from a dictionary.
        
        Args:
            config: Dictionary containing configuration overrides
            
        Logs warnings for unknown configuration keys.
        """
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                game_logger.debug("Set %s to %s", key, value)
            else:
                game_logger.warning("Unknown configuration key: %s", key)

    def _validate_config(self) -> None:
        """
        Validate configuration settings for consistency and constraints.
        
        Checks:
        - Screen dimensions are positive and multiples of grid size
        - Grid size divides screen dimensions evenly
        - FPS and timing values are positive
        - Score limits are reasonable
        
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            # Validate screen dimensions
            if self.screen_width <= 0 or self.screen_height <= 0:
                raise ValueError("Screen dimensions must be positive")
            
            if self.screen_width % self.grid_size != 0 or self.screen_height % self.grid_size != 0:
                raise ValueError("Screen dimensions must be multiples of grid size")
            
            # Validate timing values
            if self.fps <= 0:
                raise ValueError("FPS must be positive")
            
            if self.das_delay < 0 or self.das_repeat < 0:
                raise ValueError("DAS settings must be non-negative")
            
            # Validate score settings
            if self.max_score <= 0:
                raise ValueError("Max score must be positive")
            
            game_logger.debug("Configuration validation successful")
            
        except ValueError as e:
            game_logger.error("Configuration validation failed: %s", str(e))
            raise

    @classmethod
    def load_config(cls, filepath: str) -> 'GameConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            GameConfig: New configuration instance
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            game_logger.info("Loaded configuration from %s", filepath)
            return cls(config)
        except FileNotFoundError as e:
            game_logger.error("Configuration file not found: %s", filepath)
            raise e
        except json.JSONDecodeError as e:
            game_logger.error("Invalid JSON in configuration file: %s", filepath)
            raise e
        except Exception as e:
            game_logger.error("Unexpected error loading configuration: %s", str(e))
            raise e

    def save_config(self, filepath: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            filepath: Path to save configuration file
            
        Handles creation of parent directories if needed.
        """
        try:
            config_dict = {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('_')
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            game_logger.info("Saved configuration to %s", filepath)
        except IOError as e:
            game_logger.error("Failed to save configuration: %s", str(e))
            raise

    def get_level_speed(self, level: int) -> float:
        """
        Calculate piece falling speed for given level.
        
        Args:
            level: Current game level
            
        Returns:
            float: Time in seconds between gravity updates
        """
        return max(0.05, self.gravity_delay * (0.8 ** (level - 1)))