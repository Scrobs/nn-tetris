# File: core/score.py

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class Score:
    """
    Score handling system with persistence, overflow protection, and combo tracking.
    
    This class manages game scoring mechanics including:
    - Basic score tracking and validation
    - High score persistence
    - Combo system tracking
    - Recent points tracking for animations
    - Score overflow protection
    
    Attributes:
        max_score (int): Maximum allowable score value
        value (int): Current score value
        high_score (int): Highest achieved score
        filepath (str): Path to high score persistence file
        recent_points (List[Tuple[float, int]]): Recent score changes with timestamps
        combo_counter (int): Current combo multiplier
    """

    def __init__(self, max_score: int, filepath: Optional[str] = None):
        """
        Initialize the scoring system.

        Args:
            max_score: Maximum allowed score value
            filepath: Optional custom path for high score file storage
        
        Raises:
            ValueError: If max_score is not a positive integer
        """
        self.max_score = max_score
        self.value = 0
        self.high_score = 0
        self.filepath = filepath or os.path.join(Path.home(), ".tetris_game", "high_score.json")
        self.recent_points: List[Tuple[float, int]] = []
        self.combo_counter = -1
        
        self._validate_max_score()
        self.load_high_score()
        game_logger.debug("Score system initialized with max_score: %d", self.max_score)

    def reset(self) -> None:
        """
        Reset the scoring system to initial state.
        
        Resets:
        - Current score to 0
        - Combo counter to -1
        - Recent points list to empty
        Preserves high score
        """
        self.value = 0
        self.combo_counter = -1
        self.recent_points = []
        game_logger.debug("Score system reset: value=%d, combo=%d", self.value, self.combo_counter)

    def _validate_max_score(self) -> None:
        """
        Ensure max_score is a positive integer.
        
        Raises:
            ValueError: If max_score is invalid
        """
        if not isinstance(self.max_score, int) or self.max_score <= 0:
            game_logger.error("Invalid max_score: %s", self.max_score)
            raise ValueError("max_score must be a positive integer.")

    def add(self, points: int) -> None:
        """
        Add points to the current score with overflow protection.
        
        Args:
            points: Number of points to add
            
        Points are added with the following constraints:
        - Must be a positive integer
        - Total score cannot exceed max_score
        - Points are recorded with timestamp for animation
        """
        if not isinstance(points, int):
            game_logger.error("Attempted to add non-integer points: %s", points)
            return

        old_value = self.value
        self.value = min(self.value + points, self.max_score)
        
        if self.value == self.max_score and old_value != self.max_score:
            game_logger.warning("Score overflowed. Capped at max_score: %d", self.max_score)
        
        # Record points for animation
        current_time = time.time()
        self.recent_points.append((current_time, points))
        
        # Cleanup old points
        self.recent_points = [
            (t, p) for t, p in self.recent_points if current_time - t <= 5.0
        ]
        
        game_logger.debug("Added %d points. Total score: %d", points, self.value)

    def get_recent_points(self, window: float = 5.0) -> int:
        """
        Get the total points added in the recent time window.
        
        Args:
            window: Time window in seconds for considering recent points
            
        Returns:
            int: Sum of points added within the specified window
        """
        current_time = time.time()
        return sum(p for t, p in self.recent_points if current_time - t <= window)

    @property
    def value_display(self) -> int:
        """
        Get the current score.
        
        Returns:
            int: Current score value
        """
        return self.value

    @property
    def high_score_display(self) -> int:
        """
        Get the high score.
        
        Returns:
            int: Current high score value
        """
        return self.high_score

    def load_high_score(self) -> None:
        """
        Load the high score from persistent storage.
        
        Creates high score directory if it doesn't exist.
        Sets high score to 0 if file is missing or corrupted.
        """
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.high_score = data.get("high_score", 0)
                game_logger.debug("Loaded high score: %d", self.high_score)
            else:
                self.high_score = 0
                game_logger.debug("High score file does not exist. Starting at 0.")
        except (IOError, json.JSONDecodeError) as e:
            game_logger.error("Failed to load high score: %s", str(e))
            self.high_score = 0

    def save_high_score(self) -> None:
        """
        Save the high score to persistent storage.
        
        Updates high score if current score is higher.
        Creates storage directory if needed.
        Handles IO errors with logging.
        """
        try:
            if self.value > self.high_score:
                self.high_score = self.value
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
                
                with open(self.filepath, 'w') as f:
                    json.dump({"high_score": self.high_score}, f)
                game_logger.info("New high score saved: %d", self.high_score)
        except IOError as e:
            game_logger.error("Failed to save high score: %s", str(e))

    def update_combo(self, lines_cleared: int) -> None:
        """
        Update the combo counter based on lines cleared.
        
        Args:
            lines_cleared: Number of lines cleared in current move
            
        Increments combo counter for multiple lines.
        Resets combo counter if no lines cleared.
        """
        if lines_cleared > 1:
            self.combo_counter += 1
            game_logger.debug("Combo increased to: %d", self.combo_counter)
        else:
            self.combo_counter = 0
            game_logger.debug("Combo reset")