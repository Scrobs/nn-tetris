#!/usr/bin/env python3
"""
Unit tests for the Score class.
"""

import os
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, mock_open
from core.score import Score


@pytest.fixture
def temp_score_file():
    """Fixture providing a temporary file for score storage."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def score_instance(temp_score_file):
    """Fixture providing a Score instance with temporary file storage."""
    return Score(max_score=1000, filepath=temp_score_file)


class TestScore:
    """Test suite for the Score class."""

    def test_initialization(self, score_instance):
        """Test initial state of Score instance."""
        assert score_instance.value == 0
        assert score_instance.high_score == 0
        assert score_instance.combo_counter == -1
        assert isinstance(score_instance.recent_points, list)
        assert len(score_instance.recent_points) == 0

    def test_invalid_max_score(self):
        """Test validation of max_score parameter."""
        with pytest.raises(ValueError):
            Score(max_score=0)
        
        with pytest.raises(ValueError):
            Score(max_score=-100)
        
        with pytest.raises(ValueError):
            Score(max_score="1000")

    def test_add_points(self, score_instance):
        """Test adding points to score."""
        score_instance.add(50)
        assert score_instance.value == 50
        
        score_instance.add(30)
        assert score_instance.value == 80
        
        # Test adding negative points (should be ignored)
        score_instance.add(-20)
        assert score_instance.value == 80
        
        # Test adding non-integer points (should be ignored)
        score_instance.add(10.5)
        assert score_instance.value == 80

    def test_max_score_cap(self, score_instance):
        """Test that score doesn't exceed max_score."""
        score_instance.add(800)
        assert score_instance.value == 800
        
        score_instance.add(300)  # Would exceed max_score of 1000
        assert score_instance.value == 1000
        
        score_instance.add(50)  # Already at max
        assert score_instance.value == 1000

    def test_reset(self, score_instance):
        """Test score reset functionality."""
        score_instance.add(500)
        score_instance.combo_counter = 5
        score_instance.recent_points.append((time.time(), 100))
        
        score_instance.reset()
        
        assert score_instance.value == 0
        assert score_instance.combo_counter == -1
        assert len(score_instance.recent_points) == 0

    def test_recent_points_window(self, score_instance):
        """Test recent points tracking within time window."""
        current_time = time.time()
        
        # Add points at different times
        with patch('time.time') as mock_time:
            # Points within 5-second window
            mock_time.return_value = current_time
            score_instance.add(100)
            
            mock_time.return_value = current_time + 2
            score_instance.add(200)
            
            mock_time.return_value = current_time + 4
            score_instance.add(300)
            
            # Points outside 5-second window
            mock_time.return_value = current_time + 6
            score_instance.add(400)
            
            # Check points within window
            mock_time.return_value = current_time + 4.5
            recent_total = score_instance.get_recent_points(window=5.0)
            assert recent_total == 600  # 100 + 200 + 300

    def test_high_score_persistence(self, temp_score_file):
        """Test saving and loading high scores."""
        # Create initial score instance and set high score
        score1 = Score(max_score=1000, filepath=temp_score_file)
        score1.add(500)
        score1.save_high_score()
        
        # Create new instance and verify high score loaded
        score2 = Score(max_score=1000, filepath=temp_score_file)
        assert score2.high_score == 500

    def test_high_score_file_corruption(self, temp_score_file):
        """Test handling of corrupted high score file."""
        # Write invalid JSON to score file
        with open(temp_score_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle corrupt file gracefully
        score = Score(max_score=1000, filepath=temp_score_file)
        assert score.high_score == 0

    def test_save_high_score_file_error(self, score_instance):
        """Test handling of file write errors during high score save."""
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Permission denied")
            score_instance.add(100)
            # Should not raise exception
            score_instance.save_high_score()

    def test_update_combo(self, score_instance):
        """Test combo counter updates."""
        # Single line clear resets combo
        score_instance.update_combo(1)
        assert score_instance.combo_counter == 0
        
        # Multi-line clear increases combo
        score_instance.update_combo(2)
        assert score_instance.combo_counter == 1
        
        score_instance.update_combo(4)
        assert score_instance.combo_counter == 2
        
        # Single line clear resets combo
        score_instance.update_combo(1)
        assert score_instance.combo_counter == 0

    def test_high_score_directory_creation(self):
        """Test creation of high score directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            score_path = os.path.join(temp_dir, 'subdir', 'high_score.json')
            score = Score(max_score=1000, filepath=score_path)
            score.add(100)
            score.save_high_score()
            
            assert os.path.exists(os.path.dirname(score_path))

    def test_score_display_properties(self, score_instance):
        """Test score display property accessors."""
        score_instance.add(500)
        score_instance.high_score = 800
        
        assert score_instance.value_display == 500
        assert score_instance.high_score_display == 800

    def test_score_file_permissions(self, temp_score_file):
        """Test handling of file permission issues."""
        if os.name != 'nt':  # Skip on Windows
            # Make file read-only
            os.chmod(temp_score_file, 0o444)
            
            score = Score(max_score=1000, filepath=temp_score_file)
            score.add(100)
            # Should not raise exception
            score.save_high_score()


if __name__ == "__main__":
    pytest.main(["-v", __file__])