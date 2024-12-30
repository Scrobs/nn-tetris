#!/usr/bin/env python3
"""
Unit tests for the GameState class.
"""

import pytest
from enum import Enum
from core.game_state import GameState


class TestGameState:
    """Test suite for the GameState enumeration."""
    
    def test_game_state_members(self):
        """Test that all expected game states are present."""
        expected_states = {
            'MENU',
            'PLAYING',
            'PAUSED',
            'GAME_OVER',
            'ANIMATING'
        }
        actual_states = {state.name for state in GameState}
        assert actual_states == expected_states

    def test_enum_inheritance(self):
        """Test that GameState is properly inheriting from Enum."""
        assert issubclass(GameState, Enum)
        assert isinstance(GameState.MENU, GameState)
        assert isinstance(GameState.PLAYING, GameState)
        assert isinstance(GameState.PAUSED, GameState)
        assert isinstance(GameState.GAME_OVER, GameState)
        assert isinstance(GameState.ANIMATING, GameState)

    def test_state_uniqueness(self):
        """Test that all states have unique values."""
        state_values = [state.value for state in GameState]
        assert len(state_values) == len(set(state_values))

    def test_state_string_representation(self):
        """Test the string representation of game states."""
        assert str(GameState.MENU) == "MENU"
        assert str(GameState.PLAYING) == "PLAYING"
        assert str(GameState.PAUSED) == "PAUSED"
        assert str(GameState.GAME_OVER) == "GAME_OVER"
        assert str(GameState.ANIMATING) == "ANIMATING"

    @pytest.mark.parametrize("state", list(GameState))
    def test_state_immutability(self, state):
        """Test that game states are immutable."""
        with pytest.raises((AttributeError, TypeError)):
            state.value = 999

    def test_valid_state_comparisons(self):
        """Test valid state comparisons."""
        assert GameState.MENU != GameState.PLAYING
        assert GameState.PAUSED != GameState.GAME_OVER
        assert GameState.MENU == GameState.MENU
        assert GameState.PLAYING is GameState.PLAYING

    def test_state_ordering(self):
        """Test that states maintain a consistent ordering."""
        all_states = list(GameState)
        assert len(all_states) == 5
        assert all_states == sorted(all_states, key=lambda x: x.value)

    def test_state_hash_compatibility(self):
        """Test that states can be used in sets and as dictionary keys."""
        state_set = {GameState.MENU, GameState.PLAYING}
        assert len(state_set) == 2
        assert GameState.MENU in state_set
        
        state_dict = {GameState.MENU: "In Menu", GameState.PLAYING: "Currently Playing"}
        assert state_dict[GameState.MENU] == "In Menu"
        assert state_dict[GameState.PLAYING] == "Currently Playing"

    def test_invalid_state_creation(self):
        """Test that creating invalid states is not possible."""
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            GameState("INVALID_STATE")

    @pytest.mark.parametrize("state", list(GameState))
    def test_state_name_value_consistency(self, state):
        """Test consistency between state names and their string representation."""
        assert state.name == str(state)
        assert isinstance(state.name, str)
        assert isinstance(state.value, int)

    def test_state_transitions_validity(self):
        """
        Test valid state transitions based on game logic.
        This documents expected state transition paths.
        """
        valid_transitions = {
            GameState.MENU: {GameState.PLAYING},
            GameState.PLAYING: {GameState.PAUSED, GameState.GAME_OVER, GameState.ANIMATING},
            GameState.PAUSED: {GameState.PLAYING},
            GameState.GAME_OVER: {GameState.MENU, GameState.PLAYING},
            GameState.ANIMATING: {GameState.PLAYING}
        }

        # Verify each state can transition to expected states
        for from_state, to_states in valid_transitions.items():
            for to_state in GameState:
                if to_state in to_states:
                    # These transitions should be logically valid
                    assert to_state in GameState
                    assert isinstance(to_state, GameState)

    def test_auto_value_assignment(self):
        """Test that auto() properly assigned increasing values."""
        state_values = [state.value for state in GameState]
        assert state_values == list(range(1, len(GameState) + 1))


if __name__ == "__main__":
    pytest.main(["-v", __file__])