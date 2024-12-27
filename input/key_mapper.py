# File: input/key_mapper.py
import json
import pygame
from typing import Optional, Dict

class KeyMapper:
    def __init__(self, config_file: str):
        """
        Initialize the KeyMapper with a configuration file.
        :param config_file: Path to the JSON configuration file for key mappings.
        """
        self.config_file = config_file
        self.key_to_action: Dict[int, str] = {}
        self.load_mappings()

    def load_mappings(self) -> None:
        """
        Load key mappings from the configuration file.
        """
        try:
            with open(self.config_file, "r") as f:
                mappings = json.load(f)
            # Convert key names to Pygame key constants
            self.key_to_action = {pygame.key.key_code(v.upper()): k for k, v in mappings.items()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading key mappings: {e}")
            self.key_to_action = {}

    def map_key_to_action(self, key: int) -> Optional[str]:
        """
        Map a raw key event to a logical action.
        :param key: The raw Pygame key constant.
        :return: The logical action name, or None if not mapped.
        """
        return self.key_to_action.get(key)

    def remap_key(self, action: str, new_key: int) -> bool:
        """
        Remap a logical action to a new key.
        :param action: The action to remap.
        :param new_key: The new key constant.
        :return: True if remapping was successful, False otherwise.
        """
        # Remove existing key mapping for the action
        for k, v in list(self.key_to_action.items()):
            if v == action:
                del self.key_to_action[k]
                break
        # Add new key mapping
        self.key_to_action[new_key] = action
        return True

    def save_mappings(self) -> None:
        """
        Save the current key mappings back to the configuration file.
        """
        try:
            # Reverse the mapping for saving
            reverse_mapping = {v: pygame.key.name(k).upper() for k, v in self.key_to_action.items()}
            with open(self.config_file, "w") as f:
                json.dump(reverse_mapping, f, indent=4)
        except IOError as e:
            print(f"Error saving key mappings: {e}")
