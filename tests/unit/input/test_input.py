import unittest
import logging
from unittest.mock import Mock
import pygame
from input.action_registry import ActionRegistry
from input.key_mapper import KeyMapper
from input.input_processor import InputProcessor
from input.input_handler import InputHandler


class TestActionRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ActionRegistry()

    def test_register_and_get_callback(self):
        """
        Test that actions can be registered and retrieved correctly.
        """
        mock_callback = Mock()
        self.registry.register_action("TEST_ACTION", mock_callback)
        self.assertEqual(self.registry.get_callback("TEST_ACTION"), mock_callback)

    def test_validate_actions(self):
        """
        Test that missing actions trigger a warning.
        """
        mock_callback = Mock()
        self.registry.register_action("ACTION_1", mock_callback)
        with self.assertLogs(level="WARNING") as log:
            self.registry.validate_actions(["ACTION_1", "ACTION_2"])
        self.assertIn("No callback registered for action 'ACTION_2'", log.output[0])


class TestKeyMapper(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.config_file = "tests/fixtures/input_config.json"
        self.mapper = KeyMapper(self.config_file)

    def test_load_mappings(self):
        """
        Test that key mappings are loaded correctly.
        """
        self.assertEqual(self.mapper.map_key_to_action(pygame.key.key_code("LEFT")), "MOVE_LEFT")
        self.assertEqual(self.mapper.map_key_to_action(pygame.key.key_code("RIGHT")), "MOVE_RIGHT")

    def test_remap_key(self):
        """
        Test that keys can be remapped correctly.
        """
        new_key = pygame.key.key_code("LEFT")
        self.mapper.remap_key("MOVE_LEFT", new_key)
        self.assertEqual(self.mapper.map_key_to_action(new_key), "MOVE_LEFT")

    def test_save_mappings(self):
        """
        Test that key mappings are saved correctly.
        """
        new_key = pygame.key.key_code("LEFT")
        self.mapper.remap_key("MOVE_LEFT", new_key)
        self.mapper.save_mappings()
        self.mapper.load_mappings()
        self.assertEqual(self.mapper.map_key_to_action(new_key), "MOVE_LEFT")


class TestInputProcessor(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.mapper = KeyMapper("tests/fixtures/input_config.json")
        self.registry = ActionRegistry()
        self.processor = InputProcessor(self.mapper, self.registry)

    def test_process_keydown_event(self):
        """
        Test that keydown events are processed correctly.
        """
        self.mapper.key_to_action[pygame.key.key_code("A")] = "MOVE_LEFT"
        mock_callback = Mock()
        self.registry.register_action("MOVE_LEFT", mock_callback)

        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code("A"))
        self.processor.process_event(event)

        mock_callback.assert_called_once()

    def test_process_keyup_event(self):
        """
        Test that keyup events are processed correctly.
        """
        key = pygame.key.key_code("A")
        self.processor.pressed_keys.add(key)

        event = pygame.event.Event(pygame.KEYUP, key=key)
        self.processor.process_event(event)

        self.assertNotIn(key, self.processor.pressed_keys)


class TestInputHandler(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.config_file = "tests/fixtures/input_config.json"
        self.callbacks = {
            "MOVE_LEFT": Mock(),
            "MOVE_RIGHT": Mock(),
        }
        self.handler = InputHandler(self.config_file, self.callbacks)

    def test_handle_event(self):
        """
        Test that InputHandler processes events correctly.
        """
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.key.key_code("LEFT"))
        self.handler.handle_event(event)
        self.callbacks["MOVE_LEFT"].assert_called_once()

    def test_update(self):
        """
        Test that InputHandler's update method does not raise errors.
        """
        try:
            self.handler.update(0.1)
        except Exception as e:
            self.fail(f"InputHandler.update raised an exception: {e}")
