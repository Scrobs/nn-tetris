#!/usr/bin/env python3

import random
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ai.dqn_agent import DQNAgent


class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        """
        Setup the DQNAgent with test parameters.
        """
        self.state_shape = (4, 4, 1)  # Example state shape (e.g., Tetris grid)
        self.action_size = 5  # Example number of actions
        self.agent = DQNAgent(
            state_shape=self.state_shape,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.995,
            batch_size=2,
            memory_size=10,
        )

    @patch("ai.dqn_agent.Sequential")  # Adjusted import path to match the actual usage
    def test_build_model(self, mock_sequential):
        """
        Test the model building process.
        """
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        # Call _build_model and ensure Sequential was invoked
        model = self.agent._build_model()
        mock_sequential.assert_called_once()
        self.assertEqual(mock_model, model)


    def test_update_target_model(self):
        """
        Test updating the target model's weights.
        """
        self.agent.model.get_weights = MagicMock(return_value="mock_weights")
        self.agent.target_model.set_weights = MagicMock()

        self.agent.update_target_model()
        self.agent.model.get_weights.assert_called_once()
        self.agent.target_model.set_weights.assert_called_once_with("mock_weights")

    def test_remember(self):
        """
        Test the remember method to ensure experiences are stored.
        """
        initial_length = len(self.agent.memory)
        state = np.zeros(self.state_shape)
        action = 0
        reward = 1
        next_state = np.ones(self.state_shape)
        done = False

        self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_length + 1)

    @patch("numpy.random.rand")
    def test_act_exploration(self, mock_random):
        """
        Test the act method when epsilon forces exploration.
        """
        mock_random.return_value = 0.5  # Force exploration (<= epsilon)
        self.agent.epsilon = 1.0  # High epsilon for exploration

        action = self.agent.act(np.zeros(self.state_shape))
        self.assertIn(action, range(self.action_size))  # Valid random action

    @patch("tensorflow.keras.models.Sequential")
    def test_act_exploitation(self, mock_sequential):
        """
        Test the act method when epsilon forces exploitation.
        """
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model
        mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # Mock Q-values
        self.agent.model = mock_model  # Use mock model for the agent
        self.agent.epsilon = 0.0  # Force exploitation

        action = self.agent.act(np.zeros(self.state_shape))
        self.assertEqual(action, 4)  # Expect action with max Q-value

    @patch("random.sample")
    @patch("ai.dqn_agent.Sequential")
    def test_replay(self, mock_sequential, mock_random_sample):
        """
        Test the replay method to ensure it trains the model correctly.
        """
        # Mock model
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model
        mock_model.predict.side_effect = [
            np.zeros((2, self.action_size)),  # Q-values for states
            np.zeros((2, self.action_size)),  # Q-values for next states
        ]
        mock_model.fit = MagicMock()

        # Inject the mock model into the agent
        self.agent.model = mock_model
        self.agent.target_model = mock_model

        # Mock experiences to ensure memory meets batch_size
        mock_experiences = [
            (np.zeros(self.state_shape), 0, 1, np.ones(self.state_shape), False),
            (np.ones(self.state_shape), 1, 2, np.zeros(self.state_shape), True),
        ]
        mock_random_sample.return_value = mock_experiences

        # Extend memory to ensure batch_size is met
        for _ in range(self.agent.batch_size):
            self.agent.memory.append(random.choice(mock_experiences))

        # Call replay and check if fit is called
        self.agent.replay()
        mock_model.fit.assert_called_once()  # Verify fit was called


    @patch("tensorflow.keras.models.load_model")
    def test_load_model(self, mock_load_model):
        """
        Test loading a saved model.
        """
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Ensure weights are compatible
        mock_model.get_weights.return_value = self.agent.model.get_weights()

        self.agent.load_model("test_model.h5")
        mock_load_model.assert_called_once_with("test_model.h5")
        self.assertEqual(self.agent.model, mock_model)

    @patch("tensorflow.keras.models.Sequential")
    def test_save_model(self, mock_sequential):
        """
        Test saving the current model.
        """
        mock_model = MagicMock()
        self.agent.model = mock_model

        self.agent.save_model("test_model.h5")
        mock_model.save.assert_called_once_with("test_model.h5")

    def tearDown(self):
        """
        Clean up resources.
        """
        self.agent = None


if __name__ == "__main__":
    unittest.main()
