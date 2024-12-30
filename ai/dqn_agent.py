#!/usr/bin/env python3

import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for Tetris.
    """

    def __init__(
        self,
        state_shape: tuple,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10000,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # Main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Build the neural network model.
        """
        model = Sequential()
        model.add(Input(shape=self.state_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """
        Update the weights of the target model to match the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Decide on an action based on the current state.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """
        Train the model using a batch of experiences from memory.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict Q-values for current and next states
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q_values[i])
            q_values[i][actions[i]] = target

        # Train the model
        self.model.fit(states, q_values, epochs=1, verbose=0)

        # Reduce epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        """
        Save the model to a file.
        """
        self.model.save(file_path)

    def load_model(self, file_path):
        """
        Load the model from a file.
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(file_path)
        self.update_target_model()
