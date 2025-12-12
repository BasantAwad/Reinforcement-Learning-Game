from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def act(self, observation):
        """Select an action based on the observation."""
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Learn from the experience."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the agent model."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the agent model."""
        pass
