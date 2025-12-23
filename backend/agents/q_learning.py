"""
Q-Learning - Temporal Difference Control Algorithm

Implementation based on Sutton & Barto "Reinforcement Learning: An Introduction"
Chapter 6: Temporal-Difference Learning

Algorithm (Off-Policy TD Control):
----------------------------------
Initialize Q(s,a) arbitrarily
Repeat for each episode:
    Initialize s
    Repeat for each step:
        Choose a from s using policy derived from Q (ε-greedy)
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
    until s is terminal

Key Properties:
- Off-policy: learns optimal policy while following ε-greedy
- Uses max Q(s',a') regardless of action actually taken
- Converges to optimal Q* with probability 1
"""

import numpy as np
import random
from .base_agent import BaseAgent


class TabularQLearningAgent(BaseAgent):
    """
    Q-Learning Agent for Tabular Reinforcement Learning.
    
    Off-policy TD control algorithm that learns the optimal action-value
    function Q*(s,a) while following an ε-greedy exploration policy.
    
    Update Rule:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Where:
        α = learning rate (step size)
        γ = discount factor
        ε = exploration rate for ε-greedy policy
    """
    
    def __init__(self, action_space, state_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize Q-Learning agent.
        
        Args:
            action_space: Gymnasium action space
            state_space_size: Number of discrete states
            alpha: Learning rate (0 < α ≤ 1)
            gamma: Discount factor (0 < γ ≤ 1)
            epsilon: Exploration probability for ε-greedy
        """
        super().__init__(action_space)
        
        # Initialize Q-table to zeros
        self.q_table = np.zeros((state_space_size, action_space.n), dtype=np.float64)
        
        # Hyperparameters
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor  
        self.epsilon = epsilon  # Exploration rate
    
    def act(self, observation):
        """
        Select action using ε-greedy policy.
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (greedy action)
        
        Args:
            observation: Current state
            
        Returns:
            int: Action to take
        """
        state = observation
        
        # Exploration: random action
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # Exploitation: greedy action (argmax Q(s,a))
        return int(np.argmax(self.q_table[state]))
    
    def step(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            done: Whether episode terminated
        """
        # Get current Q-value
        current_q = self.q_table[state, action]
        
        # Get maximum Q-value for next state (off-policy)
        # If episode done, next state has no value
        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        
        # Q-learning update: Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)
    
    def save(self, path):
        """Save Q-table to file."""
        np.save(path, self.q_table)
    
    def load(self, path):
        """Load Q-table from file."""
        self.q_table = np.load(path)
    
    def get_q_table(self):
        """Return copy of Q-table."""
        return self.q_table.copy()
    
    def get_epsilon(self):
        """Return current exploration rate."""
        return self.epsilon

