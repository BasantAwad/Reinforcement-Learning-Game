"""
SARSA - Temporal Difference Control Algorithm

Implementation based on Sutton & Barto "Reinforcement Learning: An Introduction"
Chapter 6: Temporal-Difference Learning

Algorithm (On-Policy TD Control):
---------------------------------
Initialize Q(s,a) arbitrarily
Repeat for each episode:
    Initialize s
    Choose a from s using policy derived from Q (ε-greedy)
    Repeat for each step:
        Take action a, observe r, s'
        Choose a' from s' using policy derived from Q (ε-greedy)
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        s ← s', a ← a'
    until s is terminal

Key Properties:
- On-policy: learns and improves the same ε-greedy policy
- Uses Q(s',a') where a' is selected by current policy
- Name comes from: State-Action-Reward-State-Action
- More conservative than Q-Learning for stochastic environments
"""

import numpy as np
import random
from .base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    """
    SARSA Agent for Tabular Reinforcement Learning.
    
    On-policy TD control algorithm that learns the value of the policy
    being followed (ε-greedy) and improves it gradually.
    
    Update Rule:
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    
    Where a' is chosen by the current ε-greedy policy (not max).
    
    Difference from Q-Learning:
        - SARSA: Uses Q(s',a') where a' follows ε-greedy ← ON-POLICY
        - Q-Learning: Uses max Q(s',a') ← OFF-POLICY
    """
    
    def __init__(self, action_space, state_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize SARSA agent.
        
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
        
        Args:
            observation: Current state
            
        Returns:
            int: Action to take
        """
        state = observation
        
        # ε-greedy action selection
        if random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        return int(np.argmax(self.q_table[state]))  # Exploit
    
    def step(self, state, action, reward, next_state, done):
        """
        Update Q-table using SARSA (on-policy) update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Where a' is selected using the current ε-greedy policy.
        
        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            done: Whether episode terminated
        """
        # Select next action a' using current policy (ε-greedy)
        # This is the key difference from Q-Learning!
        if done:
            next_q = 0.0
        else:
            if random.random() < self.epsilon:
                next_action = self.action_space.sample()
            else:
                next_action = int(np.argmax(self.q_table[next_state]))
            next_q = self.q_table[next_state, next_action]
        
        # SARSA update: use Q(s',a') where a' follows the policy
        current_q = self.q_table[state, action]
        target = reward + self.gamma * next_q
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

