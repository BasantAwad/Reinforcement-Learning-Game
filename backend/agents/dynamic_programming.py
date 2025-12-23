"""
Dynamic Programming - Value Iteration Algorithm

Implementation based on Sutton & Barto "Reinforcement Learning: An Introduction"
Chapter 4: Dynamic Programming

Algorithm:
---------
Initialize V(s) = 0 for all states
Repeat until convergence:
    For each state s:
        V(s) = max_a Σ_s',r p(s',r|s,a)[r + γV(s')]
        
Extract policy:
    π(s) = argmax_a Σ_s',r p(s',r|s,a)[r + γV(s')]

Parameters:
-----------
gamma (float): Discount factor (0 < γ ≤ 1)
theta (float): Convergence threshold
"""

import numpy as np


class DynamicProgrammingAgent:
    """
    Dynamic Programming Agent using Value Iteration.
    
    Requires environment with discrete state/action spaces and known
    transition probabilities P[s][a] = [(prob, next_state, reward, done), ...]
    
    This is an offline algorithm that computes the optimal policy before acting.
    
    Update Rule:
        V(s) = max_a Σ p(s',r|s,a)[r + γV(s')]
    
    Then extract greedy policy:
        π(s) = argmax_a Σ p(s',r|s,a)[r + γV(s')]
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-8):
        """
        Initialize Value Iteration agent.
        
        Args:
            env: Gymnasium environment with P attribute
            gamma: Discount factor [0, 1]
            theta: Convergence threshold for value iteration
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # Get environment dimensions
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        
        # Initialize value function and policy
        self.V = np.zeros(self.num_states, dtype=np.float64)
        self.policy = np.zeros(self.num_states, dtype=np.int32)
        
        self.iterations = 0
    
    def value_iteration(self, threshold=None):
        """
        Perform value iteration to find optimal value function.
        
        Args:
            threshold: Optional override for convergence threshold
            
        Returns:
            int: Number of iterations until convergence
        """
        if threshold is None:
            threshold = self.theta
            
        iteration = 0
        
        while True:
            delta = 0  # Track maximum change in value function
            
            # Update value for each state
            for s in range(self.num_states):
                v_old = self.V[s]
                
                # Compute Q-values for all actions
                q_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    # Sum over all possible next states
                    for prob, next_s, reward, done in self.env.unwrapped.P[s][a]:
                        # Bellman backup
                        q_values[a] += prob * (reward + self.gamma * self.V[next_s] * (1 - done))
                
                # Update value function: V(s) = max_a Q(s,a)
                self.V[s] = np.max(q_values)
                
                # Track convergence
                delta = max(delta, abs(v_old - self.V[s]))
            
            iteration += 1
            
            # Check for convergence
            if delta < threshold:
                break
        
        self.iterations = iteration
        
        # Extract optimal policy after convergence
        self._extract_policy()
        
        return iteration
    
    def _extract_policy(self):
        """Extract greedy policy from value function."""
        for s in range(self.num_states):
            # Compute Q-values for all actions
            q_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                for prob, next_s, reward, done in self.env.unwrapped.P[s][a]:
                    q_values[a] += prob * (reward + self.gamma * self.V[next_s] * (1 - done))
            
            # Select action with highest Q-value
            self.policy[s] = int(np.argmax(q_values))
    
    def act(self, observation):
        """
        Select action according to computed policy.
        
        Args:
            observation: Current state
            
        Returns:
            int: Action to take
        """
        return int(self.policy[observation])
    
    def step(self, *args):
        """
        No-op for compatibility with online learning agents.
        Value iteration computes policy offline.
        """
        pass
    
    def get_value_function(self):
        """Return the learned value function."""
        return self.V.copy()
    
    def get_policy(self):
        """Return the learned policy."""
        return self.policy.copy()

