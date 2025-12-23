"""
REINFORCE - Monte Carlo Policy Gradient Algorithm

Implementation based on Sutton & Barto "Reinforcement Learning: An Introduction"  
Chapter 13: Policy Gradient Methods

Algorithm (Vanilla Policy Gradient):
------------------------------------
Initialize policy parameters θ arbitrarily
Repeat for each episode:
    Generate episode using π_θ: s_0,a_0,r_1,...,s_T
    For each step t = 0,...,T-1:
        G_t ← return from step t
        θ ← θ + α*G_t*∇log(π(a_t|s_t,θ))

Policy Gradient Theorem:
    ∇J(θ) ∝ E[G_t * ∇log(π(a|s,θ))]

Where:
    - π(a|s,θ) is the policy (softmax in tabular case)
    - G_t is the discounted return from time t
    - ∇log(π) is the score function

Key Properties:
- Directly optimizes policy, not value function
- Monte Carlo: uses full episode returns
- High variance, but unbiased
- Works with continuous action spaces (with neural nets)
"""

import numpy as np
from .base_agent import BaseAgent


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE (Vanilla Policy Gradient) Agent.
    
    Tabular implementation using:
    - State-action preferences (logits)
    - Softmax policy: π(a|s) = exp(h(s,a)) / Σ_b exp(h(s,b))
    - Monte Carlo returns with baseline normalization
    
    Update Rule:
        θ ← θ + α*G_t*∇log(π(a_t|s_t,θ))
    
    For softmax policy:
        ∇log(π(a|s)) = 1(a=a_t) - π(a|s)
    """
    
    def __init__(self, action_space, state_space_size, lr=0.01, gamma=0.99):
        """
        Initialize REINFORCE agent.
        
        Args:
            action_space: Gymnasium action space
            state_space_size: Number of discrete states
            lr: Learning rate (step size α)
            gamma: Discount factor (0 < γ ≤ 1)
        """
        super().__init__(action_space)
        
        self.state_size = state_space_size
        self.n_actions = action_space.n
        
        # Preference table (logits): h(s,a)
        # Policy is π(a|s) = softmax(h(s,:))
        self.preferences = np.zeros((self.state_size, self.n_actions), dtype=np.float32)
        
        # Hyperparameters
        self.lr = lr        # Learning rate α
        self.gamma = gamma  # Discount factor γ
        
        # Episode trajectory buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _softmax(self, preferences):
        """
        Compute softmax probabilities from preferences (logits).
        
        π(a|s) = exp(h(s,a)) / Σ_b exp(h(s,b))
        
        Uses numerical stability trick: subtract max before exp
        """
        z = preferences - np.max(preferences)  # Stability
        exp_prefs = np.exp(z)
        return exp_prefs / np.sum(exp_prefs)
    
    def act(self, observation):
        """
        Sample action from current policy π(a|s,θ).
        
        Args:
            observation: Current state s
            
        Returns:
            int: Sampled action a ~ π(·|s)
        """
        state = observation
        
        # Get policy π(a|s) via softmax
        probs = self._softmax(self.preferences[state])
        
        # Sample action from categorical distribution
        action = int(np.random.choice(self.n_actions, p=probs))
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """
        Store transition and update policy at episode end.
        
        REINFORCE is a Monte Carlo method: updates only when episode completes.
        
        Args:
            state: Current state s_t
            action: Action taken a_t
            reward: Reward received r_t
            next_state: Next state s_{t+1}
            done: Whether episode terminated
        """
        # Store transition in episode buffer
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
        # Only update at end of episode
        if done:
            self._update_policy()
    
    def _update_policy(self):
        """
        Perform REINFORCE policy update using complete episode.
        
        Steps:
        1. Compute discounted returns G_t for each timestep
        2. Normalize returns (baseline for variance reduction)
        3. Update preferences using policy gradient
        """
        T = len(self.episode_rewards)
        
        # Step 1: Compute returns G_t = Σ_{k=t}^T γ^{k-t} * r_k
        returns = []
        G = 0.0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        
        # Step 2: Normalize returns (improves stability)
        # This acts as a baseline: b(s_t) = mean(G)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Step 3: Policy gradient update
        for t in range(T):
            s_t = self.episode_states[t]
            a_t = self.episode_actions[t]
            G_t = returns[t]
            
            # Compute current policy π(a|s_t)
            probs = self._softmax(self.preferences[s_t])
            
            # Gradient of log-policy: ∇log(π(a|s))
            # For softmax: ∇log(π(a|s)) = 1(a=a_t) - π(a|s)
            grad_log_pi = np.zeros(self.n_actions, dtype=np.float32)
            grad_log_pi[:] = -probs  # -π(a|s) for all actions
            grad_log_pi[a_t] += 1.0  # +1 for taken action
            
            # REINFORCE update: θ ← θ + α*G_t*∇log(π)
            self.preferences[s_t] += self.lr * G_t * grad_log_pi
        
        # Clear episode buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def save(self, path):
        """Save policy parameters (preference table)."""
        np.save(path, self.preferences)
    
    def load(self, path):
        """Load policy parameters (preference table)."""
        self.preferences = np.load(path)
    
    def get_policy(self, state):
        """
        Get policy distribution π(·|s) for given state.
        
        Args:
            state: State to query
            
        Returns:
            np.array: Probability distribution over actions
        """
        return self._softmax(self.preferences[state])

