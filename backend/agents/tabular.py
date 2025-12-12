import numpy as np
import random
from .base_agent import BaseAgent

class TabularQLearningAgent(BaseAgent):
    """
    Q-Learning Agent (Off-Policy TD Control)
    Update rule: Q(s,a) += α[r + γ*max_a'(Q(s',a')) - Q(s,a)]
    Uses max Q-value of next state (off-policy: learns optimal policy)
    """
    def __init__(self, action_space, state_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(action_space)
        self.q_table = np.zeros((state_space_size, action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, observation):
        state = observation
        if random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])

    def step(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        # Q-Learning: use max Q-value of next state (off-policy)
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max * (not done))
        self.q_table[state, action] = new_value

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


class SARSAAgent(BaseAgent):
    """
    SARSA Agent (On-Policy TD Control)
    Update rule: Q(s,a) += α[r + γ*Q(s',a') - Q(s,a)]
    Uses actual next action a' taken by policy (on-policy: learns from behavior)
    
    Key difference from Q-Learning:
    - Q-Learning uses max(Q(s',a')) - learns greedy optimal policy
    - SARSA uses Q(s',a') where a' is the actual next action - learns current policy
    """
    def __init__(self, action_space, state_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(action_space)
        self.q_table = np.zeros((state_space_size, action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.next_action = None  # Store next action for SARSA update

    def act(self, observation):
        state = observation
        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def step(self, state, action, reward, next_state, done):
        # Choose next action using epsilon-greedy (on-policy)
        if random.random() < self.epsilon:
            next_action = self.action_space.sample()
        else:
            next_action = np.argmax(self.q_table[next_state])
        
        old_value = self.q_table[state, action]
        # SARSA: use Q-value of actual next action (on-policy)
        next_q = self.q_table[next_state, next_action] if not done else 0
        
        new_value = old_value + self.alpha * (reward + self.gamma * next_q - old_value)
        self.q_table[state, action] = new_value
        
        # Store next action for consistency (though we recompute in step)
        self.next_action = next_action

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


class DynamicProgrammingAgent:
    """
    Dynamic Programming Agent (Value Iteration)
    Requires environment with known transition probabilities P[s][a]
    Computes optimal policy offline before acting
    """
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.v_table = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

    def value_iteration(self, threshold=1e-8):
        while True:
            delta = 0
            for s in range(self.num_states):
                v = self.v_table[s]
                # Bellman Optimality Update
                q_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    for prob, next_s, reward, done in self.env.unwrapped.P[s][a]:
                        q_values[a] += prob * (reward + (self.gamma * self.v_table[next_s] * (not done)))
                
                self.v_table[s] = np.max(q_values)
                delta = max(delta, abs(v - self.v_table[s]))
            
            if delta < threshold:
                break
        
        # Extract Policy
        for s in range(self.num_states):
            q_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                for prob, next_s, reward, done in self.env.unwrapped.P[s][a]:
                    q_values[a] += prob * (reward + (self.gamma * self.v_table[next_s] * (not done)))
            self.policy[s] = np.argmax(q_values)

    def act(self, observation):
        return self.policy[observation]
        
    def step(self, *args):
        pass  # DP is offline planning, doesn't update online

