import numpy as np
import random
from .base_agent import BaseAgent

class TabularQLearningAgent(BaseAgent):
    def __init__(self, action_space, state_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(action_space)
        self.q_table = np.zeros((state_space_size, action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, observation):
        # Observation must be an integer index for tabular
        state = observation
        if random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])

    def step(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

class DynamicProgrammingAgent:
    def __init__(self, env, gamma=0.99):
        # Env must expose P[state][action] = [(prob, next_state, reward, done)...]
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
        pass # DP is offline planning, doesn't update online
