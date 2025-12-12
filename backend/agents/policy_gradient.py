import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from .base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_size = self._get_conv_out(input_shape)
        
        self.actor = nn.Sequential(
            nn.Linear(self.fc_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.fc_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.actor(conv_out), self.critic(conv_out)

class PolicyGradientAgent(BaseAgent):
    def __init__(self, input_shape, action_space, lr=1e-4, gamma=0.99):
        super().__init__(action_space)
        self.gamma = gamma
        self.model = ActorCritic(input_shape, action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []

    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        probs, value = self.model(state)
        dist = torch.distributions.Categorical(logits=probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        return action.item()

    def step(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        if done:
            self.update()

    def update(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        policy_loss = []
        value_loss = []
        
        for log_prob, value, R in zip(self.log_probs, self.values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([[R]])))
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs = []
        self.values = []
        self.rewards = []

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
